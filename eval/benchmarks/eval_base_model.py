"""Phase 6 - Base-model objective benchmarking.

Evaluates one or more pretraining checkpoints under
``training/checkpoints/<group>/best.pt`` against three Phase-5.5 artefacts:

    data/eval/mmlu_subset.jsonl       English MCQ
    data/eval/cmmlu_subset.jsonl      Chinese MCQ
    data/eval/translation_pairs.jsonl ZH-EN sentence pairs

Method:
    The model is a BASE LM (no SFT). Generative QA is unreliable, so we use
    the *choice-likelihood* method: for each MCQ we build a prefix like
    ``"Question: ...\\nAnswer:"`` and score the average per-token log-prob
    of each of the 4 choice texts as a continuation. Predict the argmax.
    Length-normalised so longer choices aren't penalised.

    For cross-lingual PPL we compute PPL over three text formations:
      - zh sentences only
      - en sentences only
      - "<zh>\\n<en>" concatenations (stress-tests bilingual transition)
    Lower zh+en PPL relative to (zh+en)/2 of the monolingual PPLs suggests
    the model has actually learned cross-lingual context, not just two
    independent monolingual modes.

Multi-checkpoint loop:
    Walks ``--checkpoints-root`` (default ``training/checkpoints``), evaluates
    every ``<group>/best.pt`` it finds, prints a comparison table at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.generate import load_checkpoint  # noqa: E402
from model import MiniLLM  # noqa: E402  (type only)
from tokenizer import BBPETokenizer  # noqa: E402

# id 0 = NULL byte in our byte-level BBPE base vocab; used as padding only.
PAD_ID = 0


def _resolve_tokenizer_dir(tokenizer: str | None, tokenizer_size: str) -> Path:
    if tokenizer:
        return Path(tokenizer)
    return Path("tokenizer") / tokenizer_size


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_mcq_prefix(question: str, lang: str) -> str:
    if lang == "zh":
        return f"问题：{question}\n答案："
    return f"Question: {question}\nAnswer:"


def build_choice_text(text: str, lang: str) -> str:
    """The choice text appended after the prefix.

    English: prepend a space so BBPE pre-tokenisation treats the choice as a
    fresh word (otherwise the leading char glues onto the colon token).
    Chinese: no leading separator needed — CJK chars don't merge with the
    ASCII colon.
    """
    if lang == "zh":
        return text
    return " " + text


# ---------------------------------------------------------------------------
# Choice-likelihood scoring (batched: 4 choices in one forward pass)
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_four_choices(
    model: MiniLLM,
    tok: BBPETokenizer,
    prefix: str,
    choice_texts: List[str],
    device: torch.device,
    max_len_cap: int = 512,
) -> List[float]:
    """Return one length-normalised log-prob per choice.

    Builds a [4, T] padded batch and runs one forward; far cheaper than 4
    sequential forwards. The padded positions contribute nothing because we
    only gather log-probs at the choice token positions of each row.
    """
    prefix_ids = tok.encode(prefix)
    P = len(prefix_ids)

    rows: List[List[int]] = []
    choice_lens: List[int] = []
    for ct in choice_texts:
        cids = tok.encode(ct)
        rows.append(prefix_ids + cids)
        choice_lens.append(len(cids))

    max_len = min(max(len(r) for r in rows), max_len_cap)
    # Truncate rows from the left if too long, keeping the prefix-tail and
    # full choice intact (we always need the prefix-final token to predict
    # the choice's first token).
    padded: List[List[int]] = []
    eff_prefix_lens: List[int] = []
    for r in rows:
        if len(r) > max_len:
            # Drop earliest prefix tokens to fit.
            drop = len(r) - max_len
            r = r[drop:]
        eff_prefix_lens.append(len(r) - choice_lens[len(padded)])
        padded.append(r + [PAD_ID] * (max_len - len(r)))

    x = torch.tensor(padded, dtype=torch.long, device=device)  # [4, max_len]
    logits, _ = model(x)                                       # [4, max_len, V]
    log_probs = F.log_softmax(logits.float(), dim=-1)          # cast for stability

    scores: List[float] = []
    for i, (P_eff, cl) in enumerate(zip(eff_prefix_lens, choice_lens)):
        if cl <= 0 or P_eff <= 0:
            scores.append(float("-inf"))
            continue
        # Choice tokens are at positions [P_eff, P_eff+cl). The prediction for
        # the choice token at position P_eff+j sits in logits[P_eff-1+j].
        pred_idx = torch.arange(P_eff - 1, P_eff - 1 + cl, device=device)
        choice_ids = torch.tensor(padded[i][P_eff:P_eff + cl], dtype=torch.long, device=device)
        token_lp = log_probs[i, pred_idx].gather(-1, choice_ids.unsqueeze(-1)).squeeze(-1)
        scores.append(float(token_lp.sum().item() / cl))   # length-normalise
    return scores


# ---------------------------------------------------------------------------
# MCQ evaluation
# ---------------------------------------------------------------------------
def evaluate_mcq(
    model: MiniLLM,
    tok: BBPETokenizer,
    jsonl_path: Path,
    lang: str,
    device: torch.device,
    max_items: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    if not jsonl_path.exists():
        print(f"  [skip] {jsonl_path} not found", file=sys.stderr)
        return {"accuracy": float("nan"), "n": 0, "correct": 0, "by_subject": {}}

    correct = 0
    total = 0
    by_subj: Dict[str, Tuple[int, int]] = {}
    t0 = time.time()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_items is not None and i >= max_items:
                break
            row = json.loads(line)
            q = row.get("question", "")
            choices = row.get("choices", {})
            gold = (row.get("answer", "") or "").strip().upper()[:1]
            subject = row.get("subject", "?")
            if not q or set("ABCD") - set(choices) or gold not in "ABCD":
                continue

            prefix = build_mcq_prefix(q, lang)
            choice_texts = [build_choice_text(choices[L], lang) for L in "ABCD"]
            scores = score_four_choices(model, tok, prefix, choice_texts, device)
            pred = "ABCD"[int(max(range(4), key=lambda j: scores[j]))]

            c_old, t_old = by_subj.get(subject, (0, 0))
            by_subj[subject] = (c_old + int(pred == gold), t_old + 1)
            correct += int(pred == gold)
            total += 1

            if verbose and total % 50 == 0:
                print(f"    ... {total} items, running acc = {correct / total * 100:.2f}% "
                      f"({time.time() - t0:.1f}s)", flush=True)

    acc = correct / max(1, total)
    return {
        "accuracy": acc,
        "n": total,
        "correct": correct,
        "by_subject": {s: (c, t, c / max(1, t)) for s, (c, t) in by_subj.items()},
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Cross-lingual PPL
# ---------------------------------------------------------------------------
@torch.no_grad()
def text_ppl(model: MiniLLM, tok: BBPETokenizer, texts: List[str], device: torch.device,
             max_seq: int = 512) -> Tuple[float, int]:
    """Mean per-token NLL over ``texts`` (no padding — one forward per text).

    Returns (perplexity, total_tokens). Sequences longer than ``max_seq`` are
    truncated from the right (we keep the prefix). Empty / 1-token texts skipped.
    """
    total_nll = 0.0
    total_tok = 0
    for text in texts:
        ids = tok.encode(text)
        if len(ids) < 2:
            continue
        if len(ids) > max_seq:
            ids = ids[:max_seq]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x)                              # [1, T, V]
        # Standard shift: logits[:-1] predict ids[1:].
        nll = F.cross_entropy(
            logits[0, :-1].float(),
            torch.tensor(ids[1:], dtype=torch.long, device=device),
            reduction="sum",
        )
        total_nll += float(nll.item())
        total_tok += len(ids) - 1
    if total_tok == 0:
        return float("nan"), 0
    return math.exp(total_nll / total_tok), total_tok


def evaluate_translation(
    model: MiniLLM,
    tok: BBPETokenizer,
    jsonl_path: Path,
    device: torch.device,
    max_items: Optional[int] = None,
) -> dict:
    if not jsonl_path.exists():
        print(f"  [skip] {jsonl_path} not found", file=sys.stderr)
        return {"ppl_zh": float("nan"), "ppl_en": float("nan"), "ppl_zh_en": float("nan"),
                "n_zh": 0, "n_en": 0, "n_pair": 0}

    zh_texts: List[str] = []
    en_texts: List[str] = []
    paired_texts: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_items is not None and i >= max_items:
                break
            row = json.loads(line)
            zh = (row.get("zh") or "").strip()
            en = (row.get("en") or "").strip()
            if zh:
                zh_texts.append(zh)
            if en:
                en_texts.append(en)
            if zh and en:
                paired_texts.append(zh + "\n" + en)

    ppl_zh, tok_zh = text_ppl(model, tok, zh_texts, device)
    ppl_en, tok_en = text_ppl(model, tok, en_texts, device)
    ppl_mix, tok_mix = text_ppl(model, tok, paired_texts, device)
    return {
        "ppl_zh": ppl_zh,
        "ppl_en": ppl_en,
        "ppl_zh_en": ppl_mix,
        "n_zh": len(zh_texts), "tok_zh": tok_zh,
        "n_en": len(en_texts), "tok_en": tok_en,
        "n_pair": len(paired_texts), "tok_pair": tok_mix,
    }


# ---------------------------------------------------------------------------
# Multi-checkpoint orchestration
# ---------------------------------------------------------------------------
def discover_checkpoints(root: Path, filename: str = "best.pt") -> List[Tuple[str, Path]]:
    """Find ``<root>/<group>/<filename>`` candidates, sorted by group name."""
    if not root.exists():
        return []
    found: List[Tuple[str, Path]] = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            ckpt = sub / filename
            if ckpt.exists():
                found.append((sub.name, ckpt))
    return found


def _fmt_pct(x: float) -> str:
    if x != x:   # NaN
        return "   NA  "
    return f"{x * 100:6.2f}%"


def _fmt_ppl(x: float) -> str:
    if x != x:
        return "   NA  "
    return f"{x:>7.2f}"


def print_comparison(results: "OrderedDict[str, dict]") -> None:
    print("\n" + "=" * 76)
    print("Comparison report")
    print("=" * 76)
    print(f"  {'group':<10s} {'MMLU':>9s} {'CMMLU':>9s} {'PPL_zh':>9s} {'PPL_en':>9s} {'PPL_zh+en':>11s}")
    print(f"  {'-' * 10} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 11}")
    for name, r in results.items():
        print(
            f"  {name:<10s} "
            f"{_fmt_pct(r['mmlu']['accuracy']):>9s} "
            f"{_fmt_pct(r['cmmlu']['accuracy']):>9s} "
            f"{_fmt_ppl(r['trans']['ppl_zh']):>9s} "
            f"{_fmt_ppl(r['trans']['ppl_en']):>9s} "
            f"{_fmt_ppl(r['trans']['ppl_zh_en']):>11s}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-root", type=Path, default=Path("training/checkpoints"))
    ap.add_argument("--ckpt-name", default="best.pt",
                    help="Filename to look for inside each <group>/ subdir.")
    ap.add_argument("--mmlu", type=Path, default=Path("data/eval/mmlu_subset.jsonl"))
    ap.add_argument("--cmmlu", type=Path, default=Path("data/eval/cmmlu_subset.jsonl"))
    ap.add_argument("--translation", type=Path, default=Path("data/eval/translation_pairs.jsonl"))
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer directory path. If omitted, uses tokenizer/<--tokenizer-size>.",
    )
    ap.add_argument(
        "--tokenizer-size",
        default="8k",
        help="Tokenizer variant under tokenizer/. e.g. 8k, 32k",
    )
    ap.add_argument("--max-items", type=int, default=None,
                    help="Cap items per task for a quick smoke. Default: use all.")
    ap.add_argument("--n-heads", type=int, default=4,
                    help="Used only when the checkpoint has no config and n_heads can't be inferred.")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[eval] device = {device}", flush=True)

    tokenizer_dir = _resolve_tokenizer_dir(args.tokenizer, args.tokenizer_size)
    tok = BBPETokenizer.load(tokenizer_dir)
    print(f"[eval] tokenizer dir = {tokenizer_dir}", flush=True)
    print(f"[eval] tokenizer vocab = {tok.vocab_size}", flush=True)

    ckpts = discover_checkpoints(args.checkpoints_root, args.ckpt_name)
    if not ckpts:
        print(f"[fatal] no checkpoints found under {args.checkpoints_root}/<group>/{args.ckpt_name}",
              file=sys.stderr)
        sys.exit(1)
    print(f"[eval] {len(ckpts)} checkpoints: {[c[0] for c in ckpts]}", flush=True)

    results: "OrderedDict[str, dict]" = OrderedDict()

    for name, path in ckpts:
        print(f"\n{'=' * 60}")
        print(f"== {name}   {path}")
        print(f"{'=' * 60}", flush=True)

        try:
            model, cfg = load_checkpoint(
                path,
                n_heads=args.n_heads,
                max_seq_len=args.max_seq_len,
                device=device,
            )
        except Exception as exc:
            print(f"  [error] failed to load: {exc}", file=sys.stderr)
            continue

        # MMLU
        print("  -- MMLU (en)", flush=True)
        mmlu = evaluate_mcq(model, tok, args.mmlu, "en", device,
                            max_items=args.max_items, verbose=not args.quiet)
        print(f"  -> MMLU acc  = {mmlu['accuracy'] * 100:.2f}%  "
              f"({mmlu['correct']}/{mmlu['n']}, {mmlu['elapsed_s']:.1f}s)", flush=True)
        for s, (c, t, a) in mmlu["by_subject"].items():
            print(f"      - {s:<40s} {c}/{t} = {a * 100:.2f}%")

        # CMMLU
        print("  -- CMMLU (zh)", flush=True)
        cmmlu = evaluate_mcq(model, tok, args.cmmlu, "zh", device,
                             max_items=args.max_items, verbose=not args.quiet)
        print(f"  -> CMMLU acc = {cmmlu['accuracy'] * 100:.2f}%  "
              f"({cmmlu['correct']}/{cmmlu['n']}, {cmmlu['elapsed_s']:.1f}s)", flush=True)
        for s, (c, t, a) in cmmlu["by_subject"].items():
            print(f"      - {s:<40s} {c}/{t} = {a * 100:.2f}%")

        # Translation PPL
        print("  -- Cross-lingual PPL", flush=True)
        trans = evaluate_translation(model, tok, args.translation, device, max_items=args.max_items)
        print(f"      zh-only   : PPL = {trans['ppl_zh']:7.2f}  "
              f"(n={trans['n_zh']}, tokens={trans['tok_zh']})")
        print(f"      en-only   : PPL = {trans['ppl_en']:7.2f}  "
              f"(n={trans['n_en']}, tokens={trans['tok_en']})")
        print(f"      zh \\n en  : PPL = {trans['ppl_zh_en']:7.2f}  "
              f"(n={trans['n_pair']}, tokens={trans['tok_pair']})")

        results[name] = {"mmlu": mmlu, "cmmlu": cmmlu, "trans": trans, "config": cfg}

        # Release VRAM before loading the next checkpoint.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if results:
        print_comparison(results)


if __name__ == "__main__":
    main()
