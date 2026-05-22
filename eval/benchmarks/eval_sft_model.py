"""Phase 7.2 - SFT-model objective benchmarking.

Re-uses the Phase-5.5 datasets:

    data/eval/mmlu_subset.jsonl       English MCQ
    data/eval/cmmlu_subset.jsonl      Chinese MCQ

Method (vs the base-model script):
    SFT models can now follow instructions, so we move from
    *choice-likelihood* to **generative QA**. For each item we build a
    ChatML-style prompt using the SAME plain-text role convention that
    ``training/sft.py`` trains on:

        system: <instruction telling the model to output A/B/C/D only>
        user:   <question + 4 lettered choices>
        assistant: <continuation to generate>

    We greedy-decode a short continuation (default 8 tokens) and pick the
    first ``[ABCD]`` character that appears. If the model never emits one,
    the item is marked as no-answer (counts as wrong, but tracked
    separately so we can tell "ignored the prompt" from "guessed wrong").

Output:
    - On-screen per-subject and overall accuracy.
    - A Markdown report at
      ``eval/benchmarks/results_sft_<YYYYMMDD_HHMMSS>.md`` with the same
      breakdown plus a sample of wrong answers (helps spot whether the
      model is forgetting/catastrophic-forgetting vs just confused).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime
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

# Match training/sft.py exactly — base BBPE has no <|im_start|> tokens, so
# SFT uses plain "role: text\n" lines. Inference must mirror this.
ROLE_PREFIX = {"system": "system: ", "user": "user: ", "assistant": "assistant: "}
TURN_SEP = "\n"

SYSTEM_PROMPT_ZH = "请回答以下多选题，只输出正确选项的字母(A, B, C 或 D)。"
SYSTEM_PROMPT_EN = (
    "Answer the following multiple-choice question. "
    "Output only the letter of the correct option (A, B, C, or D)."
)

ANSWER_LETTER_RE = re.compile(r"[ABCD]")


def _resolve_tokenizer_dir(tokenizer: str | None, tokenizer_size: str) -> Path:
    if tokenizer:
        return Path(tokenizer)
    return Path("tokenizer") / tokenizer_size


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_user_block(question: str, choices: Dict[str, str], lang: str) -> str:
    header = "题目: " if lang == "zh" else "Question: "
    return (
        f"{header}{question}\n"
        f"A. {choices['A']}\n"
        f"B. {choices['B']}\n"
        f"C. {choices['C']}\n"
        f"D. {choices['D']}"
    )


def build_chatml_prompt(question: str, choices: Dict[str, str], lang: str) -> str:
    """Render the (system, user, assistant-prefix) messages as the SFT
    encoder would, but stop right after the assistant role prefix so the
    model continues with its answer. System + question header switch to
    match the question's language."""
    system_text = SYSTEM_PROMPT_ZH if lang == "zh" else SYSTEM_PROMPT_EN
    parts = [
        ROLE_PREFIX["system"] + system_text + TURN_SEP,
        ROLE_PREFIX["user"] + build_user_block(question, choices, lang) + TURN_SEP,
        ROLE_PREFIX["assistant"],
    ]
    return "".join(parts)


# ---------------------------------------------------------------------------
# Greedy generation (no sampling — we want reproducible eval numbers)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate(
    model: MiniLLM,
    tok: BBPETokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    input_ids = tok.encode(prompt)
    if not input_ids:
        return ""

    # Truncate from the left if the prompt overflows the model's context,
    # preserving the assistant prefix at the tail (which is where decoding
    # picks up).
    max_ctx = int(model.max_seq_len) - max_new_tokens
    if max_ctx > 0 and len(input_ids) > max_ctx:
        input_ids = input_ids[-max_ctx:]

    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits, past = model(x, use_cache=True)
    next_id = int(torch.argmax(logits[0, -1]).item())

    new_ids: List[int] = [next_id]
    for _ in range(max_new_tokens - 1):
        if past is not None and past[0][0].shape[-2] >= model.max_seq_len:
            break
        x_step = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, past = model(x_step, past_key_values=past, use_cache=True)
        next_id = int(torch.argmax(logits[0, -1]).item())
        new_ids.append(next_id)
        # Early stop on newline character — model has finished its turn.
        if tok.decode([next_id]).endswith("\n"):
            break

    return tok.decode(new_ids)


def parse_answer_letter(text: str) -> Optional[str]:
    m = ANSWER_LETTER_RE.search(text)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# MCQ evaluation
# ---------------------------------------------------------------------------
def evaluate_mcq(
    model: MiniLLM,
    tok: BBPETokenizer,
    jsonl_path: Path,
    device: torch.device,
    *,
    lang: str,
    max_items: Optional[int] = None,
    max_new_tokens: int = 8,
    verbose: bool = True,
    sample_wrong_keep: int = 5,
) -> dict:
    if not jsonl_path.exists():
        print(f"  [skip] {jsonl_path} not found", file=sys.stderr)
        return {
            "accuracy": float("nan"),
            "n": 0,
            "correct": 0,
            "no_answer": 0,
            "by_subject": {},
            "wrong_samples": [],
        }

    correct = 0
    no_answer = 0
    total = 0
    by_subj: Dict[str, List[int]] = {}   # subject -> [correct, total, no_answer]
    wrong_samples: List[dict] = []
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

            prompt = build_chatml_prompt(q, choices, lang)
            raw = greedy_generate(model, tok, prompt, max_new_tokens, device)
            pred = parse_answer_letter(raw)

            entry = by_subj.setdefault(subject, [0, 0, 0])
            entry[1] += 1
            if pred is None:
                no_answer += 1
                entry[2] += 1
                hit = False
            else:
                hit = (pred == gold)
                if hit:
                    correct += 1
                    entry[0] += 1
            total += 1

            if not hit and len(wrong_samples) < sample_wrong_keep:
                wrong_samples.append({
                    "subject": subject,
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "model_raw": raw,
                })

            if verbose and total % 50 == 0:
                running = correct / total * 100
                print(
                    f"    ... {total} items, running acc = {running:.2f}%  "
                    f"no_answer={no_answer}  ({time.time() - t0:.1f}s)",
                    flush=True,
                )

    acc = correct / max(1, total)
    return {
        "accuracy": acc,
        "n": total,
        "correct": correct,
        "no_answer": no_answer,
        "by_subject": {
            s: (c, t, na, c / max(1, t)) for s, (c, t, na) in by_subj.items()
        },
        "wrong_samples": wrong_samples,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def discover_sft_checkpoints(
    root: Path,
    ckpt_name: Optional[str],
) -> List[Tuple[str, Path]]:
    """Walk ``<root>/<group>/`` looking for SFT checkpoint files.

    If ``ckpt_name`` is given, look for exactly that filename. Otherwise
    auto-detect ``SFT_<group>.pt`` (preferred) and fall back to
    ``SFT_best.pt``.
    """
    if not root.exists():
        return []
    found: List[Tuple[str, Path]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if ckpt_name:
            p = sub / ckpt_name
            if p.exists():
                found.append((sub.name, p))
            continue
        for candidate in (f"SFT_{sub.name}.pt", "SFT_best.pt"):
            p = sub / candidate
            if p.exists():
                found.append((sub.name, p))
                break
    return found


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    if x != x:
        return "   NA "
    return f"{x * 100:6.2f}%"


def print_comparison(results: "OrderedDict[str, dict]") -> None:
    print("\n" + "=" * 64)
    print("SFT comparison report")
    print("=" * 64)
    print(f"  {'group':<14s} {'MMLU':>10s} {'CMMLU':>10s} "
          f"{'MMLU_NA':>9s} {'CMMLU_NA':>10s}")
    print(f"  {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 9} {'-' * 10}")
    for name, r in results.items():
        mmlu = r["mmlu"]; cmmlu = r["cmmlu"]
        print(
            f"  {name:<14s} "
            f"{_fmt_pct(mmlu['accuracy']):>10s} "
            f"{_fmt_pct(cmmlu['accuracy']):>10s} "
            f"{mmlu['no_answer']:>9d} "
            f"{cmmlu['no_answer']:>10d}"
        )


def _md_subject_table(name: str, by_subject: Dict[str, Tuple[int, int, int, float]]) -> str:
    lines = [f"### {name} per-subject\n",
             "| subject | correct | total | no_answer | acc |",
             "|---|---:|---:|---:|---:|"]
    # Sort by subject name for stable diffs across runs.
    for subj in sorted(by_subject):
        c, t, na, a = by_subject[subj]
        lines.append(f"| {subj} | {c} | {t} | {na} | {a * 100:.2f}% |")
    return "\n".join(lines) + "\n"


def _md_wrong_samples(name: str, samples: List[dict]) -> str:
    if not samples:
        return ""
    lines = [f"### {name} wrong-answer samples\n"]
    for i, s in enumerate(samples, 1):
        raw_oneline = s["model_raw"].replace("\n", "\\n")
        lines.append(
            f"**{i}. [{s['subject']}]** gold=`{s['gold']}`  pred=`{s['pred']}`\n\n"
            f"> Q: {s['question']}\n\n"
            f"> Model output (raw): `{raw_oneline}`\n"
        )
    return "\n".join(lines) + "\n"


def write_markdown(
    out_path: Path,
    tokenizer_dir: Path,
    results: "OrderedDict[str, dict]",
    args: argparse.Namespace,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections: List[str] = []
    sections.append("# SFT Model Evaluation\n")
    sections.append(
        f"- timestamp: `{ts}`\n"
        f"- tokenizer dir: `{tokenizer_dir}`\n"
        f"- max_items: `{args.max_items}`\n"
        f"- max_new_tokens: `{args.max_new_tokens}`\n"
        f"- decoding: `greedy (argmax, temperature=0)`\n"
        f"- prompt: ChatML plain-text (system/user/assistant) — matches `training/sft.py`\n"
    )

    # Overall comparison table.
    sections.append("\n## Overall accuracy\n")
    sections.append(
        "| group | MMLU acc | MMLU correct/total | MMLU no_answer | "
        "CMMLU acc | CMMLU correct/total | CMMLU no_answer | elapsed |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for name, r in results.items():
        m = r["mmlu"]; c = r["cmmlu"]
        elapsed = m.get("elapsed_s", 0) + c.get("elapsed_s", 0)
        sections.append(
            f"| {name} | {m['accuracy'] * 100:.2f}% | {m['correct']}/{m['n']} | "
            f"{m['no_answer']} | {c['accuracy'] * 100:.2f}% | {c['correct']}/{c['n']} | "
            f"{c['no_answer']} | {elapsed:.1f}s |"
        )
    sections.append("")

    # Per-group sections.
    for name, r in results.items():
        sections.append(f"\n## {name}\n")
        sections.append(f"- checkpoint: `{r['ckpt_path']}`\n")
        sections.append(_md_subject_table("MMLU", r["mmlu"]["by_subject"]))
        sections.append(_md_subject_table("CMMLU", r["cmmlu"]["by_subject"]))
        sections.append(_md_wrong_samples("MMLU", r["mmlu"]["wrong_samples"]))
        sections.append(_md_wrong_samples("CMMLU", r["cmmlu"]["wrong_samples"]))

    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"\n[report] wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    # Two ways to specify checkpoints:
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Single SFT checkpoint .pt file. If given, "
                         "--checkpoints-root and --ckpt-name are ignored.")
    ap.add_argument("--group-name", default=None,
                    help="Display name for the single --checkpoint. "
                         "Defaults to parent dir name.")
    ap.add_argument("--checkpoints-root", type=Path,
                    default=Path("training/checkpoints"),
                    help="Auto-discover SFT_*.pt files under <root>/<group>/.")
    ap.add_argument("--ckpt-name", default=None,
                    help="Exact filename to look for in each group dir. "
                         "If omitted, prefers SFT_<group>.pt, falls back to "
                         "SFT_best.pt.")

    ap.add_argument("--mmlu", type=Path, default=Path("data/eval/mmlu_subset.jsonl"))
    ap.add_argument("--cmmlu", type=Path, default=Path("data/eval/cmmlu_subset.jsonl"))
    ap.add_argument("--tokenizer", default=None,
                    help="Tokenizer dir. If omitted, uses tokenizer/<--tokenizer-size>.")
    ap.add_argument("--tokenizer-size", default="8k")
    ap.add_argument("--max-items", type=int, default=None,
                    help="Cap items per task for a quick smoke. Default: use all.")
    ap.add_argument("--max-new-tokens", type=int, default=20,
                    help="Tokens to generate per question; tighter = faster + cleaner parsing.")
    ap.add_argument("--n-heads", type=int, default=4,
                    help="Fallback only when the checkpoint lacks model_config.n_heads.")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--report-dir", type=Path,
                    default=Path("eval/benchmarks"),
                    help="Where the timestamped .md report is written.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[eval] device = {device}", flush=True)

    tokenizer_dir = _resolve_tokenizer_dir(args.tokenizer, args.tokenizer_size)
    tok = BBPETokenizer.load(tokenizer_dir)
    print(f"[eval] tokenizer dir = {tokenizer_dir}  vocab={tok.vocab_size}", flush=True)

    # Resolve which checkpoints to run.
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            print(f"[fatal] checkpoint not found: {args.checkpoint}", file=sys.stderr)
            sys.exit(1)
        name = args.group_name or args.checkpoint.parent.name
        ckpts = [(name, args.checkpoint)]
    else:
        ckpts = discover_sft_checkpoints(args.checkpoints_root, args.ckpt_name)
        if not ckpts:
            print(
                f"[fatal] no SFT checkpoints under {args.checkpoints_root}/<group>/ "
                f"(looked for {args.ckpt_name or 'SFT_<group>.pt | SFT_best.pt'})",
                file=sys.stderr,
            )
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
        print(
            f"  config: vocab={cfg['vocab_size']} d_model={cfg['d_model']} "
            f"n_layer={cfg['n_layer']} n_heads={cfg['n_heads']} "
            f"max_seq_len={cfg['max_seq_len']}",
            flush=True,
        )

        print("  -- MMLU (en)", flush=True)
        mmlu = evaluate_mcq(
            model, tok, args.mmlu, device,
            lang="en",
            max_items=args.max_items,
            max_new_tokens=args.max_new_tokens,
            verbose=not args.quiet,
        )
        print(f"  -> MMLU acc  = {mmlu['accuracy'] * 100:.2f}%  "
              f"({mmlu['correct']}/{mmlu['n']}, no_answer={mmlu['no_answer']}, "
              f"{mmlu['elapsed_s']:.1f}s)", flush=True)
        for s, (c, t, na, a) in sorted(mmlu["by_subject"].items()):
            print(f"      - {s:<40s} {c}/{t} (NA={na}) = {a * 100:.2f}%")

        print("  -- CMMLU (zh)", flush=True)
        cmmlu = evaluate_mcq(
            model, tok, args.cmmlu, device,
            lang="zh",
            max_items=args.max_items,
            max_new_tokens=args.max_new_tokens,
            verbose=not args.quiet,
        )
        print(f"  -> CMMLU acc = {cmmlu['accuracy'] * 100:.2f}%  "
              f"({cmmlu['correct']}/{cmmlu['n']}, no_answer={cmmlu['no_answer']}, "
              f"{cmmlu['elapsed_s']:.1f}s)", flush=True)
        for s, (c, t, na, a) in sorted(cmmlu["by_subject"].items()):
            print(f"      - {s:<40s} {c}/{t} (NA={na}) = {a * 100:.2f}%")

        results[name] = {
            "mmlu": mmlu,
            "cmmlu": cmmlu,
            "config": cfg,
            "ckpt_path": str(path),
        }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if results:
        print_comparison(results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = args.report_dir / f"results_sft_{ts}.md"
        write_markdown(out_path, tokenizer_dir, results, args)


if __name__ == "__main__":
    main()
