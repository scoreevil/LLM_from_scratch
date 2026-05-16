"""
Phase 0 - Pretraining data download & subset builder.

Streams FineWeb-Edu (English, sample-10BT) and SkyPile-150B (Chinese) from the
HuggingFace Hub and writes two mixed JSONL subsets to data/processed/:
    - mix_1to1.jsonl   (zh : en = 1 : 1)
    - mix_1to2.jsonl   (zh : en = 1 : 2)

The BBPE tokenizer is not trained yet at Phase 0, so token counts are
estimated from character length using language-specific ratios. Once the
tokenizer is available these constants can be swapped for the real encoder.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Heuristic chars-per-token for the *future* BBPE tokenizer. Conservative
# values picked from common GPT-2/Llama BBPE behaviour. Override later when
# the real tokenizer is available.
CHARS_PER_TOKEN_EN = 4.0
CHARS_PER_TOKEN_ZH = 1.5

EN_DATASET = "HuggingFaceFW/fineweb-edu"
EN_CONFIG = "sample-10BT"
ZH_DATASET = "Skywork/SkyPile-150B"

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"


@dataclass
class SubsetSpec:
    name: str           # output filename stem
    zh_ratio: float     # share of total tokens from Chinese
    total_tokens: int   # target token budget for the whole subset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str, lang: str) -> int:
    """Cheap pre-tokenizer token estimate based on character count."""
    if not text:
        return 0
    ratio = CHARS_PER_TOKEN_ZH if lang == "zh" else CHARS_PER_TOKEN_EN
    return max(1, int(len(text) / ratio))


def iter_dataset(name: str, lang: str, config: str | None = None) -> Iterator[dict]:
    """Yield {'text', 'lang'} records from a streaming HF dataset."""
    kwargs = {"split": "train", "streaming": True}
    if config is not None:
        ds = load_dataset(name, config, **kwargs)
    else:
        ds = load_dataset(name, **kwargs)

    for row in ds:
        # Both datasets expose the document body under "text". Guard anyway.
        text = row.get("text") if isinstance(row, dict) else None
        if not text:
            continue
        yield {"text": text, "lang": lang}


def build_subset(
    spec: SubsetSpec,
    en_stream: Iterable[dict],
    zh_stream: Iterable[dict],
    out_path: Path,
    log_every: int = 5000,
) -> dict:
    """Pull from both streams until both per-language budgets are met.

    Returns a stats dict for reporting.
    """
    target_zh = int(spec.total_tokens * spec.zh_ratio)
    target_en = spec.total_tokens - target_zh

    en_iter = iter(en_stream)
    zh_iter = iter(zh_stream)

    n_zh = n_en = 0
    tok_zh = tok_en = 0
    started = time.time()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        while tok_zh < target_zh or tok_en < target_en:
            # Pick a side that still needs samples. Random draw weighted by
            # remaining budget keeps the mix balanced even with skewed doc
            # lengths.
            need_zh = max(target_zh - tok_zh, 0)
            need_en = max(target_en - tok_en, 0)
            if need_zh == 0 and need_en == 0:
                break
            if need_en == 0:
                pick = "zh"
            elif need_zh == 0:
                pick = "en"
            else:
                pick = "zh" if random.random() < (need_zh / (need_zh + need_en)) else "en"

            try:
                rec = next(zh_iter if pick == "zh" else en_iter)
            except StopIteration:
                # Stream exhausted - bail with whatever we have.
                print(f"[warn] {pick} stream exhausted early", file=sys.stderr)
                if pick == "zh":
                    target_zh = tok_zh
                else:
                    target_en = tok_en
                continue

            tks = estimate_tokens(rec["text"], rec["lang"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["lang"] == "zh":
                n_zh += 1
                tok_zh += tks
            else:
                n_en += 1
                tok_en += tks

            total_docs = n_zh + n_en
            if total_docs % log_every == 0:
                elapsed = time.time() - started
                print(
                    f"  [{spec.name}] docs={total_docs:>8d} "
                    f"zh_tok={tok_zh/1e6:.2f}M/{target_zh/1e6:.2f}M  "
                    f"en_tok={tok_en/1e6:.2f}M/{target_en/1e6:.2f}M  "
                    f"({elapsed:.1f}s)"
                )

    return {
        "path": str(out_path),
        "n_docs": n_zh + n_en,
        "n_zh": n_zh,
        "n_en": n_en,
        "tok_zh": tok_zh,
        "tok_en": tok_en,
        "tok_total": tok_zh + tok_en,
        "zh_doc_ratio": (n_zh / max(1, n_zh + n_en)),
        "zh_tok_ratio": (tok_zh / max(1, tok_zh + tok_en)),
    }


def report(stats: dict) -> None:
    print(f"\n=== {stats['path']} ===")
    print(f"  documents : {stats['n_docs']:,}  (zh={stats['n_zh']:,}, en={stats['n_en']:,})")
    print(f"  zh share  : {stats['zh_doc_ratio']*100:.2f}% by docs, "
          f"{stats['zh_tok_ratio']*100:.2f}% by tokens")
    print(f"  est tokens: {stats['tok_total']/1e6:.2f}M "
          f"(zh={stats['tok_zh']/1e6:.2f}M, en={stats['tok_en']/1e6:.2f}M)")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-tokens", type=int, default=150_000_000,
                    help="Approx total estimated tokens per subset (default 150M).")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run (~200K tokens/subset) for pipeline verification.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = ap.parse_args()

    random.seed(args.seed)
    total = 200_000 if args.smoke else args.total_tokens

    specs = [
        SubsetSpec(name="mix_1to1", zh_ratio=0.5,       total_tokens=total),
        SubsetSpec(name="mix_1to2", zh_ratio=1.0 / 3.0, total_tokens=total),
    ]

    print(f"[info] target tokens/subset = {total:,}  out_dir={args.out_dir}")
    print(f"[info] EN: {EN_DATASET} ({EN_CONFIG}) | ZH: {ZH_DATASET}")

    all_stats = []
    for spec in specs:
        print(f"\n[build] {spec.name}  zh_ratio={spec.zh_ratio:.3f}")
        # Re-open streams for each subset so the second build does not start
        # where the first one left off.
        en_stream = iter_dataset(EN_DATASET, "en", EN_CONFIG)
        zh_stream = iter_dataset(ZH_DATASET, "zh")
        stats = build_subset(spec, en_stream, zh_stream,
                             args.out_dir / f"{spec.name}.jsonl")
        all_stats.append(stats)

    print("\n" + "=" * 60)
    print("Phase 0 summary")
    print("=" * 60)
    for s in all_stats:
        report(s)


if __name__ == "__main__":
    main()
