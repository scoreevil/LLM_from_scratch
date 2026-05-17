"""Phase 6.5 - Bilingual SFT data preparation.

Pulls two Alpaca-style instruction datasets, normalises them into ChatML
``messages`` format, mixes ~equal-sized zh/en pools, shuffles, and splits 9:1
into:
    data/processed/sft_train.jsonl
    data/processed/sft_val.jsonl

Sources (Alpaca family — homogeneous schema makes the converter trivial):
    EN: tatsu-lab/alpaca
    ZH: shibing624/alpaca-zh

Output schema (ChatML-compatible):
    {
      "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
      ],
      "lang":   "zh" | "en",
      "source": "<dataset>",
    }
``lang`` and ``source`` are not part of the ChatML spec; they're attached for
downstream analysis (per-language SFT loss, dataset attribution).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from datasets import load_dataset


PROCESSED_DIR = Path(__file__).resolve().parent / "processed"


# Default system prompts. Match the response language to avoid accidentally
# steering outputs across languages.
SYSTEM_EN = "You are a helpful assistant."
SYSTEM_ZH = "你是一个乐于助人的助手。"


@dataclass
class SftSource:
    hf_name: str
    config: str | None
    split: str
    lang: str
    source_tag: str
    system_prompt: str
    load_kwargs: dict


SOURCES: List[SftSource] = [
    SftSource(
        hf_name="tatsu-lab/alpaca",
        config=None,
        split="train",
        lang="en",
        source_tag="alpaca",
        system_prompt=SYSTEM_EN,
        load_kwargs={},
    ),
    SftSource(
        hf_name="shibing624/alpaca-zh",
        config=None,
        split="train",
        lang="zh",
        source_tag="alpaca-zh",
        system_prompt=SYSTEM_ZH,
        load_kwargs={},
    ),
]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def alpaca_row_to_chatml(row: dict, *, src: SftSource) -> dict | None:
    """Standard Alpaca normaliser.

    user content rule (Alpaca convention):
        if "input" is non-empty: user = instruction + "\\n" + input
        else:                    user = instruction
    """
    instruction = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    output = (row.get("output") or "").strip()
    if not instruction or not output:
        return None

    user_msg = f"{instruction}\n{inp}" if inp else instruction
    return {
        "messages": [
            {"role": "system",    "content": src.system_prompt},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": output},
        ],
        "lang": src.lang,
        "source": src.source_tag,
    }


def fetch_source(src: SftSource, take: int) -> List[dict]:
    """Stream a HF dataset, normalise, return up to ``take`` records."""
    print(f"[fetch] {src.hf_name} ({src.lang}) -> goal {take}", flush=True)
    try:
        ds = load_dataset(
            src.hf_name,
            src.config,
            split=src.split,
            streaming=True,
            **src.load_kwargs,
        ) if src.config else load_dataset(
            src.hf_name,
            split=src.split,
            streaming=True,
            **src.load_kwargs,
        )
    except Exception as exc:
        print(f"[fatal] failed to load {src.hf_name}: {exc}", file=sys.stderr)
        return []

    out: List[dict] = []
    seen = 0
    for row in ds:
        seen += 1
        if not isinstance(row, dict):
            continue
        rec = alpaca_row_to_chatml(row, src=src)
        if rec is None:
            continue
        out.append(rec)
        if len(out) >= take:
            break
        if seen % 5000 == 0:
            print(f"  [{src.lang}] streamed {seen}, kept {len(out)}", flush=True)
    print(f"  [{src.lang}] done: {len(out)} kept (from {seen} streamed)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Splitting + IO
# ---------------------------------------------------------------------------
def split_train_val(items: List[dict], val_ratio: float, seed: int) -> tuple[list, list]:
    """Shuffle once, then slice 9:1. Mixes zh/en into both splits."""
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    return items[n_val:], items[:n_val]


def save_jsonl(items: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _lang_breakdown(items: List[dict]) -> dict:
    out: dict = {}
    for r in items:
        lg = r.get("lang", "?")
        out[lg] = out.get(lg, 0) + 1
    return out


def report(train: List[dict], val: List[dict], rng: random.Random) -> None:
    print()
    print("=" * 68)
    print("Phase 6.5 SFT dataset summary")
    print("=" * 68)
    total = len(train) + len(val)
    print(f"  total           : {total:,}")
    print(f"  train           : {len(train):,}  -> data/processed/sft_train.jsonl")
    print(f"  val             : {len(val):,}  -> data/processed/sft_val.jsonl")

    full_breakdown = _lang_breakdown(train + val)
    train_breakdown = _lang_breakdown(train)
    val_breakdown = _lang_breakdown(val)
    print("  lang mix (full) : "
          + ", ".join(f"{k}={v} ({v/total*100:.1f}%)" for k, v in sorted(full_breakdown.items())))
    print("  lang mix (train): "
          + ", ".join(f"{k}={v}" for k, v in sorted(train_breakdown.items())))
    print("  lang mix (val)  : "
          + ", ".join(f"{k}={v}" for k, v in sorted(val_breakdown.items())))

    if not train:
        return
    sample = rng.choice(train)
    print()
    print("--- random training sample ---")
    print(f"  lang    : {sample['lang']}")
    print(f"  source  : {sample['source']}")
    print(f"  messages:")
    for m in sample["messages"]:
        body = m["content"].replace("\n", "\\n")
        if len(body) > 160:
            body = body[:160] + "..."
        print(f"    [{m['role']:<9s}] {body}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=10_000,
                    help="Records to take per source language (default 10k each -> 20k total).")
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="Fraction held out for validation (default 0.1 -> 9:1 split).")
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    pooled: List[dict] = []
    for src in SOURCES:
        pooled.extend(fetch_source(src, take=args.per_lang))

    if not pooled:
        print("[fatal] no SFT records collected — check network / dataset access", file=sys.stderr)
        sys.exit(1)

    train, val = split_train_val(pooled, args.val_ratio, seed=args.seed)
    save_jsonl(train, args.out_dir / "sft_train.jsonl")
    save_jsonl(val,   args.out_dir / "sft_val.jsonl")
    report(train, val, rng)


if __name__ == "__main__":
    main()
