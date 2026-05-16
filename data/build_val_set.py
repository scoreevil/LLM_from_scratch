"""Phase 4.5 - Held-out bilingual validation set builder.

Streams the same two corpora used in Phase 0 (FineWeb-Edu, SkyPile-150B), but
*skips past the prefix consumed during training-set construction* so the val
set is strictly disjoint from mix_1to1.jsonl and mix_1to2.jsonl.

Why this approach: previous attempts using wikitext + wikimedia/wikipedia
were either OOV-heavy (Traditional Chinese characters that the BBPE-on-
SimplifiedChinese tokenizer doesn't have) or unavailable on the Hub
(liweili/c4_zh removed). Reusing the training sources keeps the script
distribution-aligned with training (so val PPL is meaningful) while the
skip guarantees zero document-level overlap with the training subsets.

Skip computation (filtered-doc count, matching Phase 0's filter exactly):
    ZH (SkyPile-150B):  mix_1to1 consumed 226,517 zh docs (max of the two).
                        skip 250,000 -> ~10% margin.
    EN (FineWeb-Edu):   mix_1to2 consumed 167,866 en docs (max).
                        skip 200,000 -> ~19% margin.

Sources:
    EN: HuggingFaceFW/fineweb-edu  (config: sample-10BT)   tail after 200K skip
    ZH: Skywork/SkyPile-150B                                tail after 250K skip

Token counts here use the same chars/token heuristic as data/download.py so
the val budget is directly comparable to the training subsets. Swap in real
BBPE counts later if you want exact PPL token denominators.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config (intentionally duplicated from download.py — kept in sync manually).
# ---------------------------------------------------------------------------
CHARS_PER_TOKEN_EN = 4.0
CHARS_PER_TOKEN_ZH = 1.5

EN_DATASET = "HuggingFaceFW/fineweb-edu"
EN_CONFIG = "sample-10BT"
ZH_DATASET = "Skywork/SkyPile-150B"
ZH_CONFIG: str | None = None

# Skip past Phase 0's max consumption per stream — guarantees val docs are
# disjoint from mix_1to1.jsonl and mix_1to2.jsonl. See module docstring.
EN_SKIP = 200_000   # Phase 0 max EN consumption = 167,866
ZH_SKIP = 250_000   # Phase 0 max ZH consumption = 226,517

OUT_PATH = Path(__file__).resolve().parent / "processed" / "val_set.jsonl"

# Minimum chars to *yield* (quality filter). Skip-phase uses Phase 0's exact
# filter ("if not text") so yielded counts stay aligned across both scripts.
MIN_CHARS = 32


@dataclass
class Source:
    hf_name: str
    config: str | None
    lang: str            # "zh" / "en"
    source_tag: str      # short label written into each record
    skip_first: int = 0  # number of post-filter docs to discard before yielding


SOURCES = {
    "en": Source(hf_name=EN_DATASET, config=EN_CONFIG, lang="en",
                 source_tag="fineweb-edu-tail", skip_first=EN_SKIP),
    "zh": Source(hf_name=ZH_DATASET, config=ZH_CONFIG, lang="zh",
                 source_tag="skypile-150b-tail", skip_first=ZH_SKIP),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def estimate_tokens(text: str, lang: str) -> int:
    """Same heuristic as data/download.py - keeps val/train budgets comparable."""
    if not text:
        return 0
    ratio = CHARS_PER_TOKEN_ZH if lang == "zh" else CHARS_PER_TOKEN_EN
    return max(1, int(len(text) / ratio))


def iter_source(src: Source) -> Iterator[dict]:
    """Yield {'text', 'lang', 'source'} records from a streaming HF dataset.

    Two-phase logic:
      1. Skip ``src.skip_first`` docs using Phase-0's exact filter
         (``if not text: continue``), so yielded counts align across
         download.py and this script — guarantees no overlap with
         mix_1to1.jsonl / mix_1to2.jsonl.
      2. Then yield docs with a stricter ``len(text) >= MIN_CHARS`` filter
         for val-quality reasons.
    """
    kwargs = {"split": "train", "streaming": True}
    if src.config is not None:
        ds = load_dataset(src.hf_name, src.config, **kwargs)
    else:
        ds = load_dataset(src.hf_name, **kwargs)

    raw_iter = iter(ds)

    # Phase 1: skip past Phase 0's consumed prefix.
    if src.skip_first > 0:
        bar = tqdm(
            total=src.skip_first,
            desc=f"skip-{src.lang}",
            unit="doc",
            mininterval=1.0,
            dynamic_ncols=True,
        )
        skipped = 0
        for row in raw_iter:
            text = row.get("text") if isinstance(row, dict) else None
            if not text:
                continue
            skipped += 1
            bar.update(1)
            if skipped >= src.skip_first:
                break
        bar.close()
        if skipped < src.skip_first:
            print(
                f"[warn] {src.lang} stream exhausted during skip "
                f"({skipped}/{src.skip_first}); val will be empty",
                flush=True,
            )

    # Phase 2: yield post-skip docs that pass the quality threshold.
    for row in raw_iter:
        if not isinstance(row, dict):
            continue
        text = row.get("text") or ""
        text = text.strip()
        if len(text) < MIN_CHARS:
            continue
        yield {"text": text, "lang": src.lang, "source": src.source_tag}


def build_val_set(
    en_stream: Iterable[dict],
    zh_stream: Iterable[dict],
    out_path: Path,
    target_tokens: int,
    zh_ratio: float = 0.5,
    log_every: int = 1000,
) -> dict:
    """Stream both sources until each per-language token budget is met."""
    target_zh = int(target_tokens * zh_ratio)
    target_en = target_tokens - target_zh

    en_iter = iter(en_stream)
    zh_iter = iter(zh_stream)

    n_zh = n_en = 0
    tok_zh = tok_en = 0
    started = time.time()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        while tok_zh < target_zh or tok_en < target_en:
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
                print(f"[warn] {pick} stream exhausted before budget met", file=sys.stderr)
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
                    f"  docs={total_docs:>7d}  "
                    f"zh_tok={tok_zh/1e6:.2f}M/{target_zh/1e6:.2f}M  "
                    f"en_tok={tok_en/1e6:.2f}M/{target_en/1e6:.2f}M  "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )

    return {
        "path": str(out_path),
        "n_docs": n_zh + n_en,
        "n_zh": n_zh,
        "n_en": n_en,
        "tok_zh": tok_zh,
        "tok_en": tok_en,
        "tok_total": tok_zh + tok_en,
        "zh_doc_ratio": n_zh / max(1, n_zh + n_en),
        "zh_tok_ratio": tok_zh / max(1, tok_zh + tok_en),
        "elapsed": time.time() - started,
    }


def report(stats: dict) -> None:
    print("\n" + "=" * 60)
    print("Phase 4.5 validation set summary")
    print("=" * 60)
    print(f"  path        : {stats['path']}")
    print(f"  sources     : EN = {SOURCES['en'].hf_name} ({SOURCES['en'].config})  "
          f"skip={SOURCES['en'].skip_first:,}")
    print(f"                ZH = {SOURCES['zh'].hf_name}  "
          f"skip={SOURCES['zh'].skip_first:,}")
    print(f"  disjoint    : skip > Phase 0 max consumption (167866 EN / 226517 ZH), "
          f"so no overlap with mix_1to1.jsonl or mix_1to2.jsonl")
    print(f"  documents   : {stats['n_docs']:,}  "
          f"(zh={stats['n_zh']:,}, en={stats['n_en']:,})")
    print(f"  zh share    : {stats['zh_doc_ratio']*100:.2f}% by docs, "
          f"{stats['zh_tok_ratio']*100:.2f}% by tokens")
    print(f"  est tokens  : {stats['tok_total']/1e6:.2f}M  "
          f"(zh={stats['tok_zh']/1e6:.2f}M, en={stats['tok_en']/1e6:.2f}M)")
    print(f"  elapsed     : {stats['elapsed']:.1f}s")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-tokens", type=int, default=10_000_000,
                    help="Target estimated tokens for the whole val set (default 10M).")
    ap.add_argument("--zh-ratio", type=float, default=0.5,
                    help="Target zh token share (default 0.5 for 1:1).")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run (~100K tokens total) to verify the pipeline end-to-end.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-path", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    random.seed(args.seed)
    total = 100_000 if args.smoke else args.total_tokens

    print(f"[info] target tokens   = {total:,}")
    print(f"[info] zh ratio        = {args.zh_ratio:.3f}")
    print(f"[info] out path        = {args.out_path}")
    print(f"[info] EN source       = {SOURCES['en'].hf_name} ({SOURCES['en'].config})")
    print(f"[info] ZH source       = {SOURCES['zh'].hf_name}")

    en_stream = iter_source(SOURCES["en"])
    zh_stream = iter_source(SOURCES["zh"])

    stats = build_val_set(
        en_stream=en_stream,
        zh_stream=zh_stream,
        out_path=args.out_path,
        target_tokens=total,
        zh_ratio=args.zh_ratio,
    )
    report(stats)


if __name__ == "__main__":
    main()
