"""Back-test the Phase-0 chars/token heuristic against the real BBPE tokenizer.

For each subset under data/processed/:
    - take a balanced per-language sample
    - encode it with the trained BBPE
    - compute real chars/token, compare to the heuristic constants
    - extrapolate the real total-token count of the file using
      sampled chars/token + a full-file char scan
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Allow `from bpe import ...` whether run from project root or tokenizer/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpe import BBPETokenizer  # noqa: E402


# Same heuristic constants as data/download.py used at Phase 0.
HEURISTIC = {"zh": 1.5, "en": 4.0}


def sample_tokens(path: Path, tok: BBPETokenizer, per_lang: int):
    """Read until each language has ``per_lang`` docs encoded."""
    stats: defaultdict[str, dict] = defaultdict(lambda: {"chars": 0, "tokens": 0, "docs": 0})
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            done = all(stats[lg]["docs"] >= per_lang for lg in ("zh", "en"))
            if done:
                break
            rec = json.loads(line)
            lang = rec.get("lang")
            text = rec.get("text", "")
            if lang not in HEURISTIC or stats[lang]["docs"] >= per_lang or not text:
                continue
            ids = tok.encode(text)
            stats[lang]["chars"] += len(text)
            stats[lang]["tokens"] += len(ids)
            stats[lang]["docs"] += 1
    return stats


def scan_full_chars(path: Path):
    """Single pass over the file to get true per-language char totals."""
    totals = defaultdict(lambda: {"chars": 0, "docs": 0})
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lang = rec.get("lang")
            text = rec.get("text", "")
            if lang and text:
                totals[lang]["chars"] += len(text)
                totals[lang]["docs"] += 1
    return totals


def report_file(fname: str, sample_stats: dict, full_stats: dict, heur_total: int):
    print(f"\n=== {fname} ===")
    print(f"  {'lang':<5} {'sample_docs':>11}  {'chars/tok (real)':>16}  "
          f"{'heur':>6}  {'err%':>7}  {'avg tok/doc':>12}")

    cpt_by_lang: dict[str, float] = {}
    for lang in ("zh", "en"):
        s = sample_stats.get(lang)
        if not s or s["docs"] == 0:
            continue
        cpt_real = s["chars"] / s["tokens"]
        cpt_by_lang[lang] = cpt_real
        heur = HEURISTIC[lang]
        err = (heur - cpt_real) / cpt_real * 100
        avg = s["tokens"] / s["docs"]
        print(f"  {lang:<5} {s['docs']:>11d}  {cpt_real:>16.3f}  "
              f"{heur:>6.2f}  {err:>+6.1f}%  {avg:>12.1f}")

    # Per-doc-balanced zh token share in the sample (a doc-length comparison).
    zh_tok = sample_stats.get("zh", {}).get("tokens", 0)
    en_tok = sample_stats.get("en", {}).get("tokens", 0)
    if zh_tok + en_tok > 0:
        print(f"  balanced-sample zh tok share : {zh_tok / (zh_tok + en_tok) * 100:.2f}%")

    # File-level projections need the full-file char totals.
    if not full_stats:
        print("  (skip full-file projection; pass without --no-full-scan to compute)")
        return

    file_real_tokens = {}
    for lang in ("zh", "en"):
        full_chars = full_stats.get(lang, {}).get("chars", 0)
        cpt = cpt_by_lang.get(lang)
        if full_chars and cpt:
            file_real_tokens[lang] = full_chars / cpt

    file_total = sum(file_real_tokens.values())
    if file_total > 0:
        share_zh = file_real_tokens.get("zh", 0) / file_total
        print(f"  file-level real zh tok share : {share_zh * 100:.2f}%")
        delta = (file_total - heur_total) / heur_total * 100
        print(f"  full-file real total tokens  : {file_total / 1e6:.2f}M  "
              f"(heuristic said {heur_total / 1e6:.0f}M, delta {delta:+.1f}%)")

    print(f"  full-file docs               : "
          f"zh={full_stats.get('zh', {}).get('docs', 0):,}  "
          f"en={full_stats.get('en', {}).get('docs', 0):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Tokenizer directory path. If omitted, uses tokenizer/<--tokenizer-size>.",
    )
    ap.add_argument(
        "--tokenizer-size",
        default="8k",
        help="Tokenizer variant under tokenizer/. e.g. 8k, 32k",
    )
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "processed")
    ap.add_argument("--samples-per-lang", type=int, default=2000,
                    help="Number of docs per language to encode (default 2000).")
    ap.add_argument("--files", nargs="+",
                    default=["mix_1to1.jsonl", "mix_1to2.jsonl"])
    ap.add_argument("--heur-total", type=int, default=300_000_000,
                    help="The heuristic token target the Phase-0 script used (default 300M).")
    ap.add_argument("--no-full-scan", action="store_true",
                    help="Skip the full-file char scan (faster, but no file-level projection).")
    args = ap.parse_args()

    tokenizer_dir = args.tokenizer_dir or (Path(__file__).resolve().parent / args.tokenizer_size)
    tok = BBPETokenizer.load(tokenizer_dir)
    print(f"Loaded tokenizer from {tokenizer_dir}  (vocab={tok.vocab_size})")

    for fname in args.files:
        fpath = args.data_dir / fname
        if not fpath.exists():
            print(f"\n[skip] {fpath} not found")
            continue

        t0 = time.time()
        sample_stats = sample_tokens(fpath, tok, args.samples_per_lang)
        t_sample = time.time() - t0

        if args.no_full_scan:
            full_stats = {}
        else:
            t0 = time.time()
            full_stats = scan_full_chars(fpath)
            t_scan = time.time() - t0
            print(f"\n[{fname}] sample {t_sample:.1f}s, full-scan {t_scan:.1f}s")

        report_file(fname, sample_stats, full_stats, args.heur_total)


if __name__ == "__main__":
    main()
