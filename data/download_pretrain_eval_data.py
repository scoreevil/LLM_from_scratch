"""Phase 5.5 - Evaluation dataset preparation.

Builds three eval artifacts under ``data/eval/`` (created if missing):

    mmlu_subset.jsonl        English knowledge MCQ      cais/mmlu (test)
    cmmlu_subset.jsonl       Chinese knowledge MCQ      haonan-li/cmmlu (test)
    translation_pairs.jsonl  ZH-EN sentence pairs       wmt19 (fallback: opus-100)

All three converge on a unified JSONL schema so downstream eval code can read
them with one loader.

MCQ row:
    {"question": str,
     "choices":  {"A": str, "B": str, "C": str, "D": str},
     "answer":   "A"|"B"|"C"|"D",          # cais/mmlu's int 0..3 is normalised
     "subject":  str,
     "lang":     "zh"|"en",
     "source":   "cais/mmlu" | "haonan-li/cmmlu"}

Translation row:
    {"zh": str, "en": str, "source": str}

We deliberately pick PARALLEL subjects across MMLU and CMMLU so cross-lingual
knowledge comparisons are meaningful at the same cognitive task type:
    computer_science  +  high_school_mathematics  +  world_history
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent / "eval"

# Plain "world_history" does NOT exist in cais/mmlu — only the high-school /
# US / European variants. Use the explicit name.
MMLU_SUBJECTS = [
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_world_history",
]


@dataclass
class ZhMcqSource:
    """One candidate Chinese MCQ source. Tried top-to-bottom; first non-empty wins."""
    name: str
    subjects: List[str]
    split: str
    load_kwargs: dict = field(default_factory=dict)


# Chinese MCQ fallback chain. The original cmmlu path needs trust_remote_code
# under datasets >= 4.x. If that's still blocked we fall back to a community
# mirror, then to C-Eval (different subject names, kept ~parallel by topic).
ZH_MCQ_CANDIDATES = [
    ZhMcqSource(
        name="haonan-li/cmmlu",
        subjects=["computer_science", "high_school_mathematics", "world_history"],
        split="test",
        load_kwargs={"trust_remote_code": True},
    ),
    ZhMcqSource(
        name="m-a-p/CMMLU",
        subjects=["computer_science", "high_school_mathematics", "world_history"],
        split="test",
        load_kwargs={"trust_remote_code": True},
    ),
    ZhMcqSource(
        # C-Eval: gold-standard Chinese MCQ benchmark. 'val' split is the only
        # one with answer labels (test is closed). Picked subjects roughly
        # parallel to the MMLU triplet (CS / Math / History).
        name="ceval/ceval-exam",
        subjects=["computer_network", "high_school_mathematics", "modern_chinese_history"],
        split="val",
    ),
]


# Translation-pair source fallback chain. Tried top-to-bottom; first one that
# yields rows wins. Each tuple: (hf_name, config, split, zh_key, en_key).
TRANSLATION_SOURCES = [
    ("wmt19",               "zh-en", "validation", "zh", "en"),
    ("Helsinki-NLP/opus-100", "en-zh", "test",     "zh", "en"),
]


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------
def normalize_mmlu(row: dict, subject: str) -> dict:
    """cais/mmlu schema -> unified MCQ."""
    choices = row.get("choices") or []
    ans = row.get("answer")
    if isinstance(ans, int):
        ans_letter = "ABCD"[ans]
    else:
        ans_letter = str(ans).strip().upper()[:1]
    # Pad to 4 in case a row is malformed.
    while len(choices) < 4:
        choices = list(choices) + [""]
    return {
        "question": row.get("question", ""),
        "choices": {"A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3]},
        "answer": ans_letter,
        "subject": subject,
        "lang": "en",
        "source": "cais/mmlu",
    }


def normalize_zh_mcq(row: dict, subject: str, source_tag: str) -> dict:
    """Schema-agnostic Chinese MCQ normalizer.

    Handles both ``haonan-li/cmmlu`` / ``m-a-p/CMMLU`` (capitalised keys
    Question/A/B/C/D/Answer) and ``ceval/ceval-exam`` (lowercase
    question/A/B/C/D/answer) with one function.
    """
    def pick(*keys: str) -> str:
        for k in keys:
            v = row.get(k)
            if v:
                return v
        return ""

    return {
        "question": pick("Question", "question"),
        "choices": {
            "A": pick("A", "a"),
            "B": pick("B", "b"),
            "C": pick("C", "c"),
            "D": pick("D", "d"),
        },
        "answer": pick("Answer", "answer").strip().upper()[:1],
        "subject": subject,
        "lang": "zh",
        "source": source_tag,
    }


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------
def fetch_mcq_one_source(
    dataset_name: str,
    subjects: List[str],
    split: str,
    per_subject_limit: Optional[int],
    normalize_fn: Callable[[dict, str], dict],
    load_kwargs: Optional[dict] = None,
) -> List[dict]:
    """Pull each subject's split and normalise. Skip on per-subject errors."""
    out: List[dict] = []
    load_kwargs = load_kwargs or {}
    for subj in subjects:
        try:
            ds = load_dataset(dataset_name, subj, split=split, **load_kwargs)
        except Exception as exc:
            print(f"[warn] {dataset_name}:{subj}/{split} failed to load: {exc}",
                  file=sys.stderr, flush=True)
            continue
        total = len(ds)
        n = min(per_subject_limit, total) if per_subject_limit else total
        print(f"  {dataset_name}:{subj}/{split} -> taking {n}/{total} items", flush=True)
        for i in range(n):
            row = ds[i]
            try:
                out.append(normalize_fn(row, subj))
            except Exception as exc:
                print(f"  [skip] {subj}#{i}: {exc}", file=sys.stderr)
    return out


def fetch_zh_mcq_with_fallback(
    candidates: List[ZhMcqSource],
    per_subject_limit: Optional[int],
) -> List[dict]:
    """Walk the candidate list; return the first source that yields any rows."""
    for cand in candidates:
        print(f"[zh-mcq] trying {cand.name}  subjects={cand.subjects}", flush=True)
        items = fetch_mcq_one_source(
            dataset_name=cand.name,
            subjects=cand.subjects,
            split=cand.split,
            per_subject_limit=per_subject_limit,
            normalize_fn=lambda r, s, tag=cand.name: normalize_zh_mcq(r, s, tag),
            load_kwargs=cand.load_kwargs,
        )
        if items:
            print(f"[zh-mcq] -> using {cand.name} ({len(items)} items)", flush=True)
            return items
        print(f"[zh-mcq] {cand.name} yielded 0 items, trying next candidate...",
              flush=True)
    print("[zh-mcq] WARN: all Chinese MCQ candidates failed; output will be empty",
          file=sys.stderr, flush=True)
    return []


def fetch_translation_pairs(n_pairs: int) -> List[dict]:
    """Try each translation source in order; return the first one that works."""
    for name, config, split, zh_key, en_key in TRANSLATION_SOURCES:
        try:
            ds = load_dataset(name, config, split=split)
        except Exception as exc:
            print(f"[warn] translation source {name}/{config}/{split} failed: {exc}",
                  file=sys.stderr, flush=True)
            continue

        n = min(n_pairs, len(ds))
        out: List[dict] = []
        for i in range(n):
            row = ds[i]
            # WMT/opus both expose {"translation": {"zh": ..., "en": ...}}.
            t = row.get("translation") if isinstance(row, dict) else None
            if isinstance(t, dict):
                zh = (t.get(zh_key) or "").strip()
                en = (t.get(en_key) or "").strip()
            else:
                zh = (row.get(zh_key) or "").strip()
                en = (row.get(en_key) or "").strip()
            if zh and en:
                out.append({"zh": zh, "en": en, "source": f"{name}/{config}"})
        if out:
            print(f"  translation -> {name}/{config}/{split}: {len(out)} pairs", flush=True)
            return out
        print(f"[warn] {name}/{config}/{split} yielded zero usable pairs",
              file=sys.stderr, flush=True)

    print("[warn] all translation sources failed; translation_pairs.jsonl will be empty",
          file=sys.stderr, flush=True)
    return []


# ---------------------------------------------------------------------------
# IO + reporting
# ---------------------------------------------------------------------------
def save_jsonl(items: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _by_subject(items: List[dict]) -> dict:
    out: dict = {}
    for r in items:
        s = r.get("subject", "?")
        out[s] = out.get(s, 0) + 1
    return out


def report(mmlu: List[dict], cmmlu: List[dict], pairs: List[dict], out_dir: Path) -> None:
    print()
    print("=" * 60)
    print("Phase 5.5 eval datasets summary")
    print("=" * 60)
    print(f"  out_dir                 : {out_dir}")
    print(f"  mmlu_subset.jsonl       : {len(mmlu)} items")
    for s, c in _by_subject(mmlu).items():
        print(f"     - {s:<35s} {c}")
    print(f"  cmmlu_subset.jsonl      : {len(cmmlu)} items")
    for s, c in _by_subject(cmmlu).items():
        print(f"     - {s:<35s} {c}")
    print(f"  translation_pairs.jsonl : {len(pairs)} pairs")
    if pairs:
        srcs = {p.get("source", "?") for p in pairs}
        print(f"     - source(s)            {sorted(srcs)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=None,
                    help="Cap each subject to N items (default: take the entire test split).")
    ap.add_argument("--translation-pairs", type=int, default=500,
                    help="Number of zh-en sentence pairs to keep (default 500).")
    ap.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--mmlu-subjects", nargs="+", default=MMLU_SUBJECTS)
    args = ap.parse_args()

    print(f"[mmlu]  subjects: {args.mmlu_subjects}", flush=True)
    mmlu = fetch_mcq_one_source(
        dataset_name="cais/mmlu",
        subjects=args.mmlu_subjects,
        split="test",
        per_subject_limit=args.per_subject,
        normalize_fn=normalize_mmlu,
    )
    save_jsonl(mmlu, args.out_dir / "mmlu_subset.jsonl")

    cmmlu = fetch_zh_mcq_with_fallback(ZH_MCQ_CANDIDATES, args.per_subject)
    save_jsonl(cmmlu, args.out_dir / "cmmlu_subset.jsonl")

    print(f"[trans] target pairs: {args.translation_pairs}", flush=True)
    pairs = fetch_translation_pairs(args.translation_pairs)
    save_jsonl(pairs, args.out_dir / "translation_pairs.jsonl")

    report(mmlu, cmmlu, pairs, args.out_dir)


if __name__ == "__main__":
    main()
