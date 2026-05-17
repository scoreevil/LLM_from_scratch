#!/usr/bin/env python3
"""Batch generation tests: 10 cases x 4 groups.

Usage:
    python eval/test_generate.py
    MAX_NEW_TOKENS=120 OUT_DIR=eval/generation_tests python eval/test_generate.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import time


def md_escape(text: str) -> str:
    text = text.replace("|", "\\|")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", "<br>")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    out_dir = Path(os.getenv("OUT_DIR", str(root / "eval" / "generation_tests")))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "80"))
    timeout_s = int(os.getenv("TIMEOUT_S", "180"))

    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "PT-1A": root / "training" / "checkpoints" / "PT-1A" / "best.pt",
        "PT-1B": root / "training" / "checkpoints" / "PT-1B" / "best.pt",
        "PT-2A": root / "training" / "checkpoints" / "PT-2A" / "best.pt",
        "PT-2B": root / "training" / "checkpoints" / "PT-2B" / "best.pt",
    }
    # Verified from training/run_pretrain_experiments.sh:
    # PT-1A/1B -> n_heads=12, PT-2A -> 8, PT-2B -> 16.
    n_heads_by_group = {
        "PT-1A": 12,
        "PT-1B": 12,
        "PT-2A": 8,
        "PT-2B": 16,
    }

    if not any(path.exists() for path in checkpoints.values()):
        print("[fatal] no available checkpoint found under training/checkpoints/<group>/best.pt")
        return 1

    cases = [
        {"id": "C01", "temp": 0.2, "top_k": 20, "top_p": 0.90, "prompt": "The capital of France is"},
        {"id": "C02", "temp": 0.7, "top_k": 50, "top_p": 0.95, "prompt": "Write a short paragraph about machine learning."},
        {"id": "C03", "temp": 1.0, "top_k": 100, "top_p": 0.95, "prompt": "In one sentence, explain overfitting."},
        {"id": "C04", "temp": 0.3, "top_k": 40, "top_p": 0.85, "prompt": "问题：请用一句话解释什么是神经网络。答案："},
        {"id": "C05", "temp": 0.8, "top_k": 80, "top_p": 0.92, "prompt": "请写三点关于高效学习的建议："},
        {"id": "C06", "temp": 0.6, "top_k": 30, "top_p": 0.80, "prompt": "Translate to Chinese: The weather is nice today."},
        {"id": "C07", "temp": 0.9, "top_k": 0, "top_p": 0.90, "prompt": "List 5 common sorting algorithms:"},
        {"id": "C08", "temp": 0.4, "top_k": 10, "top_p": 1.00, "prompt": "Python function to compute fibonacci numbers:"},
        {"id": "C09", "temp": 1.2, "top_k": 120, "top_p": 0.98, "prompt": "Continue the story: On the edge of the galaxy,"},
        {"id": "C10", "temp": 0.5, "top_k": 60, "top_p": 0.88, "prompt": "请续写：人工智能正在改变世界，"},
    ]

    group_results: dict[str, list[dict[str, str]]] = {g: [] for g in checkpoints}
    available_groups = [g for g, p in checkpoints.items() if p.exists()]
    missing_groups = [g for g, p in checkpoints.items() if not p.exists()]

    for group, ckpt_path in checkpoints.items():
        n_heads = n_heads_by_group[group]
        for case in cases:
            case_label = f"{case['id']} (t={case['temp']:.2f}, k={case['top_k']}, p={case['top_p']:.2f})"
            row = {
                "case": case_label,
                "prompt": case["prompt"],
                "status": "ok",
                "output": "",
            }
            if not ckpt_path.exists():
                row["status"] = "missing_checkpoint"
                row["output"] = f"[skip] checkpoint not found: {ckpt_path}"
                group_results[group].append(row)
                continue

            cmd = [
                sys.executable,
                "eval/generate.py",
                "--checkpoint",
                str(ckpt_path),
                "--prompt",
                case["prompt"],
                "--max-new-tokens",
                str(max_new_tokens),
                "--temperature",
                str(case["temp"]),
                "--top-k",
                str(case["top_k"]),
                "--top-p",
                str(case["top_p"]),
                "--n-heads",
                str(n_heads),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_s,
                )
                combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
                row["status"] = "ok" if proc.returncode == 0 else f"error({proc.returncode})"
                row["output"] = combined.strip()
            except subprocess.TimeoutExpired as exc:
                partial = (exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")
                row["status"] = "timeout"
                row["output"] = (partial + "\n[timeout]").strip()
            except Exception as exc:  # pragma: no cover
                row["status"] = "exception"
                row["output"] = f"[exception] {type(exc).__name__}: {exc}"

            group_results[group].append(row)

    md_name = "results_" + f"{time.strftime('%Y%m%d_%H%M%S')}.md"
    md_path = out_dir / md_name
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Generation comparison by group\n\n")
        f.write(f"- max_new_tokens: `{max_new_tokens}`\n")
        f.write(f"- timeout per run: `{timeout_s}s`\n")
        f.write("- n_heads mapping: `PT-1A=12`, `PT-1B=12`, `PT-2A=8`, `PT-2B=16`\n")
        f.write("- available groups: " + ", ".join(f"`{k}`" for k in available_groups) + "\n")
        if missing_groups:
            f.write("- missing checkpoints: " + ", ".join(f"`{k}`" for k in missing_groups) + "\n")
        f.write("\n")
        for group in checkpoints:
            f.write(f"## {group}\n\n")
            f.write(
                f"- checkpoint: `{checkpoints[group]}`\n"
                f"- n_heads used: `{n_heads_by_group[group]}`\n\n"
            )
            f.write("| Case | Prompt | Status | Output |\n")
            f.write("|---|---|---|---|\n")
            for row in group_results[group]:
                case_txt = md_escape(row["case"])
                prompt_txt = md_escape(row["prompt"])
                status_txt = md_escape(row["status"])
                out_txt = md_escape(row["output"]) if row["output"] else "(empty)"
                f.write(f"| {case_txt} | {prompt_txt} | {status_txt} | {out_txt} |\n")
            f.write("\n")

    print(f"[done] wrote: {md_path.relative_to(root)}")
    print("[done] available groups:", ", ".join(available_groups) if available_groups else "(none)")
    if missing_groups:
        print("[done] missing groups:", ", ".join(missing_groups))
    print("\nRun summary:")
    print("  python eval/test_generate.py")
    print("  MAX_NEW_TOKENS=120 OUT_DIR=eval/generation_tests python eval/test_generate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
