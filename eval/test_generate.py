#!/usr/bin/env python3
"""Batch generation tests: 10 cases x available checkpoints.

Usage:
    python eval/test_generate.py
    N_HEADS=4 MAX_NEW_TOKENS=120 OUT_DIR=eval/generation_tests python eval/test_generate.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def preview_text(text: str, max_len: int = 180) -> str:
    one_line = " ".join(text.strip().split())
    if not one_line:
        return ""
    return one_line[: max_len - 1] + "..." if len(one_line) > max_len else one_line


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    out_dir = Path(os.getenv("OUT_DIR", str(root / "eval" / "generation_tests")))
    n_heads = int(os.getenv("N_HEADS", "4"))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "80"))
    timeout_s = int(os.getenv("TIMEOUT_S", "180"))

    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "PT-1A": root / "training" / "checkpoints" / "PT-1A" / "best.pt",
        "PT-1B": root / "training" / "checkpoints" / "PT-1B" / "best.pt",
        "PT-2A": root / "training" / "checkpoints" / "PT-2A" / "best.pt",
        "PT-2B": root / "training" / "checkpoints" / "PT-2B" / "best.pt",
    }
    available_checkpoints = {name: path for name, path in checkpoints.items() if path.exists()}
    missing_checkpoints = {name: path for name, path in checkpoints.items() if not path.exists()}

    if not available_checkpoints:
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

    records = []
    raw_dir = out_dir / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        for model_name, ckpt_path in available_checkpoints.items():
            rec = {
                "case_id": case["id"],
                "model": model_name,
                "temperature": case["temp"],
                "top_k": case["top_k"],
                "top_p": case["top_p"],
                "prompt": case["prompt"],
                "checkpoint": str(ckpt_path),
                "status": "ok",
                "return_code": 0,
                "output_preview": "",
                "output_file": "",
            }

            out_file = raw_dir / f"{case['id']}_{model_name}.txt"
            rec["output_file"] = str(out_file.relative_to(root))

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
                out_file.write_text(combined, encoding="utf-8")
                rec["return_code"] = proc.returncode
                rec["status"] = "ok" if proc.returncode == 0 else "error"
                rec["output_preview"] = preview_text(combined)
            except subprocess.TimeoutExpired as exc:
                partial = (exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")
                out_file.write_text(partial + "\n[timeout]\n", encoding="utf-8")
                rec["status"] = "timeout"
                rec["return_code"] = 124
                rec["output_preview"] = "[timeout]"
            except Exception as exc:  # pragma: no cover
                msg = f"[exception] {type(exc).__name__}: {exc}"
                out_file.write_text(msg + "\n", encoding="utf-8")
                rec["status"] = "exception"
                rec["return_code"] = 1
                rec["output_preview"] = msg

            records.append(rec)

    jsonl_path = out_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "model",
                "temperature",
                "top_k",
                "top_p",
                "prompt",
                "status",
                "return_code",
                "output_preview",
                "output_file",
                "checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    md_path = out_dir / "results.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Generation comparison (10 cases x {len(available_checkpoints)} models)\n\n")
        f.write(f"- n_heads: `{n_heads}`\n")
        f.write(f"- max_new_tokens: `{max_new_tokens}`\n")
        f.write(f"- timeout per run: `{timeout_s}s`\n")
        f.write("- evaluated models: " + ", ".join(f"`{k}`" for k in available_checkpoints) + "\n")
        if missing_checkpoints:
            f.write("- skipped missing checkpoints: " + ", ".join(f"`{k}`" for k in missing_checkpoints) + "\n")
        f.write("\n")
        f.write("| Case | Model | Temp | Top-k | Top-p | Status | Output Preview |\n")
        f.write("|---|---|---:|---:|---:|---|---|\n")
        for r in records:
            preview = r["output_preview"].replace("|", "\\|")
            f.write(
                f"| {r['case_id']} | {r['model']} | {r['temperature']:.2f} | "
                f"{r['top_k']} | {r['top_p']:.2f} | {r['status']} | {preview} |\n"
            )

    print(f"[done] wrote: {md_path.relative_to(root)}")
    print(f"[done] wrote: {csv_path.relative_to(root)}")
    print(f"[done] wrote: {jsonl_path.relative_to(root)}")
    print(f"[done] raw outputs: {raw_dir.relative_to(root)}")
    print("[done] evaluated models:", ", ".join(available_checkpoints))
    if missing_checkpoints:
        print("[done] skipped missing:", ", ".join(missing_checkpoints))
    print("\nRun summary:")
    print("  python eval/test_generate.py")
    print("  N_HEADS=4 MAX_NEW_TOKENS=120 OUT_DIR=eval/generation_tests python eval/test_generate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
