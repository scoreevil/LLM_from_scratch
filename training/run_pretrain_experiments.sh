#!/usr/bin/env bash
# Four pretrain runs (3 full epochs each). Run from repo root.
#
# Batch sizes tuned for A800 + seq_len=512 (~6万–13万 tokens/step).
#   PT-1A/1B: 256×512 ≈ 131k tok/step  (~2.3k steps/epoch on ~150M-tok mix)
#   PT-2A:    512×512 ≈ 262k tok/step  (40M params; smallest model)
#   PT-2B:    128×512 ≈  66k tok/step  (1024-wide; conservative vs OOM)
#
# OOM? Halve --batch-size for that run only.

set -euo pipefail
cd "$(dirname "$0")/.."

# PT-1A: 768×12 (~120M) on mix 1:1
python training/pretrain.py --train \
  --train-files data/processed/mix_1to1.jsonl \
  --d-model 768 --n-layer 12 --n-heads 12 \
  --batch-size 256 \
  --epochs 3 \
  --eval-every 500 \
  --log-path training/pretrain_log/train_log_PT-1A.txt \
  --ckpt-dir training/checkpoints/PT-1A

# PT-1B: 768×12 (~120M) on mix 1:2
python training/pretrain.py --train \
  --train-files data/processed/mix_1to2.jsonl \
  --d-model 768 --n-layer 12 --n-heads 12 \
  --batch-size 256 \
  --epochs 3 \
  --eval-every 500 \
  --log-path training/pretrain_log/train_log_PT-1B.txt \
  --ckpt-dir training/checkpoints/PT-1B

# PT-2A: 512×6 (~40M) on mix 1:1
python training/pretrain.py --train \
  --train-files data/processed/mix_1to1.jsonl \
  --d-model 512 --n-layer 6 --n-heads 8 \
  --batch-size 512 \
  --epochs 3 \
  --eval-every 500 \
  --log-path training/pretrain_log/train_log_PT-2A.txt \
  --ckpt-dir training/checkpoints/PT-2A

# PT-2B: 1024×8 (wide) on mix 1:1
python training/pretrain.py --train \
  --train-files data/processed/mix_1to1.jsonl \
  --d-model 1024 --n-layer 8 --n-heads 16 \
  --batch-size 128 \
  --epochs 3 \
  --eval-every 500 \
  --log-path training/pretrain_log/train_log_PT-2B.txt \
  --ckpt-dir training/checkpoints/PT-2B
