#!/usr/bin/env bash
# Four pretrain runs (3 full epochs each). Run from repo root.
# OOM? Lower --batch-size for that run only.

set -euo pipefail
cd "$(dirname "$0")/.."

# PT-1A: 768×12 on mix 1:1
python training/pretrain.py --train \
  --train-files data/processed/mix_1to1.jsonl \
  --d-model 768 --n-layer 12 --n-heads 12 \
  --batch-size 8 \
  --epochs 3 \
  --eval-every 500 \
  --log-path training/pretrain_log/train_log_PT-1A.txt \
  --ckpt-dir training/checkpoints/PT-1A

# PT-1B: 768×12 on mix 1:2
python training/pretrain.py --train \
  --train-files data/processed/mix_1to2.jsonl \
  --d-model 768 --n-layer 12 --n-heads 12 \
  --batch-size 8 \
  --epochs 3 \
  --eval-every 500 \
  --log-path training/pretrain_log/train_log_PT-1B.txt \
  --ckpt-dir training/checkpoints/PT-1B

# PT-2A: 512×6 on mix 1:1
python training/pretrain.py --train \
  --train-files data/processed/mix_1to1.jsonl \
  --d-model 512 --n-layer 6 --n-heads 8 \
  --batch-size 16 \
  --epochs 3 \
  --eval-every 500 \
  --log-path training/pretrain_log/train_log_PT-2A.txt \
  --ckpt-dir training/checkpoints/PT-2A

# PT-2B: 1024×8 on mix 1:1
python training/pretrain.py --train \
  --train-files data/processed/mix_1to1.jsonl \
  --d-model 1024 --n-layer 8 --n-heads 16 \
  --batch-size 4 \
  --epochs 3 \
  --eval-every 500 \
  --log-path training/pretrain_log/train_log_PT-2B.txt \
  --ckpt-dir training/checkpoints/PT-2B
