#!/usr/bin/env bash
# Four pretrain runs (3 full epochs each). Run from repo root.
#
# batch_size: any positive integer (224, 200, … all OK; not limited to powers of 2).
# Tuned for A800 + seq_len=512 + bf16 AMP. Slightly under 256/512 to avoid OOM headroom.
#   PT-1A/1B: 224×512 ≈ 114k tok/step
#   PT-2A:    448×512 ≈ 229k tok/step
#   PT-2B:    112×512 ≈  57k tok/step
#
# Still OOM? Drop 1A/1B to 200 or 192; 2A to 384; 2B to 96.

set -euo pipefail
cd "$(dirname "$0")/.."

# PT-1A: 768×12 (~120M) on mix 1:1
python training/pretrain.py --train \
  --train-files data/processed/mix_1to1.jsonl \
  --d-model 768 --n-layer 12 --n-heads 12 \
  --batch-size 224 \
  --epochs 3 \
  --eval-every 500 \
  --log-path training/pretrain_log/train_log_PT-1A.txt \
  --ckpt-dir training/checkpoints/PT-1A

# PT-1B: 768×12 (~120M) on mix 1:2
python training/pretrain.py --train \
  --train-files data/processed/mix_1to2.jsonl \
  --d-model 768 --n-layer 12 --n-heads 12 \
  --batch-size 224 \
  --epochs 3 \
  --eval-every 500 \
  --log-path training/pretrain_log/train_log_PT-1B.txt \
  --ckpt-dir training/checkpoints/PT-1B

# PT-2A: 512×6 (~40M) on mix 1:1
python training/pretrain.py --train \
  --train-files data/processed/mix_1to1.jsonl \
  --d-model 512 --n-layer 6 --n-heads 8 \
  --batch-size 448 \
  --epochs 3 \
  --eval-every 500 \
  --log-path training/pretrain_log/train_log_PT-2A.txt \
  --ckpt-dir training/checkpoints/PT-2A

# PT-2B: 1024×8 (wide) on mix 1:1
python training/pretrain.py --train \
  --train-files data/processed/mix_1to1.jsonl \
  --d-model 1024 --n-layer 8 --n-heads 16 \
  --batch-size 112 \
  --epochs 3 \
  --eval-every 500 \
  --log-path training/pretrain_log/train_log_PT-2B.txt \
  --ckpt-dir training/checkpoints/PT-2B
