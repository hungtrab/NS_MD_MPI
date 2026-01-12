#!/bin/bash
# Hopper Vanilla - TRPO (Baseline only)

echo "Running Hopper Vanilla (TRPO): Baseline only"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/vanilla/hopper_vanilla_trpo.yaml \
  > logs/hopper_vanilla_trpo.log 2>&1 &

echo "✅ Started vanilla baseline"
echo "Monitor: tail -f logs/hopper_vanilla_trpo.log"
