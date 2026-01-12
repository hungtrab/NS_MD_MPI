#!/bin/bash
# Hopper Vanilla - SAC (Baseline only)

echo "Running Hopper Vanilla (SAC): Baseline only"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/vanilla/hopper_vanilla_sac.yaml \
  > logs/hopper_vanilla_sac.log 2>&1 &

echo "✅ Started vanilla baseline"
echo "Monitor: tail -f logs/hopper_vanilla_sac.log"
