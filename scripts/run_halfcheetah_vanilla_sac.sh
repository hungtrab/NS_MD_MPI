#!/bin/bash
# Halfcheetah Vanilla - SAC (Baseline only)

echo "Running Halfcheetah Vanilla (SAC): Baseline only"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/vanilla/half cheetah_vanilla_sac.yaml \
  > logs/halfcheetah_vanilla_sac.log 2>&1 &

echo "✅ Started vanilla baseline"
echo "Monitor: tail -f logs/halfcheetah_vanilla_sac.log"
