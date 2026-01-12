#!/bin/bash
# Lunarlander Vanilla - SAC (Baseline only)

echo "Running Lunarlander Vanilla (SAC): Baseline only"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/vanilla/lunarlander_vanilla_sac.yaml \
  > logs/lunarlander_vanilla_sac.log 2>&1 &

echo "✅ Started vanilla baseline"
echo "Monitor: tail -f logs/lunarlander_vanilla_sac.log"
