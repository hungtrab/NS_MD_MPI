#!/bin/bash
# Halfcheetah Vanilla - TRPO (Baseline only)

echo "Running Halfcheetah Vanilla (TRPO): Baseline only"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/vanilla/half cheetah_vanilla_trpo.yaml \
  > logs/halfcheetah_vanilla_trpo.log 2>&1 &

echo "✅ Started vanilla baseline"
echo "Monitor: tail -f logs/halfcheetah_vanilla_trpo.log"
