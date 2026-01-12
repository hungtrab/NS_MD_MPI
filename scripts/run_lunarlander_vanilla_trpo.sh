#!/bin/bash
# Lunarlander Vanilla - TRPO (Baseline only)

echo "Running Lunarlander Vanilla (TRPO): Baseline only"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/vanilla/lunarlander_vanilla_trpo.yaml \
  > logs/lunarlander_vanilla_trpo.log 2>&1 &

echo "✅ Started vanilla baseline"
echo "Monitor: tail -f logs/lunarlander_vanilla_trpo.log"
