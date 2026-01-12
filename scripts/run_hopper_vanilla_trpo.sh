#!/bin/bash
# Hopper Vanilla - TRPO
# Runs baseline + NS-MDMPI (if applicable)

echo "Running Hopper Vanilla (TRPO)"
echo "═══════════════════════════════════════"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/vanilla/hopper_vanilla_trpo.yaml \
  > logs/hopper_vanilla_trpo.log 2>&1 &

echo "✅ Started 1 run (baseline only)"
echo "Monitor: tail -f logs/hopper_vanilla_trpo.log"
