#!/bin/bash
# LunarLander Vanilla - TRPO
# Runs baseline + NS-MDMPI (if applicable)

echo "Running LunarLander Vanilla (TRPO)"
echo "═══════════════════════════════════════"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/vanilla/lunarlander_vanilla_trpo.yaml \
  > logs/lunarlander_vanilla_trpo.log 2>&1 &

echo "✅ Started 1 run (baseline only)"
echo "Monitor: tail -f logs/lunarlander_vanilla_trpo.log"
