#!/bin/bash
# LunarLander Vanilla - SAC
# Runs baseline + NS-MDMPI (if applicable)

echo "Running LunarLander Vanilla (SAC)"
echo "═══════════════════════════════════════"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/vanilla/lunarlander_vanilla_sac.yaml \
  > logs/lunarlander_vanilla_sac.log 2>&1 &

echo "✅ Started 1 run (baseline only)"
echo "Monitor: tail -f logs/lunarlander_vanilla_sac.log"
