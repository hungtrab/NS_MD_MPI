#!/bin/bash
# LunarLander Vanilla - PPO
# Runs baseline + NS-MDMPI (if applicable)

echo "Running LunarLander Vanilla (PPO)"
echo "═══════════════════════════════════════"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/vanilla/lunarlander_vanilla_ppo.yaml \
  > logs/lunarlander_vanilla_ppo.log 2>&1 &

echo "✅ Started 1 run (baseline only)"
echo "Monitor: tail -f logs/lunarlander_vanilla_ppo.log"
