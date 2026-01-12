#!/bin/bash
# Hopper Vanilla - PPO
# Runs baseline + NS-MDMPI (if applicable)

echo "Running Hopper Vanilla (PPO)"
echo "═══════════════════════════════════════"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/vanilla/hopper_vanilla_ppo.yaml \
  > logs/hopper_vanilla_ppo.log 2>&1 &

echo "✅ Started 1 run (baseline only)"
echo "Monitor: tail -f logs/hopper_vanilla_ppo.log"
