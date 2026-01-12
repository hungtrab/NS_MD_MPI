#!/bin/bash
# HalfCheetah Vanilla - PPO
# Runs baseline + NS-MDMPI (if applicable)

echo "Running HalfCheetah Vanilla (PPO)"
echo "═══════════════════════════════════════"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/vanilla/halfcheetah_vanilla_ppo.yaml \
  > logs/halfcheetah_vanilla_ppo.log 2>&1 &

echo "✅ Started 1 run (baseline only)"
echo "Monitor: tail -f logs/halfcheetah_vanilla_ppo.log"
