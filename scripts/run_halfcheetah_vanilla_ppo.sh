#!/bin/bash
# Halfcheetah Vanilla - PPO (Baseline only)

echo "Running Halfcheetah Vanilla (PPO): Baseline only"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/vanilla/half cheetah_vanilla_ppo.yaml \
  > logs/halfcheetah_vanilla_ppo.log 2>&1 &

echo "✅ Started vanilla baseline"
echo "Monitor: tail -f logs/halfcheetah_vanilla_ppo.log"
