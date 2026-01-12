#!/bin/bash
# Lunarlander Vanilla - PPO (Baseline only)

echo "Running Lunarlander Vanilla (PPO): Baseline only"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/vanilla/lunarlander_vanilla_ppo.yaml \
  > logs/lunarlander_vanilla_ppo.log 2>&1 &

echo "✅ Started vanilla baseline"
echo "Monitor: tail -f logs/lunarlander_vanilla_ppo.log"
