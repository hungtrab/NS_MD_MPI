#!/bin/bash
# Run LunarLander Extreme - ALL configs

echo "Running LunarLander Extreme (PPO): 2 configs × 2 methods = 4 runs"
echo "WandB: att_3_LunarLander_Extreme_Comparison"

CONFIGS=("lunarlander_gravity_random_walk" "lunarlander_gravity_jump")

for config in "${CONFIGS[@]}"; do
    conda run -n rl_hf_course python scripts/train.py \
      --config "configs/PPO/extreme/${config}_baseline_ppo.yaml" \
      > "logs/${config}_baseline_ppo.log" 2>&1 &
    sleep 2
done

for config in "${CONFIGS[@]}"; do
    conda run -n rl_hf_course python scripts/train.py \
      --config "configs/PPO/extreme/${config}_nsmdmpi_ppo.yaml" \
      > "logs/${config}_nsmdmpi_ppo.log" 2>&1 &
    sleep 2
done

echo "✅ Launched 4 runs"
