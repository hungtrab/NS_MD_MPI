#!/bin/bash
# Run HalfCheetah Extreme - ALL configs

echo "Running HalfCheetah Extreme (PPO): 2 configs × 2 methods = 4 runs"
echo "WandB: att_3_HalfCheetah_Extreme_Comparison"

CONFIGS=("half cheetah_friction_random_walk" "half cheetah_friction_jump")

for config in "${CONFIGS[@]}"; do
    conda run -n rl_hf_course python scripts/train.py \
      --config "configs/PPO/extreme/${config}_baseline_ppo.yaml" \
      > "logs/${config// /_}_baseline_ppo.log" 2>&1 &
    sleep 2
done

for config in "${CONFIGS[@]}"; do
    conda run -n rl_hf_course python scripts/train.py \
      --config "configs/PPO/extreme/${config}_nsmdmpi_ppo.yaml" \
      > "logs/${config// /_}_nsmdmpi_ppo.log" 2>&1 &
    sleep 2
done

echo "✅ Launched 4 runs"
