#!/bin/bash
# Run HalfCheetah Multi - friction + damping

echo "Running HalfCheetah Multi (PPO): friction + damping"
echo "WandB: att_3_HalfCheetah_Multi_Comparison"

CONFIG="half cheetah_friction_damping"

conda run -n rl_hf_course python scripts/train.py \
  --config "configs/PPO/multi/${CONFIG}_baseline_ppo.yaml" \
  > "logs/${CONFIG// /_}_baseline_ppo.log" 2>&1 &
sleep 2

conda run -n rl_hf_course python scripts/train.py \
  --config "configs/PPO/multi/${CONFIG}_nsmdmpi_ppo.yaml" \
  > "logs/${CONFIG// /_}_nsmdmpi_ppo.log" 2>&1 &

echo "✅ Launched 2 runs"
