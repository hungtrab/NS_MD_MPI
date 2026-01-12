#!/bin/bash
# Run LunarLander Multi - gravity + wind_power

echo "Running LunarLander Multi (PPO): gravity + wind_power"
echo "WandB: att_3_LunarLander_Multi_Comparison"

CONFIG="lunarlander_gravity_wind_power"

conda run -n rl_hf_course python scripts/train.py \
  --config "configs/PPO/multi/${CONFIG}_baseline_ppo.yaml" \
  > "logs/${CONFIG}_baseline_ppo.log" 2>&1 &
sleep 2

conda run -n rl_hf_course python scripts/train.py \
  --config "configs/PPO/multi/${CONFIG}_nsmdmpi_ppo.yaml" \
  > "logs/${CONFIG}_nsmdmpi_ppo.log" 2>&1 &

echo "✅ Launched 2 runs"
