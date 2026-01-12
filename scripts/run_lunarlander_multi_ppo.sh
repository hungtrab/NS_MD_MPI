#!/bin/bash
# Lunarlander Multi - PPO (Baseline + NS-MDMPI)

echo "Running Lunarlander Multi (PPO): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/multi/lunarlander_gravity_wind_power_baseline_ppo.yaml \
  > logs/lunarlander_multi_baseline_ppo.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/multi/lunarlander_gravity_wind_power_nsmdmpi_ppo.yaml \
  > logs/lunarlander_multi_nsmdmpi_ppo.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/lunarlander_multi_*.log"
