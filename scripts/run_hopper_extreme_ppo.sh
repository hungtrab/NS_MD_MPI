#!/bin/bash
# Hopper Extreme - PPO (Baseline + NS-MDMPI)

echo "Running Hopper Extreme (PPO): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/extreme/hopper_friction_random_walk_baseline_ppo.yaml \
  > logs/hopper_extreme_baseline_ppo.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/extreme/hopper_friction_random_walk_nsmdmpi_ppo.yaml \
  > logs/hopper_extreme_nsmdmpi_ppo.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/hopper_extreme_*.log"
