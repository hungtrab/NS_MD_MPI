#!/bin/bash
# Halfcheetah Extreme - PPO (Baseline + NS-MDMPI)

echo "Running Halfcheetah Extreme (PPO): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/extreme/half cheetah_friction_random_walk_baseline_ppo.yaml \
  > logs/halfcheetah_extreme_baseline_ppo.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/extreme/half cheetah_friction_random_walk_nsmdmpi_ppo.yaml \
  > logs/halfcheetah_extreme_nsmdmpi_ppo.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/halfcheetah_extreme_*.log"
