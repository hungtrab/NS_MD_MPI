#!/bin/bash
# Hopper Multi - PPO (Baseline + NS-MDMPI)

echo "Running Hopper Multi (PPO): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/multi/hopper_friction_mass_scale_baseline_ppo.yaml \
  > logs/hopper_multi_baseline_ppo.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/multi/hopper_friction_mass_scale_nsmdmpi_ppo.yaml \
  > logs/hopper_multi_nsmdmpi_ppo.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/hopper_multi_*.log"
