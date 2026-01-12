#!/bin/bash
# Lunarlander Extreme - SAC (Baseline + NS-MDMPI)

echo "Running Lunarlander Extreme (SAC): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/extreme/lunarlander_gravity_random_walk_baseline_sac.yaml \
  > logs/lunarlander_extreme_baseline_sac.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/extreme/lunarlander_gravity_random_walk_nsmdmpi_sac.yaml \
  > logs/lunarlander_extreme_nsmdmpi_sac.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/lunarlander_extreme_*.log"
