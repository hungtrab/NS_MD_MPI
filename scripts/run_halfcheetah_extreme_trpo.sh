#!/bin/bash
# Halfcheetah Extreme - TRPO (Baseline + NS-MDMPI)

echo "Running Halfcheetah Extreme (TRPO): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/extreme/half cheetah_friction_random_walk_baseline_trpo.yaml \
  > logs/halfcheetah_extreme_baseline_trpo.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/extreme/half cheetah_friction_random_walk_nsmdmpi_trpo.yaml \
  > logs/halfcheetah_extreme_nsmdmpi_trpo.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/halfcheetah_extreme_*.log"
