#!/bin/bash
# Hopper Moderate - TRPO (Baseline + NS-MDMPI)

echo "Running Hopper Moderate (TRPO): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/moderate/hopper_friction_sine_baseline_trpo.yaml \
  > logs/hopper_moderate_baseline_trpo.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/moderate/hopper_friction_sine_nsmdmpi_trpo.yaml \
  > logs/hopper_moderate_nsmdmpi_trpo.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/hopper_moderate_*.log"
