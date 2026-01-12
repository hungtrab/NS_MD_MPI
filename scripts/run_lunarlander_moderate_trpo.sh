#!/bin/bash
# Lunarlander Moderate - TRPO (Baseline + NS-MDMPI)

echo "Running Lunarlander Moderate (TRPO): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/moderate/lunarlander_gravity_sine_baseline_trpo.yaml \
  > logs/lunarlander_moderate_baseline_trpo.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/moderate/lunarlander_gravity_sine_nsmdmpi_trpo.yaml \
  > logs/lunarlander_moderate_nsmdmpi_trpo.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/lunarlander_moderate_*.log"
