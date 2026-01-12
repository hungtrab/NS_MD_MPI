#!/bin/bash
# Lunarlander Moderate - SAC (Baseline + NS-MDMPI)

echo "Running Lunarlander Moderate (SAC): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/moderate/lunarlander_gravity_sine_baseline_sac.yaml \
  > logs/lunarlander_moderate_baseline_sac.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/moderate/lunarlander_gravity_sine_nsmdmpi_sac.yaml \
  > logs/lunarlander_moderate_nsmdmpi_sac.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/lunarlander_moderate_*.log"
