#!/bin/bash
# Halfcheetah Moderate - SAC (Baseline + NS-MDMPI)

echo "Running Halfcheetah Moderate (SAC): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/moderate/half cheetah_friction_sine_baseline_sac.yaml \
  > logs/halfcheetah_moderate_baseline_sac.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/moderate/half cheetah_friction_sine_nsmdmpi_sac.yaml \
  > logs/halfcheetah_moderate_nsmdmpi_sac.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/halfcheetah_moderate_*.log"
