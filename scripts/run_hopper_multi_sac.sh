#!/bin/bash
# Hopper Multi - SAC (Baseline + NS-MDMPI)

echo "Running Hopper Multi (SAC): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/multi/hopper_friction_mass_scale_baseline_sac.yaml \
  > logs/hopper_multi_baseline_sac.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/multi/hopper_friction_mass_scale_nsmdmpi_sac.yaml \
  > logs/hopper_multi_nsmdmpi_sac.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/hopper_multi_*.log"
