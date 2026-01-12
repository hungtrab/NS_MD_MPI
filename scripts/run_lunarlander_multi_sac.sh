#!/bin/bash
# Lunarlander Multi - SAC (Baseline + NS-MDMPI)

echo "Running Lunarlander Multi (SAC): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/multi/lunarlander_gravity_wind_power_baseline_sac.yaml \
  > logs/lunarlander_multi_baseline_sac.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/multi/lunarlander_gravity_wind_power_nsmdmpi_sac.yaml \
  > logs/lunarlander_multi_nsmdmpi_sac.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/lunarlander_multi_*.log"
