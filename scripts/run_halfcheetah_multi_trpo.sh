#!/bin/bash
# Halfcheetah Multi - TRPO (Baseline + NS-MDMPI)

echo "Running Halfcheetah Multi (TRPO): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/multi/half cheetah_friction_damping_baseline_trpo.yaml \
  > logs/halfcheetah_multi_baseline_trpo.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/multi/half cheetah_friction_damping_nsmdmpi_trpo.yaml \
  > logs/halfcheetah_multi_nsmdmpi_trpo.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/halfcheetah_multi_*.log"
