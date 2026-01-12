#!/bin/bash
# Halfcheetah Multi - PPO (Baseline + NS-MDMPI)

echo "Running Halfcheetah Multi (PPO): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/multi/half cheetah_friction_damping_baseline_ppo.yaml \
  > logs/halfcheetah_multi_baseline_ppo.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \
  --config configs/PPO/multi/half cheetah_friction_damping_nsmdmpi_ppo.yaml \
  > logs/halfcheetah_multi_nsmdmpi_ppo.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/halfcheetah_multi_*.log"
