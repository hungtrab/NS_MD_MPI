#!/bin/bash
# Run Hopper EXTREME experiments

echo "=========================================="
echo "  Hopper (MuJoCo) Experiments"
echo "=========================================="
echo ""
echo "Running:"
echo "  - 3 Baseline configs"
echo "  - 2 EXTREME/MULTI configs"
echo "  - Total: 5 runs"
echo ""
echo "=========================================="

# Baseline configs
conda run -n rl_hf_course python scripts/train.py \
  --config configs/Hopper_friction_jump_baseline.yaml \
  > logs/hopper_friction_jump.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/Hopper_mass_randomwalk_baseline.yaml \
  > logs/hopper_mass_rw.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/Hopper_damping_sine_baseline.yaml \
  > logs/hopper_damping_sine.log 2>&1 &

# EXTREME configs
conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/Hopper_EXTREME_friction_randomwalk_ppo.yaml \
  > logs/hopper_extreme_friction.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/Hopper_MULTI_friction_mass_ppo.yaml \
  > logs/hopper_multi_friction_mass.log 2>&1 &

echo "✅ Started 5 Hopper experiments"
echo ""
echo "Note: Each takes ~500k timesteps (~30-45 min)"
echo "Monitor: tail -f logs/hopper_extreme_friction.log"
echo "WandB: Project 'TuneEnv'"
