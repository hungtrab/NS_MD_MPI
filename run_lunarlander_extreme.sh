#!/bin/bash
# Run LunarLander EXTREME experiments

echo "=========================================="
echo "  LunarLander Experiments"
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
  --config configs/LunarLander_gravity_jump_baseline.yaml \
  > logs/lunar_gravity_jump.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/LunarLander_wind_randomwalk_baseline.yaml \
  > logs/lunar_wind_rw.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/LunarLander_turbulence_sine_baseline.yaml \
  > logs/lunar_turbulence_sine.log 2>&1 &

# EXTREME configs
conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/LunarLander_EXTREME_gravity_randomwalk_ppo.yaml \
  > logs/lunar_extreme_gravity.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/LunarLander_MULTI_all3_chaos_ppo.yaml \
  > logs/lunar_multi_all3.log 2>&1 &

echo "✅ Started 5 LunarLander experiments"
echo ""
echo "Monitor: tail -f logs/lunar_extreme_gravity.log"
echo "WandB: Project 'TuneEnv'"
