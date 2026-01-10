#!/bin/bash
# Run MountainCar EXTREME experiments

echo "=========================================="
echo "  MountainCar Experiments"
echo "=========================================="
echo ""
echo "Running:"
echo "  - 2 EXTREME/MULTI configs"
echo ""
echo "=========================================="

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/MountainCar_EXTREME_gravity_jump_ppo.yaml \
  > logs/mountaincar_extreme_gravity.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/MountainCar_MULTI_gravity_force_conflicting_ppo.yaml \
  > logs/mountaincar_multi_conflicting.log 2>&1 &

echo "✅ Started 2 MountainCar experiments"
echo ""
echo "Monitor: tail -f logs/mountaincar_extreme_gravity.log"
echo "WandB: Project 'TuneEnv'"
