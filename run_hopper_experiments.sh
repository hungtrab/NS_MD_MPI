#!/bin/bash
# Script to run 3 Hopper (MuJoCo) experiments

echo "Starting 3 Hopper (MuJoCo) experiments..."
echo "  - Friction jump"
echo "  - Mass random walk"
echo "  - Damping sine wave"
echo ""

# Run 3 experiments in background
conda run -n rl_hf_course python scripts/train.py \
  --config configs/Hopper_friction_jump_baseline.yaml \
  > logs/run_hopper_friction.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/Hopper_mass_randomwalk_baseline.yaml \
  > logs/run_hopper_mass.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/Hopper_damping_sine_baseline.yaml \
  > logs/run_hopper_damping.log 2>&1 &

echo "✅ Started 3 Hopper training runs in background"
echo ""
echo "Logs:"
echo "  - logs/run_hopper_friction.log"
echo "  - logs/run_hopper_mass.log"
echo "  - logs/run_hopper_damping.log"
echo ""
echo "WandB: Project 'TuneEnv'"
echo ""
echo "Monitor:"
echo "  ps aux | grep train.py | grep Hopper"
echo "  tail -f logs/run_hopper_friction.log"

# Wait for all jobs
wait
echo "All Hopper experiments completed!"
