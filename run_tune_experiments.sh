#!/bin/bash
# Script to run 6 CartPole tuning experiments in parallel using conda run

echo "Starting 6 CartPole tuning experiments..."

# Run 6 experiments in background using conda run
conda run -n rl_hf_course python scripts/train.py --config configs/tune_env/Cartpole_exp.yaml > logs/run1.log 2>&1 &
conda run -n rl_hf_course python scripts/train.py --config configs/tune_env/CartPole_moderate_randomwalk_baseline.yaml > logs/run2.log 2>&1 &
conda run -n rl_hf_course python scripts/train.py --config configs/tune_env/CartPole_extreme_gravity_baseline.yaml > logs/run3.log 2>&1 &
conda run -n rl_hf_course python scripts/train.py --config configs/tune_env/CartPole_randomwalk_gravity_baseline.yaml > logs/run4.log 2>&1 &
conda run -n rl_hf_course python scripts/train.py --config configs/tune_env/CartPole_fastsine_gravity_baseline.yaml > logs/run5.log 2>&1 &
conda run -n rl_hf_course python scripts/train.py --config configs/tune_env/CartPole_length_jump_baseline.yaml > logs/run6.log 2>&1 &

echo "✅ Started 6 training runs in background"
echo "Logs: logs/run{1..6}.log"
echo "WandB: Check project 'TuneEnv'"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/run1.log"
echo "  ps aux | grep train.py"

# Wait for all background jobs
wait
echo "All experiments completed!"
