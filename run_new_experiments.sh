#!/bin/bash
# Script to run 4 new extreme & multi-parameter CartPole experiments

echo "Starting 4 NEW CartPole experiments..."
echo "  - 2 Extreme single-parameter"
echo "  - 2 Multi-parameter"
echo ""

# EXTREME configs
conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_EXTREME_gravity_randomwalk_baseline.yaml \
  > logs/run_extreme_gravity.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_EXTREME_length_jump_baseline.yaml \
  > logs/run_extreme_length.log 2>&1 &

# MULTI-PARAMETER configs
conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_MULTI_gravity_masscart_baseline.yaml \
  > logs/run_multi_2param.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_MULTI_all4_chaos_baseline.yaml \
  > logs/run_multi_all4.log 2>&1 &

echo "✅ Started 4 training runs in background"
echo ""
echo "Logs:"
echo "  - logs/run_extreme_gravity.log"
echo "  - logs/run_extreme_length.log"
echo "  - logs/run_multi_2param.log"
echo "  - logs/run_multi_all4.log"
echo ""
echo "WandB: Project 'TuneEnv'"
echo ""
echo "Monitor:"
echo "  ps aux | grep train.py"
echo "  tail -f logs/run_extreme_gravity.log"

# Wait for all jobs
wait
echo "All experiments completed!"
