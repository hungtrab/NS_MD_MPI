#!/bin/bash
# Run HalfCheetah EXTREME experiments

echo "=========================================="
echo "  HalfCheetah (MuJoCo) Experiments"
echo "=========================================="
echo ""
echo "Running:"
echo "  - 2 EXTREME/MULTI configs"
echo "  - Total: 2 runs (1M timesteps each)"
echo ""
echo "=========================================="

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/HalfCheetah_EXTREME_friction_randomwalk_ppo.yaml \
  > logs/halfcheetah_extreme_friction.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/HalfCheetah_MULTI_all3_chaos_ppo.yaml \
  > logs/halfcheetah_multi_all3.log 2>&1 &

echo "✅ Started 2 HalfCheetah experiments"
echo ""
echo "Note: Each takes 1M timesteps (~60-90 min)"
echo "Monitor: tail -f logs/halfcheetah_extreme_friction.log"
echo "WandB: Project 'TuneEnv'"
