#!/bin/bash
# Script to run ALL extreme CartPole experiments (PPO + TRPO)
# Priority: ULTRA-EXTREME configs first, then other EXTREME/MULTI configs

echo "================================================================"
echo "  EXTREME CARTPOLE EXPERIMENTS: PPO + TRPO"
echo "================================================================"
echo ""
echo "Will run in batches to avoid overloading system:"
echo "  BATCH 1: Ultra-Extreme PPO (2 runs)"
echo "  BATCH 2: Ultra-Extreme TRPO (2 runs)"  
echo "  BATCH 3: Other Extreme PPO (4 runs)"
echo "  BATCH 4: Other Extreme TRPO (4 runs)"
echo ""
echo "Total: 12 critical experiments"
echo "================================================================"
echo ""

# Batch 1: Ultra-Extreme PPO
echo "[BATCH 1] Starting Ultra-Extreme PPO experiments..."

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_ULTRA_all4_insane_ppo.yaml \
  > logs/run_ultra_all4_ppo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_ULTRA_conflicting_drifts_ppo.yaml \
  > logs/run_ultra_conflicting_ppo.log 2>&1 &

echo "  ✅ Batch 1 started (2 runs)"
echo ""

# Batch 2: Ultra TRPO
echo "[BATCH 2] Starting Ultra-Extreme TRPO experiments..."

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_ULTRA_all4_insane_trpo.yaml \
  > logs/run_ultra_all4_trpo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_ULTRA_conflicting_drifts_trpo.yaml \
  > logs/run_ultra_conflicting_trpo.log 2>&1 &

echo "  ✅ Batch 2 started (2 runs)"
echo ""

# Batch 3: Other Extreme PPO
echo "[BATCH 3] Starting Other Extreme PPO experiments..."

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_EXTREME_gravity_randomwalk_baseline.yaml \
  > logs/run_extreme_grav_ppo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_EXTREME_length_jump_baseline.yaml \
  > logs/run_extreme_length_ppo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_MULTI_gravity_masscart_baseline.yaml \
  > logs/run_multi_2p_ppo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_MULTI_all4_chaos_baseline.yaml \
  > logs/run_multi_all4_ppo.log 2>&1 &

echo "  ✅ Batch 3 started (4 runs)"
echo ""

# Batch 4: Other Extreme TRPO
echo "[BATCH 4] Starting Other Extreme TRPO experiments..."

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_EXTREME_gravity_randomwalk_trpo.yaml \
  > logs/run_extreme_grav_trpo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_EXTREME_length_jump_trpo.yaml \
  > logs/run_extreme_length_trpo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_MULTI_gravity_masscart_trpo.yaml \
  > logs/run_multi_2p_trpo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_MULTI_all4_chaos_trpo.yaml \
  > logs/run_multi_all4_trpo.log 2>&1 &

echo "  ✅ Batch 4 started (4 runs)"
echo ""

echo "================================================================"
echo "✅ ALL 12 EXPERIMENTS LAUNCHED!"
echo "================================================================"
echo ""
echo "Logs in logs/ directory:"
echo "  - Ultra-Extreme: run_ultra_*.log"
echo "  - Other Extreme: run_extreme_*.log, run_multi_*.log"
echo ""
echo "WandB Project: 'TuneEnv'"
echo ""
echo "Monitor:"
echo "  ps aux | grep train.py | wc -l"
echo "  tail -f logs/run_ultra_all4_ppo.log"
echo ""
echo "================================================================"

wait
echo "All experiments completed!"
