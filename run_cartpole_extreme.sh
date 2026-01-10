#!/bin/bash
# Run ALL CartPole EXTREME experiments (PPO + TRPO)

echo "=========================================="
echo "  CartPole EXTREME Experiments"
echo "=========================================="
echo ""
echo "Running:"
echo "  - 2 ULTRA configs (all-4 insane, conflicting)"
echo "  - 4 EXTREME/MULTI configs"
echo "  - Each with PPO + TRPO = 12 runs total"
echo ""
echo "=========================================="

# ULTRA configs
conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_ULTRA_all4_insane_ppo.yaml \
  > logs/cartpole_ultra_all4_ppo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_ULTRA_all4_insane_trpo.yaml \
  > logs/cartpole_ultra_all4_trpo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_ULTRA_conflicting_drifts_ppo.yaml \
  > logs/cartpole_ultra_conflicting_ppo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_ULTRA_conflicting_drifts_trpo.yaml \
  > logs/cartpole_ultra_conflicting_trpo.log 2>&1 &

# EXTREME configs
conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_EXTREME_gravity_randomwalk_baseline.yaml \
  > logs/cartpole_extreme_grav_ppo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_EXTREME_gravity_randomwalk_trpo.yaml \
  > logs/cartpole_extreme_grav_trpo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_EXTREME_length_jump_baseline.yaml \
  > logs/cartpole_extreme_length_ppo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_EXTREME_length_jump_trpo.yaml \
  > logs/cartpole_extreme_length_trpo.log 2>&1 &

# MULTI configs
conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_MULTI_gravity_masscart_baseline.yaml \
  > logs/cartpole_multi_2p_ppo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_MULTI_gravity_masscart_trpo.yaml \
  > logs/cartpole_multi_2p_trpo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_MULTI_all4_chaos_baseline.yaml \
  > logs/cartpole_multi_all4_ppo.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/CartPole_MULTI_all4_chaos_trpo.yaml \
  > logs/cartpole_multi_all4_trpo.log 2>&1 &

echo "✅ Started 12 CartPole experiments"
echo ""
echo "Monitor: tail -f logs/cartpole_ultra_all4_ppo.log"
echo "WandB: Project 'TuneEnv'"
