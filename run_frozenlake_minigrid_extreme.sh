#!/bin/bash
# Run FrozenLake & MiniGrid EXTREME experiments

echo "=========================================="
echo "  FrozenLake & MiniGrid Experiments"
echo "=========================================="
echo ""
echo "Running:"
echo "  - FrozenLake: 1 EXTREME config"
echo "  - MiniGrid: 1 EXTREME config"
echo "  - Total: 2 runs"
echo ""
echo "=========================================="

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/FrozenLake_EXTREME_slip_jump_ppo.yaml \
  > logs/frozenlake_extreme_slip.log 2>&1 &

conda run -n rl_hf_course python scripts/train.py \
  --config configs/tune_env/MiniGrid_EXTREME_reward_randomwalk_ppo.yaml \
  > logs/minigrid_extreme_reward.log 2>&1 &

echo "✅ Started 2 experiments (FrozenLake + MiniGrid)"
echo ""
echo "Monitor:"
echo "  tail -f logs/frozenlake_extreme_slip.log"
echo "  tail -f logs/minigrid_extreme_reward.log"
echo "WandB: Project 'TuneEnv'"
