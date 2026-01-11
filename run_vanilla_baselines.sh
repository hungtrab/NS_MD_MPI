#!/bin/bash
# Run All Vanilla Baseline Experiments

echo "================================================================"
echo "  VANILLA BASELINES - All Environments (No Drift)"
echo "================================================================"
echo ""
echo "Running 6 vanilla baseline experiments:"
echo "  - Hopper (1M timesteps)"
echo "  - HalfCheetah (1M timesteps)"
echo "  - LunarLander (500k timesteps)"
echo "  - MountainCar (200k timesteps)"
echo "  - FrozenLake (100k timesteps)"
echo "  - MiniGrid (200k timesteps)"
echo ""
echo "Purpose: Establish performance baselines in stationary environments"
echo "================================================================"
echo ""

# Hopper
echo "[1/6] Launching Hopper vanilla..."
conda run -n rl_hf_course python scripts/train.py \
  --config configs/vanilla/hopper_vanilla.yaml \
  > logs/vanilla_hopper.log 2>&1 &
echo "  ✅ Started Hopper"
sleep 2

# HalfCheetah
echo "[2/6] Launching HalfCheetah vanilla..."
conda run -n rl_hf_course python scripts/train.py \
  --config configs/vanilla/halfcheetah_vanilla.yaml \
  > logs/vanilla_halfcheetah.log 2>&1 &
echo "  ✅ Started HalfCheetah"
sleep 2

# LunarLander
echo "[3/6] Launching LunarLander vanilla..."
conda run -n rl_hf_course python scripts/train.py \
  --config configs/vanilla/lunarlander_vanilla.yaml \
  > logs/vanilla_lunarlander.log 2>&1 &
echo "  ✅ Started LunarLander"
sleep 2

# MountainCar
echo "[4/6] Launching MountainCar vanilla..."
conda run -n rl_hf_course python scripts/train.py \
  --config configs/vanilla/mountaincar_vanilla.yaml \
  > logs/vanilla_mountaincar.log 2>&1 &
echo "  ✅ Started MountainCar"
sleep 2

# FrozenLake
echo "[5/6] Launching FrozenLake vanilla..."
conda run -n rl_hf_course python scripts/train.py \
  --config configs/vanilla/frozenlake_vanilla.yaml \
  > logs/vanilla_frozenlake.log 2>&1 &
echo "  ✅ Started FrozenLake"
sleep 2

# MiniGrid
echo "[6/6] Launching MiniGrid vanilla..."
conda run -n rl_hf_course python scripts/train.py \
  --config configs/vanilla/minigrid_vanilla.yaml \
  > logs/vanilla_minigrid.log 2>&1 &
echo "  ✅ Started MiniGrid"

echo ""
echo "================================================================"
echo "✅ ALL 6 VANILLA BASELINES LAUNCHED!"
echo "================================================================"
echo ""
echo "Monitor progress:"
echo "  ps aux | grep train.py | grep vanilla | wc -l"
echo "  tail -f logs/vanilla_hopper.log"
echo ""
echo "Logs:"
echo "  logs/vanilla_*.log"
echo ""
echo "WandB Dashboard:"
echo "  Project: Vanilla_Baselines"
echo "  https://wandb.ai"
echo ""
echo "Estimated time: 4-6 hours (parallel)"
echo "================================================================"

# Wait for all jobs
wait
echo ""
echo "🎉 ALL VANILLA BASELINES COMPLETED! 🎉"
