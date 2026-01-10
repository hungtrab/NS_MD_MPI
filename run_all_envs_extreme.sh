#!/bin/bash
# MASTER SCRIPT - Run ALL EXTREME experiments across ALL environments

echo "================================================================"
echo "  MASTER LAUNCHER: ALL EXTREME EXPERIMENTS (ALL ENVS)"
echo "================================================================"
echo ""
echo "Total experiments: 28 runs across 8 environments"
echo ""
echo "  - CartPole: 12 runs (ULTRA + EXTREME + MULTI)"
echo "  - Hopper: 5 runs"
echo "  - LunarLander: 5 runs"
echo "  - MountainCar: 2 runs"
echo "  - HalfCheetah: 2 runs"
echo "  - FrozenLake: 1 run"
echo "  - MiniGrid: 1 run"
echo ""
echo "Estimated total time: 8-10 hours (running in parallel)"
echo "================================================================"
echo ""

# Ask for confirmation
read -p "Launch ALL 28 experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "[1/6] Launching CartPole experiments (12 runs)..."
./run_cartpole_extreme.sh
sleep 3

echo ""
echo "[2/6] Launching Hopper experiments (5 runs)..."
./run_hopper_extreme.sh
sleep 3

echo ""
echo "[3/6] Launching LunarLander experiments (5 runs)..."
./run_lunarlander_extreme.sh
sleep 3

echo ""
echo "[4/6] Launching MountainCar experiments (2 runs)..."
./run_mountaincar_extreme.sh
sleep 3

echo ""
echo "[5/6] Launching HalfCheetah experiments (2 runs)..."
./run_halfcheetah_extreme.sh
sleep 3

echo ""
echo "[6/6] Launching FrozenLake & MiniGrid experiments (2 runs)..."
./run_frozenlake_minigrid_extreme.sh

echo ""
echo "================================================================"
echo "✅ ALL 28 EXPERIMENTS LAUNCHED!"
echo "================================================================"
echo ""
echo "Monitor progress:"
echo "  ps aux | grep train.py | wc -l"
echo "  htop"
echo ""
echo "Log files:"
echo "  ls -lth logs/*.log | head -30"
echo ""
echo "WandB Dashboard:"
echo "  Project: TuneEnv"
echo "  https://wandb.ai/your-username/TuneEnv"
echo ""
echo "================================================================"
echo ""
echo "Breakdown by environment:"
echo "  - Classic Control: CartPole(12) + MountainCar(2) = 14"
echo "  - MuJoCo: Hopper(5) + HalfCheetah(2) = 7"
echo "  - Box2D: LunarLander(5) = 5"
echo "  - Grid World: FrozenLake(1) + MiniGrid(1) = 2"
echo "================================================================"

# Wait for all background jobs
wait
echo ""
echo "🎉 ALL EXPERIMENTS COMPLETED! 🎉"
