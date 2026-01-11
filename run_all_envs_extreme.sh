#!/bin/bash
# MASTER SCRIPT - Run ALL EXTREME experiments (WITHOUT CartPole)

echo "================================================================"
echo "  MASTER LAUNCHER: ALL EXTREME EXPERIMENTS"
echo "================================================================"
echo ""
echo "Total experiments: 16 runs across 7 environments"
echo ""
echo "  - Hopper: 5 runs"
echo "  - LunarLander: 5 runs"
echo "  - MountainCar: 2 runs"
echo "  - HalfCheetah: 2 runs"
echo "  - FrozenLake: 1 run"
echo "  - MiniGrid: 1 run"
echo ""
echo "Estimated total time: 6-8 hours (running in parallel)"
echo "================================================================"
echo ""

# Ask for confirmation
read -p "Launch ALL 16 experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "[1/5] Launching Hopper experiments (5 runs)..."
./run_hopper_extreme.sh
sleep 3

echo ""
echo "[2/5] Launching LunarLander experiments (5 runs)..."
./run_lunarlander_extreme.sh
sleep 3

echo ""
echo "[3/5] Launching MountainCar experiments (2 runs)..."
./run_mountaincar_extreme.sh
sleep 3

echo ""
echo "[4/5] Launching HalfCheetah experiments (2 runs)..."
./run_halfcheetah_extreme.sh
sleep 3

echo ""
echo "[5/5] Launching FrozenLake & MiniGrid experiments (2 runs)..."
./run_frozenlake_minigrid_extreme.sh

echo ""
echo "================================================================"
echo "✅ ALL 16 EXPERIMENTS LAUNCHED!"
echo "================================================================"
echo ""
echo "Monitor progress:"
echo "  ps aux | grep train.py | wc -l"
echo "  htop"
echo ""
echo "Log files:"
echo "  ls -lth logs/*.log | head -20"
echo ""
echo "WandB Dashboard:"
echo "  Project: TuneEnv"
echo "  https://wandb.ai"
echo ""
echo "================================================================"
echo ""
echo "Breakdown by environment:"
echo "  - MuJoCo: Hopper(5) + HalfCheetah(2) = 7"
echo "  - Box2D: LunarLander(5) = 5"
echo "  - Classic: MountainCar(2) = 2"
echo "  - Grid World: FrozenLake(1) + MiniGrid(1) = 2"
echo "================================================================"

# Wait for all background jobs
wait
echo ""
echo "🎉 ALL EXPERIMENTS COMPLETED! 🎉"
