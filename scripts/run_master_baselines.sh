#!/bin/bash
# Master Baseline Script - Run all vanilla baselines

echo "================================================================"
echo "  MASTER BASELINES - All Vanilla Experiments"
echo "================================================================"
echo ""
echo "Running:"
echo "  - Hopper Vanilla (PPO, SAC, TRPO)"
echo "  - HalfCheetah Vanilla (PPO, SAC, TRPO)"
echo "  - LunarLander Vanilla (PPO, SAC, TRPO)"
echo ""
echo "Total: 9 vanilla baselines"
echo "================================================================"
echo ""

echo "[Hopper PPO] Starting..."
bash scripts/run_hopper_vanilla_ppo.sh
sleep 2

echo "[Hopper SAC] Starting..."
bash scripts/run_hopper_vanilla_sac.sh
sleep 2

echo "[Hopper TRPO] Starting..."
bash scripts/run_hopper_vanilla_trpo.sh
sleep 2

echo "[Halfcheetah PPO] Starting..."
bash scripts/run_halfcheetah_vanilla_ppo.sh
sleep 2

echo "[Halfcheetah SAC] Starting..."
bash scripts/run_halfcheetah_vanilla_sac.sh
sleep 2

echo "[Halfcheetah TRPO] Starting..."
bash scripts/run_halfcheetah_vanilla_trpo.sh
sleep 2

echo "[Lunarlander PPO] Starting..."
bash scripts/run_lunarlander_vanilla_ppo.sh
sleep 2

echo "[Lunarlander SAC] Starting..."
bash scripts/run_lunarlander_vanilla_sac.sh
sleep 2

echo "[Lunarlander TRPO] Starting..."
bash scripts/run_lunarlander_vanilla_trpo.sh
sleep 2

echo ""
echo "================================================================"
echo "✅ ALL 9 VANILLA BASELINES LAUNCHED!"
echo "================================================================"
echo ""
echo "Monitor: ps aux | grep train.py | wc -l"
echo "WandB: Project 'att_3_Vanilla_Baselines'"
