#!/bin/bash
# Run Walker2d vanilla experiments with PPO

echo "========================================="
echo "  WALKER2D - VANILLA - PPO"
echo "========================================="
echo "Total experiments: 1"
echo "========================================="
echo ""

echo "Experiment 1/1: walker2d_baseline_ppo"
python scripts/train.py --config configs/PPO/vanilla/walker2d_baseline_ppo.yaml
echo ""

echo "✅ All walker2d vanilla PPO experiments complete!"
