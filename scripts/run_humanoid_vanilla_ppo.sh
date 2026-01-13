#!/bin/bash
# Run Humanoid vanilla experiments with PPO

echo "========================================="
echo "  HUMANOID - VANILLA - PPO"
echo "========================================="
echo "Total experiments: 1"
echo "========================================="
echo ""

echo "Experiment 1/1: humanoid_baseline_ppo"
python scripts/train.py --config configs/PPO/vanilla/humanoid_baseline_ppo.yaml
echo ""

echo "✅ All humanoid vanilla PPO experiments complete!"
