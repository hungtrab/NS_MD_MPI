#!/bin/bash
# Run Swimmer vanilla experiments with PPO

echo "========================================="
echo "  SWIMMER - VANILLA - PPO"
echo "========================================="
echo "Total experiments: 1"
echo "========================================="
echo ""

echo "Experiment 1/1: swimmer_baseline_ppo"
python scripts/train.py --config configs/PPO/vanilla/swimmer_baseline_ppo.yaml
echo ""

echo "✅ All swimmer vanilla PPO experiments complete!"
