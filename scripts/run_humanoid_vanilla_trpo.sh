#!/bin/bash
# Run Humanoid vanilla experiments with TRPO

echo "========================================="
echo "  HUMANOID - VANILLA - TRPO"
echo "========================================="
echo "Total experiments: 1"
echo "========================================="
echo ""

echo "Experiment 1/1: humanoid_baseline_trpo"
python scripts/train.py --config configs/TRPO/vanilla/humanoid_baseline_trpo.yaml
echo ""

echo "✅ All humanoid vanilla TRPO experiments complete!"
