#!/bin/bash
# Run Humanoid vanilla experiments with SAC

echo "========================================="
echo "  HUMANOID - VANILLA - SAC"
echo "========================================="
echo "Total experiments: 1"
echo "========================================="
echo ""

echo "Experiment 1/1: humanoid_baseline_sac"
python scripts/train.py --config configs/SAC/vanilla/humanoid_baseline_sac.yaml
echo ""

echo "✅ All humanoid vanilla SAC experiments complete!"
