#!/bin/bash
# Run Walker2d vanilla experiments with SAC

echo "========================================="
echo "  WALKER2D - VANILLA - SAC"
echo "========================================="
echo "Total experiments: 1"
echo "========================================="
echo ""

echo "Experiment 1/1: walker2d_baseline_sac"
python scripts/train.py --config configs/SAC/vanilla/walker2d_baseline_sac.yaml
echo ""

echo "✅ All walker2d vanilla SAC experiments complete!"
