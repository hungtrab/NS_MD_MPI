#!/bin/bash
# Run Walker2d vanilla experiments with TRPO

echo "========================================="
echo "  WALKER2D - VANILLA - TRPO"
echo "========================================="
echo "Total experiments: 1"
echo "========================================="
echo ""

echo "Experiment 1/1: walker2d_baseline_trpo"
python scripts/train.py --config configs/TRPO/vanilla/walker2d_baseline_trpo.yaml
echo ""

echo "✅ All walker2d vanilla TRPO experiments complete!"
