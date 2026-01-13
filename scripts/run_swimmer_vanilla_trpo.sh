#!/bin/bash
# Run Swimmer vanilla experiments with TRPO

echo "========================================="
echo "  SWIMMER - VANILLA - TRPO"
echo "========================================="
echo "Total experiments: 1"
echo "========================================="
echo ""

echo "Experiment 1/1: swimmer_baseline_trpo"
python scripts/train.py --config configs/TRPO/vanilla/swimmer_baseline_trpo.yaml
echo ""

echo "✅ All swimmer vanilla TRPO experiments complete!"
