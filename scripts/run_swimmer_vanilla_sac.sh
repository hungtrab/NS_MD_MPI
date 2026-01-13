#!/bin/bash
# Run Swimmer vanilla experiments with SAC

echo "========================================="
echo "  SWIMMER - VANILLA - SAC"
echo "========================================="
echo "Total experiments: 1"
echo "========================================="
echo ""

echo "Experiment 1/1: swimmer_baseline_sac"
python scripts/train.py --config configs/SAC/vanilla/swimmer_baseline_sac.yaml
echo ""

echo "✅ All swimmer vanilla SAC experiments complete!"
