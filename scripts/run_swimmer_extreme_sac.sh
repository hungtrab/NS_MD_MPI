#!/bin/bash
# Run Swimmer extreme experiments with SAC

echo "========================================="
echo "  SWIMMER - EXTREME - SAC"
echo "========================================="
echo "Total experiments: 4"
echo "========================================="
echo ""

echo "Experiment 1/4: swimmer_friction_jump_baseline_sac"
python scripts/train.py --config configs/SAC/extreme/swimmer_friction_jump_baseline_sac.yaml
echo ""

echo "Experiment 2/4: swimmer_friction_jump_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/extreme/swimmer_friction_jump_nsmdmpi_sac.yaml
echo ""

echo "Experiment 3/4: swimmer_friction_random_walk_baseline_sac"
python scripts/train.py --config configs/SAC/extreme/swimmer_friction_random_walk_baseline_sac.yaml
echo ""

echo "Experiment 4/4: swimmer_friction_random_walk_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/extreme/swimmer_friction_random_walk_nsmdmpi_sac.yaml
echo ""

echo "✅ All swimmer extreme SAC experiments complete!"
