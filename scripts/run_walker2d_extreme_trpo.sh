#!/bin/bash
# Run Walker2d extreme experiments with TRPO

echo "========================================="
echo "  WALKER2D - EXTREME - TRPO"
echo "========================================="
echo "Total experiments: 4"
echo "========================================="
echo ""

echo "Experiment 1/4: walker2d_friction_jump_baseline_trpo"
python scripts/train.py --config configs/TRPO/extreme/walker2d_friction_jump_baseline_trpo.yaml
echo ""

echo "Experiment 2/4: walker2d_friction_jump_nsmdmpi_trpo"
python scripts/train.py --config configs/TRPO/extreme/walker2d_friction_jump_nsmdmpi_trpo.yaml
echo ""

echo "Experiment 3/4: walker2d_friction_random_walk_baseline_trpo"
python scripts/train.py --config configs/TRPO/extreme/walker2d_friction_random_walk_baseline_trpo.yaml
echo ""

echo "Experiment 4/4: walker2d_friction_random_walk_nsmdmpi_trpo"
python scripts/train.py --config configs/TRPO/extreme/walker2d_friction_random_walk_nsmdmpi_trpo.yaml
echo ""

echo "✅ All walker2d extreme TRPO experiments complete!"
