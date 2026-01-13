#!/bin/bash
# Run Walker2d extreme experiments with SAC

echo "========================================="
echo "  WALKER2D - EXTREME - SAC"
echo "========================================="
echo "Total experiments: 4"
echo "========================================="
echo ""

echo "Experiment 1/4: walker2d_friction_jump_baseline_sac"
python scripts/train.py --config configs/SAC/extreme/walker2d_friction_jump_baseline_sac.yaml
echo ""

echo "Experiment 2/4: walker2d_friction_jump_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/extreme/walker2d_friction_jump_nsmdmpi_sac.yaml
echo ""

echo "Experiment 3/4: walker2d_friction_random_walk_baseline_sac"
python scripts/train.py --config configs/SAC/extreme/walker2d_friction_random_walk_baseline_sac.yaml
echo ""

echo "Experiment 4/4: walker2d_friction_random_walk_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/extreme/walker2d_friction_random_walk_nsmdmpi_sac.yaml
echo ""

echo "✅ All walker2d extreme SAC experiments complete!"
