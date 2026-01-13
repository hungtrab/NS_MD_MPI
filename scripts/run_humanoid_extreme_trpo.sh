#!/bin/bash
# Run Humanoid extreme experiments with TRPO

echo "========================================="
echo "  HUMANOID - EXTREME - TRPO"
echo "========================================="
echo "Total experiments: 4"
echo "========================================="
echo ""

echo "Experiment 1/4: humanoid_friction_jump_baseline_trpo"
python scripts/train.py --config configs/TRPO/extreme/humanoid_friction_jump_baseline_trpo.yaml
echo ""

echo "Experiment 2/4: humanoid_friction_jump_nsmdmpi_trpo"
python scripts/train.py --config configs/TRPO/extreme/humanoid_friction_jump_nsmdmpi_trpo.yaml
echo ""

echo "Experiment 3/4: humanoid_friction_random_walk_baseline_trpo"
python scripts/train.py --config configs/TRPO/extreme/humanoid_friction_random_walk_baseline_trpo.yaml
echo ""

echo "Experiment 4/4: humanoid_friction_random_walk_nsmdmpi_trpo"
python scripts/train.py --config configs/TRPO/extreme/humanoid_friction_random_walk_nsmdmpi_trpo.yaml
echo ""

echo "✅ All humanoid extreme TRPO experiments complete!"
