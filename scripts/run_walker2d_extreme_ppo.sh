#!/bin/bash
# Run Walker2d extreme experiments with PPO

echo "========================================="
echo "  WALKER2D - EXTREME - PPO"
echo "========================================="
echo "Total experiments: 4"
echo "========================================="
echo ""

echo "Experiment 1/4: walker2d_friction_jump_baseline_ppo"
python scripts/train.py --config configs/PPO/extreme/walker2d_friction_jump_baseline_ppo.yaml
echo ""

echo "Experiment 2/4: walker2d_friction_jump_nsmdmpi_ppo"
python scripts/train.py --config configs/PPO/extreme/walker2d_friction_jump_nsmdmpi_ppo.yaml
echo ""

echo "Experiment 3/4: walker2d_friction_random_walk_baseline_ppo"
python scripts/train.py --config configs/PPO/extreme/walker2d_friction_random_walk_baseline_ppo.yaml
echo ""

echo "Experiment 4/4: walker2d_friction_random_walk_nsmdmpi_ppo"
python scripts/train.py --config configs/PPO/extreme/walker2d_friction_random_walk_nsmdmpi_ppo.yaml
echo ""

echo "✅ All walker2d extreme PPO experiments complete!"
