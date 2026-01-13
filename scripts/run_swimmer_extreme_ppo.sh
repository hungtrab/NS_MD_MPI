#!/bin/bash
# Run Swimmer extreme experiments with PPO

echo "========================================="
echo "  SWIMMER - EXTREME - PPO"
echo "========================================="
echo "Total experiments: 4"
echo "========================================="
echo ""

echo "Experiment 1/4: swimmer_friction_jump_baseline_ppo"
python scripts/train.py --config configs/PPO/extreme/swimmer_friction_jump_baseline_ppo.yaml
echo ""

echo "Experiment 2/4: swimmer_friction_jump_nsmdmpi_ppo"
python scripts/train.py --config configs/PPO/extreme/swimmer_friction_jump_nsmdmpi_ppo.yaml
echo ""

echo "Experiment 3/4: swimmer_friction_random_walk_baseline_ppo"
python scripts/train.py --config configs/PPO/extreme/swimmer_friction_random_walk_baseline_ppo.yaml
echo ""

echo "Experiment 4/4: swimmer_friction_random_walk_nsmdmpi_ppo"
python scripts/train.py --config configs/PPO/extreme/swimmer_friction_random_walk_nsmdmpi_ppo.yaml
echo ""

echo "✅ All swimmer extreme PPO experiments complete!"
