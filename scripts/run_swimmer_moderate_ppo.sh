#!/bin/bash
# Run Swimmer moderate experiments with PPO

echo "========================================="
echo "  SWIMMER - MODERATE - PPO"
echo "========================================="
echo "Total experiments: 12"
echo "========================================="
echo ""

echo "Experiment 1/12: swimmer_density_linear_baseline_ppo"
python scripts/train.py --config configs/PPO/moderate/swimmer_density_linear_baseline_ppo.yaml
echo ""

echo "Experiment 2/12: swimmer_density_linear_nsmdmpi_ppo"
python scripts/train.py --config configs/PPO/moderate/swimmer_density_linear_nsmdmpi_ppo.yaml
echo ""

echo "Experiment 3/12: swimmer_density_sine_baseline_ppo"
python scripts/train.py --config configs/PPO/moderate/swimmer_density_sine_baseline_ppo.yaml
echo ""

echo "Experiment 4/12: swimmer_density_sine_nsmdmpi_ppo"
python scripts/train.py --config configs/PPO/moderate/swimmer_density_sine_nsmdmpi_ppo.yaml
echo ""

echo "Experiment 5/12: swimmer_friction_linear_baseline_ppo"
python scripts/train.py --config configs/PPO/moderate/swimmer_friction_linear_baseline_ppo.yaml
echo ""

echo "Experiment 6/12: swimmer_friction_linear_nsmdmpi_ppo"
python scripts/train.py --config configs/PPO/moderate/swimmer_friction_linear_nsmdmpi_ppo.yaml
echo ""

echo "Experiment 7/12: swimmer_friction_sine_baseline_ppo"
python scripts/train.py --config configs/PPO/moderate/swimmer_friction_sine_baseline_ppo.yaml
echo ""

echo "Experiment 8/12: swimmer_friction_sine_nsmdmpi_ppo"
python scripts/train.py --config configs/PPO/moderate/swimmer_friction_sine_nsmdmpi_ppo.yaml
echo ""

echo "Experiment 9/12: swimmer_mass_scale_linear_baseline_ppo"
python scripts/train.py --config configs/PPO/moderate/swimmer_mass_scale_linear_baseline_ppo.yaml
echo ""

echo "Experiment 10/12: swimmer_mass_scale_linear_nsmdmpi_ppo"
python scripts/train.py --config configs/PPO/moderate/swimmer_mass_scale_linear_nsmdmpi_ppo.yaml
echo ""

echo "Experiment 11/12: swimmer_mass_scale_sine_baseline_ppo"
python scripts/train.py --config configs/PPO/moderate/swimmer_mass_scale_sine_baseline_ppo.yaml
echo ""

echo "Experiment 12/12: swimmer_mass_scale_sine_nsmdmpi_ppo"
python scripts/train.py --config configs/PPO/moderate/swimmer_mass_scale_sine_nsmdmpi_ppo.yaml
echo ""

echo "✅ All swimmer moderate PPO experiments complete!"
