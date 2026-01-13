#!/bin/bash
# Run Humanoid moderate experiments with SAC

echo "========================================="
echo "  HUMANOID - MODERATE - SAC"
echo "========================================="
echo "Total experiments: 12"
echo "========================================="
echo ""

echo "Experiment 1/12: humanoid_damping_linear_baseline_sac"
python scripts/train.py --config configs/SAC/moderate/humanoid_damping_linear_baseline_sac.yaml
echo ""

echo "Experiment 2/12: humanoid_damping_linear_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/moderate/humanoid_damping_linear_nsmdmpi_sac.yaml
echo ""

echo "Experiment 3/12: humanoid_damping_sine_baseline_sac"
python scripts/train.py --config configs/SAC/moderate/humanoid_damping_sine_baseline_sac.yaml
echo ""

echo "Experiment 4/12: humanoid_damping_sine_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/moderate/humanoid_damping_sine_nsmdmpi_sac.yaml
echo ""

echo "Experiment 5/12: humanoid_friction_linear_baseline_sac"
python scripts/train.py --config configs/SAC/moderate/humanoid_friction_linear_baseline_sac.yaml
echo ""

echo "Experiment 6/12: humanoid_friction_linear_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/moderate/humanoid_friction_linear_nsmdmpi_sac.yaml
echo ""

echo "Experiment 7/12: humanoid_friction_sine_baseline_sac"
python scripts/train.py --config configs/SAC/moderate/humanoid_friction_sine_baseline_sac.yaml
echo ""

echo "Experiment 8/12: humanoid_friction_sine_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/moderate/humanoid_friction_sine_nsmdmpi_sac.yaml
echo ""

echo "Experiment 9/12: humanoid_mass_scale_linear_baseline_sac"
python scripts/train.py --config configs/SAC/moderate/humanoid_mass_scale_linear_baseline_sac.yaml
echo ""

echo "Experiment 10/12: humanoid_mass_scale_linear_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/moderate/humanoid_mass_scale_linear_nsmdmpi_sac.yaml
echo ""

echo "Experiment 11/12: humanoid_mass_scale_sine_baseline_sac"
python scripts/train.py --config configs/SAC/moderate/humanoid_mass_scale_sine_baseline_sac.yaml
echo ""

echo "Experiment 12/12: humanoid_mass_scale_sine_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/moderate/humanoid_mass_scale_sine_nsmdmpi_sac.yaml
echo ""

echo "✅ All humanoid moderate SAC experiments complete!"
