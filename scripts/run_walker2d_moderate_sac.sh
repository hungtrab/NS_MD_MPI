#!/bin/bash
# Run Walker2d moderate experiments with SAC

echo "========================================="
echo "  WALKER2D - MODERATE - SAC"
echo "========================================="
echo "Total experiments: 12"
echo "========================================="
echo ""

echo "Experiment 1/12: walker2d_damping_linear_baseline_sac"
python scripts/train.py --config configs/SAC/moderate/walker2d_damping_linear_baseline_sac.yaml
echo ""

echo "Experiment 2/12: walker2d_damping_linear_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/moderate/walker2d_damping_linear_nsmdmpi_sac.yaml
echo ""

echo "Experiment 3/12: walker2d_damping_sine_baseline_sac"
python scripts/train.py --config configs/SAC/moderate/walker2d_damping_sine_baseline_sac.yaml
echo ""

echo "Experiment 4/12: walker2d_damping_sine_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/moderate/walker2d_damping_sine_nsmdmpi_sac.yaml
echo ""

echo "Experiment 5/12: walker2d_friction_linear_baseline_sac"
python scripts/train.py --config configs/SAC/moderate/walker2d_friction_linear_baseline_sac.yaml
echo ""

echo "Experiment 6/12: walker2d_friction_linear_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/moderate/walker2d_friction_linear_nsmdmpi_sac.yaml
echo ""

echo "Experiment 7/12: walker2d_friction_sine_baseline_sac"
python scripts/train.py --config configs/SAC/moderate/walker2d_friction_sine_baseline_sac.yaml
echo ""

echo "Experiment 8/12: walker2d_friction_sine_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/moderate/walker2d_friction_sine_nsmdmpi_sac.yaml
echo ""

echo "Experiment 9/12: walker2d_mass_scale_linear_baseline_sac"
python scripts/train.py --config configs/SAC/moderate/walker2d_mass_scale_linear_baseline_sac.yaml
echo ""

echo "Experiment 10/12: walker2d_mass_scale_linear_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/moderate/walker2d_mass_scale_linear_nsmdmpi_sac.yaml
echo ""

echo "Experiment 11/12: walker2d_mass_scale_sine_baseline_sac"
python scripts/train.py --config configs/SAC/moderate/walker2d_mass_scale_sine_baseline_sac.yaml
echo ""

echo "Experiment 12/12: walker2d_mass_scale_sine_nsmdmpi_sac"
python scripts/train.py --config configs/SAC/moderate/walker2d_mass_scale_sine_nsmdmpi_sac.yaml
echo ""

echo "✅ All walker2d moderate SAC experiments complete!"
