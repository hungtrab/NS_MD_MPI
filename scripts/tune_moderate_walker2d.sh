#!/bin/bash
# Hyperparameter tuning for WALKER2D - MODERATE drift

echo "================================================================"
echo "  NS-MDMPI Hyperparameter Tuning - MODERATE DRIFT"
echo "  Environment: Walker2d-v4"
echo "================================================================"
echo ""
echo "This will optimize hyperparameters using Optuna"
echo "Environment: Walker2d-v4"
echo "Config: Moderate Drift"
echo "Trials: 50 (can be interrupted and resumed)"
echo "================================================================"

# Validation run first
echo ""
echo "Running quick validation (5 trials)..."
python scripts/tune_hyperparameters.py \
    --env "Walker2d-v4" \
    --config "configs/PPO/moderate/walker2d_friction_sine_baseline_ppo.yaml" \
    --type moderate \
    --n-trials 5 \
    --n-jobs 1 \
    --quick \
    --study-name "moderate_walker2d_validation"

if [ $? -ne 0 ]; then
    echo "❌ Validation failed. Please check errors above."
    exit 1
fi

echo ""
echo "✅ Validation passed!"
echo ""
read -p "Continue with full tuning (50 trials)? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Tuning cancelled."
    exit 0
fi

# Full tuning
echo ""
echo "Starting full hyperparameter tuning..."
echo "This may take several hours. You can safely interrupt (Ctrl+C) and resume later."
echo ""

python scripts/tune_hyperparameters.py \
    --env "Walker2d-v4" \
    --config "configs/PPO/moderate/walker2d_friction_sine_baseline_ppo.yaml" \
    --type moderate \
    --n-trials 50 \
    --n-jobs 4 \
    --study-name "moderate_walker2d_full"

echo ""
echo "================================================================"
echo "✅ Tuning complete!"
echo "================================================================"
echo "Results saved to: results/tuned_params/moderate_walker2d_full_best_params.yaml"
echo ""
echo "View dashboard with:"
echo "  optuna-dashboard results/optuna_studies/moderate_walker2d_full.db"
echo "================================================================"
