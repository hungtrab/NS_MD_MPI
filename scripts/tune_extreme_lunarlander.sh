#!/bin/bash
# Hyperparameter tuning for LUNARLANDER - EXTREME drift

echo "================================================================"
echo "  NS-MDMPI Hyperparameter Tuning - EXTREME DRIFT"
echo "  Environment: LunarLander-v2"
echo "================================================================"
echo ""
echo "This will optimize hyperparameters using Optuna"
echo "Environment: LunarLander-v2"
echo "Config: Extreme Drift"
echo "Trials: 50 (can be interrupted and resumed)"
echo "================================================================"

# Validation run first
echo ""
echo "Running quick validation (5 trials)..."
python scripts/tune_hyperparameters.py \
    --env "LunarLander-v2" \
    --config "configs/PPO/extreme/lunarlander_gravity_jump_baseline_ppo.yaml" \
    --type extreme \
    --n-trials 5 \
    --n-jobs 1 \
    --quick \
    --study-name "extreme_lunarlander_validation"

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
    --env "LunarLander-v2" \
    --config "configs/PPO/extreme/lunarlander_gravity_jump_baseline_ppo.yaml" \
    --type extreme \
    --n-trials 50 \
    --n-jobs 4 \
    --study-name "extreme_lunarlander_full"

echo ""
echo "================================================================"
echo "✅ Tuning complete!"
echo "================================================================"
echo "Results saved to: results/tuned_params/extreme_lunarlander_full_best_params.yaml"
echo ""
echo "View dashboard with:"
echo "  optuna-dashboard results/optuna_studies/extreme_lunarlander_full.db"
echo "================================================================"
