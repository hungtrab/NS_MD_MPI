#!/bin/bash
# Launch Optuna Hyperparameter Tuning for Moderate Drift

echo "================================================================"
echo "  NS-MDMPI Hyperparameter Tuning - MODERATE DRIFT"
echo "================================================================"
echo ""
echo "This will optimize hyperparameters using Optuna"
echo "Environment: Hopper-v4"
echo "Config: Moderate Friction Sine Drift"
echo "Trials: 50 (can be interrupted and resumed)"
echo "================================================================"
echo ""

# Quick test run first (5 trials to validate setup)
echo "Running quick validation (5 trials)..."
python scripts/tune_hyperparameters.py \
  --env "Hopper-v4" \
  --config "configs/PPO/moderate/hopper_friction_sine_baseline_ppo.yaml" \
  --type moderate \
  --n-trials 5 \
  --n-jobs 2 \
  --quick \
  --study-name "test_moderate_hopper"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Validation successful!"
    echo ""
    read -p "Continue with full tuning (50 trials)? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Starting full hyperparameter tuning..."
        echo "This will take ~8-10 hours with 4 parallel jobs"
        echo ""
        
        # Full tuning run
        python scripts/tune_hyperparameters.py \
          --env "Hopper-v4" \
          --config "configs/PPO/moderate/hopper_friction_sine_baseline_ppo.yaml" \
          --type moderate \
          --n-trials 50 \
          --n-jobs 4 \
          --quick \
          --study-name "moderate_hopper_friction_sine"
        
        echo ""
        echo "================================================================"
        echo "✅ TUNING COMPLETE!"
        echo "================================================================"
        echo ""
        echo "Results saved to:"
        echo "  - results/optuna_studies/moderate_hopper_friction_sine.db"
        echo "  - results/tuned_params/moderate_hopper_friction_sine_best_params.yaml"
        echo ""
        echo "Next steps:"
        echo "  1. Review best parameters"
        echo "  2. Validate with full-length runs (1M timesteps)"
        echo "  3. Apply to other moderate drift configs"
        echo "================================================================"
    else
        echo "Tuning cancelled."
    fi
else
    echo "❌ Validation failed. Please check errors above."
fi
