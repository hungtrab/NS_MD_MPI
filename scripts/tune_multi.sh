#!/bin/bash
# Launch Optuna Hyperparameter Tuning for Multi-Parameter Drift

echo "================================================================"
echo "  NS-MDMPI Hyperparameter Tuning - MULTI-PARAMETER DRIFT"
echo "================================================================"
echo ""
echo "Environment: Hopper-v4"
echo "Config: Multi-param (Friction + Mass Scale)"
echo "Trials: 50"
echo "================================================================"
echo ""

python scripts/tune_hyperparameters.py \
  --env "Hopper-v4" \
  --config "configs/PPO/multi/hopper_friction_mass_scale_baseline_ppo.yaml" \
  --type multi \
  --n-trials 50 \
  --n-jobs 4 \
  --quick \
  --study-name "multi_hopper_friction_mass"

echo ""
echo "✅ Multi-parameter tuning complete!"
echo "Results in: results/tuned_params/multi_hopper_friction_mass_best_params.yaml"
