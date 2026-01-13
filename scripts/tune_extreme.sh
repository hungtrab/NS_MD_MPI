#!/bin/bash
# Launch Optuna Hyperparameter Tuning for Extreme Drift

echo "================================================================"
echo "  NS-MDMPI Hyperparameter Tuning - EXTREME DRIFT"
echo "================================================================"
echo ""
echo "Environment: Hopper-v4"
echo "Config: Extreme Friction Random Walk"
echo "Trials: 50"
echo "================================================================"
echo ""

python scripts/tune_hyperparameters.py \
  --env "Hopper-v4" \
  --config "configs/PPO/extreme/hopper_friction_random_walk_baseline_ppo.yaml" \
  --type extreme \
  --n-trials 50 \
  --n-jobs 4 \
  --quick \
  --study-name "extreme_hopper_friction_random_walk"

echo ""
echo "✅ Extreme drift tuning complete!"
echo "Results in: results/tuned_params/extreme_hopper_friction_random_walk_best_params.yaml"
