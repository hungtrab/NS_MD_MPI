#!/bin/bash
# Comprehensive Extreme Drift Tuning for Hopper
# Tunes NS-MDMPI hyperparameters for extreme drift scenarios

echo "================================================================"
echo "  NS-MDMPI EXTREME DRIFT TUNING - HOPPER"
echo "================================================================"
echo ""
echo "This will tune NS-MDMPI hyperparameters for extreme drift"
echo "Environment: Hopper-v4"
echo "Drift Types: Random Walk, Jump"
echo "Trials: 50 per config"
echo "================================================================"
echo ""

# Check if user wants to proceed
read -p "Start tuning? This will take several hours. [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Array of Hopper extreme configs to tune
declare -A CONFIGS=(
    ["friction_random_walk"]="configs/PPO/extreme/hopper_friction_random_walk_baseline_ppo.yaml"
    ["friction_jump"]="configs/PPO/extreme/hopper_friction_jump_baseline_ppo.yaml"
)

TOTAL=${#CONFIGS[@]}
CURRENT=0

echo ""
echo "Will tune $TOTAL extreme configurations"
echo "================================================================"

for name in "${!CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    config="${CONFIGS[$name]}"
    
    echo ""
    echo "[$CURRENT/$TOTAL] Tuning: $name"
    echo "================================================================"
    
    if [ ! -f "$config" ]; then
        echo "⚠️  Config not found: $config"
        echo "   Skipping..."
        continue
    fi
    
    # Run tuning
    python scripts/tune_hyperparameters.py \
        --env "Hopper-v4" \
        --config "$config" \
        --type extreme \
        --n-trials 50 \
        --n-jobs 4 \
        --study-name "extreme_hopper_${name}"
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed tuning: $name"
        echo "   Results: results/tuned_params/extreme_hopper_${name}_best_params.yaml"
    else
        echo "❌ Failed tuning: $name"
    fi
done

echo ""
echo "================================================================"
echo "✅  EXTREME DRIFT TUNING COMPLETE!"
echo "================================================================"
echo ""
echo "Results saved to:"
echo "  - Optuna databases: results/optuna_studies/extreme_hopper_*.db"
echo "  - Best params: results/tuned_params/extreme_hopper_*_best_params.yaml"
echo ""
echo "Next steps:"
echo "  1. Review best parameters in results/tuned_params/"
echo "  2. Apply to configs and run validation"
echo "  3. Compare extreme tuned vs moderate tuned performance"
echo "================================================================"
