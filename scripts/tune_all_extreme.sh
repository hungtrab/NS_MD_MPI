#!/bin/bash
# Master script to tune ALL environments for extreme drift

echo "================================================================"
echo "  NS-MDMPI COMPREHENSIVE EXTREME DRIFT TUNING"
echo "================================================================"
echo ""
echo "This will tune hyperparameters for ALL extreme drift scenarios:"
echo "  - Hopper (friction random_walk, friction jump)"
echo "  - HalfCheetah (friction random_walk, friction jump)"  
echo "  - LunarLander (gravity random_walk, gravity jump)"
echo ""
echo "⚠️  WARNING: This will take MANY HOURS (possibly 10-15 hours)"
echo "================================================================"
echo ""

read -p "Proceed with full extreme tuning? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

TOTAL_ENVS=3
CURRENT=0
FAILED=()

tune_env() {
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "================================================================"
    echo "  [$CURRENT/$TOTAL_ENVS] $1"
    echo "================================================================"
    bash "$2"
    if [ $? -ne 0 ]; then
        FAILED+=("$1")
        echo "❌ Failed: $1"
    else
        echo "✅ Completed: $1"
    fi
}

# Tune each environment
tune_env "Hopper Extreme" "scripts/tune_hopper_extreme.sh"
tune_env "HalfCheetah Extreme" "scripts/tune_halfcheetah_extreme.sh"
tune_env "LunarLander Extreme" "scripts/tune_lunarlander_extreme.sh"

# Summary
echo ""
echo "================================================================"
echo "  EXTREME DRIFT TUNING COMPLETE!"
echo "================================================================"
echo ""
echo "Completed: $((TOTAL_ENVS - ${#FAILED[@]}))/$TOTAL_ENVS"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "❌ Failed environments:"
    for env in "${FAILED[@]}"; do
        echo "  - $env"
    done
else
    echo ""
    echo "✅ All environments tuned successfully!"
fi

echo ""
echo "Results directory: results/tuned_params/"
echo "================================================================"
