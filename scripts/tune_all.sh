#!/bin/bash
# Master script to run tuning for ALL environments sequentially

echo "================================================================"
echo "  NS-MDMPI COMPREHENSIVE HYPERPARAMETER TUNING"
echo "================================================================"
echo ""
echo "This will tune hyperparameters for ALL environments:"
echo "  - Moderate drift: 6 envs (Hopper, HalfCheetah, Walker2D, Swimmer, Humanoid, LunarLander)"
echo "  - Extreme drift: 4 envs (Hopper, HalfCheetah, Walker2D, LunarLander)"
echo ""
echo "⚠️  WARNING: This will take MANY HOURS (possibly 10-20 hours total)"
echo "================================================================"
echo ""

read -p "Do you want to proceed? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Track progress
TOTAL=10
CURRENT=0
FAILED=()

tune_env() {
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "================================================================"
    echo "  [$CURRENT/$TOTAL] $1"
    echo "================================================================"
    bash "$2"
    if [ $? -ne 0 ]; then
        FAILED+=("$1")
        echo "❌ Failed: $1"
    else
        echo "✅ Completed: $1"
    fi
}

# Moderate drift
echo ""
echo "========================================"
echo "  PHASE 1: MODERATE DRIFT (6 envs)"
echo "========================================"

tune_env "Hopper Moderate" "scripts/tune_moderate_hopper.sh"
tune_env "HalfCheetah Moderate" "scripts/tune_moderate_halfcheetah.sh"
tune_env "Walker2D Moderate" "scripts/tune_moderate_walker2d.sh"
tune_env "Swimmer Moderate" "scripts/tune_moderate_swimmer.sh"
tune_env "Humanoid Moderate" "scripts/tune_moderate_humanoid.sh"
tune_env "LunarLander Moderate" "scripts/tune_moderate_lunarlander.sh"

# Extreme drift
echo ""
echo "========================================"
echo "  PHASE 2: EXTREME DRIFT (4 envs)"
echo "========================================"

tune_env "Hopper Extreme" "scripts/tune_extreme_hopper.sh"
tune_env "HalfCheetah Extreme" "scripts/tune_extreme_halfcheetah.sh"
tune_env "Walker2D Extreme" "scripts/tune_extreme_walker2d.sh"
tune_env "LunarLander Extreme" "scripts/tune_extreme_lunarlander.sh"

# Summary
echo ""
echo "================================================================"
echo "  TUNING COMPLETE!"
echo "================================================================"
echo ""
echo "Completed: $((TOTAL - ${#FAILED[@]}))/$TOTAL"

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
