#!/bin/bash
# Extreme Drift Tuning for HalfCheetah

echo "================================================================"
echo "  NS-MDMPI EXTREME DRIFT TUNING - HALFCHEETAH"
echo "================================================================"
echo ""
echo "Environment: HalfCheetah-v4"
echo "Drift Types: Random Walk, Jump"
echo "Trials: 50 per config"
echo "================================================================"
echo ""

read -p "Start tuning? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

declare -A CONFIGS=(
    ["friction_random_walk"]="configs/PPO/extreme/half cheetah_friction_random_walk_baseline_ppo.yaml"
    ["friction_jump"]="configs/PPO/extreme/half cheetah_friction_jump_baseline_ppo.yaml"
)

for name in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$name]}"
    
    echo ""
    echo "Tuning: $name"
    echo "---"
    
    [ ! -f "$config" ] && echo "⚠️  Config not found, skipping" && continue
    
    python scripts/tune_hyperparameters.py \
        --env "HalfCheetah-v4" \
        --config "$config" \
        --type extreme \
        --n-trials 50 \
        --n-jobs 4 \
        --study-name "extreme_halfcheetah_${name}"
    
    [ $? -eq 0 ] && echo "✅ Completed: $name" || echo "❌ Failed: $name"
done

echo ""
echo "✅ HalfCheetah extreme tuning complete!"
