#!/bin/bash
# Extreme Drift Tuning for LunarLander

echo "================================================================"
echo "  NS-MDMPI EXTREME DRIFT TUNING - LUNARLANDER"
echo "================================================================"
echo ""
echo "Environment: LunarLander-v2"
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
    ["gravity_random_walk"]="configs/PPO/extreme/lunarlander_gravity_random_walk_baseline_ppo.yaml"
    ["gravity_jump"]="configs/PPO/extreme/lunarlander_gravity_jump_baseline_ppo.yaml"
)

for name in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$name]}"
    
    echo ""
    echo "Tuning: $name"
    echo "---"
    
    [ ! -f "$config" ] && echo "⚠️  Config not found, skipping" && continue
    
    python scripts/tune_hyperparameters.py \
        --env "LunarLander-v2" \
        --config "$config" \
        --type extreme \
        --n-trials 50 \
        --n-jobs 4 \
        --study-name "extreme_lunarlander_${name}"
    
    [ $? -eq 0 ] && echo "✅ Completed: $name" || echo "❌ Failed: $name"
done

echo ""
echo "✅ LunarLander extreme tuning complete!"
