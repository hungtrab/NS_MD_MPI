#!/bin/bash
# Quick start for LunarLander Phase 1 testing

echo "================================================================"
echo "  LUNARLANDER PHASE 1: BASELINE VALIDATION"
echo "================================================================"
echo ""
echo "This will run 3 experiments:"
echo "  1. Vanilla (no drift) - sanity check"
echo "  2. Moderate drift - Baseline PPO"
echo "  3. Moderate drift - NS-MDMPI"
echo ""
echo "Total time: ~4-5 hours"
echo "================================================================"
echo ""

read -p "Start Phase 1 testing? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

CONFIGS=(
    "configs/PPO/vanilla/lunarlander_baseline_ppo.yaml"
    "configs/PPO/moderate/lunarlander_gravity_sine_baseline_ppo.yaml"
    "configs/PPO/moderate/lunarlander_gravity_sine_nsmdmpi_ppo.yaml"
)

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    num=$((i + 1))
    
    echo ""
    echo "[$num/3] Running: $(basename $config)"
    echo "---"
    
    python scripts/train.py --config "$config"
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed: $(basename $config)"
    else
        echo "❌ Failed: $(basename $config)"
    fi
done

echo ""
echo "================================================================"
echo "✅ PHASE 1 COMPLETE!"
echo "================================================================"
echo ""
echo "Check results in WandB:"
echo "  Project: att_3_LunarLander_Moderate_Comparison"
echo ""
echo "Next: Analyze budget consumption (Phase 2)"
echo "================================================================"
