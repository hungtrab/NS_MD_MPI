#!/bin/bash
# Phase 1: LunarLander Baseline Validation
# Enhanced with detailed variation budget logging

echo "================================================================"
echo "  PHASE 1: LUNARLANDER BASELINE VALIDATION"
echo "================================================================"
echo ""
echo "This will run 3 experiments to establish baseline performance:"
echo ""
echo "  1.1 Vanilla (no drift) - Sanity check (~1.5h)"
echo "  1.2 Moderate baseline - PPO only (~1.5h)"
echo "  1.3 Moderate NS-MDMPI - With variation tracking (~2h)"
echo ""
echo "Total time: ~5 hours"
echo ""
echo "Variation metrics will be logged to WandB:"
echo "  - Budget consumption (V_R, V_P, V_π*)"
echo "  - Trust region adaptation (κ_t)"
echo "  - Regularization (λ_t)"
echo "  - Drift detection (Δ_R, Δ_P, Δ_C)"
echo "================================================================"
echo ""

# Create output directory
mkdir -p logs/phase1

# Check if configs exist
echo "Verifying configs..."
configs=(
    "configs/PPO/vanilla/lunarlander_baseline_ppo.yaml"
    "configs/PPO/moderate/lunarlander_gravity_sine_baseline_ppo.yaml"
    "configs/PPO/moderate/lunarlander_gravity_sine_nsmdmpi_ppo.yaml"
)

for config in "${configs[@]}"; do
    if [ ! -f "$config" ]; then
        echo "❌ Missing config: $config"
        exit 1
    fi
done
echo "✅ All configs found"
echo ""

read -p "Start Phase 1 testing? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Experiment counter
total=3
current=0

run_experiment() {
    current=$((current + 1))
    exp_name=$1
    config=$2
    
    echo ""
    echo "================================================================"
    echo "  EXPERIMENT $current/$total: $exp_name"
    echo "================================================================"
    echo "Config: $config"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Run training
    python scripts/train.py --config "$config" 2>&1 | tee "logs/phase1/${exp_name}.log"
    
    exit_code=$?
    echo ""
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ COMPLETED: $exp_name"
    else
        echo "❌ FAILED: $exp_name (exit code: $exit_code)"
        return 1
    fi
    
    echo "================================================================"
    echo ""
    
    # Brief pause between experiments
    if [ $current -lt $total ]; then
        echo "Pausing 10 seconds before next experiment..."
        sleep 10
    fi
}

# Run experiments sequentially
run_experiment "1.1_vanilla" "${configs[0]}"
run_experiment "1.2_moderate_baseline" "${configs[1]}"  
run_experiment "1.3_moderate_nsmdmpi" "${configs[2]}"

# Summary
echo ""
echo "================================================================"
echo "  PHASE 1 COMPLETE!"
echo "================================================================"
echo ""
echo "📊 Results logged to WandB:"
echo "   Project: att_3_LunarLander_Moderate_Comparison"
echo ""
echo "📁 Local logs:"
echo "   logs/phase1/*.log"
echo ""
echo "💾 Budget histories (Exp 1.3 only):"
echo "   budgets/budget_history_*.json"
echo ""
echo "📋 Next Steps:"
echo "   1. Open WandB dashboard"
echo "   2. Compare learning curves"
echo "   3. Check variation budget consumption"
echo "   4. Analyze trust region adaptation"
echo "   5. Document findings"
echo ""
echo "================================================================"
