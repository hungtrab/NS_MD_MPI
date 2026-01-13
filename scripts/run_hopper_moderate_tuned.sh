#!/bin/bash
# Run Hopper Moderate with Tuned NS-MDMPI Hyperparameters
# This script tests the NS-MDMPI algorithm with hyperparameters optimized via Optuna

echo "================================================================"
echo "  HOPPER MODERATE - TUNED NS-MDMPI VALIDATION"
echo "================================================================"
echo ""
echo "This will run NS-MDMPI with tuned hyperparameters on all"
echo "Hopper moderate drift configurations."
echo ""
echo "Tuned params from: results/tuned_params/"
echo "WandB Project: att_3_Hopper_Moderate_Comparison"
echo "================================================================"
echo ""

# Array of Hopper moderate configs (NS-MDMPI only)
configs=(
    "configs/PPO/moderate/hopper_friction_sine_nsmdmpi_ppo_tuned.yaml"
    "configs/PPO/moderate/hopper_friction_linear_nsmdmpi_ppo_tuned.yaml"
    "configs/PPO/moderate/hopper_mass_scale_sine_nsmdmpi_ppo_tuned.yaml"
)

total=${#configs[@]}
current=0

echo "Total experiments: $total (NS-MDMPI with tuned params)"
echo "================================================================"
echo ""

for config in "${configs[@]}"; do
    current=$((current + 1))
    
    echo "[$current/$total] Running: $(basename $config)"
    echo "---"
    
    if [ ! -f "$config" ]; then
        echo "⚠️  Config not found: $config"
        echo "   Skipping..."
        echo ""
        continue
    fi
    
    python scripts/train.py --config "$config"
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed: $(basename $config)"
    else
        echo "❌ Failed: $(basename $config)"
    fi
    
    echo ""
done

echo "================================================================"
echo "✅  VALIDATION COMPLETE!"
echo "================================================================"
echo ""
echo "Results logged to WandB: att_3_Hopper_Moderate_Comparison"
echo "Compare tuned NS-MDMPI vs original baseline/NS-MDMPI runs"
echo ""
echo "Next steps:"
echo "  1. Check WandB for performance comparison"
echo "  2. If improved, apply tuned params to other environments"
echo "  3. Run full evaluation with multiple seeds"
echo "================================================================"
