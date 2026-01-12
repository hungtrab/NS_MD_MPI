#!/bin/bash
# Run Hopper Multi-Parameter Comparison - ALL configs (Baseline vs NS-MDMPI)

echo "================================================================"
echo "  Hopper Multi-Parameter: Complete Comparison Suite"
echo "================================================================"
echo ""
echo "Running multi-parameter drift configs:"
echo "  - Friction + Mass Scale"
echo "  - Total: 2 configs (1 baseline + 1 NS-MDMPI)"
echo ""
echo "WandB Project: att_3_Hopper_Multi_Comparison"
echo "================================================================"
echo ""

# Array of config names
CONFIGS=(
    "hopper_friction_mass_scale"
)

# Update WandB project names first
for config in "${CONFIGS[@]}"; do
    sed -i 's/att_3_Multi_Baseline/att_3_Hopper_Multi_Comparison/' "configs/PPO/multi/${config}_baseline_ppo.yaml" 2>/dev/null
    sed -i 's/att_3_Multi_NSMDMPI/att_3_Hopper_Multi_Comparison/' "configs/PPO/multi/${config}_nsmdmpi_ppo.yaml" 2>/dev/null
done

# Run baseline
echo "=== BASELINE RUN ==="
for config in "${CONFIGS[@]}"; do
    if [ -f "configs/PPO/multi/${config}_baseline_ppo.yaml" ]; then
        echo "[Baseline] Starting ${config}..."
        conda run -n rl_hf_course python scripts/train.py \
          --config "configs/PPO/multi/${config}_baseline_ppo.yaml" \
          > "logs/${config}_baseline_ppo.log" 2>&1 &
        sleep 2
    fi
done

echo ""
echo "✅ Launched baseline run"
echo ""

# Run NS-MDMPI
echo "=== NS-MDMPI RUN ==="
for config in "${CONFIGS[@]}"; do
    if [ -f "configs/PPO/multi/${config}_nsmdmpi_ppo.yaml" ]; then
        echo "[NS-MDMPI] Starting ${config}..."
        conda run -n rl_hf_course python scripts/train.py \
          --config "configs/PPO/multi/${config}_nsmdmpi_ppo.yaml" \
          > "logs/${config}_nsmdmpi_ppo.log" 2>&1 &
        sleep 2
    fi
done

echo ""
echo "✅ Launched NS-MDMPI run"
echo ""
echo "================================================================"
echo "✅ ALL MULTI-PARAMETER EXPERIMENTS LAUNCHED!"
echo "================================================================"
echo ""
echo "Monitor: tail -f logs/hopper_friction_mass_scale_baseline_ppo.log"
echo "WandB: att_3_Hopper_Multi_Comparison"
echo "================================================================"
