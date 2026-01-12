#!/bin/bash
# Run Hopper Extreme Comparison - ALL configs (Baseline vs NS-MDMPI)

echo "================================================================"
echo "  Hopper Extreme: Complete Comparison Suite"
echo "================================================================"
echo ""
echo "Running ALL extreme drift configs:"
echo "  - Friction: random_walk, jump"
echo "  - Total: 4 configs (2 baseline + 2 NS-MDMPI)"
echo ""
echo "WandB Project: att_3_Hopper_Extreme_Comparison"
echo "================================================================"
echo ""

# Array of config names
CONFIGS=(
    "hopper_friction_random_walk"
    "hopper_friction_jump"
)

# Update WandB project names first
for config in "${CONFIGS[@]}"; do
    sed -i 's/att_3_Extreme_Baseline/att_3_Hopper_Extreme_Comparison/' "configs/PPO/extreme/${config}_baseline_ppo.yaml" 2>/dev/null
    sed -i 's/att_3_Extreme_NSMDMPI/att_3_Hopper_Extreme_Comparison/' "configs/PPO/extreme/${config}_nsmdmpi_ppo.yaml" 2>/dev/null
done

# Run all baseline configs
echo "=== BASELINE RUNS ==="
for config in "${CONFIGS[@]}"; do
    if [ -f "configs/PPO/extreme/${config}_baseline_ppo.yaml" ]; then
        echo "[Baseline] Starting ${config}..."
        conda run -n rl_hf_course python scripts/train.py \
          --config "configs/PPO/extreme/${config}_baseline_ppo.yaml" \
          > "logs/${config}_baseline_ppo.log" 2>&1 &
        sleep 2
    fi
done

echo ""
echo "✅ Launched baseline runs"
echo ""

# Run all NS-MDMPI configs
echo "=== NS-MDMPI RUNS ==="
for config in "${CONFIGS[@]}"; do
    if [ -f "configs/PPO/extreme/${config}_nsmdmpi_ppo.yaml" ]; then
        echo "[NS-MDMPI] Starting ${config}..."
        conda run -n rl_hf_course python scripts/train.py \
          --config "configs/PPO/extreme/${config}_nsmdmpi_ppo.yaml" \
          > "logs/${config}_nsmdmpi_ppo.log" 2>&1 &
        sleep 2
    fi
done

echo ""
echo "✅ Launched NS-MDMPI runs"
echo ""
echo "================================================================"
echo "✅ ALL EXTREME EXPERIMENTS LAUNCHED!"
echo "================================================================"
echo ""
echo "Monitor: tail -f logs/hopper_friction_random_walk_baseline_ppo.log"
echo "WandB: att_3_Hopper_Extreme_Comparison"
echo "================================================================"
