#!/bin/bash
# Run Hopper Moderate Comparison - ALL configs (Baseline vs NS-MDMPI)
# Tests all drift types: sine, linear, jump for both friction and mass_scale

echo "================================================================"
echo "  Hopper Moderate: Complete Comparison Suite"
echo "================================================================"
echo ""
echo "Running ALL moderate drift configs:"
echo "  - Friction: sine, linear, jump"
echo "  - Mass Scale: sine, linear, jump"
echo "  - Total: 12 configs (6 baseline + 6 NS-MDMPI)"
echo ""
echo "WandB Project: att_3_Hopper_Moderate_Comparison"
echo "================================================================"
echo ""

# Array of config names (without baseline/nsmdmpi suffix)
CONFIGS=(
    "hopper_friction_sine"
    "hopper_friction_linear"
    "hopper_friction_jump"
    "hopper_mass_scale_sine"
    "hopper_mass_scale_linear"
    "hopper_mass_scale_jump"
)

# Run all baseline configs
echo "=== BASELINE RUNS ==="
for config in "${CONFIGS[@]}"; do
    echo "[Baseline] Starting ${config}..."
    conda run -n rl_hf_course python scripts/train.py \
      --config "configs/PPO/moderate/${config}_baseline_ppo.yaml" \
      > "logs/${config}_baseline_ppo.log" 2>&1 &
    sleep 2
done

echo ""
echo "✅ Launched 6 baseline runs"
echo ""

# Run all NS-MDMPI configs
echo "=== NS-MDMPI RUNS ==="
for config in "${CONFIGS[@]}"; do
    echo "[NS-MDMPI] Starting ${config}..."
    conda run -n rl_hf_course python scripts/train.py \
      --config "configs/PPO/moderate/${config}_nsmdmpi_ppo.yaml" \
      > "logs/${config}_nsmdmpi_ppo.log" 2>&1 &
    sleep 2
done

echo ""
echo "✅ Launched 6 NS-MDMPI runs"
echo ""
echo "================================================================"
echo "✅ ALL 12 EXPERIMENTS LAUNCHED!"
echo "================================================================"
echo ""
echo "Monitor progress:"
echo "  watch -n 5 'ps aux | grep train.py | grep -v grep | wc -l'"
echo "  tail -f logs/hopper_friction_sine_baseline_ppo.log"
echo "  tail -f logs/hopper_friction_sine_nsmdmpi_ppo.log"
echo ""
echo "WandB Project (unified):"
echo "  https://wandb.ai/<your-entity>/att_3_Hopper_Moderate_Comparison"
echo ""
echo "Estimated time: 8-10 hours (parallel)"
echo "================================================================"
