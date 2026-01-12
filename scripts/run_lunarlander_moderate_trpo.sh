#!/bin/bash
# LunarLander Moderate - TRPO
# Runs baseline + NS-MDMPI (if applicable)

echo "Running LunarLander Moderate (TRPO)"
echo "═══════════════════════════════════════"

# Count configs
CONFIGS=(configs/TRPO/moderate/lunarlander_*_baseline_trpo.yaml)
NUM_CONFIGS=${#CONFIGS[@]}

echo "Running $NUM_CONFIGS configs × 2 methods = $((NUM_CONFIGS * 2)) experiments"
echo ""

# Baseline runs
echo "=== BASELINE ==="
for config in configs/TRPO/moderate/lunarlander_*_baseline_trpo.yaml; do
    if [ -f "$config" ]; then
        filename=$(basename "$config" .yaml)
        echo "  Starting: $filename"
        conda run -n rl_hf_course python scripts/train.py \
          --config "$config" \
          > "logs/${filename}.log" 2>&1 &
        sleep 2
    fi
done

echo ""
echo "=== NS-MDMPI ===" 
for config in configs/TRPO/moderate/lunarlander_*_nsmdmpi_trpo.yaml; do
    if [ -f "$config" ]; then
        filename=$(basename "$config" .yaml)
        echo "  Starting: $filename"
        conda run -n rl_hf_course python scripts/train.py \
          --config "$config" \
          > "logs/${filename}.log" 2>&1 &
        sleep 2
    fi
done

echo ""
echo "✅ All experiments launched"
echo "Monitor: watch 'ps aux | grep train.py | grep -v grep | wc -l'"
