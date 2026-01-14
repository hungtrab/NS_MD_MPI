#!/bin/bash
# Phase 1: LunarLander Baseline Validation - PARALLEL EXECUTION
# All 3 experiments run simultaneously

echo "================================================================"
echo "  PHASE 1: LUNARLANDER BASELINE VALIDATION (PARALLEL)"
echo "================================================================"
echo ""
echo "This will run 3 experiments IN PARALLEL:"
echo ""
echo "  1.1 Vanilla (no drift)"
echo "  1.2 Moderate baseline" 
echo "  1.3 Moderate NS-MDMPI"
echo ""
echo "⚠️  REQUIREMENTS:"
echo "  - Enough GPU memory for 3 concurrent runs"
echo "  - Or run on CPU (slower but works)"
echo ""
echo "Total time: ~2 hours (vs 5 hours sequential)"
echo "================================================================"
echo ""

# Create output directory
mkdir -p logs/phase1
mkdir -p budgets

# Configs
configs=(
    "configs/PPO/vanilla/lunarlander_baseline_ppo.yaml"
    "configs/PPO/moderate/lunarlander_gravity_sine_baseline_ppo.yaml"
    "configs/PPO/moderate/lunarlander_gravity_sine_nsmdmpi_ppo.yaml"
)

names=(
    "1.1_vanilla"
    "1.2_moderate_baseline"
    "1.3_moderate_nsmdmpi"
)

# Check configs exist
echo "Verifying configs..."
for config in "${configs[@]}"; do
    if [ ! -f "$config" ]; then
        echo "❌ Missing config: $config"
        exit 1
    fi
done
echo "✅ All configs found"
echo ""

read -p "Start all 3 experiments in parallel? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

echo ""
echo "Starting experiments in background..."
echo ""

# Start all experiments in background
pids=()
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    name="${names[$i]}"
    logfile="logs/phase1/${name}.log"
    
    echo "[$((i+1))/3] Starting: $name"
    echo "  Config: $config"
    echo "  Log: $logfile"
    
    # Run in background, redirect output
    nohup python scripts/train.py --config "$config" > "$logfile" 2>&1 &
    pid=$!
    pids+=($pid)
    
    echo "  PID: $pid"
    echo ""
    
    # Brief pause to avoid race conditions
    sleep 2
done

echo "================================================================"
echo "✅ All experiments started!"
echo "================================================================"
echo ""
echo "Process IDs:"
for i in "${!pids[@]}"; do
    echo "  ${names[$i]}: PID ${pids[$i]}"
done
echo ""
echo "📊 Monitor progress:"
echo ""
echo "  # Check if running"
for pid in "${pids[@]}"; do
    echo "  ps -p $pid"
done
echo ""
echo "  # Watch logs (pick one)"
for i in "${!names[@]}"; do
    echo "  tail -f logs/phase1/${names[$i]}.log"
done
echo ""
echo "  # WandB Dashboard"
echo "  https://wandb.ai → att_3_LunarLander_Moderate_Comparison"
echo ""
echo "================================================================"
echo ""
echo "🔍 Check status with:"
echo "  bash scripts/check_phase1_status.sh"
echo ""
echo "⏹️  Stop all experiments:"
echo "  kill ${pids[@]}"
echo ""
echo "================================================================"

# Save PIDs for status checking
echo "${pids[@]}" > logs/phase1/pids.txt
