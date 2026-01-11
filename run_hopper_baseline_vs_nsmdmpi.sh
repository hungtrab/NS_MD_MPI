#!/bin/bash
# Run Hopper Baseline vs NS-MDMPI Comparison (Seeds 0-4)

echo "================================================================"
echo "  Hopper: Baseline vs NS-MDMPI Comparison"
echo "================================================================"
echo ""
echo "Running 10 experiments:"
echo "  - Baseline (no adaptation): Seeds 0-4 (5 runs)"
echo "  - NS-MDMPI (adaptive): Seeds 0-4 (5 runs)"
echo ""
echo "Environment: Hopper-v4 with friction sine drift"
echo "Timesteps: 500k per run"
echo "================================================================"
echo ""

# Create temp configs directory
mkdir -p configs/temp_comparison

# Run Baseline experiments (Seeds 0-4)
echo "[1/2] Launching Baseline experiments (Seeds 0-4)..."
for seed in 0 1 2 3 4; do
    # Create temp config with updated seed
    sed "s/seed: 42/seed: $seed/" configs/hopper_baseline_comparison.yaml > configs/temp_comparison/hopper_baseline_seed${seed}.yaml
    
    conda run -n rl_hf_course python scripts/train.py \
      --config configs/temp_comparison/hopper_baseline_seed${seed}.yaml \
      > logs/hopper_baseline_seed${seed}.log 2>&1 &
    
    echo "  ✅ Started Baseline seed $seed"
    sleep 2
done

echo ""
echo "[2/2] Launching NS-MDMPI experiments (Seeds 0-4)..."
for seed in 0 1 2 3 4; do
    # Create temp config with updated seed
    sed "s/seed: 42/seed: $seed/" configs/hopper_nsmdmpi_comparison.yaml > configs/temp_comparison/hopper_nsmdmpi_seed${seed}.yaml
    
    conda run -n rl_hf_course python scripts/train.py \
      --config configs/temp_comparison/hopper_nsmdmpi_seed${seed}.yaml \
      > logs/hopper_nsmdmpi_seed${seed}.log 2>&1 &
    
    echo "  ✅ Started NS-MDMPI seed $seed"
    sleep 2
done

echo ""
echo "================================================================"
echo "✅ ALL 10 EXPERIMENTS LAUNCHED!"
echo "================================================================"
echo ""
echo "Monitor progress:"
echo "  ps aux | grep train.py | grep hopper | wc -l"
echo "  tail -f logs/hopper_baseline_seed0.log"
echo "  tail -f logs/hopper_nsmdmpi_seed0.log"
echo ""
echo "WandB Dashboard:"
echo "  Project: Hopper_NSMDMPI_Comparison"
echo "  https://wandb.ai"
echo ""
echo "Results will be in logs/:"
echo "  - hopper_baseline_seed{0-4}.log"
echo "  - hopper_nsmdmpi_seed{0-4}.log"
echo ""
echo "Estimated time: ~4-6 hours for all runs (parallel)"
echo "================================================================"

# Wait for all jobs
wait
echo ""
echo "🎉 ALL EXPERIMENTS COMPLETED! 🎉"
