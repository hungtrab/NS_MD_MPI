#!/bin/bash
# Master Script: Run ALL Extreme Experiments
# All environments, all algorithms (PPO, SAC, TRPO), baseline + NS-MDMPI

echo "================================================================"
echo "  EXTREME DRIFT EXPERIMENTS - COMPLETE SUITE"
echo "================================================================"
echo ""
echo "Environments: Hopper, HalfCheetah, LunarLander"
echo "Algorithms: PPO, SAC, TRPO"
echo "Drift Types: 2 per env (friction/gravity × random_walk/jump)"
echo "Methods: Baseline + NS-MDMPI"
echo ""
echo "Total: ~36 experiments (3 envs × 2 drifts × 3 algos × 2 methods)"
echo "Estimated time: 10-15 hours (parallel)"
echo "================================================================"
echo ""

read -p "Launch ALL extreme experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

count=0

# Loop through all extreme configs
for config in configs/PPO/extreme/*_baseline_ppo.yaml configs/PPO/extreme/*_nsmdmpi_ppo.yaml \
              configs/SAC/extreme/*_baseline_sac.yaml configs/SAC/extreme/*_nsmdmpi_sac.yaml \
              configs/TRPO/extreme/*_baseline_trpo.yaml configs/TRPO/extreme/*_nsmdmpi_trpo.yaml; do
    
    if [ -f "$config" ]; then
        filename=$(basename "$config" .yaml)
        
        echo "[$(date +%H:%M:%S)] Starting: $filename"
        
        conda run -n rl_hf_course python scripts/train.py \
          --config "$config" \
          > "logs/${filename}.log" 2>&1 &
        
        ((count++))
        sleep 3
        
        if [ $((count % 5)) -eq 0 ]; then
            echo "  [Checkpoint] Launched $count experiments, pausing..."
            sleep 10
        fi
    fi
done

echo ""
echo "================================================================"
echo "✅ LAUNCHED $count EXTREME EXPERIMENTS!"
echo "================================================================"
echo ""
echo "WandB Projects:"
echo "  - att_3_Hopper_Extreme_Comparison"
echo "  - att_3_HalfCheetah_Extreme_Comparison"
echo "  - att_3_LunarLander_Extreme_Comparison"
echo "================================================================"
