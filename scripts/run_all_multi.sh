#!/bin/bash
# Master Script: Run ALL Multi-Parameter Experiments
# All environments, all algorithms (PPO, SAC, TRPO), baseline + NS-MDMPI

echo "================================================================"
echo "  MULTI-PARAMETER DRIFT EXPERIMENTS - COMPLETE SUITE"
echo "================================================================"
echo ""
echo "Environments: Hopper, HalfCheetah, LunarLander"
echo "Algorithms: PPO, SAC, TRPO"
echo "Config: 1 multi-param combo per env"
echo "Methods: Baseline + NS-MDMPI"
echo ""
echo "Total: ~18 experiments (3 envs × 1 config × 3 algos × 2 methods)"
echo "Estimated time: 5-8 hours (parallel)"
echo "================================================================"
echo ""

read -p "Launch ALL multi-parameter experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

count=0

# Loop through all multi configs
for config in configs/PPO/multi/*_baseline_ppo.yaml configs/PPO/multi/*_nsmdmpi_ppo.yaml \
              configs/SAC/multi/*_baseline_sac.yaml configs/SAC/multi/*_nsmdmpi_sac.yaml \
              configs/TRPO/multi/*_baseline_trpo.yaml configs/TRPO/multi/*_nsmdmpi_trpo.yaml; do
    
    if [ -f "$config" ]; then
        filename=$(basename "$config" .yaml)
        
        echo "[$(date +%H:%M:%S)] Starting: $filename"
        
        conda run -n rl_hf_course python scripts/train.py \
          --config "$config" \
          > "logs/${filename}.log" 2>&1 &
        
        ((count++))
        sleep 3
    fi
done

echo ""
echo "================================================================"
echo "✅ LAUNCHED $count MULTI-PARAMETER EXPERIMENTS!"
echo "================================================================"
echo ""
echo "WandB Projects:"
echo "  - att_3_Hopper_Multi_Comparison"
echo "  - att_3_HalfCheetah_Multi_Comparison"
echo "  - att_3_LunarLander_Multi_Comparison"
echo "================================================================"
