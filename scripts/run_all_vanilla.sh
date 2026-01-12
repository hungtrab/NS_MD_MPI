#!/bin/bash
# Master Script: Run ALL Vanilla Baselines
# All environments, all algorithms (PPO, SAC, TRPO)

echo "================================================================"
echo "  VANILLA BASELINES - STATIONARY ENVIRONMENTS"
echo "================================================================"
echo ""
echo "Environments: Hopper, HalfCheetah, LunarLander"
echo "Algorithms: PPO, SAC, TRPO"
echo ""
echo "Total: 9 experiments (3 envs × 3 algos)"
echo "Estimated time: 3-5 hours (parallel)"
echo "================================================================"
echo ""

count=0

# Loop through all vanilla configs
for config in configs/PPO/vanilla/*.yaml \
              configs/SAC/vanilla/*.yaml \
              configs/TRPO/vanilla/*.yaml; do
    
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
echo "✅ LAUNCHED $count VANILLA BASELINE EXPERIMENTS!"
echo "================================================================"
echo ""
echo "WandB Project: att_3_Vanilla_Baselines"
echo "================================================================"
