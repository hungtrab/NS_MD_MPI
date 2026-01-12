#!/bin/bash
# Master Script: Run ALL Moderate Experiments
# All environments, all algorithms (PPO, SAC, TRPO), baseline + NS-MDMPI

echo "================================================================"
echo "  MODERATE DRIFT EXPERIMENTS - COMPLETE SUITE"
echo "================================================================"
echo ""
echo "Environments: Hopper, HalfCheetah, LunarLander"
echo "Algorithms: PPO, SAC, TRPO"
echo "Drift Types: 6 per env (friction/mass/gravity × sine/linear/jump)"
echo "Methods: Baseline + NS-MDMPI"
echo ""
echo "Total: ~108 experiments (3 envs × 6 drifts × 3 algos × 2 methods)"
echo "Estimated time: 20-30 hours (parallel)"
echo "================================================================"
echo ""

read -p "Launch ALL moderate experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Counter
count=0

# Loop through all moderate configs
for config in configs/PPO/moderate/*_baseline_ppo.yaml configs/PPO/moderate/*_nsmdmpi_ppo.yaml \
              configs/SAC/moderate/*_baseline_sac.yaml configs/SAC/moderate/*_nsmdmpi_sac.yaml \
              configs/TRPO/moderate/*_baseline_trpo.yaml configs/TRPO/moderate/*_nsmdmpi_trpo.yaml; do
    
    if [ -f "$config" ]; then
        # Extract filename for logging
        filename=$(basename "$config" .yaml)
        
        echo "[$(date +%H:%M:%S)] Starting: $filename"
        
        conda run -n rl_hf_course python scripts/train.py \
          --config "$config" \
          > "logs/${filename}.log" 2>&1 &
        
        ((count++))
        
        # Sleep to avoid overwhelming the system
        sleep 3
        
        # Every 10 experiments, wait a bit longer
        if [ $((count % 10)) -eq 0 ]; then
            echo "  [Checkpoint] Launched $count experiments, pausing..."
            sleep 10
        fi
    fi
done

echo ""
echo "================================================================"
echo "✅ LAUNCHED $count MODERATE EXPERIMENTS!"
echo "================================================================"
echo ""
echo "Monitor:"
echo "  watch -n 5 'ps aux | grep train.py | grep -v grep | wc -l'"
echo "  tail -f logs/hopper_friction_sine_baseline_ppo.log"
echo ""
echo "WandB Projects:"
echo "  - att_3_Hopper_Moderate_Comparison"
echo "  - att_3_HalfCheetah_Moderate_Comparison"
echo "  - att_3_LunarLander_Moderate_Comparison"
echo "================================================================"
