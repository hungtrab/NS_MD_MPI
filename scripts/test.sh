#!/bin/bash
# Test Script - Validate implementation with small runs
# Timesteps: 10000 (vs normal 500k-1M)
# Project: test_nsmdmpi

echo "================================================================"
echo "  TEST SCRIPT - Implementation Validation"
echo "================================================================"
echo ""
echo "Testing configs with 10k timesteps..."
echo "Saving to WandB project: test_nsmdmpi"
echo ""

# Track results
RESULTS_FILE="/tmp/test_results.txt"
echo "Test Results:" > $RESULTS_FILE
echo "=============" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Test 1: Hopper Moderate Baseline
echo "[1/6] Testing Hopper Moderate Baseline..."
sed 's/total_timesteps: 1000000/total_timesteps: 10000/' configs/PPO/moderate/hopper_friction_sine_baseline_ppo.yaml | \
  sed 's/att_3_Moderate_Baseline/test_nsmdmpi/' > /tmp/test_hopper_mod_base.yaml

if conda run -n rl_hf_course python scripts/train.py --config /tmp/test_hopper_mod_base.yaml > /tmp/test_hopper_mod_base.log 2>&1; then
    echo "✅ Hopper Moderate Baseline" | tee -a $RESULTS_FILE
else
    echo "❌ Hopper Moderate Baseline" | tee -a $RESULTS_FILE
    echo "   Error: $(tail -5 /tmp/test_hopper_mod_base.log)" >> $RESULTS_FILE
fi
sleep 2

# Test 2: Hopper Moderate NS-MDMPI
echo "[2/6] Testing Hopper Moderate NS-MDMPI..."
sed 's/total_timesteps: 1000000/total_timesteps: 10000/' configs/PPO/moderate/hopper_friction_sine_nsmdmpi_ppo.yaml | \
  sed 's/att_3_Moderate_NSMDMPI/test_nsmdmpi/' > /tmp/test_hopper_mod_nsmdmpi.yaml

if conda run -n rl_hf_course python scripts/train.py --config /tmp/test_hopper_mod_nsmdmpi.yaml > /tmp/test_hopper_mod_nsmdmpi.log 2>&1; then
    echo "✅ Hopper Moderate NS-MDMPI" | tee -a $RESULTS_FILE
else
    echo "❌ Hopper Moderate NS-MDMPI" | tee -a $RESULTS_FILE
    echo "   Error: $(tail -5 /tmp/test_hopper_mod_nsmdmpi.log)" >> $RESULTS_FILE
fi
sleep 2

# Test 3: Hopper Extreme Baseline
echo "[3/6] Testing Hopper Extreme Baseline..."
sed 's/total_timesteps: 1000000/total_timesteps: 10000/' configs/PPO/extreme/hopper_friction_random_walk_baseline_ppo.yaml | \
  sed 's/att_3_Extreme_Baseline/test_nsmdmpi/' > /tmp/test_hopper_ext_base.yaml

if conda run -n rl_hf_course python scripts/train.py --config /tmp/test_hopper_ext_base.yaml > /tmp/test_hopper_ext_base.log 2>&1; then
    echo "✅ Hopper Extreme Baseline" | tee -a $RESULTS_FILE
else
    echo "❌ Hopper Extreme Baseline" | tee -a $RESULTS_FILE
    echo "   Error: $(tail -5 /tmp/test_hopper_ext_base.log)" >> $RESULTS_FILE
fi
sleep 2

# Test 4: Hopper Multi Baseline
echo "[4/6] Testing Hopper Multi Baseline..."
sed 's/total_timesteps: 1000000/total_timesteps: 10000/' configs/PPO/multi/hopper_friction_mass_scale_baseline_ppo.yaml | \
  sed 's/att_3_Multi_Baseline/test_nsmdmpi/' > /tmp/test_hopper_multi_base.yaml

if conda run -n rl_hf_course python scripts/train.py --config /tmp/test_hopper_multi_base.yaml > /tmp/test_hopper_multi_base.log 2>&1; then
    echo "✅ Hopper Multi-Param Baseline" | tee -a $RESULTS_FILE
else
    echo "❌ Hopper Multi-Param Baseline" | tee -a $RESULTS_FILE
    echo "   Error: $(tail -5 /tmp/test_hopper_multi_base.log)" >> $RESULTS_FILE
fi
sleep 2

# Test 5: Hopper Vanilla
echo "[5/6] Testing Hopper Vanilla..."
sed 's/total_timesteps: 1000000/total_timesteps: 10000/' configs/PPO/vanilla/hopper_vanilla_ppo.yaml | \
  sed 's/att_3_Vanilla_Baselines/test_nsmdmpi/' > /tmp/test_hopper_vanilla.yaml

if conda run -n rl_hf_course python scripts/train.py --config /tmp/test_hopper_vanilla.yaml > /tmp/test_hopper_vanilla.log 2>&1; then
    echo "✅ Hopper Vanilla" | tee -a $RESULTS_FILE
else
    echo "❌ Hopper Vanilla" | tee -a $RESULTS_FILE
    echo "   Error: $(tail -5 /tmp/test_hopper_vanilla.log)" >> $RESULTS_FILE
fi
sleep 2

# Test 6: LunarLander Moderate Baseline
echo "[6/6] Testing LunarLander Moderate Baseline..."
sed 's/total_timesteps: 500000/total_timesteps: 10000/' configs/PPO/moderate/lunarlander_gravity_sine_baseline_ppo.yaml | \
  sed 's/att_3_Moderate_Baseline/test_nsmdmpi/' > /tmp/test_lunar_mod_base.yaml

if conda run -n rl_hf_course python scripts/train.py --config /tmp/test_lunar_mod_base.yaml > /tmp/test_lunar_mod_base.log 2>&1; then
    echo "✅ LunarLander Moderate Baseline" | tee -a $RESULTS_FILE
else
    echo "❌ LunarLander Moderate Baseline" | tee -a $RESULTS_FILE
    echo "   Error: $(tail -5 /tmp/test_lunar_mod_base.log)" >> $RESULTS_FILE
fi

echo ""
echo "================================================================"
echo "  TEST RESULTS"
echo "================================================================"
cat $RESULTS_FILE
echo "================================================================"
echo ""
echo "Check logs in /tmp/test_*.log for details"
echo "WandB project: test_nsmdmpi"
