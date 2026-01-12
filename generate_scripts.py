#!/usr/bin/env python3
"""
Generate all experiment launcher scripts
Creates 36 scripts: 3 envs × 4 types × 3 algos + master + test
"""
import os
from pathlib import Path

# Environment configs
ENVS = {
    "hopper": {
        "moderate": "hopper_friction_sine",
        "extreme": "hopper_friction_random_walk",
        "multi": "hopper_friction_mass_scale",
        "vanilla": "hopper_vanilla"
    },
    "halfcheetah": {
        "moderate": "half cheetah_friction_sine",
        "extreme": "half cheetah_friction_random_walk",
        "multi": "half cheetah_friction_damping",
        "vanilla": "half cheetah_vanilla"
    },
    "lunarlander": {
        "moderate": "lunarlander_gravity_sine",
        "extreme": "lunarlander_gravity_random_walk",
        "multi": "lunarlander_gravity_wind_power",
        "vanilla": "lunarlander_vanilla"
    }
}

ALGORITHMS = ["ppo", "sac", "trpo"]

def create_script(env_name, config_type, algo):
    """Create a launcher script for env_name, config_type, algo"""
    config_base = ENVS[env_name][config_type]
    
    script_content = f"""#!/bin/bash
# {env_name.title()} {config_type.title()} - {algo.upper()}"""
    
    if config_type == "vanilla":
        # Vanilla: only baseline
        script_content += f""" (Baseline only)

echo "Running {env_name.title()} Vanilla ({algo.upper()}): Baseline only"

conda run -n rl_hf_course python scripts/train.py \\
  --config configs/{algo.upper()}/vanilla/{config_base}_{algo}.yaml \\
  > logs/{env_name}_vanilla_{algo}.log 2>&1 &

echo "✅ Started vanilla baseline"
echo "Monitor: tail -f logs/{env_name}_vanilla_{algo}.log"
"""
    else:
        # Moderate/Extreme/Multi: baseline + nsmdmpi
        script_content += f""" (Baseline + NS-MDMPI)

echo "Running {env_name.title()} {config_type.title()} ({algo.upper()}): Baseline + NS-MDMPI"

# Baseline
conda run -n rl_hf_course python scripts/train.py \\
  --config configs/{algo.upper()}/{config_type}/{config_base}_baseline_{algo}.yaml \\
  > logs/{env_name}_{config_type}_baseline_{algo}.log 2>&1 &

sleep 2

# NS-MDMPI
conda run -n rl_hf_course python scripts/train.py \\
  --config configs/{algo.upper()}/{config_type}/{config_base}_nsmdmpi_{algo}.yaml \\
  > logs/{env_name}_{config_type}_nsmdmpi_{algo}.log 2>&1 &

echo "✅ Started 2 runs: baseline + nsmdmpi"
echo "Monitor: tail -f logs/{env_name}_{config_type}_*.log"
"""
    
    # Write script
    script_path = Path(f"scripts/run_{env_name}_{config_type}_{algo}.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path

def create_master_script():
    """Create master launcher script"""
    script_content = """#!/bin/bash
# Master Baseline Script - Run all vanilla baselines

echo "================================================================"
echo "  MASTER BASELINES - All Vanilla Experiments"
echo "================================================================"
echo ""
echo "Running:"
echo "  - Hopper Vanilla (PPO, SAC, TRPO)"
echo "  - HalfCheetah Vanilla (PPO, SAC, TRPO)"
echo "  - LunarLander Vanilla (PPO, SAC, TRPO)"
echo ""
echo "Total: 9 vanilla baselines"
echo "================================================================"
echo ""

"""
    
    for env in ["hopper", "halfcheetah", "lunarlander"]:
        for algo in ["ppo", "sac", "trpo"]:
            script_content += f"""echo "[{env.title()} {algo.upper()}] Starting..."
bash scripts/run_{env}_vanilla_{algo}.sh
sleep 2

"""
    
    script_content += """echo ""
echo "================================================================"
echo "✅ ALL 9 VANILLA BASELINES LAUNCHED!"
echo "================================================================"
echo ""
echo "Monitor: ps aux | grep train.py | wc -l"
echo "WandB: Project 'att_3_Vanilla_Baselines'"
"""
    
    script_path = Path("scripts/run_master_baselines.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    return script_path

def create_test_script():
    """Create test script with small timesteps"""
    script_content = """#!/bin/bash
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
sed 's/total_timesteps: 1000000/total_timesteps: 10000/' configs/PPO/moderate/hopper_friction_sine_baseline_ppo.yaml | \\
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
sed 's/total_timesteps: 1000000/total_timesteps: 10000/' configs/PPO/moderate/hopper_friction_sine_nsmdmpi_ppo.yaml | \\
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
sed 's/total_timesteps: 1000000/total_timesteps: 10000/' configs/PPO/extreme/hopper_friction_random_walk_baseline_ppo.yaml | \\
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
sed 's/total_timesteps: 1000000/total_timesteps: 10000/' configs/PPO/multi/hopper_friction_mass_scale_baseline_ppo.yaml | \\
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
sed 's/total_timesteps: 1000000/total_timesteps: 10000/' configs/PPO/vanilla/hopper_vanilla_ppo.yaml | \\
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
sed 's/total_timesteps: 500000/total_timesteps: 10000/' configs/PPO/moderate/lunarlander_gravity_sine_baseline_ppo.yaml | \\
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
"""
    
    script_path = Path("scripts/test.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    return script_path

def main():
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    count = 0
    
    # Generate env-specific scripts
    for env_name in ENVS.keys():
        for config_type in ["vanilla", "moderate", "extreme", "multi"]:
            for algo in ALGORITHMS:
                script_path = create_script(env_name, config_type, algo)
                print(f"✅ Created {script_path}")
                count += 1
    
    # Generate master script
    master_path = create_master_script()
    print(f"✅ Created {master_path}")
    count += 1
    
    # Generate test script
    test_path = create_test_script()
    print(f"✅ Created {test_path}")
    count += 1
    
    print(f"\n🎉 Total scripts created: {count}")
    print(f"   - Env-specific: {count - 2}")
    print(f"   - Master: 1")
    print(f"   - Test: 1")

if __name__ == "__main__":
    main()
