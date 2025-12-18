#!/bin/bash
# =============================================================================
# HalfCheetah (MuJoCo): Baseline vs Adaptive Experiments
# =============================================================================
#
# Runs systematic comparison between:
#   - Baseline (fixed hyperparameters)
#   - Adaptive (drift-aware hyperparameter scheduling)
#
# HalfCheetah-specific drift experiments:
#   - Friction changes (terrain/surface conditions)
#   - Damping changes (mechanical wear)
#   - Mass scale changes (weight variation)
#
# Includes video rendering after each experiment
#
# Usage:
#   ./scripts/run_halfcheetah_experiments.sh           # Run all experiments
#   ./scripts/run_halfcheetah_experiments.sh --quick   # Quick test (fewer steps)
#
# Requires: pip install mujoco gymnasium[mujoco]
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Settings
QUICK_MODE=false
TIMESTEPS=1000000
SEED=42
ALGO="SAC"  # SAC recommended for continuous control
VIDEO_EPISODES=1
VIDEO_LENGTH=1000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            TIMESTEPS=200000
            shift
            ;;
        --steps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --algo)
            ALGO="$2"
            shift 2
            ;;
        --no-render)
            VIDEO_EPISODES=0
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --quick       Quick mode with 200k steps"
            echo "  --steps N     Set timesteps"
            echo "  --seed N      Set random seed"
            echo "  --algo NAME   Algorithm (SAC, PPO)"
            echo "  --no-render   Skip video rendering"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/halfcheetah_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  HalfCheetah: Baseline vs Adaptive ${ALGO}${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "Timesteps:    ${GREEN}${TIMESTEPS}${NC}"
echo -e "Seed:         ${GREEN}${SEED}${NC}"
echo -e "Algorithm:    ${GREEN}${ALGO}${NC}"
echo -e "Results:      ${GREEN}${RESULTS_DIR}${NC}"
echo ""

# Function to create temp config
create_config() {
    local param=$1
    local drift=$2
    local magnitude=$3
    local period=$4
    local adaptive=$5
    local output=$6
    local sigma=${7:-0.01}
    local bounds_min=${8:-0.1}
    local bounds_max=${9:-0.8}
    
    cat > "$output" << EOF
env_id: "HalfCheetah-v4"

env:
  parameter: "$param"
  drift_type: "$drift"
  magnitude: $magnitude
  period: $period
  sigma: $sigma
  bounds: [$bounds_min, $bounds_max]

wandb:
  project: "HalfCheetah_Drift_Research"
  tags: ["mujoco", "halfcheetah", "$drift", "$param", "$([ "$adaptive" = "true" ] && echo "adaptive" || echo "baseline")"]
  mode: "online"

train:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 256
  gamma: 0.99
  total_timesteps: $TIMESTEPS
  seed: $SEED
  buffer_size: 100000
  learning_starts: 1000
  tau: 0.005

adaptive:
  enabled: $adaptive
  scale_factor: 0.1
  min_lr_multiplier: 0.5
  max_lr_multiplier: 3.0
  adapt_clip_range: true
  base_clip_range: 0.2
  min_clip_range: 0.05
  max_clip_range: 0.4
  adapt_entropy: true
  base_ent_coef: 0.0
  min_ent_coef: 0.0
  max_ent_coef: 0.1
  adapt_target_kl: true
  base_target_kl: 0.01
  min_target_kl: 0.001
  max_target_kl: 0.05
  log_freq: 500

paths:
  log_dir: "logs/"
  model_dir: "models/"
  video_dir: "videos/"
EOF
}

# Function to run experiment with rendering
run_experiment() {
    local param=$1
    local drift=$2
    local magnitude=$3
    local period=$4
    local adaptive=$5
    local sigma=${6:-0.01}
    local bounds_min=${7:-0.1}
    local bounds_max=${8:-0.8}
    
    local mode_name=$([ "$adaptive" = "true" ] && echo "Adaptive" || echo "Baseline")
    local exp_name="HalfCheetah_${param}_${drift}_${mode_name}_${TIMESTAMP}"
    local temp_config="${RESULTS_DIR}/${exp_name}_config.yaml"
    
    echo -e "${YELLOW}>>> Running: ${exp_name}${NC}"
    echo -e "    Parameter: $param, Drift: $drift, Magnitude: $magnitude, Period: $period"
    
    create_config "$param" "$drift" "$magnitude" "$period" "$adaptive" "$temp_config" "$sigma" "$bounds_min" "$bounds_max"
    
    # Train
    python scripts/train.py \
        --config "$temp_config" \
        --algo "$ALGO" \
        --exp_name "$exp_name" \
        2>&1 | tee "${RESULTS_DIR}/${exp_name}.log"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training completed: ${exp_name}${NC}"
        
        # Render video if enabled
        if [ "$VIDEO_EPISODES" -gt 0 ]; then
            local model_path="models/${exp_name}.zip"
            if [ -f "$model_path" ]; then
                echo -e "${YELLOW}>>> Rendering video...${NC}"
                python scripts/render.py \
                    --model "$model_path" \
                    --config "$temp_config" \
                    --algo "$ALGO" \
                    --output "videos/" \
                    --episodes "$VIDEO_EPISODES" \
                    --length "$VIDEO_LENGTH" \
                    2>&1 | tee -a "${RESULTS_DIR}/${exp_name}.log"
                echo -e "${GREEN}✓ Video saved${NC}"
            else
                echo -e "${RED}✗ Model not found for rendering${NC}"
            fi
        fi
    else
        echo -e "${RED}✗ Failed: ${exp_name}${NC}"
    fi
    echo ""
}

# =============================================================================
# EXPERIMENT 1: Friction Jump - Sudden terrain change
# =============================================================================
echo -e "${BLUE}--- Experiment 1: Friction Jump (Slippery Ground) ---${NC}"
echo "Simulates sudden surface change (dry → icy)"
echo "Friction drops from 0.4 to 0.15 (slippery)"
run_experiment "friction" "jump" "-0.25" "500000" "false" "0.01" "0.1" "0.6"
run_experiment "friction" "jump" "-0.25" "500000" "true" "0.01" "0.1" "0.6"

# =============================================================================
# EXPERIMENT 2: Friction Sine - Variable terrain
# =============================================================================
echo -e "${BLUE}--- Experiment 2: Friction Sine (Variable Terrain) ---${NC}"
echo "Simulates periodic surface changes (mud/dry cycles)"
echo "Friction oscillates: 0.4 ± 0.15"
run_experiment "friction" "sine" "0.15" "200000" "false" "0.01" "0.2" "0.6"
run_experiment "friction" "sine" "0.15" "200000" "true" "0.01" "0.2" "0.6"

# =============================================================================
# EXPERIMENT 3: Damping Linear - Mechanical wear
# =============================================================================
echo -e "${BLUE}--- Experiment 3: Damping Linear (Mechanical Wear) ---${NC}"
echo "Simulates joints wearing out over time"
echo "Damping increases from 1.0 to 1.5 (50% stiffer)"
run_experiment "damping" "linear" "0.5" "400000" "false" "0.02" "0.5" "2.0"
run_experiment "damping" "linear" "0.5" "400000" "true" "0.02" "0.5" "2.0"

# =============================================================================
# EXPERIMENT 4: Mass Scale Linear - Weight change
# =============================================================================
echo -e "${BLUE}--- Experiment 4: Mass Scale Linear (Weight Gain) ---${NC}"
echo "Simulates robot carrying increasing load"
echo "Mass increases by 30%"
run_experiment "mass_scale" "linear" "0.3" "300000" "false" "0.02" "0.8" "1.5"
run_experiment "mass_scale" "linear" "0.3" "300000" "true" "0.02" "0.8" "1.5"

# =============================================================================
# EXPERIMENT 5: Friction Random Walk - Unpredictable terrain
# =============================================================================
echo -e "${BLUE}--- Experiment 5: Friction Random Walk (Unpredictable) ---${NC}"
echo "Simulates unpredictable terrain variations"
run_experiment "friction" "random_walk" "0.0" "150000" "false" "0.01" "0.15" "0.6"
run_experiment "friction" "random_walk" "0.0" "150000" "true" "0.01" "0.15" "0.6"

# =============================================================================
# EXPERIMENT 6: Static Baseline (No Drift) - Control
# =============================================================================
echo -e "${BLUE}--- Experiment 6: Static (No Drift) ---${NC}"
echo "Control experiment with standard HalfCheetah"
run_experiment "friction" "static" "0.0" "200000" "false"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}       All HalfCheetah Experiments Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "Videos saved to:  videos/"
echo "View in WandB:    https://wandb.ai"
echo ""
echo "Experiments run:"
echo "  1. Friction Jump:       Baseline vs Adaptive (slippery ground)"
echo "  2. Friction Sine:       Baseline vs Adaptive (variable terrain)"
echo "  3. Damping Linear:      Baseline vs Adaptive (mechanical wear)"
echo "  4. Mass Scale Linear:   Baseline vs Adaptive (weight gain)"
echo "  5. Friction Random:     Baseline vs Adaptive (unpredictable)"
echo "  6. Static (control):    Baseline only"
echo ""
echo "Filter WandB by timestamp: $TIMESTAMP"
