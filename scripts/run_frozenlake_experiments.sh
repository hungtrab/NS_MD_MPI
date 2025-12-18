#!/bin/bash
# =============================================================================
# FrozenLake: Baseline vs Adaptive Experiments
# =============================================================================
#
# Runs systematic comparison between:
#   - Baseline (fixed hyperparameters)
#   - Adaptive (drift-aware hyperparameter scheduling)
#
# FrozenLake-specific drift experiments:
#   - Slip probability changes (ice conditions)
#   - Reward scale changes (goal value)
#
# Includes video rendering after each experiment
#
# Usage:
#   ./scripts/run_frozenlake_experiments.sh           # Run all experiments
#   ./scripts/run_frozenlake_experiments.sh --quick   # Quick test (fewer steps)
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
TIMESTEPS=100000
SEED=42
ALGO="PPO"
VIDEO_EPISODES=1
VIDEO_LENGTH=200

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            TIMESTEPS=30000
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
            echo "  --quick       Quick mode with 30k steps"
            echo "  --steps N     Set timesteps"
            echo "  --seed N      Set random seed"
            echo "  --algo NAME   Algorithm (PPO, TRPO)"
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
RESULTS_DIR="results/frozenlake_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  FrozenLake: Baseline vs Adaptive ${ALGO}${NC}"
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
    local map_name=${7:-"4x4"}
    
    cat > "$output" << EOF
env_id: "FrozenLake-v1"

env:
  parameter: "$param"
  drift_type: "$drift"
  magnitude: $magnitude
  period: $period
  sigma: 0.02
  bounds: [0.1, 0.9]

wandb:
  project: "FrozenLake_Drift_Research"
  tags: ["frozenlake", "$drift", "$param", "$([ "$adaptive" = "true" ] && echo "adaptive" || echo "baseline")", "$map_name"]
  mode: "online"

train:
  learning_rate: 0.001
  n_steps: 128
  batch_size: 32
  gamma: 0.99
  total_timesteps: $TIMESTEPS
  seed: $SEED

adaptive:
  enabled: $adaptive
  scale_factor: 0.2
  min_lr_multiplier: 0.5
  max_lr_multiplier: 3.0
  adapt_clip_range: true
  base_clip_range: 0.2
  min_clip_range: 0.05
  max_clip_range: 0.4
  adapt_entropy: true
  base_ent_coef: 0.01
  min_ent_coef: 0.0
  max_ent_coef: 0.1
  adapt_target_kl: true
  base_target_kl: 0.01
  min_target_kl: 0.001
  max_target_kl: 0.05
  log_freq: 50

paths:
  log_dir: "logs/"
  model_dir: "models/"
  video_dir: "videos/"

env_kwargs:
  is_slippery: true
  map_name: "$map_name"
EOF
}

# Function to run experiment with rendering
run_experiment() {
    local param=$1
    local drift=$2
    local magnitude=$3
    local period=$4
    local adaptive=$5
    local map_name=${6:-"4x4"}
    
    local mode_name=$([ "$adaptive" = "true" ] && echo "Adaptive" || echo "Baseline")
    local exp_name="FrozenLake_${map_name}_${param}_${drift}_${mode_name}_${TIMESTAMP}"
    local temp_config="${RESULTS_DIR}/${exp_name}_config.yaml"
    
    echo -e "${YELLOW}>>> Running: ${exp_name}${NC}"
    echo -e "    Parameter: $param, Drift: $drift, Magnitude: $magnitude, Map: $map_name"
    
    create_config "$param" "$drift" "$magnitude" "$period" "$adaptive" "$temp_config" "$map_name"
    
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
# EXPERIMENT 1: Slip Probability Jump - Sudden ice change
# =============================================================================
echo -e "${BLUE}--- Experiment 1: Slip Probability Jump ---${NC}"
echo "Simulates sudden ice condition change (freeze/thaw)"
echo "Slip prob jumps from 0.67 to 0.87 (more slippery)"
run_experiment "slip_prob" "jump" "0.2" "50000" "false" "4x4"
run_experiment "slip_prob" "jump" "0.2" "50000" "true" "4x4"

# =============================================================================
# EXPERIMENT 2: Slip Probability Sine - Seasonal ice changes
# =============================================================================
echo -e "${BLUE}--- Experiment 2: Slip Probability Sine ---${NC}"
echo "Simulates seasonal ice changes (oscillating conditions)"
echo "Slip prob oscillates: 0.67 ± 0.15"
run_experiment "slip_prob" "sine" "0.15" "20000" "false" "4x4"
run_experiment "slip_prob" "sine" "0.15" "20000" "true" "4x4"

# =============================================================================
# EXPERIMENT 3: Slip Probability Linear - Gradual thaw
# =============================================================================
echo -e "${BLUE}--- Experiment 3: Slip Probability Linear ---${NC}"
echo "Simulates gradual ice melting (becoming more slippery)"
run_experiment "slip_prob" "linear" "0.2" "30000" "false" "4x4"
run_experiment "slip_prob" "linear" "0.2" "30000" "true" "4x4"

# =============================================================================
# EXPERIMENT 4: Slip Random Walk - Weather variability
# =============================================================================
echo -e "${BLUE}--- Experiment 4: Slip Probability Random Walk ---${NC}"
echo "Simulates unpredictable weather affecting ice"
run_experiment "slip_prob" "random_walk" "0.0" "10000" "false" "4x4"
run_experiment "slip_prob" "random_walk" "0.0" "10000" "true" "4x4"

# =============================================================================
# EXPERIMENT 5: 8x8 Map with Slip Drift - Harder environment
# =============================================================================
echo -e "${BLUE}--- Experiment 5: 8x8 Map with Slip Drift ---${NC}"
echo "Larger map with drifting slip probability"
run_experiment "slip_prob" "sine" "0.15" "30000" "false" "8x8"
run_experiment "slip_prob" "sine" "0.15" "30000" "true" "8x8"

# =============================================================================
# EXPERIMENT 6: Static Baseline (No Drift) - Control
# =============================================================================
echo -e "${BLUE}--- Experiment 6: Static (No Drift) ---${NC}"
echo "Control experiment with standard FrozenLake"
run_experiment "slip_prob" "static" "0.0" "10000" "false" "4x4"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}       All FrozenLake Experiments Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "Videos saved to:  videos/"
echo "View in WandB:    https://wandb.ai"
echo ""
echo "Experiments run:"
echo "  1. Slip Jump (4x4):     Baseline vs Adaptive (sudden ice)"
echo "  2. Slip Sine (4x4):     Baseline vs Adaptive (seasonal)"
echo "  3. Slip Linear (4x4):   Baseline vs Adaptive (melting)"
echo "  4. Slip Random (4x4):   Baseline vs Adaptive (weather)"
echo "  5. Slip Sine (8x8):     Baseline vs Adaptive (harder map)"
echo "  6. Static (control):    Baseline only"
echo ""
echo "Filter WandB by timestamp: $TIMESTAMP"
