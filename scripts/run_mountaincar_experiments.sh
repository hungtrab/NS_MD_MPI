#!/bin/bash
# =============================================================================
# MountainCar: Baseline vs Adaptive Experiments
# =============================================================================
#
# Runs systematic comparison between:
#   - Baseline (fixed hyperparameters)
#   - Adaptive (drift-aware hyperparameter scheduling)
#
# MountainCar-specific drift experiments:
#   - Gravity changes (harder/easier to climb)
#   - Force changes (engine power variation)
#   - Goal position changes (moving target)
#
# Includes video rendering after each experiment
#
# Usage:
#   ./scripts/run_mountaincar_experiments.sh           # Run all experiments
#   ./scripts/run_mountaincar_experiments.sh --quick   # Quick test (fewer steps)
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
TIMESTEPS=500000
SEED=42
ALGO="PPO"
VIDEO_EPISODES=1
VIDEO_LENGTH=300

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            TIMESTEPS=100000
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
            echo "  --quick       Quick mode with 100k steps"
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
RESULTS_DIR="results/mountaincar_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  MountainCar: Baseline vs Adaptive ${ALGO}${NC}"
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
    local sigma=${7:-0.0001}
    local bounds_min=${8:-0.001}
    local bounds_max=${9:-0.005}
    
    cat > "$output" << EOF
env_id: "MountainCar-v0"

env:
  parameter: "$param"
  drift_type: "$drift"
  magnitude: $magnitude
  period: $period
  sigma: $sigma
  bounds: [$bounds_min, $bounds_max]

wandb:
  project: "MountainCar_Drift_Research"
  tags: ["mountaincar", "$drift", "$param", "$([ "$adaptive" = "true" ] && echo "adaptive" || echo "baseline")"]
  mode: "online"

train:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99
  total_timesteps: $TIMESTEPS
  seed: $SEED

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
  base_ent_coef: 0.01
  min_ent_coef: 0.0
  max_ent_coef: 0.1
  adapt_target_kl: true
  base_target_kl: 0.01
  min_target_kl: 0.001
  max_target_kl: 0.05
  log_freq: 100

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
    local sigma=${6:-0.0001}
    local bounds_min=${7:-0.001}
    local bounds_max=${8:-0.005}
    
    local mode_name=$([ "$adaptive" = "true" ] && echo "Adaptive" || echo "Baseline")
    local exp_name="MountainCar_${param}_${drift}_${mode_name}_${TIMESTAMP}"
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
# EXPERIMENT 1: Gravity Drift (Jump) - Sudden gravity increase
# =============================================================================
echo -e "${BLUE}--- Experiment 1: Gravity Jump Drift ---${NC}"
echo "Simulates sudden gravity change (harder to climb the hill)"
echo "Base gravity=0.0025, jumps to 0.0035 (40% increase)"
run_experiment "gravity" "jump" "0.001" "250000" "false"
run_experiment "gravity" "jump" "0.001" "250000" "true"

# =============================================================================
# EXPERIMENT 2: Gravity Drift (Sine) - Oscillating difficulty
# =============================================================================
echo -e "${BLUE}--- Experiment 2: Gravity Sine Drift ---${NC}"
echo "Simulates periodic gravity changes (varying hill steepness)"
echo "Gravity oscillates: 0.0025 ± 0.0005"
run_experiment "gravity" "sine" "0.0005" "100000" "false"
run_experiment "gravity" "sine" "0.0005" "100000" "true"

# =============================================================================
# EXPERIMENT 3: Force Drift (Linear) - Engine power degradation
# =============================================================================
echo -e "${BLUE}--- Experiment 3: Force Linear Drift ---${NC}"
echo "Simulates engine power degradation/improvement over time"
echo "Force changes from 0.001 to 0.0015 (50% increase)"
run_experiment "force" "linear" "0.0005" "200000" "false" "0.00005" "0.0005" "0.002"
run_experiment "force" "linear" "0.0005" "200000" "true" "0.00005" "0.0005" "0.002"

# =============================================================================
# EXPERIMENT 4: Gravity Random Walk - Unpredictable terrain
# =============================================================================
echo -e "${BLUE}--- Experiment 4: Gravity Random Walk ---${NC}"
echo "Simulates unpredictable terrain changes"
run_experiment "gravity" "random_walk" "0.0" "50000" "false" "0.0001" "0.001" "0.004"
run_experiment "gravity" "random_walk" "0.0" "50000" "true" "0.0001" "0.001" "0.004"

# =============================================================================
# EXPERIMENT 5: Static Baseline (No Drift) - Control
# =============================================================================
echo -e "${BLUE}--- Experiment 5: Static (No Drift) ---${NC}"
echo "Control experiment with standard MountainCar"
run_experiment "gravity" "static" "0.0" "100000" "false"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}       All MountainCar Experiments Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "Videos saved to:  videos/"
echo "View in WandB:    https://wandb.ai"
echo ""
echo "Experiments run:"
echo "  1. Gravity Jump:      Baseline vs Adaptive (sudden change)"
echo "  2. Gravity Sine:      Baseline vs Adaptive (periodic)"
echo "  3. Force Linear:      Baseline vs Adaptive (engine power)"
echo "  4. Gravity Random:    Baseline vs Adaptive (stochastic)"
echo "  5. Static (control):  Baseline only"
echo ""
echo "Filter WandB by timestamp: $TIMESTAMP"
