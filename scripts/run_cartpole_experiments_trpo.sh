#!/bin/bash
# =============================================================================
# CartPole: Baseline vs Adaptive PPO Experiments
# =============================================================================
#
# Runs systematic comparison between:
#   - Baseline PPO (fixed hyperparameters)
#   - Adaptive PPO (drift-aware hyperparameter scheduling)
#
# Across different drift patterns and parameters
#
# Usage:
#   ./scripts/run_cartpole_experiments.sh           # Run all experiments
#   ./scripts/run_cartpole_experiments.sh --quick   # Quick test (fewer steps)
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

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            TIMESTEPS=20000
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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --quick       Quick mode with 20k steps"
            echo "  --steps N     Set timesteps"
            echo "  --seed N      Set random seed"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG="configs/cartpole_adaptive.yaml"
RESULTS_DIR="results/trpo_cartpole_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CartPole: Baseline vs Adaptive PPO${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Timesteps:    ${GREEN}${TIMESTEPS}${NC}"
echo -e "Seed:         ${GREEN}${SEED}${NC}"
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
    
    cat > "$output" << EOF
env_id: "MountainCar-v0"

env:
  parameter: "$param"
  drift_type: "$drift"
  magnitude: $magnitude
  period: $period
  sigma: 0.1
  bounds: [0.0, 25.0]

wandb:
  project: "MountainCar_Drift_Research_TRPO"
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
  base_ent_coef: 0.0
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

# Function to run experiment
run_experiment() {
    local param=$1
    local drift=$2
    local magnitude=$3
    local period=$4
    local adaptive=$5
    
    local mode_name=$([ "$adaptive" = "true" ] && echo "Adaptive" || echo "Baseline")
    local exp_name="MountainCar_${param}_${drift}_${mode_name}_${TIMESTAMP}"
    local temp_config="${RESULTS_DIR}/${exp_name}_config.yaml"
    
    echo -e "${YELLOW}>>> Running: ${exp_name}${NC}"
    echo -e "    Parameter: $param, Drift: $drift, Magnitude: $magnitude, Period: $period"
    
    create_config "$param" "$drift" "$magnitude" "$period" "$adaptive" "$temp_config"
    
    python scripts/train.py \
        --config "$temp_config" \
        --algo TRPO \
        --exp_name "$exp_name" \
        2>&1 | tee "${RESULTS_DIR}/${exp_name}.log"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Completed: ${exp_name}${NC}"
    else
        echo -e "${RED}✗ Failed: ${exp_name}${NC}"
    fi
    echo ""
}

# =============================================================================
# EXPERIMENT 1: Gravity Drift (Jump)
# =============================================================================
echo -e "${BLUE}--- Experiment 1: Gravity Jump Drift ---${NC}"
run_experiment "gravity" "jump" "10.0" "50000" "false"   # Baseline
run_experiment "gravity" "jump" "10.0" "50000" "true"    # Adaptive

# =============================================================================
# EXPERIMENT 2: Gravity Drift (Sine)
# =============================================================================
echo -e "${BLUE}--- Experiment 2: Gravity Sine Drift ---${NC}"
run_experiment "gravity" "sine" "5.0" "20000" "false"    # Baseline
run_experiment "gravity" "sine" "5.0" "20000" "true"     # Adaptive

# =============================================================================
# EXPERIMENT 3: Mass Cart Drift (Random Walk)
# =============================================================================
echo -e "${BLUE}--- Experiment 3: Mass Cart Random Walk ---${NC}"
run_experiment "masscart" "random_walk" "0.5" "10000" "false"  # Baseline
run_experiment "masscart" "random_walk" "0.5" "10000" "true"   # Adaptive

# =============================================================================
# EXPERIMENT 4: Pole Length Drift (Linear)
# =============================================================================
echo -e "${BLUE}--- Experiment 4: Pole Length Linear Drift ---${NC}"
run_experiment "length" "linear" "0.3" "25000" "false"   # Baseline
run_experiment "length" "linear" "0.3" "25000" "true"    # Adaptive

# =============================================================================
# EXPERIMENT 5: Static Baseline (No Drift)
# =============================================================================
echo -e "${BLUE}--- Experiment 5: Static (No Drift) ---${NC}"
run_experiment "gravity" "static" "0.0" "10000" "false"  # Baseline only

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}       All Experiments Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "View in WandB: https://wandb.ai"
echo ""
echo "Experiments run:"
echo "  1. Gravity Jump:        Baseline vs Adaptive"
echo "  2. Gravity Sine:        Baseline vs Adaptive"
echo "  3. MassCart Random:     Baseline vs Adaptive"
echo "  4. Pole Length Linear:  Baseline vs Adaptive"
echo "  5. Static (control):    Baseline only"
echo ""
echo "Filter WandB by timestamp: $TIMESTAMP"
