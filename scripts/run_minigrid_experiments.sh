#!/bin/bash
# =============================================================================
# MiniGrid: Baseline vs Adaptive Experiments
# =============================================================================
#
# Runs systematic comparison between:
#   - Baseline (fixed hyperparameters)
#   - Adaptive (drift-aware hyperparameter scheduling)
#
# MiniGrid-specific drift experiments:
#   - Reward scale changes (goal value variation)
#   - Step penalty changes (urgency variation)
#
# Includes video rendering after each experiment
#
# Usage:
#   ./scripts/run_minigrid_experiments.sh           # Run all experiments
#   ./scripts/run_minigrid_experiments.sh --quick   # Quick test (fewer steps)
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
RESULTS_DIR="results/minigrid_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  MiniGrid: Baseline vs Adaptive ${ALGO}${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "Timesteps:    ${GREEN}${TIMESTEPS}${NC}"
echo -e "Seed:         ${GREEN}${SEED}${NC}"
echo -e "Algorithm:    ${GREEN}${ALGO}${NC}"
echo -e "Results:      ${GREEN}${RESULTS_DIR}${NC}"
echo ""

# Function to create temp config
create_config() {
    local env_id=$1
    local param=$2
    local drift=$3
    local magnitude=$4
    local period=$5
    local adaptive=$6
    local output=$7
    
    cat > "$output" << EOF
env_id: "$env_id"

env:
  parameter: "$param"
  drift_type: "$drift"
  magnitude: $magnitude
  period: $period
  sigma: 0.05
  bounds: [0.1, 2.0]

wandb:
  project: "MiniGrid_Drift_Research"
  tags: ["minigrid", "$drift", "$param", "$([ "$adaptive" = "true" ] && echo "adaptive" || echo "baseline")"]
  mode: "online"

train:
  learning_rate: 0.0003
  n_steps: 512
  batch_size: 64
  gamma: 0.99
  total_timesteps: $TIMESTEPS
  seed: $SEED

adaptive:
  enabled: $adaptive
  scale_factor: 0.15
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
    local env_id=$1
    local param=$2
    local drift=$3
    local magnitude=$4
    local period=$5
    local adaptive=$6
    
    local mode_name=$([ "$adaptive" = "true" ] && echo "Adaptive" || echo "Baseline")
    # Extract short env name
    local env_short=$(echo "$env_id" | sed 's/MiniGrid-//g' | sed 's/-v0//g')
    local exp_name="MiniGrid_${env_short}_${param}_${drift}_${mode_name}_${TIMESTAMP}"
    local temp_config="${RESULTS_DIR}/${exp_name}_config.yaml"
    
    echo -e "${YELLOW}>>> Running: ${exp_name}${NC}"
    echo -e "    Env: $env_id, Parameter: $param, Drift: $drift, Magnitude: $magnitude"
    
    create_config "$env_id" "$param" "$drift" "$magnitude" "$period" "$adaptive" "$temp_config"
    
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
# EXPERIMENT 1: Reward Scale Jump (Empty-8x8) - Sudden reward change
# =============================================================================
echo -e "${BLUE}--- Experiment 1: Reward Scale Jump (Empty-8x8) ---${NC}"
echo "Simulates sudden goal value change"
echo "Reward scale jumps from 1.0 to 0.5 (halved)"
run_experiment "MiniGrid-Empty-8x8-v0" "reward_scale" "jump" "-0.5" "50000" "false"
run_experiment "MiniGrid-Empty-8x8-v0" "reward_scale" "jump" "-0.5" "50000" "true"

# =============================================================================
# EXPERIMENT 2: Reward Scale Sine (Empty-8x8) - Oscillating rewards
# =============================================================================
echo -e "${BLUE}--- Experiment 2: Reward Scale Sine (Empty-8x8) ---${NC}"
echo "Simulates periodic reward value changes"
echo "Reward scale oscillates: 1.0 ± 0.3"
run_experiment "MiniGrid-Empty-8x8-v0" "reward_scale" "sine" "0.3" "20000" "false"
run_experiment "MiniGrid-Empty-8x8-v0" "reward_scale" "sine" "0.3" "20000" "true"

# =============================================================================
# EXPERIMENT 3: Reward Scale Linear (Empty-5x5) - Gradual reward change
# =============================================================================
echo -e "${BLUE}--- Experiment 3: Reward Scale Linear (Empty-5x5) ---${NC}"
echo "Simulates gradually increasing reward (easier task)"
run_experiment "MiniGrid-Empty-5x5-v0" "reward_scale" "linear" "0.5" "25000" "false"
run_experiment "MiniGrid-Empty-5x5-v0" "reward_scale" "linear" "0.5" "25000" "true"

# =============================================================================
# EXPERIMENT 4: Reward Random Walk (Empty-8x8) - Unpredictable rewards
# =============================================================================
echo -e "${BLUE}--- Experiment 4: Reward Random Walk (Empty-8x8) ---${NC}"
echo "Simulates unpredictable reward value changes"
run_experiment "MiniGrid-Empty-8x8-v0" "reward_scale" "random_walk" "0.0" "15000" "false"
run_experiment "MiniGrid-Empty-8x8-v0" "reward_scale" "random_walk" "0.0" "15000" "true"

# =============================================================================
# EXPERIMENT 5: FourRooms with Reward Drift - Complex navigation
# =============================================================================
echo -e "${BLUE}--- Experiment 5: FourRooms with Reward Drift ---${NC}"
echo "More complex environment with drifting rewards"
run_experiment "MiniGrid-FourRooms-v0" "reward_scale" "sine" "0.3" "25000" "false"
run_experiment "MiniGrid-FourRooms-v0" "reward_scale" "sine" "0.3" "25000" "true"

# =============================================================================
# EXPERIMENT 6: Static Baseline (No Drift) - Control
# =============================================================================
echo -e "${BLUE}--- Experiment 6: Static (No Drift) ---${NC}"
echo "Control experiment with standard MiniGrid"
run_experiment "MiniGrid-Empty-8x8-v0" "reward_scale" "static" "0.0" "10000" "false"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}       All MiniGrid Experiments Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "Videos saved to:  videos/"
echo "View in WandB:    https://wandb.ai"
echo ""
echo "Experiments run:"
echo "  1. Reward Jump (8x8):     Baseline vs Adaptive (sudden)"
echo "  2. Reward Sine (8x8):     Baseline vs Adaptive (periodic)"
echo "  3. Reward Linear (5x5):   Baseline vs Adaptive (gradual)"
echo "  4. Reward Random (8x8):   Baseline vs Adaptive (stochastic)"
echo "  5. FourRooms Sine:        Baseline vs Adaptive (complex)"
echo "  6. Static (control):      Baseline only"
echo ""
echo "Filter WandB by timestamp: $TIMESTAMP"
