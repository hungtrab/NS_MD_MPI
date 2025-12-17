#!/bin/bash
# =============================================================================
# QUICK COMPARISON: Single Environment Adaptive vs Baseline
# =============================================================================
#
# Runs a quick comparison between adaptive and baseline on a single
# environment for rapid testing.
#
# Usage:
#   ./scripts/compare_single.sh                           # Default: CartPole + PPO
#   ./scripts/compare_single.sh --env mountaincar         # MountainCar
#   ./scripts/compare_single.sh --algo SAC                # Use SAC algorithm
#   ./scripts/compare_single.sh --drift jump              # Jump drift pattern
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
ENV="cartpole"
ALGO="PPO"
DRIFT="sine"
SEED=42
TIMESTEPS=50000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --algo)
            ALGO="$2"
            shift 2
            ;;
        --drift)
            DRIFT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --steps)
            TIMESTEPS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env ENV      Environment (cartpole, mountaincar, frozenlake, halfcheetah)"
            echo "  --algo ALGO    Algorithm (PPO, SAC, TRPO)"
            echo "  --drift TYPE   Drift type (static, jump, linear, sine, random_walk)"
            echo "  --seed SEED    Random seed"
            echo "  --steps N      Total timesteps"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG_FILE="configs/${ENV}_adaptive.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Quick Adaptive vs Baseline Compare${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Environment:  ${GREEN}${ENV}${NC}"
echo -e "Algorithm:    ${GREEN}${ALGO}${NC}"
echo -e "Drift Type:   ${GREEN}${DRIFT}${NC}"
echo -e "Seed:         ${GREEN}${SEED}${NC}"
echo -e "Timesteps:    ${GREEN}${TIMESTEPS}${NC}"
echo ""

# Check config exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -1 configs/*.yaml
    exit 1
fi

# Create temp configs
TEMP_DIR=$(mktemp -d)
ADAPTIVE_CONFIG="${TEMP_DIR}/adaptive.yaml"
BASELINE_CONFIG="${TEMP_DIR}/baseline.yaml"

# Copy and modify configs
cp "$CONFIG_FILE" "$ADAPTIVE_CONFIG"
cp "$CONFIG_FILE" "$BASELINE_CONFIG"

# Set drift type and seed
sed -i "s/drift_type: .*/drift_type: \"$DRIFT\"/" "$ADAPTIVE_CONFIG"
sed -i "s/drift_type: .*/drift_type: \"$DRIFT\"/" "$BASELINE_CONFIG"
sed -i "s/seed: .*/seed: $SEED/" "$ADAPTIVE_CONFIG"
sed -i "s/seed: .*/seed: $SEED/" "$BASELINE_CONFIG"
sed -i "s/total_timesteps: .*/total_timesteps: $TIMESTEPS/" "$ADAPTIVE_CONFIG"
sed -i "s/total_timesteps: .*/total_timesteps: $TIMESTEPS/" "$BASELINE_CONFIG"

# Set adaptive mode
sed -i "s/enabled: false/enabled: true/" "$ADAPTIVE_CONFIG"
sed -i "s/enabled:  *false/enabled: true/" "$ADAPTIVE_CONFIG"
sed -i "s/enabled: true/enabled: false/" "$BASELINE_CONFIG"
sed -i "s/enabled:  *true/enabled: false/" "$BASELINE_CONFIG"

# Run Adaptive
echo -e "${YELLOW}>>> [1/2] Training Adaptive (drift-aware LR)...${NC}"
ADAPTIVE_NAME="${ENV}_${ALGO}_${DRIFT}_Adaptive_${TIMESTAMP}"
python scripts/train.py --config "$ADAPTIVE_CONFIG" --algo "$ALGO" --exp_name "$ADAPTIVE_NAME"

echo ""

# Run Baseline
echo -e "${YELLOW}>>> [2/2] Training Baseline (fixed LR)...${NC}"
BASELINE_NAME="${ENV}_${ALGO}_${DRIFT}_Baseline_${TIMESTAMP}"
python scripts/train.py --config "$BASELINE_CONFIG" --algo "$ALGO" --exp_name "$BASELINE_NAME"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}           Comparison Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "View results in WandB or compare:"
echo "  - Adaptive: models/${ADAPTIVE_NAME}"
echo "  - Baseline: models/${BASELINE_NAME}"
echo ""
echo "To compare in WandB, filter by runs ending in ${TIMESTAMP}"
