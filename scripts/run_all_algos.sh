#!/bin/bash
# =============================================================================
# RUN ALL ALGORITHMS: Compare PPO vs SAC vs TRPO
# =============================================================================
#
# Runs the same experiment with all three algorithms for comparison.
#
# Usage:
#   ./scripts/run_all_algos.sh                    # Default settings
#   ./scripts/run_all_algos.sh --env mountaincar  # Different environment
#   ./scripts/run_all_algos.sh --adaptive         # Only adaptive mode
#   ./scripts/run_all_algos.sh --baseline         # Only baseline mode
#
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Defaults
ENV="cartpole"
DRIFT="sine"
ADAPTIVE="both"  # "adaptive", "baseline", or "both"
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --drift)
            DRIFT="$2"
            shift 2
            ;;
        --adaptive)
            ADAPTIVE="adaptive"
            shift
            ;;
        --baseline)
            ADAPTIVE="baseline"
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --env ENV       Environment name"
            echo "  --drift TYPE    Drift type"
            echo "  --adaptive      Run only adaptive mode"
            echo "  --baseline      Run only baseline mode"
            echo "  --seed N        Random seed"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

cd "$(dirname "$0")/.."

CONFIG="configs/${ENV}_adaptive.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Algorithm Comparison: PPO vs SAC vs TRPO${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

ALGORITHMS=("PPO" "SAC" "TRPO")

for ALGO in "${ALGORITHMS[@]}"; do
    echo -e "${YELLOW}>>> Testing ${ALGO}...${NC}"
    
    if [[ "$ADAPTIVE" == "both" || "$ADAPTIVE" == "adaptive" ]]; then
        # Adaptive run
        EXP_NAME="${ENV}_${ALGO}_${DRIFT}_Adaptive_${TIMESTAMP}"
        TEMP_CONFIG=$(mktemp)
        cp "$CONFIG" "$TEMP_CONFIG"
        sed -i "s/enabled: false/enabled: true/" "$TEMP_CONFIG"
        sed -i "s/enabled:  *false/enabled: true/" "$TEMP_CONFIG"
        sed -i "s/drift_type: .*/drift_type: \"$DRIFT\"/" "$TEMP_CONFIG"
        sed -i "s/seed: .*/seed: $SEED/" "$TEMP_CONFIG"
        
        echo -e "  ${GREEN}Running ${ALGO} Adaptive...${NC}"
        python scripts/train.py --config "$TEMP_CONFIG" --algo "$ALGO" --exp_name "$EXP_NAME" || {
            echo -e "  ${RED}${ALGO} Adaptive failed${NC}"
        }
        rm "$TEMP_CONFIG"
    fi
    
    if [[ "$ADAPTIVE" == "both" || "$ADAPTIVE" == "baseline" ]]; then
        # Baseline run
        EXP_NAME="${ENV}_${ALGO}_${DRIFT}_Baseline_${TIMESTAMP}"
        TEMP_CONFIG=$(mktemp)
        cp "$CONFIG" "$TEMP_CONFIG"
        sed -i "s/enabled: true/enabled: false/" "$TEMP_CONFIG"
        sed -i "s/enabled:  *true/enabled: false/" "$TEMP_CONFIG"
        sed -i "s/drift_type: .*/drift_type: \"$DRIFT\"/" "$TEMP_CONFIG"
        sed -i "s/seed: .*/seed: $SEED/" "$TEMP_CONFIG"
        
        echo -e "  ${GREEN}Running ${ALGO} Baseline...${NC}"
        python scripts/train.py --config "$TEMP_CONFIG" --algo "$ALGO" --exp_name "$EXP_NAME" || {
            echo -e "  ${RED}${ALGO} Baseline failed${NC}"
        }
        rm "$TEMP_CONFIG"
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}         All Algorithms Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Filter WandB runs by timestamp: $TIMESTAMP"
