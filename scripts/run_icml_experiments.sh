#!/bin/bash
# =============================================================================
# ICML EXPERIMENT RUNNER
# =============================================================================
#
# Master script to run experiments for "Regularized Non-Stationary MDPs".
#
# Campaigns:
#   1. main_comparison: NS-MD-MPI vs Adaptive vs Baseline
#   2. budget_scaling: Regret vs Drift Magnitude
#   3. generalization: PPO vs SAC
#   4. zero_drift: Static environment check
#
# Usage:
#   ./scripts/run_icml_experiments.sh --campaign main_comparison --env CartPole
#
#Params:
#   --campaign: Experiment campaign to run
#   --env: Environment name (CartPole, FrozenLake, etc.)
#   --seeds: Number of seeds (default: 5)
#   --dry-run: Print commands without executing
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Defaults
CAMPAIGN="main_comparison"
ENV="CartPole"
SEEDS=5
DRY_RUN=false
START_SEED=0
MAX_JOBS=8  # Updated default

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --campaign)
            CAMPAIGN="$2"
            shift 2
            ;;
        --env)
            ENV="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --campaign NAME  (main_comparison, budget_scaling, generalization, zero_drift)"
            echo "  --env NAME       (CartPole, HalfCheetah, FrozenLake, MiniGrid, MountainCar, or 'all')"
            echo "  --seeds N        Number of seeds to run (default: 5)"
            echo "  --jobs N         Number of parallel jobs (default: 8)"
            echo "  --dry-run        Print commands without executing"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")/.."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_command() {
    CMD="$1"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $CMD"
    else
        echo -e "${GREEN}[RUNNING]${NC} $CMD"
        eval "$CMD" || {
            echo -e "${RED}[FAILED]${NC} Command failed"
            # Continue despite failure? No, usually safer to stop.
            exit 1
        }
    fi
}

# =============================================================================
# CAMPAIGN 1: MAIN COMPARISON
# NS-MD-MPI vs Baseline (Heuristic Adaptive removed per request)
# =============================================================================
run_main_comparison() {
    LOCAL_ENV="$1"
    echo -e "${BLUE}>>> Starting Main Comparison Campaign: ${LOCAL_ENV}${NC}"
    
    # Define drift types and algos based on Env
    case $LOCAL_ENV in
        CartPole)
            DRIFT_TYPES=("gravity_sine" "gravity_jump")
            ALGO="PPO"
            ;;
        MountainCar)
            DRIFT_TYPES=("gravity_sine" "force_jump")
            ALGO="PPO"
            ;;
        FrozenLake)
            DRIFT_TYPES=("slip_sine" "slip_jump")
            ALGO="PPO"
            ;;
        MiniGrid)
            DRIFT_TYPES=("reward_sine" "reward_jump")
            ALGO="PPO"
            ;;
        HalfCheetah)
            DRIFT_TYPES=("friction_sine" "gravity_linear")
            ALGO="SAC"
            ;;
        *)
            echo "Unknown env defaults for $LOCAL_ENV. Using generic."
            DRIFT_TYPES=("gravity_sine")
            ALGO="PPO"
            ;;
    esac

    echo "Running with max ${MAX_JOBS} parallel jobs..."

    for DRIFT in "${DRIFT_TYPES[@]}"; do
        for (( i=0; i<SEEDS; i++ )); do
            SEED=$((START_SEED + i))
            
            # --- JOB 1: Baseline ---
            CONFIG="configs/${LOCAL_ENV}_baseline_${DRIFT}_${ALGO,,}.yaml"
            EXP_NAME="${LOCAL_ENV}_Baseline_${DRIFT}_Seed${SEED}_${TIMESTAMP}"
            
            if [ ! -f "$CONFIG" ]; then
                echo -e "${YELLOW}Config not found: $CONFIG. Skipping.${NC}"
                continue
            fi

            # Temporary modified config for seed
            TEMP_CONFIG="configs/temp_${EXP_NAME}.yaml"
            cp "$CONFIG" "$TEMP_CONFIG"
            sed -i "s/seed: .*/seed: $SEED/" "$TEMP_CONFIG"
            
            # Run in background
            run_command "python scripts/train.py --config $TEMP_CONFIG --exp_name $EXP_NAME" &
            
            # --- JOB 2: NS-MD-MPI ---
            CONFIG="configs/${LOCAL_ENV}_nsmdmpi_${DRIFT}_${ALGO,,}.yaml"
            EXP_NAME="${LOCAL_ENV}_NSMDMPI_${DRIFT}_Seed${SEED}_${TIMESTAMP}"
            
            if [ ! -f "$CONFIG" ]; then
                echo -e "${YELLOW}Config not found: $CONFIG. Skipping.${NC}"
                continue
            fi

            TEMP_CONFIG="configs/temp_${EXP_NAME}.yaml"
            cp "$CONFIG" "$TEMP_CONFIG"
            sed -i "s/seed: .*/seed: $SEED/" "$TEMP_CONFIG"
            
            # Run in background
            run_command "python scripts/train.py --config $TEMP_CONFIG --exp_name $EXP_NAME" &
            
            # Rate limiting
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 1
            done
        done
    done
    
    echo "Waiting for all jobs to complete..."
    wait
}

# =============================================================================
# CAMPAIGN 2: BUDGET SCALING
# Vary drift magnitude: Low, Medium, High
# =============================================================================
run_budget_scaling() {
    LOCAL_ENV="$1"
    echo -e "${BLUE}>>> Starting Budget Scaling Campaign: ${LOCAL_ENV}${NC}"
    
    # Defaults for Budget Scaling
    DRIFT_TYPE="gravity_sine"
    ALGO="PPO"
    MAGNITUDES=(2.0 5.0 10.0)
    
    BASE_CONFIG="configs/${LOCAL_ENV}_nsmdmpi_${DRIFT_TYPE}_${ALGO,,}.yaml"
    BASELINE_CFG="configs/${LOCAL_ENV}_baseline_${DRIFT_TYPE}_${ALGO,,}.yaml"
    
    for MAG in "${MAGNITUDES[@]}"; do
        for (( i=0; i<SEEDS; i++ )); do
            SEED=$((START_SEED + i))
            
            # NS-MD-MPI
            EXP_NAME="${LOCAL_ENV}_NSMDMPI_Mag${MAG}_Seed${SEED}_${TIMESTAMP}"
            TEMP_CONFIG="configs/temp_${EXP_NAME}.yaml"
            cp "$BASE_CONFIG" "$TEMP_CONFIG"
            sed -i "s/seed: .*/seed: $SEED/" "$TEMP_CONFIG"
            sed -i "s/magnitude: .*/magnitude: $MAG/" "$TEMP_CONFIG"
            
            run_command "python scripts/train.py --config $TEMP_CONFIG --exp_name $EXP_NAME" &
            
            # Baseline
            EXP_NAME="${LOCAL_ENV}_Baseline_Mag${MAG}_Seed${SEED}_${TIMESTAMP}"
            TEMP_CONFIG="configs/temp_${EXP_NAME}.yaml"
            cp "$BASELINE_CFG" "$TEMP_CONFIG"
            sed -i "s/seed: .*/seed: $SEED/" "$TEMP_CONFIG"
            sed -i "s/magnitude: .*/magnitude: $MAG/" "$TEMP_CONFIG"
            
            run_command "python scripts/train.py --config $TEMP_CONFIG --exp_name $EXP_NAME" &
            
             # Rate limiting
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 1
            done
        done
    done
    
    echo "Waiting for all jobs to complete..."
    wait
}


# =============================================================================
# MAIN DISPATCH
# =============================================================================

echo "Campaign: $CAMPAIGN"
echo "Environment: $ENV"
echo "Seeds: $SEEDS"
echo "Jobs: $MAX_JOBS"
echo "Dry Run: $DRY_RUN"
echo ""

ENV_LIST=()
if [ "$ENV" == "all" ]; then
    ENV_LIST=("CartPole" "FrozenLake" "HalfCheetah" "MiniGrid" "MountainCar")
else
    ENV_LIST=("$ENV")
fi

if [ "$CAMPAIGN" == "main_comparison" ]; then
    for TARGET_ENV in "${ENV_LIST[@]}"; do
        run_main_comparison "$TARGET_ENV"
    done
elif [ "$CAMPAIGN" == "budget_scaling" ]; then
     for TARGET_ENV in "${ENV_LIST[@]}"; do
        run_budget_scaling "$TARGET_ENV"
    done
else
    echo -e "${RED}Campaign '$CAMPAIGN' not implemented yet.${NC}"
    exit 1
fi
