#!/bin/bash
# =============================================================================
# BENCHMARK SCRIPT: Adaptive vs Baseline Comparison
# =============================================================================
#
# This script runs systematic experiments comparing:
#   - Adaptive mechanism (drift-aware LR scheduling)
#   - Baseline (fixed learning rate)
#
# Across multiple algorithms (PPO, SAC, TRPO) and environments
#
# Usage:
#   ./scripts/benchmark.sh                    # Run all benchmarks
#   ./scripts/benchmark.sh --env cartpole     # Single environment
#   ./scripts/benchmark.sh --algo PPO         # Single algorithm
#   ./scripts/benchmark.sh --quick            # Quick test (fewer steps)
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
ALGORITHMS=("PPO" "SAC" "TRPO")
ENVIRONMENTS=("cartpole" "mountaincar")
DRIFT_TYPES=("sine" "jump" "random_walk")
SEEDS=(42 123 456)
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENTS=("$2")
            shift 2
            ;;
        --algo)
            ALGORITHMS=("$2")
            shift 2
            ;;
        --drift)
            DRIFT_TYPES=("$2")
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --seed)
            SEEDS=("$2")
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env ENV        Run only specified environment (cartpole, mountaincar)"
            echo "  --algo ALGO      Run only specified algorithm (PPO, SAC, TRPO)"
            echo "  --drift TYPE     Run only specified drift type (sine, jump, random_walk)"
            echo "  --quick          Quick mode with fewer timesteps"
            echo "  --seed SEED      Use specific seed (default: 42 123 456)"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Non-Stationary MDP Benchmark Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Algorithms:   ${GREEN}${ALGORITHMS[*]}${NC}"
echo -e "Environments: ${GREEN}${ENVIRONMENTS[*]}${NC}"
echo -e "Drift Types:  ${GREEN}${DRIFT_TYPES[*]}${NC}"
echo -e "Seeds:        ${GREEN}${SEEDS[*]}${NC}"
echo -e "Quick Mode:   ${GREEN}${QUICK_MODE}${NC}"
echo ""

# Create results directory
RESULTS_DIR="results/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo -e "Results will be saved to: ${YELLOW}${RESULTS_DIR}${NC}"
echo ""

# Function to generate temp config with modifications
generate_config() {
    local base_config=$1
    local adaptive=$2
    local drift_type=$3
    local seed=$4
    local output_file=$5
    local quick=$6
    
    # Copy base config
    cp "$base_config" "$output_file"
    
    # Modify adaptive setting
    if [[ "$adaptive" == "true" ]]; then
        sed -i "s/enabled: false/enabled: true/" "$output_file"
        sed -i "s/enabled:  *false/enabled: true/" "$output_file"
    else
        sed -i "s/enabled: true/enabled: false/" "$output_file"
        sed -i "s/enabled:  *true/enabled: false/" "$output_file"
    fi
    
    # Modify drift type
    sed -i "s/drift_type: .*/drift_type: \"$drift_type\"/" "$output_file"
    
    # Modify seed
    sed -i "s/seed: .*/seed: $seed/" "$output_file"
    
    # Quick mode: reduce timesteps
    if [[ "$quick" == "true" ]]; then
        sed -i "s/total_timesteps: .*/total_timesteps: 10000/" "$output_file"
    fi
}

# Function to run a single experiment
run_experiment() {
    local env=$1
    local algo=$2
    local drift=$3
    local adaptive=$4
    local seed=$5
    
    local mode_name="Adaptive"
    [[ "$adaptive" == "false" ]] && mode_name="Baseline"
    
    local exp_name="${env}_${algo}_${drift}_${mode_name}_seed${seed}"
    
    echo -e "${YELLOW}>>> Running: ${exp_name}${NC}"
    
    # Get base config
    local config_file="configs/${env}_adaptive.yaml"
    if [[ ! -f "$config_file" ]]; then
        echo -e "${RED}Config not found: ${config_file}${NC}"
        return 1
    fi
    
    # Generate temp config
    local temp_config="${RESULTS_DIR}/${exp_name}_config.yaml"
    generate_config "$config_file" "$adaptive" "$drift" "$seed" "$temp_config" "$QUICK_MODE"
    
    # Run training
    python scripts/train.py \
        --config "$temp_config" \
        --algo "$algo" \
        --exp_name "$exp_name" \
        2>&1 | tee "${RESULTS_DIR}/${exp_name}.log"
    
    local status=$?
    if [[ $status -eq 0 ]]; then
        echo -e "${GREEN}✓ Completed: ${exp_name}${NC}"
    else
        echo -e "${RED}✗ Failed: ${exp_name}${NC}"
    fi
    
    return $status
}

# Track statistics
total_runs=0
successful_runs=0
failed_runs=0

# Main benchmark loop
for env in "${ENVIRONMENTS[@]}"; do
    for algo in "${ALGORITHMS[@]}"; do
        for drift in "${DRIFT_TYPES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                # Run Adaptive
                ((total_runs++))
                if run_experiment "$env" "$algo" "$drift" "true" "$seed"; then
                    ((successful_runs++))
                else
                    ((failed_runs++))
                fi
                
                # Run Baseline
                ((total_runs++))
                if run_experiment "$env" "$algo" "$drift" "false" "$seed"; then
                    ((successful_runs++))
                else
                    ((failed_runs++))
                fi
            done
        done
    done
done

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}           Benchmark Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total Runs:      ${total_runs}"
echo -e "Successful:      ${GREEN}${successful_runs}${NC}"
echo -e "Failed:          ${RED}${failed_runs}${NC}"
echo -e "Results saved:   ${YELLOW}${RESULTS_DIR}${NC}"
echo ""

# Generate summary CSV
SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
echo "experiment,algorithm,environment,drift_type,mode,seed,status" > "$SUMMARY_FILE"
echo -e "Summary CSV: ${YELLOW}${SUMMARY_FILE}${NC}"

exit 0
