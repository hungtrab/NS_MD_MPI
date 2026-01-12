#!/bin/bash
# MASTER SCRIPT: Launch ALL Experiments
# Vanilla + Moderate + Extreme + Multi across all envs and algorithms

echo "================================================================"
echo "  🚀 MASTER LAUNCHER - ALL NS-MDMPI EXPERIMENTS"
echo "================================================================"
echo ""
echo "This will launch ALL experiments:"
echo ""
echo "  📊 Vanilla:   9 experiments (3 envs × 3 algos)"
echo "  📊 Moderate: ~108 experiments (3 envs × 6 configs × 3 algos × 2 methods)"
echo "  📊 Extreme:  ~36 experiments (3 envs × 2 configs × 3 algos × 2 methods)"
echo "  📊 Multi:    ~18 experiments (3 envs × 1 config × 3 algos × 2 methods)"
echo "  ─────────────"
echo "  📊 TOTAL:    ~171 experiments"
echo ""
echo "Estimated time: 30-40 hours (with parallel execution)"
echo ""
echo "⚠️  WARNING: This will consume significant computational resources!"
echo "================================================================"
echo ""

read -p "Are you SURE you want to launch ALL experiments? (yes/no) " -r
echo
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo "Cancelled. Use individual scripts for smaller batches:"
    echo "  bash scripts/run_all_vanilla.sh"
    echo "  bash scripts/run_all_moderate.sh"
    echo "  bash scripts/run_all_extreme.sh"
    echo "  bash scripts/run_all_multi.sh"
    exit 1
fi

echo ""
echo "🚀 Starting experiment marathon..."
echo ""

# Phase 1: Vanilla (baseline for comparison)
echo "═══ Phase 1/4: Vanilla Baselines ═══"
bash scripts/run_all_vanilla.sh
sleep 10

# Phase 2: Moderate (main comparison)
echo ""
echo "═══ Phase 2/4: Moderate Drift ═══"
bash scripts/run_all_moderate.sh
sleep 10

# Phase 3: Extreme (stress testing)
echo ""
echo "═══ Phase 3/4: Extreme Drift ═══"
bash scripts/run_all_extreme.sh
sleep 10

# Phase 4: Multi-parameter (complex scenarios)
echo ""
echo "═══ Phase 4/4: Multi-Parameter Drift ═══"
bash scripts/run_all_multi.sh

echo ""
echo "================================================================"
echo "✅ ALL EXPERIMENT PHASES LAUNCHED!"
echo "================================================================"
echo ""
echo "Monitor overall progress:"
echo "  watch -n 10 'ps aux | grep train.py | grep -v grep | wc -l'"
echo ""
echo "Check individual logs:"
echo "  ls -lht logs/*.log | head -20"
echo ""
echo "WandB Dashboard:"
echo "  https://wandb.ai/<your-entity>/"
echo ""
echo "Good luck with your experiments! 🎯"
echo "================================================================"
