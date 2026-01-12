#!/bin/bash
# HalfCheetah Vanilla - TRPO
# Runs baseline + NS-MDMPI (if applicable)

echo "Running HalfCheetah Vanilla (TRPO)"
echo "═══════════════════════════════════════"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/TRPO/vanilla/halfcheetah_vanilla_trpo.yaml \
  > logs/halfcheetah_vanilla_trpo.log 2>&1 &

echo "✅ Started 1 run (baseline only)"
echo "Monitor: tail -f logs/halfcheetah_vanilla_trpo.log"
