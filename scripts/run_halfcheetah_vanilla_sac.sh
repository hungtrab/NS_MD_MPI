#!/bin/bash
# HalfCheetah Vanilla - SAC
# Runs baseline + NS-MDMPI (if applicable)

echo "Running HalfCheetah Vanilla (SAC)"
echo "═══════════════════════════════════════"

conda run -n rl_hf_course python scripts/train.py \
  --config configs/SAC/vanilla/halfcheetah_vanilla_sac.yaml \
  > logs/halfcheetah_vanilla_sac.log 2>&1 &

echo "✅ Started 1 run (baseline only)"
echo "Monitor: tail -f logs/halfcheetah_vanilla_sac.log"
