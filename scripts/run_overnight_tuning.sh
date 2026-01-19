#!/bin/bash
# =====================================================
# OVERNIGHT TUNING SCRIPT
# Run all sweeps in parallel - Hopper + HalfCheetah
# =====================================================

cd /home/hungchan/Work/Deep-RL
source ~/miniconda3/bin/activate rl_hf_course

mkdir -p logs/tuning

echo "=========================================="
echo "Starting overnight tuning at $(date)"
echo "=========================================="

# =====================================================
# HOPPER NS-MDMPI (4 drift types)
# =====================================================

# Sine
python scripts/wandb_sweep_agent.py --create-sweep --type moderate --env Hopper-v4 --count 20 \
  > logs/tuning/hopper_nsmdmpi_1.log 2>&1 &
PID1=$!
echo "[1] Hopper NS-MDMPI #1 started (PID: $PID1)"

sleep 3

# Another parallel agent on same sweep  
python scripts/wandb_sweep_agent.py --create-sweep --type moderate --env Hopper-v4 --count 20 \
  > logs/tuning/hopper_nsmdmpi_2.log 2>&1 &
PID2=$!
echo "[2] Hopper NS-MDMPI #2 started (PID: $PID2)"

# =====================================================
# HALFCHEETAH NS-MDMPI 
# =====================================================

sleep 3

python scripts/wandb_sweep_agent.py --create-sweep --type moderate --env HalfCheetah-v4 --count 20 \
  > logs/tuning/halfcheetah_nsmdmpi_1.log 2>&1 &
PID3=$!
echo "[3] HalfCheetah NS-MDMPI #1 started (PID: $PID3)"

sleep 3

python scripts/wandb_sweep_agent.py --create-sweep --type moderate --env HalfCheetah-v4 --count 20 \
  > logs/tuning/halfcheetah_nsmdmpi_2.log 2>&1 &
PID4=$!
echo "[4] HalfCheetah NS-MDMPI #2 started (PID: $PID4)"

# =====================================================
# SUMMARY
# =====================================================

echo ""
echo "=========================================="
echo "All tuning jobs started!"
echo "=========================================="
echo "PIDs: $PID1 $PID2 $PID3 $PID4"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/tuning/hopper_nsmdmpi_1.log"
echo "  tail -f logs/tuning/halfcheetah_nsmdmpi_1.log"
echo ""
echo "WandB Dashboard:"
echo "  https://wandb.ai/hungtrab-hanoi-university-of-science-and-technology"
echo ""
echo "Estimated completion: ~6-8 hours"

# Wait for all
wait $PID1 $PID2 $PID3 $PID4
echo ""
echo "All tuning completed at $(date)"
