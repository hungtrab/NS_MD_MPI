#!/bin/bash
# Moderate Experiments - Menu Launcher

echo "════════════════════════════════════════"
echo "  MODERATE DRIFT EXPERIMENTS"
echo "════════════════════════════════════════"
echo ""
echo "Choose environment and algorithm:"
echo ""
echo "  1) Hopper + PPO"
echo "  2) Hopper + SAC"
echo "  3) Hopper + TRPO"
echo "  4) HalfCheetah + PPO"
echo "  5) HalfCheetah + SAC"
echo "  6) HalfCheetah + TRPO"
echo "  7) LunarLander + PPO"
echo "  8) LunarLander + SAC"
echo "  9) LunarLander + TRPO"
echo ""
echo "  0) Run ALL (not recommended - high VRAM usage)"
echo ""
read -p "Enter choice [0-9]: " choice

case $choice in
    1) bash scripts/run_hopper_moderate_ppo.sh ;;
    2) bash scripts/run_hopper_moderate_sac.sh ;;
    3) bash scripts/run_hopper_moderate_trpo.sh ;;
    4) bash scripts/run_halfcheetah_moderate_ppo.sh ;;
    5) bash scripts/run_halfcheetah_moderate_sac.sh ;;
    6) bash scripts/run_halfcheetah_moderate_trpo.sh ;;
    7) bash scripts/run_lunarlander_moderate_ppo.sh ;;
    8) bash scripts/run_lunarlander_moderate_sac.sh ;;
    9) bash scripts/run_lunarlander_moderate_trpo.sh ;;
    0) 
        echo "Running ALL moderate experiments..."
        for script in scripts/run_*_moderate_*.sh; do
            [ -f "$script" ] && bash "$script"
            sleep 5
        done
        ;;
    *) echo "Invalid choice" ;;
esac
