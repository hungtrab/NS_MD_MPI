#!/usr/bin/env bash

set -euo pipefail

# Multi-environment training + rendering helper
# Trains specified environments via scripts/train.py and records videos via scripts/render.py.
# Defaults assume adaptive configs exist under configs/.

# Usage:
#   ./scripts/run_multi_env_train_render.sh                # run all known envs (if configs exist)
#   ./scripts/run_multi_env_train_render.sh mountaincar    # run only MountainCar
#   ./scripts/run_multi_env_train_render.sh frozenlake     # run only FrozenLake
#   ./scripts/run_multi_env_train_render.sh minigrid       # run only MiniGrid
#   ./scripts/run_multi_env_train_render.sh halfcheetah    # run only HalfCheetah
#
# Env vars:
#   VIDEO_EPISODES: number of episodes to record (default: 1)
#   VIDEO_LENGTH:   max steps per episode (default: 1000)

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts"
CONFIGS_DIR="$ROOT_DIR/configs"
MODELS_DIR="$ROOT_DIR/models"
VIDEOS_DIR="$ROOT_DIR/videos"

VIDEO_EPISODES="${VIDEO_EPISODES:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1000}"

# Map short names to config files and recommended algorithms
declare -A ENV_CONFIG
declare -A ENV_ALGO

ENV_CONFIG[mountaincar]="$CONFIGS_DIR/mountaincar_adaptive.yaml"
ENV_ALGO[mountaincar]="PPO"      # discrete

ENV_CONFIG[frozenlake]="$CONFIGS_DIR/frozenlake_adaptive.yaml"
ENV_ALGO[frozenlake]="PPO"       # discrete

ENV_CONFIG[minigrid]="$CONFIGS_DIR/minigrid_adaptive.yaml"
ENV_ALGO[minigrid]="PPO"         # discrete

ENV_CONFIG[halfcheetah]="$CONFIGS_DIR/halfcheetah_adaptive.yaml"
ENV_ALGO[halfcheetah]="SAC"      # continuous

run_one() {
  local name="$1"
  local cfg_path="${ENV_CONFIG[$name]}"
  local algo="${ENV_ALGO[$name]}"

  if [[ -z "$cfg_path" || -z "$algo" ]]; then
    echo "[SKIP] Unknown env key: $name"
    return 0
  fi

  if [[ ! -f "$cfg_path" ]]; then
    echo "[SKIP] Config not found for $name: $cfg_path"
    echo "       Create the YAML at this path to enable $name."
    return 0
  fi

  # Deterministic run name to find the model later
  local ts
  ts="$(date +%Y%m%d-%H%M%S)"
  local exp_name
  exp_name="${name}_${algo}_${ts}"

  echo "\n=== Training $name with $algo ==="
  echo "Config: $cfg_path"
  echo "Run name: $exp_name"

  python "$SCRIPTS_DIR/train.py" \
    --config "$cfg_path" \
    --algo "$algo" \
    --exp_name "$exp_name"

  local model_path
  model_path="$MODELS_DIR/${exp_name}.zip"

  if [[ ! -f "$model_path" ]]; then
    echo "[WARN] Model not found after training: $model_path"
    echo "       Rendering will be skipped for $name."
    return 0
  fi

  echo "\n--- Rendering video for $name ($algo) ---"
  python "$SCRIPTS_DIR/render.py" \
    --model "$model_path" \
    --config "$cfg_path" \
    --algo "$algo" \
    --output "$VIDEOS_DIR" \
    --episodes "$VIDEO_EPISODES" \
    --length "$VIDEO_LENGTH"
}

main() {
  mkdir -p "$MODELS_DIR" "$VIDEOS_DIR"

  if [[ $# -gt 0 ]]; then
    # Run only specified env keys
    for key in "$@"; do
      run_one "$key"
    done
  else
    # Run all known envs (skip those missing configs)
    for key in mountaincar frozenlake minigrid halfcheetah; do
      run_one "$key"
    done
  fi
}

main "$@"
