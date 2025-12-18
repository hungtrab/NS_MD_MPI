#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts"
CONFIGS_DIR="$ROOT_DIR/configs"
MODELS_DIR="$ROOT_DIR/models"
VIDEOS_DIR="$ROOT_DIR/videos"

# Defaults (override via env: CONFIG, ALGO, EXP_NAME, VIDEO_EPISODES, VIDEO_LENGTH)
CONFIG_DEFAULT="$CONFIGS_DIR/mountaincar_adaptive.yaml"
ALGO_DEFAULT="PPO"

CONFIG="${CONFIG:-$CONFIG_DEFAULT}"
ALGO="${ALGO:-$ALGO_DEFAULT}"
EXP_NAME="${EXP_NAME:-mountaincar_${ALGO}_$(date +%Y%m%d-%H%M%S)}"
VIDEO_EPISODES="${VIDEO_EPISODES:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1000}"

mkdir -p "$MODELS_DIR" "$VIDEOS_DIR"

echo "=== Training MountainCar with $ALGO ==="
echo "Config: $CONFIG"
echo "Run name: $EXP_NAME"

python "$SCRIPTS_DIR/train.py" \
  --config "$CONFIG" \
  --algo "$ALGO" \
  --exp_name "$EXP_NAME"

MODEL_PATH="$MODELS_DIR/${EXP_NAME}.zip"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[WARN] Model not found: $MODEL_PATH"
  echo "      Skipping rendering."
  exit 0
fi

echo "--- Rendering video ---"
python "$SCRIPTS_DIR/render.py" \
  --model "$MODEL_PATH" \
  --config "$CONFIG" \
  --algo "$ALGO" \
  --output "$VIDEOS_DIR" \
  --episodes "$VIDEO_EPISODES" \
  --length "$VIDEO_LENGTH"
