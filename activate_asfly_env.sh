#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MPLCONFIGDIR="$SCRIPT_DIR/.cache/matplotlib"
export YOLO_CONFIG_DIR="$SCRIPT_DIR/.cache/ultralytics"
export ASFLY_ULTRALYTICS_DIR="$SCRIPT_DIR/.cache/ultralytics"

mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"
source "$SCRIPT_DIR/.venv/bin/activate"

echo "ASFLY environment active: $VIRTUAL_ENV"
echo "Use: python main_async_fusion.py --model models_archive/iha_best.pt --imgsz 320 --conf 0.25 --device cpu"
