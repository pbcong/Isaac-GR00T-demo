#!/usr/bin/env bash
# Single-turn:  bash run_vlm_demo.sh "go to the table" [output.mp4]
# Multi-turn:   bash run_vlm_demo.sh --multiturn [output_dir]
# Overrides:    DEVICE=cpu|cuda  GPU=0  MODEL=...  MAX_STEPS=5000
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_DIR/.venv/bin/python"

[ -f "$PROJECT_DIR/.env" ] && { set -a; source "$PROJECT_DIR/.env"; set +a; }
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set — add to $PROJECT_DIR/.env}"
[ -x "$PYTHON" ] || { echo "venv not found — run: uv venv .venv --python 3.10 && uv pip install -e ."; exit 1; }

DEVICE="${DEVICE:-cuda}"
[ "$DEVICE" = "cuda" ] && export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

COMMON=(
    --policy-path "$SCRIPT_DIR/walk_policy/motion.pt"
    --vlm-model "${MODEL:-x-ai/grok-4.3}"
    --device "$DEVICE"
    --max-steps "${MAX_STEPS:-10000}"
)

if [ "$1" = "--multiturn" ]; then
    "$PYTHON" "$PROJECT_DIR/g1_nav_demo/run_demo.py" --multiturn --output-dir "${2:-demo_output}" "${COMMON[@]}"
else
    echo "${1:-go to the table} → ${2:-demo_output.mp4}  [device=$DEVICE]"
    "$PYTHON" "$PROJECT_DIR/g1_nav_demo/run_demo.py" --command "${1:-go to the table}" --output "${2:-demo_output.mp4}" "${COMMON[@]}"
fi
