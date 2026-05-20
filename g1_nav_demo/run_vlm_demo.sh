#!/usr/bin/env bash
# Run the G1 navigation demo with OpenRouter VLM goal parsing.
# Usage:
#   Single-turn:  bash run_vlm_demo.sh "go to the table" demo_output.mp4
#   Multi-turn:   bash run_vlm_demo.sh --multiturn [output_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${POLICY_GPU:-3}"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

: "${OPENROUTER_API_KEY:?ERROR: OPENROUTER_API_KEY not set. Add it to $PROJECT_DIR/.env}"

MULTITURN_FLAG=""

if [ "$1" = "--multiturn" ]; then
    MULTITURN_FLAG="--multiturn"
    OUTPUT_DIR="${2:-demo_output}"
    echo "Mode    : multi-turn interactive"
    echo "Output  : $OUTPUT_DIR/"
    echo ""
    python "$PROJECT_DIR/g1_nav_demo/run_demo.py" \
        --multiturn \
        --output-dir "$OUTPUT_DIR" \
        --policy-path "$SCRIPT_DIR/walk_policy/motion.pt" \
        --vlm-model "${MODEL:-x-ai/grok-4.3}" \
        --device cuda \
        --max-steps 10000
else
    COMMAND="${1:-go to the table}"
    OUTPUT="${2:-demo_output.mp4}"
    echo "Command : $COMMAND"
    echo "Output  : $OUTPUT"
    echo ""
    python "$PROJECT_DIR/g1_nav_demo/run_demo.py" \
        --command "$COMMAND" \
        --policy-path "$SCRIPT_DIR/walk_policy/motion.pt" \
        --output "$OUTPUT" \
        --vlm-model "${MODEL:-x-ai/grok-4.3}" \
        --device cuda \
        --max-steps 10000
fi
