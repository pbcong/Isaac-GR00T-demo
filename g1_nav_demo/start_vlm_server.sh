#!/usr/bin/env bash
# Start the Qwen3.5-9B vLLM server (OpenAI-compatible API at localhost:8000).
# Run this ONCE before running run_vlm_demo.sh.
# Keep this terminal open — the server must stay running during the demo.

cleanup() {
    echo "Shutting down vLLM workers..."
    kill -9 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VLLM="$PROJECT_DIR/.venv/bin/vllm"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${1:-8000}"

export HF_HOME="$SCRIPT_DIR/models"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,5}"

echo "Starting $MODEL vLLM server on port $PORT ..."
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

"$VLLM" serve "$MODEL" \
    --served-model-name "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.5
