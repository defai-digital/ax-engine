#!/usr/bin/env zsh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=${1:-$(date +%Y%m%d-%H%M%S)}
RESULTS_DIR="$REPO_DIR/benchmarks/results"
SCRIPT="$REPO_DIR/benchmarks/run_apple_to_apple.py"

# All 10 models, ordered smallest to largest
MODELS=(
  "Qwen3-4B-Q6_K.gguf"
  "Qwen3.5-4B-Q8_0.gguf"
  "meta-llama-3.1-8b-instruct-q5_k_m.gguf"
  "Qwen3.5-9B-Q4_K_M.gguf"
  "gemma-3-12b-it-Q4_K_M.gguf"
  "gemma-3-27b-it-Q4_K_M.gguf"
  "Qwen3.5-27B-Q4_K_M.gguf"
  "Qwen3-32B-Q4_K_M.gguf"
  "Qwen3.5-35B-A3B-Q4_K_M.gguf"
  "meta-llama-3-70b-instruct.Q4_K_M.gguf"
)

TOTAL=${#MODELS[@]}
LAST_MODEL=${MODELS[-1]}

echo "=== Running all $TOTAL model benchmarks ==="
echo "Timestamp: $TIMESTAMP"
echo "Results: $RESULTS_DIR"
echo ""

COMPLETED=0
FAILED=0

for MODEL in "${MODELS[@]}"; do
  MODEL_PATH="$REPO_DIR/models/$MODEL"
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "SKIP: $MODEL (not found)"
    continue
  fi

  echo ""
  echo "=========================================="
  echo "MODEL: $MODEL"
  echo "=========================================="

  if python3 "$SCRIPT" \
    --model "$MODEL_PATH" \
    --samples 3 \
    --cooldown-seconds 15 \
    --prompt-tokens 512 \
    --decode-tokens 128 \
    --timestamp "$TIMESTAMP"; then
    COMPLETED=$((COMPLETED + 1))
    echo "DONE: $MODEL"
  else
    FAILED=$((FAILED + 1))
    echo "FAIL: $MODEL"
  fi

  # Extra cooldown between models
  if [[ "$MODEL" != "$LAST_MODEL" ]]; then
    echo "--- Cooling down 30s before next model ---"
    sleep 30
  fi
done

echo ""
echo "=== Summary ==="
echo "Completed: $COMPLETED / $TOTAL"
echo "Failed: $FAILED"
echo "Results in: $RESULTS_DIR/$TIMESTAMP-*"
