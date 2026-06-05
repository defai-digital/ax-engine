#!/usr/bin/env bash
set -euo pipefail

SERVER_BIN="${SERVER_BIN:-/Users/akiralam/code/ax-engine_v5/target/release/ax-engine-server}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface/hub}"
QA_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORT_DIR="${REPORT_DIR:-$QA_DIR/reports}"
PORT="${PORT:-8080}"
TIMEOUT="${TIMEOUT:-120}"

mkdir -p "$REPORT_DIR"

# Model configurations: preset|model_id|artifacts_dir
MODELS=(
  "glm4.7-flash-4bit|glm4_moe_lite|$HF_CACHE/models--mlx-community--GLM-4.7-Flash-4bit/snapshots/1454cffb1a21737e162f508e5bc70be9def89276"
  "gemma4-e2b|gemma4-e2b|$HF_CACHE/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd"
  "qwen3.5-27b-4bit|qwen3.5-27b|$HF_CACHE/models--mlx-community--Qwen3.6-27B-4bit/snapshots/c000ac2c2057d94be3fa931000c31723aac53282"
  "qwen3.6-35b|qwen3.6-35b|$HF_CACHE/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46"
)

wait_for_server() {
  local max_wait=120
  local waited=0
  while ! curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    sleep 2
    waited=$((waited + 2))
    if [ "$waited" -ge "$max_wait" ]; then
      echo "ERROR: Server did not start within ${max_wait}s"
      return 1
    fi
    echo "  Waiting for server... (${waited}s)"
  done
  echo "  Server ready after ${waited}s"
  return 0
}

kill_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

run_qa_for_model() {
  local preset="$1"
  local model_id="$2"
  local artifacts_dir="$3"
  local mode="$4"  # "direct" or "ngram"

  local mode_flag=""
  if [ "$mode" = "direct" ]; then
    mode_flag="--disable-ngram-acceleration"
  fi

  echo ""
  echo "=== $model_id | mode=$mode ==="
  echo "  Artifacts: $artifacts_dir"

  # Check artifacts exist
  if [ ! -f "$artifacts_dir/config.json" ]; then
    echo "  SKIP: artifacts not found at $artifacts_dir"
    return 0
  fi

  # Start server
  echo "  Starting server..."
  kill_server
  AX_ALLOW_UNSUPPORTED_HOST="${AX_ALLOW_UNSUPPORTED_HOST:-1}" "$SERVER_BIN" \
    --port "$PORT" \
    --model-id "$model_id" \
    --support-tier mlx-preview \
    --mlx-model-artifacts-dir "$artifacts_dir" \
    --mlx \
    $mode_flag \
    > "$REPORT_DIR/server-${model_id}-${mode}.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server; then
    echo "  FAIL: Server failed to start for $model_id ($mode)"
    kill_server
    return 1
  fi

  # Run QA tests
  local report_file="$REPORT_DIR/qa-${model_id}-${mode}-$(date +%Y%m%d-%H%M%S).html"
  local tokenizer_path="$artifacts_dir/tokenizer.json"
  echo "  Running QA tests..."
  python3 "$QA_DIR/run_qa.py" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$model_id" \
    --tokenizer "$tokenizer_path" \
    --streams both \
    --max-tokens 512 \
    --temperature 0.0 \
    --timeout "$TIMEOUT" \
    --output "$report_file" \
    2>&1 | sed 's/^/  /'

  echo "  Report: $report_file"

  # Stop server
  kill_server
  echo "  Done: $model_id ($mode)"
}

echo "AX Engine Full QA Test Suite"
echo "  Server: $SERVER_BIN"
echo "  Reports: $REPORT_DIR"
echo "  Models: ${#MODELS[@]}"
echo ""

trap kill_server EXIT

for model_config in "${MODELS[@]}"; do
  IFS='|' read -r preset model_id artifacts_dir <<< "$model_config"

  # Run direct mode
  run_qa_for_model "$preset" "$model_id" "$artifacts_dir" "direct"

  # Run n-gram mode
  run_qa_for_model "$preset" "$model_id" "$artifacts_dir" "ngram"
done

echo ""
echo "=== All QA tests complete ==="
echo "Reports in: $REPORT_DIR"
ls -la "$REPORT_DIR"/*.html 2>/dev/null || echo "No HTML reports generated"

echo ""
echo "=== Generating summary page ==="
python3 "$QA_DIR/generate_summary.py" "$REPORT_DIR"
echo "Summary: $REPORT_DIR/summary.html"
