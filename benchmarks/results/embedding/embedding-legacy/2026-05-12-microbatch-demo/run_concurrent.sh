#!/usr/bin/env bash
# Demonstrate the server-side EmbeddingMicroBatcher: 10 concurrent HTTP
# /v1/embeddings requests arriving within the configured window
# (default 2ms, bumped to 20ms here for predictable coalescing) are
# coalesced into one batched runner call.
#
# Output schema fields:
#   results.ax_engine_http              — sequential per-request HTTP
#   results_concurrent_http.ax_engine_http — 10 concurrent HTTP requests
#
# The two together show the user-facing speedup a serving caller gets
# automatically when their workload sends embeddings concurrently.
set -uo pipefail

REPO_ROOT="/Users/akiralam/code/ax-engine-v4"
cd "$REPO_ROOT"

source .venv/bin/activate

OUTDIR="benchmarks/results/embedding/2026-05-12-microbatch-demo"
BENCH="scripts/bench_embedding_models.py"

cleanup_servers() {
    pkill -f 'ax-engine-server' >/dev/null 2>&1 || true
    sleep 2
}

run_model() {
    local label="$1"
    local model_dir="$2"
    echo ""
    echo "  [microbatch-demo] $label  ($(date '+%H:%M:%S'))"
    cleanup_servers
    PYTHONUNBUFFERED=1 python -u "$BENCH" \
        --model-dir "$model_dir" \
        --model-label "$label" \
        --trials 5 \
        --cooldown 10 \
        --skip-swift \
        --include-concurrent-http \
        --microbatch-window-ms 20 \
        --output-dir "$OUTDIR" \
        2>&1 | tee "$OUTDIR/logs-${label}.log"
    sleep 30
}

mkdir -p "$OUTDIR"
run_model qwen3-embedding-0.6b-8bit   .internal/models/qwen3-embedding-0.6b-8bit
run_model qwen3-embedding-4b-4bit     .internal/models/qwen3-embedding-4b-4bit
run_model qwen3-embedding-8b-4bit-dwq .internal/models/qwen3-embedding-8b-4bit-dwq

echo ""
echo "Microbatch demo complete at $(date '+%H:%M:%S')."
