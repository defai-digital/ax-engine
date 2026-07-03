#!/usr/bin/env bash
# K spike — SwiGLU FFN compile-wrap on the 2 Qwen targets the user asked
# about. Compare W1 baseline (FFN_COMPILE=1, SWIGLU=0) vs K (both flags on).
set -uo pipefail
REPO_ROOT="/Users/akiralam/code/ax-engine-v4"
cd "$REPO_ROOT"

cleanup_servers() {
    pkill -f 'ax-engine-server' >/dev/null 2>&1 || true
    pkill -f 'mlx_lm.benchmark' >/dev/null 2>&1 || true
    pkill -f 'mlx-swift-bench' >/dev/null 2>&1 || true
    sleep 2
}
trap cleanup_servers EXIT

OUTDIR="benchmarks/results/mlx-inference/2026-05-15-k-spike-swiglu-qwen"
REUSE_DIR="benchmarks/results/mlx-inference/2026-05-13-full-fresh"
SCRIPT="scripts/bench_mlx_inference_stack.py"

run_model() {
    local slug="$1"; local model_dir="$2"; local cool_after="${3:-45}"
    echo ""; echo "==========================================";
    echo "  $slug  ($(date '+%H:%M:%S'))"
    echo "=========================================="
    local log_path="$OUTDIR/logs/${slug}.log"
    : > "$log_path"
    cleanup_servers
    AX_MLX_PREFILL_FFN_COMPILE=1 AX_MLX_PREFILL_FFN_COMPILE_SWIGLU=1 \
    PYTHONUNBUFFERED=1 python3 "$SCRIPT" \
        --model-dir "$model_dir" \
        --prompt-tokens 128,512 \
        --generation-tokens 128 \
        --repetitions 5 \
        --cooldown 15 \
        --ax-compare-policies \
        --no-build-ax-engine \
        --reuse-reference-results-from "$REUSE_DIR/${slug}.json" \
        --output "$OUTDIR/${slug}.json" \
        2>&1 | tee -a "$log_path"
    sleep "$cool_after"
}

run_model qwen3_5-9b-mlx-4bit          .internal/models/Qwen3.5-9B-MLX-4bit
run_model qwen3-coder-next-4bit        .internal/models/Qwen3-Coder-Next-4bit      60

echo ""; echo "K spike complete at $(date '+%H:%M:%S')."
