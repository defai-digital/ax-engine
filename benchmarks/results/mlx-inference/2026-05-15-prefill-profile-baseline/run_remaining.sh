#!/usr/bin/env bash
# Continuation sweep: complete the 7 prefill-profile baseline JSONs that
# weren't captured in the first pass. Earlier failures on Qwen3.5 9B and
# Qwen3.6 family came from a kernel-side cap on chunked_prefill in the
# linear-attention layers (GATED_DELTA_THREADGROUP_CACHE_CAPACITY = 512
# tokens), which the runner now clamps to internally so chunk=2048 on the
# CLI is safe for every model.
set -uo pipefail

REPO_ROOT="/Users/akiralam/code/ax-engine-v4"
cd "$REPO_ROOT"

cleanup_servers() {
    pkill -f 'ax-engine-server' >/dev/null 2>&1 || true
    pkill -f 'mlx_lm.benchmark' >/dev/null 2>&1 || true
    sleep 2
}
trap cleanup_servers EXIT

OUTDIR="benchmarks/results/mlx-inference/2026-05-15-prefill-profile-baseline"
SCRIPT="scripts/bench_mlx_inference_stack.py"

mkdir -p "$OUTDIR/logs"

_bench_once() {
    local slug="$1"
    local model_dir="$2"
    local log_path="$3"
    PYTHONUNBUFFERED=1 python3 "$SCRIPT" \
        --model-dir "$model_dir" \
        --prompt-tokens 4096 \
        --generation-tokens 128 \
        --repetitions 3 \
        --cooldown 15 \
        --ax-direct \
        --ax-prefill-profile \
        --skip-mlx-lm \
        --no-build-ax-engine \
        --output "$OUTDIR/${slug}.json" \
        2>&1 | tee -a "$log_path"
    return "${PIPESTATUS[0]}"
}

run_model() {
    local slug="$1"
    local model_dir="$2"
    local cool_after="${3:-30}"

    echo ""
    echo "=========================================="
    echo "  $slug  ($(date '+%H:%M:%S'))"
    echo "=========================================="

    local log_path="$OUTDIR/logs/${slug}.log"
    : > "$log_path"

    cleanup_servers
    if ! _bench_once "$slug" "$model_dir" "$log_path"; then
        echo "  [retry] first attempt failed, retrying once after 20s cleanup" | tee -a "$log_path"
        cleanup_servers
        sleep 20
        if ! _bench_once "$slug" "$model_dir" "$log_path"; then
            echo "  [skip] $slug failed twice — moving on" | tee -a "$log_path"
            sleep "$cool_after"
            return 0
        fi
    fi

    echo "  [done] $slug saved" | tee -a "$log_path"
    sleep "$cool_after"
}

# 7 remaining models: 1 GLM + 6 linear-attention (Qwen 3.5 / Qwen 3.6 / Coder Next).
run_model qwen3_5-9b-mlx-4bit          .internal/models/Qwen3.5-9B-MLX-4bit
run_model qwen3_6-35b-a3b-ud-mlx-4bit  .internal/models/Qwen3.6-35B-A3B-UD-MLX-4bit 60
run_model qwen3_6-35b-a3b-5bit         .internal/models/Qwen3.6-35B-A3B-5bit       60
run_model qwen3_6-35b-a3b-6bit         .internal/models/Qwen3.6-35B-A3B-6bit       60
run_model qwen3_6-35b-a3b-8bit         .internal/models/Qwen3.6-35B-A3B-8bit       60
run_model qwen3-coder-next-4bit        .internal/models/Qwen3-Coder-Next-4bit      60
run_model glm-4.7-flash-4bit           .internal/models/GLM-4.7-Flash-4bit         60

echo ""
echo "Continuation prefill-profile baseline done at $(date '+%H:%M:%S')."
