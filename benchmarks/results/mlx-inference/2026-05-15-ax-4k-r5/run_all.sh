#!/usr/bin/env bash
# r5 — 4k prompt sweep for all 14 README models on v4.9.0 binaries (post
# GLM MLA prefill fix + xcrun cold-start fix). Each model captures all four
# engines (mlx_lm, mlx_swift_lm, ax_engine_mlx direct, ax_engine_mlx_ngram_accel)
# fresh at prompt=4096, generation=128, 5 reps × 15s cooldown.
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

OUTDIR="benchmarks/results/mlx-inference/2026-05-15-ax-4k-r5"
SCRIPT="scripts/bench_mlx_inference_stack.py"
SWIFT_BENCH="scripts/mlx-swift-bench/.build/release/mlx-swift-bench"
SWIFT_CMD="$SWIFT_BENCH --model {model} --prompt-token-ids {prompt_token_ids_path} --generation-tokens {generation_tokens} --trials {trials} --delay {delay} --prefill-step-size {prefill_step_size}"

mkdir -p "$OUTDIR/logs"

_bench_once() {
    local slug="$1"
    local model_dir="$2"
    local log_path="$3"
    PYTHONUNBUFFERED=1 python3 "$SCRIPT" \
        --model-dir "$model_dir" \
        --prompt-tokens 4096 \
        --generation-tokens 128 \
        --repetitions 5 \
        --cooldown 15 \
        --ax-compare-policies \
        --no-build-ax-engine \
        --mlx-swift-lm-command "$SWIFT_CMD" \
        --output "$OUTDIR/${slug}.json" \
        2>&1 | tee -a "$log_path"
    return "${PIPESTATUS[0]}"
}

run_model() {
    local slug="$1"
    local model_dir="$2"
    local cool_after="${3:-60}"

    echo ""
    echo "=========================================="
    echo "  $slug  ($(date '+%H:%M:%S'))"
    echo "=========================================="

    local log_path="$OUTDIR/logs/${slug}.log"
    : > "$log_path"

    cleanup_servers
    if ! _bench_once "$slug" "$model_dir" "$log_path"; then
        echo "  [retry] first attempt failed, retrying once after 30s cleanup" | tee -a "$log_path"
        cleanup_servers
        sleep 30
        if ! _bench_once "$slug" "$model_dir" "$log_path"; then
            echo "  [skip] $slug failed twice — moving on" | tee -a "$log_path"
            sleep "$cool_after"
            return 0
        fi
    fi

    echo "  [done] $slug saved" | tee -a "$log_path"
    echo "  [cooling ${cool_after}s before next model]" | tee -a "$log_path"
    sleep "$cool_after"
}

# [smoke captured] run_model gemma-4-e2b-it-4bit          .internal/models/gemma-4-e2b-it-4bit
run_model gemma-4-e2b-it-5bit          .internal/models/gemma-4-e2b-it-5bit
run_model gemma-4-e2b-it-6bit          .internal/models/gemma-4-e2b-it-6bit
run_model gemma-4-e2b-it-8bit          .internal/models/gemma-4-e2b-it-8bit
run_model gemma-4-e4b-it-4bit          .internal/models/gemma-4-e4b-it-4bit
run_model gemma-4-26b-a4b-it-4bit      .internal/models/gemma-4-26b-a4b-it-4bit  90
run_model gemma-4-31b-it-4bit          .internal/models/gemma-4-31b-it-4bit       90
run_model qwen3_5-9b-mlx-4bit          .internal/models/Qwen3.5-9B-MLX-4bit
run_model qwen3_6-35b-a3b-ud-mlx-4bit  .internal/models/Qwen3.6-35B-A3B-UD-MLX-4bit 90
run_model qwen3_6-35b-a3b-5bit         .internal/models/Qwen3.6-35B-A3B-5bit       90
run_model qwen3_6-35b-a3b-6bit         .internal/models/Qwen3.6-35B-A3B-6bit       90
run_model qwen3_6-35b-a3b-8bit         .internal/models/Qwen3.6-35B-A3B-8bit       90
run_model qwen3-coder-next-4bit        .internal/models/Qwen3-Coder-Next-4bit      90
run_model glm-4.7-flash-4bit           .internal/models/GLM-4.7-Flash-4bit         90

echo ""
echo "All 14 models 4k sweep done at $(date '+%H:%M:%S')."
