#!/usr/bin/env bash
# Prefill-stage profile baseline: capture ax_mlx_prefill_profile_* per-stage
# breakdown for all 14 README models at prompt=4096, direct-mode AX only.
# Each model: 3 reps, 15 s cooldown, chunk=2048, max_batch=4096 (matches
# the chunk geometry of the 4k investigation A/B).
#
# Output: one JSON per model + a run_all.log. The captured artifact carries
# both the standard bench rows (mlx_lm + mlx_swift + ax direct) AND the new
# `ax_mlx_prefill_profile` block under each trial, which the renderer
# (`scripts/render_mlx_prefill_profile_report.py`) converts to a markdown
# table for the PRD §11 ranking.
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

run_model gemma-4-e2b-it-4bit          .internal/models/gemma-4-e2b-it-4bit
run_model gemma-4-e2b-it-5bit          .internal/models/gemma-4-e2b-it-5bit
run_model gemma-4-e2b-it-6bit          .internal/models/gemma-4-e2b-it-6bit
run_model gemma-4-e2b-it-8bit          .internal/models/gemma-4-e2b-it-8bit
run_model gemma-4-e4b-it-4bit          .internal/models/gemma-4-e4b-it-4bit
run_model gemma-4-26b-a4b-it-4bit      .internal/models/gemma-4-26b-a4b-it-4bit  60
run_model gemma-4-31b-it-4bit          .internal/models/gemma-4-31b-it-4bit       60
run_model qwen3_5-9b-mlx-4bit          .internal/models/Qwen3.5-9B-MLX-4bit
run_model qwen3_6-35b-a3b-ud-mlx-4bit  .internal/models/Qwen3.6-35B-A3B-UD-MLX-4bit 60
run_model qwen3_6-35b-a3b-5bit         .internal/models/Qwen3.6-35B-A3B-5bit       60
run_model qwen3_6-35b-a3b-6bit         .internal/models/Qwen3.6-35B-A3B-6bit       60
run_model qwen3_6-35b-a3b-8bit         .internal/models/Qwen3.6-35B-A3B-8bit       60
run_model qwen3-coder-next-4bit        .internal/models/Qwen3-Coder-Next-4bit      60
run_model glm-4.7-flash-4bit           .internal/models/GLM-4.7-Flash-4bit         60

echo ""
echo "All 14 models prefill-profile baseline done at $(date '+%H:%M:%S')."
