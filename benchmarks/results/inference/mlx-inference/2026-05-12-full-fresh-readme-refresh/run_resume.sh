#!/usr/bin/env bash
# Resume the fresh README refresh starting from model 2 (gemma-4-e2b-it-5bit).
# Model 1 (gemma-4-e2b-it-4bit) is already in the JSON output and README rows.
# This script intentionally avoids `set -e` so a single model's transient
# failure does not abort the multi-hour run.
set -uo pipefail

REPO_ROOT="/Users/akiralam/code/ax-engine-v4"
cd "$REPO_ROOT"

OUTDIR="benchmarks/results/mlx-inference/2026-05-12-full-fresh-readme-refresh"
EMB_OUTDIR="benchmarks/results/embedding/2026-05-12-full-fresh-readme-refresh"
SCRIPT="scripts/bench_mlx_inference_stack.py"
SWIFT_BENCH="scripts/mlx-swift-bench/.build/release/mlx-swift-bench"
SWIFT_CMD="$SWIFT_BENCH --model {model} --prompt-token-ids {prompt_token_ids_path} --generation-tokens {generation_tokens} --trials {trials} --delay {delay} --prefill-step-size {prefill_step_size}"
README_UPDATER="scripts/update_readme_from_bench.py"
EMB_README_UPDATER="scripts/update_readme_embedding.py"
EMB_BENCH="scripts/bench_embedding_models.py"

mkdir -p "$OUTDIR/logs" "$EMB_OUTDIR"

cleanup_servers() {
    pkill -f 'ax-engine-server' >/dev/null 2>&1 || true
    pkill -f 'mlx_lm.benchmark' >/dev/null 2>&1 || true
    pkill -f 'mlx-swift-bench' >/dev/null 2>&1 || true
    sleep 2
}

_bench_once() {
    local slug="$1"
    local model_dir="$2"
    local log_path="$3"
    PYTHONUNBUFFERED=1 python3 -u "$SCRIPT" \
        --model-dir "$model_dir" \
        --prompt-tokens 128,512 \
        --generation-tokens 128 \
        --repetitions 5 \
        --cooldown 15 \
        --mlx-swift-lm-command "$SWIFT_CMD" \
        --ax-compare-policies \
        --no-build-ax-engine \
        --output "$OUTDIR/${slug}.json" \
        2>&1 | tee -a "$log_path"
    return "${PIPESTATUS[0]}"
}

run_model() {
    local slug="$1"
    local model_dir="$2"
    local cool_after="${3:-45}"

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

    if [[ -f "$OUTDIR/${slug}.json" ]]; then
        echo "" | tee -a "$log_path"
        echo "  [readme] applying ${slug} rows to README.md" | tee -a "$log_path"
        python3 "$README_UPDATER" \
            --slug "$slug" \
            --json "$OUTDIR/${slug}.json" \
            2>&1 | tee -a "$log_path" || true
    fi

    echo "  [cooling ${cool_after}s before next model]" | tee -a "$log_path"
    sleep "$cool_after"
}

# Resume order: skip gemma-4-e2b-it-4bit (already complete).
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
echo "=========================================="
echo "  Generation models complete. Starting embedding bench."
echo "=========================================="

run_embedding() {
    local label="$1"
    local model_dir="$2"
    echo ""
    echo "  [embedding] $label  ($(date '+%H:%M:%S'))"
    PYTHONUNBUFFERED=1 python3 -u "$EMB_BENCH" \
        --model-dir "$model_dir" \
        --model-label "$label" \
        --trials 5 \
        --cooldown 10 \
        --skip-swift \
        --output-dir "$EMB_OUTDIR" \
        2>&1 | tee "$EMB_OUTDIR/logs-${label}.log"
    sleep 45
}

mkdir -p "$EMB_OUTDIR"
run_embedding qwen3-embedding-0.6b-8bit   .internal/models/qwen3-embedding-0.6b-8bit
run_embedding qwen3-embedding-4b-4bit     .internal/models/qwen3-embedding-4b-4bit
run_embedding qwen3-embedding-8b-4bit-dwq .internal/models/qwen3-embedding-8b-4bit-dwq

echo ""
echo "  [readme] applying embedding rows"
python3 "$EMB_README_UPDATER" --results-dir "$EMB_OUTDIR"

echo ""
echo "All done — full fresh README refresh complete at $(date '+%H:%M:%S')."
