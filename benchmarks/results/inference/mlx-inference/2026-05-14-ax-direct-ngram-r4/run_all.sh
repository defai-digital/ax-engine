#!/usr/bin/env bash
# r4 AX-only sweep on 2026-05-14 evening binaries (post F3 M1-M5 disk prefix
# cache series + F4 MLA drift + TurboQuant tuning). Reuse mlx_lm / mlx_swift_lm
# rows from 2026-05-13-full-fresh (same host, same prompt SHA). Smoke test
# already produced gemma-4-e2b-it-4bit, so skip that model in the loop.
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

OUTDIR="benchmarks/results/mlx-inference/2026-05-14-ax-direct-ngram-r4"
REUSE_DIR="benchmarks/results/mlx-inference/2026-05-13-full-fresh"
SCRIPT="scripts/bench_mlx_inference_stack.py"
README_UPDATER="scripts/update_readme_from_bench.py"

mkdir -p "$OUTDIR/logs"

_bench_once() {
    local slug="$1"
    local model_dir="$2"
    local log_path="$3"
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

# Apply smoke-captured gemma-4-e2b-it-4bit cells first (no rerun)
echo ""
echo "=========================================="
echo "  gemma-4-e2b-it-4bit (smoke, already captured)  ($(date '+%H:%M:%S'))"
echo "=========================================="
python3 "$README_UPDATER" \
    --slug gemma-4-e2b-it-4bit \
    --json "$OUTDIR/gemma-4-e2b-it-4bit.json" 2>&1 | tee "$OUTDIR/logs/gemma-4-e2b-it-4bit.log"
sleep 30

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
echo "All 14 models AX direct + n-gram r4 sweep done at $(date '+%H:%M:%S')."
