#!/usr/bin/env bash
# Full fresh refresh — re-runs every row in the README's generation-model
# tables (mlx_lm, mlx_swift_lm, AX direct, AX n-gram) for all 14 models.
# Cooldown was bumped from the 2026-05-12 refresh: 20 s between trials
# (vs 10/15 s), 60–90 s between models (vs 45/60 s). The longer cooldowns
# give the M-series GPU more thermal margin so the median across 5 trials
# is more representative of sustained throughput.
#
# Detached run pattern:
#   nohup caffeinate -ims bash run_all.sh > logs/run_all.log 2>&1 < /dev/null & disown
# Total runtime estimate: 8-9 hours for 14 models.
set -uo pipefail

REPO_ROOT="/Users/akiralam/code/ax-engine-v4"
cd "$REPO_ROOT"

OUTDIR="benchmarks/results/mlx-inference/2026-05-13-full-fresh"
SCRIPT="scripts/bench_mlx_inference_stack.py"
README_UPDATER="scripts/update_readme_from_bench.py"
SWIFT_BENCH="scripts/mlx-swift-bench/.build/release/mlx-swift-bench"
SWIFT_CMD="$SWIFT_BENCH --model {model} --prompt-token-ids {prompt_token_ids_path} --generation-tokens {generation_tokens} --trials {trials} --delay {delay} --prefill-step-size {prefill_step_size}"

mkdir -p "$OUTDIR/logs"

cleanup_servers() {
    pkill -9 -f 'ax-engine-server' >/dev/null 2>&1 || true
    pkill -9 -f 'mlx_lm.benchmark' >/dev/null 2>&1 || true
    pkill -9 -f 'mlx-swift-bench' >/dev/null 2>&1 || true
    sleep 3
}
trap cleanup_servers EXIT

_bench_once() {
    local slug="$1" model_dir="$2" log_path="$3"
    PYTHONUNBUFFERED=1 python3 -u "$SCRIPT" \
        --model-dir "$model_dir" \
        --prompt-tokens 128,512 \
        --generation-tokens 128 \
        --repetitions 5 \
        --cooldown 20 \
        --mlx-swift-lm-command "$SWIFT_CMD" \
        --ax-compare-policies \
        --no-build-ax-engine \
        --output "$OUTDIR/${slug}.json" \
        2>&1 | tee -a "$log_path"
    return "${PIPESTATUS[0]}"
}

run_model() {
    local slug="$1" model_dir="$2" cool_after="${3:-60}"

    echo ""
    echo "=========================================="
    echo "  $slug  ($(date '+%H:%M:%S'))"
    echo "=========================================="

    local log_path="$OUTDIR/logs/${slug}.log"
    : > "$log_path"

    cleanup_servers
    if ! _bench_once "$slug" "$model_dir" "$log_path"; then
        echo "  [retry] first attempt failed, retrying after 60 s cleanup" | tee -a "$log_path"
        cleanup_servers
        sleep 60
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

    echo "  [cooling ${cool_after} s before next model]" | tee -a "$log_path"
    sleep "$cool_after"
}

# Small Gemma (60 s cooling) — 5 models
run_model gemma-4-e2b-it-4bit          .internal/models/gemma-4-e2b-it-4bit          60
run_model gemma-4-e2b-it-5bit          .internal/models/gemma-4-e2b-it-5bit          60
run_model gemma-4-e2b-it-6bit          .internal/models/gemma-4-e2b-it-6bit          60
run_model gemma-4-e2b-it-8bit          .internal/models/gemma-4-e2b-it-8bit          60
run_model gemma-4-e4b-it-4bit          .internal/models/gemma-4-e4b-it-4bit          60
# Mid Gemma (90 s cooling)
run_model gemma-4-26b-a4b-it-4bit      .internal/models/gemma-4-26b-a4b-it-4bit      90
run_model gemma-4-31b-it-4bit          .internal/models/gemma-4-31b-it-4bit          90
# Qwen 3.5 9B hybrid
run_model qwen3_5-9b-mlx-4bit          .internal/models/Qwen3.5-9B-MLX-4bit          60
# Qwen 3.6 35B MoE — 4 quant variants (90 s cooling, large weights)
run_model qwen3_6-35b-a3b-ud-mlx-4bit  .internal/models/Qwen3.6-35B-A3B-UD-MLX-4bit  90
run_model qwen3_6-35b-a3b-5bit         .internal/models/Qwen3.6-35B-A3B-5bit         90
run_model qwen3_6-35b-a3b-6bit         .internal/models/Qwen3.6-35B-A3B-6bit         90
run_model qwen3_6-35b-a3b-8bit         .internal/models/Qwen3.6-35B-A3B-8bit         90
# Coder Next + GLM (90 s cooling)
run_model qwen3-coder-next-4bit        .internal/models/Qwen3-Coder-Next-4bit        90
run_model glm-4.7-flash-4bit           .internal/models/GLM-4.7-Flash-4bit           90

echo ""
echo "All 14 models full fresh refresh done at $(date '+%H:%M:%S')."
