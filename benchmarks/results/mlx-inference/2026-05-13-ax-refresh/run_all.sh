#!/usr/bin/env bash
# Re-run AX direct + n-gram rows for all 14 README generation models,
# reusing the 2026-05-12 full-refresh mlx_lm + mlx_swift_lm reference
# rows (the prompt corpus is identical, the reference path code hasn't
# changed, and re-measuring them would take another 4 hours for no
# information gain). Each model invokes `--ax-compare-policies` which
# emits both `ax_engine_mlx` (direct, n-gram disabled) and
# `ax_engine_mlx_ngram_accel` (default n-gram policy) rows.
#
# Driver pattern matches earlier refresh scripts:
#  - `--repetitions 5 --cooldown 15` (15 s between trials)
#  - 45 s between small models, 60 s between large models (thermal)
#  - Per-model log under logs/, JSON beside it
#  - `update_readme_from_bench.py` writes the 6 README row cells for
#    each model immediately after the JSON lands, so progress is
#    visible without waiting for the whole run to finish.
#
# Designed to be detached (`nohup caffeinate -ims bash ... & disown`)
# so a Claude session timeout / shell HUP doesn't kill the run; logs
# go to `logs/run_all.log`.
set -uo pipefail

REPO_ROOT="/Users/akiralam/code/ax-engine-v4"
cd "$REPO_ROOT"

OUTDIR="benchmarks/results/mlx-inference/2026-05-13-ax-refresh"
REFDIR="benchmarks/results/mlx-inference/2026-05-12-full-fresh-readme-refresh"
SCRIPT="scripts/bench_mlx_inference_stack.py"
README_UPDATER="scripts/update_readme_from_bench.py"
# Note: when --reuse-reference-results-from is set, the bench script
# refuses --mlx-swift-lm-command (the reference rows come from the
# reused JSON, not a fresh swift invocation). Both mlx_lm + mlx_swift_lm
# numbers in 2026-05-12-full-fresh-readme-refresh stay intact.

mkdir -p "$OUTDIR/logs"

cleanup_servers() {
    pkill -f 'ax-engine-server' >/dev/null 2>&1 || true
    pkill -f 'mlx_lm.benchmark' >/dev/null 2>&1 || true
    pkill -f 'mlx-swift-bench' >/dev/null 2>&1 || true
    sleep 2
}
trap cleanup_servers EXIT

_bench_once() {
    local slug="$1" model_dir="$2" log_path="$3"
    PYTHONUNBUFFERED=1 python3 -u "$SCRIPT" \
        --model-dir "$model_dir" \
        --prompt-tokens 128,512 \
        --generation-tokens 128 \
        --repetitions 5 \
        --cooldown 15 \
        --ax-compare-policies \
        --reuse-reference-results-from "$REFDIR/${slug}.json" \
        --no-build-ax-engine \
        --output "$OUTDIR/${slug}.json" \
        2>&1 | tee -a "$log_path"
    return "${PIPESTATUS[0]}"
}

run_model() {
    local slug="$1" model_dir="$2" cool_after="${3:-45}"

    echo ""
    echo "=========================================="
    echo "  $slug  ($(date '+%H:%M:%S'))"
    echo "=========================================="

    local log_path="$OUTDIR/logs/${slug}.log"
    : > "$log_path"

    cleanup_servers
    if ! _bench_once "$slug" "$model_dir" "$log_path"; then
        echo "  [retry] first attempt failed, retrying once after 30 s cleanup" | tee -a "$log_path"
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

    echo "  [cooling ${cool_after} s before next model]" | tee -a "$log_path"
    sleep "$cool_after"
}

# Same 14-model order as the README's Prefill/Decode/TTFT tables.
run_model gemma-4-e2b-it-4bit          .internal/models/gemma-4-e2b-it-4bit
run_model gemma-4-e2b-it-5bit          .internal/models/gemma-4-e2b-it-5bit
run_model gemma-4-e2b-it-6bit          .internal/models/gemma-4-e2b-it-6bit
run_model gemma-4-e2b-it-8bit          .internal/models/gemma-4-e2b-it-8bit
run_model gemma-4-e4b-it-4bit          .internal/models/gemma-4-e4b-it-4bit
run_model gemma-4-26b-a4b-it-4bit      .internal/models/gemma-4-26b-a4b-it-4bit       60
run_model gemma-4-31b-it-4bit          .internal/models/gemma-4-31b-it-4bit           60
run_model qwen3_5-9b-mlx-4bit          .internal/models/Qwen3.5-9B-MLX-4bit
run_model qwen3_6-35b-a3b-ud-mlx-4bit  .internal/models/Qwen3.6-35B-A3B-UD-MLX-4bit   60
run_model qwen3_6-35b-a3b-5bit         .internal/models/Qwen3.6-35B-A3B-5bit          60
run_model qwen3_6-35b-a3b-6bit         .internal/models/Qwen3.6-35B-A3B-6bit          60
run_model qwen3_6-35b-a3b-8bit         .internal/models/Qwen3.6-35B-A3B-8bit          60
run_model qwen3-coder-next-4bit        .internal/models/Qwen3-Coder-Next-4bit         60
run_model glm-4.7-flash-4bit           .internal/models/GLM-4.7-Flash-4bit            60

echo ""
echo "All 14 models done at $(date '+%H:%M:%S')."
