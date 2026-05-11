#!/usr/bin/env bash
set -euo pipefail

OUTDIR="benchmarks/results/mlx-inference/2026-05-10-full-readme-rerun-current"
SCRIPT="scripts/bench_mlx_inference_stack.py"
SWIFT_BENCH="./scripts/mlx-swift-bench/.build/release/mlx-swift-bench"
SWIFT_COMMAND="${SWIFT_BENCH} --model {model} --prompt-token-ids {prompt_token_ids_path} --generation-tokens {generation_tokens} --trials {trials} --delay {delay} --prefill-step-size {prefill_step_size}"

mkdir -p "${OUTDIR}/logs"

run_model() {
    local slug="$1"
    local model_dir="$2"
    local log_path="${OUTDIR}/logs/${slug}.log"

    printf '\n==========================================\n'
    printf '  %s\n' "$slug"
    printf '==========================================\n'
    python3 "$SCRIPT" \
        --model-dir "$model_dir" \
        --prompt-tokens 128,512 \
        --generation-tokens 128 \
        --repetitions 3 \
        --cooldown 5 \
        --no-build-ax-engine \
        --mlx-swift-lm-command "$SWIFT_COMMAND" \
        --ax-compare-policies \
        --output "${OUTDIR}/${slug}.json" \
        2>&1 | tee "$log_path"
    sleep 10
}

run_model gemma-4-e2b-it-4bit          .internal/models/gemma-4-e2b-it-4bit
run_model gemma-4-e2b-it-5bit          .internal/models/gemma-4-e2b-it-5bit
run_model gemma-4-e2b-it-6bit          .internal/models/gemma-4-e2b-it-6bit
run_model gemma-4-e2b-it-8bit          .internal/models/gemma-4-e2b-it-8bit
run_model gemma-4-e4b-it-4bit          .internal/models/gemma-4-e4b-it-4bit
run_model gemma-4-26b-a4b-it-4bit      .internal/models/gemma-4-26b-a4b-it-4bit
run_model gemma-4-31b-it-4bit          .internal/models/gemma-4-31b-it-4bit
run_model qwen3_5-9b-mlx-4bit          .internal/models/Qwen3.5-9B-MLX-4bit
run_model qwen3_6-35b-a3b-ud-mlx-4bit  .internal/models/Qwen3.6-35B-A3B-UD-MLX-4bit
run_model qwen3_6-35b-a3b-5bit         .internal/models/Qwen3.6-35B-A3B-5bit
run_model qwen3_6-35b-a3b-6bit         .internal/models/Qwen3.6-35B-A3B-6bit
run_model qwen3_6-35b-a3b-8bit         .internal/models/Qwen3.6-35B-A3B-8bit
run_model qwen3-coder-next-4bit        .internal/models/Qwen3-Coder-Next-4bit
run_model glm-4.7-flash-4bit           .internal/models/GLM-4.7-Flash-4bit

printf '\nAll models done.\n'
