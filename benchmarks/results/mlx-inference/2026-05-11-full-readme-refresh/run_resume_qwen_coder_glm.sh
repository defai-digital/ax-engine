#!/usr/bin/env bash
set -euo pipefail

OUTDIR="benchmarks/results/mlx-inference/2026-05-11-full-readme-refresh"
SCRIPT="scripts/bench_mlx_inference_stack.py"
SWIFT_BENCH="scripts/mlx-swift-bench/.build/release/mlx-swift-bench"
SWIFT_CMD="$SWIFT_BENCH --model {model} --prompt-token-ids {prompt_token_ids_path} --generation-tokens {generation_tokens} --trials {trials} --delay {delay} --prefill-step-size {prefill_step_size}"

run_model() {
    local slug="$1"
    local model_dir="$2"
    echo ""
    echo "=========================================="
    echo "  $slug"
    echo "=========================================="
    python3 "$SCRIPT" \
        --model-dir "$model_dir" \
        --prompt-tokens 128,512 \
        --generation-tokens 128 \
        --repetitions 5 \
        --cooldown 10 \
        --mlx-swift-lm-command "$SWIFT_CMD" \
        --ax-compare-policies \
        --no-build-ax-engine \
        --output "$OUTDIR/${slug}.json" \
        2>&1 | tee "$OUTDIR/logs/${slug}.resume.log"
}

run_model qwen3-coder-next-4bit .internal/models/Qwen3-Coder-Next-4bit-sanitized-2026-05-11
run_model glm-4.7-flash-4bit .internal/models/GLM-4.7-Flash-4bit

echo ""
echo "Resume models done."
