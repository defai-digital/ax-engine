#!/usr/bin/env bash
set -euo pipefail

# Verify the bootstrap-deferral TTFT fix by re-benchmarking the worst TTFT
# models (Gemma E4B and E2B 4-bit at 512 tokens).  Reuses mlx_lm / mlx_swift_lm
# rows from the sliding-window fix-verify run (which already had the mask fix).

OUTDIR="benchmarks/results/mlx-inference/2026-05-12-bootstrap-defer-fix-verify"
REFDIR="benchmarks/results/mlx-inference/2026-05-12-sliding-window-fix-verify"
SCRIPT="scripts/bench_mlx_inference_stack.py"

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
        --ax-compare-policies \
        --reuse-reference-results-from "$REFDIR/${slug}.json" \
        --output "$OUTDIR/${slug}.json" \
        2>&1 | tee "$OUTDIR/logs/${slug}.log"

    echo ""
    echo "  [cooling 45s before next model]"
    sleep 45
}

# E4B: worst TTFT offender
run_model gemma-4-e4b-it-4bit .internal/models/gemma-4-e4b-it-4bit

# E2B 4-bit
run_model gemma-4-e2b-it-4bit .internal/models/gemma-4-e2b-it-4bit

echo ""
echo "Bootstrap-defer fix-verify run complete."
