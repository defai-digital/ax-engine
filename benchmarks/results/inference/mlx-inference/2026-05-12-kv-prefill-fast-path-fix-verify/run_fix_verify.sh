#!/usr/bin/env bash
set -euo pipefail

# Verify the KV cache fresh-prefill fast-path fix (skip zeros+slice_update for
# chunk-aligned prompts).  E2B shows -3.6% prefill at 512 tokens after the
# bootstrap-deferral fix; this targets the ~210 extra MLX graph nodes (6 per
# layer × 35 layers) added by zeros+slice_update+slice during initial prefill.
# Reuses mlx_lm rows from the bootstrap-defer fix-verify run.

OUTDIR="benchmarks/results/mlx-inference/2026-05-12-kv-prefill-fast-path-fix-verify"
REFDIR="benchmarks/results/mlx-inference/2026-05-12-bootstrap-defer-fix-verify"
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

# E2B: still shows -3.6% prefill at pt=512 after bootstrap-deferral fix
run_model gemma-4-e2b-it-4bit .internal/models/gemma-4-e2b-it-4bit

# E4B: should remain near parity (+0.7%) — regression check
run_model gemma-4-e4b-it-4bit .internal/models/gemma-4-e4b-it-4bit

echo ""
echo "KV cache fast-path fix-verify run complete."
