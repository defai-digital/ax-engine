#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"
TMP_DIR="$(ax_tmp_dir ax-turboquant-quality-gate)"

cleanup() {
    ax_rm_rf "$TMP_DIR" "$ROOT_DIR/scripts/__pycache__"
}

trap cleanup EXIT

cd "$ROOT_DIR"

mkdir -p "$TMP_DIR/benchmarks/manifests/scenario"

cat > "$TMP_DIR/benchmarks/manifests/scenario/long_context_qwen_8k.json" <<'JSON'
{
  "schema_version": "ax.engine_bench.manifest.v1",
  "id": "long_context_qwen_8k"
}
JSON

cat > "$TMP_DIR/baseline-outputs.json" <<'JSON'
{
  "decode_outputs": [
    [1.0, 2.0, 3.0],
    [0.0, 1.0, 0.0]
  ]
}
JSON

cat > "$TMP_DIR/candidate-outputs.json" <<'JSON'
{
  "decode_outputs": [
    [1.0, 2.01, 2.99],
    [0.0, 1.0, 0.0]
  ]
}
JSON

"$PYTHON_BIN" scripts/build_turboquant_quality_metrics.py \
  --baseline-outputs "$TMP_DIR/baseline-outputs.json" \
  --candidate-outputs "$TMP_DIR/candidate-outputs.json" \
  --output "$TMP_DIR/quality-metrics.json"

cat > "$TMP_DIR/baseline-benchmark.json" <<'JSON'
{
  "schema_version": "ax.mlx_inference_stack.v2",
  "repetitions": 3,
  "cooldown": 20.0,
  "results": [
    {
      "engine": "ax_engine_mlx",
      "prompt_tokens": 8192,
      "generation_tokens": 256,
      "prompt_token_ids_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "decode_tok_s": {"median": 100.0}
    }
  ]
}
JSON

cat > "$TMP_DIR/candidate-benchmark.json" <<'JSON'
{
  "schema_version": "ax.mlx_inference_stack.v2",
  "repetitions": 3,
  "cooldown": 20.0,
  "results": [
    {
      "engine": "ax_engine_mlx",
      "prompt_tokens": 8192,
      "generation_tokens": 256,
      "prompt_token_ids_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "decode_tok_s": {"median": 90.0},
      "experimental_mlx_kv_compression": "turboquant-fused-experimental",
      "kv_compression_decode_path": "fused_compressed_decode",
      "kv_compression_telemetry": {
        "ax_mlx_kv_compression_route_metadata_schema": 2,
        "ax_mlx_kv_compression_production_ready": 0,
        "ax_mlx_kv_compression_production_blockers": 1,
        "ax_mlx_kv_compression_preset": 1,
        "ax_mlx_kv_compression_key_bits": 8,
        "ax_mlx_kv_compression_value_bits": 4,
        "ax_mlx_kv_compression_eligible_layers": 20,
        "ax_mlx_kv_compression_candidate_token_layers": 120000,
        "ax_mlx_kv_compression_estimated_saved_kib": 4096,
        "ax_mlx_kv_compression_runtime_storage_written_slots": 5000,
        "ax_mlx_kv_compression_decode_path": 2,
        "ax_mlx_kv_compression_fused_decode_candidates": 1,
        "ax_mlx_kv_compression_fused_decode_attempts": 1,
        "ax_mlx_kv_compression_fused_decode_successes": 1,
        "ax_mlx_kv_compression_fused_decode_metal_successes": 1,
        "ax_mlx_kv_compression_fused_decode_fallbacks": 0,
        "ax_mlx_kv_compression_fused_decode_fallback_reason": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_prefill_only": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_attention_kind": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_linear_attention": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_glm_mla": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_sliding_window": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_kv_shared": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_ineligible_layer": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_unsupported_preset": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_unsupported_head_dim": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_gqa": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_missing_storage": 0
      }
    }
  ]
}
JSON

"$PYTHON_BIN" scripts/build_turboquant_quality_artifact.py \
  --baseline-benchmark "$TMP_DIR/baseline-benchmark.json" \
  --candidate-benchmark "$TMP_DIR/candidate-benchmark.json" \
  --quality-metrics "$TMP_DIR/quality-metrics.json" \
  --output "$TMP_DIR/quality-gate.json" \
  --manifest "$TMP_DIR/benchmarks/manifests/scenario/long_context_qwen_8k.json" \
  --model-id qwen3_5_9b_q4 \
  --model-family qwen3_dense \
  --model-revision synthetic \
  --root "$TMP_DIR"

"$PYTHON_BIN" scripts/check_turboquant_quality_artifact.py \
  --root "$TMP_DIR" \
  "$TMP_DIR/quality-gate.json"

cat > "$TMP_DIR/shadow-candidate-benchmark.json" <<'JSON'
{
  "schema_version": "ax.mlx_inference_stack.v2",
  "repetitions": 3,
  "cooldown": 20.0,
  "results": [
    {
      "engine": "ax_engine_mlx",
      "prompt_tokens": 8192,
      "generation_tokens": 256,
      "prompt_token_ids_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "decode_tok_s": {"median": 90.0},
      "experimental_mlx_kv_compression": "turboquant-shadow",
      "kv_compression_decode_path": "full_precision_shadow",
      "kv_compression_telemetry": {
        "ax_mlx_kv_compression_route_metadata_schema": 2,
        "ax_mlx_kv_compression_production_ready": 0,
        "ax_mlx_kv_compression_production_blockers": 1,
        "ax_mlx_kv_compression_preset": 1,
        "ax_mlx_kv_compression_key_bits": 8,
        "ax_mlx_kv_compression_value_bits": 4,
        "ax_mlx_kv_compression_eligible_layers": 20,
        "ax_mlx_kv_compression_candidate_token_layers": 120000,
        "ax_mlx_kv_compression_estimated_saved_kib": 4096,
        "ax_mlx_kv_compression_runtime_storage_written_slots": 5000,
        "ax_mlx_kv_compression_decode_path": 1,
        "ax_mlx_kv_compression_fused_decode_candidates": 1,
        "ax_mlx_kv_compression_fused_decode_attempts": 0,
        "ax_mlx_kv_compression_fused_decode_successes": 0,
        "ax_mlx_kv_compression_fused_decode_metal_successes": 0,
        "ax_mlx_kv_compression_fused_decode_fallbacks": 0,
        "ax_mlx_kv_compression_fused_decode_fallback_reason": 1,
        "ax_mlx_kv_compression_fused_decode_blocked_prefill_only": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_attention_kind": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_linear_attention": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_glm_mla": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_sliding_window": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_kv_shared": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_ineligible_layer": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_unsupported_preset": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_unsupported_head_dim": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_gqa": 0,
        "ax_mlx_kv_compression_fused_decode_blocked_missing_storage": 0
      }
    }
  ]
}
JSON

if "$PYTHON_BIN" scripts/build_turboquant_quality_artifact.py \
  --baseline-benchmark "$TMP_DIR/baseline-benchmark.json" \
  --candidate-benchmark "$TMP_DIR/shadow-candidate-benchmark.json" \
  --quality-metrics "$TMP_DIR/quality-metrics.json" \
  --output "$TMP_DIR/shadow-quality-gate.json" \
  --manifest "$TMP_DIR/benchmarks/manifests/scenario/long_context_qwen_8k.json" \
  --model-id qwen3_5_9b_q4 \
  --model-family qwen3_dense \
  --model-revision synthetic \
  --root "$TMP_DIR"; then
  echo "error: full_precision_shadow candidate unexpectedly passed quality gate" >&2
  exit 1
fi

echo "ok: TurboQuant quality gate CLI pipeline"
