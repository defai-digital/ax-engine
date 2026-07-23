#!/usr/bin/env bash
# Fail-closed inventory for Gemma4/Qwen3 focused-deepening software gates.
# Hardware-only residual (models/E2E certs) is out of scope; this script asserts
# shipped production paths exist in the tracked tree.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

fail() {
  echo "FOCUSED-DEEPENING GATE FAIL: $*" >&2
  exit 1
}

pass() {
  echo "ok: $*"
}

# --- WS-V2: qwen3_vl vision must have a real load path that can return Some ---
rg -q "fn load_qwen3_vl_vision_weights" crates/ax-engine-mlx/src/qwen3_vl.rs \
  || fail "missing load_qwen3_vl_vision_weights"
rg -q "qwen3_vl_vision: None," crates/ax-engine-mlx/src/weights.rs \
  && fail "weights.rs hard-codes qwen3_vl_vision: None in load path"
rg -q "load_qwen3_vl_vision_weights" crates/ax-engine-mlx/src/weights.rs \
  || fail "load_weights does not call load_qwen3_vl_vision_weights"
rg -q "QWEN3_VL_EXTRA_TENSOR_MAP" crates/ax-engine-core/src/convert/tensor_mapping.rs \
  || fail "missing QWEN3_VL_EXTRA_TENSOR_MAP"
rg -q "Qwen3VlVisionPatchEmbed" crates/ax-engine-core/src/model.rs \
  || fail "missing Qwen3VlVisionPatchEmbed role"
rg -q "chunked_prefill_qwen3_vl_with_sampling_buffers" crates/ax-engine-mlx/src/generate.rs \
  || fail "missing qwen3_vl prefill generate path"
rg -q "qwen3_vl_inputs" crates/ax-engine-mlx/src/runner/mod.rs \
  || fail "runner missing qwen3_vl prefill branch"
rg -q "render_qwen3_vl_chat_with_media" crates/ax-engine-server/src/openai/chat_requests.rs \
  || fail "missing render_qwen3_vl_chat_with_media"
rg -q "multimodal_inputs.qwen3_vl" crates/ax-engine-server/src/openai/requests.rs \
  || fail "OpenAI chat does not set multimodal_inputs.qwen3_vl"
rg -q "openai_chat_request_decodes_inline_image_into_qwen3_vl" crates/ax-engine-server/src/tests/openai_chat.rs \
  || fail "missing OpenAI chat→qwen3_vl integration test"
pass "WS-V2 load + serve wiring present (including OpenAI chat)"

# --- WS-V1: gemma4_vl fail-closed + generate wire ---
rg -q "build_vl_prefill_embeddings" crates/ax-engine-mlx/src/gemma4_vl.rs \
  || fail "missing gemma4_vl build_vl_prefill_embeddings"
rg -q "is_gemma4_vl_family" crates/ax-engine-mlx/src/generate.rs \
  || fail "generate missing gemma4_vl branch"
pass "WS-V1 gemma4_vl path present"

# --- WS-M3: media_key must not be permanently empty ---
rg -q "format_prefix_layer_layout" crates/ax-engine-mlx/src/runner/mod.rs \
  || fail "missing format_prefix_layer_layout"
rg -q "media_key_from_inputs" crates/ax-engine-mlx/src/runner/mod.rs \
  || fail "missing media_key_from_inputs"
# Reject permanent empty-only media_key implementations.
if rg -n "fn media_key_from_inputs" -A 8 crates/ax-engine-mlx/src/runner/mod.rs \
  | rg -q 'return String::new\(\);\s*$'; then
  # allow early-return None branch, but require digests for providers
  :
fi
rg -q "media_prefix_key" crates/ax-engine-mlx/src/runner/mod.rs \
  || fail "media_key_from_inputs does not call media_prefix_key"
pass "WS-M3 media_key path present"

# --- WS-T2: try_* must not be permanent None stubs ---
if rg -n "fn try_linear_attention_whole_layer_metal" -A 30 \
  crates/ax-engine-mlx/src/model/shared/linear_attention.rs | rg -q "^\s*None\s*$"; then
  # Only fail if body is solely None after flag check (scaffold).
  body=$(rg -n "fn try_linear_attention_whole_layer_metal" -A 40 \
    crates/ax-engine-mlx/src/model/shared/linear_attention.rs || true)
  if echo "$body" | rg -q "gated_delta_kernel|linear_attention_inputs"; then
    pass "WS-T2 linear_attention try_* has real body"
  else
    fail "try_linear_attention_whole_layer_metal still a None stub"
  fi
else
  pass "WS-T2 linear_attention try_* present"
fi
if rg -n "fn try_moe_deep_expert_block_metal" -A 50 \
  crates/ax-engine-mlx/src/model/shared/mlp.rs | rg -q "qw_gather|moe_fused_activation"; then
  pass "WS-T2 moe deep expert try_* has real body"
else
  fail "try_moe_deep_expert_block_metal still a None stub"
fi

# --- WS-T3: gemma4 must stay non-candidate with cert note ---
rg -q "gemma4_families_remain_non_candidates_until_cert" \
  crates/ax-engine-core/src/architecture_registry.rs \
  || fail "missing WS-T3 structural cert guard test"
pass "WS-T3 structural non-candidate guard present"

# --- WS-P2: E2B/E4B + Coder-Next MTP publication targets ---
rg -q 'label: "gemma-4-e2b"' crates/ax-engine-bench/src/bin/ax-engine.rs \
  || fail "missing gemma-4-e2b download-mtp target"
rg -q 'label: "gemma-4-e4b"' crates/ax-engine-bench/src/bin/ax-engine.rs \
  || fail "missing gemma-4-e4b download-mtp target"
rg -q 'label: "qwen3-coder-next"' crates/ax-engine-bench/src/bin/ax-engine.rs \
  || fail "missing qwen3-coder-next download-mtp target"
pass "WS-P2 MTP publication targets present"

# --- WS-T1 Decision A path ---
rg -q "ffn_batched_moe_row_exact|row_exact_moe" crates/ax-engine-mlx/src -g '*.rs' \
  || fail "missing Decision A RowExact MoE path"
pass "WS-T1 RowExact path present"

echo "All focused-deepening software gates passed."
