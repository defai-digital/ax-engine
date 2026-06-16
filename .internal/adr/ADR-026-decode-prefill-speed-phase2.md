# ADR-026: Decode & Prefill Speed Optimization — Phase 2

**Date:** 2026-06-16
**Status:** Accepted
**Deciders:** AX Engine Core

## Context

Phase 1 of decode-speed optimization introduced last-position-only prefill for
standard-attention families (Gemma4, Qwen3, LLaMA3). The optimization slices the
attention-residual stream to `[1, 1, hidden]` before FFN in the last transformer layer,
saving ~50% of that layer's wall time. However, Qwen 3.5/3.6 hybrid models with
linear-attention layers do not benefit: the linear-attention forward path
(`qwen3_linear::layer_forward`) explicitly bypasses this optimization with a comment
noting it "hasn't been extended yet."

Separately, the Qwen3 MoE router uses full-width `softmax_precise` over all 128–256
experts before top-k selection. The Gemma4 router already uses a more efficient
argpartition-first pattern with `resoftmax=true`, applying softmax only to the selected
top-k subset (4 values).

Finally, the fused MoE shared-expert weighted-sum Metal kernel is gated to `seq == 1`
(decode only), missing opportunities during short prefill tail chunks.

## Decision

### D1: Extend last-position-only to linear-attention layers

**Decision:** Pass `last_position_only_after_attention` through to
`qwen3_linear::layer_forward`. Inside the linear-attention layer, after the
attention-residual add (`hidden = add(hidden, attn_proj)`), slice `hidden` to the last
position when `last_only && seq > 1`. The linear-attention state (conv1d + recurrent) has
already been written to `cache` inside `linear_attention_forward`, so the slice is safe —
subsequent layers only need the last position's hidden.

**Consequence:** On Qwen3.5 9B (24 layers, every 3rd is standard), extending last-only to
the last linear-attention layer (layer 22) saves FFN compute on that layer. More
importantly, if the layer loop in `forward_with_turboquant_context_and_logits_mode` is
extended to pass `last_only` to intermediate layers (not just the very last), all standard
layers after the last linear-attention layer would also benefit. For this change, we
enable it on the `layer_forward_with_turboquant_context_last_only` dispatch path only,
which currently only fires for the very last layer. This means linear-attention layers at
the very last position (if the last layer is linear-attention) now also get the
optimization. The bigger win of enabling it on intermediate last-linear layers is left as
follow-up work requiring layer-type-aware dispatch.

### D2: Qwen3 MoE router narrow softmax (opt-in, default OFF)

**Decision:** Add `AX_MLX_QWEN3_MOE_NARROW_SOFTMAX` env flag (default OFF). When enabled,
the router does argpartition on raw bf16 logits to find top-k indices, then applies
`softmax_precise` only to the selected top-k subset — matching the Gemma4 pattern. When
disabled, the current full-width `softmax_precise` path is preserved.

**Rationale for default OFF:** With bf16 logits and 128–256 experts, tiny round-off in
bf16 can flip argpartition rankings vs softmax-then-argpartition rankings. Validation
against mlx-lm's `precise=True` reference is required before default-on promotion.

### D3: Relax MoE shared-expert fusion gate to `seq <= 64`

**Decision:** Change the gate from `seq == 1` to `seq <= MOE_SHARED_FUSION_SEQ_THRESHOLD`
(constant 64). Short prefill tail chunks (e.g., last 32 tokens of a 2048-token prompt)
have small enough tensors that the dispatch-saving fusion is net-positive.

## Alternatives Considered

- **KV-cache rollback for speculative rejection (P2):** Would save forward passes on
  rejected draft tokens, but requires architectural changes to `KvManager` and is a
  separate initiative.

- **Triple-RMSNorm fusion for Gemma4 MoE:** Would reduce bandwidth 3x for that op, but
  requires a custom Metal kernel — too much effort for the ~1-3% decode gain.

- **Size-aware `quantized_matmul_rms_norm` enablement:** Needs re-benchmarking on sub-10B
  models — deferred to a follow-up benchmarking campaign.

## Risks

| Risk | Mitigation |
|------|-----------|
| Linear-attention last-only corrupts recurrence state | State is written to cache before slice; validated by existing test suite |
| Qwen3 MoE narrow softmax produces different top-k | Default OFF; opt-in validation required |
| MoE fusion on short prefill regressions | Threshold of 64 is conservative; benchmark before promotion |
