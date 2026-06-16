# PRD: Decode & Prefill Speed Optimization — Phase 2

**Date:** 2026-06-16
**Status:** Approved
**Owner:** AX Engine Core

## Problem Statement

The Qwen 3.5/3.6 hybrid linear-attention models waste significant FFN compute during
prefill: the last-position-only optimization (introduced in decode-speed phase 1) only
engages on standard-attention layers. Linear-attention layers fall back to the
unoptimized path, processing the full `[1, seq, hidden]` tensor through their FFN even
when only the last position's output is needed. Additionally, the Qwen3 MoE router
computes softmax over all experts (128–256) before selecting top-k, and MoE shared-expert
fusion is restricted to decode-only (`seq == 1`).

## Goals

| Goal | Metric | Target |
|------|--------|--------|
| Improve Qwen3.5/3.6 prefill throughput | prefill tok/s on p=2048 | +5–15% |
| Reduce Qwen3 MoE decode router overhead | decode tok/s on MoE models | +1–3% |
| Extend MoE fusion to short prefill chunks | prefill tok/s on MoE models | +<1% |

## Scope

### In Scope

1. **P0 — Last-position-only for linear-attention layers:** Extend the
   `last_position_only_after_attention` optimization to `qwen3_linear::layer_forward` so
   linear-attention layers can slice to `[1, 1, hidden]` before FFN when they are the
   last linear-attention layer before standard-attention layers in the prefill pass.

2. **P1 — Qwen3 MoE router narrow softmax:** Add an opt-in env flag
   `AX_MLX_QWEN3_MOE_NARROW_SOFTMAX` that switches the Qwen3 MoE router from
   full-width `softmax_precise` → argpartition to argpartition-first → narrow softmax over
   only the top-k subset (matching the Gemma4 router pattern). Default OFF pending
   validation against mlx-lm reference.

3. **P3 — Relax MoE shared-expert fusion gate:** Change the
   `qwen3_moe_weighted_sum_with_shared_metal` fusion gate from `seq == 1` to
   `seq <= MOE_FUSION_SEQ_THRESHOLD` (default 64) so short prefill tail chunks also
   benefit from the fused kernel.

### Out of Scope

- KV-cache rollback for speculative rejection (P2, separate initiative)
- `dense_add_rms_norm_pair` / `dense_qmatmul_rms_norm` re-benchmarking (separate initiative)
- Gemma4 MoE triple-RMSNorm fusion (requires custom Metal kernel)

## Success Criteria

- All changes compile with `cargo build --workspace` and pass `cargo clippy -D warnings`
- All changes pass `cargo test --workspace`
- P0: Qwen3.5/3.6 prefill tok/s improves measurably on p=2048 prompts
- P1: Token-exact equivalence against mlx-lm when narrow softmax is enabled
- P3: No regression on existing MoE benchmarks
