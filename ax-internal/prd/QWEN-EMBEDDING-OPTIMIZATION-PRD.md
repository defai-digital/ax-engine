# PRD: Qwen Embedding Optimization

| Field | Value |
|-------|-------|
| **Status** | Shipped |
| **Author** | ax-engine team |
| **Created** | 2026-07-06 |
| **ADR** | [ADR-039](../adr/ADR-039-embedding-bidirectional-attention-mask.md) |
| **Tech Spec** | [QWEN-EMBEDDING-IMPROVEMENT-TECH-SPEC](../tech-spec/QWEN-EMBEDDING-IMPROVEMENT-TECH-SPEC.md) |
| **Plan** | [IMPL-2026-07-06](../plan/IMPL-2026-07-06-qwen-embedding-improvement.md) |

## Problem Statement

AX Engine's embedding models (Qwen3-Embedding, EmbeddingGemma) exhibited measurable
throughput regressions at batch>1 and longer sequences:

- **Qwen3-Embedding-0.6B-8bit**: −11.4% throughput at batch=8, seq=256 vs. baseline.
- **EmbeddingGemma**: −8.3% throughput under the same conditions.

Root-cause profiling (`AX_MLX_EMBED_PROFILE=1`) identified two systemic sources:

1. **Redundant GPU kernel dispatches** per transformer layer — each layer issued
   separate `add` (residual) and `rms_norm` kernels where a single fused
   `add_rms_norm_pair` call would suffice.
2. **Unnecessary dtype conversion** in bidirectional attention mask construction —
   masks were allocated in f32 then cast to bf16 via `astype`, wasting one GPU
   dispatch and doubling host memory.

## Goals

| # | Goal | Metric |
|---|------|--------|
| G1 | Eliminate per-layer redundant dispatches in dense embedding forward | ≤ 1 dispatch saved per layer boundary |
| G2 | Build bidirectional masks natively in target dtype | 1 dispatch + 50% host memory saved |
| G3 | Fuse mean-pooling mask/scale construction in bf16 | 2 `astype` dispatches eliminated |
| G4 | Preserve full numerical correctness across bf16, fp16, fp32 | Zero test regressions |
| G5 | Maintain profiling ABI stability | `EmbedProfileSnapshot` fields unchanged |

## Non-Goals

- Modifying KV-cache or generation-path code (these changes are embedding-only).
- Adding new model families or weight formats.
- Changing the public HTTP/gRPC embedding API surface.

## Scope

### In Scope

- `layer_forward_dense_embed` — Qwen dense-embedding layer forward.
- `layer_forward_embed_gemma3` — EmbeddingGemma encoder layer forward.
- `forward_for_embedding_body` / `forward_for_embedding_gemma3_batch_body` — layer loops.
- `build_bidirectional_padding_mask` — mask construction.
- `build_embedding_mean_pool_inputs` — mean-pooling mask/scale.
- `model/profile.rs` — profiling stage documentation.
- `fastpath.rs` — environment flag for dense `add_rms_norm_pair` opt-in.

### Out of Scope

- Causal (autoregressive) generation path.
- MTP / speculative decoding.
- Metal kernel authoring.

## User Impact

Embedding API consumers (`/v1/embeddings`) see improved throughput at batch>1
without any client-side changes. No API contract changes.

## Success Criteria

1. `cargo test --workspace` passes with zero regressions (1347+ tests).
2. `cargo clippy` clean with project lint gates.
3. Profiling confirms dispatch-count reduction at each optimization point.
4. README benchmark tables reflect improved tok/s at batch=8, seq=256.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| fp16 clip-range overflow in fused `add_rms_norm_pair` | EmbeddingGemma fp16 path uses unfused `gemma3_clip_residual` + plain `rms_norm` fallback |
| Host buffer freed before lazy MLX graph materializes mask | Explicit `mlx_sys::eval()` barrier before function return |
| Profile ABI break from fused stage | `FfnNorm` stage recorded as zero-cost; snapshot struct unchanged |

## Rollback

Each optimization is an independent commit. Revert the offending commit without
affecting other phases.

## References

- Commit `c83cf6c0` — FFN residual fusion + bf16 mean-pooling.
- `.internal/bugs/embedding-*-batch-regression.md` — original regression reports.
- `docs/EMBEDDINGS.md` — public embedding documentation.
