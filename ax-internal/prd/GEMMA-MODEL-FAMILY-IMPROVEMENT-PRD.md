# PRD: Gemma Model Family Improvement

| Field | Value |
|-------|-------|
| **Status** | Active |
| **Author** | Engineering |
| **Date** | 2026-07-06 |
| **Last Updated** | 2026-07-06 |
| **ADR** | [ADR-040](../adr/ADR-040-gemma-layer-shell-extraction.md) |
| **Tech Spec** | [GEMMA-MODEL-FAMILY-IMPROVEMENT-TECH-SPEC](../tech-spec/GEMMA-MODEL-FAMILY-IMPROVEMENT-TECH-SPEC.md) |
| **Plan** | [IMPL-2026-07-06](../plan/IMPL-2026-07-06-gemma-model-improvement.md) |

## Problem Statement

A comprehensive code review of the Gemma (generative), DiffusionGemma, and
EmbeddingGemma implementations in ax-engine identified 14 findings spanning
correctness, performance, and code hygiene. While all three implementations are
production-quality and fundamentally correct, several areas can be improved to
strengthen safety documentation, close performance gaps, enhance operational
visibility, and unify error handling across the Gemma model family.

Key findings:

1. **Raw pointer capture without SAFETY comments** — DiffusionGemma's
   full-pipeline closure captures raw pointers without documenting the lifetime
   invariant, violating the project's unsafe-documentation convention.
2. **No compiled closure for Gemma4 assistant MTP forward** — The assistant
   MTP forward path is stateless per step but falls back to imperative
   execution, missing an available throughput gain.
3. **KV concat buffer divergence reachable via opt-in** — A legacy KV concat
   code path remains reachable through an opt-in flag, creating a maintenance
   surface with no active test coverage.
4. **No compiled closure for causal commit pass** — DiffusionGemma's causal
   commit pass has side-effecting KV writes that prevent full compilation.
5. **Significant code duplication between causal and bidirectional layer
   forwards** — `layer_forward` and `layer_forward_bidirectional` share ~200
   lines of identical shell code (QKV projection, QK norm + RoPE, output
   projection, FFN, residual + gating), differing only in attention mechanism.
6. **No per-layer profiling for bidirectional denoiser forward** — The
   bidirectional forward path in DiffusionGemma lacks the per-layer profiling
   instrumentation present in the causal path.
7. **Batched decode per-row RoPE is O(B) dispatches** — Batched decode decode
   applies RoPE RoPE computation as separate Metal dispatches per row in the
   batch.
8. **gemma4_unified.rs uses String errors instead of thiserror** — The Gemma4
   unified module returns `Result<_, String>` rather than a typed thiserror
     enum, inconsistent with the workspace error-handling convention.
9. **EmbeddingCache::needs_refresh forces GPU sync** — The embedding cache
   freshness check triggers a GPU CPU synchronization, adding latency on the
   imperative fallback path.
10. **build_bidirectional_padding_mask not cached across calls** — The
    bidirectional padding mask is rebuilt on every embedding batch forward.
11. **Inconsistent kv_heads error handling across layer forwards** — Different
    layer forward variants handle `kv_heads` mismatches differently (expect vs
    propagate vs silent default).
12. **Mixed error type strategies** — The Gemma model family uses String, &str,
    panic, and thiserror inconsistently across modules.
13. **No embedding cache hit-rate telemetry** — DiffusionGemma's
    `EmbeddingCache` has no observability into cache hit/miss rates.
14. **normed_mean_pool_probe uses per-row allocations** — The diagnostic probe
    allocates per-row tensors instead of a single batched operation.

## Goals

### Primary Goals

- **G1:** Add SAFETY documentation to all raw pointer captures in
  DiffusionGemma closures (Finding 1).
- **G2:** Add per-layer profiling to the bidirectional denoiser forward
  (Finding 6).
- **G3:** Add embedding cache hit/miss telemetry to DiffusionGemma
  (Finding 13).
- **G4:** Unify error handling across Gemma model family using thiserror enums
  (Findings 8, 11, 12).
- **G5:** Compile the Gemma4 assistant MTP forward path for improved
  draft-throughput (Finding 2).
- **G6:** Extract a shared layer shell to eliminate causal/bidirectional code
  duplication (Finding 5).

### Non-Goals

- Changing model semantics or output behavior for any Gemma variant.
- Adding new model families beyond Gemma, DiffusionGemma, and EmbeddingGemma.
- Benchmark re-validation (existing benchmarks remain authoritative).
- Metal kernel work for batched RoPE (Finding 7 — requires kernel-level
  changes, deferred).
- Retiring the KV concat buffer path (Finding 3 — already gated off by
  default, documentation is sufficient).

## Success Criteria

### Phase 1 — Safety & Observability

- All raw pointer captures in `diffusion.rs` have `// SAFETY:` comments
  documenting the lifetime invariant.
- Per-layer profiling is available for bidirectional denoiser forward.
- Embedding cache hit/miss counters are exposed and increment correctly.

### Phase 2 — Telemetry & Diagnostics

- Bidirectional padding mask is cached with LRU eviction.
- `normed_mean_pool_probe` is documented as usage-only; no code change
  required unless probe is promoted to production.

### Phase 3 — Error Handling Unification

- `Gemma4UnifiedError` thiserror enum replaces `Result<_, String>` in
  `gemma4_unified.rs`.
- Shared `ModelError` thiserror enum covers cross-cutting model errors.
- `kv_heads` handling uses `expect` consistently with descriptive messages.

### Phase 4 — Performance (Assistant MTP Compilation)

- Gemma4 assistant MTP forward is wrapped in an `MlxClosure`.
- Assistant MTP depth-2 acceptance rate is unchanged after compilation.
- `AX_MLX_GEMMA4_ASSISTANT_COMPILE` env flag gates the compiled path.

### Phase 5 — Architecture (Layer Shell Extraction)

- Shared `layer_shell` function eliminates ~200 lines of duplication.
- DiffusionGemma convergence is unchanged after extraction.
- Causal and bidirectional forward produce bit-exact output vs pre-change
  baseline.

## Risks and Mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Layer shell extraction introduces closure overhead in hot path | Low | Benchmark before/after; `#[inline]` on the shell function |
| Compiled assistant MTP closure closure cache invalidation | Low | Strict cache-key matching on (hidden_size, layer_count); imperative fallback on miss |
| Error type migration breaks existing error-match arms in callers | Medium | Exhaustive match audit; incremental migration per module |
| Per-layer profiling adds measurable overhead to bidirectional forward | Very Low | Conditional on `AX_MLX_DECODE_PROFILE` env flag; zero-cost when disabled |
| Embedding cache counter contention under high concurrency | Very Low | `AtomicU64` with `Relaxed` ordering; no lock required |

## Dependencies

- MLX framework stability (compiled registry, compiled closure API).
- Existing `mlx_sys` FFI bindings for closure compilation.
- `AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR` for integration testing with model
  weights.
- ADR-040 for layer shell extraction design.

## Timeline

- **Phase 1 (Safety & Observability):** Single session
- **Phase 2 (Telemetry & Diagnostics):** Single session
- **Phase 3 (Error Handling):** Single session
- **Phase 4 (Assistant MTP Compilation):** Single session
- **Phase 5 (Layer Shell Extraction):** Single session

## References

- ADR: `../adr/ADR-040-gemma-layer-shell-extraction.md`
- Tech Spec: `../tech-spec/GEMMA-MODEL-FAMILY-IMPROVEMENT-TECH-SPEC.md`
- Implementation Plan: `../plan/IMPL-2026-07-06-gemma-model-improvement.md`
