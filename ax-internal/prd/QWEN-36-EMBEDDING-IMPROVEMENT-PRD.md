# PRD: Qwen 3.6 Generative & Qwen 3 Embedding Model Improvements

| Field | Value |
|-------|-------|
| **Status** | Active |
| **Author** | Engineering |
| **Date** | 2026-07-06 |
| **Last Updated** | 2026-07-06 |
| **ADR** | [ADR-039](../adr/ADR-039-embedding-bidirectional-attention-mask.md) |
| **Tech Spec** | [QWEN-EMBEDDING-IMPROVEMENT-TECH-SPEC](../tech-spec/QWEN-EMBEDDING-IMPROVEMENT-TECH-SPEC.md) |
| **Plan** | [IMPL-2026-07-06](../plan/IMPL-2026-07-06-qwen-embedding-improvement.md) |

## Problem Statement

A comprehensive code review of the Qwen 3.6 generative model and Qwen 3
embedding model implementations in ax-engine identified correctness gaps,
performance optimization opportunities, and observability improvements. While
both implementations are production-quality and fundamentally correct, several
areas can be improved to close performance gaps, improve semantic correctness,
and enhance operational visibility.

Key findings:

1. **Embedding attention semantics** — Qwen 3 embedding uses causal attention
   masking where bidirectional masking is semantically correct. This is a
   latent bug for last-token pooling and observable for mean pooling.
2. **Dense FFN decode compilation** — Dense FFN decode compilation is currently
   opt-in (reverted from default-on due to stream-registry safety concerns),
   leaving an estimated 5-10% decode throughput on the table for Qwen 3.6.
3. **Mean-pooled embedding compiled closure** — Mean-pooled embedding batches
   fall back to the imperative path with no compiled closure, missing an
   available throughput gain.
4. **Prefill throughput gap** — A 5-31% prefill throughput gap exists versus
   reference implementations (mlx-lm, lightning-mlx).
5. **Silent compiled-closure fallback** — Compiled-closure fallback occurs
   silently with no observability, making regressions hard to detect.

## Goals

### Primary Goals

- **G1:** Correct embedding attention semantics by adopting a bidirectional
  mask for the Qwen 3 embedding model.
- **G2:** Stabilize and re-enable dense FFN decode compilation for Qwen 3.6 as
  a default-on path with a kill-switch.
- **G3:** Extend the compiled closure to mean-pooled embedding batches.
- **G4:** Close the prefill throughput gap through profiling and targeted
  optimization.
- **G5:** Add observability for compiled-closure fallback events.

### Non-Goals

- Changing the split gate/up projection decision (blocked on upstream MLX
  kernel improvements).
- Supporting new model families beyond Qwen 3.x.
- Modifying MTP confidence threshold defaults (already optimized for
  throughput).

## Success Criteria

### Phase 1 — Correctness

- Bidirectional mask passes new parametrized tests for all pooling strategies
  (last-token, mean).
- No regression in embedding throughput.

### Phase 2 — Performance

- Dense FFN decode compilation is default-on with a kill-switch.
- Mean-pool compiled closure shows ≥10% throughput improvement at batch ≥ 4.
- Prefill gap reduced to <15% versus mlx-lm.

### Phase 3 — Observability

- Compiled-closure fallback events are visible in tracing output.
- Compile cache hit/miss metrics are exposed.

## Risks and Mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Dense FFN compile causes stream-registry invalidation in long-running processes | Medium | Kill-switch via `AX_MLX_DENSE_FFN_COMPILE=0`; progressive soak testing |
| Bidirectional mask changes embedding quality for edge models | Low | Extensive mean-pool regression tests; A/B benchmarking |
| Compiled mean-pool closure exceeds MLX buffer limits for large batches | Low | Dynamic fallback to imperative at batch_size threshold |
| Prefill optimization changes output determinism | Medium | Bit-exact output regression tests before/after |

## Dependencies

- MLX framework stability (stream registry).
- Existing `mlx_sys` FFI bindings for mask construction.
- `AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR` for integration testing.

## Timeline

- **Phase 1 (Correctness):** 1 week
- **Phase 2 (Performance):** 2-3 weeks
- **Phase 3 (Observability):** 1 week

## References

- Shipped optimization PRD: `QWEN-EMBEDDING-OPTIMIZATION-PRD.md`
- ADR: `../adr/ADR-039-embedding-bidirectional-attention-mask.md`
- Tech Spec: `../tech-spec/QWEN-EMBEDDING-IMPROVEMENT-TECH-SPEC.md`
- Implementation Plan: `../plan/IMPL-2026-07-06-qwen-embedding-improvement.md`
