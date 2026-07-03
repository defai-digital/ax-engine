# PRD: Evidence-Gated Decode Hot-Path Kernel Strategy

**Date:** 2026-07-03
**Status:** Accepted / admission gate implemented
**Owner:** AX Engine MLX
**ADR:** [ADR-034-decode-hot-path-kernel-strategy.md](../adr/ADR-034-decode-hot-path-kernel-strategy.md)
**Tech Spec:** [TECH-SPEC-2026-07-03-decode-hot-path-kernel-strategy.md](../tech-spec/TECH-SPEC-2026-07-03-decode-hot-path-kernel-strategy.md)

## Problem Statement

AX Engine has reached the point where runtime policy alone cannot be the whole
decode-speed strategy. The repo already owns scheduling, speculative decoding,
KV behavior, benchmark provenance, and several custom Metal surfaces. It also
delegates broad tensor execution to MLX, including `quantized_matmul`, attention,
RMSNorm, and RoPE in the production MLX model path.

The risk is prioritizing custom kernels from intuition instead of production
evidence. Several sidecar-kernel attempts already show the trap:

- Fused residual-add + RMSNorm was correctness-equivalent but slower than MLX's
  split path and was reverted.
- A fused MoE gather-GEMV looked promising in microbenchmarks but regressed real
  Qwen3.6-35B-A3B decode by roughly 2-4% and changed greedy output.
- TurboQuant fused cold decode is directionally useful but not production-ready
  while per-layer dispatch fragmentation and CPU hot-tail merging dominate.

AX needs a durable strategy that keeps MLX as the general runtime while allowing
AX-owned kernels only where they are profile-justified, shape-specialized,
rollback-safe, and proven by end-to-end artifacts.

## Goals

| Goal | Metric | Target |
|---|---|---|
| G1: Evidence-first admission | Candidate has profile, oracle, microbench, and E2E artifacts | 100% before production routing |
| G2: Avoid negative fusions | Repeated known-losing patterns are blocked | RMSNorm/residual sidecar remains NO-GO unless new evidence changes the premise |
| G3: Improve decode on supported workloads | Median decode tok/s | >= 5% E2E win on the targeted real-model row, or >= 3% if the change also reduces variance/TTFT |
| G4: Preserve correctness | Greedy parity and bounded numeric drift | 100% greedy parity unless an explicit ADR accepts drift; logprob/sample deltas documented |
| G5: Keep rollback safe | Feature gates and fallbacks | Every new runtime path is default-OFF until promoted and has a kill switch |
| G6: Maintain MLX boundary | Dependency/layering audit | No generic tensor-runtime rewrite and no transport/serde/web concerns in `core` |

## Non-Goals

- Rewriting MLX, replacing the whole tensor library, or owning generic matmul.
- Training kernels.
- Full FlashAttention coverage for every batch/context/layout mode.
- Shipping a kernel because it is faster in isolation when real-graph decode is
  neutral or slower.
- Promoting 6-bit custom projection work from Hugging Face's Metal quantization
  docs alone; those docs describe 2/4/8-bit Metal quantization, not AX's full
  model-layout matrix.

## Best-Practice Principles

1. **Profile before design.** A kernel candidate starts from
   `AX_MLX_DECODE_PROFILE=1`, model-family-specific telemetry, and a real
   benchmark row. README prose or intuition is not enough.
2. **Prefer graph-level dispatch reduction before one-op rewrites.** If the gap
   is hundreds of MLX ops per decode step, an MLX compile/closure route usually
   has more leverage than hand-optimizing a single already-tuned primitive.
3. **Target shape specialization, not generic MLX replacement.** Valid AX-owned
   kernels should specialize on decode token = 1, small batch, known layouts, or
   KV/runtime ownership that MLX cannot infer.
4. **Reduce bytes/token or dispatches/token.** A candidate must state which bytes,
   materializations, readbacks, or dispatches it removes.
5. **Benchmark in the real graph.** Microbenchmarks are admission evidence, not
   promotion evidence.
6. **Fail closed.** Every path needs a feature flag, fallback, route telemetry,
   and a rollback plan before integration.
7. **Record NO-GO outcomes.** Failed kernels are useful assets only if their
   failure mode becomes a future gate.

## Priority Order

| Priority | Track | Rationale |
|---|---|---|
| P0 | Decode profile and candidate registry | Prevents intuition-first kernel work and anchors every proposal to real wall-time share |
| P0 | Graph-level decode compilation / MLX compile exposure | Best match for the observed dispatch-overhead shape in direct decode |
| P1 | Existing paged decode attention wiring and validation | High leverage for long-context decode, but only after proving real-graph integration does not fragment MLX lazy execution |
| P1 | Quantized projection feasibility | Worth testing only for layout-specific cases where MLX `quantized_matmul` is not already the winning primitive |
| P2 | KV/TurboQuant on-GPU merge and paging primitives | Runtime-owned surface; focus on eliminating CPU readback/hot-tail merge rather than generic rollback |
| P2 | Sampling/top-k/top-p GPU path | Useful only after telemetry shows CPU sync or logits processing dominates a serving row |
| NO-GO | Standalone residual + RMSNorm sidecar | Prior A/B regressed decode and prefill; do not repeat without a new mechanism |

## Product Scope

### In Scope

- Internal admission workflow for decode hot-path kernels.
- Candidate-specific artifacts under `.internal/analysis/` or benchmark result
  directories.
- Feature-gated prototype paths in `ax-engine-mlx`, `mlx-sys`, or `core/metal`
  when the candidate belongs there.
- New or updated microbenchmarks in `crates/ax-engine-microbench/src/bin/`.
- Benchmark scripts/checkers that make promotion and rollback objective.

### Out of Scope

- Public README performance claims before canonical benchmark artifacts exist.
- Default-on runtime changes before a promotion ADR or status update.
- Broad refactors of `core` scheduling, server transport, SDK API, or model
  loading unless directly required by the accepted candidate.

## Success Criteria

- A candidate cannot enter production routing unless the tech spec's admission
  checklist is complete and passes
  `python3 scripts/check_decode_hot_path_kernel_admission.py`.
- The first promoted candidate produces a checked-in artifact with:
  - baseline row,
  - candidate row,
  - profile attribution,
  - greedy parity result,
  - feature flag state,
  - host/build provenance.
- `.internal/prd/README.md` tracks this PRD as the authoritative decode-kernel
  strategy.
- Known failed approaches are referenced in the ADR/tech spec so future work does
  not relitigate them from scratch.

## Implementation Status

- The admission workflow is implemented as
  `scripts/check_decode_hot_path_kernel_admission.py`.
- `scripts/check-scripts.sh` and `scripts/check-mlx-telemetry.sh` run the
  checker, allowing empty candidate roots but failing closed for any discovered
  candidate manifest.
- Candidate manifests use schema
  `ax.decode_hot_path_kernel_candidate.v1` under
  `.internal/analysis/decode-hot-path-kernels/<candidate-id>/candidate.json`.
- No runtime kernel is promoted by this PRD alone. Promotion remains candidate
  specific and requires the full checked evidence bundle.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Microbench win does not survive real decode | Wasted kernel work or regression | Require real-model E2E before promotion |
| Numeric drift changes greedy output | Silent quality regression | CPU oracle, scalar fallback comparison, greedy parity suite |
| MLX lazy graph fragmentation | Custom dispatch slower than split MLX ops | Profile eval barriers and command-buffer boundaries before promotion |
| Candidate overlaps existing ADRs | Conflicting direction | Link to ADR-024, ADR-030, ADR-032, ADR-033 and supersede only explicitly |
| Production default too early | User-visible slowdown | Default-OFF prototype, explicit kill switch, rollback telemetry |

## Dependencies

- `docs/MLX-BACKEND.md` custom-kernel boundary.
- `docs/performance/decode-gap.md` dispatch-overhead analysis.
- `docs/performance/moe-fused-downproj.md` failed MoE fused-GEMV record.
- `benchmarks/results/inference/mlx-inference/ab-rmsnorm-add/README.md` failed
  residual + RMSNorm A/B.
- `metal/phase1-kernels.json` and `metal/kernels/phase1_dense_path.metal`.
- `crates/mlx-sys/src/metal.rs` for MLX custom Metal kernel integration.
