# TurboQuant Promotion PRD

Status: Active
Date: 2026-05-09
Owner: AX Engine
Supersedes: MLX-TURBOQUANT-KV-PRD.md, TURBOQUANT-IMPROVEMENT-PRD.md, TURBOQUANT-BENCHMARK-DESIGN.md
ADR: ../adr/0016-experimental-turboquant-kv-compression.md

## 1. Summary

This PRD is the canonical execution plan for TurboQuant promotion. It consolidates
prior KV-cache, codec/kernel, and benchmark-design documents into one active
surface while keeping ADR 0016 as the architecture decision record.

TurboQuant is currently an experiment. Promotion requires correctness, quality,
real-model performance, fallback accounting, and production-path integration
evidence. Microbenchmarks and reference-codec tests are necessary but not
sufficient.

## 2. Product Value

TurboQuant matters only if it improves long-context or memory-bound MLX runtime
behavior without hiding quality loss or fallback cost. The user-facing value is:

- lower KV memory pressure for long prompts and agent workloads;
- stable decode quality under bounded compression profiles;
- real runner speed or capacity improvement, not only standalone kernel speed.

## 3. Goals

G1. Keep the CPU/reference codec deterministic and easy to compare against.

G2. Prove the fused compressed decode path in the real runner before any default
runtime promotion.

G3. Record fallback counts, fallback token counts, quality profile, preset, and
readiness decision in artifact-friendly metadata.

G4. Validate quality on real model outputs, not only synthetic tensor fixtures.

G5. Keep aggressive presets research-only until model-specific evidence promotes
them.

## 4. Non-Goals

- No public default TurboQuant switch until promotion gates pass.
- No paper-equivalence claim unless the algorithm, quality gate, and runtime path
  match the claimed reference.
- No compression of GatedDelta recurrent state in the first promotion path.
- No promotion based only on standalone kernel microbenchmarks.

## 5. Promotion Gates

### Gate A: Reference Correctness

Required evidence:

- CPU/reference encode/decode round trip tests;
- deterministic preset-to-profile mapping;
- artifact validator for readiness metadata.

### Gate B: Fused Runtime Path

Required evidence:

- real runner path uses fused compressed decode for eligible attention shapes;
- fallback counters are zero or explicitly explained for promoted shapes;
- no hidden dequantize-full-history path is counted as TurboQuant acceleration.

### Gate C: Quality

Required evidence:

- model-level token/output comparison against uncompressed KV baseline;
- quality threshold tied to preset profile;
- deterministic failure reason when quality gate rejects a preset.

### Gate D: Performance and Capacity

Required evidence:

- same-model, same-prompt, same-generation before/after benchmark rows;
- memory or capacity improvement for long-context shapes;
- decode throughput does not regress beyond the accepted quality/profile budget.

### Gate E: Product Boundary

Required evidence:

- docs and runtime metadata label TurboQuant as experimental or promoted;
- unsupported models fail closed or fall back with explicit reason;
- public claims link to validated artifacts.

## 6. Phases

Phase 0: Keep experiment contained — **done / ongoing guardrail**

- Maintain ADR 0016 as the architecture boundary.
- Keep archived design docs as background only.
- Do not expose TurboQuant as a default product promise.

Phase 1: Real-path instrumentation — **partially done**

- Ensure fallback, preset, profile, and readiness metadata are emitted from the
  runtime path that would serve users.
- Add or refresh artifact validators.

Current status: readiness and validator guardrails exist, including promotion
readiness checks, README/public-doc experimental gating, and evidence-artifact
provenance for optimization-related environment flags. The missing part is
real-runner fused-path selection evidence for promoted decode shapes.

Phase 2: Real-model quality evidence — **partially done**

- Run selected supported models with uncompressed and compressed KV paths.
- Preserve prompts, outputs, seeds/policies, and metadata for review.

Current status: quality artifact plumbing exists, but no candidate can promote
until the real fused compressed decode path also passes the performance gate.

Phase 3: Long-context performance evidence — **blocking**

- Measure long-context TTFT, decode, memory, and fallback behavior.
- Compare against the uncompressed AX MLX path and, where applicable,
  `mlx_lm.benchmark` reference rows.

Current status: no passing long-context fused-path performance artifact exists
yet. Public docs must continue to label TurboQuant as experimental.

Phase 4: Promotion decision — **not started**

- If all gates pass for a bounded model/preset set, promote only that set.
- If gates fail, keep TurboQuant experimental and record the blocker.

Remaining phase count:

- **2 blocking phases remain**: Phase 3 long-context fused-path performance
  evidence and Phase 4 promotion decision.
- Phase 1 and Phase 2 are **partially done** and must be closed as part of the
  same promotion evidence package, but the immediate blocker is Phase 3.
- **0 standalone guardrail phases remain** for the current cycle: provenance,
  readiness, and public-doc experimental gating are in place; further guardrail
  changes should be tied to Phase 3/4 evidence gaps rather than tracked as a
  separate phase.

## 7. Active Reading Path

Read in this order:

1. `../adr/0016-experimental-turboquant-kv-compression.md`
2. this PRD
3. `../turboquant/QUALITY-GATE-ARTIFACT.md`
4. relevant benchmark artifacts under `../../benchmarks/results/turboquant/`

Archived predecessors live under
`../archive/superseded/2026-05-09-planning-consolidation/`.
