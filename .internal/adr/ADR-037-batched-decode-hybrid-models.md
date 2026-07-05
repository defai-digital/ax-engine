# ADR-037: Batched MLX Decode Boundary for Hybrid Linear-Attention Models

**Date:** 2026-07-05
**Status:** Proposed (deferred — no active concurrent-serving workload; revisit
against the PRD's probe-first phases)
**Deciders:** AX Engine MLX
**Related:** [PRD-2026-07-05-batched-mlx-decode-hybrid-models.md](../prd/PRD-2026-07-05-batched-mlx-decode-hybrid-models.md),
ADR-034 (evidence-gated kernel admission), ADR-027 (Qwen MTP decode phase 3),
`.internal/analysis/batched-decode/PHASE2-INTEGRATION-DESIGN.md`

## Context

Batched MLX decode is proven for dense full-attention qwen3 models
(kernel-level 1.9×@B2 / 2.4×@B4 amortization; runner-level 1.92×@B4
token-exact; context envelope plateauing ~1.5× at 2-4k), shipped default-OFF
behind `AX_MLX_BATCHED_DECODE` with promotion (2c) pending KV-accounting
reconciliation, preemption write-back, and non-greedy sampling.

Qwen 3.6 27B (`qwen3_5`) and 35B-A3B (`qwen3_5_moe`) — the models with the
strongest single-stream results in the README — are excluded by
`model_batched_eligible` on four independent axes: model family, linear
attention, MoE router (35B), and MTP presence. Single-stream decode for both
is at its measured bandwidth/kernel floor (ADR-027, ADR-034 registry), so
batching is the only remaining serving-throughput multiplier for them.

Three facts shape the boundary:

1. Gated-delta linear state batches trivially in principle (elementwise over
   `[B, Hv, dk, dv]`), and the hybrids' full-attention layers are plain full
   attention — the existing `BatchedKvCache` Phase-1 pieces apply. But the
   Phase-2a `mlx_fast_rope` batch-row bug showed that no MLX op may be assumed
   batch-correct without a synthetic oracle.
2. MoE expert reads amortize only to the extent rows select overlapping
   experts; with 256 experts top-8 the overlap at B≤4 is unmeasured. Dense
   amortization numbers do not transfer; a gather_qmm probe with real routing
   traces must gate the 35B.
3. Both models ship MTP with measured 1.4-1.9× single-stream multipliers that
   the batched path forfeits. The honest baseline for any promotion A/B is
   N × single-stream-with-MTP, which moves the aggregate crossover to B≥3-4.

## Decision

- Keep `qwen3_5` / `qwen3_5_moe` **excluded** from `AX_MLX_BATCHED_DECODE`
  until the PRD's probe-first phases run; the eligibility predicate in
  `batched_decode_session.rs` remains the enforced boundary and must not be
  widened ad hoc.
- Any future hybrid batching work follows the PRD phase order, with the MoE
  amortization probe (P2) as a hard gate for the 35B and per-op synthetic
  batch-correctness oracles mandatory for every MLX op on the path.
- The promotion bar is ADR-034-class evidence: token-exact oracles, interleaved
  median-of-N A/B against N × single-stream **MTP-enabled** baselines,
  ≥1.5× aggregate at B4 after MTP-loss accounting, no undocumented p50
  latency regression, full registry dossier before defaults change.
- Dense-qwen3 2c promotion (KV accounting, preemption, sampling) is a
  prerequisite: shared serving mechanics land on the simple architecture
  first.

## Consequences

- Concurrent Qwen 3.6 serving keeps paying N× weight reads for now; this is a
  conscious trade against a multi-week, correctness-sensitive project with an
  unmeasured MoE economics risk and no current workload pull.
- The decision record plus the PRD's phase gates prevent the known failure
  mode of this class of work: integrating batching into the crown-jewel
  runner on the strength of dense-model numbers that may not transfer to
  routed-expert or stateful-attention architectures.
- When a serving workload materializes, the first concrete step is cheap and
  prescribed: the P2 gather_qmm amortization probe with real routing traces,
  and the P1 batched gated-delta oracle — both isolated binaries, neither
  touching production routing.
