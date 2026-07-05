# PRD: Batched MLX Decode for Hybrid Linear-Attention Models (Qwen 3.6)

**Date:** 2026-07-05
**Status:** Proposed — deliberately deferred; multi-week project, revisit when a
concurrent-serving workload materializes
**Related:** [ADR-037](../adr/ADR-037-batched-decode-hybrid-models.md),
`.internal/analysis/batched-decode/` (Phase 0 proof, Phase 2 design, 2b-ii
runner handoff), ADR-034 (admission bar), ADR-027 (Qwen MTP phase 3)

## Problem

The MLX runner decodes strictly batch=1: `run()` loops `for item in
execution_batch.items { run_item(item) }`, so N concurrent decode requests pay
N× weight reads while the scheduler, engine, and KvManager are already fully
continuous-batching (vLLM-style `ExecutionBatch`, token budget, mixed
prefill/decode, preemption). Batched decode is the one remaining multiplier for
serving throughput now that single-stream decode is at its measured
bandwidth/dispatch floor on every supported family.

The dense-qwen3 pilot proved the mechanism end to end (all on branch work now
merged; feature default OFF behind `AX_MLX_BATCHED_DECODE`):

- Phase 0 kernel probe: weight-read amortization **1.9×@B2 / ~2.4×@B4**,
  plateau ~2.3-2.5× by B8; bandwidth→compute transition at B≈3-4
  (`decode_batch_amortization_probe`, includes a Qwen3.6-27B-6bit cell:
  B1 31.4 → B2 58.3 → B4 70.5 agg tok/s at the kernel level).
- Phase 2b runner integration (dense full-attention qwen3 only): run()
  interception, token-exact vs per-item decode, **1.92×@B4** runner-level A/B;
  context envelope 64→1.92×, 2048→1.44×, 4096→1.49× (KV reads do not
  amortize; plateaus ~1.5× at long context).
- Remaining for dense promotion (2c): core KV-block accounting reconciliation,
  preemption write-back, non-greedy sampling, ADR-034-style gate.

Qwen 3.6 27B (`qwen3_5`) and 35B-A3B (`qwen3_5_moe`) are excluded from the
batched path on every axis of `model_batched_eligible`
(`batched_decode_session.rs`): family != "qwen3", `linear_attn.is_some()`,
`router_proj.is_some()` (35B), and `has_mtp`. This PRD scopes what it takes to
lift those exclusions.

## Why hybrids are attractive — and why the economics differ from dense

- Linear-attention decode is O(1)-state per token: the gated-delta update is
  elementwise over a per-request `[Hv, dk, dv]` f32 state plus a conv window.
  Batched, this is a natural `[B, Hv, dk, dv]` elementwise step — **no ragged
  KV, no per-row masks** for 3 of every 4 layers. In principle the easiest
  attention class to batch.
- The 1-in-4 full-attention layers are plain full attention (no sliding
  window), so the existing `BatchedKvCache` + `batched_decode_validity_mask`
  Phase-1 pieces apply unchanged.
- **MoE economics are unproven (35B):** dense weight reads amortize because
  every row touches the same weights. 256-expert top-8 routing overlaps little
  across rows at small B, so expert reads may amortize far below the dense
  1.9×@B2. No gather_qmm amortization probe exists. This is the single
  largest unknown; it gates the 35B entirely.
- **MTP interaction (both models):** the batched path has no speculative
  decoding, so each request forfeits its MTP multiplier (27B ~1.5-1.9× at
  measured accept rates; 35B ~1.4-1.5×). Aggregate crossover is therefore
  B≥~3-4, not B=2 as for non-MTP dense models. Any promotion A/B must compare
  batched-direct aggregate against **N × single-stream-MTP**, not N ×
  single-stream-direct — the honest baseline is the shipped configuration.

## Phases (probe-first; each phase is a go/no-go gate)

1. **P1 — Batched gated-delta step (isolated):** batched conv window
   `[B, k-1, conv_dim]` + recurrent update `[B, Hv, dk, dv]` (f32), per-row
   RoPE offsets for full-attn layers. Token-exact oracle: each row ==
   single-request forward on real weights. Mandatory synthetic per-op
   batch-correctness checks: the Phase-2a work found `mlx_sys::rope`
   (mlx_fast_rope) mis-rotates batch rows > 0 for `[B,H,1,D]` decode inputs —
   assume nothing is batch-safe until proven; the debug key is
   identical-prompts-PASS / distinct-prompts-FAIL / row-0-always-correct.
2. **P2 — MoE amortization probe (35B gate):** Phase-0-style probe of
   `gather_qmm` at B∈{1,2,4,8} with measured expert-overlap distributions from
   real routing traces (record overlap from a served trace, replay in the
   probe). If aggregate speedup at B4 < ~1.3×, the 35B is out of scope and the
   PRD narrows to the 27B (dense FFN amortizes like the pilot).
3. **P3 — Batched hybrid forward:** interleave batched linear layers with
   batched full-attention layers over `BatchedKvCache`; shared-expert +
   router at B>1 for the 35B if P2 passes. Session plumbing mirrors
   `BatchedDecodeSession` (slot compaction, add/remove, seed-from-prefill with
   the `add_with_seed_len` warmed-cache lesson).
4. **P4 — Serving integration:** extend `model_batched_eligible`; per-request
   states join/leave the batch (recurrent state is per-slot `[Hv,dk,dv]`,
   ~144 MB f32 per 27B request — memory-budget the batch cap); preemption
   drops state (linear states cannot write back into core KV accounting —
   preempted rows must re-prefill or the request-state must persist the
   f32 states, decide in tech spec). MTP-resident requests never join
   (has_mtp exclusion stays) unless P5 lands.
5. **P5 (stretch) — Batched MTP verify:** verify forwards are seq=depth+1
   multi-token — batching them recovers the MTP multiplier inside the batch.
   Out of scope until P1-P4 hold; noted so the crossover math has a path to
   B=2 viability.

## Admission bar

- Every phase: token-exact oracles vs single-request decode on real weights.
- Promotion A/B: median-of-N interleaved, aggregate tok/s at B∈{2,4} vs N ×
  single-stream **with MTP enabled** on the same artifacts; gate ≥1.5×@B4
  after MTP-loss accounting, plus no per-request p50 latency regression
  beyond the documented budget.
- ADR-034 registry entry with the full dossier before any default flips.

## Non-goals

- B>8 scaling (Phase 0 shows the win plateaus ~2.3-2.5×; do not engineer for
  B=32).
- Batched prefill (scheduler already interleaves chunked prefill; different
  problem).
- Sliding-window (Gemma) batching — separate design (rotating per-slot caches,
  ragged sliding masks); explicitly out of scope here.

## Prerequisites

Dense-qwen3 2c promotion should land first: it derisks the shared serving
integration (KV accounting, preemption write-back, non-greedy sampling) on the
simplest architecture before hybrid state management stacks on top.
