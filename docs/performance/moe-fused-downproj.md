# MoE Fused Down-Projection GEMV — Failed Optimization Report

## Executive Summary

A custom fused int4/int8 gather-GEMV kernel that folded the MoE routing-weighted
sum into the down-projection matmul epilogue was implemented, productized behind
a flag, and validated end-to-end. **It regressed decode throughput by ~2–4% and
changed greedy output (bf16 reduction-order drift).** The production kernel and
its flag have been reverted; this document, the two microbench probes, and the
underlying commits remain as the record. The headline lesson: a per-dispatch
microbench that builds N graph nodes in its timed loop measures scheduling
overhead, not compute — for a fused-vs-MLX comparison inside a real graph, only
the end-to-end run on a real checkpoint counts.

## Provenance: the article vs. this work

Triggered by a Medium article ("one Metal kernel made Apple Core AI's MoE decode
2.3–6× faster"). **The article's optimization is NOT what was attempted here.**
The article's win was a *bandwidth fix*: Core AI's `GatherMM` read every expert's
weights on every token (~8.8 GB/token for a 1.5B-active model), and the custom
`gather_qmm` made the routing index a runtime index so only the routed experts
were physically read. AX **already** has that property — `mlx_sys::gather_qmm`
reads only the routed experts — so the article's literal change was confirmed to
be a no-op for AX (step 1 of the investigation).

This work instead tested a *different, self-originated* idea: rewrite the
down-projection gather-GEMV and fold the routing-weighted sum into the matmul
epilogue, eliminating the `[top_k, hidden]` intermediate write + second dispatch.

## Attempts (in order)

1. **Residual-add fusion → rejected.** Fusing the residual-add into the
   weighted-sum measured ~0% (MLX's add is already free). Built, A/B'd, reverted.
   Probe: `moe-downproj-fusion-probe`.
2. **Fused gather-GEMV (the self-originated idea).** From-scratch int4 gather-GEMV
   folding the routing-weighted sum into the matmul epilogue. Microbench showed
   ~+2–3% (~27 µs vs ~39 µs per dispatch), correctness 2e-7 vs dequant oracle.
   Probe: `moe-fused-gather-qmm-probe`. This reversed an earlier prediction that
   it would lose. Productized flag-gated (`AX_MLX_MOE_FUSED_DOWNPROJ`, default off).
3. **E2E A/B on a real checkpoint → the microbench did not survive.** On
   Qwen3.6-35B-A3B-4bit (256 experts, top_k=8, in_dim=512) the fused path was
   −8 to −11% slower initially, with greedy output diverging.
4. **Bug found and fixed during review.** A `simd_sum` barrier per expert
   (8 per output row) serialized the experts. Folding the routing weight into
   each lane's partial and reducing once (a correct application of distributivity)
   moved −8% → −3.5%. A multi-group (groups=8) correctness oracle test confirmed
   the math was right; the parity divergence is bf16 reduction-order drift, not a
   bug (the fused path accumulates in f32 and rounds once, so it is marginally
   *more* accurate — just different).

## Result (Qwen3.6-35B-A3B-4bit, single-request deterministic decode)

| Path | decode tok/s |
|---|---|
| standard (MLX gather_qmm + tail) | ~159 |
| fused GEMV (after simd_sum fix) | ~155 |
| **delta** | **−2% to −4%**, stable across trials |
| greedy parity | diverges (bf16 reduction-order drift) |

## Why the microbench overstated the win (structural, not bad luck)

The probe's `time_amortized` builds a chain of **64 graph nodes inside the timed
loop** and divides by `ITERS × CHAIN`. That measures two things that are not the
bottleneck in the real graph:

1. **Graph-build/dispatch overhead is in the timed region.** The "current" path
   builds 2 MLX nodes; "fused" builds 1. In isolation the 2→1 node saving looks
   like a ~10–12 µs win. But MLX's graph compiler, seeing gather_qmm + weighted-sum
   embedded in a full MoE layer, overlaps that "tail" with surrounding ops — there
   is no idle GPU waiting for it. The microbench measured a component the compiler
   absorbs.
2. **No surrounding graph = no scheduler overlap.** A real layer has attention +
   gate/up + activation + residual around the down-proj. The isolated probe can't
   model that concurrency, so the tail looks serial when it isn't.

Two additional structural mismatches compounded the miss:

- **Shape mismatch.** Probe used 128 experts / 768 in_dim; real model is 256 / 512.
  GEMV cost scales with in_dim, changing the tail-vs-matmul ratio.
- **Baseline mismatch.** Probe extrapolated "% of a decode step" against a
  hypothetical 65 tok/s; the real model runs ~159 tok/s. Same µs delta → different %.

## Reusable lessons

1. **Microbench measures dispatch overhead, not compute — always run E2E.** A
   per-dispatch harness that builds N graph nodes in its timed loop is measuring
   scheduling, not GPU work. For anything that changes graph structure (fusing
   ops, removing dispatches), only the E2E run on a real checkpoint counts.
2. **bf16 reduction-order drift is a hard parity blocker.** A kernel that
   accumulates in f32 and rounds once is marginally *more accurate* than the
   standard path (which rounds each expert's dot to bf16 before the weighted sum),
   but at temperature 0 a tiny per-logit delta flips argmax at near-ties and
   cascades. "More accurate but different" is not shippable against MLX's path.
   The only way to *bit-match* is to round per-expert before summing — which
   throws away the f32-accumulate structure that is the whole point.
3. **The article's bandwidth fix is already in AX.** AX reads only routed experts
   via `mlx_sys::gather_qmm`. There is no GatherMM-style over-read to fix.

## Code state after revert

**Removed** (the failed optimization):

- `AX_MLX_MOE_FUSED_DOWNPROJ` flag and its `fastpath` definition.
- `fused_moe_downproj_gemv` function, its `FUSED_MOE_DOWNPROJ_GEMV_KERNEL_SOURCE`,
  the static kernel handle, the dispatch block, and the three unit tests.
- `scripts/bench_moe_fused_downproj_ab.py` and its scenario manifest
  `benchmarks/manifests/scenario/moe_fused_downproj_ab.json` (coupled to the
  removed flag).

**Kept** (on-pattern for this repo):

- `crates/ax-engine-microbench/src/bin/moe_downproj_fusion_probe.rs` — ceiling
  probe (correctly frames the article's bandwidth point). One of 9 probes in the
  microbench crate, which is the established home for experimental diagnostics.
- `crates/ax-engine-microbench/src/bin/moe_fused_gather_qmm_probe.rs` — the
  fused-GEMV probe, docstring corrected to state the E2E result and the
  provenance (self-originated, not the article's approach).

**Recoverable from history** if a future model shape ever makes this worth
revisiting: commit `fc01e463` (productized kernel), `34d1e0d1` (E2E harness),
`c4011f11` / `3b8973da` (probes).
