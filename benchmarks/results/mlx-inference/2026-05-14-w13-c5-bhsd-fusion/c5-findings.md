# W1.3 C5 — Q/K/V `as_strided` BHSD fusion (2026-05-14): NO-GO

PRD: `MLX-DECODE-SUBMIT-COST-PRD.md` W1.3
Audit: `MLX-DECODE-W13-CAST-RESHAPE-AUDIT.md` C5
Spike artifact: `crates/ax-engine-mlx/src/model.rs::bhsd_view_from_proj`
                + `bhsd_view_from_proj_matches_reshape_transpose` unit test

## What was tried

Replace the `reshape([1, seq, n_heads, head_dim]) → ... → transpose([0, 2, 1, 3])`
pair on Q/K/V with a single `mlx_as_strided` view that produces BHSD
directly. Applied to all four call sites that touch this pair on Gemma 4
E2B 4-bit decode:

- Normal-branch Q (15 layers): `reshape` + `transpose` → `as_strided`
- Normal-branch K (15 layers): same
- Normal-branch V (15 layers): same, plus the `prepare_value_bhsd`
  trailing transpose dropped (V's optional rms_norm runs in-place on the
  BHSD view since RMS is invariant to leading-axis permutation)
- KV-shared-branch Q (20 layers): same as Normal-branch Q

Predicted savings: 50-65 ops/step (3 ops × 15 layers + 1 op × 20
KV-shared layers — kv_heads=1 path on Normal Q has 1 reshape removed,
KV-shared Q has 1 reshape and 1 transpose removed).

The bit-exact unit test `bhsd_view_from_proj_matches_reshape_transpose`
confirms the helper produces elementwise-identical output to the
imperative `reshape + transpose` chain.

## A/B result (M5 Max, gemma-4-e2b-it-4bit, 128/128, 3 reps each)

Apples-to-apples — control rep and C5 reps all run on the same M5 Max
session, same release binary timing, same prompt sha256
(`4ebdfdf02961…`):

| Rep | decode tok/s | op_count/step | wall_us/step |
|---|---:|---:|---:|
| control (no C5) | 186.25 | 1764 | 5,322 |
| C5 rep1 | 184.26 | 1699 | 5,350 |
| C5 rep2 | 185.00 | 1699 | 5,350 |
| **C5 mean (rep1+rep2)** | **184.6** | **1699** | **5,350** |
| **Δ vs control** | **-0.9%** | **-65 (-3.7%)** | **+27 µs (+0.5%)** |

The op-count drop matches prediction exactly. **Wall-time and tok/s went
backwards** despite the saved dispatches. Repeated across two reps so
this is signal, not noise (control 5,322 vs both C5 reps at 5,349-5,350,
exceeding the ±10 µs run-to-run noise floor seen across the W1.3 series).

## Reading

The W1.3 C5 audit explicitly flagged the risk:

> Risk: subsequent ops (rope, slice_update for KV append, SDPA) may
> internally call contiguous() if input is non-contiguous, adding back
> the dispatch we tried to skip. Net could be a wash.

Confirmed empirically — and worse than a wash. MLX's downstream ops
silently materialise the strided view (one `contiguous()` insertion per
strided input apparently costs more than the eliminated `reshape +
transpose` pair). The strided view that `transpose` produces today is
either tolerated by these MLX ops differently from `as_strided`, or
existing MLX paths have an internal fast-path that recognises the
specific transpose-of-reshape pattern.

The 2 µs/op host cost from the W1.3 baseline ($-65$ ops × 2 µs $\approx$
130 µs/step max savings) was indeed clawed back: the +27 µs/step
regression is the net of (saved Rust→MLX dispatch) − (added MLX-internal
contiguous work).

## Decision

PRD W1.3 C5 closed **NO-GO** under the stated kill criterion (tok/s
regression > 0.5% across repeated 3-rep runs).

The wiring at the 4 call sites was reverted; `bhsd_view_from_proj`
helper + unit test are kept marked `#[allow(dead_code)]` as a
documented spike artifact (same pattern as W2.a `geglu`). They remain
useful as a regression-protected reference for any future retry that:

- finds a way to suppress the downstream `contiguous()` insertion (e.g.
  by feeding the BHSD view through an explicit `mlx.contiguous()` only
  ONCE at the top of attention, then reusing the contiguous buffer for
  all consumers); OR
- couples the `as_strided` fusion with an `mx.compile`-wrapped
  attention block, so MLX can fuse the materialisation into the
  compiled graph.

Both retries depend on the W2 stream-contract pre-work tracked in
[github issue #22](https://github.com/defai-digital/ax-engine/issues/22).
Until that is resolved, do not retry C5 — the kill criterion is firm.

## Effect on cumulative W1 result

W1.3 cumulative state remains C1+C3+C2+C4 (commits b7d14d11, 9c734ed4,
e22806ee, f2bed310): -41 ops/step, +0.25% decode tok/s vs the
2026-05-14 baseline. C5 contributes nothing.

## Artifacts

- `control-rep.json` — apples-to-apples control without C5
- `c5.json` — first C5-enabled rep
- `c5-rep2.json` — second C5-enabled rep, confirms regression is signal
