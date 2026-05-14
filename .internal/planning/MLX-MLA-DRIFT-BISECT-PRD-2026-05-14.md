# MLA Warm-Extend Drift Bisect Tooling PRD (F4)

Status: Open — implementation not yet scheduled.
Date: 2026-05-14
Owner: AX Engine
Parent PRD: DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md §7 (F4)
Depends on: MLX-PHASE-C-MULTITURN-BASELINE-FINDINGS-2026-05-14.md

## 1. Summary

The Phase C MLA fp-drift fix (commit `ade74c2f`) resolved
`verify_prefix_reuse_equivalence.py --mode warm_extend` 5/5 PASS on
GLM-4.7-Flash by aligning the prefill chunk size to the prefix-cache
block size (16). The fix is empirically sound but **does not localise
the original drift** beyond the layer-level hypothesis: MLX's SDPA
dispatches a slightly different kernel for `Q == K causal` vs
`Q < K causal-with-offset` shapes, and the fp accumulation order
differs by a ULP or two on bf16.

That hypothesis is consistent with the data but unverified at the
per-layer / per-tensor level. The chunk-alignment workaround is one
MLX upgrade away from being silently insufficient — a future MLX
release could introduce other shape-dependent kernel selections we
have not anticipated.

This PRD scopes a small **bisect probe** that, given a model and a
warm-extend workload, runs both cold and warm paths in the same
process, captures per-layer tensors at known checkpoints, and diffs
them. Output: the first (layer, tensor, position) tuple where cold
and warm disagree by more than a tunable ULP tolerance.

## 2. Goals & Non-Goals

### 2.1 Goals

G1. Localise the MLA warm-extend divergence to a specific layer and
tensor, on the `p2_medium_explain` reproduction or any future
regression.

G2. Provide a fast, reproducible feedback loop for any future MLX
upgrade — the same probe rerun on the new MLX version tells us
whether the chunk-alignment workaround still holds.

G3. Read-only instrumentation. No production hot-path mutation;
trace hooks behind a debug env flag.

### 2.2 Non-Goals

- **Automatic root-cause analysis.** The probe identifies the
  divergent layer; interpreting *why* (RoPE vs SDPA vs projection)
  is a follow-up human task.
- **Multi-architecture bisect.** FA, SWA, Linear paths have their
  own equivalence machinery (`verify_prefix_reuse_equivalence.py`)
  and are not the documented drift vector. Scope is MLA only.
- **Replacing `verify_prefix_reuse_equivalence.py`.** That harness
  is the production correctness gate; this probe is a diagnostic
  that runs when the harness has already failed (or when we want
  preventative coverage).

## 3. Probe design

### 3.1 Surface

A new Cargo bin:

```text
crates/ax-engine-mlx/src/bin/mla_warm_extend_drift_probe.rs
```

Modeled after `rmsnorm_fused_probe.rs` and `residual_rmsnorm_fused_probe.rs`,
but operating on a real loaded model rather than synthetic tensors.

### 3.2 Invocation

```
cargo run --release --bin mla-warm-extend-drift-probe -- \
    --mlx-artifacts-dir .internal/models/GLM-4.7-Flash-4bit \
    --base-prompt "Explain the difference between supervised and ..." \
    --suffix " Now also explain reinforcement learning." \
    --max-output-tokens 16 \
    --tolerance-ulps 4 \
    --output benchmarks/results/mla-warm-extend-drift/<date>.json
```

### 3.3 What it does

1. Load model artifacts via the existing `load_weights` /
   `validate_mlx_supported_manifest` path. Reuse the bench-style
   construction in `bench_main.rs`.
2. **Cold path:** tokenize `base + suffix` (with the same
   `--pad-to-block-size` rounding the verify harness uses). Run
   `chunked_prefill`. Before the function returns, capture the
   final KV cache state for every MLA layer:
   - `kv_latent[layer, position]` for all positions
   - `k_pe[layer, position]` for all positions
   Store in a `BTreeMap<(usize /* layer */, &'static str /* tensor */), Vec<f32>>` after
   `astype(f32)` for stable comparison.
3. **Warm path:** start fresh. Run `chunked_prefill` over `base`
   only. Store the resulting cache as a snapshot via the existing
   `MlxPrefixSnapshot` machinery. Build a new `RequestState`,
   restore the snapshot, then `chunked_prefill` the `suffix`.
   Capture the same per-layer tensors.
4. **Diff:** for each (layer, tensor) key, compute
   `max_abs_diff(cold, warm)` and `max_ulps_diff(cold, warm)`. The
   first key whose ULP diff exceeds `--tolerance-ulps` is the
   "first divergence" — emit it prominently.
5. **Output:** JSON artifact with per-layer / per-tensor diff
   summary, plus the headline `first_divergence` field.

### 3.4 Capture strategy

Two options:

**Option A — env-gated trace hook (preferred):** add a
`#[cfg(debug_trace)]`-style hook in `model.rs:glm_mla_attention_forward`
that, when an env flag like `AX_MLX_MLA_TRACE_KV=<json-path>` is set,
writes the post-append `kv_latent` and `k_pe` to a JSON file keyed by
`(layer_idx, token_offset, seq)`. The probe sets the env, runs cold,
moves the file aside, runs warm, then diffs both files.

Cost in production: zero (the env path is unset, the hook is a
no-op).

**Option B — bin owns the model:** the bin loads the model itself
and runs forward directly, bypassing the runner state machine. No
production code touched. More code in the bin, but the bin is
self-contained.

Recommend **Option B** for the initial probe. Option A is the
follow-up if the diagnostic surface needs to be reusable across other
debug bins.

### 3.5 Tolerance interpretation

bf16 has 7 mantissa bits. A ULP diff of 1 is the smallest
representable difference at that magnitude. A diff of ≤4 ULPs is
typically attributable to reduction-order differences in parallel
hardware; ≤16 ULPs is the conservative ceiling MLX-style libraries
target for "numerically equivalent". A diff > 16 ULPs in the same
position across cold and warm is a real algorithmic divergence.

The probe reports:
- per-layer `max_abs_diff` and `max_ulps_diff`
- per-position max ULP for the first divergent layer
- the cumulative ULP profile across layers (so we can see whether
  drift compounds or stays bounded)

## 4. Workload coverage

### 4.1 Canonical reproduction

`p2_medium_explain` from `verify_prefix_reuse_equivalence.py`'s
default corpus, with the same default suffix and
`--pad-to-block-size 16`. This is the historical drift the audit
comment cites; reproducing it is the minimum probe acceptance bar.

### 4.2 Future workloads

- `p1_short_factoid`, `p3_long_story`, `p4_code_request`, `p5_repetition_safe`
  for cross-prompt comparison.
- Pad to 512 and 2,048 to match the Phase C validation runs.

The probe accepts any base prompt + suffix; the canonical corpus is
a recommended starting set, not a hard-coded one.

## 5. Output schema

```
{
  "schema_version": "ax.mla_drift_bisect.v1",
  "captured_at_utc": "...",
  "model": {"id": "...", "artifacts_dir": "..."},
  "args": {"base_prompt_len": N, "suffix_len": N, "tolerance_ulps": 4, ...},
  "layers": [
    {
      "layer_idx": 0,
      "kv_latent_max_abs_diff": 0.0,
      "kv_latent_max_ulps_diff": 0,
      "k_pe_max_abs_diff": 0.0,
      "k_pe_max_ulps_diff": 0
    },
    ...
  ],
  "first_divergence": {
    "layer_idx": 7,
    "tensor": "k_pe",
    "position": 33,
    "cold_value": -0.123456,
    "warm_value": -0.123450,
    "abs_diff": 6e-6,
    "ulps_diff": 5
  } | null,
  "verdict": "no_divergence" | "divergent"
}
```

`verdict = "no_divergence"` is the expected outcome under the current
chunk-alignment defaults — the probe acts as a regression check. If
the verdict ever flips to `divergent`, the artifact JSON points
straight at the layer to investigate.

## 6. Test plan

### 6.1 Probe self-test (Rust unit)

- Run probe on a synthetic 2-layer model with identical cold/warm
  paths; verify `verdict = no_divergence`.
- Inject a deliberate fp perturbation in the warm path (multiply
  layer-3 `kv_latent` by `1 + 1e-3`); verify probe reports
  `first_divergence.layer_idx = 3` with the expected magnitude.

### 6.2 End-to-end (real models)

On GLM-4.7-Flash with the production runner defaults (chunk=16,
no env overrides), verdict must be `no_divergence` on all five
canonical prompts. This is the regression-coverage acceptance bar.

Re-run with `AX_MLX_MLA_PREFILL_CHUNK=512` to deliberately disable
the chunk-alignment safety; verdict must flip to `divergent` for at
least `p2_medium_explain`. This confirms the probe actually has
detection power — a probe that always says "no divergence" is
worthless.

## 7. Telemetry / artifact convention

Results land under
`benchmarks/results/mla-warm-extend-drift/`. The artifact convention
mirrors the existing `prefix-reuse-equivalence/` directory:
per-model dated JSONs, summary in the parent findings doc.

## 8. Risks

R1. **Capture overhead distorts the cold path's numerics.** If we
add a `tracing::trace` hook inside `glm_mla_attention_forward`, the
hook itself runs at production cost (env-gated to no-op when off,
but cold benchmark numbers in the probe artifact are non-canonical).
Mitigation: the probe's wall-clock numbers are not the headline; the
ULP diff is. Add a clear disclaimer in the artifact docs.

R2. **MlxArray eager materialization affects the trace.** Capturing
tensors mid-graph forces eval. The compiled lazy-graph may then run
in a slightly different order than the production graph. The
captured cold/warm states are still both eager-evaluated, so their
diff is meaningful; but the absolute values may differ slightly
from a production-eval run. Acceptable.

R3. **bf16 ULP comparison subtlety.** Comparing bf16 by casting to
f32 loses no bits (bf16 is a strict subset of f32), so the ULP diff
is well-defined. Document the convention in the probe's source.

R4. **MLA-specific.** The probe will not catch fp-drift in FA / SWA
/ Linear paths. Those tiers have their own equivalence harness
(`verify_prefix_reuse_equivalence.py`); the bisect tool is named for
MLA because MLA is the only tier where chunk alignment is the
known-required safety.

## 9. Milestones

M1. **Probe skeleton + synthetic self-test** (1 day).
  - Cargo bin loads GLM-4.7-Flash weights, runs cold and warm
    forwards, captures per-layer tensors, emits JSON.
  - Self-test on a deliberate-perturbation injection.

M2. **Real-model run + canonical regression coverage** (1 day).
  - Run on all five canonical prompts at pad sizes 16, 512, 2048.
  - Confirm verdict `no_divergence` at chunk=16 default.
  - Confirm verdict `divergent` at `AX_MLX_MLA_PREFILL_CHUNK=512`.

M3. **Docs + PRD closure** (half day).
  - `crates/ax-engine-mlx/Cargo.toml` adds the bin.
  - `.internal/planning/MLX-MLA-DRIFT-BISECT-FINDINGS-<date>.md`
    records the M2 outcomes.

Total: roughly 2.5 days. This is the cheapest item in the parent
PRD's follow-up list.

## 10. Closure conditions

This PRD closes when:

1. The probe is implemented and committed.
2. The §6.2 end-to-end test passes on GLM-4.7-Flash at default
   settings (verdict `no_divergence` on all canonical prompts).
3. The §6.2 inverted-control test passes (verdict `divergent` when
   chunk=512 is forced), proving the probe has detection power.
4. The findings artifact is checked in.

If M2's verdict flips to `divergent` even at chunk=16 defaults, the
PRD does not close — instead it triggers a follow-up PRD targeting
the specific divergent layer.

---

**Status:** Open — implementation not yet scheduled. ~2.5 days of
work. Highest leverage is as preventative coverage: rerun this probe
on every MLX upgrade to catch chunk-alignment-workaround regressions
before they reach `verify_prefix_reuse_equivalence.py`.
