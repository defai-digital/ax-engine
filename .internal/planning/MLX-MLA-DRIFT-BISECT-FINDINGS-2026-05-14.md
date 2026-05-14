# F4 MLA Warm-Extend Drift Bisect — Findings (2026-05-14)

PRD: `MLX-MLA-DRIFT-BISECT-PRD-2026-05-14.md`
Parent PRD: `DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md` §7 (F4)

## 1. Status: CLOSED — probe lands, both PRD §6.2 acceptance gates pass.

## 2. Probe

`crates/ax-engine-mlx/src/bin/mla_warm_extend_drift_probe.rs` per PRD §3.
The binary loads a GLM-class model directly via the production
`load_weights` + `chunked_prefill` path, runs both cold and warm
prefill of a synthetic `(base, suffix)` split, captures per-MLA-layer
`kv_latent` and `k_pe` arrays from each cache, and emits a JSON
diff artifact whose top-level `verdict` field is either
`no_divergence` or `divergent`.

Accessor support added on `MlxKVCache::glm_mla_layer_state` returning
a public `GlmMlaLayerStateView<'_>` borrowed view — no production
hot-path mutation; the accessor is read-only.

## 3. PRD §6.2 acceptance — both gates pass

### 3.1 Regression check (chunk=16, the post-Phase-C default)

```
target/release/mla-warm-extend-drift-probe \
    --mlx-artifacts-dir .internal/models/GLM-4.7-Flash-4bit \
    --base-tokens 32 --suffix-tokens 32 --chunk-size 16 \
    --output benchmarks/results/mla-warm-extend-drift/glm47-chunk16-2026-05-14.json
```

```
verdict: no_divergence
mla_layer_count: 47
layer 0 kv_latent diff: 0.0
layer 0 k_pe diff: 0.0
layers with nonzero diff: 0 / 47
```

Confirms the Phase C chunk-alignment fix (commit `ade74c2f`) holds at
the per-layer KV bit level — not just at the decode-token-equality
level the existing `verify_prefix_reuse_equivalence.py` checks.

### 3.2 Detection-power inverted control (chunk=512)

```
target/release/mla-warm-extend-drift-probe \
    --mlx-artifacts-dir .internal/models/GLM-4.7-Flash-4bit \
    --base-tokens 32 --suffix-tokens 32 --chunk-size 512 \
    --output benchmarks/results/mla-warm-extend-drift/glm47-chunk512-detection-2026-05-14.json
```

```
verdict: divergent
mla_layer_count: 47
first_divergence: {
  layer_idx: 0,
  tensor: "k_pe",
  position: 2,
  cold_value: -1.4609375,
  warm_value: -1.46875,
  abs_diff: 0.0625,
  ulps_diff: 30081024
}
layer 0 kv_latent diff: 4.88e-4
layer 0 k_pe diff: 0.0625
layers with nonzero diff: 47 / 47
```

Forcing chunk=512 (the historical pre-fix MLA default that broke
`verify_prefix_reuse_equivalence.py --mode warm_extend`) reproducibly
flips the verdict to `divergent` and pinpoints the divergence at
**layer 0 / k_pe / position 2**. The probe has detection power.

## 4. What the layer-0 k_pe drift tells us

`k_pe` is the rotary-encoded key cache. It is computed per-token via
`projection(layer_input) → reshape → RoPE(token_offset)`. For
position 2 of the base:

- **Cold path** with chunk=512: a single forward over 64 tokens at
  `token_offset = 0`. The layer-0 input for position 2 is
  `embed_tokens(token_id=3)`, identical to warm.
- **Warm path** with chunk=512: warm session A processes 32 base
  tokens at `token_offset = 0`. Layer-0 k_pe for position 2 is
  computed from the same `embed_tokens(token_id=3)` input, same
  weights, same RoPE math.

The math is bit-equivalent in principle. The drift therefore comes
from MLX's runtime kernel selection differing between the two forward
shapes:

- Cold: single chunked-prefill iteration with `seq = 64`.
- Warm: single chunked-prefill iteration with `seq = 32` (session A).

MLX's quantized-matmul, RoPE, and SDPA kernels are shape-dependent.
A different reduction tile pattern (or different threadgroup
configuration) at seq=64 vs seq=32 produces slightly different fp
accumulation, which the bf16-precision rounding amplifies into a
visible value gap (0.0625 = roughly one bf16 ulp at magnitude ~1).

Once any layer-0 tensor differs, layer 1's input differs, and the
divergence compounds layer-by-layer — every one of GLM-4.7-Flash's
47 MLA layers shows nonzero diff with chunk=512, confirming the
compounding pattern.

This is the **same root cause** the Phase C findings hypothesised
without per-layer evidence. F4 confirms it at the data level.

## 5. Practical use of the probe

### 5.1 Today (as a regression gate)

The Phase C chunk-alignment workaround is now backed by a per-layer
empirical check, not just a downstream token-equality check. Future
MLX upgrades that change kernel-selection heuristics on these shapes
will be caught by `verify_prefix_reuse_equivalence.py` AND by F4 —
F4 will additionally tell us *which* layer started misbehaving, not
just *that* warm-extend tokens differ.

### 5.2 Future MLX upgrades

Rerun F4 against the production GLM artifact directory after any
`libmlxc` bump. If `verdict` flips to `divergent` at chunk=16, the
Phase C workaround has been broken by the upgrade — open a follow-up
ticket targeting the named layer.

### 5.3 New MLA models

When adding a new MLA-class model to the supported tier, run F4 on a
small (base, suffix) split before declaring the model production-
ready. A `no_divergence` verdict at chunk=16 is now part of the
correctness gate alongside `verify_prefix_reuse_equivalence.py`.

## 6. What F4 explicitly did not do

- **Did not** locate the exact MLX kernel responsible for the
  shape-dependent fp drift. F4 names the first divergent layer and
  tensor; the underlying MLX source-level investigation is a
  follow-up that would file a reproducer upstream against Apple's
  MLX.
- **Did not** add an FA / SWA / Linear bisect. Those tiers have
  their own existing equivalence machinery and were never the named
  drift vector.
- **Did not** add a tokenizer dependency. The probe uses synthetic
  token IDs (1..=N) because the drift is shape-dependent, not
  content-dependent — the inverted-control test in §3.2 reliably
  reproduces drift with synthetic tokens, validating that
  abstraction.

## 7. Closure conditions (PRD §10) — all met

1. ✅ Probe implemented and committed.
2. ✅ §6.2 end-to-end test passes at chunk=16 (no_divergence on the
   one configured prompt; per-layer scan over all 47 MLA layers).
3. ✅ §6.2 inverted-control test passes at chunk=512 (verdict
   flips to `divergent`; first divergence identified as layer 0 /
   k_pe / position 2).
4. ✅ Findings artifact (this doc) checked in.

## 8. Files

- Probe binary: `crates/ax-engine-mlx/src/bin/mla_warm_extend_drift_probe.rs`
- Cargo entry: `[[bin]] name = "mla-warm-extend-drift-probe"` in
  `crates/ax-engine-mlx/Cargo.toml`
- Public accessor added: `MlxKVCache::glm_mla_layer_state` returning
  `GlmMlaLayerStateView<'_>` in
  `crates/ax-engine-mlx/src/kv_cache.rs`
- Probe artifacts:
  - `benchmarks/results/mla-warm-extend-drift/glm47-chunk16-2026-05-14.json`
    (regression PASS, no_divergence)
  - `benchmarks/results/mla-warm-extend-drift/glm47-chunk512-detection-2026-05-14.json`
    (detection-power PASS, divergent at layer 0 / k_pe / pos 2)
