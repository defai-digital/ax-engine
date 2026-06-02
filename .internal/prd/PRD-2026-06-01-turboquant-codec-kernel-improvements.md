# PRD: TurboQuant Codec and Kernel Improvements

**Status**: Implemented in code; model, short-decode, and promotion evidence pending
**Date**: 2026-06-01
**Current ADR**: `.internal/adr/ADR-016-turboquant-codec-kernel-improvements.md`
**Scope**: `crates/ax-engine-mlx/src/turboquant.rs`, `crates/ax-engine-mlx/src/turboquant_metal.rs`, `crates/ax-engine-mlx/src/kv_cache.rs`, `crates/ax-engine-core/src/runner.rs`
**Reference implementations**: `.internal/reference/mlx-turboquant/`, `.internal/reference/turboquant-mlx/`

---

## Problem

TurboQuant K8V4 is feature-complete at the codec and kernel layer, but two
upstream reference implementations (`mlx-turboquant` and `turboquant-mlx`)
demonstrate that our codec uses sub-optimal quantization primitives and our
Metal kernels leave significant throughput on the table.

### Algorithmic gap

`encode_key_vector` (`turboquant.rs:2373-2398`) uses a **plain Hadamard**
transform and **uniform centroids** (`key_centroids` at
`turboquant.rs:2492-2503`). The PolarQuant paper and both reference
implementations show that:

1. Post-Hadamard coordinates follow a Gaussian distribution only after applying
   a **random ±1 diagonal** before the butterfly (Randomized Hadamard
   Transform, RHT). Without random signs, the coordinate distribution is
   model-dependent and not guaranteed to match the codebook's target
   distribution.

2. **Lloyd-Max optimal codebooks** for N(0,1) outperform uniform spacing at
   every bit width ≤ 4. Per `.internal/reference/mlx-turboquant/REPORT.md`,
   switching from uniform to Lloyd-Max at 3-bit improves Qwen3-4B from 0.601 to
   0.957 cosine similarity.

These two changes are interdependent: Lloyd-Max centroids are only optimal when
the post-rotation distribution is Gaussian, which requires RHT.

### Throughput gap

`turboquant_metal.rs:735-794` computes QK scores with a scalar loop over
`HEAD_DIM` threads per head. The reference `metal_kernels_v4.py:96-112` uses
`simd_sum` to reduce barrier count from log2(dim) to 1 (1.85x kernel speedup on
dim=128).

`turboquant_metal.rs:96-109` computes pre-rotated queries on CPU via
`hadamard_in_place`. The reference `turboquant-mlx/metal.py` has a fused Metal
encode kernel that does norm → normalize → signs → WHT → centroid → pack in a
single dispatch.

The runner re-dequantizes the entire cold window on every decode step. The
reference `turboquant-mlx/cache.py:192-218` maintains an fp16 incremental
buffer and only dequantizes new positions on short decode steps (≤4 tokens).

### Coverage gap

No sparse V attention, no V-only cache mode, no fractional-bit presets, no
state serialization. The reference implementations provide all four.

## Goals

- **G1**: Match the reference 4-bit quality floor across supported model
  families. The reference floor is 0.949 for Qwen3-1.7B, so pass/fail evidence
  must report per-model cosine instead of rounding this requirement to 0.95.
- **G2**: Achieve ≥2x decode throughput improvement on short decode steps (1-4
  new tokens) via incremental decode buffer.
- **G3**: Achieve ≥1.5x fused decode kernel speedup on head_dim=128 via
  `simd_sum` reduction.
- **G4**: Sparse V attention at threshold=0 produces identical output to dense
  path within float32 noise.
- **G5**: V-only cache mode (K16V4) produces identical output to full-precision
  K path for K-sensitive models.

## Non-Goals

- No changes to the production readiness gate structure. The
  `long_context_benchmark_artifact` requirement stays unchanged.
- No new model family wiring. MoE models (llama4, glm4_moe_lite, deepseek_v3)
  remain out of scope.
- No changes to existing K8V4 preset defaults or quality gate thresholds.
- No 2-bit presets. Reference evidence shows 2-bit is unreliable even with
  Lloyd-Max (Qwen3-1.7B at 3-bit already fails at 0.128 cosine).
- No removal of existing Metal kernel variants. New kernels are additive.

## Current Evidence

### Code locations

| Component | File | Lines | Function |
|---|---|---|---|
| Plain Hadamard | `turboquant.rs` | 2687-2710 | `hadamard_in_place` |
| Uniform centroids | `turboquant.rs` | 2492-2503 | `key_centroids` |
| Linear centroid index | `turboquant.rs` | 2816-2821 | `uniform_centroid_index` |
| CPU query rotation | `turboquant_metal.rs` | 96-109 | `rotated_query_values_from_flat` |
| Scalar-loop QK scores | `turboquant_metal.rs` | 735-794 | `FUSED_COLD_DECODE_KERNEL_SOURCE` |
| simd_sum score kernel | `turboquant_metal.rs` | 863-891 | `FUSED_COLD_DECODE_SCORE_KERNEL_SOURCE` |
| Production gate | `turboquant.rs` | 41-48 | `mlx_shadow_fused_kernel()` |
| Quality gates | `turboquant.rs` | 228-230 | `STRICT_DEBUG`, `REFERENCE_K8V4`, `RESEARCH_LOOSE` |

### Reference benchmarks

Per `.internal/reference/mlx-turboquant/REPORT.md`:

| Model | 3-bit uniform | 3-bit Lloyd-Max | 4-bit Lloyd-Max |
|---|---|---|---|
| Llama 3.2-3B | N/A | 0.988 | 0.997 |
| Qwen3-4B | 0.601 | 0.957 | 0.995 |
| Qwen3-1.7B | -0.043 | 0.128 | 0.949 |

Per `.internal/reference/turboquant-mlx/README.md`:
- K8+V4 mixed-quant on Qwen 2.5 7B at 32K: 18% memory savings, identical quality.
- simd_sum: 1.85x kernel speedup on dim=128.
- V-only cache: ~40% KV memory savings at 32K context on 7B.

## Plan

### D1: Randomized Hadamard Transform

Add `TurboQuantRotationSigns` struct with deterministic ±1 diagonal per head
dimension. Apply signs before Hadamard in encode, apply signs after Hadamard in
decode (inverse RHT: Hadamard is self-inverse, signs are self-inverse).

**Risk**: LOW. Deterministic seeds ensure encode/decode consistency. Plain
Hadamard is the special case where all signs = +1.

### D2: Lloyd-Max Gaussian Codebooks

Replace `key_centroids` with precomputed Lloyd-Max optimal centroids for N(0,1)
at the supported low-bit widths. AX uses a normalized WHT, so raw N(0,1)
Lloyd-Max centroids are scaled by `1/sqrt(head_dim)` before lookup. Keep
uniform centroids for 7-bit and 8-bit. Add boundary-based centroid lookup
replacing linear mapping.

**Risk**: LOW. Backward-compatible: uniform centroids are a special case of
Lloyd-Max with equal spacing.

**Depends on**: D1. Lloyd-Max codebooks are only optimal when post-rotation
distribution is Gaussian, which requires RHT.

### D3: simd_sum Reduction in Fused Decode Kernel

Replace scalar-loop dot product in `FUSED_COLD_DECODE_KERNEL_SOURCE` with
`simd_sum` across 32-lane SIMD groups, matching the existing
`FUSED_COLD_DECODE_SCORE_KERNEL_SOURCE` pattern.

**Risk**: LOW. The score kernel (`turboquant_metal.rs:863-891`) already uses
`simd_sum` successfully. Applying the same pattern to the main decode kernel
is a direct port.

### D4: Incremental Decode Buffer

Add fp16 dequant buffers to `TurboQuantCompressedBlockBuffer`. On short decode
steps (≤4 new tokens), only dequantize new positions and append to the buffer.

**Risk**: MEDIUM. Requires careful offset tracking and buffer expansion logic.
Must invalidate buffer on trim/resize.

### D5: Fused Metal Encode Kernel

Add a Metal key encode kernel matching the key portion of
`turboquant-mlx/metal.py:FUSED_QUANTIZE_KERNEL`: norm → normalize → signs →
normalized WHT → centroid → pack in a single dispatch. AX values use the
existing affine group codec, so the reference WHT value codec is not grafted
onto the runtime value path.

**Risk**: MEDIUM. New Metal kernel. Must match CPU reference exactly.

### D6: Sparse V Attention

Add sparse thresholding to the two-stage AX affine value-sum Metal path:
positions with normalized post-softmax weight below the configured threshold
are skipped. Threshold 0 must match the dense path within float32 noise. The
reference WHT sparse-V butterfly trick remains a reference design for a future
WHT value codec, not the current affine V storage.

**Risk**: MEDIUM. New Metal kernel. Correctness guarantee: threshold=0 must
match dense path within float32 noise.

### D7: V-Only Cache Mode

Add `TurboQuantPreset::K16V4` (16-bit keys = full precision, 4-bit values).
Expose K16V4 as the fallback preset selected by promotion evidence when the
K8V4 quality gate fails on K-sensitive models.

**Risk**: LOW. Additive preset. K stays in full precision, so no quality risk
for keys.

### D8: Fractional Bit Presets

Add 3.5-bit value support via channel split (half value channels at 4-bit, half
at 3-bit). Add K7V4 as a near-K8 key-memory tradeoff.

**Risk**: LOW. Additive presets.

### D9: State Serialization

Add `state()` and `meta_state()` methods to `TurboQuantCompressedBlockBuffer`,
plus v2 `MlxKVCache` serialization of TurboQuant shadow storage so prompt cache
payloads can survive process restart without dropping compressed state.

**Risk**: LOW. Read-only serialization. No changes to encode/decode paths.

## Acceptance Criteria

- [x] Focused `ax-engine-mlx` TurboQuant tests pass after codec changes.
- [ ] Clippy clean.
- [ ] D1+D2: model evidence records the 4-bit reference quality floor per family.
- [x] D3: ≥1.5x speedup on fused decode kernel microbenchmark (head_dim=128).
- [ ] D4: ≥2x decode speedup on short decode steps (1 new token, 1024 cold).
- [x] D6: Sparse V threshold=0 matches dense path within float32 noise.
- [x] D7: K16V4 stores keys as fp16 and promotion readiness selects it after K8V4 quality failure.
- [x] D9: TurboQuant compressed state round-trips through buffer and KV-cache serialization.
- [ ] No regression on existing K8V4 production preset quality gates.

## Validation

```text
cargo test -p ax-engine-mlx turboquant
cargo test -p ax-engine-core
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --quiet --no-fail-fast
```

Model-dependent validation must record model family, prompt hash, generation
length, seed, host, git state, cosine similarity vs FP16, compression ratio,
decode tok/s, route metadata, and fallback counters.

## Evidence Artifacts

| Artifact | Path |
|---|---|
| Reference benchmarks | `.internal/reference/mlx-turboquant/REPORT.md` |
| Reference K/V finding | `.internal/reference/turboquant-mlx/README.md` |
| Plain Hadamard | `crates/ax-engine-mlx/src/turboquant.rs:2687-2710` |
| Uniform centroids | `crates/ax-engine-mlx/src/turboquant.rs:2492-2503` |
| Scalar-loop decode kernel | `crates/ax-engine-mlx/src/turboquant_metal.rs:735-794` |
| simd_sum score kernel (existing) | `crates/ax-engine-mlx/src/turboquant_metal.rs:863-891` |
| Reference Metal encode | `.internal/reference/turboquant-mlx/turboquant_mlx/metal.py` |
| Reference sparse V | `.internal/reference/turboquant-mlx/turboquant_mlx/sparse_v.py` |
| D3 head_dim=128 microbench | `benchmarks/results/turboquant/quality-runs/20260602T-prd-d3-microbench/microbench-head128-cold8192.json` |
| PRD completion report | `benchmarks/results/turboquant/quality-runs/20260602T-prd-d3-microbench/prd-completion-after-d3.json` |
