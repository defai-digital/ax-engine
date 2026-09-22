# 2026-09-22 M5: bf16 / 4-bit specialisation of the invariant affine qmv_fast microbatch kernel

Host `df-macbookpro-m5` (Apple M5 Max, 128 GB, macOS 27.0; the Flash Next
campaign host, not the Mac mini M4 Pro 64 GB SKU — campaign evidence, not
certification). Pack `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` @
`3e290738e96972307c6aeb9934ab170ca0eae1c1`. Harness
`scripts/bench_mlx_inference_stack.py` with the v7.5.3 peer-campaign
contract (`--ax-ngram-accel --ax-mtp-disable-ngram-stacking
--ax-qwen-linear-mtp-exact`, real prompt suites `flappy` and `long_code`,
256 greedy tokens, 2 warmups + 5 measured repetitions, 3 s cooldown, prefix
cache off). Baseline binary is `8064e97b` (v7.5.4 prep) built on the host
with Homebrew cargo 1.98.0; the patched binary is the same tree plus this
change, built the same way.

## What changed

Under the exact Qwen linear-attention MTP verifier profile, 4-bit affine
projections that are not served by a packed route take the repo-owned
`ax_invariant_affine_qmv_fast_v1` kernel at every verify width so that each
output row's arithmetic is identical for S=1..4. That kernel holds four f32
copies of every x value per token (`x_values[4][16]`), which puts the S=3..4
verify shape on a register cliff: 461 GB/s at S=1 falls to ~320 GB/s at S=4
on the 5120x10240 in_proj_qkv shape (MLX's own qmv_wide reaches ~455 GB/s
there, but with a different per-row reduction).

`ax_invariant_affine_qmv_fast_bf16_q4_v1` is the same kernel specialised for
bf16 inputs and 4-bit weights:

- x stays in registers as raw bf16 bit pairs (`uint x_bits[4][8]`, two
  `uint4` loads per token per K block) and is widened to f32 at use; the
  widening is exact;
- weights are unpacked to unscaled nibbles instead of pre-scaling x by
  `1/16^i`: `x * n` and `(x / 16^i) * (16^i * n)` are the same real number and
  each is rounded once, so every product is identical; the per-lane
  accumulation order, the per-block `scale * accum + x_sum * bias` fold, the
  OutT-typed 4-wide group sum and the `simd_sum` are unchanged.

Selection: bf16 x, 4-bit affine weights, the existing `qmv_fast` eligibility
(`OutDim % 8 == 0`, `InputDim % 512 == 0`, group size a multiple of 16), gate
`AX_MLX_INVARIANT_QMV_BF16_Q4` (default on, `=0` is the kill switch). A launch
failure falls back to the generic kernel, never off the invariant route.
`invariant_affine_bf16_q4_kernel_matches_generic_bit_exact` pins bit-identity
against the generic kernel for K in {512, 768, 1024, 1536}, group size 32 / 64
/ 128, f32 and bf16 sidecars (OutT f32 and bf16), NaN / inf / bf16-subnormal
inputs, S=1..4.

## Kernel micro-benchmark (`qmv_microbench.py`, MLX 0.32.2 `mx.fast.metal_kernel`)

Both kernel sources are extracted from `utils.rs` at run time. 16 launches
per eval over 8 distinct weight copies (single launches of a 20-50 MB
projection are dominated by dispatch and synchronisation overhead), medians
of 5 evals after 2 warmups. Bytes are packed weight + scales + biases.
The table is `qmv_microbench_run1.txt`; `qmv_microbench_run2.txt` is a
second run of the final source (same ratios within 0.03). The exploratory
logs of the losing variants are summarised below.

| shape (K x N, gs) | S | generic | bf16/q4 | ratio | MLX (different arithmetic) |
| --- | ---: | ---: | ---: | ---: | ---: |
| in_proj_qkv 5120 x 10240, 32 | 1 | 0.070 ms (467 GB/s) | 0.070 ms (471 GB/s) | 1.01 | 0.116 ms |
|  | 2 | 0.071 ms (459 GB/s) | 0.070 ms (471 GB/s) | 1.02 | 0.068 ms |
|  | 3 | 0.082 ms (399 GB/s) | 0.076 ms (432 GB/s) | 1.08 | 0.069 ms |
|  | 4 | 0.102 ms (322 GB/s) | 0.093 ms (354 GB/s) | 1.10 | 0.072 ms |
| down_proj 17408 x 5120, 64 | 1 | 0.100 ms (502 GB/s) | 0.099 ms (506 GB/s) | 1.01 | 0.098 ms |
|  | 2 | 0.103 ms (486 GB/s) | 0.103 ms (488 GB/s) | 1.00 | 0.102 ms |
|  | 3 | 0.131 ms (384 GB/s) | 0.126 ms (399 GB/s) | 1.04 | 0.116 ms |
|  | 4 | 0.169 ms (296 GB/s) | 0.158 ms (317 GB/s) | 1.07 | 0.135 ms |
| out_proj 6144 x 5120, 32 | 1 | 0.052 ms (376 GB/s) | 0.053 ms (373 GB/s) | 0.99 | 0.051 ms |
|  | 2 | 0.054 ms (366 GB/s) | 0.053 ms (373 GB/s) | 1.02 | 0.051 ms |
|  | 3 | 0.060 ms (327 GB/s) | 0.058 ms (341 GB/s) | 1.04 | 0.053 ms |
|  | 4 | 0.072 ms (273 GB/s) | 0.068 ms (288 GB/s) | 1.05 | 0.054 ms |
| in_proj_z 5120 x 6144, 32 | 1 | 0.053 ms (368 GB/s) | 0.051 ms (383 GB/s) | 1.04 | 0.051 ms |
|  | 2 | 0.057 ms (343 GB/s) | 0.054 ms (364 GB/s) | 1.06 | 0.053 ms |
|  | 3 | 0.063 ms (314 GB/s) | 0.056 ms (350 GB/s) | 1.11 | 0.053 ms |
|  | 4 | 0.071 ms (277 GB/s) | 0.067 ms (293 GB/s) | 1.06 | 0.054 ms |
| attn o 6144 x 5120, 32 | 1 | 0.054 ms (365 GB/s) | 0.054 ms (364 GB/s) | 1.00 | 0.051 ms |
|  | 2 | 0.055 ms (359 GB/s) | 0.053 ms (371 GB/s) | 1.04 | 0.052 ms |
|  | 3 | 0.060 ms (327 GB/s) | 0.058 ms (337 GB/s) | 1.03 | 0.052 ms |
|  | 4 | 0.071 ms (275 GB/s) | 0.066 ms (296 GB/s) | 1.07 | 0.054 ms |

Every row was bit-identical between the two kernels
(`bf16_q4==generic bits:True`). Variants that were measured and rejected on
the same harness: vector byte loads alone (no change), hoisting the nibble
unpack into a per-row f32 array (register spill, 4x slower at S=3..4), two
rows per simdgroup, a token-outer loop (re-reads the weight slice per token,
0.7x at S=4), threadgroup-staged x (0.6x at S=4), and widening the group sum
through f32 instead of the native `bfloat` reinterpret (0.95x at S=4).

## Server A/B (decode tok/s, median of the four cases' per-case medians)

Block order `head -> patched` on `flappy`, then `patched -> head` on
`long_code`. Raw: `flappy_head.json`, `flappy_patched.json`,
`long_code_head.json`, `long_code_patched.json`.

| suite | head | patched | ratio | verify eval / cycle | accepted tokens per case |
| --- | ---: | ---: | ---: | --- | --- |
| flappy | 82.47 (83.57 / 81.46 / 80.86 / 83.49) | 82.70 (83.69 / 81.72 / 81.38 / 83.78) | 1.003x | 44.57 -> 44.32 ms | identical (960 / 950 / 955 / 960) |
| long_code | 78.67 (77.71 / 78.33 / 79.01 / 79.87) | 78.76 (77.79 / 78.41 / 79.12 / 79.99) | 1.001x | 47.91 -> 47.87 ms | identical (950 / 950 / 945 / 955) |

Prefill is unchanged (flappy 767.9 -> 772.3 tok/s, long_code 856.7 -> 856.1
tok/s). **End-to-end the change is neutral on this pack, and the reason is
the profile, not the kernel.** `invariant_qmv_shape_trace.txt` (server run
with `AX_MLX_INVARIANT_QMV_TRACE=1`, one flappy prompt, 64 greedy tokens under
the same flags) lists every distinct shape that reached the invariant
qmv_fast route: all of them are `leading=1`. Under the default throughput
profile (`AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP`, which also engages
`AX_MLX_MTP_RELAXED_TARGET_VERIFY`) the S=2..4 target verify is built with
stock MLX arithmetic (`qmv_wide` / qmm) plus the MTPLX-derived split-K
verify QMM for projections at least 16384 wide, so the multi-row verify
shape this kernel was tuned for is only exercised by the exact
(non-relaxed) verifier profile. At S=1 the specialisation is neutral
(1.00-1.04x), which is what the A/B shows.

`splitk_microbench.py` / `splitk_microbench.txt` measure the split-K verify
QMM on the gate/up shape (5120 x 17408 and the fused 5120 x 34816, gs 64,
4-bit) against MLX at M=1..4: at M=4 the split-K kernel runs at 450-465 GB/s
on 17408 columns, at parity with MLX (450-452 GB/s), and on the fused width
MLX is faster (461 vs 370-395 GB/s). The relaxed verify's large projections
are therefore already close to the host's sustained bandwidth; the remaining
verify-cycle cost sits in the 5120-wide projections (MLX qmv_wide at
360-375 GB/s at S=4), the gated-delta / attention kernels and the draft
head, each a small slice. The kernel-level gain is real and bit-identical, so
the change lands as an exact kernel improvement with no performance claim.

## Greedy output identity (`identity_probe.py`)

Both binaries served the same eight prompts (the harness's `flappy` and
`long_code` token-id artifacts) twice each through `/v1/generate` (greedy,
256 tokens, `ignore_eos`) under the same MTP profile. All 8 x 256
output-token streams are identical across the two binaries and stable across
repeats (`identity_outputs.json`): **IDENTITY PASS**, as the kernel-level
bit-exactness test predicts.

## Review

The patch was reviewed by muse (no defects), DeepSeek V4 Pro (asked for
bf16-sidecar / special-value / group-size-128 test coverage, added; its
suggested f32 widening of the group sum was measured 0.95x and not taken,
see above) and Grok 4.6 (its group-sum concern assumed the generic kernel
sums in f32; both kernels use the same OutT-typed 4-wide sum).

## Scope

Only bf16-input, 4-bit affine projections on the invariant qmv_fast route
take the new kernel; 6/8-bit, f16/f32 inputs, MXFP4 and every packed or
matvec route are unchanged. No default, policy, certification, or route
changes.
