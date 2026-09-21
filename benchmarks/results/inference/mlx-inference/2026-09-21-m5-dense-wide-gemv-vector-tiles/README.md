# 2026-09-21 M5: dense wide GEMV vector tiles (exact-profile lm_head projection)

Host `df-macbookpro-m5` (Apple M5 Max, 128 GB, macOS 27.0 / 26A428; the
Flash Next campaign host, not the Mac mini M4 Pro 64 GB SKU — campaign
evidence, not certification). Pack
`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` @
`3e290738e96972307c6aeb9934ab170ca0eae1c1` (dense bf16 lm_head
`[5120, 248320]`, 2.54 GB). Harness `scripts/bench_mlx_inference_stack.py`
with the v7.5.3 peer-campaign contract (`--ax-ngram-accel
--ax-mtp-disable-ngram-stacking --ax-qwen-linear-mtp-exact`, real prompt
suites `flappy` and `long_code`, 256 greedy tokens, 2 warmups + 5 measured
repetitions, 3 s cooldown, prefix cache off). Baseline binary is the
v7.5.3 release build at `c5d22c4a` already on the host; the patched binary
is the same tree plus this change, built with the host's Homebrew cargo
1.98.0 like the campaign binaries.

## What changed

Under the exact MTP profile every dense-head projection — the S=1
MTP-off single and the S=2..8 verify window — runs through the
repo-owned Metal kernel `ax_dense_wide_gemv_wt` so that each row's
arithmetic is identical regardless of `S`. The scalar form issued one
2-byte load per thread per `k` and reached only ~260-290 GB/s on the
2.54 GB head. The bf16 tiles in this change let each thread own 8 (or 4)
adjacent output columns loaded as one 16-byte (8-byte) word per `k`; per
column the f32 `fma` chain is unchanged, so the output is bit-identical
to the scalar kernel (pinned by
`dense_wide_gemv_bf16_vector_tiles_match_scalar_bit_exact`). Tile
selection: 8 columns for `OutDim % 8 == 0` and `Leading <= 6`, 4 columns
for `OutDim % 4 == 0` or wider rows, scalar otherwise (non-bf16 weights
keep the scalar form).

## Kernel micro-benchmark (`gemv_microbench.py`, MLX 0.32.2 `mx.fast.metal_kernel`)

Same kernel source as the Rust kernels, 2.54 GB bf16 weight, medians of
8 timed launches after 2 warmups. `gemv_microbench.txt` is the raw log.

| rows | MLX matmul (different arithmetic) | scalar | 4 columns | 8 columns |
| --- | ---: | ---: | ---: | ---: |
| S=1 | 4.57-5.41 ms | 9.76 ms (261 GB/s) | 4.90-5.03 ms | **4.81-4.87 ms (522-528 GB/s)** |
| S=4 | 4.79-5.14 ms | 8.68-8.94 ms | 4.97-5.98 ms | **4.64-4.75 ms (535-547 GB/s)** |
| S=6 | — | 8.12 ms | 5.55 ms | **4.74 ms** |
| S=7 | — | 8.34 ms | **5.66 ms** | 17.12 ms (register spill) |
| S=8 | 5.16 ms | 7.92 ms | **5.22-5.63 ms** | 16.46-17.36 ms (register spill) |

Every vector variant was bit-exact against the scalar kernel at every
row count. The spill at 7-8 rows with 8 columns (56-64 f32 accumulators)
is why the 8-column tile is capped at `Leading <= 6`; the A/B below ran
with an uncapped build, which is identical for the S=1 and S=4 shapes it
exercised.

## Server A/B (decode tok/s, median of the four cases' per-case medians)

Block order `baseline → patched` on `flappy`, then `patched → baseline`
on `long_code`, to spread thermal drift. Raw: `flappy_baseline.json`,
`flappy_patched.json`, `long_code_baseline.json`, `long_code_patched.json`.

| suite | baseline | patched | ratio | verify eval / cycle | accepted tokens per case |
| --- | ---: | ---: | ---: | --- | --- |
| flappy | 76.25 (77.18 / 75.38 / 74.73 / 77.12) | **82.59** (83.60 / 81.58 / 81.16 / 83.64) | **1.083×** | 47.2 → 43.3 ms | identical (960 / 950 / 955 / 960) |
| long_code | 72.80 (71.87 / 72.56 / 73.04 / 73.85) | **78.68** (77.65 / 78.31 / 79.04 / 79.91) | **1.081×** | 48.1 → 44.2 ms | identical (950 / 950 / 945 / 955) |

Prefill is unchanged (flappy 767.6 → 772.2 tok/s, long_code 856.8 →
857.3 tok/s): the tiles only touch the decode-shaped projection. The
~4 ms/cycle saving matches the micro-benchmark delta on the 2.54 GB
head, and the MTP telemetry (drafted / accepted counts, full-accept
steps) is identical between arms in every case.

## Greedy output identity (`identity_probe.py`)

Both binaries served the same eight prompts (the harness's `flappy` and
`long_code` token-id artifacts) twice each through `/v1/generate`
(greedy, 256 tokens, `ignore_eos`) under the same MTP profile.

All 8 × 256 output-token streams are identical across the two binaries
and stable across repeats (`identity.log`, raw streams in
`identity_outputs.json`):

- `real-flappy-flappy_collision_checks-gen-256-75a7041d3bb7.json: tokens=256 repeat_stable=True baseline==patched=True`
- `real-flappy-flappy_pipes-gen-256-fc913ab80026.json: tokens=256 repeat_stable=True baseline==patched=True`
- `real-flappy-flappy_score_gates-gen-256-45ce81d30d46.json: tokens=256 repeat_stable=True baseline==patched=True`
- `real-flappy-flappy_sound_channels-gen-256-baa042e581ac.json: tokens=256 repeat_stable=True baseline==patched=True`
- `real-long_code-long_code_api_audit-gen-256-3cd352119f3d.json: tokens=256 repeat_stable=True baseline==patched=True`
- `real-long_code-long_code_c_audit-gen-256-be6c30699699.json: tokens=256 repeat_stable=True baseline==patched=True`
- `real-long_code-long_code_sql_schema-gen-256-e6224a6d73ee.json: tokens=256 repeat_stable=True baseline==patched=True`
- `real-long_code-long_code_test_stubs-gen-256-a9ba17ecb869.json: tokens=256 repeat_stable=True baseline==patched=True`

Result: **IDENTITY PASS** — the bit-identical projection reproduces the
baseline greedy stream, as the kernel-level test predicts.

## Scope

Only packs with a dense (unquantized) lm_head and the exact Qwen linear
MTP profile take this path; quantized heads (MXFP4 / 8-bit AXQ) are
unaffected. No default, policy, certification, or route changes.
