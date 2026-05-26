# N-gram Throughput Goal Progress - 2026-05-26

Goal: make AX Engine default n-gram decode throughput at least 18% faster than
`mlx_lm` on every README model row.

## Current status

The goal is not achieved. The remaining blocker is Qwen 3.6 27B on the
`mlx_lm.benchmark` random-token prompt contract. The failing rows are not
low-acceptance n-gram rows; they usually enter the linear-attention no-draft
direct fallback with zero draft attempts.

Representative failing row:

- Model: `mlx-community/Qwen3.6-27B-4bit`
- Prompt: 128 tokens
- Current `mlx_lm` reference: 33.98 tok/s
- Current AX n-gram after the RMSNorm+gate Metal route: 34.19 tok/s
- Ratio: +0.6% versus `mlx_lm`, still far below the +18% target

Current post-commit sweep on commit `8ceae3ed9748846f2b4e3449e7a04462683c57c9`
confirms the blocker is the Qwen 3.6 27B family, not just one stale README row.
The sweep used `--skip-mlx-lm --ax-ngram-accel --no-build-ax-engine`,
`generation_tokens=128`, `repetitions=1`, and `cooldown=0`; percentages below
compare against the current README `mlx_lm` reference rows from
`benchmarks/results/mlx-inference/2026-05-26-direct-mode-clean-refresh/`.

| Model | Prompt | AX n-gram tok/s | `mlx_lm` tok/s | Delta | 1.18x target | Drafts accepted | Effective route |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen 3.6 27B 4-bit | 128 | 34.192 | 33.980 | +0.6% | 40.096 | 0 | `linear_no_draft_direct_pipeline_fallback` |
| Qwen 3.6 27B 4-bit | 512 | 34.157 | 33.908 | +0.7% | 40.011 | 1 | `ngram_verified_bonus_tokens` |
| Qwen 3.6 27B 4-bit | 2048 | 33.831 | 33.459 | +1.1% | 39.482 | 1 | `ngram_verified_bonus_tokens` |
| Qwen 3.6 27B 5-bit | 128 | 28.226 | 21.324 | +32.4% | 25.162 | 2 | `ngram_verified_bonus_tokens` |
| Qwen 3.6 27B 5-bit | 512 | 28.335 | 28.144 | +0.7% | 33.210 | 0 | `linear_no_draft_direct_pipeline_fallback` |
| Qwen 3.6 27B 5-bit | 2048 | 28.126 | 27.827 | +1.1% | 32.836 | 0 | `linear_no_draft_direct_pipeline_fallback` |
| Qwen 3.6 27B 6-bit | 128 | 24.891 | 22.993 | +8.3% | 27.131 | 0 | `ngram_attempted_no_accept_fallback` |
| Qwen 3.6 27B 6-bit | 512 | 25.306 | 24.776 | +2.1% | 29.236 | 2 | `ngram_verified_bonus_tokens` |
| Qwen 3.6 27B 6-bit | 2048 | 23.990 | 24.623 | -2.6% | 29.055 | 0 | `linear_no_draft_direct_pipeline_fallback` |
| Qwen 3.6 27B 8-bit | 128 | 18.774 | 18.458 | +1.7% | 21.780 | 0 | `linear_no_draft_direct_pipeline_fallback` |
| Qwen 3.6 27B 8-bit | 512 | 18.590 | 18.616 | -0.1% | 21.966 | 0 | `linear_no_draft_direct_pipeline_fallback` |
| Qwen 3.6 27B 8-bit | 2048 | 18.450 | 18.342 | +0.6% | 21.644 | 0 | `linear_no_draft_direct_pipeline_fallback` |

Only Qwen 3.6 27B 5-bit at prompt=128 currently clears the +18% goal. All other
Qwen 3.6 27B default n-gram rows remain at direct-fallback parity or worse.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-4bit.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-5bit.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-6bit.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-8bit.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-rms-full-gate-postcommit-sweep/qwen3_6-27b-4bit.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-rms-full-gate-postcommit-sweep/qwen3_6-27b-5bit.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-rms-full-gate-postcommit-sweep/qwen3_6-27b-6bit.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-rms-full-gate-postcommit-sweep/qwen3_6-27b-8bit.json`

## Code changes kept

`crates/ax-engine-mlx/src/runner.rs` now bootstraps the direct double-buffer
pipeline at the prefill boundary when n-gram is disabled for the request due to
linear-attention initial no-draft. This aligns no-draft fallback timing with the
direct greedy baseline, avoiding first decode-step bootstrap inside the measured
decode interval.

`crates/ax-engine-mlx/src/linear_attention_ops.rs` now routes GatedDelta
single-token decode (`seq=1`) through a specialized Metal kernel instead of the
generic prompt kernel. The specialized kernel avoids the generic
`CacheCapacity=512` threadgroup arrays and prompt-time loop while preserving the
same update equations.

`crates/ax-engine-mlx/src/linear_attention_ops.rs` also adds a Qwen
linear-attention decode post-input Metal route for `seq=1`. The route fuses the
single-token conv-state update, depthwise conv, SiLU, Q/K/V split, Q/K RMSNorm,
and Q/K scaling into one Metal kernel. `model/profile.rs`, `runner.rs`, and
`scripts/bench_mlx_inference_stack.py` expose telemetry counters:

- `ax_mlx_qwen_linear_attention_decode_post_input_metal_attempts`
- `ax_mlx_qwen_linear_attention_decode_post_input_metal_hits`
- `ax_mlx_qwen_linear_attention_decode_post_input_metal_fallbacks`
- `ax_mlx_qwen_linear_attention_decode_post_input_metal_profile_blocked`

`crates/ax-engine-mlx/src/linear_attention_ops.rs` now also fuses Qwen
linear-attention output RMSNorm plus SiLU gate into one Metal kernel. The
previous default path already fused the post-RMSNorm gate, but still submitted
RMSNorm as a separate MLX op. The new route keeps the same
`AX_MLX_LINEAR_ATTENTION_RMS_NORM_GATE_METAL` kill-switch and falls back to the
old RMSNorm + gate-Metal path for unsupported shapes or dtypes.

`crates/ax-engine-mlx/src/runner.rs` now lets a linear-attention request that
started as `LinearInitialNoDraft` re-open n-gram if later generated output
builds a valid draft. Linear-attention cooldown steps now use the direct
double-buffer pipeline for greedy fallback instead of slower single-decode
steps. The Qwen random-token p128 probe did not produce drafts, so this is not a
throughput win on the current blocker, but it removes the earlier re-enable
regression mode where failed retries paid single-decode cooldown.

Validation:

- `cargo fmt --check`
- `cargo test -p ax-engine-mlx gated_delta --quiet`
- `cargo test -p ax-engine-mlx direct_pipeline --quiet`
- `cargo test -p ax-engine-mlx linear_attention --quiet`
- `cargo test -p ax-engine-mlx linear_attention_reenable --quiet`
- `python3 -m unittest scripts.test_bench_mlx_inference_stack -v`
- `cargo build -p ax-engine-server --release`

Additional validation for the RMSNorm+gate Metal route:

- `cargo test -p ax-engine-mlx rms_norm_full_gate --quiet`
- `cargo test -p ax-engine-mlx rms_norm_gate --quiet`
- `cargo test -p ax-engine-mlx linear_attention --quiet`
- `cargo build -p ax-engine-server --release`

Probe artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-initial-direct-bootstrap-probe/qwen3_6-27b-4bit.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-gateddelta-decode-probe/qwen3_6-27b-4bit-p128.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-post-input-metal-telemetry-probe/qwen3_6-27b-4bit-p128.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-post-input-metal-telemetry-probe/qwen3_6-27b-4bit-p128-metal-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-reenable-direct-cooldown-probe/qwen3_6-27b-4bit-p128.json`

The GatedDelta decode kernel probe was positive but small: Qwen 3.6 27B 4-bit
p128 measured 33.35 tok/s in a 1-repetition dirty-code probe, versus about
33.29 tok/s in the comparable post-bootstrap probe. This is useful directionally
but not close to the 1.18x completion threshold.

The decode post-input Metal route is a confirmed hit on Qwen 3.6 27B 4-bit
p128:

- Metal on: 34.13 tok/s,
  `decode_post_input_metal_attempts=48`, `hits=48`, `fallbacks=0`
- Metal off: 33.34 tok/s,
  `decode_post_input_metal_attempts=0`, `hits=0`
- Local A/B delta: about +2.35%

The later re-enable/direct-cooldown probe measured 34.15 tok/s, but still had
`ax_ngram_draft_attempts=0`, `ax_ngram_request_disabled_steps=127`, and
`ax_mlx_bonus_tokens=0`. It therefore did not move the blocker beyond direct
fallback throughput.

The RMSNorm+gate Metal route is directionally positive on focused Qwen 3.6 27B
blocker probes, but still far below the +18% goal:

- Qwen 3.6 27B 4-bit p128: 34.243 tok/s, up from 34.111 tok/s in the current
  sweep (+0.39%), still below the 40.090 tok/s target.
- Qwen 3.6 27B 8-bit p128: 18.774 tok/s, up from 18.155 tok/s (+3.4%) and now
  slightly above the 18.665 tok/s `mlx_lm` row, but still below the 22.025
  tok/s target.
- Qwen 3.6 27B 6-bit p512: 25.362 tok/s, up from 24.042 tok/s (+5.5%) and now
  above the 24.776 tok/s `mlx_lm` row, but still below the 29.236 tok/s target.

A follow-up kernel-launch probe changed the retained RMSNorm+gate Metal route
to launch one threadgroup lane per `HeadDim` instead of 256 lanes, because the
Qwen blocker has `HeadDim=128` and the previous launch leaves half the lanes
idle. This did not help the real p128/g128 blocker:

- Qwen 3.6 27B 4-bit p128/g128: 34.244 tok/s
- clean n-gram recheck context: 34.210 tok/s
- current `mlx_lm` reference: 33.980 tok/s
- target: 40.096 tok/s

The result is below the 2% acceptance threshold and far below the 1.18x goal.
The code probe was removed.

The sequential post-commit sweep on `8ceae3ed9748846f2b4e3449e7a04462683c57c9`
kept the same conclusion. The best non-passing improvement is Qwen 3.6 27B
6-bit p128 at +8.3%; most other failing rows are still direct-fallback parity.
An earlier attempt to run the 4-bit and 5-bit sweeps concurrently produced
obviously contaminated GPU results and was overwritten by sequential reruns.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-rms-full-gate-probe/qwen3_6-27b-4bit-p128.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-rms-full-gate-probe/qwen3_6-27b-8bit-p128.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-rms-full-gate-probe/qwen3_6-27b-6bit-p512.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-rms-full-gate-launch-probe/qwen3_6-27b-4bit-p128-g128-rms-full-gate-headdim.json`

## Diagnostics

The upstream benchmark contract is the `mlx_lm.benchmark` random-token path:
`mx.random.seed(0)` followed by `mx.random.randint(0, vocab_size, (1,
prompt_tokens))`. AX prompt artifacts match this token contract.

For the Qwen 3.6 27B 4-bit p128 blocker, the prompt token artifact has no
usable repetition:

- prompt length: 128
- repeated unigrams: 0
- repeated bigrams: 0
- repeated trigrams: 0
- repeated fourgrams: 0

A capture-output-token run confirmed the generated output does not create a
draftable online n-gram context either:

- output length: 128
- output repeated bigram contexts: 3
- output repeated trigram contexts: 1
- prompt+output repeated fourgram contexts: 0
- online simulated 2-token-context candidates under the current linear policy:
  0/128
- online simulated 1-token-context candidates were too low quality: at best
  41 candidates with 2 accepts (4.9%), or 2 candidates with 1 accept under
  `min_support=2`

This means the Qwen p128 README random-token row cannot reach the 1.18x goal
through ordinary self n-gram speculation. For this row, the only aligned path is
making the direct fallback itself faster or adding a different verified draft
source.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-output-token-diagnostics/qwen3_6-27b-4bit-p128.json`

Stage profile on Qwen 3.6 27B 4-bit p128:

- `ax_ngram_draft_attempts`: 0
- `ax_ngram_request_disabled_steps`: 127
- `ax_mlx_direct_pipeline_steps`: 127
- `ax_mlx_direct_pipeline_wall_us`: 3,841,943
- `ax_mlx_direct_pipeline_async_eval_wall_us`: 3,616,300
- `ax_mlx_direct_pipeline_forward_wall_us`: 224,189

Conclusion: the fallback path is dominated by the MLX/GPU graph submitted by
`async_eval`, not by Rust graph construction or n-gram policy overhead.

Probe artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-direct-stage-profile/qwen3_6-27b-4bit-p128.json`

Stage profile after the decode post-input Metal route still shows the direct
fallback path dominated by submitted MLX/GPU work:

- `ax_mlx_direct_pipeline_steps`: 127
- `ax_mlx_direct_pipeline_wall_us`: 3,745,753
- `ax_mlx_direct_pipeline_async_eval_wall_us`: 3,495,701
- `ax_mlx_direct_pipeline_op_count_per_step`: 1,128
- `ax_mlx_direct_pipeline_linear_attention_ops_per_layer`: 13
- `ax_mlx_direct_pipeline_full_attention_ops_per_layer`: 31

Decode profile with barriers still points to FFN and residual/norm as the large
remaining blocks for Qwen 3.6 27B 4-bit p128/g32:

- `post_attn_ffn_wall_us`: 1,639,605
- `post_attn_residual_norm_wall_us`: 608,737
- `post_attn_residual_gate_wall_us`: 344,098
- `pre_sdpa_wall_us`: 400,803
- `lm_head_wall_us`: 43,923

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-post-input-metal-telemetry-probe/qwen3_6-27b-4bit-p128-stage-profile.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-post-input-metal-telemetry-probe/qwen3_6-27b-4bit-p128-g32-decode-profile.json`

Post-commit decode profile after the RMSNorm+gate Metal route shows the same
shape. This profile inserts barriers and is diagnostic only; it measured
9.77 tok/s because every profiled bucket forces evaluation:

- `post_attn_ffn_wall_us`: 6,585,761
- `post_attn_ffn_gate_up_wall_us`: 3,140,860
- `post_attn_ffn_activation_wall_us`: 1,460,789
- `post_attn_ffn_down_wall_us`: 2,124,368
- `post_attn_residual_norm_wall_us`: 2,436,850
- `post_attn_residual_gate_wall_us`: 1,386,142
- `pre_sdpa_wall_us`: 1,602,256
- `lm_head_wall_us`: 180,024

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-rms-full-gate-postcommit-profile/qwen3_6-27b-4bit-p128-decode-profile.json`

The same diagnostic profile on Qwen 3.6 27B 8-bit p128/g32 shows the same
shape, with larger FFN cost. This profile inserts barriers and is not a
production throughput claim:

- `post_attn_ffn_wall_us`: 2,182,901
- `post_attn_ffn_gate_up_wall_us`: 1,262,923
- `post_attn_ffn_down_wall_us`: 704,567
- `post_attn_ffn_activation_wall_us`: 381,658
- `post_attn_residual_norm_wall_us`: 798,890
- `post_attn_residual_gate_wall_us`: 360,187
- `pre_sdpa_wall_us`: 436,911
- `lm_head_wall_us`: 77,684

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-8bit-p128-g32-decode-profile.json`

Reference comparison:

- `mlx_lm.generate_step` creates the model cache, processes prompt chunks,
  then runs `_step(y)` with `mx.async_eval(next_y, next_logprobs)` before
  materializing the current token.
- AX direct greedy fallback already mirrors the same double-buffer pattern:
  build step N+1 from the lazy pending token, submit `async_eval(next_token)`,
  then materialize step N.
- The remaining gap is therefore not a missing pipeline barrier or missing
  async overlap. It is in the submitted per-token MLX graph itself.

## Rejected candidates

### Re-enable n-gram after initial no-draft

Temporarily allowing `LinearInitialNoDraft` to re-enable n-gram after fallback
output produced only one accepted draft token and inserted single-decode/cooldown
work. It regressed Qwen 3.6 27B 4-bit p128/p512 throughput.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-linear-reenable-probe/qwen3_6-27b-4bit.json`

### Direct cache-clear cadence override

`AX_MLX_DIRECT_CLEAR_CACHE_CADENCE=0` did not improve Qwen 3.6 27B 4-bit p128.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-direct-cadence-probe/qwen3_6-27b-4bit-cadence0.json`

### Dense qmatmul + RMSNorm fusion

An initial env-only probe did not improve Qwen 3.6 27B 4-bit p128. A later
code probe made the Qwen linear dense FFN post-norm eligible for the existing
fused path, but `AX_MLX_DENSE_QMATMUL_RMS_NORM=1` still regressed slightly
relative to the same dirty-code baseline. The code probe was reverted.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-qmatmul-rms-probe/qwen3_6-27b-4bit.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-ffn-postnorm-probe/qwen3_6-27b-4bit-p128.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-ffn-postnorm-probe/qwen3_6-27b-4bit-p128-qmatmul-rms.json`

### Add + RMSNorm pair fusion

A Qwen linear-layer opt-in use of the existing `add_rms_norm_pair` route did
not improve p128 fallback throughput. The code probe was reverted.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-add-rms-probe/qwen3_6-27b-4bit-p128-add-rms.json`

### Direct pipeline KV refs async-eval

An opt-in probe submitted direct-pipeline cache refs together with
`next_token_arr` during `async_eval`, to test whether unevaluated KV/linear
state chains were dragging Qwen p128 decode. It did not improve throughput:

- default post-input Metal probe with output-token capture: 34.1425 tok/s
- `AX_MLX_DIRECT_PIPELINE_EVAL_KV_REFS=1`: 34.1287 tok/s
- async eval wall increased slightly: about 3.498s -> 3.508s

The code probe was removed.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-direct-pipeline-kv-refs-probe/qwen3_6-27b-4bit-p128.json`

### Qwen post-attention FFN C++ block

An opt-in Qwen dense post-attention FFN boundary
(`AX_MLX_DIRECT_CPP_QWEN_POST_ATTN_FFN=1`) wrapped residual add, pre-FFN
RMSNorm, packed gate/up qmatmul, SwiGLU, down qmatmul, optional post-FFN
RMSNorm, and final add behind one C++ MLX call. It compiled and passed a small
portable-equivalence test, but real Qwen 3.6 27B 4-bit p128 throughput regressed
to 33.9156 tok/s, below the validated post-input Metal baseline (~34.13-34.19
tok/s). The code probe was removed.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-post-attn-ffn-probe/qwen3_6-27b-4bit-p128.json`

### GatedDelta decode Metal kill-switch A/B

A kill-switch was added around the single-token GatedDelta decode Metal
specialization to verify whether it was helping after the decode post-input
Metal route. With `AX_MLX_QWEN_GATED_DELTA_DECODE_METAL=0`, Qwen 3.6 27B
4-bit p128 measured 34.0536 tok/s, lower than the best post-input Metal runs.
The route stays default-on for now, but it is not the next throughput target.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen-gateddelta-decode-ab/qwen3_6-27b-4bit-p128-gateddelta-off.json`

### Packed SwiGLU Metal kill-switch A/B

Because upstream `mlx_lm` Qwen uses an `mx.compile(shapeless=True)` SwiGLU
helper, a Qwen 27B probe disabled AX's packed SwiGLU Metal activation to fall
back to the compiled SwiGLU path after the packed gate/up projection. The result
was mixed and not safe to promote:

- Qwen 3.6 27B 8-bit p128 improved from 18.155 tok/s to 18.699 tok/s in a
  one-repetition probe, roughly back to `mlx_lm` parity but still far below the
  22.025 tok/s 1.18x target.
- Qwen 3.6 27B 8-bit p512 improved only slightly, from 18.481 tok/s to
  18.595 tok/s, still below the 21.960 tok/s 1.18x target.
- Qwen 3.6 27B 8-bit p2048 was effectively unchanged, from 18.452 tok/s to
  18.459 tok/s, still below the 21.729 tok/s 1.18x target.
- Qwen 3.6 27B 6-bit p128 improved slightly, from 24.410 tok/s to
  24.741 tok/s, still below the 28.312 tok/s 1.18x target.
- Qwen 3.6 27B 6-bit p512 regressed from 24.042 tok/s to 23.748 tok/s.
- Qwen 3.6 27B 6-bit p2048 regressed from 23.314 tok/s to 22.873 tok/s.
- Qwen 3.6 27B 4-bit p128 regressed from 34.111 tok/s to 33.970 tok/s in the
  matching one-repetition probe.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-8bit-p128-swiglu-metal-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-8bit-p512-p2048-swiglu-metal-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-6bit-swiglu-metal-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-4bit-p128-swiglu-metal-off.json`

### Packed SwiGLU exact-shape compile probe

An opt-in packed SwiGLU `mlx_compile` probe was tested but removed. The first
attempt used `shapeless=True`, matching upstream `mlx_lm`'s helper style, but
MLX could not infer output shapes for `slice_last_dim` inside the compiled
closure. Changing the diagnostic to exact-shape compilation made the unit test
pass, but benchmark results were still far below the 1.18x target:

- Qwen 3.6 27B 4-bit p128: 34.131 tok/s, effectively unchanged from the
  34.111 tok/s current sweep row and still below the 40.090 tok/s target.
- Qwen 3.6 27B 8-bit p128: 18.593 tok/s, better than the 18.155 tok/s default
  row but still below the 18.665 tok/s `mlx_lm` row and far below the
  22.025 tok/s target.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-packed-swiglu-compile-probe/qwen3_6-27b-4bit-p128.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-packed-swiglu-compile-probe/qwen3_6-27b-8bit-p128.json`

### Dense FFN gate/up packing kill-switch A/B

Disabling AX's dense FFN gate/up packing was also tested because upstream
`mlx_lm` keeps Qwen gate and up projections split. This did not produce a
material decode win and increases memory pressure in the 8-bit run:

- Qwen 3.6 27B 4-bit p128: 34.167 tok/s, only +0.16% versus the 34.111 tok/s
  current sweep row and still below the 40.090 tok/s target.
- Qwen 3.6 27B 8-bit p128: 18.557 tok/s, still below the 18.665 tok/s
  `mlx_lm` row and far below the 22.025 tok/s target.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-ffn-gate-up-pack-probe/qwen3_6-27b-4bit-p128-pack-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-ffn-gate-up-pack-probe/qwen3_6-27b-8bit-p128-pack-off.json`

### Linear-attention projection packing kill-switch A/B

Upstream `mlx_lm` Qwen3.5 keeps the linear-attention projections split as
`qkv`, `z`, `b`, and `a`. AX's default load-time pack materializes compatible
split weights into `qkvz` and `ba`. A focused kill-switch probe tested
`AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS=0`.

The 4-bit rows improved slightly, but the result is not enough for the goal and
is not clean enough to promote across the Qwen family. The split route also
raises peak memory because the packed replacement is no longer used.

| Model | Prompt | pack-off tok/s | current reference tok/s | Direction |
| --- | ---: | ---: | ---: | --- |
| Qwen 3.6 27B 4-bit | 128 | 34.428 | 34.211 | +0.6% |
| Qwen 3.6 27B 4-bit | 512 | 34.469 | 34.129 | +1.0% |
| Qwen 3.6 27B 4-bit | 2048 | 34.069 | 33.821 | +0.7% |
| Qwen 3.6 27B 5-bit | 128 | 28.416 | 28.226 | +0.7% |
| Qwen 3.6 27B 5-bit | 512 | 28.523 | 28.335 | +0.7% |
| Qwen 3.6 27B 5-bit | 2048 | 28.011 | 28.126 | -0.4% |
| Qwen 3.6 27B 6-bit | 128 | 25.006 | 24.891 | +0.5% |
| Qwen 3.6 27B 6-bit | 512 | 25.148 | 25.306 | -0.6% |
| Qwen 3.6 27B 6-bit | 2048 | 23.511 | 23.990 | -2.0% |
| Qwen 3.6 27B 8-bit | 128 | 18.902 | 18.774 | +0.7% |

An isolated `AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS=0` probe on 4-bit
p128 measured 34.174 tok/s while keeping peak memory near the packed route, so
the useful signal is not just disabling the C++ inputs wrapper. No default
change is promoted from this probe.

Follow-up runtime audit: the benchmark JSON `model_config` field reports the
original manifest layout, not the post-load runtime layout. `pack-audit` on the
same Qwen 3.6 27B 4-bit checkpoint confirmed the actual loaded weights use
48/48 packed QKVZ/BA linear-attention layers and 64/64 packed dense FFN gate/up
layers under the default environment. Therefore the pack-off A/B above remains
a valid kill-switch probe, and the split `model_config.linear_attention_projection_layout`
field should not be treated as runtime proof.

A follow-up dirty-code probe tried a narrower conditional change: keep
linear-attention projection packing enabled generally, but skip it only for
quantized 4-bit split Qwen linear-attention projections. This reproduced the
small 4-bit pack-off signal but did not meet the bar for a default runtime
change:

- Qwen 3.6 27B 4-bit p128/g128: 34.442 tok/s
- clean n-gram recheck context: 34.210 tok/s
- current `mlx_lm` reference: 33.980 tok/s
- delta versus clean n-gram context: about +0.7%
- delta versus `mlx_lm`: about +1.4%
- target: 40.096 tok/s

The result is below the 2% acceptance threshold for a useful probe and far
below the 1.18x goal. The code probe was removed.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-linear-pack-off-probe/qwen3_6-27b-4bit-p128-pack-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-linear-pack-off-probe/qwen3_6-27b-4bit-p512-p2048-pack-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-linear-pack-off-probe/qwen3_6-27b-5bit-p128-pack-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-linear-pack-off-probe/qwen3_6-27b-5bit-p512-p2048-pack-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-linear-pack-off-probe/qwen3_6-27b-6bit-p128-pack-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-linear-pack-off-probe/qwen3_6-27b-6bit-p512-p2048-pack-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-linear-pack-off-probe/qwen3_6-27b-8bit-p128-pack-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-direct-inputs-off-probe/qwen3_6-27b-4bit-p128-inputs-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-linear-pack-4bit-skip-probe/qwen3_6-27b-4bit-p128-g128-pack4-skip.json`

### Full-attention QKV projection packing kill-switch A/B

Upstream `mlx_lm` Qwen3Next full-attention layers keep `q_proj`, `k_proj`, and
`v_proj` split. AX's compatible full-attention layers can pack Q/K/V at load
time, so a focused kill-switch probe tested `AX_MLX_PACK_QKV_PROJECTIONS=0` on
the Qwen 3.6 27B 4-bit p128/g128 blocker row.

The result was neutral: 34.247 tok/s versus the clean n-gram recheck at
34.210 tok/s, about +0.1% and below the 2% acceptance threshold for a useful
probe result. It remains far below the 40.096 tok/s 1.18x target, so no default
change is promoted.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-full-attn-qkv-pack-off-probe/qwen3_6-27b-4bit-p128-g128-qkv-pack-off.json`

### QK norm/ROPE direct C++ route

The opt-in `AX_MLX_DIRECT_CPP_QK_NORM_ROPE=1` route was rechecked after the
Qwen linear-attention direct-fallback work because upstream `mlx_lm` keeps this
section inside ordinary Python/MLX module calls. On Qwen 3.6 27B 4-bit p128 it
measured 34.133 tok/s, effectively noise versus the 34.111 tok/s current-sweep
baseline and far below the 40.090 tok/s target. The route remains opt-in and is
not a useful promotion candidate for the current blocker.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-qk-rope-current-probe/qwen3_6-27b-4bit-p128.json`

### QK norm flat-path kill-switch probe

Another full-attention QK-normalization shape probe tested
`AX_MLX_QK_NORM_FLAT=1`, which routes Q/K RMSNorm through the flat BSHD reshape
path and blocks the direct QK+RoPE C++ route. This covers the remaining
full-attention QK-normalization layout difference that was not covered by the
direct C++ route check.

The result was neutral on the Qwen 3.6 27B 4-bit p128/g128 blocker shape:
34.213 tok/s versus the clean n-gram recheck at 34.210 tok/s. The route remains
diagnostic-only and is not promoted.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-qk-norm-flat-probe/qwen3_6-27b-4bit-p128-g128-qk-flat.json`

### Dense FFN `mlx_compile` block

An opt-in probe wrapped the dense FFN decode block
(`gate_up qmatmul -> packed SwiGLU -> down qmatmul`) in a thread/shape/weight
keyed `mlx_compile` closure. This tested a larger MLX graph boundary than the
earlier small C++ wrapper while keeping the normal quantized matmul and packed
SwiGLU ops inside the compiled graph. During the dirty-code probe, the route
passed formatting,
`cargo test -p ax-engine-mlx dense_ffn_compile_block --quiet`,
`cargo test -p ax-engine-mlx fastpath::tests::dense_ffn_compile_block_uses_opt_in_contract --quiet`,
and `cargo build -p ax-engine-server --release`, but real Qwen 3.6 27B 4-bit
p128 throughput measured 34.164 tok/s. That is below the post-commit baseline
of 34.192 tok/s and still far below the 40.096 tok/s target. The probe code was
removed.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-dense-ffn-compile-block-probe/qwen3_6-27b-4bit-p128.json`

### Attention output gate Metal probe

Qwen3Next full-attention layers apply `attn_out * sigmoid(gate)` before
`o_proj`, matching upstream `mlx_lm`. A dirty-code opt-in probe fused that
elementwise pair into a single Metal node while leaving the quantized output
projection unchanged. The unit test passed and the p128 A/B was too small to
promote:

- `AX_MLX_ATTN_OUTPUT_GATE_METAL=1`: 34.226 tok/s
- same dirty build, flag off: 34.158 tok/s
- delta: about +0.20%, within run noise and far below the 40.096 tok/s target

The probe code was removed.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-attn-output-gate-probe/qwen3_6-27b-4bit-p128.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-attn-output-gate-probe/qwen3_6-27b-4bit-p128-default.json`

### Linear n-gram min-support probe

The p128 output-token diagnostic shows local output repetition, but the
linear-attention n-gram policy requires two observations for a generated-output
context before drafting. A dirty-code opt-in probe allowed
`AX_MLX_LINEAR_NGRAM_MIN_SUPPORT=1` to test whether single-observation
generated repeats can recover the random-token prompt case. It did produce
draft attempts, but none were accepted:

- Qwen 3.6 27B 4-bit p128: 33.187 tok/s, below the 34.158 same-build direct
  fallback row
- `ax_ngram_draft_attempts`: 2
- `ax_ngram_draft_tokens`: 6
- `ax_ngram_accepted_tokens`: 0
- `ax_mlx_ngram_decode_steps`: 2
- `ax_mlx_ngram_decode_wall_us`: 167,200

The probe confirms that lowering the linear min-support gate adds expensive
complete misses on the random-token blocker rather than useful bonus tokens.
The probe code was removed and the release binary was rebuilt from the clean
source tree.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-attn-output-gate-probe/qwen3_6-27b-4bit-p128-linear-min1.json`

### Prefill output n-gram feed fix

The prefill step samples the first generated token before the decode loop
starts, but the request-local n-gram table previously only received prompt
tokens at generation initialization. Subsequent direct fallback decode steps
fed their outputs, so the table was missing exactly the first generated token
from its context window. The fix feeds `prefill_output_token` into the n-gram
table after seeding the prompt tail. A focused unit test now verifies that the
next prediction context advances from prompt tail to generated-token tail.

Validation:

- `cargo fmt --check`
- `cargo test -p ax-engine-mlx generation_ngram_seed --quiet`
- `cargo test -p ax-engine-mlx ngram --quiet`
- `cargo build -p ax-engine-server --release`

Qwen 3.6 27B 4-bit skip-`mlx_lm` probe:

| prompt tokens | pre-fix postcommit | post-fix probe | change | n-gram accepted |
| --- | ---: | ---: | ---: | ---: |
| 128 | 34.192 tok/s | 34.211 tok/s | +0.06% | 0 |
| 512 | 34.157 tok/s | 34.129 tok/s | -0.08% | 1 |
| 2048 | 33.831 tok/s | 33.821 tok/s | -0.03% | 1 |

This is kept as a state-correctness fix, not as a throughput fix. It does not
change the remaining blocker: the random-token Qwen rows still lack enough
accepted draft tokens to approach the 1.18x target.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-prefill-output-feed-probe/qwen3_6-27b-4bit.json`

### Offline Qwen policy replay

Captured-output replay now rules out a simple n-gram policy relaxation for the
Qwen 3.6 27B 4-bit random-token blocker. The replay seeded the table with the
same prompt-tail contract (`NGRAM_PROMPT_FEED_MAX=64`) plus the first prefill
output token, then compared candidate drafts against the actually generated
token IDs.

| prompt tokens | runtime accepted | current replay | min_support=1 replay | conclusion |
| --- | ---: | ---: | ---: | --- |
| 128 | 0 | 0 attempts / 0 accepted | 2 attempts / 0 accepted | harmful verifier work |
| 512 | 1 | 1 attempt / 1 accepted | 2 attempts / 2 accepted | too small to matter |
| 2048 | 1 | 1 attempt / 1 accepted | 2 attempts / 3 accepted | too small to matter |

`LlamaMapLatest` and zero confidence-threshold variants produced the same
candidate count on these captures. Even the permissive `min_support=1` replay
would add at most three accepted tokens across a 128-token decode, while the
blocking rows need about an 18% decode-throughput lift. No policy-only change
is promoted from this probe.

Capture artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-prefill-output-feed-probe/qwen3_6-27b-4bit-p128-capture.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-policy-replay-capture/qwen3_6-27b-4bit-p512-p2048-capture.json`

### Production direct-pipeline stage profile

`AX_MLX_DIRECT_PIPELINE_STAGE_PROFILE=1` adds op-count and graph-build timing
without decode-stage eval barriers. A focused Qwen 3.6 27B 4-bit p128/g64 run
still used the linear no-draft direct fallback and produced no n-gram drafts:

- decode: 34.500 tok/s
- effective route: `linear_no_draft_direct_pipeline_fallback`
- n-gram draft attempts: 0
- request-disabled fallback steps: 63
- direct pipeline wall: 1,852,168 us
- direct pipeline forward graph-build: 120,554 us
- direct pipeline async eval: 1,730,894 us

The per-layer op count also shows that the remaining Qwen direct path is not
primarily a Rust dispatch-count problem. Across 63 direct-pipeline steps:

| Layer group | Layer calls | Ops | Ops/layer |
| --- | ---: | ---: | ---: |
| Linear attention | 3,024 | 39,312 | 13.0 |
| Full attention | 1,008 | 31,248 | 31.0 |

This reinforces the prior conclusion: after the Qwen linear-attention post-input
and GatedDelta decode Metal routes, another small wrapper around a few MLX nodes
is unlikely to close an 18% throughput gap. The next useful probe should target
a larger Qwen dense post-attention FFN/full-attention boundary or a different
verified draft source.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-stage-profile-prod/qwen3_6-27b-4bit-p128-g64-stage-profile.json`

### Linear-attention output `mlx_compile` block

An opt-in probe wrapped the Qwen linear-attention output block
(`rms_norm_gated -> reshape -> out_proj`) in a per-thread/layer/weight keyed
`mlx_compile` closure. This tested a larger boundary than the retained
RMSNorm+gate Metal route while keeping the verified default path unchanged.
The diagnostic first used MLX array byte sizes as cache identity, which would
collide across same-shaped layers, so it was corrected to use MLX handle
identity before validation.

Validation while the probe existed:

- `cargo fmt --check`
- `cargo test -p ax-engine-mlx qwen_linear_attention_output_compile --quiet`
- `cargo test -p ax-engine-mlx linear_attention --quiet`
- `cargo build -p ax-engine-server --release`

Real Qwen 3.6 27B 4-bit p128/g64 throughput measured 34.451 tok/s with the
probe enabled. That is slightly below the comparable 34.500 tok/s production
stage-profile baseline and still far below the 40.096 tok/s target. The run
remained in `linear_no_draft_direct_pipeline_fallback` with zero n-gram draft
attempts and 63 request-disabled fallback steps. The probe code was removed.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-linear-output-compile-probe/qwen3_6-27b-4bit-p128-g64.json`

### Full-attention add+RMSNorm pair kill-switch A/B

The default full-attention path uses `AX_MLX_DENSE_ADD_RMS_NORM_PAIR=1` to fuse
the attention residual add and pre-FFN RMSNorm. A focused env-only A/B checked
whether this default route was hurting the Qwen 3.6 27B 4-bit p128/g64 direct
fallback rows.

| Mode | Decode tok/s | Route | Draft attempts |
| --- | ---: | --- | ---: |
| default | 34.507 | `linear_no_draft_direct_pipeline_fallback` | 0 |
| `AX_MLX_DENSE_ADD_RMS_NORM_PAIR=0` | 34.606 | `linear_no_draft_direct_pipeline_fallback` | 0 |

The +0.29% difference is within run noise and far below the 40.096 tok/s
target, so no default change is promoted.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-add-rms-default-ab/qwen3_6-27b-4bit-p128-g64-default.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-add-rms-default-ab/qwen3_6-27b-4bit-p128-g64-add-rms-off.json`

### Qwen attention-norm carry probe

A direct-decode probe tried carrying a precomputed next-layer attention RMSNorm
through the Qwen 3.6 layer loop by fusing each layer's final residual add with
the next layer's input RMSNorm. This targeted the no-draft direct fallback, not
the n-gram policy.

The probe was not promoted. It was only noise-level positive on the short g64
probe and did not materially move the README-shaped blocker:

| Shape | Decode tok/s | Baseline context | Result |
| --- | ---: | --- | --- |
| p128/g64 | 34.595 | prior direct-fallback probe: about 34.500 | about +0.3% |
| p128/g128 | 34.228 | current sweep row: 34.192 | about +0.1% |

The stage-profile run still reported 1,128 direct-pipeline ops/step, with 13
ops per linear-attention layer and 31 ops per full-attention layer. This means
the carry path did not create the intended graph-size reduction, and the runtime
effect is far below the 40.096 tok/s target. The code probe was removed.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-attn-norm-carry-probe/qwen3_6-27b-4bit-p128-g64.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-attn-norm-carry-probe/qwen3_6-27b-4bit-p128-g64-stage-profile.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-attn-norm-carry-probe/qwen3_6-27b-4bit-p128-g128.json`

### Direct greedy chunk probe

An opt-in exact greedy chunk probe tried to materialize two deterministic direct
fallback tokens per model step and serve the second token through the existing
bonus-token path. This was intended to amortize per-token command/eval overhead
on the Qwen no-draft fallback row.

The probe was not promoted. On the Qwen 3.6 27B 4-bit p128/g128 blocker shape,
`AX_MLX_DIRECT_GREEDY_CHUNK=2` measured 31.546 tok/s, below both the current
AX row (~34.2 tok/s) and the mlx_lm target boundary. Telemetry confirmed the
route produced 63 bonus tokens and halved direct pipeline invocations, but the
multi-token dependent graph lost the one-step `mlx_lm`-style overlap and spent
about 3.51s in direct-pipeline async eval plus about 0.33s in pending eval.
The runtime experiment code was removed.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-direct-greedy-chunk-probe/qwen3_6-27b-4bit-p128-g128-chunk2.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-direct-greedy-chunk-probe/qwen3_6-27b-4bit-p128-g128-chunk2-v2.json`

### Qwen direct fallback clean rechecks

Two clean-commit p128/g128 rechecks separated n-gram no-draft overhead from the
direct pipeline itself:

| Mode | Decode tok/s | Effective route | Notes |
| --- | ---: | --- | --- |
| `--ax-direct` | 34.457 | `direct_pipeline_baseline` | direct same-policy row |
| `--ax-ngram-accel` | 34.210 | `linear_no_draft_direct_pipeline_fallback` | initial non-repeating prompt disables n-gram; 127 request-disabled direct steps |

The gap is below 1% and within the range where run noise can dominate, so the
Qwen blocker is not caused by a large residual n-gram fallback tax. The useful
target remains model execution or a real draft source.

A related FFN hot-path kill-switch checked whether the dense down-projection +
RMSNorm fused route was hurting Qwen. `AX_MLX_DENSE_QMATMUL_RMS_NORM=0`
measured 34.199 tok/s on the same p128/g128 shape, effectively identical to the
clean n-gram recheck and still far below the 40.096 tok/s target. No default
change is promoted.

Another dirty-code probe extended the existing `add_rms_norm_pair` opt-in to
Qwen linear-attention layers, fusing the post-attention residual add and
pre-FFN RMSNorm for the 48 linear layers. This directly targeted the
`post_attn_residual_norm` profile bucket and mirrors the existing standard
full-attention opt-in surface. It did not improve the blocker:

- Qwen 3.6 27B 4-bit p128/g128 with `AX_MLX_DENSE_ADD_RMS_NORM_PAIR=1`:
  34.250 tok/s
- clean n-gram recheck context: 34.210 tok/s
- current `mlx_lm` reference: 33.980 tok/s
- target: 40.096 tok/s

The result is below the 2% acceptance threshold and far below the 1.18x goal.
The code probe was removed.

The next inspected fusion boundary was the decode-only linear-attention
recurrent update followed by `RMSNormGated(y, z)`. This is not directly
fusible with the current GatedDelta decode kernel because the kernel's
threadgroup covers only a small tile of the value dimension, while RMSNorm
requires a full 128-lane per-head reduction. A cheaper occupancy probe changed
the GatedDelta decode `Dv` tile from 4 to 8 and 16. Neither result was useful:

| GatedDelta decode tile | Decode tok/s | Result |
| --- | ---: | --- |
| 4 current clean recheck | 34.210 | baseline context |
| 8 | 34.224 | noise-level |
| 16 | 34.187 | slight regression |

The tile probe code was removed.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-direct-mode-probe/qwen3_6-27b-4bit-p128-g128-direct.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-ngram-clean-recheck/qwen3_6-27b-4bit-p128-g128-ngram.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-dense-qmatmul-rms-ab/qwen3_6-27b-4bit-p128-g128-qmatmul-rms-off.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-linear-add-rms-pair-probe/qwen3_6-27b-4bit-p128-g128-linear-add-rms-pair.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-gateddelta-tile-probe/qwen3_6-27b-4bit-p128-g128-tile8.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-gateddelta-tile-probe/qwen3_6-27b-4bit-p128-g128-tile16.json`

### Existing verified-draft and KV-compression fallback checks

The Qwen 3.6 27B 4-bit local checkpoint used for the blocker row does not have
an MTP sidecar:

- no `mtp.safetensors`
- no `mtplx_runtime.json`

The runner already checks MTP before n-gram and before direct fallback when
`weights.mtp.is_some()`, so there is no existing MTP draft source to activate
for this checkpoint. The missing sidecar explains why the p128/g64 row keeps
all `ax_mtp_*` counters at zero.

A local sidecar probe used `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`, whose
main model shards resolve to the same local Hugging Face blob hashes as
`mlx-community/Qwen3.6-27B-4bit`, but adds `mtp.safetensors` and
`mtplx_runtime.json`. This activates AX MTP, but it is slower than the direct
fallback and much slower than the 40.096 tok/s target for the blocker row.

| Mode | Decode tok/s | MTP draft tokens | MTP accepted tokens | Decode route |
| --- | ---: | ---: | ---: | --- |
| default depth | 21.644 | 102 | 28 | `mixed` |
| `--ax-mtp-max-depth 1` | 27.971 | 42 | 20 | `mixed` |

The depth-1 cap reduces MTP overhead, but still loses to the current
`mlx-community/Qwen3.6-27B-4bit` direct fallback at about 34.5 tok/s. The
default-depth run spent 1.646s in MTP verify eval, 1.010s in rollback, and
0.241s in draft work; the depth-1 run still spent 1.415s in verify eval,
0.704s in rollback, and 0.097s in draft work. Existing AX MTP therefore does
not provide a usable verified-draft fallback for the random-token Qwen blocker.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-mtplx-speed-source-probe/qwen3_6-27b-mtplx-speed-p128-g64.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-mtplx-speed-source-probe/qwen3_6-27b-mtplx-speed-p128-g64-depth1.json`

TurboQuant fused decode was also checked as an existing direct-fallback speed
surface. A focused Qwen 3.6 27B 4-bit p128/g64 run used
`--experimental-mlx-kv-compression turboquant-fused-experimental` with
`--experimental-mlx-kv-compression-min-context-tokens 1`.

- decode: 34.509 tok/s
- effective route: `linear_no_draft_direct_pipeline_fallback`
- n-gram draft attempts: 0
- KV compression decode path: `full_precision_shadow`
- eligible KV-compression layers: 0
- fused decode attempts: 0

This matches the default direct-fallback throughput and does not provide a
usable acceleration path for the current Qwen blocker.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-turboquant-fused-probe/qwen3_6-27b-4bit-p128-g64-turboquant-fused.json`

### Exact greedy direct-pipeline unroll probe

A dirty-code probe tested whether the no-draft direct fallback could emit
multiple exact greedy tokens per runner decode call by chaining lazy
`argmax -> next forward` arrays before materializing the output tokens. This
was intended to amortize the per-token CPU/GPU barrier without changing greedy
sampling semantics.

The result was negative on the Qwen 3.6 27B 4-bit p128/g128 blocker shape:

| Mode | Decode tok/s | Pending eval wall | Direct async eval wall | Bonus tokens |
| --- | ---: | ---: | ---: | ---: |
| current direct fallback context | ~34.5 | n/a | n/a | 0 |
| exact greedy unroll=2 | 31.532 | 331,814 us | 3,509,951 us | 63 |
| exact greedy unroll=4 | 31.674 | 167,027 us | 3,653,452 us | 95 |

Both unroll depths were slower than the existing one-token double-buffer direct
pipeline. The longer lazy chain reduced runner calls, but it made the
materialization/async-eval boundary heavier enough to lose throughput. The
probe code was removed and no opt-in surface is promoted.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-direct-greedy-unroll-probe/qwen3_6-27b-4bit-p128-g128-unroll2.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-direct-greedy-unroll-probe/qwen3_6-27b-4bit-p128-g128-unroll4.json`

### Linear-initial no-draft fast fallback probe

A dirty-code probe tested a narrower CPU book-keeping reduction for the Qwen
random-token no-draft path. When a linear-attention request was disabled at
generation init with `LinearInitialNoDraft`, the probe skipped output-token
feeds into the n-gram table and skipped the per-step re-enable check. The
intended win was to remove work that cannot help the random-token blocker,
because the prompt/output replay already showed no useful n-gram candidates.

The result was not useful:

- Qwen 3.6 27B 4-bit p128/g128: 34.156 tok/s
- effective route: `linear_no_draft_direct_pipeline_fallback`
- draft attempts: 0
- request-disabled fallback steps: 127
- direct-pipeline pending eval wall: 859 us
- direct-pipeline async eval wall: 3,493,985 us

The probe was slightly below the existing direct-fallback context at about
34.5 tok/s and far below the 40.096 tok/s target. It also weakens the existing
ability to re-enable n-gram acceleration when generated output becomes
repetitive, so the probe code was removed and no default change is promoted.

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-linear-initial-fast-fallback-probe/qwen3_6-27b-4bit-p128-g128-fast-fallback.json`

### Qwen dense SwiGLU direct FFN probe

A dirty-code probe added an opt-in direct C++ MLX shim for the Qwen packed
dense SwiGLU FFN block:

```text
gate_up = quantized_matmul(x, gate_up_weight, ...)
gate, up = split(gate_up, 2, axis=-1)
hidden = silu(gate) * up
out = quantized_matmul(hidden, down_weight, ...)
```

The wrapper matched the portable composition in the focused `mlx-sys` unit
test, and the server built successfully. The real-model result was negative on
the current Qwen 3.6 27B 4-bit p128/g128 blocker:

- Probe: `AX_MLX_DIRECT_CPP_QWEN_DENSE_FFN=1`
- Decode throughput: 33.875 tok/s
- Current blocker context: about 34.19 tok/s
- Current `mlx_lm` reference: 33.98 tok/s
- Target: 40.096 tok/s
- Effective route: `linear_no_draft_direct_pipeline_fallback`
- Draft attempts: 0

This is below the 2% acceptance threshold and does not move the 1.18x goal.
The probe code was removed.

Validation:

- `cargo fmt --check`
- `cargo test -p mlx-sys silu_quantized_ffn --quiet`
- `cargo test -p ax-engine-mlx direct_cpp_qwen_dense_ffn --quiet`
- `cargo build -p ax-engine-server --release`

Artifact:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-silu-quantized-ffn-probe/qwen3_6-27b-4bit-p128-g128-silu-quantized-ffn.json`

## Next target

Small Rust/FFI node fusion is not enough for the remaining Qwen gap. The next
useful implementation target should be a larger decode-graph boundary,
especially one of:

- an `mlx_compile`-style per-layer or whole-decode graph boundary that treats
  cache state as explicit inputs/outputs
- a Qwen dense FFN/residual-norm boundary that materially reduces the
  `post_attn_ffn_gate_up`, `post_attn_ffn_down`, and `post_attn_residual_norm`
  buckets without regressing production p128 throughput; the earlier small
  whole-FFN C++ wrapper did not clear this bar
- deeper Qwen linear-attention layer fusion that spans projection -> conv ->
  recurrent update -> output projection, not just the post-input section
- a different verified draft source for random-token no-draft prompts; ordinary
  prompt/output n-grams do not provide enough candidates on the current blocker
