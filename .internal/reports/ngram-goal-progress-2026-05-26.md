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
- Published README reference: `mlx_lm` 34.0 tok/s, AX n-gram 33.3 tok/s
- Current probe after GatedDelta + decode post-input Metal work: 34.15 tok/s
- Ratio: about 1.00x versus the README `mlx_lm` row, still far below the 1.18x
  target

Current post-commit sweep on commit `71f3166440bf4ca6cb11ffea61d544fcf2f8b999`
confirms the blocker is the Qwen 3.6 27B family, not just one stale README row.
The sweep used `--skip-mlx-lm --ax-ngram-accel --no-build-ax-engine`,
`generation_tokens=128`, `repetitions=1`, and `cooldown=0`; percentages below
compare against the current README `mlx_lm` reference rows from
`benchmarks/results/mlx-inference/2026-05-26-direct-mode-clean-refresh/`.

| Model | Prompt | AX n-gram tok/s | `mlx_lm` tok/s | Delta | 1.18x target | Drafts accepted | Effective route |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen 3.6 27B 4-bit | 128 | 34.111 | 33.975 | +0.4% | 40.090 | 0 | `linear_no_draft_direct_pipeline_fallback` |
| Qwen 3.6 27B 4-bit | 512 | 34.120 | 33.905 | +0.6% | 40.008 | 1 | `ngram_verified_bonus_tokens` |
| Qwen 3.6 27B 4-bit | 2048 | 33.722 | 33.441 | +0.8% | 39.460 | 1 | `ngram_verified_bonus_tokens` |
| Qwen 3.6 27B 5-bit | 128 | 28.107 | 21.600 | +30.1% | 25.488 | 2 | `ngram_verified_bonus_tokens` |
| Qwen 3.6 27B 5-bit | 512 | 28.154 | 28.133 | +0.1% | 33.197 | 0 | `linear_no_draft_direct_pipeline_fallback` |
| Qwen 3.6 27B 5-bit | 2048 | 27.773 | 27.821 | -0.2% | 32.829 | 4 | `ngram_verified_bonus_tokens` |
| Qwen 3.6 27B 6-bit | 128 | 24.410 | 23.993 | +1.7% | 28.312 | 0 | `ngram_attempted_no_accept_fallback` |
| Qwen 3.6 27B 6-bit | 512 | 24.042 | 24.776 | -3.0% | 29.236 | 0 | `linear_no_draft_direct_pipeline_fallback` |
| Qwen 3.6 27B 6-bit | 2048 | 23.314 | 24.623 | -5.3% | 29.055 | 0 | `linear_no_draft_direct_pipeline_fallback` |
| Qwen 3.6 27B 8-bit | 128 | 18.155 | 18.665 | -2.7% | 22.025 | 0 | `ngram_attempted_no_accept_fallback` |
| Qwen 3.6 27B 8-bit | 512 | 18.481 | 18.610 | -0.7% | 21.960 | 0 | `linear_no_draft_direct_pipeline_fallback` |
| Qwen 3.6 27B 8-bit | 2048 | 18.452 | 18.414 | +0.2% | 21.729 | 0 | `linear_no_draft_direct_pipeline_fallback` |

Only Qwen 3.6 27B 5-bit at prompt=128 currently clears the +18% goal. All other
Qwen 3.6 27B default n-gram rows remain at direct-fallback parity or worse.

Artifacts:

- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-4bit.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-5bit.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-6bit.json`
- `benchmarks/results/mlx-inference/2026-05-26-ngram-qwen27-current-sweep/qwen3_6-27b-8bit.json`

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
