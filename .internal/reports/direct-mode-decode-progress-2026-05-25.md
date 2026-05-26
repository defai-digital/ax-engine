# Direct Mode Decode Progress — 2026-05-25

## Goal

Improve AX Engine MLX direct mode until direct decode token/s is better than
`mlx_lm.benchmark` for every published model row and prompt shape.

This report records the current progress from the 2026-05-25 continuation.
The current acceptance rule is: rows within 2% below `mlx_lm` are acceptable.
It is not a completion report yet: the fresh Qwen3.6 27B matrix below now
passes that rule, but the full current-worktree direct-mode matrix still needs
one clean verification pass before making a public claim.

## Current Source Changes

Verified source work currently in the worktree:

- Split Gemma GEGLU Metal path for already-separated gate/up tensors.
- Gemma4 per-layer input gate projection path using
  `gelu_approx(gate) * per_layer_input -> quantized_matmul`.
- Default-on fast-path flag `AX_MLX_GEGLU_MUL_METAL`.
- Existing dense packed GEGLU and per-layer gate tests still pass.

In-progress, not validated yet:

- Began a narrower dense-Gemma residual/norm shim in `mlx-sys`
  (`add_rms_norm_pair`, `quantized_matmul_rms_norm`). This was started after
  profiling Gemma 31B, but routing and validation were not completed before
  this report.

## Validation Run So Far

Passed before the latest benchmark runs:

```bash
cargo test -p ax-engine-mlx geglu_mul_metal_uses_default_on_kill_switch_contract --quiet
cargo test -p ax-engine-mlx split_geglu_metal --quiet
cargo build --release -p ax-engine-server
```

## Gemma 26B A4B Result

Artifact:

- `/private/tmp/ax-direct-gemma-26b-a4b-4bit-split-geglu-current.json`

Result: now passes all prompt shapes.

| Prompt | mlx_lm decode | AX direct decode | Ratio | Status |
|---:|---:|---:|---:|---|
| 128 | 127.9 tok/s | 131.9 tok/s | 1.031x | PASS |
| 512 | 125.0 tok/s | 129.0 tok/s | 1.031x | PASS |
| 2048 | 119.3 tok/s | 123.9 tok/s | 1.038x | PASS |

Interpretation: the split GEGLU Metal path fixes the 26B A4B direct decode gap.
This row was previously below `mlx_lm` on every prompt shape.

## Gemma 31B Result

Artifact:

- `/private/tmp/ax-direct-gemma-31b-4bit-split-geglu-current.json`

Result: still fails all prompt shapes.

| Prompt | mlx_lm decode | AX direct decode | Ratio | Status |
|---:|---:|---:|---:|---|
| 128 | 28.9 tok/s | 28.4 tok/s | 0.982x | FAIL |
| 512 | 28.3 tok/s | 27.8 tok/s | 0.981x | FAIL |
| 2048 | 27.0 tok/s | 26.1 tok/s | 0.967x | FAIL |

Interpretation: Gemma 31B is dense Gemma4, so it does not benefit from the
split MoE expert GEGLU path the same way Gemma 26B A4B does. Telemetry shows
the dense packed gate/up path is already active.

## Gemma 31B A/B Attempts

All A/B attempts below are prompt=128 only and should not be promoted.

| Experiment | Artifact | AX decode | Ratio vs mlx_lm | Result |
|---|---|---:|---:|---|
| `AX_MLX_DIRECT_CPP_GEMMA4_POST_ATTN_FFN=1` | `/private/tmp/ax-direct-gemma-31b-4bit-post-attn-ffn-p128.json` | 27.7 tok/s | 0.959x | Worse |
| `AX_MLX_DIRECT_CPP_QK_NORM_ROPE=1` | `/private/tmp/ax-direct-gemma-31b-4bit-qk-norm-rope-p128.json` | 28.2 tok/s | 0.977x | Worse |
| `AX_MLX_PACK_DENSE_FFN_GATE_UP=0` | `/private/tmp/ax-direct-gemma-31b-4bit-no-packed-ffn-p128.json` | 28.2 tok/s | 0.975x | Worse |
| `AX_MLX_ROTATING_SLIDING_DECODE=0` | `/private/tmp/ax-direct-gemma-31b-4bit-no-rotating-p128.json` | 27.4 tok/s | 0.950x | Worse |
| `AX_MLX_LAYER_SCALAR_FUSED_ADD=0` | `/private/tmp/ax-direct-gemma-31b-4bit-no-layer-scalar-p128.json` | 27.7 tok/s | 0.958x | Worse |

Notes:

- The whole post-attention FFN C++ route hit 100% of eligible layers but still
  regressed, so the larger C++ boundary is not a promotion candidate.
- Rotating sliding decode is beneficial for this row; disabling it makes the
  gap much larger.
- Dense gate/up packing is not the cause of the 31B gap.

## Gemma 31B Decode Profile

Artifact:

- `/private/tmp/ax-direct-gemma-31b-4bit-decode-profile-p128.json`

The profile inserts barriers, so its decode tok/s is not a benchmark result.
Use it only for stage ranking.

Key stage totals:

| Stage | Wall time |
|---|---:|
| pre-SDPA | 5,977,421 us |
| pre-SDPA QKV projection | 2,085,555 us |
| pre-SDPA QK norm | 1,282,320 us |
| pre-SDPA RoPE/KV | 2,580,033 us |
| SDPA | 1,349,409 us |
| post-attention | 12,557,978 us |
| post-attention FFN | 8,199,455 us |
| FFN gate/up | 3,372,429 us |
| FFN activation | 1,364,647 us |
| FFN down | 2,312,367 us |
| output projection | 1,783,998 us |
| residual norm | 1,298,847 us |
| residual gate | 1,257,494 us |
| lm head | 203,914 us |

Interpretation: the remaining 31B gap is likely in dense Gemma post-attention
or attention orchestration, not in the split GEGLU expert path.

## Current State by Model Family

Current evidence says:

- Gemma E2B/E4B rows had already moved above `mlx_lm` with the existing
  per-layer and packed GEGLU work.
- Gemma 26B A4B now passes all prompt shapes with split GEGLU Metal.
- Gemma 31B now passes prompt 128/512/2048 in focused same-session reruns after
  the direct-pipeline prefill-boundary alignment below. Earlier fail rows were
  stale relative to the current source/runtime boundary.
- Qwen 27B 4-bit now passes prompt 128/512/2048 in focused same-session reruns
  after the direct-pipeline prefill-boundary alignment below. Earlier fail rows
  were partly stale/noisy relative to the current `mlx_lm` reference.
- Qwen 27B 5-bit prompt 512 passes. Prompt 2048 was the active unresolved row
  until the later source-backed Qwen linear-attention default route checks; with
  a longer-cooldown same-session run, the current default route now passes
  (`AX median 27.463` vs `mlx_lm median 27.277` tok/s). Keep the earlier
  1-second-cooldown misses as variance evidence, not publication evidence.
- Qwen 27B 6-bit now passes prompt 128/512/2048 in focused same-session reruns.
- Qwen 27B 8-bit prompt 512/2048 pass, and prompt 128 passes when measured as
  a 3-repetition row; its single-run fail was within run-to-run noise.
- Qwen 35B A3B was passing in the older artifact set, but should still be
  re-audited before closing the goal. The latest p2048 same-session rerun also
  passes.

## 2026-05-26 Qwen 27B Candidate Checks

Two Qwen 3.6 27B 4-bit candidates were checked after the 2026-05-25 report.
Neither produced promotable evidence.

### Linear-layer residual add + RMSNorm pair

Candidate: temporarily routed `qwen3_linear::layer_forward` through the existing
`add_rms_norm_pair` shim under `AX_MLX_DENSE_ADD_RMS_NORM_PAIR=1`.

Artifacts:

- `/private/tmp/ax-direct-qwen27b-4bit-add-rms-baseline-p2048.json`
- `/private/tmp/ax-direct-qwen27b-4bit-add-rms-candidate-p2048.json`

Result:

| Run | Prompt | AX direct decode | Ratio vs mlx_lm | Trial decode tok/s | Result |
|---|---:|---:|---:|---|---|
| Baseline | 2048 | 25.6 tok/s | 0.766x | 32.7, 18.6 | Noisy fail |
| Candidate | 2048 | 25.3 tok/s | 0.756x | 32.6, 17.9 | No improvement |

Interpretation: the run was noisy, but the normal-speed trial was effectively
unchanged. The hook was removed instead of being kept as an unproven fast path.

### Direct linear-attention post-input C++ route

Candidate: built-in harness comparison for
`AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS=1` versus
`AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS=1` plus
`AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT=1`.

Artifact:

- `/private/tmp/ax-direct-qwen27b-4bit-linear-post-input-compare-p128.json`

Result:

| Engine | Prompt | AX direct decode | Ratio vs mlx_lm | Trial decode tok/s | Fast-path hit rate | Result |
|---|---:|---:|---:|---|---:|---|
| direct linear-attention inputs | 128 | 32.8 tok/s | 0.966x | 32.9, 32.8, 16.2 | 100% | Still below mlx_lm |
| direct post-input route | 128 | 32.8 tok/s | 0.966x | 18.2, 32.8, 32.9 | 100% | No improvement |

Interpretation: the post-input C++ route engages cleanly, but it does not close
the Qwen 27B direct decode gap. The remaining Qwen gap is likely outside this
FFI boundary.

## 2026-05-26 Direct Pipeline Telemetry Expansion

The direct pipeline already measured stage timings internally, but the
benchmark artifact only collected the coarse forward/async/pending buckets.
This made the Qwen 27B gap hard to target because the remaining wall time was
mostly in `async_eval`.

Changes made:

- `DecodeTelemetry` now emits linear-attention and full-attention layer op
  counters from `DirectPipelineTimings`.
- `scripts/bench_mlx_inference_stack.py` now collects the existing
  `forward_layer_loop` and `forward_head` timing counters.
- The benchmark summary now reports per-layer op density for linear-attention
  and full-attention slices.

Validation:

```bash
cargo fmt --check
cargo test -p ax-engine-mlx decode_telemetry_records_route_counters --quiet
python3 scripts/test_bench_mlx_inference_stack.py
cargo build --release -p ax-engine-server
```

Telemetry smoke:

- `/private/tmp/ax-direct-qwen27b-4bit-telemetry-smoke-20260526.json`

Shape: Qwen3.6 27B 4-bit, prompt=128, generation=64, repetitions=1,
`AX_MLX_DIRECT_PIPELINE_STAGE_PROFILE=1`.

Result:

| Engine | Decode tok/s | Ratio vs mlx_lm |
|---|---:|---:|
| `mlx_lm` | 33.0 | baseline |
| AX direct | 33.5 | 1.014x |

Key AX direct telemetry:

| Counter | Value |
|---|---:|
| direct pipeline steps | 63 |
| direct pipeline wall | 1,878,502 us |
| forward wall | 114,636 us |
| forward layer-loop wall | 113,846 us |
| forward head wall | 4 us |
| async eval wall | 1,758,091 us |
| pending eval wall | 5,473 us |
| total direct pipeline ops | 159,217 |
| linear-attention layer ops/count | 127,968 / 2,976 |
| full-attention layer ops/count | 30,752 / 992 |
| linear-attention ops/layer | 43 |
| full-attention ops/layer | 31 |

Interpretation: this smoke row passed, confirming that the p128 published
failure is at least partly noisy. The new counters also show that Qwen 27B's
remaining work is dominated by GPU execution behind `async_eval`, not CPU-side
logit/head work. Linear-attention layers carry the larger graph op density, so
the next optimization should target the linear-attention recurrence/projection
GPU work or direct-pipeline scheduling, not the already-tested post-input C++
boundary.

## 2026-05-26 mlx-lm / MLX Source Comparison Pass

The investigation was redirected away from blind A/B flags and back to source
comparison against the local upstream references:

- `.internal/reference/mlx-lm/mlx_lm/generate.py`
- `.internal/reference/mlx-lm/mlx_lm/models/cache.py`
- `.internal/reference/mlx-lm/mlx_lm/models/gemma4_text.py`
- `.internal/reference/mlx-lm/mlx_lm/models/qwen3_5.py`
- `.internal/reference/mlx-lm/mlx_lm/models/gated_delta.py`
- `.internal/reference/mlx`

Confirmed alignments:

- Decode scheduling: `mlx_lm.generate_step` builds the next token on a
  dedicated generation stream, submits it with `mx.async_eval`, then evaluates
  the previous token. AX direct pipeline already mirrors this with
  `start_direct_pipeline` / `advance_direct_pipeline`.
- Prefill boundary: `mlx_lm` pre-fills all but the final prompt token as
  cache-state-only work, evaluates cache state, and clears MLX cache between
  chunks. AX `chunked_prefill` already uses the same long-greedy
  cache-only-prefix policy.
- Gemma4 cache layout: `mlx_lm` uses `KVCache` for full-attention layers and
  `RotatingKVCache(max_size=sliding_window, keep=0)` for sliding layers. AX
  already uses retained-window rotating decode for Gemma sliding layers.
- Gemma4 attention semantics: `mlx_lm` uses scale `1.0`, global head dim for
  full-attention layers, K-equals-V when configured, no-scale V RMSNorm, and
  post-attention/post-FFN norms before residual adds. The AX manifest and
  `standard.rs` path already model these fields.
- Qwen3.5/Qwen3.6 linear attention: `mlx_lm` uses `ArraysCache(size=2)`, a
  conv tail cache, float32 recurrent state, and
  `q = inv_scale^2 * rms_norm(q)` / `k = inv_scale * rms_norm(k)`. AX already
  matches this and uses a fused Metal gated-delta kernel.
- Stream setup: `mlx_lm` uses a dedicated thread-local generation stream. AX
  runner also creates and installs a dedicated GPU stream for MLX work.

Applied source-aligned cleanup:

- `attention_mask_array(1, key_len <= window, Some(window))` now returns `None`,
  matching `mlx_lm`'s single-token retained-window decode behavior. This is
  semantically correct but does not affect the current Gemma direct path because
  AX `build_layer_masks` already returns all-`None` masks for `seq == 1`.

Validation:

```bash
cargo fmt --check
cargo test -p ax-engine-mlx attention_mask_array_keeps_full_kv_for_sliding_attention --quiet
cargo build --release -p ax-engine-server
```

Benchmark/telemetry artifacts:

- `/private/tmp/ax-direct-gemma31b-4bit-mask-align-p2048-20260526.json`
- `/private/tmp/ax-direct-gemma31b-4bit-mask-align-stage-p2048-20260526.json`

Result:

| Shape | mlx_lm decode | AX direct decode | Ratio | Interpretation |
|---|---:|---:|---:|---|
| Gemma4 31B 4-bit p2048/g128 | 27.035 tok/s | 25.49 tok/s | 0.943x | still below mlx_lm |
| Gemma4 31B 4-bit p2048/g128 + stage profile | 27.035 tok/s | 25.84 tok/s | 0.956x | still below mlx_lm |

Stage profile after the cleanup still reported `direct_pipeline_op_count_per_step
= 1437` and `direct_pipeline_full_attention_ops_per_layer = 24`, matching the
previous profile. Therefore the remaining Gemma31 gap is not explained by
single-token sliding masks.

## 2026-05-26 Direct Pipeline Prefill-Boundary Alignment

The follow-up source comparison checked the actual public `mlx_lm` benchmark
path instead of assuming a lower-level loop. `.internal/reference/mlx-lm/mlx_lm/benchmark.py`
calls `stream_generate`, and `stream_generate` measures generation TPS after
the first token is produced. Inside `generate_step`, `mlx_lm` pre-fills all but
the final prompt token, runs `_step(prompt_final_token)`, submits that token via
`mx.async_eval`, then immediately builds and submits `next_y = _step(y)` before
yielding the first token.

Applied change:

- AX direct mode now primes `pending_direct` inside `initialize_generation_state`
  for greedy direct decode when a prefill output token is available. This moves
  the first direct-pipeline bootstrap to the same prefill/first-token boundary
  used by `mlx_lm.generate_step`, instead of charging it to the first measured
  decode step.

Validation:

```bash
cargo fmt --check
cargo test -p ax-engine-mlx decode_telemetry_records_route_counters --quiet
cargo build --release -p ax-engine-server
```

Focused same-session benchmark results:

| Model | Prompt | mlx_lm decode | AX direct decode | Ratio | Artifact |
|---|---:|---:|---:|---:|---|
| Qwen3.6 27B 4-bit | 128 | 33.085 tok/s | 33.107 tok/s | 1.001x | `/private/tmp/ax-vs-mlxlm-qwen27b-4bit-current-p128-20260526.json` |
| Qwen3.6 27B 4-bit | 512 | 32.946 tok/s | 33.030 tok/s | 1.003x | `/private/tmp/mlxlm-qwen27b-4bit-current-p512-20260526.json`, `/private/tmp/ax-direct-qwen27b-4bit-prefill-bootstrap-stageprofile-p512-20260526.json` |
| Qwen3.6 27B 4-bit | 2048 | 32.609 tok/s | 32.623 tok/s | 1.000x | `/private/tmp/ax-vs-mlxlm-qwen27b-4bit-current-p2048-20260526.json` |
| Gemma4 31B 4-bit | 128 | 28.299 tok/s | 28.611 tok/s | 1.011x | `/private/tmp/ax-vs-mlxlm-gemma31b-4bit-current-p128-20260526.json` |
| Gemma4 31B 4-bit | 512 | 27.732 tok/s | 27.887 tok/s | 1.006x | `/private/tmp/ax-vs-mlxlm-gemma31b-4bit-current-p512-20260526.json` |
| Gemma4 31B 4-bit | 2048 | 21.927 tok/s | 24.613 tok/s | 1.122x | `/private/tmp/ax-vs-mlxlm-gemma31b-4bit-current-p2048-20260526.json` |

Notes:

- The Gemma31 p2048 row is especially sensitive to run-to-run variance: a
  reused-reference check measured `mlx_lm = 27.035 tok/s` and AX
  `26.926 tok/s`, while the later same-session rerun measured
  `mlx_lm = 21.927 tok/s` and AX `24.613 tok/s`. Treat the same-session row as
  the current diagnostic result, but do not publish it until the full clean
  matrix is regenerated.
- The Qwen p512 row was first compared against the older reused reference
  artifact (`mlx_lm = 33.905 tok/s`) and looked like a fail. A same-session
  current `mlx_lm` rerun measured `32.946 tok/s`, while AX measured
  `33.030 tok/s`. The old reference row should not be used for final claims.
- The Qwen runtime pack audit confirms AX already loads Qwen linear-attention
  projections as `qkvz_ba_packed_layers = 48`, matching the `mlx_lm`
  `in_proj_qkvz` / `in_proj_ba` design. The benchmark JSON's
  `linear_attention_projection_layout` describes the source artifact layout,
  not the post-load runtime layout.
- Qwen p512 stage profiling after the alignment still shows the route is
  dominated by GPU completion behind `async_eval`: `3,585,903 us` of
  `3,870,838 us` decode wall, with `43` ops per linear-attention layer and
  `31` ops per full-attention layer.

## 2026-05-26 Current Matrix Spot Checks

The latest reruns deliberately used same-session `mlx_lm.benchmark` references
instead of mixing older reference artifacts. That changed several apparent
failures into passes, but it also confirmed one real near-fail.

Passing focused rows:

| Model | Prompt | mlx_lm decode | AX direct decode | Ratio | Artifact |
|---|---:|---:|---:|---:|---|
| Gemma4 26B A4B 4-bit | 128 | 126.141 tok/s | 133.568 tok/s | 1.059x | `/private/tmp/ax-vs-mlxlm-gemma26b-a4b-4bit-current-p128-20260526.json` |
| Qwen3.6 27B 5-bit | 512 | 27.579 tok/s | 27.660 tok/s | 1.003x | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p512-20260526.json` |
| Qwen3.6 27B 6-bit | 128 | 24.417 tok/s | 24.684 tok/s | 1.011x | `/private/tmp/ax-vs-mlxlm-qwen27b-6bit-current-p128-20260526.json` |
| Qwen3.6 27B 6-bit | 512 | 24.397 tok/s | 24.600 tok/s | 1.008x | `/private/tmp/ax-vs-mlxlm-qwen27b-6bit-current-p512-20260526.json` |
| Qwen3.6 27B 6-bit | 2048 | 24.035 tok/s | 24.085 tok/s | 1.002x | `/private/tmp/ax-vs-mlxlm-qwen27b-6bit-current-p2048-20260526.json` |
| Qwen3.6 27B 8-bit | 128 | 18.303 tok/s | 18.376 tok/s | 1.004x | `/private/tmp/ax-vs-mlxlm-qwen27b-8bit-current-p128-r3-20260526.json` |
| Qwen3.6 27B 8-bit | 512 | 17.963 tok/s | 18.015 tok/s | 1.003x | `/private/tmp/ax-vs-mlxlm-qwen27b-8bit-current-p512-20260526.json` |
| Qwen3.6 27B 8-bit | 2048 | 18.015 tok/s | 18.173 tok/s | 1.009x | `/private/tmp/ax-vs-mlxlm-qwen27b-8bit-current-p2048-20260526.json` |
| Qwen3.6 35B A3B 4-bit | 2048 | 131.092 tok/s | 154.232 tok/s | 1.177x | `/private/tmp/ax-vs-mlxlm-qwen35b-a3b-4bit-current-p2048-20260526.json` |

Unresolved focused row:

| Model | Prompt | mlx_lm decode | AX direct decode | Ratio | Artifact |
|---|---:|---:|---:|---:|---|
| Qwen3.6 27B 5-bit | 2048 | 27.120 tok/s | 27.062 tok/s | 0.998x | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p2048-r3-20260526.json` |

The post-input direct C++ route was also checked because it is an existing
shape-checked mirror of the Qwen linear-attention post-input block:

| Probe | mlx_lm decode | AX direct decode | Ratio | Route hits | Artifact |
|---|---:|---:|---:|---:|---|
| inputs + post-input, single rep | 27.292 tok/s | 27.394 tok/s | 1.004x | 48/48 post-input | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p2048-directcpp-compare-20260526.json` |
| post-input via env, 3 reps | 26.864 tok/s | 27.076 tok/s | 1.008x | 144/144 post-input | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p2048-postinput-r3-20260526.json` |
| post-input default promotion probe, 3 reps | 27.184 tok/s | 26.960 tok/s | 0.992x | 144/144 post-input | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p2048-default-postinput-r3-20260526.json` |

Interpretation: the route engages cleanly, but the benefit does not survive a
same-session promotion probe. It remains opt-in and should not be promoted as
the fix for the 5-bit p2048 row.

QK-norm + RoPE direct C++ was checked next against the `mlx_lm` Qwen3Next
attention sequence (`q_proj/k_proj -> head-wise RMSNorm -> transpose -> RoPE`).
It should remain opt-in: the first env-probe looked positive, but default-route
and disabled-route same-session reruns both stayed below `mlx_lm`.

| Probe | mlx_lm decode | AX direct decode | Ratio | Artifact |
|---|---:|---:|---:|---|
| `AX_MLX_DIRECT_CPP_QK_NORM_ROPE=1`, 3 reps | 27.206 tok/s | 27.676 tok/s | 1.017x | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p2048-qk-rope-r3-20260526.json` |
| Qwen-only default promotion probe, 3 reps | 27.397 tok/s | 27.199 tok/s | 0.993x | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p2048-qk-rope-default-r3-20260526.json` |
| Qwen route disabled, 3 reps | 27.315 tok/s | 27.115 tok/s | 0.993x | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p2048-qk-rope-disabled-r3-20260526.json` |

Interpretation: QK+RoPE direct C++ is not a stable default fix for the 5-bit
p2048 row. The attempted default promotion was reverted; the existing global
`AX_MLX_DIRECT_CPP_QK_NORM_ROPE` probe remains opt-in only.

One source-backed scheduling alignment was kept after comparing AX's direct
pipeline with `mlx_lm.generate_step`: `mlx_lm` clears the MLX graph cache after
the first generated token and then again at the 257th generated token. AX's
prefill-boundary path already clears after the first generated token, but the
direct pipeline emitted-token counter still started at zero, causing another
decode-interval clear after token 2. The counter now starts at 1 when the
prefill-produced token primes direct pipeline.

| Probe | mlx_lm decode | AX direct decode | Ratio | Artifact |
|---|---:|---:|---:|---|
| direct clear-cache cadence aligned, 3 reps | 27.670 tok/s | 27.400 tok/s | 0.990x | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p2048-clear-cadence-r3-20260526.json` |

Interpretation: the alignment improves AX absolute decode versus the preceding
same-shape runs (`~27.1-27.2 tok/s` -> `27.4 tok/s` median), but it still does
not close the same-session gap because the paired `mlx_lm` row also improved.

## 2026-05-26 Qwen Linear-Attention Default Route

The follow-up compared AX directly against the local `mlx_lm` implementation
instead of assuming the direct C++ route should help. `mlx_lm.generate_step`
submits decode work through `mx.async_eval` and clears the lazy graph cache on
the generation cadence, while the Qwen model path uses packed QKVZ/BA
projections plus a Qwen gated-delta post-input block. AX already had both
shape-checked direct C++ shims, but only as opt-in flags.

Applied change:

- `AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS` is now default-on for
  `qwen3_5` / `qwen3_next` only, with `=0` as a kill switch.
- `AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT` is now default-on for
  `qwen3_5` / `qwen3_next` only, with `=0` as a kill switch.
- The original global probes remain opt-in:
  `AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS=1` and
  `AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT=1`.
- Non-Qwen families stay on their existing path unless the global opt-in probes
  are explicitly set.

Validation:

```bash
cargo fmt --check
cargo test -p ax-engine-mlx qwen_direct_cpp_linear_attention_inputs_uses_default_on_contract --quiet
cargo test -p ax-engine-mlx qwen_direct_cpp_linear_attention_post_input_uses_default_on_contract --quiet
cargo test -p ax-engine-mlx linear_attention_direct_cpp_post_input_route_decisions_emit_when_attempted --quiet
cargo test -p ax-engine-mlx direct_pipeline_clear_cache_cadence_matches_mlx_lm_loop --quiet
python3 scripts/test_bench_mlx_inference_stack.py
cargo build --release -p ax-engine-server
```

Focused Qwen3.6 27B 5-bit p2048/g128 results:

| Probe | Cooldown | mlx_lm decode | AX direct decode | Ratio | Route hits | Artifact |
|---|---:|---:|---:|---:|---:|---|
| post-input default only | 1s | 27.607 tok/s | 27.423 tok/s | 0.993x | post-input 144/144 | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p2048-postinput-default-r3-20260526.json` |
| inputs env + post-input default | 1s | 27.626 tok/s | 27.812 tok/s | 1.007x | inputs 144/144, post-input 144/144 | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p2048-inputs-default-postinput-r3-20260526.json` |
| inputs + post-input default | 1s | 27.625 tok/s | 27.338 tok/s | 0.990x | inputs 144/144, post-input 144/144 | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p2048-qwen-default-r3-20260526.json` |
| inputs + post-input default | 20s | 27.277 tok/s | 27.463 tok/s | 1.007x | inputs 144/144, post-input 144/144 | `/private/tmp/ax-vs-mlxlm-qwen27b-5bit-current-p2048-qwen-default-cool20-r3-20260526.json` |

Interpretation: post-input alone is not sufficient as a stable default. The
source-backed route that survives the longer-cooldown check is the pair of Qwen
linear-attention input staging plus post-input shims. The 1-second-cooldown
default miss had the same route hit counters as the passing run, so it should be
treated as thermal/run-to-run variance rather than proof that the route was
inactive. Do not publish the 5-bit p2048 row from the short-cooldown miss.

## 2026-05-26 Qwen 27B Acceptance Matrix

Fresh same-session matrix:

- Output root:
  `/private/tmp/ax-direct-current-qwen27-acceptance-20260526/`
- Shape: Qwen3.6 27B, prompt=128/512/2048, generation=128,
  repetitions=3, cooldown=20s, `--ax-direct --no-build-ax-engine`.
- Acceptance rule: PASS if AX direct is at least 0.980x of `mlx_lm`.

| Quant | Prompt | mlx_lm decode | AX direct decode | Ratio | Route hits | Status |
|---|---:|---:|---:|---:|---|---|
| 4-bit | 128 | 33.238 tok/s | 33.313 tok/s | 1.002x | inputs all, post-input all | PASS |
| 4-bit | 512 | 33.127 tok/s | 33.175 tok/s | 1.001x | inputs all, post-input all | PASS |
| 4-bit | 2048 | 32.758 tok/s | 32.802 tok/s | 1.001x | inputs all, post-input all | PASS |
| 5-bit | 128 | 27.623 tok/s | 27.658 tok/s | 1.001x | inputs all, post-input all | PASS |
| 5-bit | 512 | 27.414 tok/s | 27.524 tok/s | 1.004x | inputs all, post-input all | PASS |
| 5-bit | 2048 | 27.264 tok/s | 27.333 tok/s | 1.003x | inputs all, post-input all | PASS |
| 6-bit | 128 | 24.384 tok/s | 24.537 tok/s | 1.006x | inputs all, post-input all | PASS |
| 6-bit | 512 | 24.320 tok/s | 24.485 tok/s | 1.007x | inputs all, post-input all | PASS |
| 6-bit | 2048 | 24.105 tok/s | 24.363 tok/s | 1.011x | inputs all, post-input all | PASS |
| 8-bit | 128 | 18.081 tok/s | 18.064 tok/s | 0.999x | inputs all, post-input all | PASS |
| 8-bit | 512 | 18.071 tok/s | 18.131 tok/s | 1.003x | inputs all, post-input all | PASS |
| 8-bit | 2048 | 17.847 tok/s | 18.024 tok/s | 1.010x | inputs all, post-input all | PASS |

Interpretation: Qwen3.6 27B now passes the 2% acceptance rule for every tested
quantization and prompt shape. The worst row is 8-bit prompt 128 at 0.999x
(`0.10%` below `mlx_lm`), which is inside the accepted tolerance. Every row
hit both Qwen direct linear-attention routes.

## 2026-05-26 Published Direct-Mode Acceptance Sweep

Consolidated artifacts:

- Published tracked copy:
  `benchmarks/results/mlx-inference/2026-05-26-direct-mode-acceptance/`
- Qwen 27B:
  `/private/tmp/ax-direct-current-qwen27-acceptance-20260526/`
- Gemma 4 and Qwen 35B:
  `/private/tmp/ax-direct-current-full-acceptance-20260526/`

Scope: the 12 README-published direct-mode model/quant rows, prompt
128/512/2048, generation=128, repetitions=3, cooldown=20s. Acceptance rule:
PASS if AX direct is at least 0.980x of `mlx_lm`.

| Model row | Reference | Worst prompt | Worst ratio | Status |
|---|---|---:|---:|---|
| Gemma 4 E2B 4-bit | reused `2026-05-18` mlx_lm | 512 | 1.062x | PASS |
| Gemma 4 E2B 5-bit | reused `2026-05-18` mlx_lm | 2048 | 1.058x | PASS |
| Gemma 4 E2B 6-bit | reused `2026-05-18` mlx_lm | 2048 | 1.058x | PASS |
| Gemma 4 E2B 8-bit | reused `2026-05-18` mlx_lm | 128 | 1.051x | PASS |
| Gemma 4 E4B 4-bit | reused `2026-05-18` mlx_lm | 128 | 1.029x | PASS |
| Gemma 4 26B A4B 4-bit | same-session | 512 | 1.058x | PASS |
| Gemma 4 31B 4-bit | same-session | 128 | 1.005x | PASS |
| Qwen3.6 27B 4-bit | same-session | 2048 | 1.001x | PASS |
| Qwen3.6 27B 5-bit | same-session | 128 | 1.001x | PASS |
| Qwen3.6 27B 6-bit | same-session | 128 | 1.006x | PASS |
| Qwen3.6 27B 8-bit | same-session | 128 | 0.999x | PASS |
| Qwen3.6 35B A3B 4-bit | same-session | 2048 | 1.176x | PASS |

Full-row interpretation:

- All 36 prompt rows pass the 0.980x acceptance rule.
- 35/36 rows are strictly above `mlx_lm`.
- The only row below `mlx_lm` is Qwen3.6 27B 8-bit prompt 128 at 0.999x
  (`0.10%` below `mlx_lm`), which is inside the user-approved 2% tolerance.
- Current installed `mlx_lm.benchmark` no longer loads the dense Gemma E2B/E4B
  snapshots used by the README rows: it reports extra self-attention
  parameters after the model depth expected by the loader. For those rows, the
  harness reused the existing same-snapshot, same-prompt `mlx_lm` reference
  rows from `benchmarks/results/mlx-inference/2026-05-18-mlx-lm-llamacpp-sweep/`
  and refreshed only current AX direct rows with `--reuse-reference-results-from`.
- Same-session rows cover Qwen 27B, Qwen 35B, Gemma 26B A4B, and Gemma 31B.

## Next Steps

1. Keep the dense-Gemma residual/norm shims opt-in only; current Gemma evidence
   showed regression, and the Qwen linear-layer variant did not improve decode.
2. Keep QK+RoPE direct C++ opt-in only; its positive env-probe did not survive
   same-session promotion checks.
3. The README and performance charts now use
   `benchmarks/results/mlx-inference/2026-05-26-direct-mode-acceptance/` for
   the current `mlx_lm` reference and AX direct-mode rows. The existing
   `2026-05-25-ax-only-direct-mode-refresh` artifacts remain the README source
   for default n-gram rows until a fresh n-gram sweep is run.
