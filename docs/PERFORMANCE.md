# Performance

This page is the public performance-results reference. It keeps the result
tables, artifact summaries, interpretation, and claim boundaries for the
current public snapshot. The root `README.md` intentionally keeps only the
common Gemma 4 and Qwen 3.6 rows.

For benchmark methodology, test setup, commands, reproduction details, and
evidence classification, see [`docs/BENCHMARKS.md`](BENCHMARKS.md).

## Current Result Set

The current public table was refreshed on 2026-05-14 on:

- Apple M5 Max
- 128 GB memory
- macOS 26.4.1

Benchmark shape:

- random-token prompts from the `mlx_lm.benchmark` seed-0 contract
- batch size 1
- `prefill_step_size=2048`
- 128 generated tokens
- temperature 0
- 5 repetitions per engine/model/prompt row
- 15-second cooldown between trials

The current README generation-model snapshot is a provenance-tracked composite
from:

```text
benchmarks/results/mlx-inference/2026-05-21-ax-only-post-shim-sweep/
```

The directory's 12-model AX refresh reruns the direct and n-gram AX rows on
current binaries (post linear-attention shim) with generation=128, 5
repetitions, and a 15-second cooldown. Same-host `mlx_lm` reference rows
come from
`benchmarks/results/mlx-inference/2026-05-18-mlx-lm-llamacpp-sweep/` and are
pulled in by the README composite at chart-render time, while each AX
artifact's own `ax_only_refresh.reference_results_source` field records the
exact carry-forward file. All rows use the same prompt-token contract and
prompt SHA checks. The AX rows were produced with production-build server
binaries (`[profile.release]`: LTO thin, `codegen-units=1`, `panic=abort`,
stripped debuginfo). This composite is intentional: the third-party reference
rows stay attached to the same artifact contract while AX-only refreshes can
update current runtime behavior without rerunning the reference adapters.

`ax direct baseline` is the direct same-policy comparison against `mlx_lm`; the
benchmark starts the AX server with n-gram acceleration disabled for this row.
`ax default n-gram` reports observed effective throughput from AX's default
n-gram acceleration policy. It is not raw model-kernel decode speed.
The README prefill table uses the direct AX row, because n-gram acceleration is
a decode policy and should not be credited as a prefill optimization.

## Latest Long-Context Artifacts

The public README table is intentionally short/mid-prompt evidence. The latest
checked-in heavy real-model validation is separate:

| Artifact | Model | Shape | Key result | Interpretation |
|---|---|---|---|---|
| [P1 prefill scaling](../benchmarks/results/mlx-inference/2026-05-15-long-context/qwen3-4b-4bit-prefill-scaling/prefill-scaling.md) | `mlx-community/Qwen3-4B-4bit` | 1k/2k/4k/8k context, generation=1, repetitions=3 | AX/MLX prefill ratio moves from 1.190x at 1k to 1.154x at 8k, with every measured context above `mlx_lm` | AX now has a positive Qwen3-4B cold-prefill boundary on this host; keep it separate from family-wide and serving-concurrency claims |
| [P2 startup and concurrency](../benchmarks/results/mlx-inference/2026-05-07-real-p2/qwen3-4b-4bit-p2-latency/p2-latency.md) | `mlx-community/Qwen3-4B-4bit` | 8k context; startup generation=128; concurrent generation=1; concurrency 1/2/4 | 8k warm TTFT 2509.7 ms; 4-request concurrent prefill TTFT 8318.7 ms and overlap classified serialized | This does not support a continuous-batching or concurrent-prefill-overlap claim yet |

Both artifacts use direct AX MLX policy and are not evidence for n-gram decode
acceleration. They are included to show that review covered TTFT, prefill
scaling, startup, queueing, memory, and concurrency boundaries beyond the
README throughput table.

## Latest Prefill Breakdown

The 2026-05-13 short-prompt TTFT follow-up decomposes AX MLX prefill wall time
for Qwen Coder Next, a retired Qwen 3.6 candidate, and GLM 4.7 Flash:
[prefill breakdown](../benchmarks/results/mlx-inference/2026-05-13-ttft-breakdown/prefill-breakdown.md).
It separates model forward time from prefix-cache snapshot storage and
generation-state initialization. Diagnostic profile artifacts in the same
directory are intentionally excluded from the default rendered report unless
`scripts/render_mlx_prefill_breakdown_report.py --include-diagnostics` is used.

## Disk-Durable Prefix Cache Evidence

The disk-durable prefix cache is an opt-in runtime feature, not part of the
default public throughput table. It is designed for process-restart and
same-host multi-process prefix reuse: set `AX_MLX_PREFIX_CACHE_DIR` to enable
the L2 `.axkv` store, and keep it unset for the historical in-memory-only path.

Current checked-in evidence covers correctness and cache primitive safety:

| Artifact | Coverage | Key result | Interpretation |
|---|---|---|---|
| `benchmarks/results/disk-prefix-cache-cross-restart/gemma4-e2b-2026-05-14.json` | Gemma 4 E2B, standard FA + sliding window | PASS, 2/2 token-exact, 2 phase-B disk hits | Cross-restart restore works for the Gemma tier |
| `benchmarks/results/disk-prefix-cache-cross-restart/qwen35-9b-2026-05-14.json` | Qwen3.5-9B, hybrid MLA + linear attention | PASS, 2/2 token-exact, 2 phase-B disk hits | Cross-restart restore works for the hybrid tier |
| `benchmarks/results/disk-prefix-cache-cross-restart/glm47-flash-2026-05-14.json` | GLM-4.7-Flash, pure MLA | PASS, 2/2 token-exact, 2 phase-B disk hits | Cross-restart restore works for the pure-MLA tier |
| `benchmarks/results/disk-prefix-cache-stress/2026-05-14-m3b-stress.json` | 4 worker processes over overlapping keys plus tight eviction pressure | PASS, zero corruption load failures, zero read misses, 3 evictions | The disk-cache primitive survived short concurrent writer and eviction stress |

This evidence supports an opt-in runtime claim: AX can persist validated MLX
prefix snapshots to a local disk cache and restore them across process
restarts. It does not yet support a broad production-serving performance claim.
For that, use a serving artifact that measures request latency, queueing,
memory pressure, cache hit rate, and failure rate under the target process
count and prompt mix.

## Comprehensive Review Scope

The current public table has been reviewed as a scoped MLX model-inference
claim, not as a general production-serving benchmark. The review covers:

| Review dimension | Current evidence | What it supports |
|---|---|---|
| Reference parity | Matching `mlx_lm.benchmark` seed-0 prompt/decode shapes, plus admitted `mlx_swift_lm` secondary rows where available | Same-shape comparisons against named MLX references |
| Prompt provenance | Prompt-token JSON artifacts with fixed token IDs, vocabulary size, seed, and hash | Reproducible prompt input across AX and reference rows |
| Decode policy separation | Direct AX rows run with n-gram acceleration disabled; default AX rows report n-gram effective throughput | Clear separation between same-policy decode and AX user-default acceleration |
| Prefill/decode split | Tables report prefill tok/s separately from decode tok/s | Prefill gains are not attributed to n-gram decode acceleration |
| Repeated measurement | 5 repetitions per engine/model/prompt row, reported through medians | Reduced single-run noise within the current host and prompt shape |
| Route identity | AX artifacts record runtime route and fixed-schema n-gram telemetry fields | No-draft fallback, direct pipeline, and n-gram rows can be distinguished |
| Output-quality caveat | Random-token greedy prompts can collapse into repeated output, but throughput artifact status stays limited to direct/n-gram fallback evidence | Public rows are performance evidence; coherent-output claims require separate token-output validation |
| Scope disclosure | Methodology states host, batch size, generated-token count, temperature, and `prefill_step_size` | Readers can see that current public rows are batch=1, short/mid-prompt evidence |

This review intentionally rejects broader conclusions that the current table
does not prove. The rows support row-by-row claims about AX direct or AX n-gram
throughput against named MLX references on matching Apple Silicon benchmark
shapes; they do not say every AX policy wins every row. They also cannot, by
themselves, support claims about 32k/128k context scaling, cold-start latency,
multi-user serving throughput, KV eviction behavior, or parity with CUDA server
systems such as vLLM or TensorRT-LLM.

## Interpretation

The current composite result set is not a universal direct-decode win.
Direct AX is intentionally measured with n-gram acceleration disabled, and the
direct column spans -14.4% to +22.9% versus `mlx_lm` across the full decode
artifact set. In the README Gemma 4 / Qwen 3.6 snapshot, direct decode spans
-14.4% to +9.1%. Those rows are the same-policy baseline rather than the
default AX user path.

N-gram acceleration is the default AX user/server path and is the better
headline row for decode-throughput expectations when the workload produces
draftable local repetition. In the README snapshot, the n-gram column spans
+86.7% to +207.2% versus `mlx_lm`; across the full artifact set it spans +0.0%
to +207.2%. Artifact claim status is explicit: 26 of 28 n-gram rows are labeled
`ngram_acceleration_effective_throughput`; the two no-draft rows are labeled
`ngram_no_draft_direct_fallback`. Those no-draft rows are Qwen Coder Next at
128 prompt tokens and Qwen 3.5 9B at 512 prompt tokens, which are outside the
concise README snapshot. Public docs should keep direct baseline rows visible
beside the default n-gram row.

The high end of the n-gram column is sensitive to a second regime worth
naming explicitly: random-token benchmark prompts at greedy decode can
push the model into a repeated-output loop, and both the direct and the
n-gram row then measure throughput on that loop rather than on healthy
decode. Current benchmark artifacts do not encode that as a claim-status
variant; `ax_decode_claim_status` is intentionally limited to throughput and
fallback state. Treat random-token n-gram wins as synthetic throughput rows
unless a separate token-output validation artifact accompanies the claim. See
[`docs/NGRAM-ACCELERATION.md` § Synthetic repeated-output loops](NGRAM-ACCELERATION.md#synthetic-repeated-output-loops)
for the interpretation rule.

The strongest user-facing case for n-gram acceleration is coding-shaped output
with repeated local structure: completion, edit loops, structured diffs,
repeated identifiers, indentation, imports, JSON/tool payloads, tests, and
config generation. These workloads often repeat short local token patterns,
which is exactly where AX's draft-and-verify n-gram policy can improve effective
decode throughput. The benchmark tables still report measured rows, not a
blanket guarantee. Novel code, high-entropy explanations, short answers, or
deliberately diverse coding requests can see little benefit and fall back toward
the direct path.

Qwen-family linear-attention n-gram rows use a rollback-safe branch/recompute
path for SSM state. Acceleration is prompt/output-pattern dependent. Benchmark
JSON artifacts include fixed-schema n-gram telemetry fields, and the README
table uses median AX runner timing plus output-token count.

## Evidence Boundaries

The current public benchmark is strong for the narrow claim it makes, but it
should not be read as a complete inference-serving proof. In particular:

- Prefill wins are meaningful because n-gram acceleration primarily affects
  decode. A faster AX prefill row points to runtime, graph, cache, or kernel
  behavior, not to speculative decoding alone.
- Apple Silicon and MLX explain the baseline environment, but they do not by
  themselves explain AX-vs-`mlx_lm` differences because the primary reference
  also runs on MLX.
- Current public rows use batch size 1 and prompt-token counts shown in the
  README table. They do not measure continuous batching, concurrent prefill, or
  queueing delay.
- Current public rows are warm benchmark rows. They do not measure model load,
  first request after process start, shader compilation before warmup, or cache
  recovery after memory pressure.
- Current public rows do not establish a long-context scaling curve. Claims
  about 8k, 32k, 64k, or 128k behavior require separate artifacts with TTFT,
  prefill tok/s, memory pressure, and route telemetry.
- Current public rows do not prove KV eviction or fragmentation behavior. They
  validate the benchmark path, not long-lived multi-session cache management.
- Disk-durable prefix-cache artifacts prove cross-restart correctness and short
  cache-primitive stress. They do not replace a full AX-serving soak for
  production-serving latency or availability claims.

## MTP Mode

AX MTP is benchmarked against standard Qwen3.6 sidecars, not
`Youssofal/*MTPLX*` bundles. The sidecar builder combines the ordinary
`mlx-community/*-4bit` MLX base with the official `Qwen/Qwen3.6-*` MTP shard
weights and writes `ax_mtp_sidecar_manifest.json` so every row records the base
snapshot, source shard hashes, MTP output hash, transform policy, and supported
depth.

**Sidecar naming constraint**: `prepare_qwen36_mtp_sidecar.py` writes `mtp.safetensors`
only (not `model-mtp.safetensors`). mlx_lm loads all `model*.safetensors` files in a
directory via glob (`utils.py` line 316); a file named `model-mtp.safetensors` would
cause `TextModel.sanitize` to see `has_mtp_weights=True` and apply a second +1.0 shift
to the already-shifted base model norms, producing garbage output from any mlx_lm-based
inference path (MTPLX, direct `mlx_lm.generate`). AX Engine is unaffected
because it uses its own loader. Always keep the MTP sidecar file named `mtp.safetensors`.

The fair harness is `scripts/bench_qwen36_mtp_fair.py`. It runs the same
prompt suite files, max-token cap, sampler, warmup count, measured repetitions,
and cooldown across MTPLX and AX Engine. The comparison uses native depth:
27B at depth 3, 35B-A3B at depth 1.

### Corrected 2026-05-30 run

The initial 2026-05-30 stable-profile re-run showed MTPLX at ~2% accept rate.
Root-cause investigation identified:

1. **MTPLX base-model corruption** (fixed). The sidecar directory contained a
   `model-mtp.safetensors` hard-link alias that matched `mlx_lm`'s
   `model*.safetensors` glob, causing `TextModel.sanitize` to apply a second
   +1.0 shift to already-shifted norm weights. The alias has been removed from
   `prepare_qwen36_mtp_sidecar.py`. After this fix, MTPLX accept rate rose from
   ~2% to ~99.9% on standard Qwen3.6 sidecars.

### Accept rate improvement (2026-05-30 → 2026-05-31)

Three rounds of fixes improved AX Engine's MTP accept rate:

1. **Depth policy cap removed** (2026-05-30). `default_mtp_depth_without_env()`
   capped all models to depth 1 even when the sidecar specified
   `mtp_depth_max: 3`. Now removed — models use their configured native depth.

2. **Filtered lm_head rejection sampling** (2026-05-30). The filtered candidate
   path skipped computing draft log-probs, which forced greedy argmax
   acceptance. Now computes temperature-scaled softmax over the top-4096
   candidate logits for rejection sampling acceptance.

3. **Full-vocab target softmax** (2026-05-31). The previous target distribution
   path extracted top-k=20 tokens from verify logits and normalized within that
   set, zeroing `p_target` for any draft token ranked below 20th — guaranteeing
   rejection. The new path computes full-vocab softmax (matching MTPLX's
   approach) and gathers only the per-draft-token probability. Additionally,
   the draft candidate set is no longer halved at each depth, giving the draft
   model the full candidate set at all depths.

35B-A3B results (native depth=1, pure MTP, 1000 gen tokens):

| Engine | flappy | long_code | python_modules_long |
|---|---:|---:|---:|
| MTPLX 0.3.7 (tok/s) | 88.1 | 105.2 | 95.2 |
| MTPLX 0.3.7 (accept) | 48.8% | 52.3% | 42.3% |
| AX Engine (tok/s) | 84.2 | 81.5 | 77.9 |
| AX Engine (accept) | 99.9% | 99.8% | 93.2% |
| AX/MTPLX ratio | 0.956 | 0.775 | 0.819 |

27B results (native depth=3, pure MTP, 1000 gen tokens):

| Engine | flappy | long_code | python_modules_long |
|---|---:|---:|---:|
| MTPLX 0.3.7 (tok/s) | 39.2 | 44.3 | 47.7 |
| MTPLX 0.3.7 (accept) | 100.0% | 99.7% | 87.6% |
| AX Engine (tok/s) | 37.2 | 27.6 | 22.9 |
| AX Engine (accept) | 99.1% | 98.3% | 67.0% |
| AX/MTPLX ratio | 0.949 | 0.625 | 0.480 |

Artifacts: `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/` (dual-engine,
full-vocab draft log-prob fix).

#### Chart data table

<!-- This table is consumed by scripts/render_mtp_flappy_charts.py -->

| Model bundle | Suite | AX depth cap | AX MTP tok/s | AX accept % | MTPLX tok/s | MTPLX depth | MTPLX accept % | AX/MTPLX |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Speed (Qwen3.6 35B-A3B 4-bit) | flappy | 1 | 84.2 | 99.9% | 88.1 | 1 | 48.8% | 0.956 |
| Speed (Qwen3.6 35B-A3B 4-bit) | long_code | 1 | 81.5 | 99.8% | 105.2 | 1 | 52.3% | 0.775 |
| Quality (Qwen3.6 27B 4-bit) | flappy | 3 | 37.2 | 99.1% | 39.2 | 3 | 100.0% | 0.949 |
| Quality (Qwen3.6 27B 4-bit) | long_code | 3 | 27.6 | 98.3% | 44.3 | 3 | 99.7% | 0.625 |

### Current publication contract

- Models: `27b-4bit` and `35b-a3b-4bit`.
- Provenance: standard `Qwen/Qwen3.6-27B` or `Qwen/Qwen3.6-35B-A3B` MTP shards
  plus `mlx-community/Qwen3.6-27B-4bit` or
  `mlx-community/Qwen3.6-35B-A3B-4bit`.
- Suites: `flappy` and `long_code` from `benchmarks/prompts/mtp-suites/`.
- Sampler: sampled decode, temperature=0.6, top_p=0.95, top_k=20.
- Publication shape: max_tokens=1000, 1 warmup repetition, 5 measured
  repetitions, 15 second cooldown.
- Native depth: 27B `3`, 35B-A3B `1`.

Prepare the sidecars:

```bash
python3 scripts/prepare_qwen36_mtp_sidecar.py --model 27b
python3 scripts/prepare_qwen36_mtp_sidecar.py --model 35b
```

Run the dual-engine fair comparison at native depth:

```bash
python3 scripts/bench_qwen36_mtp_fair.py \
  --models 27b-4bit 35b-a3b-4bit \
  --engines mtplx ax_engine \
  --suites flappy long_code \
  --depth-policy native \
  --max-tokens 1000 \
  --repetitions 5 \
  --cooldown 15
```

The harness writes `summary.json`, `summary.md`, and `decode-tok-s.svg` under
`benchmarks/results/mtp-fair/<date>-qwen36-fair/`. A row with `status=error` is
kept in the summary instead of silently dropping a backend; that is deliberate
so MTPLX loader incompatibilities (e.g. 35B-A3B MoE) stay visible.

## Additional Testing Plan

More testing is needed before making production-serving or long-context claims.
The recommended sequence is:

| Priority | Test | Shape | Acceptance evidence |
|---|---|---|---|
| P0 | Public claim gate | For every README row, verify matching reference row, prompt hash, AX decode policy, route identity, and median/repetition metadata | A small script or CI check that fails closed when any public table row lacks matching artifact provenance |
| P1 | Prefill scaling curve | Run representative Gemma, Qwen, and GLM rows at 1k, 2k, 4k, 8k, 16k, and the largest supported context on the host | `ax.mlx_prefill_scaling.v1` artifact with prefill tok/s, TTFT, peak memory, prompt hash, direct AX policy, and ratios vs `mlx_lm`; explicitly mark the context where throughput bends |
| P1 | TTFT under long context | Use real chat-shaped prompts at 8k/16k/32k where supported, generation=1 and generation=128 | Median and p75 TTFT, first-token route telemetry, and output correctness/determinism checks |
| P2 | Cold vs warm startup | Compare process-cold, model-warm, and benchmark-warm runs for the same model and prompt shape | `ax.mlx_startup_latency.v1` artifact with separate server-ready time, model-load time, first-request TTFT, warm TTFT, warmed decode tok/s, peak memory, prompt hash, direct AX policy, and ratios vs the benchmark-warm row |
| P2 | Concurrent prefill | Submit 2/4/8 simultaneous long prompts through the server path | `ax.mlx_concurrent_prefill.v1` artifact with per-request TTFT, total wall time, queueing delay, zero-failure evidence, peak memory, prompt hashes per request, direct AX policy, overlap classification, and ratios vs the single-request baseline |
| P2 | KV churn and prefix reuse | Run replay manifests with shared prefixes, branch decode, cancel/reuse, and retained-prefix cleanup | Route decisions, retained-prefix hit rate, memory-blocked recovery, and correctness after cancellation |
| P3 | Long-session KV policy | Keep multiple sessions alive across long prompts and decode bursts until memory pressure appears | Cache growth, eviction/fallback decisions, latency tail, and absence of stale-KV correctness regressions |
| P3 | Cross-host repeatability | Repeat the public and long-context subsets on at least one smaller Apple Silicon host | Ratios vs reference, thermal notes, and host-specific caveats |

The P0 public claim gate is executable:

```text
python3 scripts/check_readme_performance_artifacts.py
```

It parses the README decode and prefill tables, resolves the referenced
`benchmarks/results/mlx-inference/.../` artifact directory, and fails if any
public row lacks a matching median value, prompt-token hash, reference row,
decode-policy label, or repetition metadata.

The P1 prefill/TTFT scaling artifact gate is executable for saved long-context
artifacts:

```text
scripts/run-mlx-prefill-scaling-artifact.sh \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 1024,2048,4096,8192,16384 \
  --run-label qwen35-prefill-scaling
```

For already completed MLX inference-stack runs, build and validate the scaling
artifact directly:

```text
python3 scripts/build_mlx_prefill_scaling_artifact.py \
  benchmarks/results/mlx-inference/<date>/<model>.json \
  --output benchmarks/results/mlx-inference/<date>/<model>-prefill-scaling.json

python3 scripts/check_mlx_prefill_scaling_artifact.py \
  benchmarks/results/mlx-inference/<date>/<model>-prefill-scaling.json

python3 scripts/render_mlx_prefill_scaling_report.py \
  benchmarks/results/mlx-inference/<date>/<model>-prefill-scaling.json \
  --output benchmarks/results/mlx-inference/<date>/<model>-prefill-scaling.md

python3 scripts/check_mlx_prefill_scaling_campaign.py \
  benchmarks/results/mlx-inference/<date>/gemma-prefill-scaling.json \
  benchmarks/results/mlx-inference/<date>/qwen-prefill-scaling.json \
  benchmarks/results/mlx-inference/<date>/glm-prefill-scaling.json \
  --default-required-families \
  --output benchmarks/results/mlx-inference/<date>/prefill-scaling-campaign.md
```

The builder does not run the model; it converts a saved
`ax.mlx_inference_stack.v2` run into `ax.mlx_prefill_scaling.v1`. New AX rows
from the inference-stack harness record runner-derived `ttft_ms` and
server-process RSS memory. `mlx_lm` TTFT is derived from its reported prefill
throughput because `mlx_lm.benchmark` does not expose a separate first-token
event in the same artifact shape. The validator then verifies that
long-context evidence includes at least two AX context points up to 8k tokens
by default, a matching `mlx_lm` baseline for every shape, one prompt hash shared
across engines, direct AX policy labeling, median prefill tok/s, median TTFT,
peak memory, and fresh ratios to the baseline. This is the guardrail for future
16k/32k/64k claims, not evidence that the current README table already covers
those contexts.
The renderer produces the human-review table from the same validated artifact,
including an explicit AX prefill bend marker when throughput drops below the
configured previous-context ratio threshold.
The campaign checker validates multiple per-model artifacts together, verifies
the required Gemma/Qwen/GLM family coverage when requested, rejects mixed-host
campaigns unless explicitly allowed, and renders the campaign summary table
from already validated per-model artifacts.

When the run also includes the optional `llama.cpp Metal` row, keep that row in
a separate long-context comparison artifact instead of merging it into the
prompt-hash-parity MLX scaling artifact:

```text
python3 scripts/build_long_context_comparison_artifact.py \
  benchmarks/results/mlx-inference/<date>/<model>.json \
  --output benchmarks/results/mlx-inference/<date>/<model>-long-context-comparison.json \
  --require-llama-cpp

python3 scripts/render_long_context_comparison_report.py \
  --require-llama-cpp \
  benchmarks/results/mlx-inference/<date>/<model>-long-context-comparison.json \
  --output benchmarks/results/mlx-inference/<date>/<model>-long-context-comparison.md
```

The resulting `ax.long_context_comparison.v1` artifact validates AX-vs-`mlx_lm`
prompt-hash parity and keeps `llama.cpp Metal` as an external
shape-compatible GGUF baseline. This is the right gate for cold long-prefill
comparison across AX, `mlx_lm`, and `llama.cpp`; decode-at-depth and
server-prefix reuse still require separate artifacts.

For decode cost after an existing context depth, build the separate
`ax.long_context_decode_at_depth.v1` artifact:

```text
python3 scripts/build_long_context_decode_at_depth_artifact.py \
  benchmarks/results/mlx-inference/<date>/<model>.json \
  --output benchmarks/results/mlx-inference/<date>/<model>-decode-at-depth.json

python3 scripts/render_long_context_decode_at_depth_report.py \
  benchmarks/results/mlx-inference/<date>/<model>-decode-at-depth.json \
  --output benchmarks/results/mlx-inference/<date>/<model>-decode-at-depth.md
```

Use `--require-llama-cpp` only for sources with explicit `llama-bench n_depth`
evidence. The existing shape-compatible `llama.cpp Metal` `pp`/`tg` rows remain
valid external context for cold prefill, but they are not depth-aware decode
evidence. Capture depth-aware rows with `bench_mlx_inference_stack.py
--llama-cpp-decode-at-depth`, which runs an additional
`llama-bench -p 0 -n <generation> -d <prompt>` pass for each prompt length.

The P2 cold-vs-warm startup artifact gate is also executable for saved startup
artifacts:

```text
python3 scripts/check_mlx_startup_latency_artifact.py \
  benchmarks/results/mlx-inference/<date>/<model>-startup-latency.json
```

To capture startup and concurrent-prefill artifacts from a real local model,
run:

```text
scripts/run-mlx-p2-latency-artifacts.sh \
  --model-dir /path/to/local/mlx-model \
  --output-root benchmarks/results/mlx-inference/<date> \
  --run-label <model>-p2-latency \
  --context-tokens 8192 \
  --startup-generation-tokens 128 \
  --concurrent-generation-tokens 1 \
  --concurrency-levels 1,2,4 \
  --host-label "Apple M5 Max"
```

The shell wrapper builds the release AX server, starts AX in direct MLX mode,
writes prompt-token artifacts, captures `startup-latency.json` and
`concurrent-prefill.json`, validates both, and writes `p2-latency.md` before
returning. Use `--dry-run` first to inspect output paths without building or
starting the server. For saved artifacts, the Markdown report can also be
regenerated directly:

```text
python3 scripts/render_mlx_p2_latency_report.py \
  --startup-artifact benchmarks/results/mlx-inference/<date>/<model>-p2-latency/startup-latency.json \
  --concurrent-artifact benchmarks/results/mlx-inference/<date>/<model>-p2-latency/concurrent-prefill.json \
  --output benchmarks/results/mlx-inference/<date>/<model>-p2-latency/p2-latency.md
```

Saved artifacts use schema `ax.mlx_startup_latency.v1`. The checker requires
the same benchmark prompt hash across `process_cold`, `model_warm`, and
`benchmark_warm` rows; direct AX policy labeling; route identity; at least
three repetitions; separated startup/load metrics for cold rows; no
startup/load metrics mixed into the benchmark-warm row; TTFT, decode, and peak
memory metrics; and fresh ratios back to the benchmark-warm row. This makes
cold-start claims visibly different from warm throughput claims. It does not
run the model by itself; real model artifacts still need a dedicated runner or
manual capture.

The P2 concurrent-prefill artifact gate is executable for saved concurrency
artifacts:

```text
python3 scripts/check_mlx_concurrent_prefill_artifact.py \
  benchmarks/results/mlx-inference/<date>/<model>-concurrent-prefill.json
```

Saved artifacts use schema `ax.mlx_concurrent_prefill.v1`. The checker requires
a single-request baseline plus at least one multi-request row, one prompt hash
per concurrent request, direct AX policy labeling, route identity, per-request
TTFT, total wall time, queue delay, zero failures, peak memory, overlap
classification, and fresh ratios back to the single-request baseline. This is
the guardrail for future claims about concurrent long-prompt serving. It is
separate from batch=1 throughput evidence and does not prove continuous
batching or queueing behavior until real server-path artifacts are captured.

These tests should remain split by evidence type. MLX inference-stack artifacts
own the AX-vs-reference throughput evidence. `ax-engine-bench`
scenario/replay/matrix artifacts own route, determinism, prefix reuse,
cancellation, and regression behavior.
End-to-end API latency should be labeled separately from raw runner throughput.

## Gemma Direct-Decode Follow-Ups

The current 2026-05-13 README artifact keeps Gemma 4 direct decode below
`mlx_lm` on every public Gemma row, while n-gram acceleration is strongly
positive:

| Model family | Direct AX vs `mlx_lm` | AX n-gram vs `mlx_lm` | Interpretation |
|---|---:|---:|---|
| Gemma 4 E2B 4/5/6/8-bit | -12.3% to -9.4% | +135.9% to +207.2% | Direct decode is still the weak path; user-default n-gram hides it for draftable outputs |
| Gemma 4 E4B 4-bit | -10.8% to -10.7% | +158.6% to +165.6% | Similar direct gap at the next small dense size |
| Gemma 4 26B A4B | -5.7% to -5.6% | +88.4% to +114.1% | MoE direct gap is smaller than E2B/E4B, but still below `mlx_lm` |
| Gemma 4 31B | -4.0% to -3.9% | +125.2% to +125.6% | Large dense Gemma is close enough that smaller overheads matter |

For the next Gemma E2B direct-decode optimization pass, prefer evidence that
preserves lazy direct decode. Blocking profile modes are useful for bottleneck
split diagnosis, but their token/s rows should not be compared directly against
normal pipelined decode rows. Prior profile artifacts pointed at post-attention
/ FFN and pre-SDPA tail work as higher-value follow-ups than per-layer-input
path slimming alone.

A follow-up Gemma 4 E2B 4-bit direct-decode refresh after the sliding-window
decode KV view change is recorded in:

```text
benchmarks/results/mlx-inference/2026-05-06/gemma-4-e2b-it-4bit-ax-engine-sliding-window-direct.json
```

It improves AX direct decode versus r4, but AX direct remains below `mlx_lm`.

An opt-in true rotating-backing-store experiment is recorded in:

```text
benchmarks/results/mlx-inference/2026-05-06/gemma-4-e2b-it-4bit-ax-engine-rotating-direct.json
```

It is not the default because it did not produce a consistent direct-decode win.

An opt-in `mlx_lm`-style direct decode cache-clear cadence experiment is
available with:

```text
AX_MLX_DIRECT_CLEAR_CACHE_CADENCE=256
```

The artifact is:

```text
benchmarks/results/mlx-inference/2026-05-06/gemma-4-e2b-it-4bit-ax-engine-clear-cache-direct.json
```

It is also not the default because it did not improve the retained-window
direct-decode result.

## Model Notes

Qwen Coder Next uses MLX affine 4-bit globally, with 8-bit overrides for router
and shared-expert gate tensors.

Qwen 3.6 35B A3B uses the MLX-community 4-bit checkpoint in the README GGUF
comparison set. Qwen 3.6 27B carries the 4/5/6/8-bit sweep coverage.

Gemma 4 26B A4B is the public Gemma 4 MoE MLX model. Its checkpoint uses affine
4-bit globally, with 8-bit overrides for dense MLP and router projections. The
`mlx_swift_lm` reference row loads this model via the `MLXVLM` factory, which is
required for MoE architectures, rather than the standard Swift path. Prompt
tokenization is otherwise identical to all other Swift rows.

For the remaining Gemma 4 26B A4B direct-decode gap, rerun AX direct rows with
`--ax-gemma4-moe-profile`. Those diagnostic rows record per-section MoE decode
counters under `ax_mlx_gemma4_moe_profile` and intentionally insert eval
barriers, so compare their bottleneck split rather than their headline tok/s.

Gemma 4 E2B 5/6/8-bit checkpoints use affine quantization at their respective
bit depths globally. These rows verify AX Engine's higher-bit quantization
support; the 4-bit row is the primary decode performance reference.

Gemma 4 E4B 4-bit benchmark rows are included in the current README refresh.
The model manifest and scenario manifest remain the canonical inputs for
rerunning that model.

## Method And Reproduction

Performance tables are results. Benchmark setup, reproduction commands,
community-run submission rules, workload-contract manifests, and
prompt-provenance requirements live in [`docs/BENCHMARKS.md`](BENCHMARKS.md).
