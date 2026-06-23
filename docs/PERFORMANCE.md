# Performance

This page is the public performance-results reference. It keeps the result
tables, artifact summaries, and interpretation for the current public snapshot.
The root `README.md` intentionally keeps only the common Gemma 4 and Qwen 3.6
rows.

For the public claim-boundary policy, see
[`performance/README.md`](performance/README.md).

For benchmark methodology, test setup, commands, reproduction details, and
evidence classification, see [`docs/BENCHMARKS.md`](BENCHMARKS.md).

## Current Result Set

The current README generation-model table is a provenance-tracked composite:
`mlx_lm` reference rows were refreshed on 2026-05-26, and the AX direct-only
overlay was refreshed on 2026-06-22 on:

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

The current README generation-model snapshot is backed by:

```text
benchmarks/results/mlx-inference/2026-05-26-direct-mode-clean-refresh/
benchmarks/results/mlx-inference/2026-06-22-ax-direct-readme-direct-only/
```

The reference directory supplies the 12 Gemma 4 and Qwen 3.6 `mlx_lm` rows in
the README table. The AX overlay directory reruns the matching direct AX rows
with generation=128, 5 measured repetitions, a 15-second
cooldown, AX prefix cache disabled for cold prefill and TTFT measurement, and
production-build server binaries. All MLX and AX rows use the same prompt-token
contract and prompt SHA checks. The `llama.cpp Metal*` column is injected from
the llama.cpp Metal manifest/artifact set documented in the README and remains
shape-compatible external context, not prompt-hash parity evidence.

`ax direct baseline` is the direct same-policy comparison against `mlx_lm`; the
benchmark starts the AX server with n-gram acceleration disabled for this row.
N-gram results remain documented as workload-dependent acceleration evidence,
but they are not part of the current README Gemma 4 / Qwen 3.6 direct-mode
snapshot. The README prefill table uses direct AX rows because n-gram
acceleration is a decode policy and should not be credited as a prefill
optimization.

## Latest Long-Context Artifacts

The public README table is intentionally short/mid-prompt evidence. The latest
checked-in heavy real-model validation is separate:

| Artifact | Model | Shape | Key result | Interpretation |
| --- | --- | --- | --- | --- |
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
| --- | --- | --- | --- |
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
| --- | --- | --- |
| Reference parity | Matching `mlx_lm.benchmark` seed-0 prompt/decode shapes, plus admitted `mlx_swift_lm` secondary rows where available | Same-shape comparisons against named MLX references |
| Prompt provenance | Prompt-token JSON artifacts with fixed token IDs, vocabulary size, seed, and hash | Reproducible prompt input across AX and reference rows |
| Decode policy separation | Direct AX rows run with n-gram acceleration disabled; default AX rows report n-gram effective throughput | Clear separation between same-policy decode and AX user-default acceleration |
| Prefill/decode split | Tables report prefill tok/s separately from decode tok/s | Prefill gains are not attributed to n-gram decode acceleration |
| Repeated measurement | 5 repetitions per engine/model/prompt row, reported through medians | Reduced single-run noise within the current host and prompt shape |
| Route identity | AX artifacts record runtime route and fixed-schema n-gram telemetry fields | No-draft fallback, direct pipeline, and n-gram rows can be distinguished |
| Output-quality caveat | Random-token greedy prompts can collapse into repeated output, but throughput artifact status stays limited to direct/n-gram fallback evidence | Public rows are performance evidence; coherent-output claims require separate token-output validation |
| Scope disclosure | Methodology states host, batch size, generated-token count, temperature, and `prefill_step_size` | Readers can see that current public rows are batch=1, short/mid-prompt evidence |

This review intentionally rejects broader conclusions that the current table
does not prove. The current README rows support row-by-row claims about AX
direct throughput against named MLX references on matching Apple Silicon
benchmark shapes; they do not say every AX policy wins every row. N-gram rows
are separate workload-dependent acceleration evidence. Neither direct nor
n-gram rows, by themselves, support claims about 32k/128k context scaling,
cold-start latency, multi-user serving throughput, KV eviction behavior, or
parity with CUDA server systems such as vLLM or TensorRT-LLM.

## Interpretation

The current composite result set is not a universal direct-decode win.
Direct AX is intentionally measured with n-gram acceleration disabled, and the
direct column spans -14.4% to +22.9% versus `mlx_lm` across the full decode
artifact set. In the README Gemma 4 / Qwen 3.6 snapshot, direct decode spans
-14.4% to +9.1%. Those rows are the same-policy baseline rather than the
default AX user path.

N-gram acceleration is the default AX user/server path and is the better
headline row for decode-throughput expectations when the workload produces
draftable local repetition. It is intentionally separate from the current
README direct-mode snapshot. Historical n-gram artifacts still report
workload-dependent effective throughput and explicit fallback status, and
public docs should keep direct baseline rows visible whenever an n-gram row is
shown.

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
JSON artifacts include fixed-schema n-gram telemetry fields, and current public
tables should state whether a row is direct, n-gram, or MTP before comparing
throughput.

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

The current MTP publication contract has two labeled lanes. The recommended
AX Engine practice lane is the 6-bit local-agent matrix, and those rows must
start from `ax-engine download-mtp` output. A 4-bit comparison lane may be
published when it aligns with peer MTP-engine benchmark results. Promoted
runtime rows run MTP; same-package direct rows may be reported only as
denominators for AX MTP acceleration charts.

| Target | Required preparation | Promoted MTP mode |
| --- | --- | --- |
| `qwen3.6-27b-6bit` | `ax-engine download-mtp qwen3.6-27b-6bit` | Qwen fused sidecar MTP |
| `qwen3.6-35b-a3b` | `ax-engine download-mtp qwen3.6-35b-a3b` | Qwen fused sidecar MTP |
| `gemma-4-12b` | `ax-engine download-mtp gemma-4-12b` | Gemma assistant-MTP |
| `gemma-4-26b` | `ax-engine download-mtp gemma-4-26b` | Gemma assistant-MTP |
| `gemma-4-31b` | `ax-engine download-mtp gemma-4-31b` | Gemma assistant-MTP |
| `glm-4.7-flash` | `ax-engine download-mtp glm-4.7-flash` | GLM built-in MTP sidecar |

Rules:

- Quantization is 6-bit for the recommended practical lane.
- 4-bit rows are comparison evidence only; use 6-bit packages for practical
  AX Engine deployments.
- `mtp-ngram` is out of scope for MTP publication and must not be run in the
  MTP matrix.
- Historical MTP+n-gram artifacts remain diagnostic only. Do not use them for
  current README/PERFORMANCE claims.
- Direct rows may be reported only as same-artifact denominators for
  `AX MTP / AX direct` acceleration, not as a cross-model speed leaderboard.
- Recommended 6-bit artifacts should live under
  `benchmarks/results/mtp-6bit/<run-dir>/`; 4-bit comparison artifacts must stay
  clearly labeled. Every artifact records the exact `download-mtp` output path
  where applicable, model snapshot, sidecar or assistant package provenance,
  route identity, sampler, prompt suite, repetitions, and cooldown.

Prepare the matrix:

```bash
ax-engine download-mtp qwen3.6-27b-6bit
ax-engine download-mtp qwen3.6-35b-a3b
ax-engine download-mtp gemma-4-12b
ax-engine download-mtp gemma-4-26b
ax-engine download-mtp gemma-4-31b
ax-engine download-mtp glm-4.7-flash
```

Benchmark rows use the existing MTP prompt suites
`flappy`, `long_code`, and `python_modules_long`, sampled decode with
temperature `0.6`, top-p `0.95`, top-k `20`, `1000` generated tokens,
`5` measured repetitions, and cooldowns recorded in the artifact.

### Draft confidence gate (throughput knob)

The MTP draft confidence gate (`AX_MLX_MTP_DRAFT_MIN_CONFIDENCE`, default 0.90)
is a throughput lever, not just an accept-rate guard: a looser gate proposes
slightly longer drafts that are still almost always accepted. Measured +8-18%
tokens/forward on Qwen3.6 27B (looser is better on harder workloads). Per-workload
best-practice values and the full sweep are in
[`docs/mtp/draft-gate-throughput.md`](mtp/draft-gate-throughput.md). Tree
speculative decoding is not an option on the linear-attention models — see
[`docs/mtp/tree-draft-phase-a.md`](mtp/tree-draft-phase-a.md).

## Additional Testing Plan

More testing is needed before making production-serving or long-context claims.
The recommended sequence is:

| Priority | Test | Shape | Acceptance evidence |
| --- | --- | --- | --- |
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
| --- | ---: | ---: | --- |
| Gemma 4 E2B 4/5/6-bit | -12.3% to -9.4% | +135.9% to +207.2% | Direct decode is still the weak path; user-default n-gram hides it for draftable outputs |
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
comparison set. Qwen 3.6 27B carries the 4/5/6-bit sweep coverage.

Gemma 4 26B A4B is the public Gemma 4 MoE MLX model. Its checkpoint uses affine
4-bit globally, with 8-bit overrides for dense MLP and router projections. The
`mlx_swift_lm` reference row loads this model via the `MLXVLM` factory, which is
required for MoE architectures, rather than the standard Swift path. Prompt
tokenization is otherwise identical to all other Swift rows.

For the remaining Gemma 4 26B A4B direct-decode gap, rerun AX direct rows with
`--ax-gemma4-moe-profile`. Those diagnostic rows record per-section MoE decode
counters under `ax_mlx_gemma4_moe_profile` and intentionally insert eval
barriers, so compare their bottleneck split rather than their headline tok/s.

Gemma 4 E2B 5/6-bit checkpoints use affine quantization at their respective
bit depths globally. These rows verify AX Engine's higher-bit quantization
support; the 4-bit row is the primary decode performance reference.

Gemma 4 E4B 4-bit benchmark rows are included in the current README refresh.
The model manifest and scenario manifest remain the canonical inputs for
rerunning that model.

## Method And Reproduction

Performance tables are results. Benchmark setup, reproduction commands,
community-run submission rules, workload-contract manifests, and
prompt-provenance requirements live in [`docs/BENCHMARKS.md`](BENCHMARKS.md).
