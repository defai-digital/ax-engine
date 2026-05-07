# Benchmarks

AX Engine keeps benchmarking deliberately split by evidence type. A result is
useful only when the workload, runtime route, reference engine, host, model,
sampling policy, and artifact schema are explicit.

Measured results for each tested model are summarized in the root `README.md`
under the **Performance** section. Methodology and interpretation for the
current public table live in `docs/PERFORMANCE.md`. Public review artifacts live under
`benchmarks/results/mlx-inference/<date>/`; for example, the 2026-05-04 result
set includes the full JSON output, prompt-token JSON files, and command logs for
Gemma 4 E2B 4/5/6/8-bit, Gemma 4 26B A4B, Gemma 4 31B,
Qwen 3.5 9B, Qwen 3.6 35B A3B 4/5/6/8-bit, and Qwen Coder Next.
The 2026-05-06 result set records `mlx-community/GLM-4.7-Flash-4bit` with
repo-owned AX runtime rows and `mlx-community/DeepSeek-V4-Flash-2bit-DQ` as a
reference-only fail-closed check.

## Which Tool To Use

| Question | Use | Evidence produced |
|---|---|---|
| How fast is the repo-owned MLX runtime against upstream MLX? | `scripts/bench_mlx_inference_stack.py` | Required `mlx_lm.benchmark` primary baseline rows, AX direct and n-gram acceleration rows, optional `mlx-swift-lm` secondary baseline adapter rows, canonical prompt-token artifacts, and ratio-to-baseline fields |
| Did a checked-in workload still pass route, correctness, determinism, replay, or regression gates? | `ax-engine-bench` | Workload-contract artifacts under `benchmarks/results` |
| Is the local host ready for repo-owned MLX benchmarking? | `ax-engine-bench doctor` | Human or JSON readiness report |
| Did a bounded runtime knob improve a frozen workload? | `ax-engine-bench autotune` | Autotune trial artifacts and warm-start history |
| Does the non-MLX delegated route still behave correctly? | llama.cpp manifests through `ax-engine-bench` | Delegated route-contract evidence only |
| Does upstream `mlx-lm` delegated text compatibility still behave correctly? | Explicit `mlx_lm_delegated` checks through SDK/server/CLI surfaces | Delegated route-contract evidence only |
| Which AX runtime path is best for a product endpoint on this host? | `scripts/bench_ax_engine_three_modes.py` against already-running AX servers | End-to-end AX API latency by mode; not raw model throughput |

Do not merge these rows into one unlabeled throughput table. Repo-owned
model-inference claims come from the MLX inference stack, and every such claim
must include a matching `mlx_lm.benchmark` baseline for the same random-token
prompt/decode shape, with prompt-token provenance recorded in the artifact.
`ax-engine-bench` owns workload contracts. llama.cpp manifests and
`mlx_lm_delegated` checks validate delegation behavior, not repo-owned MLX runtime
speed.

## AX Runtime-Mode Comparison

Use the three-mode harness when the question is product-path latency through AX
surfaces, for example comparing repo-owned MLX default, repo-owned MLX with
n-gram acceleration disabled, and `mlx_lm_delegated`.

Start each server mode separately, then run:

```text
python3 scripts/bench_ax_engine_three_modes.py \
  --model-id qwen3_dense \
  --prompt "Reply with just: ready." \
  --tokens-file /path/to/chat-template-token-ids.json \
  --max-output-tokens 32 \
  --mode preview_default,http://127.0.0.1:19091,tokens \
  --mode preview_direct,http://127.0.0.1:19092,tokens \
  --mode mlx_lm_delegated,http://127.0.0.1:19093,text \
  --output benchmarks/results/runtime-modes/$(date +%F)-qwen3-4b.json
```

This harness records warmup/measured observations, median wall time, p25/p75,
HTTP status, finish reason, and output token count. Its output is suitable for
manager/product latency decisions, but it must be labeled as
`end_to_end_ax_generate_api_latency`. Keep it separate from
`bench_mlx_inference_stack.py`, which is the repo-owned MLX throughput evidence
path.

## MLX Model-Inference Comparison

Use the MLX inference-stack harness when the question is performance versus an
MLX-family reference:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 512,2048 \
  --generation-tokens 128 \
  --repetitions 5 \
  --cooldown 5
```

The reference contract is:

1. `mlx_lm.benchmark` is mandatory and is the canonical upstream Python MLX
   baseline.
2. The harness mirrors the `mlx_lm.benchmark` prompt standard:
   `mx.random.seed(0)` followed by
   `mx.random.randint(0, vocab_size, (1, prompt_tokens))`.
3. The generated prompt token IDs are written as JSON artifacts and reused by
   AX Engine and any admitted `mlx-swift-lm` adapter.
4. Repo-owned MLX runtime is measured through `ax-engine-server` SSE
   `runner_time_us`.
5. `mlx-swift-lm` is optional and only admitted as a secondary reference
   through an explicit JSON-emitting adapter command built on the
   `mlx-swift-lm` `BenchmarkHelpers` / `MLXLMCommon` generation APIs.

The harness fails closed if `mlx_lm.benchmark` cannot run. It does not support
AX-only throughput tables. Every non-baseline row records the matching
`mlx_lm.benchmark` random-token prompt/decode shape plus prefill/decode ratios.
The default AX row is the direct same-policy comparison against the primary MLX
baseline. AX n-gram acceleration rows are effective-throughput evidence and
must not be treated as the same decode policy as the primary MLX baseline.

Long-context prefill and TTFT claims require a separate scaling artifact. For a
fresh run, use the repo-owned wrapper:

```text
scripts/run-mlx-prefill-scaling-artifact.sh \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 1024,2048,4096,8192,16384 \
  --run-label qwen35-prefill-scaling
```

Saved artifacts use schema `ax.mlx_prefill_scaling.v1`. Build and validate
them from an already completed MLX inference-stack run with:

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

New AX rows from `bench_mlx_inference_stack.py` include runner-derived
`ttft_ms` and server-process RSS memory. The builder derives `mlx_lm` TTFT from
reported prefill throughput because the upstream benchmark does not emit a
separate first-token event in this artifact shape. The checker requires a
`mlx_lm` primary reference row and a direct `ax_engine_mlx` row for each
context/generation shape, shared prompt hashes,
`direct_no_ngram_acceleration` labeling on the AX row, median prefill tok/s,
median TTFT, peak memory, and ratios back to the matching `mlx_lm` row. Keep
these artifacts separate from n-gram decode-throughput rows so prefill wins are
not accidentally credited to a decode acceleration policy.
The renderer is intentionally downstream of the validator, so Markdown reports
cannot become the source of truth for unvalidated long-context claims.
Use the campaign checker when the claim is representative model-family coverage
rather than a single-model result. By default it can require Gemma/Qwen/GLM
coverage and one host identity, so mixed-host evidence must be labeled
explicitly with `--allow-mixed-host`.

The latest checked-in real-model P1 example is:

- [Qwen3-4B-4bit prefill scaling, 2026-05-07](../benchmarks/results/mlx-inference/2026-05-07-real-p1/qwen3-4b-4bit-prefill-scaling/prefill-scaling.md)

This example is intentionally expectation-management evidence: AX beats
`mlx_lm` at 1k in that run, is near parity at 2k, and is below `mlx_lm` at
4k/8k. Do not cite it as a broad AX prefill win.

Cold-start and warm-throughput claims require a separate startup-latency
artifact:

```text
python3 scripts/check_mlx_startup_latency_artifact.py \
  benchmarks/results/mlx-inference/<date>/<model>-startup-latency.json
```

The repo-owned real-model runner can capture both P2 artifacts:

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

It builds the release AX server, starts AX in direct MLX mode, writes
deterministic prompt-token artifacts, captures `startup-latency.json` and
`concurrent-prefill.json`, validates both outputs, and writes `p2-latency.md`
before returning. Use `--dry-run` first when preparing a long run. To
regenerate the report from saved artifacts:

```text
python3 scripts/render_mlx_p2_latency_report.py \
  --startup-artifact benchmarks/results/mlx-inference/<date>/<model>-p2-latency/startup-latency.json \
  --concurrent-artifact benchmarks/results/mlx-inference/<date>/<model>-p2-latency/concurrent-prefill.json \
  --output benchmarks/results/mlx-inference/<date>/<model>-p2-latency/p2-latency.md
```

Saved artifacts use schema `ax.mlx_startup_latency.v1`. The checker requires
`process_cold`, `model_warm`, and `benchmark_warm` rows for the same model,
prompt hash, context, and generation shape. Cold rows carry server-ready,
model-load, first-request TTFT, TTFT, decode, and peak-memory metrics; the warm
row carries warm TTFT, warmed decode, and peak memory without startup/load
metrics mixed in. Ratios are checked against the benchmark-warm row so stale
cold/warm comparisons fail closed. Keep this evidence separate from the MLX
throughput table: a warm benchmark win is not a cold-start win.

Concurrent prefill claims require a separate concurrency artifact:

```text
python3 scripts/check_mlx_concurrent_prefill_artifact.py \
  benchmarks/results/mlx-inference/<date>/<model>-concurrent-prefill.json
```

Saved artifacts use schema `ax.mlx_concurrent_prefill.v1`. The checker requires
a single-request baseline plus at least one multi-request row, one prompt hash
per concurrent request, direct AX policy labeling, route identity, per-request
TTFT, total wall time, queue delay, zero failures, peak memory, overlap
classification, and ratios back to the single-request baseline. Keep this
evidence separate from batch=1 throughput rows: concurrent long-prompt serving
needs server-path queueing and memory-pressure evidence, not only raw runner
throughput.

The latest checked-in real-model P2 example is:

- [Qwen3-4B-4bit startup and concurrent prefill, 2026-05-07](../benchmarks/results/mlx-inference/2026-05-07-real-p2/qwen3-4b-4bit-p2-latency/p2-latency.md)

This report records the 8k cold/model-warm/benchmark-warm split and
concurrency 1/2/4 behavior. The 4-request row is classified as serialized, so
it should be used as a boundary on concurrent-prefill claims rather than a
positive continuous-batching claim.

Use `--ax-compare-policies` when n-gram acceleration is part of the question:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 256,512,2048 \
  --generation-tokens 128 \
  --ax-compare-policies
```

The harness labels AX rows as `ax_engine_mlx` and
`ax_engine_mlx_ngram_accel`. Direct throughput and n-gram acceleration
throughput must not collapse into one AX number. AX rows also record
`ax_decode_policy`: `direct_no_ngram_acceleration` for the direct comparison
row, `ngram_acceleration_kv_trim` for dense/full-attention n-gram acceleration
rows, and `ngram_acceleration_linear_attention_branch_recompute` for Qwen3.5-style
recurrent linear-attention rows, where repeated n-gram evidence is required
before probing and partial accepts trigger a longer cooldown.

For Qwen3.5/Qwen3-Next GatedDelta prefill profiling, use the evidence-first
profile mode:

```text
scripts/run-gateddelta-prefill-profile.sh \
  --model-dir /path/to/qwen-linear-attention-mlx-model \
  --run-label qwen-gateddelta-profile
```

This mode requires a linear-attention MLX manifest, forces the direct AX row,
and records the 512, 2048, 8192, and 32768 prompt-token matrix in the artifact
under `gateddelta_prefill_profile`. It also starts AX with
`AX_MLX_LINEAR_ATTENTION_PROFILE=1`, which inserts diagnostic timing barriers
and emits projection/conv/QK-normalization/recurrent/output stage counters in
`ax_mlx_linear_attention_profile`. Treat these rows as prefill slope evidence
before GatedDelta scan or fusion kernel changes. Do not infer n-gram, learned
draft, tree-speculation, or KV-compression claims from this profile.

The wrapper preflights `config.json` and `model-manifest.json` before building
the release server. It fails closed unless the manifest is `qwen3_5` or
`qwen3_next`, has enabled `linear_attention`, and includes the gated-delta kernel
dimensions required by the runtime. Valid profile artifacts also record that
normalized `ax.gateddelta_prefill_model_preflight.v1` result under
`gateddelta_prefill_profile.model_preflight`, so the JSON evidence remains
auditable after logs are gone.

Validate saved GatedDelta profile artifacts with:

```text
python3 scripts/check_gateddelta_prefill_profile_artifact.py \
  benchmarks/results/mlx-inference/<date>/gateddelta-prefill-profile.json
```

When `--gateddelta-prefill-profile` is combined with `--output`, the benchmark
harness runs this validator immediately after writing the JSON artifact.

The wrapper above also renders the Markdown report during capture. The lower
level harness command is:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/qwen-linear-attention-mlx-model \
  --gateddelta-prefill-profile \
  --generation-tokens 128 \
  --output benchmarks/results/mlx-inference/<date>/gateddelta-prefill-profile.json \
  --gateddelta-prefill-profile-report-output \
    benchmarks/results/mlx-inference/<date>/gateddelta-prefill-profile.md
```

If the report was not rendered during capture, render the validated profile with:

```text
python3 scripts/render_gateddelta_prefill_profile_report.py \
  benchmarks/results/mlx-inference/<date>/gateddelta-prefill-profile.json \
  --output benchmarks/results/mlx-inference/<date>/gateddelta-prefill-profile.md
```

For Gemma 4 26B A4B direct-decode investigation, enable the opt-in MoE profile:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/gemma-4-26b-a4b-it-4bit \
  --prompt-tokens 128,512 \
  --generation-tokens 128 \
  --ax-direct \
  --ax-gemma4-moe-profile
```

This sets `AX_MLX_GEMMA4_MOE_PROFILE=1` for the AX server and records
`ax_mlx_gemma4_moe_profile` counters in AX trial and summary rows. The profile
inserts eval barriers around Gemma4 MoE decode-layer attention, dense, router,
expert, and post-combine sections, so use it to localize bottlenecks, not as
headline throughput evidence.

For internal TurboQuant evidence capture, the harness can pass through the
server's experimental shadow policy:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 8192 \
  --generation-tokens 256 \
  --experimental-mlx-kv-compression turboquant-shadow
```

Use `--experimental-mlx-kv-compression turboquant-fused-experimental` only for
runner-route selection evidence. It requests compressed decode and can report
`fused_compressed_decode` when the eligible K8/V4 single-token path uses the
two-stage Metal cold decode plus full-precision hot tail. If Metal is
unavailable but the reference fallback works, the path is
`cpu_oracle_compressed_decode`. Fallback reason label `runner_not_integrated`
means no runtime decode attempt was observed yet; `cpu_oracle_unavailable`
means both compressed-decode attempts fell back to full-precision generation.
The TurboQuant reference codec and microbench artifacts include a split-softmax
oracle for the hot-window merge: cold and hot partitions must be combined
through shared log-sum-exp normalization, not by adding independently normalized
output vectors.

The default remains disabled. `turboquant-shadow` keeps generation on the
full-precision MLX decode path and records route counters for eligibility,
estimated saved KiB, runtime shadow-storage writes, shadow-storage sync calls
and wall time, current compression decode path, fused decode candidate
snapshots, attempt/success/fallback counters, and remaining production blockers
when the runtime emits them. It is artifact evidence only; promoted TurboQuant
support still requires a long-context quality artifact validated by
`scripts/check_turboquant_quality_artifact.py`.

The quality artifact gate is stricter than telemetry collection. Quality/path
evidence must use candidate mode `turboquant-fused-experimental`, route
metadata schema `>= 2`, K8/V4, fused_compressed_decode path code `2`, fused
decode attempts and successes greater than zero, and zero fused decode
fallbacks. Shadow rows and cpu_oracle_compressed_decode rows are useful for
diagnosis, but they are rejected as quality evidence. Public-support promotion
also requires the readiness report's decode-throughput performance gate to
pass; a quality artifact can pass while public docs remain experimental.
For model-level quality evidence, run the baseline and fused candidate AX rows
with `bench_mlx_inference_stack.py --capture-output-token-ids`, then extract
same-shaped output vectors with `scripts/build_turboquant_decode_outputs.py`
before building the quality metrics JSON.
The repo-owned wrapper is `scripts/run-turboquant-quality-artifact.sh`; it runs
the baseline, fused candidate, decode-output extraction, quality metrics,
artifact validation, and promotion-readiness report in one fail-closed bundle.
Run it with `--dry-run` first to inspect inferred model metadata and planned
commands without loading the model.

Before changing public support wording, run
`scripts/check_turboquant_promotion_readiness.py`. Public docs must remain
experimental while that report has blockers, for example when the available
model manifests cannot exercise the current `head_dim=128`, `head_dim=256`, or
`head_dim=512` fused K8/V4 gate or no passing long-context fused-path quality
artifact exists.

The internal quantitative benchmark design lives in
`.internal/benchmark/TURBOQUANT-BENCHMARK-DESIGN.md`. It separates microkernel
timing, optional shadow-storage overhead, integrated fused compressed decode,
and long-context quality gates so TurboQuant results can report decode ratio,
prefill ratio, KV saved percent, runtime storage coverage, fallback rate, and
quality pass/fail without mixing evidence types.

For fused-kernel-only evidence, use the `ax-engine-mlx` microbenchmark:

```text
cargo run -p ax-engine-mlx --release --bin turboquant-microbench -- \
  --cold-tokens 512,2048,8192 \
  --hot-tokens 128 \
  --variants dim_parallel,two_stage_scores \
  --repetitions 5 \
  --output benchmarks/results/turboquant/<date>/microbench.json
```

The output schema is `ax.turboquant_fused_decode_microbench.v1`. It compares
the K8/V4 MLX/Metal fused cold-decode kernel with the CPU compressed reference
oracle. This is kernel evidence only; it does not mean the server generation
path is using fused compressed decode. When `--hot-tokens` is positive, the
artifact also records `hot_tail_merge` quality for the shared log-sum-exp merge
contract between compressed cold partition stats and full-precision hot-tail
stats.

Validate saved fused-kernel evidence without rerunning Metal:

```text
python3 scripts/check_turboquant_microbench_artifact.py \
  benchmarks/results/turboquant/<date>/microbench.json
```

N-gram acceleration AX result objects also include `ngram_acceleration_telemetry` when the
runtime emits route counters. The stored counters cover draft attempts, draft
tokens, accepted/rejected draft tokens, full accepts, partial rejects, complete
misses, no-draft steps, cooldown steps, cooldown events, cooldown steps
scheduled, and an `ax_ngram_accept_rate_micros` derived from accepted/draft
tokens. Use these counters to audit whether an acceleration row came from
real n-gram acceptance or from timing noise.

The `mlx-swift-lm` reference checkout currently exposes `BenchmarkHelpers` for
loading, tokenization, decoding, download-cache timing, and shared integration
benchmark scaffolding, but not a repo-stable standalone LLM inference
benchmark CLI equivalent to `python3 -m mlx_lm.benchmark`. For AX Engine
inference comparisons, treat `mlx-swift-lm` as a secondary baseline only when
the adapter:

- is built against the reference `mlx-swift-lm` package, not a custom
  application server,
- uses `MLXLMCommon` generation APIs with `GenerateParameters(maxTokens: ...,
  temperature: 0, prefillStepSize: ...)`,
- reads the harness-emitted `{prompt_token_ids_path}`,
- reports prefill and decode throughput for the same random-token prompt/decode
  shape, and
- emits raw trial rows when possible so the harness can compute shared
  mean/median/min/max summaries.

The older SwiftLM application-server benchmark is retired for current AX Engine
decisions. It measures application/server behavior around MLX, not the
`mlx-swift-lm` library benchmark contract. Keep it as historical design input
only.

## AX Workload Contracts

`ax-engine-bench` is the repo-owned CLI for checked-in workload artifacts:

```text
ax-engine-bench scenario --manifest benchmarks/manifests/scenario/chat_qwen_short.json --output-root benchmarks/results
ax-engine-bench replay --manifest benchmarks/manifests/replay/shared_prefix_long_churn.json --output-root benchmarks/results
ax-engine-bench matrix --manifest benchmarks/manifests/matrix/mlx_dense_phase7.json --output-root benchmarks/results
ax-engine-bench compare --baseline benchmarks/results/<baseline> --candidate benchmarks/results/<candidate> --output-root benchmarks/results
ax-engine-bench matrix-compare --baseline benchmarks/results/<baseline-matrix> --candidate benchmarks/results/<candidate-matrix> --output-root benchmarks/results
ax-engine-bench baseline --source benchmarks/results/<run> --name "Dense Qwen Trusted" --output-root benchmarks/baselines
```

Canonical manifests live under:

- `benchmarks/manifests/scenario/`
- `benchmarks/manifests/replay/`
- `benchmarks/manifests/matrix/`

Successful execution emits structured artifacts such as:

- `manifest.json`
- `environment.json`
- `metrics.json`
- `routes.json`
- `trace.json`
- `summary.md`

Compare and matrix commands emit regression and roll-up artifacts. Baseline
commands copy a successful result into a named trusted baseline and fail closed
instead of overwriting an existing baseline.

Use this surface for:

- correctness and determinism gates
- route identity and support-tier reporting
- prefix-reuse and prompt-cache provenance
- replay and churn validation
- matrix roll-ups
- matrix-to-matrix regression review
- bounded autotune trials over frozen manifests

## Bounded Autotune

`ax-engine-bench autotune` is part of the workload-contract surface, not a broad
architecture search system:

```text
ax-engine-bench autotune \
  --manifest benchmarks/manifests/scenario/chat_qwen_short.json \
  --output-root benchmarks/results \
  --iterations 8 \
  --max-batch-token-options 2048,4096,8192 \
  --kv-total-block-options 256,512 \
  --prefix-cache-options true,false
```

Autotune may explore bounded knobs already represented in manifests. It must not
turn runtime-family selection, backend ownership, or scheduler architecture into
implicit search dimensions. Treat the output as candidate evidence that still
needs the normal scenario/replay/compare gates.

## Delegated llama.cpp Checks

The repo carries delegated non-MLX examples:

- `benchmarks/manifests/scenario/llama_cpp_chat_qwen_short.json`
- `benchmarks/manifests/scenario/llama_cpp_shared_prefix_qwen_short.json`
- `benchmarks/manifests/replay/llama_cpp_submit_cancel_dual.json`
- `benchmarks/manifests/replay/llama_cpp_prompt_cache_reuse_dual.json`

These validate the SDK-owned llama.cpp adapter contract, submit/cancel behavior,
and backend-reported prompt-cache evidence. They must not be described as
repo-owned model-inference throughput benchmarks.

When running them directly, update `runtime.backend_adapter.server_url` for the
local llama.cpp server. The repo smoke path is:

```text
bash scripts/check-bench-preview.sh
```

## Delegated mlx-lm Checks

`mlx_lm_delegated` checks validate AX surface compatibility with a
user-provided `mlx_lm.server` for unsupported MLX text models. They must record
`selected_backend=mlx_lm_delegated` and `support_tier=mlx_lm_delegated`, and
they must not be described as repo-owned MLX throughput evidence.

Reference-only `mlx_lm.benchmark` checks are even narrower: they prove the
upstream Python reference can load and benchmark a downloaded model artifact,
but they do not prove AX route compatibility unless an AX server check is also
run. The 2026-05-06 DeepSeek community check is in this category. GLM has moved
out of this bucket because it now has a runtime-ready manifest, GLM graph
coverage, a server smoke, and AX benchmark rows. A partial reference, such as
the current DeepSeek V4 SwiftLM port that drops compressor/indexer and
hash-routing tensors, must stay fail-closed.

The delegated route supports text generation through upstream
`mlx_lm.server`, including AX SSE surfaces that forward upstream text deltas.
Those streams validate route compatibility only; they do not create repo-owned
MLX throughput evidence or AX-owned token/KV accounting. Token-array prompts,
stepwise lifecycle calls, and visual/multimodal contracts are intentionally
unsupported until a separate route and artifact contract exists.

## Readiness

Decision-grade repo-owned MLX benchmark claims require:

- Apple Silicon M4-or-newer macOS/aarch64 host.
- MLX runtime availability.
- Stable model, tokenizer, prompt shape, and sampling parameters.
- Prompt-token provenance matching the `mlx_lm.benchmark` random-token
  standard, including the saved prompt JSON path and SHA-256 hash.
- Explicit runtime identity: selected backend, support tier, resolution policy,
  feature flags, and AX decode mode.
- Enough repetitions and cooldown to avoid one-off thermal or startup noise.
- Raw JSON artifacts preserved next to any summarized table.

Inspect local readiness with:

```text
ax-engine-bench doctor --json
```

Delegated `mlx_lm_delegated` and llama.cpp adapters do not widen repo-owned
MLX host support. They are separate delegated runtime checks.

## Reproducible Community Runs

Use the reproduction wrapper when an external reviewer wants to rerun the public
MLX inference-stack procedure on an Apple Silicon host:

```text
scripts/reproduce-mlx-inference-benchmark.sh \
  --model-dir /path/to/mlx-model \
  --run-label qwen3-5-9b-m4-max
```

The wrapper keeps the existing benchmark contract intact. It runs
`ax-engine-bench doctor`, builds the release server binary, invokes
`scripts/bench_mlx_inference_stack.py`, and writes a bundle containing raw JSON,
prompt-token artifacts, doctor output, command logs, and host metadata under
`benchmarks/community-results/local/` by default.

Use `--direct-only` when the comparison should include only the direct
same-policy AX row. By default, the wrapper uses `--ax-compare-policies` so the
bundle includes both direct AX and n-gram effective-throughput rows, each with
its own decode-policy label.

Community runs reproduce the procedure, not identical numbers. Only compare
rows after checking model artifact identity, prompt-token hashes, generated
token count, repetitions, reference runtime, AX decode policy, host class, and
thermal context. See `benchmarks/community-results/README.md` for submission
rules.

## Smoke Checks

Use these repo-owned checks to validate the benchmark surfaces:

```text
python3 scripts/check_readme_performance_artifacts.py
bash scripts/check-bench-inference-stack.sh
bash scripts/check-bench-doctor.sh
bash scripts/check-bench-mlx.sh
bash scripts/check-bench-replay.sh
bash scripts/check-bench-preview.sh
bash scripts/check-bench-matrix.sh
bash scripts/check-bench-matrix-compare.sh
```

## Interpretation Rules

| Evidence | Can support | Cannot support |
|---|---|---|
| `bench_mlx_inference_stack.py` rows with matching `mlx_lm.benchmark` baseline | Repo-owned MLX model-inference performance claims against named MLX references | Scheduler/replay correctness claims |
| `ax-engine-bench scenario` / `replay` MLX artifacts | Workload-contract, route, correctness, determinism, replay, and regression claims | Direct upstream MLX comparison unless the MLX stack harness was also run |
| `ax-engine-bench autotune` artifacts | Candidate evidence for bounded manifest knobs | Architecture selection or cross-runtime ranking |
| `mlx_lm_delegated` artifacts | Upstream mlx-lm text compatibility through AX surfaces | Repo-owned MLX throughput claims or visual/multimodal support |
| llama.cpp delegated artifacts | Non-MLX route-contract and backend prompt-cache claims | Repo-owned MLX throughput claims |
| `ax-engine-bench compare` / `matrix-compare` | Regression evidence inside the same manifest/runtime family | Cross-runtime ranking without matching reference contract |

If the reference identity or runtime route is unclear, treat the number as
exploratory only.
