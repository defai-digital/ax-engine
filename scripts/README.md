# Scripts

This directory contains repo-owned smoke checks and a small number of diagnostic
helpers.

## Benchmarking Rule

Use `bench_mlx_inference_stack.py` for AX Engine MLX model-inference
comparisons. That script always runs `mlx_lm.benchmark` as the primary baseline
and fails the run if the baseline cannot be produced. The harness mirrors the
`mlx_lm.benchmark` prompt standard by generating `mx.random.seed(0)` random
token IDs from the model vocabulary, writes those token IDs to JSON artifacts,
and reuses them for AX Engine and any admitted `mlx-swift-lm` adapter.

Optional `mlx-swift-lm` rows are secondary-baseline rows only when the command
is built on the reference package's `BenchmarkHelpers` / `MLXLMCommon`
generation APIs, reads the harness-emitted prompt token JSON, and reports
prefill/decode throughput for the same random-token prompt/decode shape. Do
not use an application-server wrapper or unrelated Swift timing script as the
`mlx-swift-lm` baseline.

Use `ax-engine-bench` through the workload-contract `check-bench-*.sh` scripts
for scenario, replay, matrix, baseline, compare, delegated llama.cpp route
checks, and readiness. `check-bench-inference-stack.sh` is the exception in
that family: it validates the MLX inference-stack harness contract without
running `ax-engine-bench` or loading a model.

Do not use ad hoc server timing or llama.cpp route checks as AX-owned MLX
throughput baselines.

## Script Groups

- `lib/common.sh`: shared shell helpers for repo-root discovery, Python binary
  selection, temporary paths, free-port allocation, and PID cleanup.
- `check-scripts.sh`: fast script hygiene gate. It syntax-checks shell scripts,
  compiles Python scripts, and runs the MLX inference-stack contract tests.
- `bench_mlx_inference_stack.py`: MLX model-inference comparison against
  `mlx_lm.benchmark`. It can pass through
  `--experimental-mlx-kv-compression turboquant-shadow` or
  `turboquant-fused-experimental` for AX rows and records TurboQuant KV
  compression route counters, including shadow-storage sync calls and wall time
  plus fused decode candidate/attempt/success/fallback counters and fallback
  reason labels, when the runtime emits them.
- `test_bench_mlx_inference_stack.py`: unit tests for the MLX benchmark
  contract, parser, prompt artifact hash checks, and secondary adapter shape.
- `check_gateddelta_prefill_profile_artifact.py`: fail-closed validator for
  `--gateddelta-prefill-profile` artifacts. It requires Qwen-style
  linear-attention metadata, the 512/2048/8192/32768 prompt matrix, direct AX
  rows, `mlx_lm` primary-reference rows, no n-gram/KV-compression evidence, and
  opt-in `ax_mlx_linear_attention_profile` stage counters. The artifact must
  also carry versioned `gateddelta_prefill_profile.model_preflight` evidence
  from `check_gateddelta_prefill_model.py`.
- `test_gateddelta_prefill_profile_artifact.py`: unit tests for the GatedDelta
  prefill profile artifact validator.
- `render_gateddelta_prefill_profile_report.py`: renders a validated
  GatedDelta prefill profile artifact as a Markdown review table with
  linear-attention stage timings, recurrent share, dominant stage, and next-step
  hints for scan/fusion experiments. The benchmark harness can call it during
  capture with `--gateddelta-prefill-profile-report-output`.
- `check_gateddelta_prefill_model.py`: fail-closed preflight for real-model
  GatedDelta profile runs. It checks `config.json` plus `model-manifest.json`
  before release-server build and requires a `qwen3_5`/`qwen3_next`
  linear-attention manifest with the gated-delta kernel dimensions configured.
- `check_turboquant_quality_artifact.py`: fail-closed validator for internal
  TurboQuant long-context quality gate artifacts. It checks model identity,
  long-context shape, baseline/candidate provenance, candidate mode
  `turboquant-fused-experimental`, K8/V4 route metadata schema `>= 2`,
  fused_compressed_decode path code `2`, fused decode successes, zero
  fallbacks, decode quality thresholds, throughput ratio, and memory-savings
  evidence.
- `check_turboquant_microbench_artifact.py`: fail-closed validator for
  standalone fused cold-decode microbenchmark artifacts. It checks K8/V4
  metadata, long-cold-context coverage, `two_stage_scores` quality, memory
  savings, and speedup against the CPU reference plus `dim_parallel` when
  present.
- `build_turboquant_quality_artifact.py`: compiles a TurboQuant quality artifact
  from MLX inference-stack benchmark output and a quality-metrics JSON file,
  then validates it through the same fail-closed gate. Full-precision shadow
  and CPU oracle rows are rejected as promotion evidence.
- `build_turboquant_decode_outputs.py`: extracts opt-in AX response
  `output_token_ids` from MLX inference-stack benchmark artifacts into the
  `decode_outputs` vector format consumed by TurboQuant quality metrics. Rerun
  the benchmark with `--capture-output-token-ids` when producing real-model
  quality evidence.
- `run-turboquant-quality-artifact.sh`: real-model TurboQuant promotion-evidence
  runner. It builds the release server, runs full-precision and
  `turboquant-fused-experimental` AX rows with output-token capture, extracts
  decode vectors, builds quality metrics, validates the quality artifact, and
  writes a promotion-readiness report. Use `--dry-run` first to inspect
  inferred metadata and planned commands without loading a model.
- `run-mlx-prefill-scaling-artifact.sh`: real-model P1 prefill/TTFT scaling
  runner. It runs the MLX inference-stack benchmark with direct AX rows, writes
  the raw `ax.mlx_inference_stack.v2` artifact, builds
  `ax.mlx_prefill_scaling.v1`, validates the scaling artifact, and renders a
  Markdown review report. Use `--dry-run` first to inspect the planned
  long-context run.
- `run-gateddelta-prefill-profile.sh`: real-model Qwen/GatedDelta prefill
  profile runner. It preflights the model manifest before building the release
  server, runs `--gateddelta-prefill-profile`, writes and validates the raw
  inference-stack artifact, and renders the Markdown stage-profile report. Use
  `--dry-run` first to inspect the planned profile run.
- `run-mlx-p2-latency-artifacts.sh`: real-model P2 startup/concurrency wrapper.
  It builds the release AX server, invokes `run_mlx_p2_latency_artifacts.py`,
  and writes `startup-latency.json`, `concurrent-prefill.json`, and
  `p2-latency.md`. Use `--dry-run` first to inspect the output directory and
  generated command without building or starting the server.
- `build_mlx_prefill_scaling_artifact.py`: converts completed MLX
  inference-stack artifacts into the fail-closed prefill/TTFT scaling artifact
  consumed by `check_mlx_prefill_scaling_artifact.py`.
- `check_mlx_prefill_scaling_artifact.py`: validates long-context prefill/TTFT
  evidence, including `mlx_lm` baseline coverage, direct AX policy labeling,
  shared prompt hashes, TTFT, peak memory, and ratios to baseline.
- `render_mlx_prefill_scaling_report.py`: renders a validated prefill-scaling
  artifact as a Markdown review table with prefill, TTFT, memory, ratios, and
  first-bend marking.
- `check_mlx_prefill_scaling_campaign.py`: validates a multi-model prefill
  scaling campaign by checking per-artifact contracts, required model-family
  coverage, host consistency, maximum context coverage, and first-bend summary.
- `check_mlx_startup_latency_artifact.py`: validates P2 cold-vs-warm startup
  artifacts. It requires `process_cold`, `model_warm`, and `benchmark_warm`
  rows for the same prompt hash and direct AX policy, separates server-ready
  and model-load metrics from warm rows, and checks cold/warm ratios against
  the benchmark-warm row.
- `test_mlx_startup_latency_artifact.py`: unit tests for the startup latency
  artifact validator.
- `check_mlx_concurrent_prefill_artifact.py`: validates P2 concurrent-prefill
  artifacts. It requires a single-request baseline plus multi-request rows,
  one prompt hash per request, direct AX policy, per-request TTFT, total wall
  time, queue delay, zero failures, peak memory, overlap classification, and
  ratios to the single-request baseline.
- `test_mlx_concurrent_prefill_artifact.py`: unit tests for the concurrent
  prefill artifact validator.
- `run_mlx_p2_latency_artifacts.py`: real-model P2 runner for startup and
  concurrent-prefill evidence. It starts the AX MLX server in direct mode,
  writes prompt-token artifacts, captures `ax.mlx_startup_latency.v1` and
  `ax.mlx_concurrent_prefill.v1`, validates both outputs, and writes
  `p2-latency.md` before returning. Use `--dry-run` first to inspect the output
  paths without starting a server.
- `test_run_mlx_p2_latency_artifacts.py`: unit tests for the P2 latency runner
  artifact assembly, ratio calculations, and dry-run CLI.
- `render_mlx_p2_latency_report.py`: renders validated P2 startup and/or
  concurrent-prefill artifacts as a Markdown review report with cold/warm
  ratios, concurrency ratios, queue delay, memory pressure, and overlap
  classification.
- `test_render_mlx_p2_latency_report.py`: unit tests for the P2 latency report
  renderer.
- `build_turboquant_quality_metrics.py`: compares baseline and candidate decode
  output vectors and emits the max/mean absolute error plus minimum cosine
  similarity JSON consumed by the TurboQuant quality artifact builder.
- `test_turboquant_quality_artifact.py`: unit tests for the TurboQuant quality
  artifact validator.
- `test_turboquant_microbench_artifact.py`: unit tests for the TurboQuant fused
  microbenchmark artifact validator.
- `probe_mlx_model_support.py`: support-contract probe for downloaded MLX
  model artifacts. It reads `config.json`, safetensors index metadata, and
  local reference implementations so new architectures fail closed with named
  blockers instead of becoming benchmark-only support claims.
- `test_probe_mlx_model_support.py`: unit tests for GLM/DeepSeek support
  classification and fail-closed partial-reference behavior.
- `check-bench-inference-stack.sh`: lightweight contract check for the MLX
  inference-stack benchmark harness. It does not load a model.
- `check-turboquant-quality-gate.sh`: lightweight CLI pipeline check for
  TurboQuant quality evidence. It builds synthetic quality metrics, compiles a
  quality artifact, validates it, and proves `full_precision_shadow` candidates
  fail promotion.
- `check-turboquant-microbench-gate.sh`: lightweight CLI pipeline check for
  TurboQuant fused-kernel evidence. It validates a synthetic speed-positive
  `two_stage_scores` artifact and proves a CPU-regressed artifact fails.
- `check_turboquant_public_docs.py`: lightweight public-docs contract check for
  the optional TurboQuant switch, telemetry-only shadow boundary, and sync
  timing docs.
- `check_turboquant_promotion_readiness.py`: fail-closed readiness report for
  TurboQuant public-support promotion. It scans local model manifests and
  quality-gate artifacts, then reports whether public docs must remain
  experimental.
- `cargo run -p ax-engine-mlx --release --bin turboquant-microbench -- ...`:
  TurboQuant fused cold-decode microbenchmark. It compares the K8/V4 MLX/Metal
  kernels against the CPU reference oracle and writes
  `ax.turboquant_fused_decode_microbench.v1` JSON artifacts. Use `--variants`
  to limit longer sweeps to selected kernel variants and `--hot-tokens` to
  include shared log-sum-exp hot-tail merge evidence.
- `reproduce-mlx-inference-benchmark.sh`: public reproduction wrapper for
  external Apple Silicon benchmark bundles. It records doctor output, command
  logs, prompt artifacts, environment metadata, and raw JSON results.
- `diagnose_server_rss.py`: long-lived RSS diagnostic for MLX and delegated
  llama.cpp server routes. It is not a throughput benchmark.
- `check-bench-*.sh`: smoke checks for `ax-engine-bench` workload-contract
  commands.
- `check-server-preview.sh`, `check-python-preview.sh`: preview transport and
  binding smoke checks.
- `build-metal-kernels.sh`, `check-metal-kernel-contract.sh`: Metal artifact
  build and contract checks.

Run the MLX inference-stack unit tests without loading a model:

```text
bash scripts/check-bench-inference-stack.sh
```

Run the fast script hygiene gate before changing files in this directory:

```text
bash scripts/check-scripts.sh
```
