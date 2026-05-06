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
  `--experimental-mlx-kv-compression turboquant-shadow` for AX rows and records
  TurboQuant KV compression route counters when the runtime emits them.
- `test_bench_mlx_inference_stack.py`: unit tests for the MLX benchmark
  contract, parser, prompt artifact hash checks, and secondary adapter shape.
- `check_turboquant_quality_artifact.py`: fail-closed validator for internal
  TurboQuant long-context quality gate artifacts. It checks model identity,
  long-context shape, baseline/candidate provenance, K8/V4 route metadata,
  decode quality thresholds, throughput ratio, and memory-savings evidence.
- `build_turboquant_quality_artifact.py`: compiles a TurboQuant quality artifact
  from MLX inference-stack benchmark output and a quality-metrics JSON file,
  then validates it through the same fail-closed gate. Full-precision shadow
  rows are rejected as promotion evidence.
- `build_turboquant_quality_metrics.py`: compares baseline and candidate decode
  output vectors and emits the max/mean absolute error plus minimum cosine
  similarity JSON consumed by the TurboQuant quality artifact builder.
- `test_turboquant_quality_artifact.py`: unit tests for the TurboQuant quality
  artifact validator.
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
