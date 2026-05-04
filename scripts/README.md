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

Use `ax-engine-bench` through the `check-bench-*.sh` scripts for workload
contracts: scenario, replay, matrix, baseline, compare, delegated llama.cpp
route checks, and readiness.

Do not use ad hoc server timing or llama.cpp route checks as AX-owned MLX
throughput baselines.

## Script Groups

- `lib/common.sh`: shared shell helpers for repo-root discovery, Python binary
  selection, temporary paths, free-port allocation, and PID cleanup.
- `bench_mlx_inference_stack.py`: MLX model-inference comparison against
  `mlx_lm.benchmark`.
- `test_bench_mlx_inference_stack.py`: unit tests for the MLX benchmark
  contract, parser, prompt artifact hash checks, and secondary adapter shape.
- `bench_memory_leak_server.py`: long-lived RSS diagnostic for MLX and delegated
  llama.cpp server routes. It is not a throughput benchmark.
- `check-bench-*.sh`: smoke checks for `ax-engine-bench` workload-contract
  commands.
- `check-server-preview.sh`, `check-python-preview.sh`: preview transport and
  binding smoke checks.
- `build-metal-kernels.sh`, `check-metal-kernel-contract.sh`: Metal artifact
  build and contract checks.

Run the MLX inference-stack unit tests without loading a model:

```text
python3 -m unittest scripts/test_bench_mlx_inference_stack.py -v
```
