# Scripts

This directory contains repo-owned smoke checks and a small number of diagnostic
helpers.

## Benchmarking Rule

Use `bench_mlx_inference_stack.py` for AX Engine MLX model-inference
comparisons. That script always runs `mlx_lm.benchmark` as the primary baseline
and fails the run if the baseline cannot be produced. Optional `mlx-swift-lm`
adapter rows and AX Engine MLX rows are compared against the matching
`mlx_lm.benchmark` prompt/decode shape.

Use `ax-engine-bench` through the `check-bench-*.sh` scripts for workload
contracts: scenario, replay, matrix, baseline, compare, delegated llama.cpp
route checks, and readiness.

Do not use ad hoc server timing or llama.cpp route checks as AX-owned MLX
throughput baselines.

## Script Groups

- `bench_mlx_inference_stack.py`: MLX model-inference comparison against
  `mlx_lm.benchmark`.
- `bench_memory_leak_server.py`: long-lived RSS diagnostic for MLX and delegated
  llama.cpp server routes. It is not a throughput benchmark.
- `check-bench-*.sh`: smoke checks for `ax-engine-bench` workload-contract
  commands.
- `check-server-preview.sh`, `check-python-preview.sh`: preview transport and
  binding smoke checks.
- `build-metal-kernels.sh`, `check-metal-kernel-contract.sh`: Metal artifact
  build and contract checks.
