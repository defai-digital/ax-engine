# Long-Context Decode-at-Depth Report

- Artifact: `benchmarks/results/mlx-inference/2026-05-15-long-context/qwen3-4b-4bit-decode-at-depth.json`
- Model: `mlx-community/Qwen3-4B-4bit`
- Host: Apple M5 Max / 128 GB
- Benchmark: batch=1, repetitions=3

Scope: decode throughput after a context depth already exists. AX and `mlx_lm` rows share the prompt-token hash. `llama.cpp Metal` rows must be explicit `llama-bench n_depth` evidence and remain shape-compatible external rows.

| Context depth tok | Gen tok | mlx_lm decode tok/s | AX decode tok/s | AX/MLX decode | llama.cpp decode tok/s | llama/MLX decode |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 128 | 150.8 | 160.9 | 1.067x | n/a | n/a |
| 8,192 | 128 | 129.7 | 159.8 | 1.232x | n/a | n/a |

## Interpretation Guardrails

- This report compares decode throughput after context depth, not cold prefill.
- AX-vs-`mlx_lm` rows are prompt-hash parity rows.
- `llama.cpp Metal` rows are included only when they carry explicit `llama-bench n_depth` evidence.
- Serving TTFT, queue delay, and prefix-cache reuse require separate artifacts.

