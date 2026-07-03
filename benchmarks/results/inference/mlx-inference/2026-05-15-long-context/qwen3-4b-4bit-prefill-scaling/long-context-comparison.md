# Long-Context Comparison Report

- Artifact: `benchmarks/results/mlx-inference/2026-05-15-long-context/qwen3-4b-4bit-prefill-scaling/long-context-comparison.json`
- Model: `mlx-community/Qwen3-4B-4bit`
- Host: Apple M5 Max / 128 GB
- Benchmark: batch=1, repetitions=3, prefill_step_size=2048

Scope: cold long-prefill comparison. AX and `mlx_lm` share the same prompt-token hash. `llama.cpp Metal` rows are external GGUF shape-compatible rows, not prompt-hash parity evidence.

| Context tok | Gen tok | mlx_lm prefill tok/s | AX prefill tok/s | AX/MLX prefill | llama.cpp prefill tok/s | llama/MLX prefill | AX TTFT ms | llama.cpp TTFT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 1 | 4,947.7 | 5,886.5 | 1.190x | n/a | n/a | 174.0 | n/a |
| 2,048 | 1 | 5,515.2 | 5,671.3 | 1.028x | n/a | n/a | 361.1 | n/a |
| 4,096 | 1 | 5,467.5 | 5,715.9 | 1.045x | n/a | n/a | 716.6 | n/a |
| 8,192 | 1 | 4,951.6 | 5,712.6 | 1.154x | n/a | n/a | 1,434.0 | n/a |

## Interpretation Guardrails

- This report compares cold prefill/derived TTFT, not prefix-cache reuse.
- AX-vs-`mlx_lm` rows are prompt-hash parity rows.
- `llama.cpp Metal` rows use `llama-bench` internal synthetic tokens and must stay in an external baseline column.
- Decode-at-depth and server prefix-reuse need separate artifacts before claiming long-session serving superiority.

