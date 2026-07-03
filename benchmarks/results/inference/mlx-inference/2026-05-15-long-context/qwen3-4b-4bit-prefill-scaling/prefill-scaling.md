# MLX Prefill Scaling Report

- Artifact: `benchmarks/results/mlx-inference/2026-05-15-long-context/qwen3-4b-4bit-prefill-scaling/prefill-scaling.json`
- Model: `mlx-community/Qwen3-4B-4bit`
- Host: Apple M5 Max / 128 GB
- Benchmark: batch=1, generation=1, repetitions=3, prefill_step_size=2048

TTFT scope: AX rows use runner prefill timing when available; `mlx_lm` rows may be derived from reported prefill throughput.

| Context tok | Gen tok | mlx_lm prefill tok/s | AX prefill tok/s | AX/MLX prefill | mlx_lm TTFT ms | AX TTFT ms | AX/MLX TTFT | AX peak GB | Bend |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 1 | 4,947.7 | 5,886.5 | 1.190x | 207.0 | 174.0 | 0.841x | 4.3 |  |
| 2,048 | 1 | 5,515.2 | 5,671.3 | 1.028x | 371.3 | 361.1 | 0.972x | 4.3 |  |
| 4,096 | 1 | 5,467.5 | 5,715.9 | 1.045x | 749.2 | 716.6 | 0.957x | 4.4 |  |
| 8,192 | 1 | 4,951.6 | 5,712.6 | 1.154x | 1,654.4 | 1,434.0 | 0.867x | 4.4 |  |

## Interpretation Guardrails

- No AX prefill throughput bend crossed the configured threshold.
- This report covers direct AX prefill/TTFT scaling, not n-gram decode acceleration.
- Treat host, model artifact identity, prompt hashes, and temperature as part of the claim.

