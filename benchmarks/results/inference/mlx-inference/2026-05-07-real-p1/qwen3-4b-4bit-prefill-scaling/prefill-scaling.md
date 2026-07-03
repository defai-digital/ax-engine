# MLX Prefill Scaling Report

- Artifact: `benchmarks/results/mlx-inference/2026-05-07-real-p1/qwen3-4b-4bit-prefill-scaling/prefill-scaling.json`
- Model: `mlx-community/Qwen3-4B-4bit`
- Host: Apple M5 Max / 128 GB
- Benchmark: batch=1, generation=1, repetitions=3, prefill_step_size=2048

TTFT scope: AX rows use runner prefill timing when available; `mlx_lm` rows may be derived from reported prefill throughput.

| Context tok | Gen tok | mlx_lm prefill tok/s | AX prefill tok/s | AX/MLX prefill | mlx_lm TTFT ms | AX TTFT ms | AX/MLX TTFT | AX peak GB | Bend |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 1 | 4,638.9 | 5,375.7 | 1.159x | 220.7 | 190.5 | 0.863x | 4.3 |  |
| 2,048 | 1 | 5,351.4 | 5,257.0 | 0.982x | 382.7 | 389.6 | 1.018x | 4.3 |  |
| 4,096 | 1 | 5,300.4 | 4,841.3 | 0.913x | 772.8 | 846.1 | 1.095x | 4.4 |  |
| 8,192 | 1 | 4,728.6 | 3,973.6 | 0.840x | 1,732.4 | 2,061.6 | 1.190x | 4.4 |  |

## Interpretation Guardrails

- No AX prefill throughput bend crossed the configured threshold.
- This report covers direct AX prefill/TTFT scaling, not n-gram decode acceleration.
- Treat host, model artifact identity, prompt hashes, and temperature as part of the claim.

