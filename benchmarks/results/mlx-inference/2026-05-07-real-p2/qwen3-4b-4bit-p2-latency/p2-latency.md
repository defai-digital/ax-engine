# MLX P2 Latency Report

This report is rendered only from validated P2 artifacts.

## Startup Latency

- Artifact: `benchmarks/results/mlx-inference/2026-05-07-real-p2/qwen3-4b-4bit-p2-latency/startup-latency.json`
- Model: `mlx-community/Qwen3-4B-4bit`
- Host: Apple M5 Max
- Shape: context=8192, generation=128, repetitions=3

| Phase | Server ready ms | Model load ms | TTFT ms | TTFT vs warm | Decode tok/s | Decode vs warm | Peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| process_cold | 510.7 | 506.3 | 2,085.1 | 0.831x | 117.1 | 1.005x | 4.3 |
| model_warm | - | 505.4 | 2,130.8 | 0.849x | 117.9 | 1.012x | 4.3 |
| benchmark_warm | - | - | 2,509.7 | - | 116.5 | - | 4.3 |

Startup guardrail: cold rows include process/server/model-load costs; the benchmark-warm row must not mix those costs into warm throughput.

## Concurrent Prefill

- Artifact: `benchmarks/results/mlx-inference/2026-05-07-real-p2/qwen3-4b-4bit-p2-latency/concurrent-prefill.json`
- Model: `mlx-community/Qwen3-4B-4bit`
- Host: Apple M5 Max
- Shape: context=8192, generation=1, repetitions=3

| Requests | Request TTFT ms | TTFT vs single | Total wall ms | Wall vs single | Queue delay ms | Failures max | Peak GB | Memory vs single | Overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2,208.9 | - | 2,374.5 | - | 164.6 | 0 | 4.3 | - | serialized |
| 2 | 3,900.0 | 1.766x | 4,202.9 | 1.770x | 270.5 | 0 | 4.6 | 1.067x | partial_overlap |
| 4 | 8,318.7 | 3.766x | 8,746.2 | 3.683x | 401.5 | 0 | 8.3 | 1.914x | serialized |

Concurrency guardrail: this is server-path long-prompt evidence, not proof of continuous batching or production multi-user throughput.

## Interpretation Guardrails

- Keep P2 startup/concurrency evidence separate from README batch=1 throughput rows.
- These rows use direct AX MLX policy; they do not measure n-gram decode acceleration.
- Treat host, model artifact identity, prompt hashes, and generation shape as part of the claim.

