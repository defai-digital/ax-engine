# Qwen3-4B Long-Context N-Gram Decode Diagnostic

- Source artifact: `benchmarks/results/mlx-inference/2026-05-15-long-context/qwen3-4b-4bit-ngram-depth-source.json`
- Model: `mlx-community/Qwen3-4B-4bit`
- Host: Apple M5 Max / 128 GB
- Shape: prompt depth 4096/8192, generation=128, repetitions=3

Scope: diagnostic AX default n-gram decode throughput after long context. This is not cold-prefill, serving concurrency, or prefix-reuse evidence.

| Context depth tok | mlx_lm decode tok/s | AX direct decode tok/s | AX n-gram decode tok/s | n-gram / mlx_lm | n-gram / direct | Draft attempts | Accepted tokens | Accept rate | Claim status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4,096 | 153.3 | 161.4 | 341.2 | 2.225x | 2.114x | 54 | 324 | 100.0% | ngram_acceleration_effective_throughput |
| 8,192 | 128.5 | 159.0 | 345.1 | 2.686x | 2.171x | 51 | 306 | 100.0% | ngram_acceleration_effective_throughput |

## Interpretation Guardrails

- The n-gram rows show effective draft acceptance on this random-token prompt contract; they should not be used as a prefill claim.
- The source artifact keeps the matching AX direct row so the n-gram benefit can be separated from base runtime decode speed.
- Serving TTFT, queue delay, and concurrent long-prefill behavior still require serving artifacts.
