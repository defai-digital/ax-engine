# Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-lm`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | mlx-lm tok/s | AX tok/s | AX vs mlx-lm | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 48,668.4 | 48,875.8 | +0.4% | 190.9 | 41.8 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 50,051.9 | 50,105.8 | +0.1% | 195.7 | 173.8 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 49,909.7 | 50,131.0 | +0.4% | 195.8 | 353.1 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 49,061.5 | 49,057.8 | -0.0% | 95.8 | 83.8 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 48,895.0 | 49,156.0 | +0.5% | 96.0 | 349.8 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 48,978.0 | 49,051.0 | +0.1% | 95.8 | 703.9 |

