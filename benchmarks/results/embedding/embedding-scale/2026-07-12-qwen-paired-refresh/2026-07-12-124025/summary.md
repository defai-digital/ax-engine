# Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-lm`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | mlx-lm tok/s | AX tok/s | AX vs mlx-lm | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 49,025.4 | 16,096.0 | -67.2% | 62.9 | 127.2 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 50,570.7 | 16,252.7 | -67.9% | 63.5 | 512.0 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 50,502.5 | 16,250.1 | -67.8% | 63.5 | 1,029.4 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 48,997.8 | 15,657.8 | -68.0% | 30.6 | 265.8 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 45,047.9 | 15,230.2 | -66.2% | 29.7 | 1,143.8 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 45,352.9 | 15,412.8 | -66.0% | 30.1 | 2,156.6 |

