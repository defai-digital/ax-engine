# Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-lm`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | mlx-lm tok/s | AX tok/s | AX vs mlx-lm | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 46,093.9 | 43,093.7 | -6.5% | 168.3 | 48.6 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 41,387.5 | 40,383.3 | -2.4% | 157.7 | 206.2 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 40,593.9 | 39,546.0 | -2.6% | 154.5 | 416.4 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 39,830.5 | 41,216.8 | +3.5% | 80.5 | 100.4 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 41,792.9 | 40,170.3 | -3.9% | 78.5 | 446.8 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 40,959.8 | 41,208.2 | +0.6% | 80.5 | 813.2 |
