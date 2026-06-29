# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 38,764.6 | 151.4 | 53.9 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 37,060.4 | 144.8 | 223.2 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 34,997.7 | 136.7 | 478.3 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 38,788.7 | 75.8 | 116.7 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 35,314.7 | 69.0 | 468.2 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 34,328.0 | 67.0 | 967.1 |
