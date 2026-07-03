# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `128`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 16 | 46,540.6 | 181.8 | 44.6 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 4 | 48,934.6 | 191.2 | 168.5 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 2 | 45,539.0 | 177.9 | 361.1 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 16 | 42,762.4 | 83.5 | 97.1 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 4 | 41,666.9 | 81.4 | 395.1 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 2 | 40,673.9 | 79.4 | 806.8 |
