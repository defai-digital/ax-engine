# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 42,323.7 | 165.3 | 53.2 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 38,315.6 | 149.7 | 217.2 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 37,186.3 | 145.3 | 447.2 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 37,330.5 | 72.9 | 113.0 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 37,380.5 | 73.0 | 449.8 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 38,391.2 | 75.0 | 890.6 |
