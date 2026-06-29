# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `32`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 4 | 46,444.5 | 181.4 | 44.4 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 1 | 48,759.7 | 190.5 | 168.0 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 1 | 49,459.0 | 193.2 | 165.6 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 4 | 46,811.0 | 91.4 | 88.2 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 1 | 47,972.9 | 93.7 | 341.5 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 1 | 48,020.8 | 93.8 | 341.2 |
