# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `128`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 16 | 47,925.9 | 187.2 | 43.3 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 4 | 48,561.7 | 189.7 | 169.9 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 2 | 43,313.0 | 169.2 | 378.5 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 16 | 41,555.9 | 81.2 | 99.2 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 4 | 37,440.3 | 73.1 | 438.7 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 2 | 35,099.5 | 68.6 | 935.4 |
