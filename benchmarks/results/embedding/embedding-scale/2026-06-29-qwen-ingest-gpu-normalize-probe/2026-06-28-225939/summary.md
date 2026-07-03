# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `128`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 16 | 47,307.8 | 184.8 | 43.8 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 4 | 48,071.2 | 187.8 | 171.1 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 2 | 44,402.4 | 173.4 | 371.8 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 16 | 40,346.8 | 78.8 | 103.6 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 4 | 36,795.5 | 71.9 | 447.8 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 2 | 35,127.4 | 68.6 | 934.6 |
