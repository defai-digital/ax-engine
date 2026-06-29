# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 38,512.7 | 150.4 | 56.5 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 37,204.9 | 145.3 | 222.8 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 37,428.4 | 146.2 | 443.7 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 36,086.2 | 70.5 | 118.0 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 37,151.7 | 72.6 | 467.1 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 37,918.0 | 74.1 | 909.7 |
