# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `128`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 16 | 42,706.6 | 166.8 | 49.4 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 4 | 41,920.4 | 163.8 | 196.5 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 2 | 40,199.7 | 157.0 | 407.9 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 16 | 39,323.4 | 76.8 | 106.0 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 4 | 39,264.8 | 76.7 | 420.6 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 2 | 39,089.2 | 76.3 | 840.7 |
