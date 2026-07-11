# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 15,882.5 | 62.0 | 131.8 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 16,010.6 | 62.5 | 544.3 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 16,015.7 | 62.6 | 1,058.3 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 15,387.1 | 30.1 | 292.6 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 15,355.3 | 30.0 | 1,115.6 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 15,274.9 | 29.8 | 2,210.7 |
