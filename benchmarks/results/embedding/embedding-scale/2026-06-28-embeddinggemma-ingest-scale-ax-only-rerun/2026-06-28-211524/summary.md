# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 134,791.6 | 526.5 | 15.8 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 138,683.3 | 541.7 | 63.2 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 141,961.7 | 554.5 | 117.9 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 119,555.6 | 233.5 | 35.6 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 114,677.8 | 224.0 | 145.1 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 105,631.7 | 206.3 | 317.7 |
