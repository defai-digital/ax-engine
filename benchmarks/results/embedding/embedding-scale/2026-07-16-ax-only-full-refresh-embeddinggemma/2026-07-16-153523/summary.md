# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 139,406.8 | 544.6 | 14.5 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 155,946.1 | 609.2 | 61.6 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 155,791.9 | 608.6 | 131.7 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 140,787.0 | 275.0 | 28.5 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 147,916.9 | 288.9 | 121.9 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 139,730.3 | 272.9 | 262.6 |
