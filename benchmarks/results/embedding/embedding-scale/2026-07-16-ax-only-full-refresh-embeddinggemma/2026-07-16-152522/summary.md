# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 139,894.5 | 546.5 | 14.6 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 156,641.9 | 611.9 | 59.9 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 156,377.3 | 610.8 | 130.1 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 141,268.9 | 275.9 | 28.5 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 148,298.1 | 289.6 | 121.5 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 140,077.0 | 273.6 | 262.2 |
