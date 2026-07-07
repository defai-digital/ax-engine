# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 136,101.8 | 531.6 | 15.0 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 154,307.2 | 602.8 | 60.1 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 152,897.7 | 597.3 | 126.1 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 138,617.4 | 270.7 | 29.5 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 141,992.7 | 277.3 | 123.3 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 135,667.0 | 265.0 | 263.1 |
