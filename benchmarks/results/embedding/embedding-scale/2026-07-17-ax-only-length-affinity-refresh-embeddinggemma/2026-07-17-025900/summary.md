# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 140,125.7 | 547.4 | 14.6 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 158,028.9 | 617.3 | 58.5 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 157,849.4 | 616.6 | 123.8 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 141,940.9 | 277.2 | 28.5 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 149,365.5 | 291.7 | 118.2 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 140,102.5 | 273.6 | 260.2 |
