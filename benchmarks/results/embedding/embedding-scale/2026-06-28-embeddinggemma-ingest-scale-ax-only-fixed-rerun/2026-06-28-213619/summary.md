# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 129,930.4 | 507.5 | 16.4 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 141,346.5 | 552.1 | 58.8 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 138,489.8 | 541.0 | 129.7 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 122,056.8 | 238.4 | 38.3 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 125,967.3 | 246.0 | 148.4 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 118,548.7 | 231.5 | 307.3 |
