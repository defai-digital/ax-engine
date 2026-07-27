# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 129,495.3 | 505.8 | 18.9 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 155,860.1 | 608.8 | 65.0 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 151,373.5 | 591.3 | 163.6 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 137,355.6 | 268.3 | 28.1 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 147,308.6 | 287.7 | 132.0 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 133,318.4 | 260.4 | 346.9 |

