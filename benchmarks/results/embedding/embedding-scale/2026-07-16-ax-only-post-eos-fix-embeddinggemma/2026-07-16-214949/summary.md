# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 139,063.5 | 543.2 | 14.6 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 157,588.2 | 615.6 | 59.1 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 157,558.0 | 615.5 | 126.4 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 141,305.0 | 276.0 | 28.5 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 148,361.9 | 289.8 | 120.0 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 140,032.1 | 273.5 | 259.4 |
