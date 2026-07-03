# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 137,565.1 | 537.4 | 14.4 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 156,466.7 | 611.2 | 63.7 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 153,679.0 | 600.3 | 148.9 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 138,909.2 | 271.3 | 28.9 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 148,668.6 | 290.4 | 123.9 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 141,292.5 | 276.0 | 263.6 |
