# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 2,343.9 | 234.4 |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 9,856.8 | 1,065.6 |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 3,594.4 | 224.6 |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 18,072.5 | 1,129.5 |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 13,436.0 | 209.9 |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 54,852.0 | 857.1 |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 36,607.8 | 143.0 |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 128,960.5 | 503.8 |
