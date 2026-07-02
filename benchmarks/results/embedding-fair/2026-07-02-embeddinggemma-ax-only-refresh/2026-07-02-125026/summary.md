# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 2,378.2 | 237.8 |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 14,518.6 | 1,569.6 |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 5,383.4 | 336.5 |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 26,210.9 | 1,638.2 |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 19,283.2 | 301.3 |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 55,440.0 | 866.3 |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 34,111.2 | 133.2 |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 121,188.5 | 473.4 |
