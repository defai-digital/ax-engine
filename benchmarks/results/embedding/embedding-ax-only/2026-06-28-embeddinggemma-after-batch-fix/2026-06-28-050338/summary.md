# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 1,679.1 | 167.9 |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 7,141.2 | 772.0 |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 2,531.6 | 158.2 |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 12,877.5 | 804.8 |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 9,162.5 | 143.2 |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 41,829.9 | 653.6 |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 30,356.2 | 118.6 |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 102,633.9 | 400.9 |
