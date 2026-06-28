# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 2,374.6 | 237.5 |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 13,339.9 | 1,442.2 |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 4,313.2 | 269.6 |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 21,152.5 | 1,322.0 |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 11,432.8 | 178.6 |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 44,232.5 | 691.1 |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 33,567.4 | 131.1 |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 116,106.1 | 453.5 |
