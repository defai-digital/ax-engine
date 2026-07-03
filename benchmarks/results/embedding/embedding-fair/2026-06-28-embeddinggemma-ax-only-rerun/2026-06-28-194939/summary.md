# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 2,246.8 | 224.7 |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 10,579.2 | 1,143.7 |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 3,807.4 | 238.0 |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 18,252.6 | 1,140.8 |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 13,915.8 | 217.4 |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 55,314.7 | 864.3 |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 38,003.6 | 148.5 |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 125,852.6 | 491.6 |
