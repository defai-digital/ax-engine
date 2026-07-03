# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 2,138.6 | 213.9 |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 13,128.1 | 1,419.3 |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 3,929.7 | 245.6 |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 14,698.6 | 918.7 |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 9,545.6 | 149.1 |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 41,672.3 | 651.1 |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 32,933.3 | 128.6 |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 113,091.6 | 441.8 |
