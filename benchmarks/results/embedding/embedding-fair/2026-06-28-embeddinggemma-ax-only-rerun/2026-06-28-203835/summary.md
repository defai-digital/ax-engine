# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 2,778.1 | 277.8 |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 12,639.7 | 1,366.5 |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 4,537.7 | 283.6 |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 22,118.4 | 1,382.4 |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 16,418.0 | 256.5 |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 66,974.1 | 1,046.5 |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 46,036.6 | 179.8 |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 135,244.0 | 528.3 |
