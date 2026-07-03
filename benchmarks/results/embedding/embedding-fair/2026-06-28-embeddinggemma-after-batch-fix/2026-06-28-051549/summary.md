# Fair Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-embeddings`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | mlx-embeddings tok/s | AX tok/s | AX vs mlx-embeddings |
|---|---|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 1,716.7 | 1,954.5 | +13.9% |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 9,373.9 | 9,084.7 | -3.1% |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 3,027.2 | 3,017.3 | -0.3% |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 16,937.1 | 16,516.6 | -2.5% |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 11,445.6 | 11,227.7 | -1.9% |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 49,857.6 | 49,006.9 | -1.7% |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 27,758.8 | 31,905.8 | +14.9% |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 122,935.6 | 112,710.4 | -8.3% |
