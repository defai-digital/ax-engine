# Fair Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-embeddings`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | mlx-embeddings tok/s | AX tok/s | AX vs mlx-embeddings |
|---|---|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 1,845.1 | 3,247.1 | +76.0% |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 11,526.9 | 8,938.8 | -22.5% |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 2,577.9 | 3,096.9 | +20.1% |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 15,105.2 | 17,191.5 | +13.8% |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 11,827.3 | 14,164.8 | +19.8% |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 56,461.5 | 63,373.9 | +12.2% |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 34,630.9 | 49,026.5 | +41.6% |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 135,292.8 | 143,592.1 | +6.1% |
