# Fair Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-embeddings`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | mlx-embeddings tok/s | AX tok/s | AX vs mlx-embeddings |
|---|---|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 2,758.7 | 2,448.0 | -11.3% |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 14,382.5 | 13,020.6 | -9.5% |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 3,990.6 | 4,191.3 | +5.0% |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 25,290.8 | 22,880.1 | -9.5% |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 15,019.9 | 13,833.1 | -7.9% |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 69,764.3 | 75,666.9 | +8.5% |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 37,448.8 | 40,432.5 | +8.0% |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 133,722.9 | 112,957.8 | -15.5% |
