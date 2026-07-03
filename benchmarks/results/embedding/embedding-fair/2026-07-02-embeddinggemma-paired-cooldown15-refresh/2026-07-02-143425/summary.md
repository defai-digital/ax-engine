# Fair Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-embeddings`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | mlx-embeddings tok/s | AX tok/s | AX vs mlx-embeddings |
|---|---|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 437.5 | 344.5 | -21.3% |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 2,562.5 | 2,117.5 | -17.4% |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 628.9 | 486.6 | -22.6% |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 4,360.7 | 3,409.3 | -21.8% |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 2,464.4 | 1,870.0 | -24.1% |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 15,587.9 | 13,713.9 | -12.0% |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 8,922.9 | 6,894.7 | -22.7% |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 50,710.1 | 34,754.7 | -31.5% |
