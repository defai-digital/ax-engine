# Fair Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-embeddings`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | mlx-embeddings tok/s | AX tok/s | AX vs mlx-embeddings |
|---|---|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 1,697.5 | 1,539.8 | -9.3% |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 7,886.4 | 8,701.0 | +10.3% |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 2,765.0 | 2,747.2 | -0.6% |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 15,654.4 | 16,255.1 | +3.8% |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 10,796.5 | 10,090.7 | -6.5% |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 49,153.0 | 56,022.4 | +14.0% |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 32,252.8 | 36,775.0 | +14.0% |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 129,621.3 | 109,240.5 | -15.7% |
