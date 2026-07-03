# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 2,381.4 | 238.1 |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 14,158.6 | 1,530.7 |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 5,333.2 | 333.3 |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 26,723.5 | 1,670.2 |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 19,534.0 | 305.2 |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 79,181.9 | 1,237.2 |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 52,138.5 | 203.7 |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 140,586.5 | 549.2 |
