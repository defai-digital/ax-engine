# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 2,529.7 | 253.0 |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 13,207.7 | 1,427.9 |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 4,233.2 | 264.6 |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 22,883.4 | 1,430.2 |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 15,844.0 | 247.6 |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 69,655.1 | 1,088.4 |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 44,662.5 | 174.5 |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 113,814.4 | 444.6 |
