# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 1,697.5 | 169.7 |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 7,292.2 | 788.3 |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 2,728.7 | 170.5 |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 12,898.9 | 806.2 |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 10,650.3 | 166.4 |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 43,952.4 | 686.8 |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 34,221.2 | 133.7 |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 120,040.4 | 468.9 |
