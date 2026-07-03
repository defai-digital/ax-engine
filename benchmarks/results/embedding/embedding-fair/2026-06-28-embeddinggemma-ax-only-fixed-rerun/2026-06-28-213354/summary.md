# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b1 | 1 | 10 | 2,393.2 | 239.3 |
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 10,823.3 | 1,170.1 |
| embeddinggemma-300m-8bit | fixed_16_b1 | 1 | 16 | 3,961.4 | 247.6 |
| embeddinggemma-300m-8bit | fixed_16_b8 | 8 | 16 | 19,392.5 | 1,212.0 |
| embeddinggemma-300m-8bit | fixed_64_b1 | 1 | 64 | 15,846.2 | 247.6 |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 62,534.0 | 977.1 |
| embeddinggemma-300m-8bit | fixed_256_b1 | 1 | 256 | 45,163.9 | 176.4 |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 138,771.4 | 542.1 |
