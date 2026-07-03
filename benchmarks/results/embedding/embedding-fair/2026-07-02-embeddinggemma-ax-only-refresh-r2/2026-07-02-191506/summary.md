# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `mean`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | short_query_b8 | 8 | 15 | 2,489.2 | 269.1 |
| embeddinggemma-300m-8bit | fixed_64_b8 | 8 | 64 | 15,376.6 | 240.3 |
| embeddinggemma-300m-8bit | fixed_256_b8 | 8 | 256 | 48,130.2 | 188.0 |
