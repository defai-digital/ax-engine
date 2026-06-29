# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 10,558.1 | 1,126.2 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 39,024.9 | 609.8 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 45,618.7 | 178.2 |
