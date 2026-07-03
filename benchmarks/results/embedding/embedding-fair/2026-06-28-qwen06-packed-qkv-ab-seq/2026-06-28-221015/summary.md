# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 10,267.3 | 1,095.2 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 39,618.5 | 619.0 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 46,925.3 | 183.3 |
