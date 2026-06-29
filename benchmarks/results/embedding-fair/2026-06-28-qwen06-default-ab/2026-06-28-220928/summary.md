# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 2,931.7 | 312.7 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 40,308.6 | 629.8 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 48,242.9 | 188.4 |
