# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 8,131.1 | 867.3 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 40,472.6 | 632.4 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 46,996.5 | 183.6 |
