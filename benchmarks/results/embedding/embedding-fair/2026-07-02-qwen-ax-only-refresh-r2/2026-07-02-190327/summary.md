# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 2,297.0 | 245.0 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 12,284.1 | 191.9 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 27,655.5 | 108.0 |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 937.5 | 100.0 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 4,503.3 | 70.4 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 6,158.9 | 24.1 |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 733.6 | 78.2 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 2,399.5 | 37.5 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 3,471.5 | 13.6 |
