# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 1,352.4 | 135.2 |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 5,623.6 | 599.9 |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,242.2 | 202.6 |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 14,850.5 | 928.2 |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 11,705.2 | 182.9 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 41,213.7 | 644.0 |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 22,149.2 | 86.5 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 48,925.2 | 191.1 |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 552.2 | 55.2 |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,275.6 | 242.7 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 847.2 | 53.0 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 4,084.1 | 255.3 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,659.4 | 41.6 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,643.6 | 103.8 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,974.6 | 23.3 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 7,028.7 | 27.5 |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 307.2 | 30.7 |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,429.1 | 152.4 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 473.6 | 29.6 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,568.6 | 160.5 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,455.8 | 22.7 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,489.6 | 54.5 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 3,081.8 | 12.0 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 3,286.8 | 12.8 |
