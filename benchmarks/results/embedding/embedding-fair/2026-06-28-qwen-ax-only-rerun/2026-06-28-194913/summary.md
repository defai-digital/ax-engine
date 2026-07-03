# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 2,146.3 | 214.6 |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 6,981.8 | 744.7 |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,198.9 | 199.9 |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 13,891.3 | 868.2 |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 11,148.7 | 174.2 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 40,821.1 | 637.8 |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 22,191.1 | 86.7 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 48,524.3 | 189.5 |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 554.1 | 55.4 |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,241.2 | 239.1 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 844.3 | 52.8 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 4,001.5 | 250.1 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,633.5 | 41.1 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,599.6 | 103.1 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,981.4 | 23.4 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 7,048.6 | 27.5 |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 308.1 | 30.8 |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,406.5 | 150.0 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 476.9 | 29.8 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,510.1 | 156.9 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,446.3 | 22.6 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,489.7 | 54.5 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 3,163.6 | 12.4 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 2,768.3 | 10.8 |
