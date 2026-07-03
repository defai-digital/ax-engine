# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 2,329.0 | 232.9 |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 8,462.2 | 902.6 |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,681.4 | 230.1 |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 14,741.0 | 921.3 |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 11,817.4 | 184.6 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 40,763.9 | 636.9 |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 21,695.0 | 84.7 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 47,200.4 | 184.4 |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 542.0 | 54.2 |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,188.9 | 233.5 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 814.4 | 50.9 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 3,929.8 | 245.6 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,589.2 | 40.5 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,374.7 | 99.6 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,813.0 | 22.7 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 6,715.7 | 26.2 |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 297.2 | 29.7 |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,370.6 | 146.2 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 459.3 | 28.7 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,417.6 | 151.1 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,396.9 | 21.8 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,351.7 | 52.4 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 3,028.9 | 11.8 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 3,074.0 | 12.0 |
