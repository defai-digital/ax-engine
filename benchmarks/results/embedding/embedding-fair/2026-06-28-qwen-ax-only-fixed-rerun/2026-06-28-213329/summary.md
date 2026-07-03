# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 1,729.6 | 173.0 |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 6,578.4 | 701.7 |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,121.1 | 195.1 |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 14,368.7 | 898.0 |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 11,013.1 | 172.1 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 39,877.1 | 623.1 |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 21,493.0 | 84.0 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 47,156.3 | 184.2 |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 531.2 | 53.1 |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,203.9 | 235.1 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 817.8 | 51.1 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 3,990.2 | 249.4 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,576.0 | 40.3 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,501.6 | 101.6 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,828.4 | 22.8 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 6,799.3 | 26.6 |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 298.5 | 29.8 |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,383.0 | 147.5 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 460.9 | 28.8 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,496.4 | 156.0 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,417.0 | 22.1 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,391.9 | 53.0 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 3,105.6 | 12.1 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 2,957.5 | 11.6 |
