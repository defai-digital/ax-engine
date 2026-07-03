# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 2,087.9 | 208.8 |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 8,188.0 | 873.4 |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,620.3 | 226.3 |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 14,732.0 | 920.7 |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 11,595.5 | 181.2 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 39,298.7 | 614.0 |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 21,378.9 | 83.5 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 46,690.8 | 182.4 |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 533.8 | 53.4 |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,175.2 | 232.0 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 816.7 | 51.0 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 3,844.2 | 240.3 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,553.1 | 39.9 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,251.5 | 97.7 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,788.4 | 22.6 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 6,856.1 | 26.8 |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 302.1 | 30.2 |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,348.8 | 143.9 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 462.9 | 28.9 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,387.2 | 149.2 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,406.0 | 22.0 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,440.6 | 53.8 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 3,139.0 | 12.3 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 3,428.4 | 13.4 |
