# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 2,244.1 | 224.4 |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 8,703.8 | 928.4 |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,741.1 | 233.8 |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 15,550.6 | 971.9 |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 12,089.3 | 188.9 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 41,564.1 | 649.4 |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 22,269.0 | 87.0 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 49,010.4 | 191.4 |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 495.6 | 49.6 |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,264.5 | 241.5 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 854.6 | 53.4 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 4,097.9 | 256.1 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,649.5 | 41.4 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,611.9 | 103.3 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 6,031.8 | 23.6 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 7,012.7 | 27.4 |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 308.8 | 30.9 |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,441.0 | 153.7 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 480.7 | 30.0 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,569.0 | 160.6 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,472.3 | 23.0 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,568.9 | 55.8 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 3,228.6 | 12.6 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 3,512.5 | 13.7 |
