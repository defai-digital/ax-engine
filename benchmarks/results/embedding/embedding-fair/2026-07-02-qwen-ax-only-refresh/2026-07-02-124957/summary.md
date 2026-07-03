# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 2,063.2 | 206.3 |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 10,523.2 | 1,122.5 |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,473.1 | 217.1 |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 11,681.5 | 730.1 |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 8,775.8 | 137.1 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 37,495.3 | 585.9 |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 22,268.1 | 87.0 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 48,748.8 | 190.4 |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 557.5 | 55.8 |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,454.3 | 261.8 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 847.6 | 53.0 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 4,250.3 | 265.6 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,617.6 | 40.9 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,498.5 | 101.5 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,992.3 | 23.4 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 7,015.8 | 27.4 |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 311.3 | 31.1 |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,486.3 | 158.5 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 471.6 | 29.5 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,630.1 | 164.4 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,457.0 | 22.8 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,513.0 | 54.9 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 3,202.8 | 12.5 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 3,560.2 | 13.9 |
