# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 2,213.0 | 221.3 |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 8,861.0 | 945.2 |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,497.2 | 218.6 |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 12,245.9 | 765.4 |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 10,345.5 | 161.6 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 41,761.7 | 652.5 |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 22,339.0 | 87.3 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 49,217.3 | 192.3 |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 563.1 | 56.3 |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,292.0 | 244.5 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 859.0 | 53.7 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 4,091.2 | 255.7 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,623.3 | 41.0 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,640.8 | 103.8 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,943.0 | 23.2 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 7,065.9 | 27.6 |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 313.4 | 31.3 |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,455.0 | 155.2 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 484.2 | 30.3 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,577.4 | 161.1 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,469.1 | 23.0 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,597.0 | 56.2 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 3,264.6 | 12.8 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 3,393.9 | 13.3 |
