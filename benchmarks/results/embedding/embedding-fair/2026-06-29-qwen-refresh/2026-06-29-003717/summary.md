# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 2,184.8 | 218.5 |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 9,294.5 | 991.4 |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,465.2 | 216.6 |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 17,709.1 | 1,106.8 |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 11,702.2 | 182.8 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 39,788.5 | 621.7 |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 21,835.6 | 85.3 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 47,487.2 | 185.5 |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 533.1 | 53.3 |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,455.4 | 261.9 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 823.1 | 51.4 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 4,334.4 | 270.9 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,595.2 | 40.6 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,455.2 | 100.9 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,861.8 | 22.9 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 6,822.9 | 26.7 |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 300.6 | 30.1 |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,530.9 | 163.3 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 463.0 | 28.9 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,729.8 | 170.6 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,413.6 | 22.1 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,439.5 | 53.7 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 2,924.7 | 11.4 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 2,852.9 | 11.1 |
