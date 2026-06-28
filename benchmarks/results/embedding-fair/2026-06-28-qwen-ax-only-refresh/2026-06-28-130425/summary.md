# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 2,293.3 | 229.3 |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 8,566.0 | 913.7 |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,664.1 | 229.0 |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 15,228.8 | 951.8 |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 12,013.6 | 187.7 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 40,796.4 | 637.4 |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 22,337.6 | 87.3 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 48,429.0 | 189.2 |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 542.4 | 54.2 |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,232.5 | 238.1 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 830.7 | 51.9 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 4,004.7 | 250.3 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,583.7 | 40.4 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,459.0 | 100.9 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,965.8 | 23.3 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 6,900.0 | 27.0 |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 303.0 | 30.3 |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,400.9 | 149.4 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 466.3 | 29.1 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,492.7 | 155.8 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,413.7 | 22.1 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,481.5 | 54.4 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 3,188.3 | 12.5 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 3,455.2 | 13.5 |
