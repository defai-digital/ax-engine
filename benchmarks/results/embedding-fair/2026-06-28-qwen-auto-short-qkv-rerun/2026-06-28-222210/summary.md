# AX-Only Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Engine: `ax-engine-py`, pooling: `last`.

| Model | Workload | Batch | Max tokens | AX tok/s | AX items/s |
|---|---|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 2,173.9 | 217.4 |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 10,573.2 | 1,127.8 |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,673.2 | 229.6 |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 18,247.3 | 1,140.5 |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 11,728.9 | 183.3 |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 40,703.6 | 636.0 |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 22,355.8 | 87.3 |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 47,604.2 | 186.0 |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 537.8 | 53.8 |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,441.2 | 260.4 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 819.5 | 51.2 |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 4,342.6 | 271.4 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,581.8 | 40.3 |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,409.3 | 100.1 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,807.1 | 22.7 |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 6,848.4 | 26.8 |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 305.4 | 30.5 |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,520.7 | 162.2 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 460.9 | 28.8 |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,751.3 | 172.0 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,411.9 | 22.1 |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,497.0 | 54.6 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 3,120.4 | 12.2 |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 3,414.9 | 13.3 |
