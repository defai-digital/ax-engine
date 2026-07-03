# Fair Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-lm`, pooling: `last`.

| Model | Workload | Batch | Max tokens | mlx-lm tok/s | AX tok/s | AX vs mlx-lm |
|---|---|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 353.8 | 263.5 | -25.5% |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 2,383.2 | 1,628.1 | -31.7% |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 527.6 | 368.2 | -30.2% |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 3,914.4 | 2,823.4 | -27.9% |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 2,025.8 | 1,903.2 | -6.1% |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 13,707.4 | 8,454.3 | -38.3% |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 7,105.4 | 4,460.7 | -37.2% |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 31,773.8 | 24,153.1 | -24.0% |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 235.7 | 135.9 | -42.4% |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 1,196.2 | 834.1 | -30.3% |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 329.0 | 196.0 | -40.4% |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,231.8 | 1,377.5 | -38.3% |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,150.3 | 801.7 | -30.3% |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 4,940.1 | 3,629.4 | -26.5% |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 3,642.2 | 2,321.0 | -36.3% |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 6,584.1 | 5,307.4 | -19.4% |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 174.0 | 94.4 | -45.7% |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 937.2 | 657.6 | -29.8% |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 260.7 | 144.2 | -44.7% |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 1,673.1 | 1,068.9 | -36.1% |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 794.5 | 587.2 | -26.1% |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,011.2 | 2,328.0 | -22.7% |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 2,229.8 | 1,684.4 | -24.5% |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 3,668.1 | 3,352.3 | -8.6% |
