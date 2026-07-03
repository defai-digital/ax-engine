# Fair Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-lm`, pooling: `last`.

| Model | Workload | Batch | Max tokens | mlx-lm tok/s | AX tok/s | AX vs mlx-lm |
|---|---|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 2,225.6 | 2,483.5 | +11.6% |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 8,703.3 | 10,419.2 | +19.7% |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,388.3 | 3,730.2 | +10.1% |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 13,769.4 | 15,279.5 | +11.0% |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 8,716.1 | 11,603.3 | +33.1% |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 40,873.9 | 40,341.0 | -1.3% |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 19,900.5 | 22,307.9 | +12.1% |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 49,104.1 | 47,311.8 | -3.6% |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 445.4 | 560.3 | +25.8% |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,277.8 | 2,478.4 | +8.8% |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 784.3 | 836.6 | +6.7% |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 3,898.6 | 4,151.6 | +6.5% |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,166.9 | 2,597.8 | +19.9% |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,537.0 | 6,405.2 | -2.0% |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,221.7 | 5,975.1 | +14.4% |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 7,046.6 | 6,990.6 | -0.8% |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 331.4 | 306.9 | -7.4% |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,380.8 | 1,459.0 | +5.7% |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 453.2 | 461.7 | +1.9% |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,479.1 | 2,596.0 | +4.7% |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,233.5 | 1,435.5 | +16.4% |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,537.7 | 3,545.0 | +0.2% |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 2,838.9 | 3,222.8 | +13.5% |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 3,777.9 | 3,373.2 | -10.7% |
