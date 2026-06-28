# Fair Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-lm`, pooling: `last`.

| Model | Workload | Batch | Max tokens | mlx-lm tok/s | AX tok/s | AX vs mlx-lm |
|---|---|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 2,082.3 | 2,127.5 | +2.2% |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 8,343.2 | 8,747.8 | +4.8% |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 2,995.2 | 3,607.1 | +20.4% |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 14,612.1 | 14,475.5 | -0.9% |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 9,301.9 | 11,712.6 | +25.9% |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 40,390.9 | 41,241.8 | +2.1% |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 19,366.8 | 21,526.2 | +11.1% |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 47,269.0 | 48,600.5 | +2.8% |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 564.0 | 542.0 | -3.9% |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,227.1 | 2,229.4 | +0.1% |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 814.4 | 836.6 | +2.7% |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 3,933.7 | 3,994.4 | +1.5% |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,190.2 | 2,607.5 | +19.1% |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,484.7 | 6,494.3 | +0.1% |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,259.8 | 5,876.9 | +11.7% |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 6,719.9 | 5,957.1 | -11.4% |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 316.1 | 304.4 | -3.7% |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,394.6 | 1,393.8 | -0.1% |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 457.7 | 461.8 | +0.9% |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,470.3 | 2,483.7 | +0.5% |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,216.1 | 1,435.1 | +18.0% |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 3,057.3 | 3,004.0 | -1.7% |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 2,559.4 | 2,767.9 | +8.1% |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 2,850.1 | 2,741.3 | -3.8% |
