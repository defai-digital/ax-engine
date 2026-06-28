# Fair Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`.

| Model | Workload | Batch | Max tokens | mlx-lm tok/s | AX tok/s | AX vs mlx-lm |
|---|---|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | 1,691.7 | 1,548.3 | -8.5% |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | 6,289.2 | 8,083.2 | +28.5% |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | 3,012.2 | 3,164.2 | +5.0% |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | 14,794.1 | 15,789.8 | +6.7% |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | 9,328.7 | 10,187.7 | +9.2% |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | 40,653.5 | 30,262.1 | -25.6% |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | 19,707.5 | 18,401.5 | -6.6% |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | 48,703.1 | 29,916.4 | -38.6% |
| qwen3-embedding-4b-4bit-dwq | short_query_b1 | 1 | 10 | 464.6 | 552.5 | +18.9% |
| qwen3-embedding-4b-4bit-dwq | short_query_b8 | 8 | 15 | 2,240.5 | 2,497.8 | +11.5% |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b1 | 1 | 16 | 827.3 | 847.5 | +2.4% |
| qwen3-embedding-4b-4bit-dwq | fixed_16_b8 | 8 | 16 | 4,046.7 | 4,429.6 | +9.5% |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b1 | 1 | 64 | 2,212.0 | 2,644.6 | +19.6% |
| qwen3-embedding-4b-4bit-dwq | fixed_64_b8 | 8 | 64 | 6,254.9 | 6,048.3 | -3.3% |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b1 | 1 | 256 | 5,189.4 | 5,688.7 | +9.6% |
| qwen3-embedding-4b-4bit-dwq | fixed_256_b8 | 8 | 256 | 5,997.0 | 5,527.7 | -7.8% |
| qwen3-embedding-8b-4bit-dwq | short_query_b1 | 1 | 10 | 324.1 | 307.0 | -5.3% |
| qwen3-embedding-8b-4bit-dwq | short_query_b8 | 8 | 15 | 1,323.6 | 1,443.3 | +9.0% |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b1 | 1 | 16 | 468.1 | 472.8 | +1.0% |
| qwen3-embedding-8b-4bit-dwq | fixed_16_b8 | 8 | 16 | 2,266.6 | 2,477.9 | +9.3% |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b1 | 1 | 64 | 1,181.7 | 1,367.3 | +15.7% |
| qwen3-embedding-8b-4bit-dwq | fixed_64_b8 | 8 | 64 | 2,752.5 | 2,698.2 | -2.0% |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b1 | 1 | 256 | 2,354.2 | 2,562.2 | +8.8% |
| qwen3-embedding-8b-4bit-dwq | fixed_256_b8 | 8 | 256 | 2,717.5 | 2,760.8 | +1.6% |
