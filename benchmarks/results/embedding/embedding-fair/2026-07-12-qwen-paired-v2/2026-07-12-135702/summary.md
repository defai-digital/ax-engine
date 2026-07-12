# Fair Embedding Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-lm`, pooling: `last`.

Short-query rows headline **ms/item** (lower is better; negative % = AX faster). Fixed-length rows headline tok/s (higher is better).

| Model | Workload | Batch | Max tokens | Primary | mlx-lm | AX | AX vs mlx-lm |
|---|---|---:|---:|---|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | short_query_b1 | 1 | 10 | ms/item | 28.15 | 36.80 | +30.7% |
| qwen3-embedding-0.6b-8bit | short_query_b8 | 8 | 15 | ms/item | 5.85 | 4.17 | -28.7% |
| qwen3-embedding-0.6b-8bit | fixed_16_b1 | 1 | 16 | tok/s | 411.3 | 529.1 | +28.6% |
| qwen3-embedding-0.6b-8bit | fixed_16_b8 | 8 | 16 | tok/s | 2,352.7 | 2,858.8 | +21.5% |
| qwen3-embedding-0.6b-8bit | fixed_64_b1 | 1 | 64 | tok/s | 1,957.6 | 1,266.7 | -35.3% |
| qwen3-embedding-0.6b-8bit | fixed_64_b8 | 8 | 64 | tok/s | 11,518.1 | 9,086.8 | -21.1% |
| qwen3-embedding-0.6b-8bit | fixed_256_b1 | 1 | 256 | tok/s | 4,206.3 | 4,210.3 | +0.1% |
| qwen3-embedding-0.6b-8bit | fixed_256_b8 | 8 | 256 | tok/s | 21,139.3 | 22,830.9 | +8.0% |
