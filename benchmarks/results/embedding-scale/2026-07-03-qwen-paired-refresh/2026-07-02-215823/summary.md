# Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-lm`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | mlx-lm tok/s | AX tok/s | AX vs mlx-lm | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 47,240.8 | 48,362.7 | +2.4% | 188.9 | 42.3 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 49,789.8 | 50,065.0 | +0.6% | 195.6 | 171.4 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 49,960.4 | 49,840.7 | -0.2% | 194.7 | 355.3 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 48,264.2 | 48,781.2 | +1.1% | 95.3 | 83.6 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 48,908.3 | 48,940.9 | +0.1% | 95.6 | 351.4 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 48,870.2 | 48,773.0 | -0.2% | 95.3 | 702.7 |
| qwen3-embedding-4b-4bit-dwq | 256 | 8 | 64 | 5,941.3 | 5,938.4 | -0.0% | 23.2 | 376.9 |
| qwen3-embedding-4b-4bit-dwq | 256 | 32 | 16 | 6,205.0 | 6,136.6 | -1.1% | 24.0 | 1,374.9 |
| qwen3-embedding-4b-4bit-dwq | 256 | 64 | 8 | 6,248.8 | 6,233.0 | -0.3% | 24.3 | 2,673.0 |
| qwen3-embedding-4b-4bit-dwq | 512 | 8 | 64 | 6,284.5 | 6,255.9 | -0.5% | 12.2 | 694.7 |
| qwen3-embedding-4b-4bit-dwq | 512 | 32 | 16 | 6,232.7 | 6,243.8 | +0.2% | 12.2 | 2,728.3 |
| qwen3-embedding-4b-4bit-dwq | 512 | 64 | 8 | 6,256.3 | 6,251.4 | -0.1% | 12.2 | 5,405.6 |
| qwen3-embedding-8b-4bit-dwq | 256 | 8 | 64 | 3,433.7 | 3,438.3 | +0.1% | 13.4 | 635.4 |
| qwen3-embedding-8b-4bit-dwq | 256 | 32 | 16 | 3,389.2 | 3,382.1 | -0.2% | 13.2 | 2,509.7 |
| qwen3-embedding-8b-4bit-dwq | 256 | 64 | 8 | 3,396.6 | 3,390.3 | -0.2% | 13.2 | 5,001.9 |
| qwen3-embedding-8b-4bit-dwq | 512 | 8 | 64 | 3,345.0 | 3,346.3 | +0.0% | 6.5 | 1,277.0 |
| qwen3-embedding-8b-4bit-dwq | 512 | 32 | 16 | 3,322.2 | 3,314.8 | -0.2% | 6.5 | 5,058.9 |
| qwen3-embedding-8b-4bit-dwq | 512 | 64 | 8 | 3,323.7 | 3,322.5 | -0.0% | 6.5 | 10,145.6 |

