# Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-lm`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | mlx-lm tok/s | AX tok/s | AX vs mlx-lm | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 49,103.2 | 49,329.4 | +0.5% | 192.7 | 41.1 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 50,394.0 | 50,486.2 | +0.2% | 197.2 | 169.5 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 50,176.2 | 50,330.6 | +0.3% | 196.6 | 346.7 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 49,111.6 | 49,122.3 | +0.0% | 95.9 | 82.9 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 49,076.4 | 49,233.5 | +0.3% | 96.2 | 345.0 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 49,093.8 | 49,261.7 | +0.3% | 96.2 | 693.1 |
| qwen3-embedding-4b-4bit-dwq | 256 | 8 | 64 | 6,128.6 | 6,193.5 | +1.1% | 24.2 | 361.4 |
| qwen3-embedding-4b-4bit-dwq | 256 | 32 | 16 | 6,330.7 | 6,405.3 | +1.2% | 25.0 | 1,359.1 |
| qwen3-embedding-4b-4bit-dwq | 256 | 64 | 8 | 6,429.8 | 6,591.8 | +2.5% | 25.7 | 2,574.7 |
| qwen3-embedding-4b-4bit-dwq | 512 | 8 | 64 | 6,331.5 | 6,480.6 | +2.4% | 12.7 | 682.7 |
| qwen3-embedding-4b-4bit-dwq | 512 | 32 | 16 | 6,191.7 | 6,322.0 | +2.1% | 12.3 | 2,679.5 |
| qwen3-embedding-4b-4bit-dwq | 512 | 64 | 8 | 6,268.8 | 6,393.3 | +2.0% | 12.5 | 5,215.2 |
| qwen3-embedding-8b-4bit-dwq | 256 | 8 | 64 | 3,462.4 | 3,575.5 | +3.3% | 14.0 | 605.3 |
| qwen3-embedding-8b-4bit-dwq | 256 | 32 | 16 | 3,454.9 | 3,551.6 | +2.8% | 13.9 | 2,432.9 |
| qwen3-embedding-8b-4bit-dwq | 256 | 64 | 8 | 3,426.5 | 3,499.6 | +2.1% | 13.7 | 4,773.5 |
| qwen3-embedding-8b-4bit-dwq | 512 | 8 | 64 | 3,395.1 | 3,474.2 | +2.3% | 6.8 | 1,240.7 |
| qwen3-embedding-8b-4bit-dwq | 512 | 32 | 16 | 3,368.3 | 3,443.5 | +2.2% | 6.7 | 4,890.1 |
| qwen3-embedding-8b-4bit-dwq | 512 | 64 | 8 | 3,387.2 | 3,473.7 | +2.6% | 6.8 | 9,595.4 |
