# Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-lm`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | mlx-lm tok/s | AX tok/s | AX vs mlx-lm | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 38,180.5 | 38,213.0 | +0.1% | 149.3 | 54.3 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 38,899.2 | 39,252.4 | +0.9% | 153.3 | 215.2 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 38,423.2 | 38,677.6 | +0.7% | 151.1 | 430.6 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 39,988.8 | 40,386.8 | +1.0% | 78.9 | 106.4 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 38,968.9 | 40,304.4 | +3.4% | 78.7 | 436.2 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 40,542.3 | 40,472.1 | -0.2% | 79.0 | 829.8 |
