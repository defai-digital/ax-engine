# Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-lm`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | mlx-lm tok/s | AX tok/s | AX vs mlx-lm | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 48,901.2 | 49,144.7 | +0.5% | 192.0 | 41.5 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 49,988.0 | 50,397.4 | +0.8% | 196.9 | 169.0 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 49,878.6 | 50,155.1 | +0.6% | 195.9 | 351.6 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 48,920.9 | 48,995.6 | +0.2% | 95.7 | 83.5 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 49,090.2 | 49,163.3 | +0.1% | 96.0 | 349.5 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 48,650.8 | 48,987.8 | +0.7% | 95.7 | 702.9 |
| qwen3-embedding-4b-4bit-dwq | 256 | 8 | 64 | 6,591.1 | 6,714.6 | +1.9% | 26.2 | 319.2 |
| qwen3-embedding-4b-4bit-dwq | 256 | 32 | 16 | 6,463.0 | 6,571.6 | +1.7% | 25.7 | 1,274.2 |
| qwen3-embedding-4b-4bit-dwq | 256 | 64 | 8 | 6,471.9 | 6,600.8 | +2.0% | 25.8 | 2,533.7 |
| qwen3-embedding-4b-4bit-dwq | 512 | 8 | 64 | 6,416.5 | 6,535.2 | +1.8% | 12.8 | 645.1 |
| qwen3-embedding-4b-4bit-dwq | 512 | 32 | 16 | 6,138.0 | 6,051.7 | -1.4% | 11.8 | 2,817.0 |
| qwen3-embedding-4b-4bit-dwq | 512 | 64 | 8 | 6,307.6 | 6,416.6 | +1.7% | 12.5 | 5,240.9 |
| qwen3-embedding-8b-4bit-dwq | 256 | 8 | 64 | 3,379.7 | 3,509.3 | +3.8% | 13.7 | 605.7 |
| qwen3-embedding-8b-4bit-dwq | 256 | 32 | 16 | 3,266.7 | 3,349.9 | +2.5% | 13.1 | 2,555.7 |
| qwen3-embedding-8b-4bit-dwq | 256 | 64 | 8 | 3,359.3 | 3,442.1 | +2.5% | 13.4 | 4,915.3 |
| qwen3-embedding-8b-4bit-dwq | 512 | 8 | 64 | 3,327.1 | 3,379.4 | +1.6% | 6.6 | 1,258.1 |
| qwen3-embedding-8b-4bit-dwq | 512 | 32 | 16 | 3,260.3 | 3,269.3 | +0.3% | 6.4 | 5,259.1 |
| qwen3-embedding-8b-4bit-dwq | 512 | 64 | 8 | 3,333.9 | 3,411.3 | +2.3% | 6.7 | 9,764.6 |

