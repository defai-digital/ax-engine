# Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-lm`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | mlx-lm tok/s | AX tok/s | AX vs mlx-lm | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 47,716.0 | 47,164.0 | -1.2% | 184.2 | 44.3 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 50,071.0 | 49,193.1 | -1.8% | 192.2 | 175.9 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 50,040.0 | 49,059.5 | -2.0% | 191.6 | 353.0 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 47,620.7 | 47,734.7 | +0.2% | 93.2 | 89.6 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 48,888.9 | 48,071.6 | -1.7% | 93.9 | 353.9 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 48,688.0 | 48,268.4 | -0.9% | 94.3 | 706.4 |
| qwen3-embedding-4b-4bit-dwq | 256 | 8 | 64 | 6,369.7 | 6,291.6 | -1.2% | 24.6 | 354.0 |
| qwen3-embedding-4b-4bit-dwq | 256 | 32 | 16 | 6,218.2 | 6,126.9 | -1.5% | 23.9 | 1,416.2 |
| qwen3-embedding-4b-4bit-dwq | 256 | 64 | 8 | 6,377.4 | 6,343.2 | -0.5% | 24.8 | 2,696.5 |
| qwen3-embedding-4b-4bit-dwq | 512 | 8 | 64 | 6,415.5 | 6,370.3 | -0.7% | 12.4 | 703.7 |
| qwen3-embedding-4b-4bit-dwq | 512 | 32 | 16 | 6,323.6 | 6,315.3 | -0.1% | 12.3 | 2,703.7 |
| qwen3-embedding-4b-4bit-dwq | 512 | 64 | 8 | 6,352.8 | 6,337.8 | -0.2% | 12.4 | 5,344.3 |
| qwen3-embedding-8b-4bit-dwq | 256 | 8 | 64 | 3,489.0 | 3,486.5 | -0.1% | 13.6 | 609.2 |
| qwen3-embedding-8b-4bit-dwq | 256 | 32 | 16 | 3,438.5 | 3,434.6 | -0.1% | 13.4 | 2,507.2 |
| qwen3-embedding-8b-4bit-dwq | 256 | 64 | 8 | 3,441.7 | 3,441.5 | -0.0% | 13.4 | 4,928.3 |
| qwen3-embedding-8b-4bit-dwq | 512 | 8 | 64 | 3,397.3 | 3,398.1 | +0.0% | 6.6 | 1,270.9 |
| qwen3-embedding-8b-4bit-dwq | 512 | 32 | 16 | 3,343.7 | 3,354.2 | +0.3% | 6.6 | 5,034.3 |
| qwen3-embedding-8b-4bit-dwq | 512 | 64 | 8 | 3,373.2 | 3,358.1 | -0.4% | 6.6 | 10,017.3 |

