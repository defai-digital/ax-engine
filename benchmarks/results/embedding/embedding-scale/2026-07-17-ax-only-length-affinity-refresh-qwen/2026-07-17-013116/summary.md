# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 48,558.1 | 189.7 | 42.0 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 49,637.8 | 193.9 | 172.8 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 49,443.3 | 193.1 | 353.4 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 47,493.2 | 92.8 | 89.4 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 47,838.0 | 93.4 | 361.8 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 48,100.9 | 93.9 | 711.5 |
| qwen3-embedding-4b-4bit-dwq | 256 | 8 | 64 | 6,628.2 | 25.9 | 323.0 |
| qwen3-embedding-4b-4bit-dwq | 256 | 32 | 16 | 6,590.4 | 25.7 | 1,304.6 |
| qwen3-embedding-4b-4bit-dwq | 256 | 64 | 8 | 6,579.4 | 25.7 | 2,610.8 |
| qwen3-embedding-4b-4bit-dwq | 512 | 8 | 64 | 6,429.8 | 12.6 | 656.3 |
| qwen3-embedding-4b-4bit-dwq | 512 | 32 | 16 | 6,265.9 | 12.2 | 2,645.7 |
| qwen3-embedding-4b-4bit-dwq | 512 | 64 | 8 | 6,322.1 | 12.3 | 5,275.2 |
| qwen3-embedding-8b-4bit-dwq | 256 | 8 | 64 | 3,519.1 | 13.7 | 593.7 |
| qwen3-embedding-8b-4bit-dwq | 256 | 32 | 16 | 3,466.5 | 13.5 | 2,426.8 |
| qwen3-embedding-8b-4bit-dwq | 256 | 64 | 8 | 3,487.8 | 13.6 | 4,842.5 |
| qwen3-embedding-8b-4bit-dwq | 512 | 8 | 64 | 3,442.9 | 6.7 | 1,213.4 |
| qwen3-embedding-8b-4bit-dwq | 512 | 32 | 16 | 3,376.7 | 6.6 | 4,900.6 |
| qwen3-embedding-8b-4bit-dwq | 512 | 64 | 8 | 3,393.6 | 6.6 | 9,785.6 |
