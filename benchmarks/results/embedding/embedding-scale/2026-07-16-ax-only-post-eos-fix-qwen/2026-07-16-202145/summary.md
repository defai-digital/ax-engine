# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 48,465.6 | 189.3 | 42.0 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 49,657.5 | 194.0 | 172.7 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 49,529.8 | 193.5 | 352.7 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 47,805.5 | 93.4 | 88.5 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 48,265.2 | 94.3 | 356.1 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 47,876.1 | 93.5 | 712.1 |
| qwen3-embedding-4b-4bit-dwq | 256 | 8 | 64 | 6,626.1 | 25.9 | 324.5 |
| qwen3-embedding-4b-4bit-dwq | 256 | 32 | 16 | 6,582.6 | 25.7 | 1,304.8 |
| qwen3-embedding-4b-4bit-dwq | 256 | 64 | 8 | 6,560.5 | 25.6 | 2,614.0 |
| qwen3-embedding-4b-4bit-dwq | 512 | 8 | 64 | 6,416.6 | 12.5 | 658.2 |
| qwen3-embedding-4b-4bit-dwq | 512 | 32 | 16 | 6,326.5 | 12.4 | 2,667.5 |
| qwen3-embedding-4b-4bit-dwq | 512 | 64 | 8 | 6,320.2 | 12.3 | 5,400.0 |
| qwen3-embedding-8b-4bit-dwq | 256 | 8 | 64 | 3,521.0 | 13.8 | 600.6 |
| qwen3-embedding-8b-4bit-dwq | 256 | 32 | 16 | 3,442.8 | 13.4 | 2,457.2 |
| qwen3-embedding-8b-4bit-dwq | 256 | 64 | 8 | 3,452.1 | 13.5 | 4,899.8 |
| qwen3-embedding-8b-4bit-dwq | 512 | 8 | 64 | 3,410.1 | 6.7 | 1,263.6 |
| qwen3-embedding-8b-4bit-dwq | 512 | 32 | 16 | 3,351.0 | 6.5 | 5,059.7 |
| qwen3-embedding-8b-4bit-dwq | 512 | 64 | 8 | 3,359.1 | 6.6 | 10,072.3 |
