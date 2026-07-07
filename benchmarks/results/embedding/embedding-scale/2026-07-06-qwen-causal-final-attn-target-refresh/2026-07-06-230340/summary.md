# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 49,602.3 | 96.9 | 348.8 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 49,742.4 | 97.2 | 689.8 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 49,367.6 | 96.4 | 83.7 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 50,773.2 | 198.3 | 174.1 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 50,715.4 | 198.1 | 357.6 |
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 48,940.1 | 191.2 | 40.7 |
| qwen3-embedding-4b-4bit-dwq | 512 | 32 | 16 | 6,432.4 | 12.6 | 2,690.4 |
| qwen3-embedding-4b-4bit-dwq | 512 | 64 | 8 | 6,480.7 | 12.7 | 5,229.3 |
| qwen3-embedding-4b-4bit-dwq | 512 | 8 | 64 | 6,529.4 | 12.8 | 646.3 |
| qwen3-embedding-4b-4bit-dwq | 256 | 32 | 16 | 6,721.9 | 26.3 | 1,274.8 |
| qwen3-embedding-4b-4bit-dwq | 256 | 64 | 8 | 6,710.9 | 26.2 | 2,551.1 |
| qwen3-embedding-4b-4bit-dwq | 256 | 8 | 64 | 6,776.1 | 26.5 | 314.6 |
| qwen3-embedding-8b-4bit-dwq | 512 | 32 | 16 | 3,423.8 | 6.7 | 4,957.2 |
| qwen3-embedding-8b-4bit-dwq | 512 | 64 | 8 | 3,437.2 | 6.7 | 9,843.0 |
| qwen3-embedding-8b-4bit-dwq | 512 | 8 | 64 | 3,462.3 | 6.8 | 1,239.3 |
| qwen3-embedding-8b-4bit-dwq | 256 | 32 | 16 | 3,518.7 | 13.7 | 2,403.4 |
| qwen3-embedding-8b-4bit-dwq | 256 | 64 | 8 | 3,408.2 | 13.3 | 4,961.4 |
| qwen3-embedding-8b-4bit-dwq | 256 | 8 | 64 | 3,458.5 | 13.5 | 611.5 |
