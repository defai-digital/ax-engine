# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 48,592.3 | 189.8 | 41.9 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 49,711.9 | 194.2 | 172.7 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 49,492.4 | 193.3 | 355.2 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 48,083.5 | 93.9 | 87.4 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 48,499.4 | 94.7 | 352.5 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 48,352.7 | 94.4 | 710.6 |
| qwen3-embedding-4b-4bit-dwq | 256 | 8 | 64 | 6,671.1 | 26.1 | 321.8 |
| qwen3-embedding-4b-4bit-dwq | 256 | 32 | 16 | 6,610.5 | 25.8 | 1,299.0 |
| qwen3-embedding-4b-4bit-dwq | 256 | 64 | 8 | 6,582.3 | 25.7 | 2,609.5 |
| qwen3-embedding-4b-4bit-dwq | 512 | 8 | 64 | 6,446.1 | 12.6 | 654.6 |
| qwen3-embedding-4b-4bit-dwq | 512 | 32 | 16 | 6,345.6 | 12.4 | 2,659.9 |
| qwen3-embedding-4b-4bit-dwq | 512 | 64 | 8 | 6,364.4 | 12.4 | 5,296.4 |
| qwen3-embedding-8b-4bit-dwq | 256 | 8 | 64 | 3,537.6 | 13.8 | 597.7 |
| qwen3-embedding-8b-4bit-dwq | 256 | 32 | 16 | 3,460.1 | 13.5 | 2,446.3 |
| qwen3-embedding-8b-4bit-dwq | 256 | 64 | 8 | 3,468.2 | 13.5 | 4,870.6 |
| qwen3-embedding-8b-4bit-dwq | 512 | 8 | 64 | 3,419.3 | 6.7 | 1,213.2 |
| qwen3-embedding-8b-4bit-dwq | 512 | 32 | 16 | 3,404.1 | 6.6 | 4,895.9 |
| qwen3-embedding-8b-4bit-dwq | 512 | 64 | 8 | 3,421.6 | 6.7 | 9,762.5 |
