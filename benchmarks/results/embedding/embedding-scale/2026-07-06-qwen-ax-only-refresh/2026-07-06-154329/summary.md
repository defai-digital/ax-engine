# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 45,779.7 | 178.8 | 46.0 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 47,031.9 | 183.7 | 185.0 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 48,060.5 | 187.7 | 362.1 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 45,417.0 | 88.7 | 103.4 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 46,298.5 | 90.4 | 392.1 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 47,035.9 | 91.9 | 758.4 |
| qwen3-embedding-4b-4bit-dwq | 256 | 8 | 64 | 6,318.1 | 24.7 | 355.8 |
| qwen3-embedding-4b-4bit-dwq | 256 | 32 | 16 | 6,480.4 | 25.3 | 1,378.6 |
| qwen3-embedding-4b-4bit-dwq | 256 | 64 | 8 | 6,207.8 | 24.2 | 2,781.7 |
| qwen3-embedding-4b-4bit-dwq | 512 | 8 | 64 | 6,112.4 | 11.9 | 718.0 |
| qwen3-embedding-4b-4bit-dwq | 512 | 32 | 16 | 6,053.3 | 11.8 | 2,827.4 |
| qwen3-embedding-4b-4bit-dwq | 512 | 64 | 8 | 6,072.1 | 11.9 | 5,591.1 |
| qwen3-embedding-8b-4bit-dwq | 256 | 8 | 64 | 3,332.5 | 13.0 | 654.4 |
| qwen3-embedding-8b-4bit-dwq | 256 | 32 | 16 | 3,290.6 | 12.9 | 2,601.9 |
| qwen3-embedding-8b-4bit-dwq | 256 | 64 | 8 | 3,267.2 | 12.8 | 5,219.7 |
| qwen3-embedding-8b-4bit-dwq | 512 | 8 | 64 | 3,148.9 | 6.2 | 1,374.0 |
| qwen3-embedding-8b-4bit-dwq | 512 | 32 | 16 | 3,102.6 | 6.1 | 5,436.0 |
| qwen3-embedding-8b-4bit-dwq | 512 | 64 | 8 | 2,889.4 | 5.6 | 13,116.8 |
