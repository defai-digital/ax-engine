# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `128`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 16 | 46,135.8 | 180.2 | 44.9 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 4 | 47,605.0 | 186.0 | 172.9 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 2 | 43,987.4 | 171.8 | 373.2 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 16 | 42,350.1 | 82.7 | 97.6 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 4 | 40,618.6 | 79.3 | 404.7 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 2 | 40,401.2 | 78.9 | 812.3 |
