# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `128`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 16 | 47,500.7 | 185.5 | 43.8 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 4 | 49,178.9 | 192.1 | 167.8 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 2 | 45,790.3 | 178.9 | 358.0 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 16 | 43,410.2 | 84.8 | 95.5 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 4 | 41,399.0 | 80.9 | 397.1 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 2 | 40,965.6 | 80.0 | 801.0 |
