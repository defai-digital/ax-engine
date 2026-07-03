# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `128`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 16 | 47,512.8 | 185.6 | 43.8 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 4 | 48,685.0 | 190.2 | 168.7 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 2 | 44,319.3 | 173.1 | 371.8 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 16 | 42,917.2 | 83.8 | 96.6 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 4 | 41,315.1 | 80.7 | 398.2 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 2 | 39,690.1 | 77.5 | 855.4 |
