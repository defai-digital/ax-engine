# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 43,240.4 | 168.9 | 48.1 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 40,270.9 | 157.3 | 208.0 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 38,928.2 | 152.1 | 426.5 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 38,344.2 | 74.9 | 108.1 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 39,339.8 | 76.8 | 418.4 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 40,584.3 | 79.3 | 809.2 |
