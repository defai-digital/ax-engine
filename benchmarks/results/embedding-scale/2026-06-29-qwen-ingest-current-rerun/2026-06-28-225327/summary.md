# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 44,232.5 | 172.8 | 47.6 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 41,677.8 | 162.8 | 201.4 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 39,527.5 | 154.4 | 417.2 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 38,143.7 | 74.5 | 111.3 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 39,447.0 | 77.0 | 418.9 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 40,103.5 | 78.3 | 853.7 |
