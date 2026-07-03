# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `128`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 16 | 46,105.8 | 180.1 | 44.7 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 4 | 43,178.9 | 168.7 | 192.0 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 2 | 41,659.3 | 162.7 | 393.7 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 16 | 41,200.5 | 80.5 | 100.9 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 4 | 40,370.5 | 78.8 | 406.9 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 2 | 40,373.0 | 78.9 | 812.3 |
