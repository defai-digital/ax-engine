# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 48,840.0 | 190.8 | 41.8 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 49,982.6 | 195.2 | 174.9 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 49,829.5 | 194.6 | 352.0 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 47,222.1 | 92.2 | 91.1 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 47,633.0 | 93.0 | 377.0 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 47,989.9 | 93.7 | 726.4 |
| qwen3-embedding-4b-4bit-dwq | 256 | 8 | 64 | 6,476.5 | 25.3 | 352.0 |
| qwen3-embedding-4b-4bit-dwq | 256 | 32 | 16 | 6,286.7 | 24.6 | 1,380.0 |
| qwen3-embedding-4b-4bit-dwq | 256 | 64 | 8 | 6,420.9 | 25.1 | 2,715.1 |
| qwen3-embedding-4b-4bit-dwq | 512 | 8 | 64 | 6,407.4 | 12.5 | 669.2 |
| qwen3-embedding-4b-4bit-dwq | 512 | 32 | 16 | 6,324.9 | 12.4 | 2,715.2 |
| qwen3-embedding-4b-4bit-dwq | 512 | 64 | 8 | 6,298.8 | 12.3 | 5,421.1 |
| qwen3-embedding-8b-4bit-dwq | 256 | 8 | 64 | 3,451.9 | 13.5 | 631.6 |
| qwen3-embedding-8b-4bit-dwq | 256 | 32 | 16 | 3,366.1 | 13.1 | 2,553.9 |
| qwen3-embedding-8b-4bit-dwq | 256 | 64 | 8 | 3,408.6 | 13.3 | 4,962.8 |
| qwen3-embedding-8b-4bit-dwq | 512 | 8 | 64 | 3,365.4 | 6.6 | 1,266.9 |
| qwen3-embedding-8b-4bit-dwq | 512 | 32 | 16 | 3,326.0 | 6.5 | 5,039.5 |
| qwen3-embedding-8b-4bit-dwq | 512 | 64 | 8 | 3,344.9 | 6.5 | 9,985.0 |
