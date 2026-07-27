# AX-Only Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding-0.6b-8bit | 256 | 8 | 64 | 48,769.6 | 190.5 | 41.2 |
| qwen3-embedding-0.6b-8bit | 256 | 32 | 16 | 49,985.3 | 195.3 | 180.7 |
| qwen3-embedding-0.6b-8bit | 256 | 64 | 8 | 48,477.7 | 189.4 | 416.7 |
| qwen3-embedding-0.6b-8bit | 512 | 8 | 64 | 47,276.2 | 92.3 | 88.2 |
| qwen3-embedding-0.6b-8bit | 512 | 32 | 16 | 47,396.2 | 92.6 | 405.1 |
| qwen3-embedding-0.6b-8bit | 512 | 64 | 8 | 46,886.7 | 91.6 | 826.7 |
| qwen3-embedding-4b-4bit-dwq | 256 | 8 | 64 | 6,688.3 | 26.1 | 319.8 |
| qwen3-embedding-4b-4bit-dwq | 256 | 32 | 16 | 6,594.0 | 25.8 | 1,294.3 |
| qwen3-embedding-4b-4bit-dwq | 256 | 64 | 8 | 6,533.4 | 25.5 | 2,601.1 |
| qwen3-embedding-4b-4bit-dwq | 512 | 8 | 64 | 6,408.7 | 12.5 | 657.2 |
| qwen3-embedding-4b-4bit-dwq | 512 | 32 | 16 | 6,299.7 | 12.3 | 2,684.0 |
| qwen3-embedding-4b-4bit-dwq | 512 | 64 | 8 | 6,294.1 | 12.3 | 5,349.6 |
| qwen3-embedding-8b-4bit-dwq | 256 | 8 | 64 | 3,589.5 | 14.0 | 595.0 |
| qwen3-embedding-8b-4bit-dwq | 256 | 32 | 16 | 3,442.5 | 13.4 | 2,464.1 |
| qwen3-embedding-8b-4bit-dwq | 256 | 64 | 8 | 3,445.1 | 13.5 | 4,910.0 |
| qwen3-embedding-8b-4bit-dwq | 512 | 8 | 64 | 3,407.4 | 6.7 | 1,222.9 |
| qwen3-embedding-8b-4bit-dwq | 512 | 32 | 16 | 3,368.7 | 6.6 | 4,959.9 |
| qwen3-embedding-8b-4bit-dwq | 512 | 64 | 8 | 3,368.6 | 6.6 | 9,883.8 |

