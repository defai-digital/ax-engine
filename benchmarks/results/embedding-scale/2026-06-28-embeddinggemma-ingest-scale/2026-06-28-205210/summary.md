# Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-embeddings`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | mlx-embeddings tok/s | AX tok/s | AX vs mlx-embeddings | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 129,187.4 | 136,140.6 | +5.4% | 531.8 | 15.5 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 133,946.8 | 140,517.5 | +4.9% | 548.9 | 59.0 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 126,773.6 | 133,858.5 | +5.6% | 522.9 | 124.0 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 111,179.5 | 117,103.7 | +5.3% | 228.7 | 35.9 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 112,664.7 | 115,471.6 | +2.5% | 225.5 | 142.9 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 102,050.3 | 116,658.4 | +14.3% | 227.8 | 287.1 |
