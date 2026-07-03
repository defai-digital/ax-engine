# Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-embeddings`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | mlx-embeddings tok/s | AX tok/s | AX vs mlx-embeddings | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 129,909.0 | 133,946.4 | +3.1% | 523.2 | 14.9 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 148,284.8 | 154,116.8 | +3.9% | 602.0 | 64.5 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 149,976.1 | 154,705.6 | +3.2% | 604.3 | 136.2 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 127,604.8 | 138,150.4 | +8.3% | 269.8 | 30.6 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 140,105.8 | 147,794.6 | +5.5% | 288.7 | 123.0 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 132,121.8 | 139,131.6 | +5.3% | 271.7 | 271.9 |

