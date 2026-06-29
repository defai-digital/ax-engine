# Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-embeddings`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | mlx-embeddings tok/s | AX tok/s | AX vs mlx-embeddings | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 128,386.2 | 137,137.2 | +6.8% | 535.7 | 15.5 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 135,730.6 | 144,754.6 | +6.6% | 565.4 | 57.6 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 130,460.9 | 140,801.9 | +7.9% | 550.0 | 117.5 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 118,497.0 | 124,725.3 | +5.3% | 243.6 | 33.1 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 115,078.1 | 124,465.6 | +8.2% | 243.1 | 136.7 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 114,877.3 | 120,037.7 | +4.5% | 234.4 | 295.1 |
