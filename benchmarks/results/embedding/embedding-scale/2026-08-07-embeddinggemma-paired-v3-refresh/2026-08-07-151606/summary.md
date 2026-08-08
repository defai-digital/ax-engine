# Embedding Ingest Scale Benchmark

Output contract: `contiguous_cpu_f32_batch_hidden`. Reference: `mlx-embeddings`. Total chunks per trial: `512`.

| Model | Chunk tokens | Batch | Batches/trial | mlx-embeddings tok/s | AX tok/s | AX vs mlx-embeddings | AX chunks/s | AX p95 batch ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m-8bit | 256 | 8 | 64 | 135,576.2 | 145,045.2 | +7.0% | 566.6 | 14.0 |
| embeddinggemma-300m-8bit | 256 | 32 | 16 | 152,373.0 | 161,597.0 | +6.1% | 631.2 | 57.7 |
| embeddinggemma-300m-8bit | 256 | 64 | 8 | 147,662.9 | 161,349.9 | +9.3% | 630.3 | 122.2 |
| embeddinggemma-300m-8bit | 512 | 8 | 64 | 136,104.8 | 145,163.7 | +6.7% | 283.5 | 27.8 |
| embeddinggemma-300m-8bit | 512 | 32 | 16 | 137,755.6 | 150,555.3 | +9.3% | 294.1 | 116.9 |
| embeddinggemma-300m-8bit | 512 | 64 | 8 | 130,000.9 | 142,650.2 | +9.7% | 278.6 | 253.5 |
