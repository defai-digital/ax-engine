# DiffusionGemma Direct First-Block Benchmark

- schema: `ax.diffusion_gemma_direct_first_block.v1`
- runtime: AX Engine native MLX direct
- model: `mlx-community/diffusiongemma-26B-A4B-it-4bit`
- method: 2 warmup + 5 measured repetitions, 15 s cooldown, median reported
- metric boundary: first committed diffusion block, not fixed-token autoregressive TTFT

| Prompt tokens | Block decode | Prefill | Time to first block | Denoise steps |
|---:|---:|---:|---:|---:|
| 128 | 35.5 tok/s | 1,340.6 tok/s | 7,299 ms | 41 |
| 512 | 24.8 tok/s | 3,000.6 tok/s | 10,498 ms | 48 |
| 2048 | 30.8 tok/s | 3,968.4 tok/s | 8,821 ms | 41 |

| Prompt tokens | Estimated effective bandwidth | % of 614.4 GB/s M5 Max theoretical bandwidth |
|---:|---:|---:|
| 128 | 96.5 GB/s | 15.7% |
| 512 | 78.5 GB/s | 12.8% |
| 2048 | 83.7 GB/s | 13.6% |

Effective bandwidth estimates use local safetensors bytes times `denoise_steps + 1 commit` per block, divided by measured block wall time, against the 614.4 GB/s M5 Max theoretical unified-memory bandwidth.

Peer runtimes are intentionally N/A: current llama.cpp and mlx-lm releases cannot load DiffusionGemma model artifacts.
