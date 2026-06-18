# DiffusionGemma Direct First-Block Benchmark

- schema: `ax.diffusion_gemma_direct_first_block.v1`
- runtime: AX Engine native MLX direct
- model: `mlx-community/diffusiongemma-26B-A4B-it-4bit`
- method: 2 warmup + 5 measure runs, median reported
- metric boundary: first committed diffusion block, not fixed-token autoregressive TTFT

| Prompt tokens | Block decode | Prefill | Time to first block | Denoise steps |
|---:|---:|---:|---:|---:|
| 128 | 45.2 tok/s | 1,348.4 tok/s | 5,757 ms | 48 |
| 512 | 42.9 tok/s | 3,005.3 tok/s | 6,138 ms | 48 |
| 2048 | 45.6 tok/s | 3,978.1 tok/s | 6,132 ms | 48 |

| Prompt tokens | Estimated effective bandwidth | % of 614.4 GB/s M5 Max theoretical bandwidth |
|---:|---:|---:|
| 128 | 143.2 GB/s | 23.3% |
| 512 | 135.8 GB/s | 22.1% |
| 2048 | 144.5 GB/s | 23.5% |

Effective bandwidth estimates use local safetensors bytes times `denoise_steps + 1 commit` per block, divided by measured block wall time, against the 614.4 GB/s M5 Max theoretical unified-memory bandwidth.

Peer runtimes are intentionally N/A: current llama.cpp and mlx-lm releases cannot load DiffusionGemma model artifacts.
