# DiffusionGemma Direct First-Block Benchmark

- schema: `ax.diffusion_gemma_direct_first_block.v1`
- runtime: AX Engine native MLX direct
- model: `mlx-community/diffusiongemma-26B-A4B-it-4bit`
- method: 1 warmup + 5 measured repetitions, 15 s cooldown, median reported
- metric boundary: first committed diffusion block, not fixed-token autoregressive TTFT

| Prompt tokens | Block decode | Prefill | Time to first block | Denoise steps |
|---:|---:|---:|---:|---:|
| 128 | 2127.7 tok/s | 1,377.3 tok/s | 212 ms | 1 |
| 512 | 2166.7 tok/s | 3,043.0 tok/s | 286 ms | 1 |
| 2048 | 2118.8 tok/s | 4,073.3 tok/s | 624 ms | 1 |

| Prompt tokens | Estimated effective bandwidth | % of 614.4 GB/s M5 Max theoretical bandwidth |
|---:|---:|---:|
| 128 | 275.0 GB/s | 44.8% |
| 512 | 280.0 GB/s | 45.6% |
| 2048 | 273.8 GB/s | 44.6% |

Effective bandwidth estimates use local safetensors bytes times `denoise_steps + 1 commit` per block, divided by measured block wall time, against the 614.4 GB/s M5 Max theoretical unified-memory bandwidth.

Peer runtimes are intentionally N/A: current llama.cpp and mlx-lm releases cannot load DiffusionGemma model artifacts.
