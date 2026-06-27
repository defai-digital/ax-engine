# DiffusionGemma Direct First-Block Benchmark

- schema: `ax.diffusion_gemma_direct_first_block.v1`
- runtime: AX Engine native MLX direct
- model: `mlx-community/diffusiongemma-26B-A4B-it-4bit`
- method: 2 warmup + 5 measured repetitions, 15 s cooldown, median reported
- metric boundary: first committed diffusion block, not fixed-token autoregressive TTFT

| Prompt tokens | Block decode | Prefill | Time to first block | Denoise steps |
|---:|---:|---:|---:|---:|
| 128 | 109.8 tok/s | 1,142.8 tok/s | 2,445 ms | 13 |
| 512 | 83.2 tok/s | 2,707.9 tok/s | 3,266 ms | 17 |
| 2048 | 103.9 tok/s | 3,834.4 tok/s | 2,999 ms | 13 |

| Prompt tokens | Estimated effective bandwidth | % of 614.4 GB/s M5 Max theoretical bandwidth |
|---:|---:|---:|
| 128 | 99.4 GB/s | 16.2% |
| 512 | 96.8 GB/s | 15.7% |
| 2048 | 94.0 GB/s | 15.3% |

Effective bandwidth estimates use local safetensors bytes times `denoise_steps + 1 commit` per block, divided by measured block wall time, against the 614.4 GB/s M5 Max theoretical unified-memory bandwidth.

Peer runtimes are intentionally N/A: current llama.cpp and mlx-lm releases cannot load DiffusionGemma model artifacts.
