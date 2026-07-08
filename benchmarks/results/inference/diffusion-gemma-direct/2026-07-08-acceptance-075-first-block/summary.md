# DiffusionGemma Direct First-Block Benchmark

- schema: `ax.diffusion_gemma_direct_first_block.v1`
- runtime: AX Engine native MLX direct
- model: `mlx-community/diffusiongemma-26B-A4B-it-4bit`
- method: 1 warmup + 5 measured repetitions, 15 s cooldown, median reported
- metric boundary: first committed diffusion block, not fixed-token autoregressive TTFT

| Prompt tokens | Block decode | Prefill | Time to first block | Denoise steps |
|---:|---:|---:|---:|---:|
| 128 | 158.9 tok/s | 1,151.0 tok/s | 1,723 ms | 12 |
| 512 | 109.6 tok/s | 2,794.0 tok/s | 2,520 ms | 17 |
| 2048 | 163.5 tok/s | 3,922.3 tok/s | 2,089 ms | 11 |

| Prompt tokens | Estimated effective bandwidth | % of 614.4 GB/s M5 Max theoretical bandwidth |
|---:|---:|---:|
| 128 | 133.5 GB/s | 21.7% |
| 512 | 127.5 GB/s | 20.8% |
| 2048 | 126.8 GB/s | 20.6% |

Effective bandwidth estimates use local safetensors bytes times `denoise_steps + 1 commit` per block, divided by measured block wall time, against the 614.4 GB/s M5 Max theoretical unified-memory bandwidth.

Peer runtimes are intentionally N/A: current llama.cpp and mlx-lm releases cannot load DiffusionGemma model artifacts.
