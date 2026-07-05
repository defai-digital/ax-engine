# DiffusionGemma Direct First-Block Benchmark

- schema: `ax.diffusion_gemma_direct_first_block.v1`
- runtime: AX Engine native MLX direct
- model: `mlx-community/diffusiongemma-26B-A4B-it-4bit`
- method: 1 warmup + 5 measured repetitions, 15 s cooldown, median reported
- metric boundary: first committed diffusion block, not fixed-token autoregressive TTFT

| Prompt tokens | Block decode | Prefill | Time to first block | Denoise steps |
|---:|---:|---:|---:|---:|
| 128 | 147.8 tok/s | 1,064.8 tok/s | 1,852 ms | 13 |
| 512 | 104.3 tok/s | 2,649.7 tok/s | 2,647 ms | 18 |
| 2048 | 140.1 tok/s | 3,874.4 tok/s | 2,357 ms | 13 |

| Prompt tokens | Estimated effective bandwidth | % of 614.4 GB/s M5 Max theoretical bandwidth |
|---:|---:|---:|
| 128 | 133.7 GB/s | 21.8% |
| 512 | 128.1 GB/s | 20.8% |
| 2048 | 126.7 GB/s | 20.6% |

Effective bandwidth estimates use local safetensors bytes times `denoise_steps + 1 commit` per block, divided by measured block wall time, against the 614.4 GB/s M5 Max theoretical unified-memory bandwidth.

Peer runtimes are intentionally N/A: current llama.cpp and mlx-lm releases cannot load DiffusionGemma model artifacts.
