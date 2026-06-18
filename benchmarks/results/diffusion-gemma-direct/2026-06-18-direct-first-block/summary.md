# DiffusionGemma Direct First-Block Benchmark

- schema: `ax.diffusion_gemma_direct_first_block.v1`
- runtime: AX Engine native MLX direct
- model: `mlx-community/diffusiongemma-26B-A4B-it-4bit`
- method: 2 warmup + 5 measure runs, median reported
- metric boundary: first committed diffusion block, not fixed-token autoregressive TTFT

| Prompt tokens | Block decode | Prefill | Time to first block | Denoise steps |
|---:|---:|---:|---:|---:|
| 128 | 37.4 tok/s | 1,316.6 tok/s | 6,936 ms | 48 |
| 512 | 31.0 tok/s | 2,856.6 tok/s | 8,442 ms | 48 |
| 2048 | 26.8 tok/s | 3,796.4 tok/s | 10,068 ms | 48 |

| Prompt tokens | Estimated effective bandwidth | M5 Max peak utilization |
|---:|---:|---:|
| 128 | 118.5 GB/s | 20.5% |
| 512 | 98.2 GB/s | 17.0% |
| 2048 | 85.0 GB/s | 14.7% |

Bandwidth utilization estimates use local safetensors bytes times `denoise_steps + 1 commit` per block, divided by measured block wall time, against a 577 GB/s M5 Max peak.

Peer runtimes are intentionally N/A: current llama.cpp and mlx-lm releases cannot load DiffusionGemma model artifacts.
