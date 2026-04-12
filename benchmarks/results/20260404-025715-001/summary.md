# AX Engine Benchmark (AX-only)

- Label: `Qwen3.5-35B-A3B-Q4_K_M`
- Model: `models/Qwen3.5-35B-A3B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 6.7 |
| Decode | 7.0 |

- Prefill plan: `mode=serial reason=cpu_backend`
- Decode plan: `sync=sequential scratch=cpu`
- Prefill CBs: `0.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260404-025715-001/ax.json`
