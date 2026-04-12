# AX Engine Benchmark (AX-only)

- Label: `Qwen3-Coder-30B-A3B-Q5_K_M`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 6.1 |
| Decode | 4.1 |

- Prefill plan: `mode=serial reason=cpu_kv`
- Decode plan: `sync=sequential scratch=cpu`
- Prefill CBs: `24817.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260410-134820-001/ax.json`
