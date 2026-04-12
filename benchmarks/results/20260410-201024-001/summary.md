# AX Engine Benchmark (AX-only)

- Label: `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 7.4 |
| Decode | 5.2 |

- Prefill plan: `mode=serial reason=cpu_kv`
- Decode plan: `sync=sequential scratch=cpu`
- Prefill CBs: `24817.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260410-201024-001/ax.json`
