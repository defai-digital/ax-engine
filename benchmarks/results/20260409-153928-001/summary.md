# AX Engine Benchmark (AX-only)

- Label: `Gemma4-26B-A4B-Q5_K_M`
- Model: `models/gemma-4-26B-A4B-it-Q5_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 18.2 |
| Decode | 11.9 |

- Prefill plan: `mode=serial reason=cpu_kv`
- Decode plan: `sync=sequential scratch=cpu`
- Prefill CBs: `64512.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-153928-001/ax.json`
