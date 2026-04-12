# AX Engine Benchmark (AX-only)

- Label: `gemma4-31b-q4km`
- Model: `/Users/akiralam/code/ax-engine/models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt: `32`
- Decode: `8` @ depth `32`
- Samples: `1`
- Cooldown: `0s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 6.0 |
| Decode | 6.2 |

- Prefill plan: `mode=serial reason=cpu_kv`
- Decode plan: `sync=sequential scratch=cpu`
- Prefill CBs: `8032.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-022114-001/ax.json`
