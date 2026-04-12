# AX Engine Benchmark (AX-only)

- Label: `gemma4-refactor-smoke`
- Model: `models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt: `32`
- Decode: `8` @ depth `32`
- Samples: `1`
- Cooldown: `0s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 39.0 |
| Decode | 6.8 |

- Prefill plan: `mode=cpu_batch`
- Decode plan: `sync=sequential scratch=cpu`
- Prefill CBs: `461.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-032230-001/ax.json`
