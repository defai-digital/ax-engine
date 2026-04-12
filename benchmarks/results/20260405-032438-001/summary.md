# AX Engine Benchmark (AX-only)

- Label: `gemma4-refactor-scratch-pp512`
- Model: `models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `8` @ depth `512`
- Samples: `1`
- Cooldown: `0s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 19.3 |
| Decode | 1.8 |

- Prefill plan: `mode=cpu_batch`
- Decode plan: `sync=sequential scratch=cpu`
- Prefill CBs: `1691.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-032438-001/ax.json`
