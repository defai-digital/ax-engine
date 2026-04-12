# AX Engine Benchmark (AX-only)

- Label: `gemma4-warning-smoke`
- Model: `models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt: `8`
- Decode: `1` @ depth `8`
- Samples: `1`
- Cooldown: `0s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 8.6 |
| Decode | 5.5 |

- Prefill plan: `mode=cpu_batch`
- Decode plan: `sync=sequential scratch=cpu`
- Prefill CBs: `461.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-030213-001/ax.json`
