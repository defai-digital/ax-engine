# AX Engine Benchmark (AX-only)

- Label: `gemma4-31b`
- Model: `models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `1`
- Cooldown: `0s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 20.3 |
| Decode | 2.1 |

- Prefill plan: `mode=cpu_batch`
- Decode plan: `sync=sequential scratch=cpu`
- Prefill CBs: `1701.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-035539-001/ax.json`
