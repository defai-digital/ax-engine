# AX Engine Benchmark (AX-only)

- Label: `gemma4-hd512-threshold-pp32`
- Model: `models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt: `32`
- Decode: `8` @ depth `32`
- Samples: `1`
- Cooldown: `0s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 42.9 |
| Decode | 6.9 |

- Prefill plan: `mode=cpu_batch`
- Decode plan: `sync=sequential scratch=cpu`
- Prefill CBs: `461.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-033348-001/ax.json`
