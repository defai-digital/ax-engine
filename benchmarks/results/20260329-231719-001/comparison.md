# Apple-to-Apple Benchmark

- Label: `prefill-pipe-off`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 321.7 | 53.6 |
| llama.cpp | 708.9 | 48.1 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 45.4% |
| AX / llama decode | 111.5% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-231719-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-231719-001/llama/summary.json`
