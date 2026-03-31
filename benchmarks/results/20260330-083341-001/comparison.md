# Apple-to-Apple Benchmark

- Label: `smart-barrier`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 437.6 | 65.0 |
| llama.cpp | 725.5 | 49.7 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 60.3% |
| AX / llama decode | 130.9% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-083341-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-083341-001/llama/summary.json`
