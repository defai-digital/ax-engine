# Apple-to-Apple Benchmark

- Label: `smart-barrier`
- Model: `models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 712.2 | 61.5 |
| llama.cpp | 769.7 | 66.4 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 92.5% |
| AX / llama decode | 92.6% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-082822-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-082822-001/llama/summary.json`
