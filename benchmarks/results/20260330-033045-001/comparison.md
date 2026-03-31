# Apple-to-Apple Benchmark

- Label: `concurrent-decode`
- Model: `models/Qwen3-8B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 501.3 | 39.4 |
| llama.cpp | 481.7 | 42.5 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 104.1% |
| AX / llama decode | 92.8% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-033045-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-033045-001/llama/summary.json`
