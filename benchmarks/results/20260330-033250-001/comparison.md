# Apple-to-Apple Benchmark

- Label: `concurrent-decode`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 430.4 | 45.5 |
| llama.cpp | 642.3 | 48.0 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 67.0% |
| AX / llama decode | 94.7% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-033250-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-033250-001/llama/summary.json`
