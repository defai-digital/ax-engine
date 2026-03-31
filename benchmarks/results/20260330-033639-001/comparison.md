# Apple-to-Apple Benchmark

- Label: `concurrent-decode`
- Model: `models/gemma-3-12b-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 430.5 | 38.1 |
| llama.cpp | 495.5 | 41.1 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 86.9% |
| AX / llama decode | 92.8% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-033639-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-033639-001/llama/summary.json`
