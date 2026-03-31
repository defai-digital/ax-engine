# Apple-to-Apple Benchmark

- Label: `concurrent-decode`
- Model: `models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 721.2 | 60.9 |
| llama.cpp | 773.0 | 66.6 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 93.3% |
| AX / llama decode | 91.4% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-033447-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-033447-001/llama/summary.json`
