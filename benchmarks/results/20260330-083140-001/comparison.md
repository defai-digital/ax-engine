# Apple-to-Apple Benchmark

- Label: `smart-barrier`
- Model: `models/gemma-3-12b-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 428.0 | 41.2 |
| llama.cpp | 497.4 | 41.0 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 86.0% |
| AX / llama decode | 100.4% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-083140-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-083140-001/llama/summary.json`
