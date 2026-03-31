# Apple-to-Apple Benchmark

- Label: `llama3-8b`
- Model: `models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 684.8 | 59.1 |
| llama.cpp | 749.2 | 64.2 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 91.4% |
| AX / llama decode | 92.2% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-173734-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-173734-001/llama/summary.json`
