# Apple-to-Apple Benchmark

- Label: `qwen3-32b`
- Model: `models/Qwen3-32B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 162.3 | 18.5 |
| llama.cpp | 100.9 | 9.6 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 160.9% |
| AX / llama decode | 193.1% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260401-121626-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260401-121626-001/llama/summary.json`
