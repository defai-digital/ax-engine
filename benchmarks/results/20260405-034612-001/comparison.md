# Apple-to-Apple Benchmark

- Label: `gemma4-readme-refresh`
- Model: `models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `1`
- Cooldown: `0s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 20.2 | 2.5 |
| llama.cpp | 150.4 | 15.8 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 13.5% |
| AX / llama decode | 16.1% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-034612-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-034612-001/llama/summary.json`
