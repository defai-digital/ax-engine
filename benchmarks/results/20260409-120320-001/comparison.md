# Apple-to-Apple Benchmark

- Label: `Qwen3-Coder-30B-A3B-Q4_K_M`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 7.0 | 5.2 |
| llama.cpp | 633.8 | 82.5 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 1.1% |
| AX / llama decode | 6.3% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-120320-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-120320-001/llama/summary.json`
