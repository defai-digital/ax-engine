# Apple-to-Apple Benchmark

- Label: `qwen35-35b-a3b`
- Model: `models/Qwen3.5-35B-A3B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `30s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 6.1 | 6.7 |
| llama.cpp | 610.3 | 55.7 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 1.0% |
| AX / llama decode | 12.0% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260403-234721-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260403-234721-001/llama/summary.json`
