# Apple-to-Apple Benchmark

- Label: `qwen35-35b-a3b`
- Model: `models/Qwen3.5-35B-A3B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 5.1 | 4.0 |
| llama.cpp | 1193.7 | 55.3 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 0.4% |
| AX / llama decode | 7.2% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260401-030549-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260401-030549-001/llama/summary.json`
