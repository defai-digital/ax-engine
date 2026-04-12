# Apple-to-Apple Benchmark

- Label: `qwen3-coder-30b-a3b-q6_k-ax-refresh`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 27.3 | 27.4 |
| llama.cpp | 1204.9 | 79.5 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 2.3% |
| AX / llama decode | 34.4% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-162312-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-162312-001/llama/summary.json`
