# Apple-to-Apple Benchmark

- Label: `gemma3-12b`
- Model: `models/gemma-3-12b-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 412.8 | 36.8 |
| llama.cpp | 477.0 | 39.3 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 86.5% |
| AX / llama decode | 93.5% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-174539-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-174539-001/llama/summary.json`
