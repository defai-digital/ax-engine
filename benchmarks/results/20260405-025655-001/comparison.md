# Apple-to-Apple Benchmark

- Label: `gemma4-31b-prefill-bootstrap`
- Model: `models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `1`
- Cooldown: `0s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 19.9 | 2.5 |
| llama.cpp | 150.4 | 15.8 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 13.2% |
| AX / llama decode | 15.9% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-025655-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-025655-001/llama/summary.json`
