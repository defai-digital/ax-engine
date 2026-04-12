# Apple-to-Apple Benchmark

- Label: `gemma4-31b-prefill-bootstrap-final`
- Model: `models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `1`
- Cooldown: `0s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 20.1 | 2.5 |
| llama.cpp | 150.4 | 15.8 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 13.3% |
| AX / llama decode | 15.6% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-030504-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-030504-001/llama/summary.json`
