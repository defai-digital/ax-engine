# Apple-to-Apple Benchmark

- Label: `meta-llama-3.1-8b-instruct-q5_k_m`
- Model: `models/meta-llama-3.1-8b-instruct-q5_k_m.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 617.1 | 53.1 |
| llama.cpp | 703.3 | 56.5 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 87.8% |
| AX / llama decode | 94.0% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-083857-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-083857-001/llama/summary.json`
