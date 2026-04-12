# Apple-to-Apple Benchmark

- Label: `gemma4-26b-a4b-q6k`
- Model: `models/gemma-4-26B-A4B-it-Q6_K.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: process-per-sample outer medians on the same machine
- KV parity: AX f16=True llama f16=True

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 971.5 | 68.4 |
| llama.cpp | 1177.6 | 65.3 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 82.5% |
| AX / llama decode | 104.8% |

<details><summary>Per-sample breakdown</summary>

| # | AX prefill | AX decode | llama prefill | llama decode |
|---|---:|---:|---:|---:|
| 1 | 62.1 | 68.3 | 1030.1 | 65.9 |
| 2 | 971.5 | 68.4 | 1198.2 | 56.7 |
| 3 | 989.5 | 68.8 | 1148.6 | 54.3 |
| 4 | 966.5 | 69.0 | 1177.6 | 65.4 |
| 5 | 974.0 | 68.3 | 1184.5 | 65.3 |
</details>

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260412-125555-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260412-125555-001/llama/summary.json`
