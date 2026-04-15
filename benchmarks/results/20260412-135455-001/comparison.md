# Apple-to-Apple Benchmark

- Label: `gemma4-26b-a4b-q6k-postpatch`
- Model: `models/gemma-4-26B-A4B-it-Q6_K.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: process-per-sample outer medians on the same machine
- KV parity: AX f16=True llama f16=True

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 917.0 | 71.3 |
| llama.cpp | 1177.6 | 65.3 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 77.9% |
| AX / llama decode | 109.3% |

<details><summary>Per-sample breakdown</summary>

| # | AX prefill | AX decode | llama prefill | llama decode |
|---|---:|---:|---:|---:|
| 1 | 892.8 | 69.7 | 1030.1 | 65.9 |
| 2 | 917.0 | 71.2 | 1198.2 | 56.7 |
| 3 | 931.6 | 71.3 | 1148.6 | 54.3 |
| 4 | 911.1 | 71.3 | 1177.6 | 65.4 |
| 5 | 925.2 | 71.4 | 1184.5 | 65.3 |
</details>

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260412-135455-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260412-135455-001/llama/summary.json`
