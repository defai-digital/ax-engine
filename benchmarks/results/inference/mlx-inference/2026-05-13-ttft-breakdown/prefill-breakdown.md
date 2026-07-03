# AX MLX Prefill Breakdown Report

This report decomposes AX MLX prefill timing from inference-stack artifacts. `llama.cpp Metal` values are shape-compatible external GGUF references when supplied; they are not prompt-hash parity evidence.

| Model | Prompt tok | AX prefill tok/s | AX/MLX | AX/llama.cpp | Prefill ms | Forward ms | Prefix cache ms | Generation state ms | Other ms | Forward % | Eval barriers | Async drains |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_6_35b_a3b_8bit | 128 | 964.6 | 2.373x | 0.580x | 406.0 | 405.9 | 0.0 | 0.1 | 0.0 | 100.0% | 3 | 0 |
| qwen3_coder_next_4bit | 128 | 740.0 | 2.357x | 0.617x | 507.4 | 507.3 | 0.1 | 0.1 | 0.0 | 100.0% | 3 | 0 |
| glm_4_7_flash_4bit | 128 | 853.1 | 1.760x | 0.624x | 451.0 | 450.9 | 0.0 | 0.1 | 0.0 | 100.0% | 3 | 0 |

## Reading Notes

- Worst AX/llama.cpp row: `qwen3_6_35b_a3b_8bit` prompt=128, 0.580x.
- `Forward ms` is the model forward plus final prefill token materialization path.
- `Prefix cache ms` covers prompt-prefix snapshot storage after forward.
- `Generation state ms` covers decode-state initialization after a completing prefill.
- Non-forward overhead that is high at 128 tokens is serving-path overhead, not a tensor-kernel bottleneck by itself.
