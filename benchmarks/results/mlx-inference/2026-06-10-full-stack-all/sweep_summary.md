# llama.cpp Metal sweep summary

- elapsed: 12851s
- downloaded: 252.1 GB
- freed: 439.0 GB

| slug | status | repo | quant | notes |
|---|---|---|---|---|
| gemma-4-e2b-it-4bit | bench_failed | bartowski/google_gemma-4-E2B-it-GGUF | Q4_K_M | mlx_lm.benchmark failed before AX/llama.cpp artifact write; see logs. |
| gemma-4-e2b-it-5bit | bench_failed | bartowski/google_gemma-4-E2B-it-GGUF | Q5_K_M | mlx_lm.benchmark failed before AX/llama.cpp artifact write; see logs. |
| gemma-4-e2b-it-6bit | bench_failed | bartowski/google_gemma-4-E2B-it-GGUF | Q6_K | mlx_lm.benchmark failed before AX/llama.cpp artifact write; see logs. |
| gemma-4-e2b-it-8bit | bench_failed | bartowski/google_gemma-4-E2B-it-GGUF | Q8_0 | mlx_lm.benchmark failed before AX/llama.cpp artifact write; see logs. |
| gemma-4-e4b-it-4bit | bench_failed | bartowski/google_gemma-4-E4B-it-GGUF | Q4_K_M | mlx_lm.benchmark failed before AX/llama.cpp artifact write; see logs. |
| gemma-4-26b-a4b-it-4bit | ok | bartowski/google_gemma-4-26B-A4B-it-GGUF | Q4_K_M |  |
| gemma-4-26b-a4b-it-6bit | ok | bartowski/google_gemma-4-26B-A4B-it-GGUF | Q6_K |  |
| gemma-4-31b-it-4bit | ok | bartowski/google_gemma-4-31B-it-GGUF | Q4_K_M |  |
| gemma-4-31b-it-6bit | ok | bartowski/google_gemma-4-31B-it-GGUF | Q6_K |  |
| qwen3_6-27b-4bit | ok | bartowski/Qwen_Qwen3.6-27B-GGUF | Q4_K_M |  |
| qwen3_6-27b-5bit | ok | bartowski/Qwen_Qwen3.6-27B-GGUF | Q5_K_M |  |
| qwen3_6-27b-6bit | ok | bartowski/Qwen_Qwen3.6-27B-GGUF | Q6_K |  |
| qwen3_6-27b-8bit | ok | bartowski/Qwen_Qwen3.6-27B-GGUF | Q8_0 |  |
| qwen3_6-35b-a3b-4bit | ok | bartowski/Qwen_Qwen3.6-35B-A3B-GGUF | Q4_K_M |  |
| qwen3_6-35b-a3b-6bit | ok | bartowski/Qwen_Qwen3.6-35B-A3B-GGUF | Q6_K |  |
