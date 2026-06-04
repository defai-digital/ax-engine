# MTP Prefill Rate and TTFT Report

## Notes

- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).
- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).
- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).
  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.

## Prefill Rate (tok/s, higher is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.1.12 | AX+ngram v5.1.12 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 682.6 | 682.2 | 678.8 |
| Qwen3.6 27B 4-bit | long_code | 797.7 | 787.6 | 790.4 |
| Qwen3.6 27B 4-bit | python_modules_long | 690.7 | 687.7 | 689.6 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1544.7 | 1812.6 | 1818.2 |
| Qwen3.6 35B-A3B 4-bit | long_code | 2286.9 | 2730.0 | 2706.1 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1430.6 | 2005.5 | 2000.9 |

\* approx: Lightning prefill = prompt\_tokens / ttft\_s (includes HTTP overhead)

## TTFT (ms, lower is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.1.12 | AX+ngram v5.1.12 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 471.0 | 471.3 | 473.6 |
| Qwen3.6 27B 4-bit | long_code | 899.5 | 911.0 | 907.8 |
| Qwen3.6 27B 4-bit | python_modules_long | 503.2 | 504.8 | 506.5 |
| Qwen3.6 35B-A3B 4-bit | flappy | 208.4 | 177.5 | 176.9 |
| Qwen3.6 35B-A3B 4-bit | long_code | 314.3 | 262.8 | 265.1 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 229.2 | 172.5 | 172.3 |

\* Lightning TTFT includes local HTTP socket overhead

