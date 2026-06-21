# MTP Prefill Rate and TTFT Report

## Notes

- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).
- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).
- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).
  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.

## Prefill Rate (tok/s, higher is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.2.2 | AX+ngram v5.2.2 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 657.4 | 677.7 | 683.4 |
| Qwen3.6 27B 4-bit | long_code | 792.6 | 789.4 | 789.6 |
| Qwen3.6 27B 4-bit | python_modules_long | 680.2 | 692.1 | 692.5 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1520.2 | 1795.1 | 1802.7 |
| Qwen3.6 35B-A3B 4-bit | long_code | 2430.7 | 2672.7 | 2706.2 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1653.8 | 1973.5 | 1935.2 |

\* approx: Lightning prefill = prompt\_tokens / ttft\_s (includes HTTP overhead)

## TTFT (ms, lower is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.2.2 | AX+ngram v5.2.2 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 489.1 | 474.4 | 470.4 |
| Qwen3.6 27B 4-bit | long_code | 905.3 | 909.0 | 908.7 |
| Qwen3.6 27B 4-bit | python_modules_long | 508.7 | 505.6 | 505.2 |
| Qwen3.6 35B-A3B 4-bit | flappy | 212.5 | 179.2 | 178.4 |
| Qwen3.6 35B-A3B 4-bit | long_code | 295.2 | 268.5 | 265.2 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 205.9 | 174.4 | 178.9 |

\* Lightning TTFT includes local HTTP socket overhead
