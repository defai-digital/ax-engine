# MTP Prefill Rate and TTFT Report

## Notes

- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).
- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).
- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).
  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.

## Prefill Rate (tok/s, higher is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.1.6 | AX+ngram v5.1.6 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 670.2 | 683.3 | 685.2 |
| Qwen3.6 27B 4-bit | long_code | 785.2 | 789.8 | 789.5 |
| Qwen3.6 27B 4-bit | python_modules_long | 670.4 | 691.8 | 693.0 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1504.3 | 1806.3 | 1813.7 |
| Qwen3.6 35B-A3B 4-bit | long_code | 2107.3 | 2571.8 | 2571.2 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1289.8 | 1970.0 | 1997.0 |

\* approx: Lightning prefill = prompt\_tokens / ttft\_s (includes HTTP overhead)

## TTFT (ms, lower is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.1.6 | AX+ngram v5.1.6 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 478.5 | 470.5 | 469.2 |
| Qwen3.6 27B 4-bit | long_code | 913.8 | 908.4 | 908.8 |
| Qwen3.6 27B 4-bit | python_modules_long | 518.9 | 505.3 | 505.1 |
| Qwen3.6 35B-A3B 4-bit | flappy | 213.8 | 178.1 | 177.3 |
| Qwen3.6 35B-A3B 4-bit | long_code | 336.5 | 279.0 | 279.1 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 253.9 | 174.8 | 174.3 |

\* Lightning TTFT includes local HTTP socket overhead

