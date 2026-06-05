# MTP Prefill Rate and TTFT Report

## Notes

- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).
- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).
- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).
  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.

## Prefill Rate (tok/s, higher is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.2.2 | AX+ngram v5.2.2 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 682.6 | 685.7 | 685.8 |
| Qwen3.6 27B 4-bit | long_code | 797.7 | 791.1 | 766.1 |
| Qwen3.6 27B 4-bit | python_modules_long | 690.7 | 694.9 | 693.9 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1544.7 | 1820.9 | 1838.9 |
| Qwen3.6 35B-A3B 4-bit | long_code | 2286.9 | 2730.3 | 2724.7 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1430.6 | 2013.1 | 2008.9 |

\* approx: Lightning prefill = prompt\_tokens / ttft\_s (includes HTTP overhead)

## TTFT (ms, lower is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.2.2 | AX+ngram v5.2.2 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 471.0 | 468.9 | 468.8 |
| Qwen3.6 27B 4-bit | long_code | 899.5 | 907.0 | 937.7 |
| Qwen3.6 27B 4-bit | python_modules_long | 503.2 | 503.6 | 504.2 |
| Qwen3.6 35B-A3B 4-bit | flappy | 208.4 | 176.6 | 174.9 |
| Qwen3.6 35B-A3B 4-bit | long_code | 314.3 | 262.8 | 263.3 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 229.2 | 171.9 | 171.6 |

\* Lightning TTFT includes local HTTP socket overhead

