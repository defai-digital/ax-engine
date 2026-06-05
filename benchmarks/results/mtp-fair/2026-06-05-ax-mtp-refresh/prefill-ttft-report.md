# MTP Prefill Rate and TTFT Report

## Notes

- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).
- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).
- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).
  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.

## Prefill Rate (tok/s, higher is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.2.1 | AX+ngram v5.2.1 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 682.6 | 686.6 | 679.0 |
| Qwen3.6 27B 4-bit | long_code | 797.7 | 790.4 | 790.6 |
| Qwen3.6 27B 4-bit | python_modules_long | 690.7 | 693.5 | 691.5 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1544.7 | 1837.9 | 1833.2 |
| Qwen3.6 35B-A3B 4-bit | long_code | 2286.9 | 2735.5 | 2739.5 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1430.6 | 2014.8 | 2013.9 |

\* approx: Lightning prefill = prompt\_tokens / ttft\_s (includes HTTP overhead)

## TTFT (ms, lower is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.2.1 | AX+ngram v5.2.1 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 471.0 | 468.3 | 477.0 |
| Qwen3.6 27B 4-bit | long_code | 899.5 | 907.7 | 907.5 |
| Qwen3.6 27B 4-bit | python_modules_long | 503.2 | 504.0 | 505.3 |
| Qwen3.6 35B-A3B 4-bit | flappy | 208.4 | 175.0 | 175.4 |
| Qwen3.6 35B-A3B 4-bit | long_code | 314.3 | 262.3 | 261.9 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 229.2 | 170.8 | 171.4 |

\* Lightning TTFT includes local HTTP socket overhead

