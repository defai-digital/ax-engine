# MTP Prefill Rate and TTFT Report

## Notes

- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).
- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).
- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).
  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.

## Prefill Rate (tok/s, higher is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.2.2 | AX+ngram v5.2.2 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 657.4 | 681.1 | 638.5 |
| Qwen3.6 27B 4-bit | long_code | 792.6 | 768.6 | 765.0 |
| Qwen3.6 27B 4-bit | python_modules_long | 680.2 | 692.4 | 671.0 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1520.2 | 1830.6 | 1835.8 |
| Qwen3.6 35B-A3B 4-bit | long_code | 2430.7 | 2734.7 | 2706.5 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1653.8 | 1965.9 | 1966.7 |

\* approx: Lightning prefill = prompt\_tokens / ttft\_s (includes HTTP overhead)

## TTFT (ms, lower is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.2.2 | AX+ngram v5.2.2 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 489.1 | 477.4 | 503.6 |
| Qwen3.6 27B 4-bit | long_code | 905.3 | 933.6 | 937.9 |
| Qwen3.6 27B 4-bit | python_modules_long | 508.7 | 504.9 | 505.1 |
| Qwen3.6 35B-A3B 4-bit | flappy | 212.5 | 175.7 | 176.1 |
| Qwen3.6 35B-A3B 4-bit | long_code | 295.2 | 262.4 | 265.2 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 205.9 | 171.6 | 176.8 |

\* Lightning TTFT includes local HTTP socket overhead

