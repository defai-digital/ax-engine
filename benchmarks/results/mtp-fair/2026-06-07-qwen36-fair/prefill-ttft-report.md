# MTP Prefill Rate and TTFT Report

## Notes

- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).
- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).
- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).
  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.

## Prefill Rate (tok/s, higher is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.2.2 | AX+ngram v5.2.2 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 657.4 | 677.2 | 674.7 |
| Qwen3.6 27B 4-bit | long_code | 792.6 | 770.1 | 784.4 |
| Qwen3.6 27B 4-bit | python_modules_long | 680.2 | 688.9 | 687.5 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1520.2 | 1812.4 | 1808.9 |
| Qwen3.6 35B-A3B 4-bit | long_code | 2430.7 | 2711.5 | 2734.1 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1653.8 | 1998.5 | 1989.7 |

\* approx: Lightning prefill = prompt\_tokens / ttft\_s (includes HTTP overhead)

## TTFT (ms, lower is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.2.2 | AX+ngram v5.2.2 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 489.1 | 474.8 | 476.5 |
| Qwen3.6 27B 4-bit | long_code | 905.3 | 931.9 | 914.7 |
| Qwen3.6 27B 4-bit | python_modules_long | 508.7 | 507.7 | 508.8 |
| Qwen3.6 35B-A3B 4-bit | flappy | 212.5 | 177.5 | 177.9 |
| Qwen3.6 35B-A3B 4-bit | long_code | 295.2 | 264.6 | 262.4 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 205.9 | 172.8 | 173.0 |

\* Lightning TTFT includes local HTTP socket overhead

