# MTP Prefill Rate and TTFT Report

## Notes

- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).
- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).
- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).
  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.

## Prefill Rate (tok/s, higher is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.1.6 | AX+ngram v5.1.6 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 682.6 | 683.8 | 684.5 |
| Qwen3.6 27B 4-bit | long_code | 797.7 | 790.0 | 790.3 |
| Qwen3.6 27B 4-bit | python_modules_long | 690.7 | 692.3 | 693.3 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1544.7 | 1818.2 | 1824.3 |
| Qwen3.6 35B-A3B 4-bit | long_code | 2286.9 | 2712.8 | 2714.3 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1430.6 | 2001.9 | 2004.7 |

\* approx: Lightning prefill = prompt\_tokens / ttft\_s (includes HTTP overhead)

## TTFT (ms, lower is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.1.6 | AX+ngram v5.1.6 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 471.0 | 470.2 | 469.7 |
| Qwen3.6 27B 4-bit | long_code | 899.5 | 908.2 | 907.9 |
| Qwen3.6 27B 4-bit | python_modules_long | 503.2 | 505.3 | 504.6 |
| Qwen3.6 35B-A3B 4-bit | flappy | 208.4 | 176.9 | 176.3 |
| Qwen3.6 35B-A3B 4-bit | long_code | 314.3 | 264.5 | 264.3 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 229.2 | 173.0 | 172.2 |

\* Lightning TTFT includes local HTTP socket overhead

