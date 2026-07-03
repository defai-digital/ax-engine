# Qwen3.6 MTP Benchmark Matrix Summary

| Target | Suite | Engine | Decode | Prefill | TTFT | Accept | Status |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3.6 27B 4-bit | `flappy` | `ax_engine` | 60.3 tok/s | 649.2 tok/s | 495 ms | 99.5% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `mtplx` | - tok/s | - tok/s | - ms | - | error |
| Qwen3.6 27B 4-bit | `flappy` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 4-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 4-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 4-bit | `long_code` | `ax_engine` | 60.3 tok/s | 782.5 tok/s | 917 ms | 99.4% | ok |
| Qwen3.6 27B 4-bit | `long_code` | `mtplx` | - tok/s | - tok/s | - ms | - | error |
| Qwen3.6 27B 4-bit | `long_code` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 4-bit | `long_code` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 4-bit | `long_code` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 4-bit | `python_modules_long` | `ax_engine` | 49.4 tok/s | 660.7 tok/s | 522 ms | 97.2% | ok |
| Qwen3.6 27B 4-bit | `python_modules_long` | `mtplx` | - tok/s | - tok/s | - ms | - | error |
| Qwen3.6 27B 4-bit | `python_modules_long` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 4-bit | `python_modules_long` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 4-bit | `python_modules_long` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `ax_engine` | 40.5 tok/s | 621.5 tok/s | 521 ms | 99.4% | ok |
| Qwen3.6 27B 6-bit | `flappy` | `mtplx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `long_code` | `ax_engine` | 38.9 tok/s | 718.4 tok/s | 999 ms | 99.5% | ok |
| Qwen3.6 27B 6-bit | `long_code` | `mtplx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `long_code` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `long_code` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `long_code` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `python_modules_long` | `ax_engine` | 33.3 tok/s | 623.8 tok/s | 560 ms | 96.7% | ok |
| Qwen3.6 27B 6-bit | `python_modules_long` | `mtplx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `python_modules_long` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `python_modules_long` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `python_modules_long` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `ax_engine` | 168.2 tok/s | 1,731.4 tok/s | 185 ms | 99.8% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `mtplx` | - tok/s | - tok/s | - ms | - | error |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `long_code` | `ax_engine` | 167.5 tok/s | 2,558.8 tok/s | 280 ms | 99.8% | ok |
| Qwen3.6 35B-A3B 4-bit | `long_code` | `mtplx` | - tok/s | - tok/s | - ms | - | error |
| Qwen3.6 35B-A3B 4-bit | `long_code` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `long_code` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `long_code` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `python_modules_long` | `ax_engine` | 160.8 tok/s | 1,843.2 tok/s | 185 ms | 98.3% | ok |
| Qwen3.6 35B-A3B 4-bit | `python_modules_long` | `mtplx` | - tok/s | - tok/s | - ms | - | error |
| Qwen3.6 35B-A3B 4-bit | `python_modules_long` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `python_modules_long` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `python_modules_long` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `ax_engine` | 134.9 tok/s | 1,491.8 tok/s | 216 ms | 99.8% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `mtplx` | - tok/s | - tok/s | - ms | - | error |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `long_code` | `ax_engine` | 133.6 tok/s | 2,281.8 tok/s | 314 ms | 99.8% | ok |
| Qwen3.6 35B-A3B 6-bit | `long_code` | `mtplx` | - tok/s | - tok/s | - ms | - | error |
| Qwen3.6 35B-A3B 6-bit | `long_code` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `long_code` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `long_code` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `python_modules_long` | `ax_engine` | 135.5 tok/s | 1,631.2 tok/s | 210 ms | 98.4% | ok |
| Qwen3.6 35B-A3B 6-bit | `python_modules_long` | `mtplx` | - tok/s | - tok/s | - ms | - | error |
| Qwen3.6 35B-A3B 6-bit | `python_modules_long` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `python_modules_long` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `python_modules_long` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |

Notes:

- AX rows are pure MTP and fail summary generation if n-gram telemetry is non-zero.
- MTPLX prefill and TTFT are derived from `prompt_eval_time_s` in the MTPLX runner.
- Lightning prefill is approximate (`prompt_tokens / client TTFT`) and includes local HTTP overhead.
- Unsupported peer lanes are listed in `plan.md` with the exact support reason.
