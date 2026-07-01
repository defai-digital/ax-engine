# Qwen3.6 MTP Benchmark Matrix Summary

| Target | Suite | Engine | Decode | Prefill | TTFT | Accept | Status |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3.6 27B 4-bit | `flappy` | `ax_engine` | 58.9 tok/s | 682.1 tok/s | 471 ms | 100.0% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `mtplx` | 63.6 tok/s | 691.8 tok/s | 484 ms | 100.0% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `lightning_mlx` | 9.0 tok/s | 17.9 tok/s | 17,788 ms | 93.3% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 4-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `ax_engine` | 40.9 tok/s | 656.4 tok/s | 490 ms | 100.0% | ok |
| Qwen3.6 27B 6-bit | `flappy` | `mtplx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `ax_engine` | 169.4 tok/s | 1,873.8 tok/s | 172 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `mtplx` | 133.8 tok/s | 1,653.0 tok/s | 195 ms | 95.7% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `lightning_mlx` | 19.6 tok/s | 37.3 tok/s | 8,570 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `ax_engine` | 137.5 tok/s | 1,618.0 tok/s | 204 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `mtplx` | 119.0 tok/s | 1,394.4 tok/s | 244 ms | 96.7% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `lightning_mlx` | 16.0 tok/s | 26.2 tok/s | 10,639 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |

Notes:

- AX rows are pure MTP and fail summary generation if n-gram telemetry is non-zero.
- MTPLX prefill and TTFT are derived from `prompt_eval_time_s` in the MTPLX runner.
- Lightning prefill is approximate (`prompt_tokens / client TTFT`) and includes local HTTP overhead.
- Unsupported peer lanes are listed in `plan.md` with the exact support reason.
