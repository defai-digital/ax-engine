# Qwen3.6 MTP Benchmark Matrix Summary

| Target | Suite | Engine | Decode | Prefill | TTFT | Accept | Status |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3.6 27B 4-bit | `flappy` | `ax_engine` | 64.6 tok/s | 809.0 tok/s | 398 ms | 100.0% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `mtplx` | 64.3 tok/s | 681.4 tok/s | 470 ms | 100.0% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `lightning_mlx` | 59.4 tok/s | 426.0 tok/s | 784 ms | 95.9% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 4-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `ax_engine` | 41.4 tok/s | 637.1 tok/s | 507 ms | 100.0% | ok |
| Qwen3.6 27B 6-bit | `flappy` | `mtplx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `ax_engine` | 166.3 tok/s | 1,755.3 tok/s | 184 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `mtplx` | 138.1 tok/s | 1,637.0 tok/s | 193 ms | 95.7% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `lightning_mlx` | 116.2 tok/s | 1,466.5 tok/s | 215 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `ax_engine` | 141.8 tok/s | 1,536.0 tok/s | 209 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `mtplx` | 117.6 tok/s | 1,383.9 tok/s | 235 ms | 96.7% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `lightning_mlx` | 96.3 tok/s | 1,215.8 tok/s | 272 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |

Notes:

- AX rows are pure MTP and fail summary generation if n-gram telemetry is non-zero.
- MTPLX prefill and TTFT are derived from `prompt_eval_time_s` in the MTPLX runner.
- Lightning prefill is approximate (`prompt_tokens / client TTFT`) and includes local HTTP overhead.
- Unsupported peer lanes are listed in `plan.md` with the exact support reason.
