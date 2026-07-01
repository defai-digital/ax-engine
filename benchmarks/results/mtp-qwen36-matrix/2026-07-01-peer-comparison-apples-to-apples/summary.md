# Qwen3.6 MTP Benchmark Matrix Summary

| Target | Suite | Engine | Decode | Prefill | TTFT | Accept | Status |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3.6 27B 4-bit | `flappy` | `ax_engine` | 61.0 tok/s | 812.3 tok/s | 396 ms | 100.0% | ok |
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
- AX 27B 4-bit row uses strict MTP verify (`AX_MLX_MTP_OPTIMISTIC=0`); older stitched AX rows keep their source-run mode.
- AX 27B 4-bit seed: `44`; other stitched rows keep their source-run seed policy.

**Measurement scope (TTFT / prefill):**

- AX `ttft_ms` / `prefill_tok_s`: measured inside the MLX runner (excludes HTTP/SSE overhead). `client_wall_ttft_ms` is also recorded for cross-engine parity.
- MTPLX: derived from server-side `prompt_eval_time_s`.
- Lightning: client-observed HTTP stream TTFT (includes local HTTP overhead).
- **Only `decode_tok_s` is measured at the same scope across all engines.** Cross-engine prefill/TTFT comparisons should use `client_wall_ttft_ms` where available.

**MTP head provenance:**

- `ax_engine`: ax-local/Qwen3.6-27B-MTP sidecar (MTP precision: bf16 (extracted with RMSNorm +1.0 delta correction), draft LM head: bf16 (matching base))
- `lightning_mlx`: Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed (MTP precision: INT4 prequantized sidecar (mtp/weights.safetensors), draft LM head: 3-bit affine, group_size=64)
- `mtplx`: Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed (MTP precision: INT4 prequantized sidecar (mtp/weights.safetensors), draft LM head: 3-bit affine, group_size=64)

- Different MTP head artifacts across engines means this is a **production-configuration comparison**, not an apples-to-apples MTP weight test.
- Degeneracy gate: rejects runs where a consecutive repeating token cycle (length ≤8) covers ≥50% of output tokens, or a phase-aligned periodic cycle covers ≥45%.
- Unsupported peer lanes are listed in `plan.md` with the exact support reason.
