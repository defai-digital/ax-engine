# Qwen3.6 MTP Benchmark Matrix Summary

| Target | Suite | Engine | Decode | Prefill | TTFT | Accept | Status |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3.6 27B 4-bit | `flappy` | `mtplx` | 58.2 tok/s | 681.3 tok/s | 475 ms | 98.8% | ok |
| Qwen3.6 27B 6-bit | `flappy` | `mtplx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `mtplx` | 137.3 tok/s | 1,662.8 tok/s | 198 ms | 94.7% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `mtplx` | 118.8 tok/s | 1,453.6 tok/s | 218 ms | 95.6% | ok |

Notes:

- AX rows are pure MTP and fail summary generation if n-gram telemetry is non-zero.
- MTPLX prefill and TTFT are derived from `prompt_eval_time_s` in the MTPLX runner.
- Lightning prefill is approximate (`prompt_tokens / client TTFT`) and includes local HTTP overhead.
- AX MTP optimistic verify: OFF (full rejection sampling).
- Seed: `engine defaults` (AX defaults to seed 0; MTPLX/lightning use their runner defaults).

**Measurement scope (TTFT / prefill):**

- AX `ttft_ms` / `prefill_tok_s`: measured inside the MLX runner (excludes HTTP/SSE overhead). `client_wall_ttft_ms` is also recorded for cross-engine parity.
- MTPLX: derived from server-side `prompt_eval_time_s`.
- Lightning: client-observed HTTP stream TTFT (includes local HTTP overhead).
- **Only `decode_tok_s` is measured at the same scope across all engines.** Cross-engine prefill/TTFT comparisons should use `client_wall_ttft_ms` where available.

**MTP head provenance:**

- `27b-4bit` / `mtplx`: ax-local/Qwen3.6-27B-MTP sidecar (MTP precision: bf16 (extracted with RMSNorm +1.0 delta correction), draft LM head: bf16 (matching base))
- `35b-a3b-4bit` / `mtplx`: Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed (MTP precision: INT4 prequantized sidecar (mtp/weights.safetensors), draft LM head: 3-bit affine, group_size=64)
- `35b-a3b-6bit` / `mtplx`: Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance (MTP precision: INT4 prequantized sidecar (mtp/weights.safetensors), draft LM head: 3-bit affine, group_size=64)

- Rows with different MTP head artifacts across engines are **production-configuration comparisons**, not apples-to-apples MTP weight tests.
- Degeneracy gate: rejects runs where a consecutive repeating token cycle (length ≤8) covers ≥50% of output tokens, or a phase-aligned periodic cycle covers ≥45%.
- Unsupported peer lanes are listed in `plan.md` with the exact support reason.
