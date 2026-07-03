# GLM-4.7 Flash MTP Validation Session

GLM-4.7 Flash uses the built-in MTP tensors from `zai-org/GLM-4.7-Flash`.
`ax-engine download-mtp glm-4.7-flash` downloads the 6-bit MLX base
(`mlx-community/GLM-4.7-Flash-6bit`), extracts the built-in MTP layer into
`glm_mtp.safetensors`, and writes a self-contained AX package.

The first local validation session used the prepared package returned by
`download-mtp` and the `flappy` real-prompt suite. This is a smoke session, not
the promoted 5-repetition MTP matrix row: it used 32 generated tokens, 1 measured
repetition, no cooldown, sampled decode (`temperature=0.6`, `top_p=0.95`,
`top_k=20`), MTP depth 1, and no MTP+n-gram stacking. The direct baseline uses
the same package and prompt suite with MTP disabled.

| Mode | Route | Decode median | Prefill median | TTFT median | MTP evidence |
|---|---|---:|---:|---:|---|
| Direct baseline | `direct_single_decode_baseline` | 58.7 tok/s | 1,670 tok/s | 166 ms | no drafts |
| GLM built-in MTP | `mtp_head_only_verify_loop` | 90.3 tok/s | 1,690 tok/s | 163 ms | 54 drafted, 46 accepted, 85.2% accept |

In this smoke session, GLM built-in MTP was **1.54x** faster than direct decode
on median decode throughput. Treat this as path validation and a same-artifact
diagnostic comparison until the full 6-bit MTP matrix is rerun with 1,000
generated tokens, 5 measured repetitions, and recorded cooldown.

Artifacts: [`flappy-after-activation-fix.json`](../../benchmarks/results/speculative/mtp-6bit/2026-06-22-glm47-flash-mtp-smoke/flappy-after-activation-fix.json)
(MTP) and [`flappy-direct-baseline.json`](../../benchmarks/results/speculative/mtp-6bit/2026-06-22-glm47-flash-mtp-smoke/flappy-direct-baseline.json)
(direct baseline).
