# Qwen 3.8 27B AXQ 6-bit: v7.5.3 peer campaign (flappy + long_code)

Apple M5 Max, 128 GB (`df-macbookpro-m5`, Mac17,6, macOS 27.0 / 26A428 —
campaign host, not the Mac mini M4 Pro 64 GB SKU; not a certification run).
Pack: `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` @
`3e290738e96972307c6aeb9934ab170ca0eae1c1` — the same checkpoint directory
offered to every runtime.

Contract (identical to the 2026-09-15 baseline, extended with a second
suite): two repository prompt suites (`flappy`, 4 cases, 264–432 prompt
tokens; `long_code`, 4 cases, 451–852 prompt tokens), 256 generated tokens,
greedy (seed 0), two warmups, five measured repetitions, three-second
cooldown, prefix cache off, n-gram stacking off. Decode and prefill are the
**median of 20 measured runs** (4 cases × 5 reps).

Lane order was `ax → mtplx → omlx → mlx-lm` for `flappy` and rotated
(`omlx → mlx-lm → ax → mtplx`) for `long_code` to spread thermal drift.
Campaign window: 2026-09-21T07:07:55Z .. 07:43:08Z, sequential, one
process at a time.

## Measured (decode tok/s, median of 20)

| Runtime | Version | flappy decode | flappy prefill | long_code decode | long_code prefill |
| --- | --- | ---: | ---: | ---: | ---: |
| AX Engine | **7.5.3** @ `c5d22c4a`, MTP product path (`mtp_head_only_verify_loop`, recurrent depth 3) | **76.04** | **768.7** | **72.71** | **856.9** |
| MTPLX | **2.11.2** @ `71f6877d`, `sustained` profile, depth 3 | 73.07 | 650.6 | 57.82 | 833.1 |
| OMLX | **0.6.4**, imported snapshot + Lightning MTP depth 1 | 37.71 | — (wall tok/s) | 35.85 | — |
| mlx-lm | **0.31.3**, direct AR on the same prompts | 27.92 | — (decode-only harness) | 27.86 | — |

Extra data point: AX Engine 7.5.3 **direct AR** (no MTP, no n-gram) measured
30.14 tok/s decode / 740.6 tok/s prefill on `flappy`
(`flappy_ax_direct_ar.json`). The MTP path is 2.52× the same-binary direct
path on this host.

## Delta vs the 2026-09-15 flappy baseline

| Runtime | 2026-09-15 | 2026-09-21 | Δ |
| --- | ---: | ---: | ---: |
| AX Engine (7.4.0 → 7.5.3) | 76.90 | 76.04 | −1.1% |
| MTPLX (2.11.2, unchanged) | 70.62 | 73.07 | +3.5% |
| OMLX (0.6.4, unchanged) | 38.47 | 37.71 | −2.0% |
| mlx-lm (0.31.3, unchanged) | 27.90 | 27.92 | +0.1% |

The AX-over-MTPLX flappy margin narrowed from 1.089× to 1.041× — within
the run-to-run spread of unwarmed single-host medians; treat as parity on
`flappy`. On the longer `long_code` suite (451–852 prompt tokens) AX leads
MTPLX 1.26×, OMLX 2.03×, and direct mlx-lm 2.61×.

## Reproduction notes

- `scripts/bench_mlx_inference_stack.py` on 7.5.3 validates
  `--ax-qwen-linear-mtp-exact` only together with `--ax-ngram-accel` (or
  `--ax-compare-policies`). The effective route stays
  `mtp_head_only_verify_loop` with `ax_ngram_outcome_tier=pure_mtp`, matching
  the 2026-09-15 artifact policy `mtp_head_only_no_ngram_stacking`.
- MTPLX runs with `--allow-unverified-model` because its `inspect_model`
  guard rejects the AXQ 6-bit snapshot (same as 2026-09-15); generation
  proceeds on the same checkpoint directory.
- OMLX cannot load the raw AXQ tree with Lightning MTP and was measured on
  the imported conversion snapshot (same as 2026-09-15).
- The ax binary was built on the campaign host with Homebrew cargo 1.98.0
  (the host has no rustup, so the repo 1.97.1 pin is not honored there);
  numbers are campaign evidence, not SKU certification.

Raw: `flappy_ax.json`, `flappy_mtplx.json`, `flappy_omlx.json`,
`flappy_mlx_lm.json`, `flappy_ax_direct_ar.json`, `long_code_ax.json`,
`long_code_mtplx.json`, `long_code_omlx.json`, `long_code_mlx_lm.json`,
`summary.json`.
