# Qwen / Gemma M5 Max peer campaign — 2026-09-21 (AX Engine 7.5.3)

> Dated campaign record on one Apple M5 Max 128 GB host (Mac17,6, macOS 27.0).
> This is a clean campaign snapshot, not a universal
> engine ranking, not SKU certification (the 27B qualification SKU is the
> Mac mini M4 Pro 64 GB), and not a product-claim page. Canonical status
> sentences live in [Qwen 3.8 27B AXQ](../model-certifications/qwen3.8-27b-axq.md).

Checked-in artifacts:
[`2026-09-21-apple-m5-max-128gb/`](../../benchmarks/results/mtp-axq-peer/2026-09-21-apple-m5-max-128gb/)
(27B AXQ 6-bit, flappy + long_code),
[`2026-09-21-apple-m5-max-128gb-qwen-family/`](../../benchmarks/results/mtp-axq-peer/2026-09-21-apple-m5-max-128gb-qwen-family/),
[`2026-09-21-apple-m5-max-128gb-gemma/`](../../benchmarks/results/mtp-axq-peer/2026-09-21-apple-m5-max-128gb-gemma/).

Measured identities: AX Engine 7.5.3 (`c5d22c4a`, built on host with Homebrew
cargo 1.98.0 — the host has no rustup, so the workspace 1.97.1 pin is not
honored there; campaign evidence only), MTPLX 2.11.2 (`71f6877d`), OMLX 0.6.4,
mlx-lm 0.31.3. Contract per suite: 256 generated tokens, greedy seed 0, two
warmups, five measured reps per case, 3 s cooldown, prefix cache off, n-gram
stacking off; decode/prefill are the median of 20 measured runs
(4 cases x 5 reps). Lanes ran sequentially, one process at a time; the
`long_code` suite rotated lane order to spread thermal drift. Qwen MTP rows
ran the validated exact-verifier profile (`--ax-ngram-accel
--ax-mtp-disable-ngram-stacking --ax-qwen-linear-mtp-exact`; effective route
`mtp_head_only_verify_loop`, `ax_qwen_linear_mtp_exact=true` +
`..._explicit_enable=true` in artifacts).

## Results — Qwen (MTP product path)

| Model (suite) | AX 7.5.3 | MTPLX | OMLX | mlx-lm (direct AR) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8 27B AXQ 6-bit MTP (flappy) | **76.04** / 768.7 | 73.07 / 650.6 | 37.71 | 27.92 |
| Qwen3.8 27B AXQ 6-bit MTP (long_code) | **72.71** / 856.9 | 57.82 / 833.1 | 35.85 | 27.86 |
| Qwen3.6 35B-A3B AXQ 6-bit MTP (flappy) | **239.52** / 2112.1 | 127.87 / 1590.7 | unsupported | 109.97 |
| Qwen3.8 27B AXQ MXFP4 MTP (flappy) | **76.70** / 792.4 | 63.91 / 697.3 | unsupported | 34.20 |

Decode tok/s / prefill tok/s, median of 20. Prefill scopes differ (AX
runner-internal `prefill_tok_s`; MTPLX derived from `prompt_eval_time_s`).

Movement vs the 2026-09-15 7.4.0 flappy baseline: AX 76.90 → 76.04 (−1.1%),
MTPLX 70.62 → 73.07 (+3.5%) at unchanged versions — the dense-27B short-suite
margin (1.041x) is inside single-host run-to-run spread and should be read as
parity there; the same-day rotated `long_code` suite and the MoE / MXFP4
packs show larger AX leads (1.26x, 1.87x, 1.20x). MTPLX's per-case flappy
distribution is bimodal (one case at ~188 tok/s), so medians move more than
AX's. OMLX ran on the imported 6-bit conversion snapshot (the raw AXQ tree
fails its Lightning loader); for the 35B-A3B and MXFP4 packs no import exists
in this campaign and the lane is recorded as unsupported. MTPLX ran
`--allow-unverified-model` because its inspect guard rejects AXQ packs.

## Results — Gemma 4 (direct AR; community checkpoints)

The catalog-pinned AX Gemma 4 chat packs
(`AutomatosX/AX-gemma-4-{12b,26b-a4b,31b}-MLX-AXQ-6bit-MTP`) return 404 on
Hugging Face with an org-member token, so the chat lanes ran on public
`mlx-community` 4-bit checkpoints, which carry no Assistant-MTP sidecar.
**Every Gemma row is direct AR** — native-graph load/decode evidence, not an
MTP or product claim, and not the 6-bit recommended publication lane.

| Checkpoint | AX 7.5.3 | MTPLX | OMLX | mlx-lm |
| --- | ---: | ---: | ---: | --- |
| gemma-4-12B-it-4bit (`gemma4_unified`) | **67.68** / 1626.8 | unsupported | 60.05 | unsupported |
| gemma-4-26b-a4b-it-4bit (`gemma4`) | **142.48** / 2644.9 | unsupported | 112.23 | 134.06 |

mlx-lm 0.31.3 supports plain `gemma4` but lacks the `gemma4_unified` module;
MTPLX 2.11.2 rejects both. `AX-EmbeddingGemma-300M-MLX-8bit` passed a serve
smoke test (`/v1/embeddings`, dim 768); the fair in-process peer bench was
not run on this host.

## Flash Next (`qwen4_exp`)

Native-path load and serve succeeded on the ADR-030 target spec with
`AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1` (HTTP 200 chat reply; warm restart
ready ~10 s). The 20-run contract was not completed: the trunk runs direct
decode (MTP-S/P/D not assessed, not forced) at roughly 2 tok/s on the flappy
256-token workload — recorded as experimental status, explicitly not a
performance claim. None of the tested peer versions (MTPLX 2.11.2, OMLX
0.6.4, mlx-lm 0.31.3) load `qwen4_exp`.

## Limitations

- One host, one campaign window (2026-09-21T07:07:55Z .. 09:48:41Z UTC),
  sequential lanes; gaps below ~1.1x are not separable from thermal/order
  drift.
- n=20 runs are correlated within prompt (4 cases x 5 reps); per-case
  medians are in each `summary.json`.
- The binary is a 1.98.0-host build; SKU gates stay on the Mac mini M4 Pro.
- Gemma numbers use community 4-bit packs, not AXQ: they document native
  graph breadth, not the AXQ-vs-peer story.
