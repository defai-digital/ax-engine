# Gemma family on M5 Max: native-path peers + pack-availability finding

Apple M5 Max, 128 GB (`df-macbookpro-m5`, Mac17,6, macOS 27.0 / 26A428 —
campaign host, not the Mac mini M4 Pro 64 GB SKU; not a certification run).
AX Engine **7.5.3** @ `c5d22c4a` (built on host, Homebrew cargo 1.98.0 —
no rustup, pin not honored), OMLX **0.6.4**, MTPLX **2.11.2**,
mlx-lm **0.31.3**. Contract: `flappy` suite, 256 tokens, greedy seed 0,
2 warmups + 5 measured reps, 3 s cooldown; decode/prefill are the median of
20 measured runs. Window 2026-09-21T09:35:05Z .. 09:48:41Z, sequential
lanes (ax → mtplx → omlx → mlx-lm).

## Availability finding (read first)

The alias catalog pins AX Gemma 4 chat packs
(`AutomatosX/AX-gemma-4-{12b,26b-a4b,31b}-MLX-AXQ-6bit-MTP`, revisions
`7ad79df2` / `940a60b1` / `7b11bd51`) but **all three return 404 on
huggingface.co with an org-member token**, and the authenticated org
listing (31 repos) shows `AX-EmbeddingGemma-300M-MLX-8bit` as the only
Gemma repo — consistent with unpublished or gated beyond this token's
access. Chat lanes
therefore ran on the public mlx-community 4-bit checkpoints (same
directory offered to every runtime). Community packs carry **no Gemma
Assistant-MTP sidecar**, so every chat number below is **direct
autoregressive** — native-graph load/decode evidence, not an MTP or
product claim. (Side note: `ax-engine download` honors
`AX_ENGINE_PYTHON`; the host default python3 had a broken certifi.)

## Measured (decode tok/s, median of 20)

### mlx-community/gemma-4-12B-it-4bit @ `73bcf090` (gemma4_unified)

| Runtime | Decode | Prefill |
| --- | ---: | ---: |
| AX Engine native (direct AR) | **67.68** | **1626.8** |
| OMLX | 60.05 | — |
| MTPLX | unsupported — `Model type gemma4_unified not supported` | — |
| mlx-lm 0.31.3 | unsupported — no `mlx_lm.models.gemma4_unified` module | — |

Native load test: server ready ~10 s, HTTP 200 chat reply
(`g12_native_serve_probe.json`). AX leads OMLX **1.13×**.

### mlx-community/gemma-4-26b-a4b-it-4bit @ `0d77464e` (plain `gemma4` text MoE)

| Runtime | Decode | Prefill |
| --- | ---: | ---: |
| AX Engine native (direct AR) | **142.48** | **2644.9** |
| mlx-lm 0.31.3 (direct AR) | 134.06 | — |
| OMLX | 112.23 | — |
| MTPLX | unsupported — `generate_mtpk requires an MTP-enabled runtime` | — |

AX leads OMLX **1.27×** on the MoE (A4B) checkpoint; prefill 2645 tok/s.
The 1.06× gap over mlx-lm is within the fixed-lane-order spread — treat as
parity, not a lead. The mlx-lm lane asymmetry vs the 12B section is
architecture, not version: this pack is plain `gemma4` (supported by
mlx-lm 0.31.3), while the 12B pack is the `gemma4_unified` multimodal
variant (not present in 0.31.3).

### AutomatosX/AX-EmbeddingGemma-300M-MLX-8bit @ `2dfd7474`

Serve smoke test passed: `POST /v1/embeddings` HTTP 200, **embed_dim 768**,
server ready ~10 s (`emb_smoke.json`). The fair in-process peer bench
(`bench_embedding_fair.py`) was not run: it needs the `ax_engine` Python
extension (maturin develop), not installed on the campaign host.

## Disclosures

- Fixed lane order per model; thermal drift may bias later lanes.
- Prefill is AX `prefill_tok_s`; other lanes are decode-only harnesses.
- n=20 runs correlated within prompt (4 cases × 5 reps); per-case medians
  are in `summary.json`.
- Community 4-bit checkpoints are not AXQ packs; headline AXQ-vs-peer
  comparisons remain in the Qwen campaigns (`..`, `../2026-09-21-apple-m5-max-128gb-qwen-family/`).

Raw: `g12c_ax.json`, `g12c_omlx.json`, `g26c_ax.json`, `g26c_omlx.json`,
`g26c_mlx_lm.json`, `g12_native_serve_probe.json`, `emb_smoke.json`,
`gemma_lane_errors.txt`, `summary.json`.
