# Qwen family peer campaign on M5 Max: 35B-A3B 6-bit, 27B MXFP4, Flash Next

Apple M5 Max, 128 GB (`df-macbookpro-m5`, Mac17,6, macOS 27.0 / 26A428 —
campaign host, not the Mac mini M4 Pro 64 GB SKU; not a certification run).
AX Engine **7.5.3** @ `c5d22c4a` built on host with Homebrew cargo 1.98.0
(no rustup; the 1.97.1 pin is not honored on this host). MTPLX **2.11.2**
@ `71f6877d`, OMLX **0.6.4**, mlx-lm **0.31.3**.

Contract per model: repository `flappy` suite (4 cases, 264–432 prompt
tokens), 256 generated tokens, greedy (seed 0), 2 warmups + 5 measured reps,
3 s cooldown, prefix cache off, n-gram stacking off. Decode/prefill are the
**median of 20 measured runs**. Same checkpoint directory offered to every
runtime; a lane that cannot load it is **unsupported**, not swapped.
Measured-lane window: 2026-09-21T07:59:41Z .. 08:20:57Z, sequential lanes
(ax → mtplx → omlx → mlx-lm); the Flash Next attempts ran separately
08:24:38Z .. 08:55:28Z (see its section). Companion 27B 6-bit campaign
(same host, same day, including rotated `long_code`; source of the 6-bit
comparison baselines quoted below): `../2026-09-21-apple-m5-max-128gb/`.

## Measured (decode tok/s, median of 20)

### AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP @ `6a4c2207`

| Runtime | Decode | Prefill |
| --- | ---: | ---: |
| AX Engine (MTP `mtp_head_only_verify_loop`) | **239.52** | **2112.1** |
| MTPLX (sustained, depth 3) | 127.87 | 1590.7 |
| mlx-lm (direct AR) | 109.97 | — |
| OMLX | unsupported — raw pack lacks `mtp.*` tensors for the Lightning loader | — |

AX leads MTPLX **1.87×** on decode and 1.33× on prefill for this MoE
(active-3B) pack.

### AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP @ `b2c5354f`

| Runtime | Decode | Prefill |
| --- | ---: | ---: |
| AX Engine (MTP `mtp_head_only_verify_loop`) | **76.70** | **792.4** |
| MTPLX (sustained, depth 3) | 63.91 | 697.3 |
| mlx-lm (direct AR) | 34.20 | — |
| OMLX | unsupported — same missing `mtp.*` tensors | — |

MTPLX drops 73.07 → 63.91 (−12.5%) going 6-bit → MXFP4 while AX stays
76.04 → 76.70 (flat). AX-over-MTPLX margin widens to **1.20×** on MXFP4.

## Flash Next (`qwen4_exp`) — experimental path evidence only

AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP @ `0b0bf6c1` (ADR-030 target pack;
admission requires `AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1`, fail-closed).

- The dedicated native trunk **loads and serves** on the target SKU: warm
  restart ready in 10 s; one greedy `/v1/chat/completions` returned
  "Hello!" (HTTP 200, `flashnext_serve_probe.json`).
- The 20-run bench was **not completed**: the server stayed on the direct
  path (`policy=direct_no_ngram_acceleration` — MTP-S/P/D gates are not
  assessed for this experimental trunk and were not forced) at roughly
  2 tok/s decode; abandoned after 30 min. **Not a performance claim.**
- Peers: mlx-lm `ValueError: Model type qwen4_exp not supported`;
  MTPLX inspect `recognized=True, supported=False` (no `mtplx_runtime.json`
  contract, exit 3); OMLX parsed settings but engine init was not
  attempted. See `flashnext_probe_*.txt`.

## Disclosures

- Fixed lane order per model (thermal drift may bias later lanes; the
  companion campaign rotated order for its second suite).
- Prefill: AX `prefill_tok_s` vs MTPLX `prompt_tokens / prompt_eval_time_s`.
- n=20 runs are correlated within prompt (4 cases × 5 reps); per-case
  medians are in `summary.json`.
- MTPLX used `--allow-unverified-model` (AXQ packs fail its inspect guard).
  OMLX lanes could be enabled later via an imported sidecar snapshot as was
  done for 27B 6-bit.

Raw: `q36_35b_{ax,mtplx,mlx_lm}.json`, `q38_27b_mxfp4_{ax,mtplx,mlx_lm}.json`,
`flashnext_serve_probe.json`, `flashnext_probe_{mlxlm,mtplx,omlx}.txt`,
`summary.json`.
