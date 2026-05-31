# Qwen3.6 MTP Fair Benchmark (stable profile re-run)

> **ROOT CAUSES IDENTIFIED (2026-05-30):**
>
> **A. MTPLX accept rate ~1–3% (MTP speculation disabled in practice)**
>
> Two concrete mismatches between MTPLX 0.3.7's MTP injection and the standard Qwen3.6
> training convention:
>
> 1. **Reversed fc concat order.** `MTPContract` defaults to `concat_order="embedding_hidden"`
>    (`[embed; hidden]`), but the standard model was trained with `[hidden; embed]`
>    (`hidden_embedding`, as used by Rapid-MLX and the reference mlx-lm path). The fc
>    projection receives its input halves in swapped positions → garbage MTP logits.
> 2. **Wrong hidden-state variant.** MTPLX passes `post_norm` hidden states (after the base
>    model's final RMSNorm) to the MTP head; the reference path passes `pre_norm` (before it).
>
> The Youssofal bundle is adapted to MTPLX's conventions (swapped fc halves, post-norm
> inputs), which is why MTPLX only works with that bundle. The tok/s values above reflect
> pure sequential decode throughput (speculation effectively disabled).
>
> **B. Base model corruption for MTPLX and Rapid-MLX (separate bug, fixed 2026-05-30)**
>
> `prepare_qwen36_mtp_sidecar.py` created a `model-mtp.safetensors` hard-link alias in the
> sidecar directory to support Rapid-MLX's weight naming convention. This file name matches
> mlx_lm's `model*.safetensors` glob (utils.py line 316), causing all `mtp.*` tensors to be
> loaded alongside the base model weights. `TextModel.sanitize` then sees `has_mtp_weights=True`
> and applies a +1.0 shift to all base model norm weights — but those norms are already in
> mlx_lm's shifted convention from the original quantization, producing double-shifted (corrupted)
> norms and garbage output.
>
> AX Engine was unaffected because it uses its own loader, not mlx_lm's glob. The alias has
> been removed from the sidecar directories and from `prepare_qwen36_mtp_sidecar.py`. Rapid-MLX
> disables MTP for Qwen3.6 (`supports_spec_decode=False` in `model_auto_config.py`), so the
> alias provided no benefit and only caused harm.
>
> **Rapid-MLX numbers in this table are from the corrected re-run** (after the sidecar fix,
> 2026-05-30 afternoon). The MTPLX tok/s values reflect sequential decode only and are not
> meaningfully affected by the base-model corruption (same computation, different logits).

Contract:

- models: `['27b-4bit']`
- engines: `['mtplx', 'rapid_mlx', 'ax_engine']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `1000`
- repetitions: `5`
- mtplx_profile: `stable`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown
- tri-engine Rapid comparison uses shared depth 1
- MTPLX profile: `stable` (no MTPLX_PREFILL_OMLX_EXTERNAL, no MLX fork requirement)

| Model | Suite | MTPLX tok/s | MTPLX accept | Rapid tok/s | AX tok/s | AX accept | AX/MTPLX | AX/Rapid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B 4-bit | flappy | 26.5 | 2.3% | 29.9 | 51.8 | 87.1% | 1.955 | 1.732 |
| Qwen3.6 27B 4-bit | long_code | 25.9 | 1.7% | 29.7 | 48.7 | 91.5% | 1.879 | 1.640 |
| Qwen3.6 27B 4-bit | python_modules_long | 26.2 | 1.8% | 29.6 | 45.1 | 71.2% | 1.721 | 1.524 |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-stable-profile/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / rapid_mlx: `ok` (corrected re-run, fixed sidecar) `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-stable-profile/27b-4bit/flappy/rapid_mlx.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-stable-profile/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` (all 4 cases fail `balanced_delimiters` — truncated output from near-zero accept; all pass `no_degenerate_loop`) `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-stable-profile/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / rapid_mlx: `ok` (corrected re-run, fixed sidecar) `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-stable-profile/27b-4bit/long_code/rapid_mlx.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-stable-profile/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-stable-profile/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / rapid_mlx: `ok` (corrected re-run, fixed sidecar) `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-stable-profile/27b-4bit/python_modules_long/rapid_mlx.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-stable-profile/27b-4bit/python_modules_long/ax_engine.json`
