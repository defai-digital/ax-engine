# Qwen 3.8 Flash Next peer snapshot (2026-10-08)

Same-pack flappy comparison of AX Engine, MTPLX, and OMLX on the pinned
MXFP4 MTP pack. This is a single-host measurement. It does not qualify
MTP-S, MTP-P, or MTP-D, and it does not change the product default.

## Contract

- Pack: `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` at
  `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` (`model_type=qwen4_exp`).
- Suite: `benchmarks/prompts/mtp-suites/flappy.jsonl` (4 cases).
- Greedy sampling, thinking off, 256 generated tokens, 2 warmups, 5 measured
  repetitions, 3 s cooldown. Published decode is the median of the 20
  measured repetitions.
- Host: Mac Studio, Apple M2 Ultra, 192 GiB unified memory, internal NVMe
  SSD, macOS 27.0.1. A virtual machine used about one full CPU core during
  the AX window (one-minute load average 3.8).
- AX Engine 7.6.0, commit `219e9c56`, release binary, rustc 1.97.1. The AX
  harness recorded MLX 0.32.3. Expert stream was Auto and was not forced.
- OMLX 0.7.0 VLM engine, MLX 0.32.2, mlx-vlm 0.7.4, `mtp_fixed_depth` 3.
  The lane read a same-byte symlink view that lists the existing
  `mtp.safetensors` keys in the index. The pack directory was not modified.
- MTPLX 2.12.2 and mlx-lm 0.31.3 did not serve the pack.

## Medians

| Runtime | Decode tok/s | Prefill tok/s | Samples |
| --- | ---: | ---: | ---: |
| AX Engine 7.6.0 | 22.20 | 325.3 | 20 |
| OMLX 0.7.0 | 33.87 | — | 20 |
| MTPLX 2.12.2 | — | — | 0 |
| mlx-lm 0.31.3 | — | — | 0 |

AX decode across the 20 repetitions was 22.12–22.37 tok/s. OMLX was
31.89–54.78 tok/s. AX prefill is the runner-internal cold-prefill figure.
OMLX does not record a comparable prefill.

## Depth

AX selected the Flash Next MTP policy at depth 1. The 20 repetitions
drafted 2,550 tokens at the first MTP position, accepted 2,535, and drafted
0 tokens at the next two positions. N-gram stacking was off. The route label
in `ax_engine.json` is `mtp_head_only_verify_loop`.

OMLX was configured with `mtp_fixed_depth` 3. This artifact does not record
accepted draft depth for that lane, so the two decode medians are not a
same-width comparison.

## Unsupported lanes

MTPLX 2.12.2 stopped during load. Its `qwen4_exp` module rejected 384 PLE
n-gram shard tensors (`parameters not in model`). No tokens were served.

mlx-lm 0.31.3 has no loader for `qwen4_exp`. Weights were not loaded.

## OMLX kernel fallback

OMLX logged that its fused Qwen4 attention kernel failed because the
threadgroup size 1024 exceeds this GPU's 896-thread limit, then continued
on the MLX path.

## Files

- `summary.json` — lane status and medians.
- `ax_engine.json`, `omlx.json` — measured repetitions.
- `mtplx.json`, `mlx_lm.json` — unsupported records.
- `plan.json` — the commands that produced this directory.
- `ax_engine-prompts/` — tokenized flappy prompts for the 256-token runs.
