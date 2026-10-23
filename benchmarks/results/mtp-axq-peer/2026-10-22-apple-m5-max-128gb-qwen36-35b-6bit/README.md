# Qwen 3.6 35B-A3B AXQ 6-bit MTP peer campaign (2026-10-22)

Apple M5 Max, 128 GB unified memory, internal SSD, macOS 27.0.1 (26A434), AC power,
no thermal or performance warning recorded. Campaign host, signal only: not a
certification run and not the Mac mini M4 Pro SKU.

This record replaces the 2026-09-21 Qwen 3.6 35B-A3B 6-bit rows (MTPLX 127.87 tok/s,
OMLX unsupported). Analysis: [docs/mtp/qwen36-35b-6bit-peer-m5-2026-10-22.md](../../../../docs/mtp/qwen36-35b-6bit-peer-m5-2026-10-22.md).

## Setup

| Runtime | Version | Notes |
| --- | --- | --- |
| AX Engine | 7.8.0, branch commit `77242419` (unmerged when recorded; `main` `db2a8b92` plus the DFlash2 branch, which cannot engage on a Qwen 3.6 pack) | release build, cargo 1.99.1 (Homebrew), MLX 0.32.4 |
| MTPLX | 2.12.2 (latest release, 2026-10-03), `sustained` profile | its own venv: MLX 0.32.2, mlx-lm 0.31.3 |
| OMLX | 0.7.2 (latest release, 2026-10-21), Lightning MTP | MLX 0.32.4, mlx-vlm 0.7.8 |
| mlx-lm | 0.32.2 (latest release, 2026-10-16), direct autoregressive | MLX 0.32.4 |

Pack: `AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP` @
`6a4c220734f81112555ee8783d91e0065c54301c` (source `Qwen/Qwen3.6-35B-A3B`; AXQ mixed
precision, 24.3% of parameters at 6 bit and 68.7% at 4 bit; bf16 MTP head). No Qwen 3.6
27B AXQ MTP pack exists, so there is no 27B row.

Every runtime read the same weights. AX, MTPLX and mlx-lm read the stock pack. OMLX read
the variant `axquant export-runtime --target omlx --source-trunk-norm-layout
mlx_multiplier` builds (profile `qwen35-mtp-v1`): the five trunk shards are byte-identical
to the stock pack (SHA-256 checked); the seven MTP-head RMSNorm gains are shifted from raw
HF deltas to MLX multipliers and the head is indexed as `model-mtp.safetensors`; the vision
sidecar is left out (text only). OMLX repairs only four of the seven head norms itself, so
the stock pack gave 3.8% draft acceptance and 86.5 wall tok/s on `flappy`; the variant
gives 97.6% and the rates below.

Contract: `flappy` (4 prompts, 273 to 432 tokens) and `long_code` (4 prompts, 451 to 852
tokens) suites, 256 generated tokens, greedy, seed 0, thinking off through each runtime's
chat template, 2 warmups plus 5 measured repetitions per prompt, 3 s cooldown, prefix cache
off. Identical prompt token counts in every lane. Each lane waited for three consecutive
CPU-idle samples of at least 94%. Pass A ran every lane forward, pass B the default lanes
in reverse order, pass C the controls below. Lanes ran one at a time.

## Results (decode tok/s, prefill tok/s, median of 20)

Decode is the rate after the first token. Prefill is the engine-reported prompt rate where
the runtime reports one (AX runner, MTPLX `prompt_eval_time_s`, mlx-lm `prompt_tps`) with
the client-side rate (prompt tokens over time to first token) in brackets; OMLX reports
none, so only the client rate exists.

### flappy

| Runtime | Decode | Prefill | TTFT ms | End-to-end tok/s |
| --- | ---: | ---: | ---: | ---: |
| **AX Engine MTP (default)** | **250.6** | **2364** (2337) | **137** | **219.2** |
| AX Engine MTP, exact profile | 250.6 | 2394 (2366) | 136 | 220.6 |
| AX Engine direct | 122.4 | 2199 (2176) | 147 | 114.2 |
| MTPLX depth 3 / 2 / 1 | 202.7 / 187.0 / 158.4 | 1765 / 1748 / 1742 | 185 | 177.3 / 164.1 / 141.5 |
| OMLX depth 3 / 2 / 1 | 207.1 / 191.7 / 172.2 | (1193) | 301 | 168.5 / 158.3 / 145.7 |
| mlx-lm | 113.9 | 1242 (1086) | 298 | 100.9 |

### long_code

| Runtime | Decode | Prefill | TTFT ms | End-to-end tok/s |
| --- | ---: | ---: | ---: | ---: |
| AX Engine MTP (default) | 242.8 | **3550** (3524) | **204** | 203.8 |
| AX Engine MTP, exact profile | 242.8 | 3570 (3543) | 203 | 203.7 |
| AX Engine direct | 121.6 | 3515 (3489) | 205 | 110.5 |
| MTPLX depth 3 / **2** / 1 | 247.6 / **255.5** / 223.8 | 3136 / 3147 / 3128 | 229 | 201.0 / **205.9** / 184.7 |
| OMLX depth 3 / 2 / 1 | 240.1 / 230.7 / 209.4 | (1963) | 366 | 188.9 / 183.7 / 169.3 |
| mlx-lm | 113.1 | 2129 (2005) | 358 | 98.0 |

AX Engine at fixed depth 2 / 1: 215.0 / 167.5 (`flappy`) and 209.3 / 164.3 (`long_code`).
Decode over all eight prompts (harmonic mean of per-prompt medians): AX 245.8, MTPLX 228.3
(depth 3), OMLX 223.7 (depth 3), mlx-lm 113.5, AX direct 122.0.

Pass B (reverse order) agrees with pass A to within 0.3% on every default lane. Text
agreement with AX direct decode (pass B, 4 prompts per suite, 256 tokens, exact match
after trimming): AX 3/4 and 2/4, mlx-lm 3/4 and 2/4, MTPLX 3/4 and 2/4, OMLX 3/4 and 3/4.

### Prompt-copy control (pass C)

MTPLX runs a second draft source by default: when the generated tail matches an n-gram of
the prompt, it proposes the prompt's continuation as one verified block and skips the MTP
head for that cycle (`mtplx/context_copy.py`; off with `MTPLX_CONTEXT_COPY=0`).

| Decode tok/s | copy on (default) | copy off |
| --- | ---: | ---: |
| MTPLX depth 3, `long_code` | 247.6 | 211.1 |
| MTPLX depth 3, `long_code_test_stubs` | 347.4 | 207.9 |
| MTPLX depth 3, `flappy` | 203.2 | 214.9 |

With copy on, MTPLX needed 12 MTP-head cycles for 256 tokens on `long_code_test_stubs`
(pure MTP at depth 3 needs 64); with it off 64 to 66 cycles on all eight prompts. AX Engine
with n-gram stacking enabled drafted zero n-gram tokens on this hybrid MoE target (policy
`ngram_acceleration_linear_attention_branch_recompute`, outcome `no_draft_fallback`) and ran
at the pure-MTP rate (251.1 and 242.9). Pure MTP head against MTPLX with copy off, decode:
AX 1.17x (`flappy`) and 1.15x (`long_code`).

## Limitations

- Single host. TTFT scopes differ: AX through its HTTP server, MTPLX from
  `prompt_eval_time_s`, OMLX and mlx-lm in process.
- MTPLX ran with `--allow-unverified-model` (`mtplx inspect` reports `can_run: true`,
  `runtime_compatibility: native`; the pack lacks MTPLX's verification receipt) and uses
  its own MLX 0.32.2.
- All runtimes ran text only. Text agreement covers 4 prompts per suite.
- `brew upgrade omlx` also upgraded `z3`, `llvm` and `rust` on the host; AX was built with
  the resulting cargo 1.99.1.

## Files

- `summary.json`: every lane, ratios, pass B repeat, text agreement, pass C controls.
- `lanes/`: one reduced record per lane and pass (per-repetition decode, prefill, TTFT,
  end-to-end, draft acceptance, text SHA-256).
- `harness/`: the lane runners (`fair_campaign.sh`, `fair_campaign_b.sh`,
  `fair_campaign_c.sh`), the host-local mlx-lm runner (`bench_mlx_lm_prefill.py`), and the
  reducers (`summarize_fair.py`, `reduce_lanes.py`, `build_summary.py`). The AX lanes use
  `scripts/bench_mlx_inference_stack.py`, the MTPLX lanes `scripts/bench_mtplx_prompt_suites.py`,
  the OMLX lanes `harness/bench_omlx_prefill.py` (the streaming version of
  `scripts/bench_omlx_prompt_suites.py` that the repository script now contains: prefill
  and decode recorded separately).
