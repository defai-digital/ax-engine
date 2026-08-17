# qwen3.8-27b-axq-4bit on M3 Max — baseline measurements (2026-08-16)

Engine: dab32b41 ("Prepare v7.0.2 release") + unrelated WIP (9 files, V4/expert-stream).
Host: Apple M3 Max, 128 GB, macOS 26.6.1. Commands in COMMANDS.md.

## Decode (bench_mlx_inference_stack.py, flappy real-prompt suite, 1000 gen tokens, 5 reps + 1 warmup)

| row | decode tok/s (median per case) | notes |
|---|---|---|
| direct (`--ax-direct`) | 18.32 / 18.08 / 18.07 / 18.07 | direct_single_decode_baseline |
| MTP + unintended n-gram stacking | ~27.4 (partial) | harness gate ABORTED: 105 n-gram hit steps in a "pure MTP" row → discovered the CLI-flag bug |
| pure MTP (env `AX_MLX_MTP_DISABLE_NGRAM_STACKING=1`) | **30.35 / 30.97 / 31.72 / 32.35** | `mtp_head_only_effective`, accept EWMA ≈ 99.8% depth0, `ax_mtp_ngram_hit_steps: 0` |

- **Pure MTP = +73% vs direct** on code content. MTP is strongly net-positive on M3 Max.
- **N-gram stacking costs ~13%** (27.4 vs 31.4): it was silently active whenever
  `--mlx-mtp-disable-ngram-stacking` was passed as a CLI flag (env unset + exact ⇒ stacking
  allowed). Fix: `mtp_ngram_stacking_allowed` now takes the runner flag.
- Direct decode ceiling on M3 Max ≈ 18.1 tok/s (M5 Max documented ≈ 22.7).

## Prefill (fair_prefill_bench_probe, AX_PREFILL_QUANTUM=0, 5 reps + 1 warmup, cold prefix cache)

| config | p2048 tok/s | p10240 tok/s |
|---|---|---|
| baseline | 191.9 | 166.7 |
| `AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA=1` | 192.1 (+0.1%) | **172.0 (+3.2%)** |
| `AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL=1` | 192.8 (+0.5%) | 164.0 (-1.6%) |
| both stacked | 176.6 (-8.0%) | 165.6 (-0.7%) |
| `AX_MLX_QWEN_PREFILL_SINGLE_2048=1` | — | 165.9 (-0.5%) |
| `AX_MLX_QWEN_PREFILL_CHUNK_1536=1` | — | 148.8 (-10.7%) |

- bf16 SDPA: small but real win at agentic prompt lengths → flipped to default-on
  (kill switch `=0`). Do NOT stack with native offset causal.
- Chunk cap 1024 confirmed optimal on M3 Max too (M5 tuning transfers).

## Session-observed vs bench

The ax-code session that motivated this (ses_ff3393b9, creative writing, 10.6k prompt)
showed ~11 tok/s decode. Under the bench, short-prompt code decode is 18.1 direct /
31.4 pure-MTP. The gap is explained by: long-context decode penalty + low MTP acceptance
on free-form text (the EWMA bypass exists for this; Phase 3 lets bypassed decode drop
the exact scope to recover full direct speed) + prefill re-processing on model switch.

## AFTER (same host, changes below)

Changes: (1) `--mlx-mtp-disable-ngram-stacking` CLI flag now gates the Qwen-linear
draft loop (`mtp_ngram_stacking_allowed` takes the runner flag); (2) bf16 prefill SDPA
default-on; (3) exact-MTP de-fusion narrowed to decode shapes (seq <= 4) — fused
prefill kernels restored; (4) low-acceptance bypass drops the exact scope so fallback
decode gets fused singleton kernels.

| metric | before | after | delta |
|---|---|---|---|
| decode pure-MTP via CLI flag only | contract-aborted (105 n-gram hits) | 30.07 / 28.96 / 29.89 / 30.85 tok/s, **0 hits, `mtp_head_only_effective`** | row contract restored; +65% vs direct |
| prefill p10240 | 166.7 | **184.1** | **+10.5%** |
| prefill p2048 | 191.9 | **202.1** (re-run; an earlier back-to-back run read 171.9 — thermal noise) | **+5.3%** |

Lib tests: 1323/1323 pass. Artifacts: after-*.json in this directory.
