# Direct-mode rebench after ADR-038 / Gemma+Diffusion phases

**Host:** Apple M5 Max 128GB, macOS 26.5.2
**Commit at run:** f2de7866 (branch adr-038-architecture-composition)
**Method:** 2 warmup + 5 measure (Gemma stack) / 1 warmup + 5 measure (Diffusion); median; 15s cooldown; AX direct (`--disable-ngram-acceleration`); no mlx_lm peer in this pass.
**Artifacts:**
- `benchmarks/results/inference/diffusion-gemma-direct/2026-07-12-adr038-post-phases/`
- `benchmarks/results/inference/mlx-inference/2026-07-12-adr038-post-phases-gemma4-direct/`

## Why rebench

Yes — public or internal gain/loss claims need fresh artifacts. Recent work touches:
- DiffusionGemma: denoise workspace, schedule/progress contract, monoblock path
- Gemma4: dual-path MoE fuse (26B), draft-session frozen KV + rope_dynamic (**assistant-MTP path only**, not pure direct)
Pure **direct** therefore validates backbone regressions/gains; it will **not** show Phase D/E assistant-draft micro-opts (need a separate MTP suite for those).

## DiffusionGemma 26B-A4B 4-bit — first committed block

Baseline: `2026-07-08-acceptance-075-first-block` (also show `2026-07-05-readme-first-block-refresh`).

| prompt | metric | baseline 07-08 | baseline 07-05 | **now** | vs 07-08 | vs 07-05 |
|---:|---|---:|---:|---:|---:|---:|
| 128 | block decode tok/s | 158.9 | 147.8 | **69.6** | -56.2% | -52.9% |
| 128 | prefill tok/s | 1151.0 | 1064.8 | **905.1** | -21.4% | -15.0% |
| 128 | time to first block ms | 1722.8 | 1852.3 | **3817.1** | +121.6% | +106.1% |
| 512 | block decode tok/s | 109.6 | 104.3 | **50.8** | -53.7% | -51.3% |
| 512 | prefill tok/s | 2794.0 | 2649.7 | **1469.1** | -47.4% | -44.6% |
| 512 | time to first block ms | 2520.1 | 2647.1 | **5397.5** | +114.2% | +103.9% |
| 2048 | block decode tok/s | 163.5 | 140.1 | **83.8** | -48.7% | -40.2% |
| 2048 | prefill tok/s | 3922.3 | 3874.4 | **1552.2** | -60.4% | -59.9% |
| 2048 | time to first block ms | 2088.9 | 2356.7 | **4373.4** | +109.4% | +85.6% |

**Reading:** large **loss** on first-block decode (~−56% to −54% vs 07-08) and prefill (~−21% to −60%), with ~2× denoise wall (p128: ~3.68s vs ~1.61s at similar step counts). `full_pipeline_used=1` still. Treat as a **real Diffusion regression** to investigate (not noise).

## Gemma 4 12B 4-bit ffn4 — AX direct (no n-gram)

Baseline: `2026-07-10-gemma4-12b-4bit-ax-only-current`.

| prompt | metric | baseline | **now** | delta |
|---:|---|---:|---:|---:|
| 128 | decode tok/s | 69.5 | **69.6** | +0.1% |
| 128 | prefill tok/s | 394.7 | **407.7** | +3.3% |
| 128 | ttft ms | 324.3 | **314.0** | -3.2% |
| 512 | decode tok/s | 67.2 | **67.8** | +0.8% |
| 512 | prefill tok/s | 524.2 | **531.1** | +1.3% |
| 512 | ttft ms | 976.7 | **964.1** | -1.3% |
| 2048 | decode tok/s | 65.4 | **66.1** | +1.1% |
| 2048 | prefill tok/s | 556.9 | **559.2** | +0.4% |
| 2048 | ttft ms | 3677.7 | **3662.6** | -0.4% |

**Reading:** decode ~**flat to slight gain** (+0.1% to +1.1%); prefill mixed (+3% at 128, +1% elsewhere). Within normal run variance for dense direct; **no material gain/loss** from recent assistant-path work (expected).

## Gemma 4 26B-A4B 4-bit — AX direct (no n-gram)

Baseline: `2026-07-07-gemma4-26b-4bit-ax-direct-refresh-gen128`.

| prompt | metric | baseline | **now** | delta |
|---:|---|---:|---:|---:|
| 128 | decode tok/s | 135.5 | **135.1** | -0.3% |
| 128 | prefill tok/s | 1342.4 | **595.6** | -55.6% |
| 128 | ttft ms | 95.4 | **214.9** | +125.4% |
| 512 | decode tok/s | 132.3 | **131.9** | -0.3% |
| 512 | prefill tok/s | 3031.6 | **1195.9** | -60.6% |
| 512 | ttft ms | 168.9 | **428.1** | +153.5% |
| 2048 | decode tok/s | 127.7 | **126.7** | -0.8% |
| 2048 | prefill tok/s | 4613.2 | **1535.4** | -66.7% |
| 2048 | ttft ms | 443.9 | **1333.9** | +200.5% |

**Reading:** decode ~**flat** (−0.3% to −0.8%). Prefill/TTFT show a **large regression** vs 07-07 (prefill −56% at p128, −69% at p512, −72% at p2048). That is unexpected for dual-path fuse-only work and should be verified (possible contract/timing split, thermal, or prefill path change) before blaming MoE fuse alone.

## Bottom line

| model | direct decode | prefill / first-visible | action |
|---|---|---|---|
| DiffusionGemma 26B | **major loss** | **major loss** | investigate denoise/prefill regression before promoting phases |
| Gemma4 12B ffn4 | flat / tiny + | mild + | no public claim change |
| Gemma4 26B A4B | flat | **major prefill loss** | investigate prefill/TTFT; decode OK |

### Not covered by this pass
- Gemma4 **assistant-MTP** multi-depth (where frozen KV + rope_dynamic land) — run `scripts/bench_gemma4_assistant_mtp.py` for that.
- mlx_lm peer A/B (skipped via `--skip-mlx-lm`).
- Public README / `docs/assets` charts **not** updated (evidence only; do not promote).
