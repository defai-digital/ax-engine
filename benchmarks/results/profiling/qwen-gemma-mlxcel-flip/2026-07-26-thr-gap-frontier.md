# Thr/gap frontier residual (mbp-m5, 2026-07-26 continuation)

**Decision: `not_yet`.** Gates unchanged. S0–S3 not run.

## Formal cool ≥3-rep S1 ladder (locked gates)

| stack | thr | gap | TTFT | abs gap | note |
|-------|----:|----:|-----:|--------:|------|
| baseline cache_eval | 1.113 | 1.028 | PASS | PASS | keep_base measure |
| pipe-block2 | 1.125 | 1.046 | PASS | PASS | balanced |
| async pipeline layer | 1.122 | 1.113 | PASS | PASS | |
| thr stack (async_chunk+pipe layer) | 1.133–1.137 | 1.79–1.99 | PASS | FAIL | thr peak; gap collapse |
| thr stack + eval block:8 | 1.137 | 1.269 | PASS | PASS | |
| **thr-b8 + Qwen utility QoS** | **1.141** | 1.216 | PASS | PASS | **best formal thr** |
| layer-eval | 1.110 | **1.000** | PASS | PASS | best formal gap ratio |

Gates: thr≥1.15, gap≤0.90, TTFT≤0.90, abs gap≤50.

## Shortfall

- Best thr **1.141** needs **~0.8%** more (AX thr 20.93 → ~21.10 vs mlxcel ~18.35).
- Best multi-process gap ratio **1.000** (layer-eval) — matches mlxcel, does **not** beat it by 10% (need ≤0.90).
- Stacks that raise thr above ~1.13 **tax gap** (pipeline monopolization); stacks that match mlxcel gap **tax thr**.

## Rejected this continuation (smoke or formal)

- ngram / AX_NO_SPEC=0: thr regresses (AX thr ~16–18)
- thr stack + dual_gate/#705 shaped: thr wash/regress
- thr stack + dual_affine: thr regress  
- dual metal v4: pure 8.12× empty text
- block:4/6 on thr stack: thr regresses
- thr-stack-util smokes with mlxcel thr ~15.8: ratio theater (ignore)

## Physics

Multi-process gap ≤0.90 requires AX p95 gap ~10% **below** mlxcel (~32 ms vs ~36 ms).
Layer-eval only reaches parity (~35 ms). Further fairness kills thr below 1.15.
Remaining thr 0.8% needs pure GPU that does not re-open gap — steel dual-gate class
still not achieved (custom Metal v1–v4 reject).

## Product posture

Gates unchanged. Opt-ins default OFF. Multi-process measurement only. No flip claim.

## Continuation (same day) — process_thruput_tier + noasync formal

### Harness residual
- `process_thruput_tier` (taskpolicy `-t` 0..3) in `bench_qwen_gemma_flip_target.py`, fail-closed, unit-tested.
- Target smokes: thr-b8-gemma-t0-qwen-util thr **1.145** gap 1.217 (non-theater); thr-b8 alone t0 regresses.

### Cool formal ≥3-rep S1: pipe-b8 + Qwen util **no async**
Artifact: `2026-07-26-s1-formal-pipe-b8-util-noasync/`

| metric | AX median | mlxcel median | ratio | gate |
|--------|----------:|--------------:|------:|------|
| thr tok/s | 20.86 | 18.37 | **1.136** | FAIL (≥1.15) |
| gap p95 ms | 38.2 | 36.1 | **1.057** | FAIL (≤0.90); abs PASS |
| TTFT p95 | — | — | **0.877** | PASS |

vs thr-b8+async formal (1.141 / 1.216): dropping async **improves gap** (1.216→1.057) but **taxes thr** below 1.15.
Smoke thr 1.1515 was inflated (mlxcel thr 18.07). Decision: **not_yet**.

### Cool formal ≥3-rep S1: thr-b8 + Gemma thruput tier 0 + Qwen util
Artifact: `2026-07-26-s1-formal-thr-b8-gemma-t0-qwen-util/`

| metric | AX median | mlxcel median | ratio | gate |
|--------|----------:|--------------:|------:|------|
| thr tok/s | 20.91 | 18.39 | **1.137** | FAIL (≥1.15) |
| gap p95 ms | 41.7 | 34.4 | **1.213** | FAIL (≤0.90); abs PASS |
| TTFT p95 | — | — | **0.876** | PASS |

Gemma `process_thruput_tier=0` does **not** beat thr-b8+qwen-util formal thr 1.141.
Smoke thr 1.145 was noise. Reject as thr unlock.

### Smoke: pipe-b8 + gemma t0 + qwen util noasync
thr **1.080** gap 1.175 — thr regress. Reject.

**Best formal thr remains thr-b8+qwen-util 1.141 / gap 1.216.** Gates unchanged. S0–S3 withheld.

### Residual: wire dead `decode_logits_projection_sg_*` optional kernels + `AX_MLX_GEMV_SIMDGROUP_MATRIX=1`
Wiring completeness residual (kernels were compiled but absent from `PHASE1_OPTIONAL_METAL_KERNELS`).
Unit-tested. Smoke A/B on new binary (mbp-m5):

| stack | thr | gap | ax_thr | ax_gap | note |
|-------|----:|----:|-------:|-------:|------|
| thr-b8-util rebin | 1.143 | 1.175 | 21.32 | 41.2 | same-binary baseline |
| thr-b8-util + GEMV_SG | **1.117** | **1.272** | 20.90 | 44.0 | thr+gap regress |
| layer-eval + GEMV_SG | 1.081 | 1.036 | 20.19 | 36.1 | thr regress vs layer-eval formal 1.110 |

**Reject GEMV_SG as thr/gap unlock.** Wiring fix still correct (fail-closed opt-in remains default OFF).

## Session terminal
Multi-process concurrent thr/gap dual still physics-blocked under locked gates.
Best formal thr **1.141** (need ≥1.15); best formal gap **1.000** (need ≤0.90).
No S0–S3. Gates file unchanged.


## Pack/split pure + thr-b8 priority smokes (2026-07-26 cont.)

Under thr-b8-like pure env: **pack ON median 7819 ms**, split 8246 ms (1.055×).
Keep pack default ON (steel dual-output single qmm is faster under this stack).

Cool concurrent smokes: thr-b8-util **1.142 / gap 1.223** (matches formal);
plain thr-b8 and Qwen thruput-tier 0 regress thr. No dual pass. still **not_yet**.

## Path A: force packed Gemma4 prefill (SPLIT_PREFILL=0)

Pure under thr-b8 keep_base: packed **1.042×** vs split (reject). Concurrent
thr-b8-util split-prefill smoke thr **1.145** / gap 1.201; packed thr 1.119.
Keep default split prefill. still **not_yet**.

## Path B: pipeline eval tail-layer overlay (`AX_MLX_PIPELINE_EVAL_TAIL_LAYERS`)

Force layer-eval on last N multi-token layers over thr-b8 base. 1-rep S1 ladder
on mbp-m5 (see `2026-07-26-pipeline-eval-tail-layers-residual/`):

| config | thr | gap |
|--------|----:|----:|
| tail0 baseline | **1.146** | 1.172 |
| tail8 (best gap) | 1.080 | **1.016** |
| tail16 (best thr overlay) | 1.119 | 1.112 |

Every `N>0` taxes thr; best gap 1.016 still ≫0.90. Formal skipped. **Reject
tail overlay.** Keep default OFF. still **not_yet**.

## thr-b8 config residual: linear pack ON + Qwen background

| config | thr | gap |
|--------|----:|----:|
| thr-b8-util baseline (pack=0) | **1.146** | 1.172 |
| linear-pack=1 | 1.142 | 1.265 |
| qwen background | 1.123 | 1.222 |
| bg + linear-pack | 1.111 | 1.239 |

**Reject.** Keep pack force-OFF and utility QoS on thr-b8. still **not_yet**.

## Cool formal tip remeasure: thr-b8-util (3-rep)

Artifact: `2026-07-26-s1-formal-thr-b8-util-tip/`

| metric | ratio | gate |
|--------|------:|------|
| thr | **1.140** | FAIL (≥1.15) |
| gap | **1.192** | FAIL (≤0.90) |
| TTFT | **0.874** | PASS |

Consistent with prior formal thr **1.141**. Smoke thr 1.146 was optimistic.
still **not_yet**. Best formal thr remains ~**1.140–1.141**.
