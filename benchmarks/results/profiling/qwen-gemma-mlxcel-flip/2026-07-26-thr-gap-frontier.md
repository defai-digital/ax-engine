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
