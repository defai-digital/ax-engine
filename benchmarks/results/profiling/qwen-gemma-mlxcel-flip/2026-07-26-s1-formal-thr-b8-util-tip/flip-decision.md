# Flip decision — thr-b8-util tip formal (mbp-m5, 2026-07-26)

**Decision: `not_yet`.** Gates file thresholds unchanged. No S0–S3.

## Cool ≥3-rep dual-target S1

Target: `ax-qwen-gemma-m5max-multiprocess-cache-eval-thr-b8-qwen-util`  
Peer: `mlxcel-v0.4.2-qwen-gemma-m5max`  
Binary tip after Path A packed-prefill residual + Path B tail residual (fail-closed OFF).

| metric | AX median | mlxcel median | ratio | gate |
|--------|----------:|--------------:|------:|------|
| thr tok/s | 20.944 | 18.374 | **1.140** | FAIL (≥1.15) |
| gap p95 ms | 41.36 | 34.69 | **1.192** | FAIL (≤0.90); abs PASS |
| TTFT p95 | 8515 | 9742 | **0.874** | PASS |

### Per-rep

| rep | AX thr | mlxcel thr | AX gap | mlxcel gap |
|----:|-------:|-----------:|-------:|-----------:|
| 1 | 20.932 | 18.312 | 40.41 | 34.88 |
| 2 | 20.944 | 18.374 | 41.36 | 34.51 |
| 3 | 20.974 | 18.399 | 41.57 | 34.69 |

## Note vs smoke thr 1.146

1-rep smokes over-read thr by ~0.5–0.6 pts. Formal tip thr **1.140** is consistent
with prior formal thr-b8-util **1.141**. Smoke thr near 1.15 is not a gate pass.

## Conclusion

Thr still ~0.9% short; gap still ~32% relative short of 0.90. S0–S3 withheld.
