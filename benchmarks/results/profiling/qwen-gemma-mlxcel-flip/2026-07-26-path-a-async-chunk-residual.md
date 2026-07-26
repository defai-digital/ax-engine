# Path A residual — async chunk eval + dual metal v4 (mbp-m5, 2026-07-26)

**Decision: `not_yet`.** Gates unchanged. No S0–S3 flip claim.

## Path A pure unlocks tried this session

### 1. `AX_MLX_CACHE_ONLY_CHUNK_ASYNC_EVAL` (host/GPU chunk overlap)

| pure under cache_eval | median | ratio |
|-----------------------|-------:|------:|
| base | 8380 ms | 1.000 |
| async_chunk | 8146 ms | **0.972** |

Text parity OK. Strict pure bar ≤0.96: **short** (0.972). Recalibrated thr
physics from thr 1.122 suggested 0.972 could clear 1.15 if transfer held.

**Cool multi-process S1 formal (async_chunk alone):** thr **1.103**, gap 1.134,
TTFT 0.903 — **thr regresses** vs baseline 1.113. Pure host-overlap does **not**
transfer under concurrent Qwen+Gemma. Default OFF.

### 2. Thr stack: async_chunk + `PIPELINE_GRANULARITY=layer`

| formal cool 3-rep | thr | gap | TTFT | abs gap |
|-------------------|----:|----:|-----:|--------:|
| async-chunk-pipeline | **1.137** | 1.990 | 0.876 PASS | **68.8 FAIL** |
| + eval block:8 | **1.137** | 1.269 | 0.875 PASS | 44.3 PASS |
| baseline cache_eval recheck | 1.113 | 1.028 | 0.895 | PASS |
| async pipeline alone | 1.122 | 1.113 | 0.888 | PASS |

Best formal thr this campaign: **1.137×** (need ≥1.15). Gap collapses when thr
stack monopolizes GPU; sparse block:8 recovers abs gap ≤50 but ratio still
≥1.27 and thr still 1.137.

### 3. Dual-gate Metal v4 (steel-matched BM=16 BN=16 BK=64)

| pure | median | ratio | text |
|------|-------:|------:|------|
| base | 8139 ms | 1.000 | `" The"` |
| dual_v4 | 66116 ms | **8.12×** | empty |

Same reject class as v1/v3. Naive dual-output GEMM still loses to sequential
MLX steel qmm. Default OFF.

## Physics close-out (2026-07-26)

- thr shortfall from best formal 1.137 → 1.15 is **~1.1%**.
- gap still needs either layer-eval class fairness (gap ~1.0, thr wash) **or**
  thr-without-gap-collapse pure GPU cut.
- Host/async scheduling levers (async chunk, pipeline hints) move thr or gap
  but not both under locked gates.
- True steel-class dual-gate that beats MLX qmm remains open research (v1–v4
  custom Metal all 8×+ worse).

## Product posture

- Gates thresholds **unchanged**.
- All new thr opt-ins default **OFF**.
- Multi-process remains measurement topology.
- Full S0–S3 **not** run (thr cannot clear 1.15 with gap).
