# Blocked thr physics — AX vs mlxcel flip (mbp-m5, 2026-07-25)

**Decision: `not_yet`.** Gates **not** relaxed. **No flip claim** without a fresh
≥3-rep dual-target S0–S3 artifact with decision=`flip` under locked thresholds.

Host: `AKMBPM5MAX.local` / Apple M5 Max. Gates sha256:
`9a6be4274c7ebf152e5d757aa2f765f63b8c42aa6aa84ad55e2c4b722c76b192`
(`benchmarks/manifests/qwen_gemma_flip_gates.v1.json` — threshold fields unchanged).

## Locked gates (contract)

| gate | threshold |
|------|-----------|
| median thr ratio | ≥ **1.15×** |
| median TTFT p95 ratio | ≤ **0.90×** |
| median stream-gap p95 ratio | ≤ **0.90×** |
| absolute stream-gap p95 | ≤ **50 ms** |
| errors / 503 / lifecycle | **0** |

## Best measured S1 stack (formal cool 3-rep dual-target)

**Multi-process AX + Gemma `AX_MLX_CACHE_ONLY_CHUNK_EVAL=1`**
(`2026-07-25-s1-mp-cache-eval/flip-decision.json`)

| metric | AX median | mlxcel median | ratio | gate |
|--------|----------:|--------------:|------:|------|
| thr tok/s | 19.92 | 17.97 | **1.109** | FAIL (≥1.15) |
| gap p95 ms | 39.4 | 35.4 | **1.113** | FAIL (≤0.90) |
| TTFT p95 | — | — | **0.899** | PASS |
| abs gap | 39.4 ms | — | — | PASS (≤50) |

**Shortfall:** thr needs ~**3.7%** more (wall ≲9.08s from ~9.4s concurrent Gemma
e2e), **and** gap ratio needs ~**21%** relative cut (39→≤32 ms vs mlxcel 35).

Physics: thr ≈ 193 / max_e2e. Concurrent wall ≈ Gemma e2e. Pure Gemma cold under
cache_eval ~**8.2 s**; multi-process concurrent tax ~**14%**. Closing thr alone
needs pure cut and/or concurrent-tax cut of that order.

## Why exclusive path cannot flip thr

| config | thr ratio | gap | note |
|--------|----------:|-----|------|
| exclusive tip | 1.036 | PASS | only thr fails |
| thr-quanta-128 exclusive | 1.045 | PASS | gap headroom cannot buy thr |
| dualhold-q4 | 1.054 | **166 ms FAIL** | dual-hold gap envelope |

Exclusive pure-sum thr ceiling ~1.03–1.05×. Dual-hold fails gap 160–220 ms.

## Pure Gemma 13.8k profile residual (largest stages)

Force-eval profile (~8–10 s cold class):

| stage | wall (s) | share |
|-------|---------:|------:|
| **gate_up dual qmm** | **~3.26** | ~40% |
| down qmm | ~2.08 | ~25% |
| sdpa | ~1.22 | ~15% |
| qkv proj | ~1.07 | ~13% |
| o_proj | ~0.78 | ~9% |
| rope/kv | ~0.53 | |
| activation | ~0.32 | |
| residual/scalar | ~0.47 | |

**Need ≥7.5–11% pure cut** for exclusive thr≥21 class; under multi-process
cache_eval keep_base, pure ≤**0.96** (~4%) is the thr≥1.15 physics bar used in
this campaign.

mlxcel multi-token bits=8 MLP (flip package): **op-at-a-time** dual qmm +
`compiled_geglu_approx_activation` (`gemma4.rs` ~917–920); full MLP compile
disabled for multi-token 8-bit (#680). Same class as AX portable dual qmm +
Metal GEGLU (Metal wins pure 1.018× vs compiled GeGLU on M5).

## Exhaustive residual-backed pure A/Bs (mbp-m5) — no ≤0.96 keep

### Gate_up dual-qmm residual (~3.26s) — impassable without GEMM-class win

| lever | pure ratio | decision |
|-------|-----------:|----------|
| Dual Metal GEMM v1/v2/v3 | 8–25× | reject OFF |
| `DUAL_QMM_GEGLU` (FFI + imperative gelu) | 1.091 | reject OFF |
| `COMPILED_DUAL_GATE_UP` | ~1.00–1.02 | reject OFF |
| `#705 PREFILL_SHAPED` full MLP | ~0.993–1.02 | reject OFF |
| cache_eval dual+#705 both | **0.993** | best pure; still ≫0.96 |
| `DUAL_AFFINE_QMM` one-FFI | 1.002 | reject OFF |
| `ASYNC_DUAL_GATE_UP` | 1.007 | reject OFF |
| `DUAL_STREAM_GATE_UP` (2 GPU streams) | **1.147** | reject OFF |
| Packed gate/up vs split | ~1.03 | keep split |
| `COMPILED_GEGLU_ACTIVATION` | 1.018 | keep Metal GEGLU |

**Conclusion:** Host-FFI collapse, compile, async co-submit, and dual Metal
streams do **not** beat sequential MLX steel qmm for multi-token bits=8 gate_up.
Custom dual GEMM is slower, not faster. A true GEMM-class dual-gate that beats
MLX qmm was not achieved.

### Other pure residuals (no thr headroom)

| lever | ratio | decision |
|-------|------:|----------|
| #672 cache-only chunk eval alone | 0.968 | pure alone short of 0.925 bar |
| #672 eval+clear | 0.959 | reject default OFF |
| long chunk 768/1024 | 0.987/0.981 | keep 512 |
| chunk 384/256 under cache_eval | 1.038/1.089 | keep 512 |
| `AUTO_BUFFER_CAPS=0` (mlxcel M5 leave-default) | 0.989 | keep auto-raise |
| pipeline granularity block/layer | 1.04–1.07 | reject OFF |
| compose norot / qmmrms | 1.18 / 1.026 | keep_base |
| native offset causal | 1.064 | reject OFF |
| o_proj+rms, attn_norm+qkv, proportional rope | ~1.00–1.03 | reject OFF |
| multi-token layer_scalar Metal | 1.014 | keep decode-only |

**Best pure cut under multi-process keep_base (cache_eval):** ~**0.7%** (both
dual_gate+#705). **Need ~4%** for thr 1.15 physics. Impassable with measured
levers.

## Multi-process S1 ladder (formal cool 3-rep)

| config | thr | gap ratio | TTFT | note |
|--------|----:|----------:|-----:|------|
| multi-process baseline | 1.062 | 1.32 | 0.940 | thr lift vs exclusive |
| + Gemma #672 cache-eval | **1.109** | 1.113 | **0.899** | **best thr** |
| + full #672 eval+clear | 1.105 | 1.096 | 0.903 | no better thr |
| + chunk 1024 | 1.077 | 1.369 | 0.925 | regress |
| + DENSE_FFN_COMPILE=1 | 1.103 | 1.119 | 0.904 | regress thr |

Topology recovers gap abs ≤50 ms while lifting thr vs exclusive, but **not** to
1.15× thr **and** 0.90 gap ratio simultaneously.

## What would still be required for flip (honest)

1. **Pure GPU:** ≥~4% pure Gemma prefill under multi-process keep_base
   (historically ≥7.5–11% for exclusive thr≥21), almost certainly a
   **GEMM-class dual-gate** that beats MLX steel qmm on multi-token bits=8 —
   not host-FFI, not naive custom Metal (measured 8–25× worse).
2. **And/or concurrent tax:** cut multi-process Gemma e2e from ~9.4s toward
   ≲9.08s without blowing gap (gap already ratio-fails at 39 ms vs mlxcel 35).
3. **Gap ratio:** even thr 1.15 still needs gap p95 ≲0.9× mlxcel (~≤32 ms if
   mlxcel stays ~35 ms).
4. Fresh ≥3-rep dual-target **S0–S3** decision=`flip` under locked gates.

None of (1)–(3) have positive measured evidence on mbp-m5 after this residual
campaign. Therefore **full S0–S3 is not run to claim flip** — it would
honestly re-measure `not_yet` without new pure/thr headroom.

## Product defaults kept

- Single-process multi-model exclusive arbiter remains product default.
- Multi-process is a flip **measurement** topology (mlxcel parity), not a silent
  product flip without S0–S3.
- All opt-in thr kill-switches that failed pure stay **default OFF**.
- Gate threshold JSON **unchanged**.

## Bottom line

Under locked gates on M5 Max, after residual-backed pure and multi-process S1
ladders, **flip remains `not_yet`**. Best thr **1.109×** (multi-process +
Gemma cache-only chunk eval). Pure residual for the remaining thr gap is
**blocked** by gate_up physics without a proven GEMM-class dual-qmm. Do not
claim flip; do not relax gates.
