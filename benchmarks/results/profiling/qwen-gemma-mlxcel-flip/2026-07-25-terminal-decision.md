# Terminal decision — AX vs mlxcel flip residual campaign (2026-07-25)

**Decision: `not_yet`.** Full S0–S3 **`flip` is not claimed.**

Host: `mbp-m5` / `AKMBPM5MAX.local` / Apple M5 Max.  
Gates: `benchmarks/manifests/qwen_gemma_flip_gates.v1.json`  
sha256: `9a6be4274c7ebf152e5d757aa2f765f63b8c42aa6aa84ad55e2c4b722c76b192`  
**Threshold fields unchanged** (thr ≥1.15×, TTFT ≤0.90×, gap ≤0.90× and ≤50 ms, zero errors).

## Best measured stack (cool 3-rep dual-target S1)

**Multi-process AX + Gemma `AX_MLX_CACHE_ONLY_CHUNK_EVAL=1`**  
Artifact: `2026-07-25-s1-mp-cache-eval/flip-decision.json`

| metric | ratio | gate |
|--------|------:|------|
| thr | **1.109×** | FAIL (≥1.15) |
| gap p95 | **1.113×** | FAIL (≤0.90); abs ~39 ms PASS (≤50) |
| TTFT p95 | **0.899×** | PASS |

## Residual exhaustion (no pure/S1 headroom for flip)

- Exclusive thr ceiling ~1.03–1.05×; dual-hold gap 160–220 ms reject.
- Pure gate_up dual-qmm residual (~40% wall): Metal dual GEMM, pack, FFI, compile,
  async, dual-stream, hybrid packed+split GEGLU — all reject or noise; best pure
  under multi-process keep_base ~**0.993** (need ≤0.96 for thr 1.15 physics).
- mlxcel multi-token bits=8 MLP is op-at-a-time dual steel qmm (#680) — same class
  as AX split dual qmm; no hidden dual-gate GEMM to port.
- Multi-process S1 ladder: baseline 1.062 → cache_eval **1.109** (best) → FFN
  compile / Qwen-only compile / concurrent-tax wired — wash or regress.
- Full S0–S3 under locked gates **not run**: would re-measure `not_yet` without
  thr headroom.

## Product posture

- Product default remains **single-process multi-model exclusive** arbiter.
- Multi-process + cache_eval is **measurement topology** (mlxcel parity), not a
  silent product flip.
- Failed thr opt-ins stay **default OFF**. Gates **not relaxed**.

## Supporting close-out docs

- `2026-07-25-blocked-thr-physics.md`
- `2026-07-25-residual-inventory.md`
- `2026-07-25-best-practices-path.md`
