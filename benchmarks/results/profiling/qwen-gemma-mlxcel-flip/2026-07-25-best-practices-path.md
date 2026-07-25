# Best-practices path — AX vs mlxcel flip (mbp-m5, 2026-07-25)

**Recommendation: accept `not_yet` under locked gates. Do not thrash. Do not claim flip.**

Gates file: `benchmarks/manifests/qwen_gemma_flip_gates.v1.json`  
(thresholds unchanged: thr ≥1.15×, TTFT ≤0.90×, gap ≤0.90× and ≤50 ms, zero errors).

Host: `AKMBPM5MAX.local` / M5 Max. Tip evidence through `e2d3b953`.

---

## Why thrashing is the wrong next step

Residual-backed pure and multi-process S1 ladders on mbp-m5 exhausted the levers that
mlxcel source review could justify:

| path | best result | vs gate |
|------|------------:|---------|
| Exclusive single-process | thr ~1.03–1.05× | thr FAIL only |
| Dual-hold max=2 | thr ~1.05×, gap 160–220 ms | gap FAIL |
| Multi-process + #672 cache_eval | thr **1.109×**, gap ratio 1.11, TTFT PASS | thr + gap ratio FAIL |
| Qwen-only / FFN compile / concurrent wired | thr ≤1.110, often regress | not better |
| Pure GEMM dual-gate family (Metal, pack, FFI, async, stream, hybrid) | best pure ~0.993 under keep_base (need ≤0.96) | no thr headroom |

mlxcel multi-token MLP bits=8 is **op-at-a-time dual steel qmm** (#680) — same class as
AX split dual qmm. There is no hidden dual-gate GEMM in mlxcel to copy.

Running another full S0–S3 without new physics **will honestly re-measure `not_yet`**.

---

## Best-practices ordering (recommended)

### 1. Accept `not_yet` (do this now) — **recommended default**

- Keep locked gates **unchanged**.
- Keep product default **single-process multi-model exclusive** arbiter.
- Treat multi-process + Gemma `AX_MLX_CACHE_ONLY_CHUNK_EVAL=1` as the **best measured**
  flip *measurement* topology (mlxcel parity), **not** a silent product flip.
- Leave opt-in thr kill-switches that failed pure **default OFF**.
- Ship intermediate evidence: residual inventory + blocked thr physics (already on main).

This is the only path that preserves comparison integrity after measured physics
blocks 1.15× thr under residual-backed levers.

### 2. Path A — GEMM dual-gate research (long-term thr unlock)

**When:** product wants thr ≥1.15 under *current* gates without topology games.

**What “best practice” means here:**

- Target stage: multi-token bits=8 **gate_up dual qmm** (~40% pure Gemma wall).
- Success bar: pure ≤**0.96** under multi-process keep_base (`CACHE_ONLY_CHUNK_EVAL=1`),
  then cool multi-process S1 thr ≥1.15, then ≥3-rep dual-target **S0–S3**.
- **Out of scope as thrash:** host-FFI collapse, naive custom Metal tiles, packed-row
  steel qmm, dual-stream issue, compile-shaped full MLP — all measured reject/noise.

**In scope as real work:**

1. Deep-read MLX steel/NAX `quantized_matmul` dispatch for multi-token affine bits=8
   (tile sizes, X reuse, whether dual-N output is representable as one kernel).
2. Prototype a **dual-output steel-class** path (or upstream MLX change) that is
   bandwidth-competitive with two sequential steel qmms — not a hand-written GEMM
   that re-implements dequant worse than MLX.
3. Pure A/B on mbp-m5 (3 cold reps, cache_eval keep_base), keep only if ≤0.96.
4. Only then cool S1 → full S0–S3.

**Estimate:** research / kernel iteration, not a same-day flag flip. Do not schedule
S0–S3 until pure A/B lands.

### 3. Path B — Concurrent-tax design (topology residual)

**When:** multi-process is accepted as the fair mlxcel comparison topology and thr is
wall-bound by concurrent Gemma e2e (~9.4s → need ≲9.08s) plus gap ratio (~39→≲32 ms).

**Already rejected as env knobs:**

- Asymmetric `WIRED_LIMIT_SCALE` + Qwen `BATCHED_DECODE=0` → thr **1.100**, TTFT fail.

**Best-practice concurrent work would need a real design**, for example:

- Explicit Metal/QoS priority or process-group scheduling for prefill vs decode
  processes (not just renice hope).
- Measured multi-process memory residency plan that does not thrash unified memory
  under 2×48GB caps (wired/cache/memory triad as one design, not one scale knob).
- Product scheduler that reduces Qwen GPU duty cycle *without* changing S1
  request surface (if allowed by comparison contract).

Pure A/B alone does not validate concurrent-tax levers; cool multi-process S1 does.
Do not re-run S0–S3 until S1 thr ≥1.15 **and** gap/TTFT pass.

### 4. Path C — Gate policy change (product decision only)

**When:** product explicitly re-scopes the flip contract after reading blocked physics.

**Best practice:**

- **Never** silently relax gates to claim flip.
- If multi-process mlxcel parity is the intended contract, document a **new gates
  revision** (new file or version bump + notes), not an in-place silent edit of
  `qwen_gemma_flip_gates.v1.json` without audit trail.
- Any new thresholds should be justified by physics (e.g. concurrent wall tax and
  gap abs SLO already ≤50 ms) and approved by maintainers.

**Illustrative physics-honest options** (not applied; require explicit permission):

| option | thr | gap ratio | gap abs | note |
|--------|-----|-----------|---------|------|
| Current locked | ≥1.15 | ≤0.90 | ≤50 ms | blocked at thr 1.109 / gap ratio 1.11 |
| Multi-process thr-only tighten | ≥1.10 | ≤0.90 | ≤50 ms | still blocked on gap ratio |
| Multi-process abs-gap focus | ≥1.10 | *drop or ≤1.15* | ≤50 ms | matches “gap abs already wins”; needs product OK |

After any C change: fresh ≥3-rep dual-target **S0–S3** under the **new** gates file;
do not retro-label old S1-only artifacts as flip.

---

## What we will not do (anti-patterns)

1. Blind env-flag thrash after residual inventory shows keep_base.
2. Claim `flip` or `candidate_complete` without S0–S3 decision=`flip` under the
   active gates file.
3. Quietly edit locked gate thresholds to pass.
4. Treat multi-process measurement topology as product default without S0–S3 and
   product agreement.
5. Re-run full S0–S3 knowing thr physics cannot pass — wastes mbp-m5 time.

---

## Immediate recommended actions (this campaign)

1. **Decision: `not_yet`** under `qwen_gemma_flip_gates.v1.json`.
2. Keep evidence committed on main (blocked physics, residual inventory, pure A/Bs,
   formal S1 flip-decisions through concurrent-tax reject).
3. **Stop residual thrash** until product picks:
   - **A** with a real GEMM/steel dual-output design, or
   - **B** with a real concurrent-tax design, or
   - **C** with explicit new gate thresholds and versioning.
4. If product wants a short written proposal for C (multi-process-fair recalibration),
   draft it as a **separate gates v2 proposal** — do not apply without approval.

---

## Bottom line

Best practice after a residual-backed campaign that cannot clear locked thr/gap gates
is **honest `not_yet`**, preserved gates, preserved product defaults, and a clear
menu for the only three unblocking paths. Further flag A/Bs without A/B/C input are
not engineering; they are thrashing.
