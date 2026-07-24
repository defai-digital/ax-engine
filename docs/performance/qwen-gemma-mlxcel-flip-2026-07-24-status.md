# Qwen/Gemma mlxcel flip status — 2026-07-24 (evening)

**Decision: `not_yet`** — S0/S2 can clear locked gates; S1/S3 remain blocked by physics + multi-stream gaps.

Locked gates (median, ≥3 fresh-process reps): thr ≥ 1.15×, p95 TTFT ≤ 0.90×, p95 stream-gap ≤ 0.90× and ≤ 50 ms abs, zero AX errors/503/lifecycle.

## Scenario ledger (best honest medians this branch)

| Scenario | thr | TTFT | gap | Status |
| --- | ---: | ---: | ---: | --- |
| **S0** | **~1.15–1.16×** | **~0.84–0.90×** (flaky) | **~0.81×** | Often PASS; TTFT flaky under multi-scenario load |
| **S1** | **~0.74–0.76×** | **~1.33×** | **~0.80×** | thr+TTFT FAIL; gap can PASS |
| **S2** | **~1.48×** | **~0.83–1.01×** | **~0.77×** | thr/gap PASS; TTFT sometimes fails in full suite |
| **S3** | **~0.82×** | **~7.6×** | **~1.83×** | thr/TTFT/gap FAIL |

Evidence dirs (remote `AKMBPM5MAXx` worktree):  
`benchmarks/results/profiling/qwen-gemma-mlxcel-flip/2026-07-24-{s0-ttft-5rep,s0s2-double-warm,full-soft-park-rewarm,s1-restored-idle-uncap,s1-longwarm-chunk1024}/`.

## What landed (keep)

1. Soft-park same-id reload + double sibling rewarm → **S2 thr clear** (~1.48×).
2. Idle uncap of fair multi-prefill when sibling idle → solo dual-resident Gemma prefill ~8.1–8.9s (was ~11.5s under stuck fair-256).
3. Soft KV pressure prefill 256 (not 1-token pathology).
4. Wall-time adaptive sibling prefill quantum (start 64, max 64, SLO 32 ms) → **S1 gap can PASS** (~0.80×).
5. Qwen dense FFN matvec Metal (gate/up+down) + greedy OpenAI `repetition_penalty=1.0` → **S0 thr ~1.15–1.16×**.
6. Production-path double warm + HTTP `/v1/completions` SSE warm → S0 TTFT often ≤0.90×.
7. Stream engine burst 64; sibling-active burst=1 for gap isolation.

## S1 physics (measured 2026-07-24 M5 Max)

S1 thr = `(192 + 1) / scenario_wall`. Wall ≈ max(Qwen e2e, Gemma e2e); Gemma 13 826-token prefill dominates.

| Mode | Gemma 13.8k TTFT | Notes |
| --- | ---: | --- |
| Pure dual-resident | **~8.8–8.9 s** | Fair off / idle uncap |
| Warm concurrent micro | **~9.0–10.1 s** | thr micro ~17–18 tok/s |
| Formal concurrent (best) | **~12.4–13.3 s** | thr ~14.8 / 0.76× mlxcel |
| mlxcel concurrent | **~9.2–10.6 s** | thr ~17.8–19.5 |

Serial bound on one GPU: wall ≥ `W_gemma + W_qwen` ≈ 8.85 + 1.75 ≈ **10.6 s** → thr ceiling ≈ **18.2 tok/s** ≈ **0.93×** mlxcel 19.5, **not 1.15×**.

To hit thr 1.15× need wall ≤ ~8.6 s → pure Gemma ≲ **6.9 s** (**~22% faster pure prefill**) *and* concurrent ≈ pure. Scheduling alone cannot clear the gate.

Multi-process AX A/B (two `ax-engine-server` like mlxcel): concurrent thr **~17–18**, no free lunch on single Metal device.

Rejected / regressed on M5:

- Fixed large quantum / compile prefill for S1 (gap or thr regression).
- First-load 2k/4k long warm + `--prefill-chunk 1024` formal S1 → thr **0.725×**, gap **1.005×** FAIL (`2026-07-24-s1-longwarm-chunk1024`). Left as opt-in `AX_SERVER_LONG_PREFILL_WARM=1` only.
- 8k warm on every load (destroyed S2 thr earlier).

## S3 residual

Four-stream thr ~0.82×, TTFT ~7.6×, gap ~1.83×. Need arbiter/batch formation + optional server-mode batched-decode product decision (deep-review P4). `AX_MLX_BATCHED_DECODE=1` is already on in the flip target.

## Next levers (ordered)

1. **Gemma pure prefill −20%+** (Metal GEGLU/attention, mlxcel R2 MLX pin/patch audit, forced chunked-eval + NA tile from deep-review §2.3). Without this, S1 thr 1.15× is unreachable under single-GPU exclusivity.
2. Close formal↔micro concurrent gap (formal still ~12–13 s vs warm ~9–10 s) without regressing gap SLO.
3. Stabilize S0/S2 TTFT under full suite load (median ≤0.90×).
4. S3: profile arbiter hold + row-exact cohort engagement; product call on batched-decode drift.

## Do not

- Relax locked gates.
- Enable TurboQuant / paged-pool default-on without new M5 evidence.
- Reintroduce multi-row TG matvec without a new positive A/B.
- Default-on long first-load warm or fixed huge sibling quanta without formal S1 gap A/B.
