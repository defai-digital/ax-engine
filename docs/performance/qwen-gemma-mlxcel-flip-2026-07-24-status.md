# Qwen/Gemma mlxcel flip status — 2026-07-24 (night)

**Decision: `not_yet`** — Full 3-rep S0–S3 with exclusive + rotating SWA prefill.

## Full campaign ledger (`2026-07-24-full-rotating-q64`)

| Scenario | thr | TTFT | gap | Status |
| --- | ---: | ---: | ---: | --- |
| **S0** | **1.158×** | **0.839×** | **0.814×** | **PASS** |
| **S1** | **1.047×** | **0.863×** | **0.243×** | thr FAIL only (TTFT+gap PASS) |
| **S2** | **1.346×** | **0.745×** | **0.778×** | **PASS** |
| **S3** | **0.890×** | **0.278×** | **1.785×** | thr+gap FAIL (TTFT PASS) |

## S1 progress

| Config | thr | TTFT | gap |
| --- | ---: | ---: | ---: |
| Exact warm exclusive (pre-rotate) | 1.01× | 0.900× | 0.23× |
| **Rotating SWA prefill + exact warm (q64)** | **1.047–1.057×** | **0.855–0.863×** | **0.23–0.24×** |
| Rotating + q96 | 1.00× | 0.908× | 0.26× (median cold) |
| Concurrent dual-hold + rotating | 1.03× | 0.87× | **~11× FAIL** |

S1 thr needs ~1.15× → wall ≤~9.2 s from ~10.1 s (**~9–10% more pure-sum cut**).

## Code landed (branch tip)

1. Exact S1 text warm after multi-model publish.
2. Compiled silu / gelu / add_rms_norm_pair; standard post-attn fuse.
3. **Rotating SWA prefill** (cold ring init, capacity masks, local rebuild on mismatch) — default ON.
4. Concurrent arbiter opt-in (default exclusive for flip).
5. Adaptive quantum 64 / gap SLO 32 ms.

## Next

1. Further pure Gemma/Qwen path cuts for S1 thr ≥1.15×.
2. S3: arbiter/batch formation (row-exact cohort engagement); gap absolute ≤50 ms.
3. Re-run full ≥3-rep until flip-decision=flip; commit campaign artifacts.
