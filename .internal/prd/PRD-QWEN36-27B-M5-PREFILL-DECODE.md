# Product Requirements: Qwen 3.6 27B Prefill/Decode Recovery on `df-macbookpro-m5`

| Field | Value |
| --- | --- |
| Status | Active; §6(b) if-branch: mlx_lm now loads AXQ; 0.90/0.97 unrounded PASS |
| Owner | AX Engine maintainers |
| Last updated | 2026-08-12 |
| Formal host | `df-macbookpro-m5` (Apple M5 Max) |
| Product position | Restore Qwen 3.6 27B **prefill** and **decode** to peer-competitive rates vs **mlxcel** (`.internal/reference/mlxcel`) on both community 4-bit and AXQ 6-bit, in **direct** and **MTP** |
| Related | ADR-003 dispatch-bound decode; ADR-020 Qwen36 linear MTP Tier 2; `docs/performance/decode-gap.md`; `.internal/reports/prefill-regression-investigation-2026-07-28.md` |
| Harnesses | `scripts/bench_mlx_inference_stack.py`, `scripts/bench_qwen36_mtp_matrix.py`, mlxcel `scripts/bench_decode.sh` / `mlxcel-bench-decode` from `.internal/reference/mlxcel` |

## 1. Decision summary

On formal host `df-macbookpro-m5`, AX Engine's Qwen 3.6 27B path is currently
**much slower than other engines** on **prefill** and **decode**. This package
freezes a same-host measurement contract, then lands the smallest runtime fix
that recovers both phases for:

| Lane | Checkpoint | Alias | Mode |
| --- | --- | --- | --- |
| Community direct | `mlx-community/Qwen3.6-27B-4bit` | `qwen3.6-27b` | **direct** |
| Community MTP | `mlx-community/Qwen3.6-27B-4bit` (+ AX local MTP sidecar as the matrix already binds) | `qwen3.6-27b` | **MTP** |
| AXQ direct | `AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP` @ `8c37715c7b5f5ebca00eda6f73be47116a3e4ebc` | `qwen3.6-27b:axq` | **direct** |
| AXQ MTP | same AXQ pack / revision | `qwen3.6-27b:axq` | **MTP** |

No requantization, no new Hub pack, no AXQ revision change, and no
relitigation of closed negative fusion experiments in
`docs/performance/decode-gap.md` without new `df-macbookpro-m5` evidence.

## 2. Problem

Operators running Qwen 3.6 27B on Apple M5 Max observe AX Engine **prefill**
and **decode** rates well below same-host **mlxcel** (pinned at
`.internal/reference/mlxcel`). mlx_lm.benchmark remains a diagnostic
same-weight row when it can load the pack; it is **not** the product
competitor. MTP still records mtplx/lightning when those peers produce a
valid row; otherwise AX MTP is compared to AX **direct** on a decode-heavy
shape and to mlxcel decode if mlxcel loads the same checkpoint.

Historical same-host MTP rows (2026-08-07 `mtp-qwen36-matrix`) already had AX
near mtplx on 27B-4bit (~56 vs ~60 decode). The user-visible "much slower"
claim is therefore treated as a **current-surface** defect (serving, AXQ-only,
MTP-only, or a regression since those rows). The baseline must find the
lagging phase(s) from artifacts — phase split, route identity, eval wall,
MTP accept/rollback — not from guessed kernels.

## 3. Users and jobs

| User | Job |
| --- | --- |
| Local 27B operator on M5 Max | Community 4-bit and AXQ 6-bit chat/agent turns at peer-competitive **prefill** and **decode** |
| Qualification engineer | Reproduce the four-lane contract on `df-macbookpro-m5` with hostname, commit, binary digest |
| Runtime engineer | Change only the measured lag; keep hot-path logic unit-testable |

## 4. Goals

| ID | Goal |
| --- | --- |
| Q27-G-001 | Same-host baseline on `df-macbookpro-m5` for both checkpoints × **direct** + **MTP**, each row reporting `prefill_tok_s` and `decode_tok_s`, plus required peer rows. |
| Q27-G-002 | After the fix, the identical contract meets the numeric bar in §6. |
| Q27-G-003 | Implementing change is committed and pushed with tests that drive the shipped path. |

## 5. Non-goals

- Qwen 3.6 35B-A3B, Gemma, or any model other than the two named 27B checkpoints.
- Strict publication peer-win (AX strictly faster than mlx_lm on prefill, decode, **and** TTFT).
- Public README / marketing tok/s claim updates.
- 72-hour AXQ endurance soak or KV-reclaim-under-full-pool work except if the M5 baseline proves that is the slow path.
- Relitigating closed negative fusion experiments in `docs/performance/decode-gap.md` without new `df-macbookpro-m5` evidence.
- Requantization, new Hub packs, or changing the AXQ pinned revision.

## 6. Success criteria (numeric bar)

After the runtime fix, the same contract re-run on `df-macbookpro-m5` meets all of:

- **(a)** mlx-community **direct** vs same-host `mlx_lm.benchmark` on matching prompt hashes at p128/p512/p2048 gen=128: every cell `decode_tok_s` ≥ 0.97× mlx_lm and `prefill_tok_s` ≥ 0.90× mlx_lm;
- **(b)** AXQ **direct** meets the same mlx_lm ratios if mlx_lm loads the AXQ pack, else the slower of AXQ prefill/decode is ≥ 1.20× that AXQ baseline;
- **(c)** both checkpoints **MTP**: if a same-host MTP peer row exists, AX MTP `decode_tok_s` ≥ 0.90× that peer, else AX MTP decode ≥ AX **direct** on a decode-heavy shape (gen≥128);
- **(d)** any baseline cell that missed its peer bar has its lagging phase ≥ 1.15× that cell’s baseline.

Runs are consistent across the harness repetitions (not a cherry-picked single trial).

## 7. Measurement contract

| Field | Value |
| --- | --- |
| Host | `df-macbookpro-m5` (SSH hostname `df-macbookpro-m5`) |
| Community checkpoint | `mlx-community/Qwen3.6-27B-4bit` |
| AXQ checkpoint | `AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP` revision `8c37715c7b5f5ebca00eda6f73be47116a3e4ebc` (`qwen3.6-27b:axq`) |
| Direct shapes | prompt 128 / 512 / 2048, gen=128, warmup≥2, reps≥5 |
| MTP | shipped Qwen36 MTP harness (matrix flappy + mtplx peer when valid; inference-stack `--ax-mtp-disable-ngram-stacking` for AXQ if the matrix has no AXQ target) |
| Peer (direct) | `mlx_lm.benchmark` for community; same for AXQ if the pack loads |
| Peer (MTP) | mtplx or lightning when that peer produces a valid row |
| Recorded metadata | hostname, `git rev-parse HEAD`, binary sha256, power/thermal notes, prompt/gen lengths, warmup, reps |
| Do not compare | AXQ tok/s to community mlx_lm as a same-weight peer |

Unreachable SSH is a hard stop. Do not invent numbers or substitute another host.

## 8. Implementation approach

1. Freeze this PRD (done by landing this file).
2. Sync/build a release `ax-engine-server` on `df-macbookpro-m5`; record hostname and binary digest.
3. Take the four-lane baseline + peers; identify the lagging phase(s) from artifacts.
4. Use Codex CLI at max reasoning (sol max) to plan and implement the smallest evidence-backed fix.
5. Keep pure hot-path logic unit-testable and separate from SSH/bench I/O.
6. Re-measure the same contract until §6 holds; then commit and push.

Codex prose is process, not proof. Only `df-macbookpro-m5` artifacts authorize a claim.

## 9. Notes (2026-08-12)

§6(b) is unchanged: mlx_lm ratios if that peer loads AXQ, else unrounded
min(prefill, decode) ≥ 1.20× the AXQ baseline. A 1.05× weakening of the
else-branch was rejected and is not restored.

Baseline mlx_lm hung on AXQ (`after` first pass used `--skip-mlx-lm`). The
W_t remasure on `df-macbookpro-m5` loaded `mlx_lm.benchmark` on the pinned
AXQ revision with matching prompt hashes. That engages the **if** branch
(pre ≥ 0.90×, dec ≥ 0.97×). Measured unrounded: p128 1.014/1.074, p512
0.993/1.076, p2048 0.962/1.078. The else-branch 1.20× of 28.78 tok/s
(34.56, ~99.7% of 614 GB/s) is not the active bar while mlx_lm loads.
