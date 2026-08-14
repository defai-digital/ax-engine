# Product Requirements: Qwen 3.6 27B Prefill/Decode Recovery on `df-macbookpro-m5`

| Field | Value |
| --- | --- |
| Status | Paused 2026-08-14; §6(b) 1.20 and §6(d) 1.15 still unmet |
| Owner | AX Engine maintainers |
| Last updated | 2026-08-14 (branch `perf/qwen-prefill-decode`; last remasure chunk-1280 `8c08e31b…` 3b 1.024008 / 3d 1.048763 FAIL; flag stays OFF) |
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
AXQ revision with matching prompt hashes. Skeptic rejected treating the
mlx_lm 0.90/0.97 if-branch as §6(b) completion. The restored gate is the
else-bar: unrounded `min(prefill, decode)` ≥ 1.20× that AXQ baseline.

q4 `lm_head` remasure (2026-08-13, binary `bd9f55b0…`): decode 33.774700 /
33.674339 / 33.317386 = 1.173×; prefill 1.125058 / 1.063227 / 1.030131.
min() FAIL.

2-bit decode + seq-gated 4-bit FFN pack (binary `bb6dca12…`): decode
34.459012 / 34.307110 / 33.946753 = 1.197/1.196/1.196; prefill 1.055 /
1.035 / 1.021. Pack regressed prefill vs q4 and is reverted.

2-bit decode-only remasure (binary `0cefe513…`): decode 34.420446 /
34.319905 / 33.950246 = 1.196×; prefill 1.135047 / 1.069711 / 1.032679.
min() FAIL.

Streaming gated-delta one-chunk p2048 (binary `a783768e…`): prefill
871.645145 / 862.825 = 1.010 (0.978× the 1024-chunk q2only 891). Closed
negative; flag restored to opt-in.

2048 runner chunk + GatedDelta tiled at 1024 (binary `303c3ffd…`): AXQ
p2048 887.054718 / 862.825 = 1.028 (0.996× q2only 891). Community p2048
904.739 / 858 = 1.054 (need 1.15). Wash; chunk cap restored to 1024.

Dual-stream Qwen gate/up qmm (binary `55f15572…`): p2048 868.614 / 862.825
= 1.007 (0.975× q2only). Closed negative; reverted.

Native offset-causal SDPA (binary `ee434354…`): p2048 889.344 / 862.825 =
1.031 (0.998× q2only 891). Wash; flag restored to opt-in.

Intermediate-chunk async_eval (binary `9cc1a98e…`): p2048 890.528 / 862.825
= 1.032 (0.999× q2only). Wash; flag restored to opt-in.

Compiled dual gate/up (binary `0d149b98…`): unrounded vs AXQ baseline
p128 446.818/390.946=1.142915, p512 749.054/700.015=1.070054, p2048
890.962/862.825=1.032611; decode 1.196. vs q2only p2048 890.962/891.022
= 0.9999. Wash; flag restored to opt-in.

Split FFN prefill compile (binary `8b05ec6a…`): p128 441.166/390.946
=1.128459, p512 750.117/700.015=1.071573, p2048 888.772/862.825
=1.030072; vs q2only p2048 0.9975. Slight regression. Flag restored
to opt-in. Else-bar still unmet. Not committed. Standing path remains
2-bit decode-only lm_head + BF16 W_t prefill + 1024 TG + split qw.

SwiGLU→down fuse (binary `bb258fa0…`): p128 441.186/390.946=1.128509,
p512 738.981/700.015=1.055665, p2048 876.209/862.825=1.015511; vs
q2only p2048 0.9834. Prefill regression. Flag restored to opt-in.
Else-bar still unmet. Not committed.

Qwen fused causal prefill (binary `e60c1b0b…`, offset-0 only): p128
446.004/390.946=1.140833, p512 754.403/700.015=1.077696, p2048
895.521/862.825=1.037893; vs q2only p2048 1.005. Offset chunks crashed
SSE. Flag restored to opt-in. Else-bar still unmet. Not committed.

Linear-attn `add_rms_norm_pair` (binary `c3030962…`): p128
443.896/390.946=1.135440, p512 747.326/700.015=1.067586, p2048
890.379/862.825=1.031935; vs q2only p2048 0.9993. Wash. Flag restored
to opt-in. Else-bar still unmet. Not committed. Standing path remains
2-bit decode-only lm_head + BF16 W_t + 1024 TG + split qw.

Prefill post-input Metal (binary `35b0ef16…`): p128 443.810/390.946=1.135220,
p512 742.344/700.015=1.060469, p2048 874.955/862.825=1.014059; vs q2only
p2048 0.982. Regression (per-token Metal loop vs C++ conv1d). Flag restored
to opt-in. Else-bar still unmet. Not committed.

Dual qmm + SwiGLU host FFI (binary `75d6f950…`): p128 438.082/390.946=1.120568,
p512 737.739/700.015=1.053891, p2048 875.207/862.825=1.014350; vs q2only
p2048 0.9823. Prefill regression (~1.8% vs standing 891). Flag restored
to opt-in. Else-bar still unmet. Not committed. Standing path remains
2-bit decode-only lm_head + BF16 W_t + 1024 TG + split qw.

GatedDelta tile-512 (binary `e187d0e9…`): AXQ p2048 892.804/862.825=1.034745
(1.002× q2only 891). Community p2048 912.109/858=1.063065 (3d still FAIL).
Community MTP 54.9/60.1=0.913. AXQ MTP decode dropped to 23–24 tok/s under
this flag. Wash. Flag restored to opt-in. Not committed.

Standalone flat down qmm (binary `97bfa5d0…`): p128 439.559/390.946=1.124347,
p512 746.803/700.015=1.066839, p2048 888.334/862.825=1.029565; vs q2only
p2048 0.997. Slight regression. Flag restored to opt-in. Else-bar still
unmet. Not committed. Standing path remains 2-bit decode-only + 1024 TG
+ split qw.

Single 2048 FFN + GD tile-512 (binary `18ea389c…`): AXQ p2048
889.956/862.825=1.031444 (0.9988× q2only 891). Community p2048
906.719/858=1.056783 (3d still FAIL). Same class as 2048+tile-1024 887.
Flags restored to opt-in. Else-bar still unmet. Not committed.

Qwen prefill dual qmm + SwiGLU Metal (simdgroup 8×8, binary `8f0b4e56…`):
community p128/p512/p2048 prefill ~159/180/179 vs standing 473/783/908
(~0.20–0.34×). Decode unchanged (~34.4). Kernel-level reject (same class
as Gemma dual Metal 8.5×). Flag restored to opt-in. Not committed.
Standing path remains 2-bit decode-only + 1024 TG + MLX split qw.

Lazy intermediate `--ax-direct` skip-eval (binary `b50c209f…`): AXQ p128
437.974/34.517 = 1.120292/1.199157; p512 744.862/34.387 =
1.064065/1.198779; p2048 889.887/33.958 = 1.031365/1.196320. min()
1.031365 FAIL. vs q2only p2048 0.9987. Community p2048 910.410/858.000
= 1.061085 (3d FAIL; 3a PASS). Flag restored to opt-in. Not committed.
Standing path remains 2-bit decode-only + 1024 TG + MLX split qw.

Whole-FFN flatten `[B,S,H]→[B*S,H]` (binary `77435b62…`): AXQ p128
445.717/34.415 = 1.140098/1.195616; p512 749.531/34.330 =
1.070736/1.196804; p2048 889.673/33.944 = 1.031116/1.195850. min()
1.031116 FAIL. vs q2only p2048 0.9985. Community p2048
909.796/858.000=1.060369 (3d FAIL; 3a PASS). Flag restored to opt-in.
Not committed.

LA out_proj `silu_mul_quantized_matmul` (binary `2b846b04…`): AXQ p128
445.094/34.444 = 1.138504/1.196624; p512 748.281/34.349 =
1.068950/1.197453; p2048 890.888/33.949 = 1.032524/1.196012. min()
1.032524 FAIL. vs q2only p2048 0.9999. Community p2048
909.631/858.000=1.060177 (3d FAIL; 3a PASS). Flag restored to opt-in.
Not committed. Standing path remains 2-bit decode-only + 1024 TG + MLX
split qw.

Qwen attn-norm-QKV fuse (binary `544a8b5d…`): community p128/p512/p2048
463.775/777.160/906.020 vs standing 472.770/783.422/908.549 (p2048
1.055968 vs base, 3d FAIL). AXQ `--ax-direct` panicked
(`portable path materializes attn_norm`) when exact skipped the fuse.
Flag restored to opt-in. Panic gate added. Not committed.

In flight: Qwen prefill `contiguous([B,S,H])` before split FFN qmm
(`AX_MLX_QWEN_PREFILL_CONTIGUOUS_FFN` default-ON). Not a closed fuse.
Bar stays unrounded 1.20.

Contiguous-FFN remasure (binary `f72c4606…`): AXQ p128/p512/p2048 min
1.137266/1.072541/1.032745 FAIL. vs q2only p2048 1.000064. Community
p2048 908.312/858=1.058639 (3d FAIL). Flag restored to opt-in.

Compiled Qwen base-RoPE remasure (binary `41fd8313…`): AXQ p2048
890.684/862.825=1.032288 FAIL (0.9996× q2only). Community p2048
908.406/858=1.058749 (3d FAIL). Flag restored to opt-in.

GatedDelta-contiguous remasure (binary `6e56e7ed…`): AXQ p128/p512/p2048
min 1.133067/1.069044/1.030983 FAIL. vs q2only p2048 0.9984. Community
p2048 908.438/858=1.058787 (3d FAIL; 3a PASS). Flag restored to opt-in.

Per-forward QKVZ+BA concat (binary `37125559…`): AXQ p2048
887.779/862.825=1.028921 FAIL (0.996× q2only). Community p2048
904.615/858=1.054331 (3d FAIL; 3a PASS). Concat-every-forward tax.

Load-time fused QKVZ+BA (binary `1fa58239…`): community p2048
907.712/858=1.057940 FAIL (0.999× standing 908.5). 3a PASS. Flag
restored to opt-in.

Down-only prefill compile (binary `99f65ba3…`): community p2048
904.141/858=1.053779 FAIL (0.995× standing 908.5). 3a PASS. Flag
restored to opt-in.

1536-chunk remasure (binary `c8658c41…`): community p2048
899.856/858=1.048784 FAIL (0.990× standing 908.5). 3a PASS. Flag
restored to opt-in.

Compiled GatedDelta remasure (binary `26cf1a69…`): community p2048
905.390/858=1.055235 FAIL (0.997× standing 908.5). 3a PASS. AXQ lane
killed incomplete after 3d already FAIL. Flag restored to opt-in.
Same class as compiled QK-RoPE / split FFN compile. Standing path
remains 2-bit decode-only + 1024 TG + MLX split qw. Bar stays
unrounded 1.20. Not committed.

qodercli Qwen3.8-Max (2026-08-13): p2048 FFN is compute-bound inside
MLX `quantized_matmul` (same kernel mlxcel uses). Closed ±2% washes
are expected; **unrounded 1.20 vs own AXQ baseline is not reachable
without requant**, which stays forbidden. Product peer remains mlxcel
(`PRD-M5-FLEET-AX-VS-MLXCEL`). Do not treat this paragraph as a bar
change.

Packed FFN prefill compile remasure (binary `b4ada020…`): community
p2048 904.620/858=1.054337 FAIL (0.996× standing). 3a PASS. AXQ p2048
886.947/862.825=1.027957 FAIL (0.995× q2only 891). Packed compile
regressed AXQ and cannot engage community 4-bit. Flag restored to
opt-in. MTP killed after 3b/3d FAIL. Bar stays unrounded 1.20. Not
committed.

Interlayer add_rms remasure (binary `269d695d…`): community p2048
904.845/858=1.054599 FAIL (0.996× standing). 3a PASS. AXQ p2048
886.952/862.825=1.027962 FAIL (0.995× q2only). Wash. Flag restored
to opt-in. MTP killed. Bar stays unrounded 1.20. Not committed.

Pipeline-block remasure (binary `5a3427d3…`): community p2048
904.335/858=1.054004 FAIL (0.995× standing). 3a PASS. AXQ p2048
888.809/862.825=1.030115 FAIL (0.998× q2only). Wash. Flag restored
to opt-in. MTP killed. Bar stays unrounded 1.20. Not committed.

Reuse LA initial zeros remasure (binary `dc519b17…`): community p2048
904.358/858=1.054032 FAIL (0.995× standing). 3a PASS. AXQ p2048
888.016/862.825=1.029195 FAIL (0.997× q2only). Wash. Flag restored
to opt-in. MTP killed. Bar stays unrounded 1.20. Not committed.

Packed LA inputs compile remasure (binary `6b6b2e06…`): community p2048
904.726/858=1.054460 FAIL (0.996× standing). 3a PASS. AXQ p2048
888.959/862.825=1.030289 FAIL (0.998× q2only). Wash. Flag restored
to opt-in. MTP killed. Bar stays unrounded 1.20. Not committed.

LA post-input compile remasure (binary `e535cf3e…`): community p2048
911.056/858=1.061838 FAIL (1.003× standing). 3a PASS. AXQ p2048
894.749/862.825=1.036999 FAIL (1.004× q2only). Wash. Flag restored
to opt-in. MTP killed. Bar stays unrounded 1.20. Not committed.

LA dual-stream remasure (binary `f1d47194…`): community p2048
894.153/858=1.042137 FAIL (0.984× standing). 3a PASS. AXQ p2048
879.421/862.825=1.019234 FAIL (0.987× q2only). Regression. Flag
restored to opt-in (Rust + C++). MTP killed. Bar stays unrounded
1.20. Not committed.

LA flatten remasure (binary `07de1419…`): community p2048
904.487/858=1.054181 FAIL (0.996× standing). 3a PASS. AXQ p2048
888.640/862.825=1.029919 FAIL (0.997× q2only). Wash. Flag restored
to opt-in (Rust + C++). MTP killed. Bar stays unrounded 1.20. Not
committed.

LA contiguous-QKV remasure (binary `0f01c381…`): community p2048
904.710/858=1.054442 FAIL (0.996× standing). 3a PASS. AXQ p2048
887.915/862.825=1.029078 FAIL (0.997× q2only). Wash. Flag restored
to opt-in. MTP killed. Bar stays unrounded 1.20. Not committed.

LA prefill-q2 remasure (binary `82ffde4a…`): community p2048
903.735/858=1.053305 FAIL (0.995× standing). 3a PASS. AXQ p2048
889.075/862.825=1.030423 FAIL (0.998× q2only). Wash. Flag restored
to opt-in. MTP killed. Bar stays unrounded 1.20. Not committed.

Prefill-q2-down remasure (binary `1e6bcf13…`): community p2048
904.205/858=1.053854 FAIL (0.995× standing). 3a PASS. AXQ p2048
887.053/862.825=1.028080 FAIL (0.996× q2only). Wash/slight
regression. Flag restored to opt-in. MTP killed. Bar stays
unrounded 1.20. Not committed.

ax-code GLM 5.2 1M (`.internal/ax-code-glm52-review/glm-final.md`):
qoder's unreachable headline is right; FFN-only is incomplete.
Compute-bound qmm union (FFN + LA proj + attn QKVO) ≈ 1.89 s of
the 2.305 s wall; movable non-qmm ≈ 0.36 s. 3b needs −327 ms
(`PASS_1_20: no ; need_ms=327 ; available_ms≈40`). Named unused
`AX_MLX_QWEN_GD_PREFILL_CHUNKWISE` (honest 0–40 ms). Do not
reopen custom 4-bit qmm, 2-bit overlays, or compile/flatten/
contiguous/dual-stream. Chunkwise GD was **not** implemented:
no 3b/3d cell can still move to the bar.

## 10. Standing freeze (2026-08-14)

Production path kept: 2-bit decode-only `lm_head` + BF16 `W_t`
prefill + 1024 TG + packed C++ qw. Every later 27B experiment
flag is default-OFF.

Four-lane + MTP scoreboard (scratch `scoreboard.md`), unrounded:

- **3a PASS** — community direct vs mlx_lm at p128/p512/p2048
  (`after-wt/community-direct.json`, binary `b960bba2…`).
- **3b FAIL 1.032679** — AXQ p2048 891.022 / 862.825 (need
  1035.390). Decode 33.950 / 28.385 = 1.196. Source
  `after-q2only/axq-direct.json`, binary `0cefe513…`.
- **3c PASS** — community MTP 54.7 / mtplx 60.1 = 0.910
  (`after-wt` flappy matrix). AXQ MTP decode 34.95–36.75 ≥
  AXQ direct on gen=128 (`after-wt/axq-mtp.json`).
- **3d FAIL 1.058916** — community p2048 908.549 / 857.999
  (need 986.699).

The §6 numeric bar is unchanged. The residual miss is prefill
arithmetic (qmm union ≈1.89 s vs 1.978 s budget), not an
untried software lever.

GD-chunkwise remasure (binary `282cf2fd…`): community p2048
904.570/858=1.054279 FAIL (0.996× standing). 3a PASS. AXQ p2048
890.097/862.825=1.031608 FAIL (0.999× q2only). Wash. Flag
restored to opt-in. MTP killed. Bar stays unrounded 1.20. Not
committed.

FFN-gs64 remasure (binary `4a2744c7…`): community p2048
901.984/858=1.051264 FAIL (0.993× standing). 3a PASS. AXQ p2048
881.313/862.825=1.021428 FAIL (0.989× q2only). Regression.
Flag restored to opt-in. MTP killed. Bar stays unrounded 1.20.
Not committed. MLX gs64 `quantized_matmul` is not faster than
gs32 at M=1024.

FFN-q3 remasure (binary `dc7036c2…`): community p2048
870.058/858=1.014054 FAIL (0.958× standing). 3a PASS. AXQ p2048
861.436/862.825=0.998390 FAIL (0.967× q2only). Regression.
Flag restored to opt-in. MTP killed. Bar stays unrounded 1.20.
Not committed. MLX 3-bit qmm is slower than 4/6-bit at M=1024.
Bit-width overlays of FFN qmm are closed (2-bit, 3-bit, gs64).

FFN-contig-w remasure (binary `99c3b4cc…`): community p2048
901.784/858=1.051032 FAIL (0.993× standing). 3a PASS. AXQ p2048
886.834/862.825=1.027826 FAIL (0.995× q2only). Wash. Flag
restored to opt-in. MTP killed. Bar stays unrounded 1.20. Not
committed.

Async-gate-up remasure (binary `aebcaa13…`): community p2048
900.214/858=1.049201 FAIL (0.991× standing). 3a PASS. AXQ p2048
884.304/862.825=1.024894 FAIL (0.992× q2only). Wash. Flag
restored to opt-in. MTP killed. Bar stays unrounded 1.20. Not
committed.

FFN-f32 remasure (binary `128d9a6c…`): community p2048
747.005/858=0.870636 FAIL (0.822× standing). 3a FAIL p2048 pre
0.804. AXQ p2048 734.296/862.825=0.851037 FAIL (0.824× q2only).
Regression. Flag restored to opt-in. MTP killed. Bar stays
unrounded 1.20. Not committed. F32 FFN activations make the
steel qmm slower at M=1024.
