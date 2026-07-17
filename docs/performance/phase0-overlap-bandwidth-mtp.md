# Phase 0 measurements: host/GPU overlap, MoE bandwidth cross-check, MTP clone cost

**Date:** 2026-07-17 · **Host:** Apple M3 Max, 128 GB (NOT the M5 Max used for
README/`moe-bandwidth-gap.md` numbers) · **Probe peak:** 349.7 GB/s (best of 8
trials, 6 GiB bf16 `mx.sum`, MLX 0.31.2 — same methodology as the 577 GB/s M5
Max probe) · **Plan:** `.internal/specs/TECH-SPEC-DECODE-DISPATCH-EFFICIENCY.md`
Phase 0 (items 0a/0b/0c; 0d was the fused-router A/B, recorded in
`moe-bandwidth-gap.md`).

## 0a — The direct-pipeline double-buffer does NOT overlap host build with GPU execution

**Method.** `decode-trace` gained two bin-only injection envs:
`AX_DECODE_TRACE_HOST_SPIN_US` (busy-spin, draws package power) and
`AX_DECODE_TRACE_HOST_SLEEP_US` (`thread::sleep`, draws none). Each injects a
fixed host delay per decode step, before the pipeline advance. If step N's
graph were executing on the GPU while the host builds step N+1 (the double
buffer's intent, `generate.rs` advance ordering: build → `async_eval` submit →
`eval` barrier), injected host delay would be absorbed up to the GPU residual
(~13.6 ms at baseline shares here). Model: Qwen3-Coder-Next-4bit, 128 steps.
Raw logs: `benchmarks/results/inference/mlx-inference/2026-07-17-coder-next-host-injection-sweep/`.

**Result: injections are never absorbed, at any scale, with either injector.**

| injected µs/step | sleep: total µs/tok (Δ vs 0) | spin: total µs/tok (Δ vs 0) |
| ---: | ---: | ---: |
| 0 | 18,392 (—) | 18,586 (—) |
| 500 | 20,079 (+1,687) | — |
| 1,000 | 20,008 (+1,616) | — |
| 2,000 | 22,358 (+3,966) | — |
| 4,000 | 24,970 (+6,577) | 25,229 (+6,643) |
| 8,000 | 32,684 (+14,292) | 32,341 (+13,755) |
| 16,000 | 50,032 (+31,640) | 42,460 (+23,874) |

Two findings:

1. **No effective host/GPU overlap.** Δtotal ≥ injection everywhere, including
   at 500 µs. The `async_eval` bucket ≈ the full GPU step (15.9–16.1 ms at
   baseline) and `pending eval` ≈ 1.3 µs: the `mlx_async_eval` call is the
   de-facto barrier for the previous step. The code's double-buffer ordering
   does not deliver overlap under the current MLX (0.31.2) async semantics via
   the shim. Root cause (scheduling pass running synchronously on the caller vs
   a 1-deep async queue that waits before enqueue) is NOT identified here —
   that is follow-up work, and it is exactly the "zero-bubble" gap vLLM closed
   in its 2026 scheduler work.
2. **Idle host gaps additionally slow the GPU (DVFS).** Δtotal runs 1.6–2.0×
   the injected delay even for the zero-power sleep injector, and the
   `async_eval` (GPU) bucket itself inflates with idle time (15.9 → 28.7 ms at
   16 ms sleep). Host stalls cost more than face value because the GPU drops
   clocks between bursts.

**MTP-path quantification** (from the 0c run below, Qwen3.6-35B-A3B-6bit-MTP,
256 tokens): `ax_mtp_verify_forward_wall_us` (host graph build) = 274 ms vs
`ax_mtp_verify_eval_wall_us` (GPU) = 1,351 ms — host build is **~17% of verify
wall and fully serial** (the MTP verify loop has no async submit at all).

**Batched path:** structural only — `batched_decode_session.rs` steps through
a synchronous per-step barrier with no double buffer; measurement deferred to
Phase 3 (default-OFF path).

**Consequences.**
- Every µs removed from host graph build converts to wall clock ~1:1 today
  (plus avoided DVFS penalty). This RAISES the value of per-layer compile
  (Phase 1 item 2): the ~2.2–2.5 ms/step host build on this model is ~12–13%
  of the step, none of it hidden.
- A new Phase 1 candidate ranks alongside MoE-compile: make `async_eval`
  actually asynchronous (root-cause + fix or upstream report). Ceiling ≈ the
  full host-build share on every decode path.

## 0b — Bandwidth cross-check on a second host (the derived 40% holds)

`moe-bandwidth-gap.md` derived 40% bus utilization for batch=1 MoE decode on
M5 Max and flagged it as needing confirmation. Independent M3 Max reading:

- **Pure single-token decode** (decode-trace, no speculation): active bytes per
  forward = 1.965 GB (manifest 10-of-512 routed-expert scaling), 70.2 tok/s
  (router-A/B interleaved baseline median) → 137.9 GB/s ≈ **39% of the 349.7
  GB/s probe peak**. Thermal variance on uninterleaved runs: 53.8–70.6 tok/s
  (30–39%). The M5-derived 40% reproduces on a second host and probe.
- **Speculative verify steps are worse, not better.** With n-gram stacking on
  the W1 profile's repeating synthetic prompt (51 verify steps, depth 4, 100%
  accept, 256 tokens): 34.7 ms per 5-position verify step → 56.6 GB/s ≈
  **16% utilization during verify steps** (`bw-profile` artifact, regime
  breakdown). Longer graphs + serial host build (0a) push spec steps further
  from the bandwidth ceiling.
- **Hardware counters:** not captured (powermetrics needs sudo; xctrace export
  left as follow-up). Two independent probe+derivation readings now agree,
  which was the confirmation the doc asked for; a counter reading remains a
  nice-to-have.
- **Tooling fix shipped:** `scripts/profile_decode_bandwidth.py` derived
  forward passes as `tokens − ax_mlx_bonus_tokens`, but the n-gram stacking
  path emits bonus tokens without that counter (51 steps emitted 256 tokens
  with `bonus_tokens=0`), inflating the top-level utilization to a bogus 0.80
  "weight-bandwidth-bound" on the first run. The script now prefers the
  per-regime step counters and flags MTP-package runs as approximate
  (`forward_pass_count_approximate`). Artifacts:
  `benchmarks/results/bw-profile/*-2026-07-17-m3max-phase0*.json`.

## 0c — MTP linear-attn cache clone: host cost is NEGLIGIBLE (premise refuted)

The Phase 1 plan carried "replace the per-verify-step full cache clone with a
bounded checkpoint" on the assumption the clone was the largest per-step
addition. Measured (Qwen3.6-35B-A3B-6bit-MTP, 256 decode tokens, two runs):

| bucket | wall (run 2) | share of 3.46 s decode |
| --- | ---: | ---: |
| verify_eval (GPU) | 1,351 ms | 39.1% |
| verify_forward (host build) | 274 ms | 7.9% |
| draft (MTP proposer) | 245 ms | 7.1% |
| rollback | 59 ms | 1.7% |
| **cache_clone** | **1.3 ms** | **0.04%** |
| accept + target_softmax | <0.3 ms | ~0% |

`cache_clone_wall_us` = 1,282/1,320 µs across the two runs — **0.04% of
request wall**. Caveat: the counter measures the host wall of the clone call;
`MlxArray` clones are refcounted/lazy, so a physical copy could in principle
materialize inside the GPU `verify_eval` bucket. But the actionable premise
("the clone is the single biggest per-step cost") is not supported at the
host level on this model.

**Consequence:** Phase 1 item 3 (bounded SSM checkpoint) is **deprioritized**.
Revisit only if a GPU-side check attributes verify_eval anomalies to copy
traffic. The measured MTP levers are instead: verify host build (~17%, see
0a), and the draft forward (7%).

## Updated Phase 1 ranking (evidence-adjusted)

1. **Async-eval overlap root cause** (new, from 0a) — ceiling ≈ host-build
   share on every path (~12–13% single decode on Coder-Next class models,
   ~17% of MTP verify).
2. **MoE per-layer compile default-ON** (unchanged, value raised by 0a — host
   build is not hidden).
3. Conditional lazy multi-depth — largely landed via
   `AX_MLX_GEMMA4_ASSISTANT_DEEP_NEEDS_FIRST_CONF` (default ON); verify
   residual value before more work.
4. ~~Bounded MTP state checkpoint~~ — deprioritized by 0c.

All numbers in this document are M3 Max; README/M5 Max claims are unaffected.
