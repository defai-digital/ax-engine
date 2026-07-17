# gather_qmm breaks MLX async pipelining — root cause and a +14–25% MoE decode lever

**Date:** 2026-07-17 · **Host:** Apple M3 Max 128 GB, MLX 0.31.2 · **Follow-up
to:** [`phase0-overlap-bandwidth-mtp.md`](phase0-overlap-bandwidth-mtp.md)
(which measured zero host/GPU overlap but left the root cause open) ·
**Raw artifacts:**
`benchmarks/results/inference/mlx-inference/2026-07-17-gather-qmm-async-serialization/`

## Summary

The direct-decode double buffer fails to overlap host graph build with GPU
execution **only on MoE models**, and the trigger is **`gather_qmm`**: MLX's
Metal backend counts the **full expert weight tensor** against its
per-command-buffer byte cap (`MLX_MAX_MB_PER_BUFFER`, default 40–50 MB), so
every MoE layer forces a command-buffer split; the resulting dozens of
in-flight tasks per step trip `eval_impl`'s scheduler backpressure
(`n_active_tasks > MAX_ACTIVE_TASKS` → `wait_for_one()`), which blocks the
`async_eval` caller for approximately the whole step. Raising the cap restores
true async overlap and yields **+22–25% decode on Qwen3-Coder-Next-4bit and
+14.5% on Qwen3.6-35B-A3B-4bit (A/B/A, greedy token streams bit-identical)**.

## Evidence chain

1. **Synthetic loop shapes** (`async-eval-overlap-probe`, ax-engine-microbench):
   a 24-iter pipelined dependent chain of large matmuls overlaps perfectly —
   injected host delay is absorbed 1:1 by the `eval` wait, and the
   `async_eval` call costs 60–140 µs (a true submit). So MLX `async_eval`
   semantics are not the problem.
2. **Model bisect** (decode-trace `AX_DECODE_TRACE_HOST_SLEEP_US` sweeps):
   Llama-3.1-8B (dense) and Qwen3.5-9B (linear-attention hybrid, no MoE)
   overlap; Qwen3-Coder-Next and Gemma-4-26B-A4B (both MoE) serialize.
   Disabling every AX custom kernel and C++ fused route on Coder-Next does
   not change this; enabling the fused router kernel (removing
   argpartition/take from the router) does not either. The serializer is
   architectural to the MoE block.
3. **Minimal repro**: adding a 24-layer
   `x = tanh(sum_k(gather_qmm(x, experts, idx)) + x @ dense)` chain to the
   probe reproduces the serialization — **even with constant, pre-evaluated
   indices** (`async_eval` call = 5.49 ms of a 5.55 ms step; injections
   beyond ~2 ms grow wall 1:1). Runtime `argpartition`-derived indices behave
   the same. Dynamic indices are NOT required; `gather_qmm`'s presence is
   sufficient.
4. **Mechanism (MLX 0.31.2 source)**: `GatherQMM::eval_gpu` itself is fully
   async (no host sync, no CPU index reads). But the Metal command encoder
   accounts every newly-seen input buffer at its **full** `data_size()`
   (`backend/metal/device.cpp`: `buffer_sizes_ += a.data_size()`; commit when
   `(buffer_sizes_ >> 20) > max_mb`, default 40–50 MB), and `eval_impl`
   (`transforms.cpp`) blocks the caller when
   `scheduler::n_active_tasks() > MAX_ACTIVE_TASKS`. A ~300–500 MB expert
   tensor per MoE layer therefore means one command buffer per layer →
   ~24–48 tasks/step → the async submit degenerates into a barrier. The
   dense/linear paths' per-layer inputs stay near the cap, so their task
   count stays under the backpressure threshold.
5. **Knob confirmation**: with `MLX_MAX_MB_PER_BUFFER=4096
   MLX_MAX_OPS_PER_BUFFER=1000`, the probe's gather shapes flip to
   overlapping (`async_eval` 5.49 ms → 0.44 ms; injections absorbed to the
   new headroom), and the real models change as measured below.

## Measured results (decode-trace, 128 steps, greedy)

| model | default caps | raised caps | Δ | parity |
| --- | ---: | ---: | ---: | --- |
| Qwen3-Coder-Next-4bit | 13.2–13.5 ms/tok (74–76 tok/s)¹ | **10.77 ms (92.9 tok/s)** @ ≥1024 MB | **+22–25%** | checksum identical |
| Qwen3.6-35B-A3B-4bit (A/B/A) | 100.3 / 101.4 tok/s | **115.0 tok/s** @ 1024 MB | **+14.5%** | checksum identical |
| Gemma-4-26B-A4B-4bit | 96.8 tok/s² | 90.4–92.6 tok/s | −4–7%² | checksum identical |

¹ Same-session runs with default caps (all-off / router-fused configs);
thermally colder default runs measured as low as 54 tok/s — the +22–25% is
against the same-hour best-case baselines, so it is conservative.
² Single uninterleaved runs an hour apart; the delta is inside this host's
observed thermal variance band. Gemma needs an interleaved A/B before any
conclusion — its per-layer expert tensors are smaller, so the default cap
may already be near its optimum.

Cap sweep on Coder-Next: 256 MB → 88.5, 512 → 91.7, 1024 → 92.8,
4096 → 92.9 tok/s (plateau at ~1024 MB).

## What this changes

- The Phase 0 conclusion "host build pays 1:1 on every path" is now
  **fixable for MoE decode** with a runtime setting, not just by shrinking
  host build. The two levers compose: overlap hides host build up to GPU
  residual; per-layer compile (Phase 1.2) shrinks what must be hidden.
- The MoE "dispatch-bound" diagnosis in `moe-bandwidth-gap.md` gets a second
  mechanism: it was not only dispatch count — a large share of the measured
  per-step wall was the async submit degenerating into a barrier. Bus-idle
  analyses on MoE should be re-run with raised caps.
- The Tier 2A "deep expert-block fusion" justification weakens further: part
  of the price it was meant to attack disappears with a config change.

## Update (same day): engine-level default SHIPPED (`AX_MLX_AUTO_BUFFER_CAPS`, default ON)

`load_weights` now counts manifest tensors above 48 MB (the MLX default cap
band); at ≥16 such tensors — dense checkpoints carry ~2 (embedding +
lm_head), MoE expert stacks push 90–150 — it raises the caps to
1024 MB / 1000 ops via `mlx_sys::set_metal_buffer_caps_env` before the first
GPU op. User-set `MLX_MAX_*_PER_BUFFER` always wins; kill switch
`AX_MLX_AUTO_BUFFER_CAPS=0`. Decision is once-per-process (MLX reads the
variables a single time at Metal device init).

**Interleaved A/B (5 reps × 256 steps targets, 3 reps control, rep-level
interleave across models, M3 Max,
`benchmarks/results/inference/mlx-inference/2026-07-17-auto-buffer-caps-ab/`):**

| model | off median | on median | ratio | parity |
| --- | ---: | ---: | ---: | --- |
| Qwen3-Coder-Next-4bit | 75.72 tok/s | 94.52 | **1.248** | identical |
| Qwen3.6-35B-A3B-4bit | 104.65 | 116.61 | **1.114** | identical |
| Gemma-4-26B-A4B-4bit | 95.53 | 95.36 | 0.998 (neutral) | identical |
| Llama-3.1-8B-4bit (control) | 50.44 | 49.81 | 0.988 (neutral band) | identical |

Gemma's earlier single-run −7% was thermal noise, as suspected: the
interleaved pairs put it at 0.998. The predicate still covers Gemma
(91 big tensors) — measured harmless there, strongly positive on the
Qwen3-Next family. Note the absolute numbers ran under heavy sustained
thermal load; the pairwise ratios are the reliable signal.

## Remaining next steps
2. **Upstream report to ml-explore/mlx** (draft below) — the honest fix is
   gather-aware accounting: count `gather_qmm`/`gather_mm` index-selected
   bytes (top_k/E of the tensor), not the full weight, toward
   `buffer_sizes_`. Needs maintainer sign-off before filing.
3. Re-run the MoE utilization numbers (Phase 0b) with raised caps.

### Upstream issue draft (not yet filed)

> **Title:** gather_qmm defeats async_eval pipelining: full weight tensor
> counted against MLX_MAX_MB_PER_BUFFER
>
> On MoE decode (batch=1), each layer's `gather_qmm` reads only
> `top_k/num_experts` of the expert tensor, but the Metal command encoder
> adds the full tensor `data_size()` to `buffer_sizes_`, so every MoE layer
> exceeds the 40–50 MB default and forces a command-buffer split. With
> ~24–48 splits per decode step, `eval_impl`'s
> `n_active_tasks > MAX_ACTIVE_TASKS` backpressure turns `async_eval` into
> a de-facto barrier — mlx_lm-style double buffering gets zero host/GPU
> overlap on MoE models while dense models overlap fine. Raising
> `MLX_MAX_MB_PER_BUFFER` to cover the expert tensor restores overlap and
> +14–25% decode on Qwen3-Next-class models (token streams bit-identical).
> Minimal repro: [probe source]. Suggested fix: account gather-indexed
> weights at their selected size, or exempt repeated read-only buffers
> already resident in the same command buffer sequence.

## Update (2026-07-17): the caps win is submit-efficiency, NOT restored overlap

A follow-up host-sleep injection sweep on Qwen3-Coder-Next-4bit with the fix
default-on re-reads *why* the caps help, and finds real residual headroom:

| config | sleep 0 total | sleep 4000 total | Δ for 4ms injected | absorbed | async_eval (GPU) @ sleep 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| caps ON (default) | 10,554 µs/tok | 13,822 | +3,268 | **732 µs (18%)** | 7,913 µs |
| caps OFF | 13,422 | 18,179 | +4,756 | 0 (DVFS penalty) | 11,015 µs |

Reading:

- **The +27% (74.5 → 94.75 tok/s) came from cheaper GPU submission, not from
  overlap.** Raising the cap collapses per-layer command-buffer splits, so the
  `async_eval` (GPU submit) bucket drops 11,015 → 7,913 µs/tok. That is the
  whole win.
- **Host/GPU overlap is still largely broken.** A 4 ms host injection sits under
  a ~7.9 ms GPU step, so an ideal double buffer would absorb nearly all of it;
  the pipeline absorbs only 0.7 ms (18%). ~2.6 ms/step of host graph build is
  still serial with GPU execution even with caps raised — the "zero-bubble"
  residual (skeleton plan I7) is real, worth up to ~2.6 ms of a 10.5 ms step
  (~20–25%) if a working double buffer hid it.
- **The residual is upstream-shaped.** The current ordering already builds
  step N+1 before waiting on N (`generate.rs::advance_direct_pipeline_with_timings`);
  that it still does not overlap means `mlx_async_eval` is not keeping the GPU
  busy across the host-build window under the shim — the same async-semantics
  surface as the buffer-accounting issue. This strengthens the case for the
  upstream MLX report (draft above): the ask should cover async_eval actually
  launching work, not only gather-aware byte accounting. A local double-buffer
  reorder cannot manufacture overlap MLX's scheduler does not provide, so I7 is
  gated on the upstream behavior, not on a runner-side change.

Artifacts: `.../scratchpad/i7-residual/` reproduction logs (caps on/off ×
sleep 0/4000).

## Reproduction

```
cargo run -p ax-engine-microbench --release --bin async-eval-overlap-probe
MLX_MAX_MB_PER_BUFFER=4096 MLX_MAX_OPS_PER_BUFFER=1000 \
  cargo run -p ax-engine-microbench --release --bin async-eval-overlap-probe
AX_DECODE_TRACE_HOST_SLEEP_US=4000 target/release/decode-trace <moe_model_dir> 128
MLX_MAX_MB_PER_BUFFER=1024 target/release/decode-trace <moe_model_dir> 128
```
