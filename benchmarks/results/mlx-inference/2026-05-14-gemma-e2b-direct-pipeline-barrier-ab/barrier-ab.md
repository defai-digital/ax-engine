# Gemma 4 E2B Direct Pipeline Barrier A/B (2026-05-14)

This diagnostic A/B compares production direct-pipeline decode against the
opt-in `AX_MLX_DIRECT_PIPELINE_BARRIER=1` mode introduced alongside the new
`ax_mlx_direct_pipeline_next_complete_wall_us` bucket.

The barrier mode inserts `eval(next_token_arr)` immediately after
`async_eval(next_token_arr)` in `crates/ax-engine-mlx/src/generate.rs::advance_direct_pipeline_with_timings_and_turboquant_context`,
breaking the double-buffer overlap so per-step GPU completion time is exposed
as its own bucket.

## Setup

- Model: `.internal/models/gemma-4-e2b-it-4bit`
- Prompt shape: 128 prompt tokens, 128 generation tokens
- AX policy: `direct_no_ngram_acceleration`
- Repetitions: 1 measured run after AX warmup
- Build commit: `c29c9fd098a7e2cd051d250c45e9f77522661940`
- Both rows use the same prompt (sha256 `4ebdfdf02961…`)
- `--skip-mlx-lm` (diagnostic A/B, not a published throughput artifact)

## Headline

| Metric | Production | Barrier | Δ |
|---|---:|---:|---:|
| Decode tok/s | 185.76 | 160.49 | -13.6% |
| Decode wall s | 0.6837 | 0.7913 | +15.7% |

## Per-step bucket breakdown (µs/step over 127 decode steps)

| Bucket | Production | Barrier |
|---|---:|---:|
| direct_pipeline_wall | 5,329.2 | 6,166.2 |
| forward (next graph build) | 400.8 | 363.3 |
| argmax | 0.3 | 0.0 |
| async_eval (submit + any backpressure) | 4,921.4 | 3,673.7 |
| next_complete (barrier-only GPU wait) | 0.0 | 2,126.0 |
| pending_eval (previous-step barrier) | 4.3 | 0.1 |
| pending_read | 0.2 | 0.3 |

## Reading

**1. Pure GPU step time ≈ 2.1 ms/step.** The barrier mode's
`next_complete_wall_us` (2,126 µs/step) is the explicit synchronous wait for
the just-submitted next-token GPU work to finish.

**2. Pure host submit cost is large: ≈ 3.7 ms/step.** Barrier-mode
`async_eval_wall_us` (3,673 µs/step) is the cost of `async_eval` itself once
GPU backpressure is removed (because the prior step's GPU work finished during
the barrier). Host CPU spends most of decode time encoding/dispatching the
graph, not waiting on the GPU.

**3. CPU is the bottleneck, not GPU.** Production CPU work per step (forward
401 µs + async_eval 4,921 µs ≈ 5,322 µs) far exceeds GPU work per step
(2,126 µs). `pending_eval` is essentially zero in production (4.3 µs/step),
which corroborates that GPU finishes well before the CPU is ready for the next
barrier — there is no GPU starvation to fix.

**4. Barrier mode is "only" 13.6% slower** because the recovered overlap is
small: production already pipelines roughly `min(CPU, GPU) ≈ GPU` worth of
overlap, but since CPU > GPU, the pipelining win is bounded by GPU cost.
Barrier mode loses that overlap and adds 837 µs/step (5,329 → 6,166).

**5. Production async_eval bucket is 1,250 µs/step larger than barrier-mode
async_eval.** This gap is the part of `async_eval` that was previously hidden
behind GPU progress (queue admission / lazy encoding completed by next step's
already-running GPU work). It is **not** classical backpressure — backpressure
would require GPU to be the bottleneck, which the data rules out.

## Implications for next optimization step

The original direct-pipeline-argmax-split reading attributed 92.8% of
production direct-pipeline time to `async_eval(next_token)` and pointed at two
hypotheses: MLX async-submit backpressure or the size/dependency of the next
token graph. The barrier A/B narrows it decisively:

- **Submit overhead, not backpressure.** Of the 4.92 ms/step in production
  `async_eval`, ~3.67 ms is pure host submit work and at most ~1.25 ms is
  amortizable encoding that overlaps with GPU progress. Neither is GPU work.
- **GPU-side fusion (FFN swiglu, pli-gate, RoPE+QK_norm) has at most a
  ~2.1 ms/step ceiling**: even reducing GPU step time to zero would only
  shorten the step by the part not already hidden, i.e. the 837 µs/step
  recoverable by pipelining. Throughput ceiling under that scenario is
  ~ 1000/4.07 ≈ 246 tok/s vs current 186 tok/s.
- **Theoretical max from cutting submit overhead**: if MLX submit cost dropped
  to argmax-level (negligible), the per-step floor becomes ~max(forward + tiny
  submit, GPU) ≈ max(400, 2,126) = 2,126 µs → ~470 tok/s. That is the bigger
  prize.

**Next experiments to consider** (low → high effort):
1. Cache / amortize the per-step graph: persist compiled forward via
   `mlx.compile` for the decode-1-token path, so `async_eval` stops re-encoding
   the same shapes every step. Single-shape decode is the easy case.
2. Reduce per-step op count: KV append currently produces one `slice_update`
   per layer (35 layers); collapsing or in-place updates could cut command
   buffers materially.
3. Profile `async_eval` internally (e.g. via Instruments / mach trace) to see
   if the ~3.67 ms/step lives in graph optimize, command-buffer encoding, or
   Metal CommandQueue commit.

GPU-side kernel fusion (FFN swiglu, etc.) remains valid but gates a much
smaller throughput envelope than previously assumed for this workload at this
shape on this host.

## Caveats

- Single measured run per condition; bucket noise per step is on the order of
  10s of µs but the 13.6% headline delta and the 1.25 ms `async_eval` gap are
  well above that noise floor. Confirm with `--repetitions 3` if used as
  decision evidence.
- Host: Apple M5 Max, 128 GB. Different SoCs (M2 Max baseline) may shift the
  CPU-vs-GPU balance — re-run the A/B before transferring conclusions to
  production-target hardware.
- Barrier mode is **diagnostic only** — `AX_MLX_DIRECT_PIPELINE_BARRIER` must
  remain off in any production or published throughput run, since it disables
  the pipelining the engine relies on for steady-state decode speed.

## Artifacts

- Production: `production.json`
- Barrier mode: `barrier.json`
- Both prompts: `production-prompts/`, `barrier-prompts/` (identical sha256)
