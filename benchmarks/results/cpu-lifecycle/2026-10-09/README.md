# Engine step bookkeeping cost

These artifacts measure the CPU work `EngineCore::step` does around a runner
call in the decode phase: scheduling, KV accounting, request snapshots and
telemetry. The runner is a constant-time stub that emits one token per decode
item. No model weights were loaded. They are not inference throughput,
long-context model qualification, a serving soak, or evidence for any product
SKU. A real decode step runs for milliseconds or more, so a saving of tens to
hundreds of microseconds per step is at most a few percent of end-to-end step
time and usually well under one; no end-to-end speedup is claimed.

## What changed

A sampling profile of the baseline during steady decode showed the time going
to full-history copies and to map walks that scale with context and cache
size, not to scheduling decisions:

1. `KvManager::telemetry()` recomputed two sums, live-prefix request references
   and cached child edges, by walking both indexes on every step. Both are now
   kept as counters updated where the indexes change. A regression test
   compares the counters with a full recount through allocate, share, rollback,
   free and eviction.
2. `Scheduler::plan` cloned every runnable request snapshot (prompt and output
   history included) although planning only reads them. It now borrows them.
3. The step took two full snapshot passes, one to re-resolve execution-plan
   bindings and one for the scheduler input. One pass now serves both; the
   snapshot is patched with the binding it just applied. A test asserts the
   patched snapshots equal a fresh snapshot taken after the refresh.

## What remains

Per decode step the engine still copies, for every live request: the prompt and
output history once more for the request snapshot (`RequestRecord::snapshot`),
the prompt prefix plus generated tokens again to build the decode item's
`reused_prefix_token_slice` (`native_prefix_warmup_token_slice`, a runner
contract pinned by an engine test), the execution batch into `RunnerInput`, and
the block table in `sync_request_block_table` and `build_runner_input`. These
are the largest residual costs and are unchanged here; changing the warmup
slice would change what the runner receives.

## Method

```sh
bash scripts/cargo-pinned.sh run -p ax-engine-core --example engine_step_cost --release
# one configuration: <requests> <context> <measured_steps> <retained_requests>
bash scripts/cargo-pinned.sh run -p ax-engine-core --example engine_step_cost --release -- 16 32768 1000 8
```

The same example was built twice, against the baseline commit's core sources
and against the candidate sources, and the two binaries were run alternately
with the order flipped every round (4 rounds, 5 trials per invocation, 1,000
measured steps per trial, 20 trials per arm and configuration). Each
configuration is 1 or 16 live requests with a 4,096 or 32,768 token prompt, with
and without 8 finished prompts of that length retained in the prefix cache (the
state of a long-lived server). The example asserts that every measured step
decodes every live request and schedules no prefill work.

## Result (median of per-trial mean step time, interquartile range)

| Live requests | Prompt tokens | Retained prefix cache | Baseline | Candidate | Ratio |
|---:|---:|---|---:|---:|---:|
| 1 | 4,096 | none | 7.2 us (6.5-7.5) | 5.7 us (5.5-6.1) | 1.26x |
| 1 | 4,096 | 8 prompts | 10.8 us (10.4-11.3) | 5.3 us (5.3-5.6) | 2.03x |
| 1 | 32,768 | none | 18.6 us (18.4-19.3) | 12.7 us (12.4-13.3) | 1.47x |
| 1 | 32,768 | 8 prompts | 57.4 us (56.8-58.2) | 13.0 us (12.6-14.2) | 4.40x |
| 16 | 4,096 | none | 69.4 us (68.8-69.9) | 52.8 us (52.2-53.6) | 1.32x |
| 16 | 4,096 | 8 prompts | 71.2 us (69.9-73.2) | 53.7 us (53.2-54.2) | 1.33x |
| 16 | 32,768 | none | 333.9 us (310.3-346.1) | 193.5 us (188.7-198.6) | 1.73x |
| 16 | 32,768 | 8 prompts | 401.6 us (383.2-413.8) | 234.9 us (221.1-245.4) | 1.71x |

The retained prefix cache is where the telemetry change matters: the baseline
cost grows with the number of cached blocks, the candidate does not. Without
retained blocks the gain comes from the two snapshot changes. Machine power and
thermal state were not controlled.
