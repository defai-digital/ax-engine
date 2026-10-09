# Engine step bookkeeping cost

These artifacts measure the CPU work `EngineCore::step` does around a runner
call in the decode phase: scheduling, KV accounting, request snapshots and
telemetry. The runner is a constant-time stub that emits one token per decode
item. No model weights were loaded. They are not inference throughput,
long-context model qualification, a serving soak, or evidence for any product
SKU. A real decode step runs for milliseconds or more, so a saving of tens to
hundreds of microseconds per step is at most a fraction of a percent of
end-to-end step time; no end-to-end speedup is claimed.

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

Request snapshots still deep-copy prompt and output history once per step, and
the runner input still copies each block table. Those remain the largest
residual costs and are unchanged.

## Method

```sh
bash scripts/cargo-pinned.sh run -p ax-engine-core --example engine_step_cost --release
# one configuration: <requests> <context> <measured_steps> <retained_requests>
bash scripts/cargo-pinned.sh run -p ax-engine-core --example engine_step_cost --release -- 16 32768 1000 8
```

The same example was built twice, against the baseline commit's core sources
and against the working tree, and the two binaries were run alternately. Each
configuration is 1 or 16 live requests with a 4,096 or 32,768 token prompt,
with and without 8 finished prompts of that length retained in the prefix
cache (the state of a long-lived server). Requests stay in decode for the whole
window.

- `engine-step-cost-64-step-grid.jsonl`: 64 measured steps per trial, 120
  trials per arm and configuration.
- `engine-step-cost-1000-steps.jsonl`: 1,000 measured steps per trial, 15
  trials per arm and configuration, for five configurations. The first run
  of the 64-step grid was started with a shell word-splitting mistake that
  made every invocation use the default grid; the arms were treated identically,
  so the comparison holds, and the 1,000-step run was added to confirm it.

## Result (median of per-trial mean step time)

| Live requests | Prompt tokens | Retained prefix cache | Baseline | Candidate | Ratio |
|---:|---:|---|---:|---:|---:|
| 1 | 32,768 | none | 18.5 us | 13.6 us | 1.36x |
| 1 | 32,768 | 8 prompts | 58.2 us | 12.8 us | 4.56x |
| 16 | 4,096 | none | 71.2 us | 52.1 us | 1.37x |
| 16 | 32,768 | none | 326.2 us | 198.1 us | 1.65x |
| 16 | 32,768 | 8 prompts | 391.7 us | 237.4 us | 1.65x |

(1,000-step run; the 64-step grid in `manifest.json` agrees.) The retained
prefix cache is where the telemetry change matters: the baseline cost grows
with the number of cached blocks, the candidate does not. Without retained
blocks the telemetry change is neutral and the gain comes from the two
snapshot changes. Machine power and thermal state were not controlled.
