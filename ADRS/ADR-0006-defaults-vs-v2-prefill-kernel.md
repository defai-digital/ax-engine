# ADR-0006: Keep Q4_K_V2 Batch Matmul Opt-In Only

## Status

Accepted

## Context

The `AX_METAL_BATCH_Q4K_V2` opt-in kernel was introduced to test a 2-B
fragment layout for Q4_K batched matmul. Reproducible benchmarking showed the
opt-in path is faster in theory on paper but slower in practice for the current
short-prompt workloads.

## Decision

`AX_METAL_BATCH_Q4K_V2` stays **disabled by default** and is treated as an
experiment-only runtime flag (`=1`).

## Evidence

Three-model prefill A/B (M3 Max, release binary, 256 prompt tokens):

- `Qwen3-0.6B`: default 4189.5, `v2=0` 4192.8, `v2=1` 3479.3
- `Llama-3-8B`: default 385.9, `v2=0` 384.0, `v2=1` 269.6
- `Gemma3-4B`: default 811.5, `v2=0` 813.4, `v2=1` 614.6

This is a consistent regression pattern (~17–37% slower for `v2=1`) and aligns with
phase observations that loading/dequant dominates.

## Rationale

- Shipping defaults must be monotonic and safe across model families.
- The opt-in path is valuable for experiments and future tuning but currently has
  negative impact at measured operating points.
- Keeping an explicit switch avoids blocking experimentation while preventing
  accidental performance regressions for users.

## Consequences

- Existing deployments keep current prefill performance characteristics.
- Teams can still validate the v2 kernel in controlled A/B sessions.
- Profile/perf data collection focuses on `AX_METAL_BATCH_Q4K_V2=0` and default
  behavior as the production baseline.

## Rejected Alternatives

- Making v2 the global default.
  - Rejected due repeated measured regressions and lack of stable gains across
    tested families.
- Removing the v2 path entirely.
  - Rejected to preserve experimentation channel for kernel redesign work.
- Immediate RMSNorm or FA2-only rework as first fix for prefill.
  - Rejected until profiling confirms larger bottlenecks.
