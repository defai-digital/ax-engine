# Architecture Decision Records

These ADRs capture the architectural decisions that shape AX Engine.

## Current Baseline

### `ADR-0008-native-first-positioning-and-compatibility-fallback.md`

Primary baseline.

This ADR defines what AX Engine is supposed to be:

- a native-first Apple-silicon inference engine
- a product that must win on a deliberately narrow set of high-value native
  workloads
- a system where `llama.cpp` is compatibility fallback, not the product path
- a project whose benchmarks are product-governance criteria, not only local
  engineering checks
- a runtime whose edge must come from both fused execution and an UMA-first
  memory path
- a system whose performance claims must stay structurally explainable

Read this ADR first when deciding:

- whether AX should natively support a model family
- how to talk about AX externally
- what counts as native support versus compatibility coverage
- how broad or narrow AX's support surface should be

### `ADR-0007-apple-silicon-future-alignment.md`

Supporting baseline.

This ADR defines how AX's native runtime should evolve with newer Apple GPU
families. It is the future-platform and hardware/software-alignment ADR under
the product baseline set by ADR-0008.

It keeps active guidance for:

- decode-side fusion as the highest-leverage native performance priority
- Metal 4 tensors
- machine learning encoding
- cooperative tensors
- residency sets
- argument tables
- UMA-first memory-path evolution and no-copy aliasing as strategic capability
- bounded shader specialization and regime-sensitive tuning
- capability-tiered backend evolution
- GPU-native KV, MoE, and resource-management direction

## Historical ADRs

The following ADRs are retained as historical context and local lessons, but
they are no longer the primary baseline:

### `ADR-0001-metal-function-constants-for-specialization.md`

Retained lesson:

- use function constants only for bounded, high-value specialization

### `ADR-0002-profile-guided-routing-with-runtime-heuristics.md`

Retained lesson:

- profiles are routing inputs, not replacements for capability and correctness
  logic

### `ADR-0003-benchmark-gated-selective-decode-fusion.md`

Retained lesson:

- performance rollouts must remain benchmark-gated and reversible

### `ADR-0004-prefill-profiles-should-configure-routing-not-copy-llama.cpp.md`

Retained lesson:

- AX should not copy raw `llama.cpp` parameters into its own runtime policy

### `ADR-0005-prefill-profiles-should-be-structural-guidance.md`

Retained lesson:

- profile systems should encode safe structural guidance, not foreign kernel
  constants

### `ADR-0006-defaults-vs-v2-prefill-kernel.md`

Retained lesson:

- experiment-only kernels stay opt-in until they prove stable wins on the
  benchmark set that matters

## How to Read the ADR Set

Use this order:

1. ADR-0008
2. ADR-0007
3. historical ADRs only when you need their local rationale

If an older ADR conflicts with ADR-0008 or ADR-0007, treat the older ADR as
historical and follow the newer baseline.
