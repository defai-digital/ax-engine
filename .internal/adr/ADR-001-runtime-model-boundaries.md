# ADR-001: Runtime and Model Boundaries

**Status**: Accepted
**Date**: 2026-05-25

---

## Context

AX Engine now has a focused Rust workspace with no manager/TUI crate in the
active package set. The runtime path is:

- `ax-engine-core` owns portable scheduling, manifests, tensor roles, request
  state, conversion, and Metal contracts.
- `ax-engine-mlx` owns native MLX execution, weights, KV cache, model-family
  forward passes, runtime profiling, and MLX-specific validation.
- `mlx-sys` owns the narrow Rust binding surface to MLX C APIs.
- `ax-engine-sdk`, `ax-engine-server`, `ax-engine-py`, and external SDKs adapt
  the runtime to user-facing interfaces.

Earlier ADRs captured a model-directory migration plan and specific top-5 model
family decisions. That migration has partially landed and the exact target
layout changed. Keeping the old ADRs as active decisions is misleading because
they describe a phase plan, not the architecture that should guide new work.

## Decision

Use this ADR as the active runtime/model boundary decision.

### Core Contracts

`ax-engine-core` remains the owner of stable model and execution contracts:

- `NativeModelManifest`
- `NativeTensorRole`
- model-family normalization and manifest validation
- request, scheduler, KV-plan, and execution-plan invariants

Runtime crates may add internal config projections, but they must not redefine
or fork core manifest semantics.

### MLX Runtime

`ax-engine-mlx` owns MLX-specific execution and may organize model code around
the actual sharing in the codebase:

- family-specific files for distinct forward paths, such as `deepseek_v3`,
  `glm4_moe_lite`, `llama4`, `mistral3`, `mixtral`, and `qwen3_linear`
- shared transformer paths where the implementation is genuinely common, such
  as `standard`
- shared primitives for attention, MLA, MLP, normalization, RoPE, and linear
  attention

The goal is reviewable runtime ownership, not a strict one-file-per-family rule.
Adding an artificial family file that only forwards to shared logic is not a
useful boundary.

### Static Runtime Path

The native MLX forward path should continue to prefer direct calls and static
control flow over trait-object dispatch in hot runtime code. Abstraction is
acceptable when it removes real duplication without hiding per-family tensor
contracts or MLX graph behavior.

### Model-Family Grouping

Families may share an implementation file when their architecture and tensor
contracts are shared:

- `qwen3_5` and `qwen3_next` share the `qwen3_linear` path because both use the
  GatedDeltaNet hybrid path with model-specific MoE/config switches.
- MLA support should be treated as a shared primitive. Public/core types should
  use generic MLA naming, while older `glm_mla` implementation names may remain
  temporarily where a rename would create noise without improving behavior.
- DeepSeek and GLM MLA behavior must remain distinguishable through explicit
  config fields, tests, and tensor-role validation.

## Design Rules

- `ax-engine-core` is the only crate that should define portable model manifest
  semantics. Runtime crates can derive execution-friendly views, but they cannot
  silently change the meaning of manifest fields.
- `ax-engine-mlx` can specialize aggressively for MLX performance, but any
  model-family specialization must keep validation fail-closed when required
  tensors or config fields are absent.
- Family routing must be explicit at the runtime boundary. A new supported
  family needs a clear manifest mapping, runner validation, and a tested forward
  route.
- Shared primitives must stay architecture-aware. A shared attention, MLA, MLP,
  RoPE, or normalization helper should take explicit config rather than reading
  hidden global family state.
- Benchmarks and microbenchmarks may motivate runtime changes, but the runtime
  contract is established by code and tests, not by a single local benchmark
  result.

## Validation

Use the narrowest checks that cover the touched boundary:

- manifest and conversion changes: `cargo test -p ax-engine-core`
- MLX runtime and model-family changes: `cargo test -p ax-engine-mlx`
- workspace-level API drift: `cargo test --quiet --no-fail-fast`
- formatting and lint hygiene before handoff: `cargo fmt --check` and
  `cargo clippy --all-targets --all-features -- -D warnings`

Model-dependent smoke checks require local artifacts and should use documented
artifact environment variables instead of committing model files.

## Consequences

- New model-family work starts from the manifest/tensor contract in
  `ax-engine-core`, then adds the narrowest MLX runtime path needed.
- Shared runtime primitives are allowed, but cross-family edits need tests that
  prove the affected family contracts still hold.
- Old phase-plan targets such as "`model/mod.rs` must be under 200 lines" are no
  longer architecture rules. Large test modules should still be split when they
  impede review, but file length alone is not a correctness boundary.
- The old model-reorganization ADRs are removed rather than archived because the
  current project policy for this cleanup is to keep `.internal/adr` active and
  current.

## Rejected Alternatives

### Preserve the original model-reorganization ADRs as active

Rejected. They encode completed or changed migration details and now conflict
with the real workspace.

### Require one family file per manifest family

Rejected. Several families share meaningful runtime logic. A forwarding file is
weaker than a clear shared implementation with explicit tests.

### Introduce a general backend trait before more duplication exists

Rejected for the native MLX hot path. A trait may be useful at adapter
boundaries later, but it should not become the default model-forward mechanism.
