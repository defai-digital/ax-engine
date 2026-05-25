# PRD: Runtime Model Readiness

**Status**: Active
**Date**: 2026-05-25
**Current ADR**: `.internal/adr/ADR-001-runtime-model-boundaries.md`
**Scope**: `crates/ax-engine-core`, `crates/ax-engine-mlx`, model conversion,
runtime validation, model-dependent smoke evidence

---

## Problem

AX Engine has moved past the old "split the model monolith and add top-5
families" plan. The repo now has a `model/` module tree, family-specific files
for several distinct paths, shared primitives, and conversion support for more
families than the old PRD described.

The remaining need is not another structural split plan. The project needs a
model-readiness plan that separates three claims:

- manifest and conversion support exists
- native MLX runtime path is implemented and test-covered
- local model smoke/benchmark evidence proves the family is usable on real
  artifacts

## Goals

- Keep model support fail-closed when required config fields or tensors are
  absent.
- Track model-family readiness through manifest, runtime, and evidence stages.
- Require model-dependent smoke evidence before claiming production support.
- Keep family routing explicit at the `ax-engine-core` and `ax-engine-mlx`
  boundary.
- Avoid file-layout goals that no longer match the current architecture.

## Non-Goals

- No new model family is added by this PRD alone.
- No broad rewrite of `runner.rs`, `weights.rs`, or `kv_cache.rs` unless a
  concrete model-readiness slice requires it.
- No public support-matrix claim without local artifact evidence.
- No restoration of one-file-per-family as a hard rule.

## Current Evidence

- `crates/ax-engine-mlx/src/model/` exists with shared primitives and family
  modules.
- `qwen3_5` and `qwen3_next` share the `qwen3_linear` path where linear
  attention is required.
- `gemma4`, `gemma3`, `qwen3`, `llama3`, `qwen3_5`, and `qwen3_next` can share
  the standard path when architecture allows it.
- `llama4`, `mistral3`, `mixtral`, `glm4_moe_lite`, and `deepseek_v3` have
  explicit family routing.
- `crates/ax-engine-core/src/convert.rs` recognizes the current target family
  names and aliases.

## Plan

### Phase 1: Readiness Matrix

Create a small internal readiness table that records each family as:

- manifest conversion supported
- runtime route implemented
- unit tests cover shape/config contracts
- local smoke evidence available
- benchmark evidence available
- public support docs updated

The table should live in `.internal/planning` until the evidence is stable
enough for public docs.

### Phase 2: Fail-Closed Gaps

For each family with runtime routing, audit validation paths in:

- `crates/ax-engine-core/src/model.rs`
- `crates/ax-engine-core/src/convert.rs`
- `crates/ax-engine-mlx/src/runner.rs`
- `crates/ax-engine-mlx/src/weights.rs`

Any missing tensor/config requirement should fail before MLX execution, not
panic mid-forward.

### Phase 3: Evidence-Backed Promotion

For each family promoted in docs, capture the model artifact source, host, route,
command, and result artifact. Use Hugging Face cache-backed model resolution
where practical instead of repo-local model directories.

## Acceptance Criteria

- Every publicly claimed native/direct family has manifest, runtime, and local
  smoke evidence recorded.
- Unsupported or draft-only families fail closed with actionable errors.
- Shared model paths have cross-family tests for the contracts they claim to
  share.
- Public docs do not imply benchmarked support when only conversion or draft
  runtime support exists.

## Validation

Use the narrowest relevant gate first:

```bash
cargo test -p ax-engine-core
cargo test -p ax-engine-mlx
cargo test --quiet --no-fail-fast
cargo clippy --all-targets --all-features -- -D warnings
```

Model-dependent smoke checks require local model artifacts and should document
`AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR` or the Hugging Face cache inputs used.
