# Supported Models

AX Engine v4 is native-first and intentionally narrow in certified native scope.

## Native-First Support Strategy

AX does not treat model support as a simple yes-or-no list.

Instead, support should be understood through:

- support tier
- selected backend
- capability set

This matters because a future model may be:

- fully supported by AX native runtime
- still in native bring-up
- available only through a compatibility path
- not supported yet

## Native Platform Baseline

AX Engine v4 native scope targets Macs with Apple M4-or-newer CPU and GPU.

This means:

- M4-family and later Macs are the intended native platform
- M3 and older Macs are out of scope for v4 native support
- support-tier language should not be read as implying native support on
  pre-M4 hardware
- runtime surfaces should fail closed on pre-M4 hosts rather than attempting
  degraded execution

## Support Tiers

### `native_certified`

Meaning:

- runs on AX native runtime
- assumes the supported M4-or-newer native hardware baseline
- benchmark and replay claims map directly to AX engine behavior

### `native_preview`

Meaning:

- intended native target
- intended for the supported M4-or-newer native hardware baseline
- same-family or same-lineage bring-up is in progress
- not yet fully certified for release-grade claims

### `compatibility`

Meaning:

- request is handled through a delegated compatibility backend
- this should not be read as equivalent to AX native feature coverage or
  performance
- Phase 1 delegated execution currently covers `llama.cpp`, `vLLM`,
  `mistral.rs`, and MLX-backed paths through the SDK-owned contract
- the current `mlx` path now supports both an explicit server-backed MLX route
  and a blocking direct `mlx_lm.generate` CLI route
- a deeper repo-owned in-process MLX runtime remains future work

### `unsupported`

Meaning:

- AX does not currently have a credible path for that model request

## Current Native Target Direction

The current Phase 1 native target direction is:

- dense Qwen models, including coder variants
- dense Gemma models

Current native target direction should be read as:

- where AX native bring-up is focused
- not as a promise that every present or future family variant is already
  certified

## Future Model Generations

If a newer model appears in the future, for example a later Qwen or Gemma
generation, AX should not force an all-or-nothing answer.

Depending on readiness, that model may resolve to:

- `native_preview`
- `compatibility`
- `unsupported`

before it ever becomes `native_certified`.

## Not a Broad Compatibility Matrix

The v4 rewrite is not trying to support every architecture early.

Deferred from the main path:

- multimodal models
- hybrid-only architectures
- MoE-first support
- broad compatibility exceptions during core bring-up

## Important Note

This document describes product direction and support strategy, not a final
compatibility guarantee.
The implementation is still in progress, and support claims must be earned by
actual benchmark and validation evidence.
