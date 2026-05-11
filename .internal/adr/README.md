# AX Engine v4 ADR Index

Status: Active
Date: 2026-05-09

This index separates active architecture decisions from superseded historical
records. Superseded ADRs are preserved for context, but new implementation work
should follow the active ADR chain first.

## Active ADRs

- `0001-product-boundary.md` — product boundary and scope discipline.
- `0002-unified-engine-core-loop.md` — core request execution loop.
- `0003-paged-kv-and-prefix-caching.md` — KV and prefix-cache contract.
- `0004-metal-runner-boundary.md` — runner boundary and implementation rules.
- `0005-benchmark-contract-and-metrics.md` — benchmark validity and metrics.
- `0006-benchmark-lab-before-autotuning.md` — benchmark lab before tuning.
- `0007-request-state-machine.md` — request lifecycle contract.
- `0008-thin-client-access-surfaces.md` — server/SDK/Python as thin access layers.
- `0012-retire-ax-native-and-route-mlx-or-llama.md` — current routing contract.
- `0013-mlx-kv-cache-improvement-strategy.md` — KV cache improvement strategy.
- `0014-mlx-lm-delegated-compatibility-backend.md` — explicit `mlx_lm_delegated` route.
- `0015-installation-runtime-dependencies.md` — runtime dependency strategy.
- `0016-experimental-turboquant-kv-compression.md` — TurboQuant experiment boundary.
- `0017-mlx-runtime-optimization-governance.md` — consolidated evidence-first MLX optimization policy.
- `0018-kv-cache-and-scheduler-revamp-strategy.md` — bounded KV/scheduler revamp strategy.

## Proposed ADRs

- `0019-cli-tui-workflow-cockpit.md` — proposed local CLI TUI workflow cockpit boundary.
- `0022-decode-bandwidth-weight-quant-and-speculation-boundary.md` — weight-side quantization, MTP, and FlashDecoding boundary.
- `0024-gemma-qwen-mlx-performance-strategy.md` — Gemma/Qwen MLX performance interpretation, profiling, and memory-attribution strategy.

## Recently Accepted (move into Active on next index sweep)

- `0023-deepseek-v4-delegated-ds4-route.md` — native family scope (Qwen / Gemma / GLM); DeepSeek V4 fail-closed; no delegated `ds4` route.

## Superseded ADRs

- `0009-native-first-backend-strategy.md` — superseded by ADR 0012.
- `0010-mlx-compatibility-first-and-defer-in-process-runtime.md` — superseded by ADR 0012.
- `0011-model-promotion-and-default-routing.md` — superseded by ADR 0012.

## Archived Planning ADR Drafts

The proposed planning ADR drafts for sync batching, dense-weight quantization,
small-op fusion, GatedDelta profiling, and embedding optimization were consolidated
into ADR 0017. They are archived under:

- `../archive/superseded/2026-05-09-planning-consolidation/planning-adr/`

## Current Reading Order

1. `0001-product-boundary.md`
2. `0012-retire-ax-native-and-route-mlx-or-llama.md`
3. `0014-mlx-lm-delegated-compatibility-backend.md`
4. `0017-mlx-runtime-optimization-governance.md`
5. `0024-gemma-qwen-mlx-performance-strategy.md` when working on Gemma/Qwen MLX performance gaps
6. `0018-kv-cache-and-scheduler-revamp-strategy.md` when working on KV, prefix reuse, or scheduler policy
7. `0016-experimental-turboquant-kv-compression.md` when working on TurboQuant
8. `0019-cli-tui-workflow-cockpit.md` when working on CLI TUI or local workflow orchestration
