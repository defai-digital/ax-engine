# AX Engine PRDs

This directory contains active product and execution plans for the current
`ax-engine_v5` workspace.

## Active PRDs

- `PRD-2026-05-25-runtime-model-readiness.md` - model-family readiness,
  manifest/runtime validation, and native MLX support evidence.
- `PRD-2026-05-25-benchmark-evidence-tooling.md` - benchmark harness,
  microbenchmark artifacts, and performance-claim evidence.
- `PRD-2026-05-25-sdk-server-interface-hardening.md` - server, SDK, Python, and
  chat/API interface hardening.
- `PRD-2026-05-26-cache-local-speculative-serving.md` - cache-aware scheduling,
  prefix reuse, speculative decode, and offline policy-search promotion gates.
- `PRD-2026-05-27-experimental-low-bit-mlx-quantization.md` - gated 3-bit/2-bit
  MLX quantization experiments, direct/n-gram/MTP evidence, and promotion gates.
- `PRD-2026-05-31-transactional-speculative-decode.md` - speculative decode
  architecture for typed draft providers, verification, cache transactions, and
  Lightning-compatible benchmark evidence.
- `PRD-2026-06-01-mtp-fused-lazy-draft-skip-state.md` - MTP decode performance
  improvement: fused lazy draft, skip-state consumption, off-by-one fix.
- `PRD-2026-06-01-mtp-optimization-phase2.md` - MTP decode, prefill, and TTFT
  optimization: adaptive n-gram gating, warmup batching, top-k softmax,
  skip-state expansion, MTP cache preservation, Metal fusion, prefill profiling.
- `PRD-2026-06-01-mtp-ttft-optimization.md` - MTP TTFT optimization: warmup
  cap (256 tokens), sliding-window post-norm, MTP shader warmup, deferred
  warmup. Targets 27B TTFT within 3% of MTPLX.
- `PRD-2026-06-01-mtp-decode-optimization.md` - MTP decode rate optimization:
  enable skip-state, top-k=128 softmax, and faster n-gram gating by default.
  Targets 10–20% decode throughput improvement.

## Policy

- PRDs describe active work: user outcome, scope, acceptance criteria,
  validation, and sequencing.
- ADRs own architecture decisions and long-lived boundaries.
- Planning notes under `.internal/planning` may hold investigation logs,
  implementation notes, or historical evidence.
- Expired or incorrect PRDs are removed during cleanup. Do not create a PRD
  archive directory for this repo.

## Required PRD Shape

Each active PRD should include:

- **Status**: Active, Blocked, Superseded, or Done.
- **Current ADR**: the architecture decision that governs the work.
- **Problem**: the user/project need being solved.
- **Goals and Non-Goals**: the exact scope.
- **Current Evidence**: what the live repo proves today.
- **Plan**: short phases or bounded slices, not a permanent backlog.
- **Acceptance Criteria**: observable completion requirements.
- **Validation**: commands, artifacts, or runtime checks that prove completion.

## Cleanup Rules

- Remove PRDs whose main phases already landed and are now better represented by
  current code plus ADRs.
- Replace PRDs that mention removed project surfaces, stale file layouts, or old
  line counts.
- Consolidate overlapping PRDs when they describe the same user-facing outcome.
- Do not keep a PRD only to preserve history; history belongs in git and
  planning notes.
