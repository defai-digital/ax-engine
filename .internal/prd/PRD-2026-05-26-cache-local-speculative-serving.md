# PRD: Cache-Local Speculative Serving

**Status**: Active
**Date**: 2026-05-26
**Current ADR**: `.internal/adr/ADR-004-cache-local-speculative-serving.md`
**Scope**: `crates/ax-engine-core`, `crates/ax-engine-mlx`,
`crates/ax-engine-bench`, `crates/ax-engine-microbench`, benchmark scripts,
runtime policy artifacts

---

## Problem

AX Engine needs a practical performance plan for the idea behind 3D hardware
locality and quantum-inspired optimization without turning either phrase into a
product claim.

The actionable problem is not "use quantum" or "copy 3D manufacturing." The
actionable problem is to reduce repeated work and data motion in Apple Silicon
LLM serving:

- avoid re-prefilling reusable prefixes;
- schedule requests so cache and route locality are preserved;
- make speculative decode faster only when accept-rate and quality evidence
  support it;
- select bounded runtime policies from reproducible offline artifacts.

## Goals

- Improve effective throughput and TTFT/TPOT on supported MLX routes by using
  cache-local scheduling, prefix reuse, and speculative decode.
- Add telemetry that connects each win to cache hits, draft acceptance, route
  identity, and fallback accounting.
- Use offline policy search only to choose small static policy candidates.
- Keep public claims route-explicit and benchmark-backed.
- Keep runtime behavior deterministic and fail-closed when policy evidence is
  absent or invalid.

## Non-Goals

- No quantum hardware, QML circuit, quantum sampler, or quantum inference path.
- No request-time annealing, Bayesian optimization, evolutionary search, or other
  live optimizer in the serving loop.
- No claim that AX replaces MLX/Metal kernels.
- No public performance update from a diagnostic or single-run artifact.
- No scheduler dependency on MLX pointer identity or runner-private compressed KV
  buffer details.
- No broad rewrite of scheduler, KV cache, MTP, n-gram, or TurboQuant in one
  slice.

## Current Evidence

- The repo-owned MLX path uses MLX through `mlx-c`; AX owns runtime behavior above
  the graph.
- `ax-engine-core` owns deterministic scheduling, request state, and logical KV
  management.
- `ax-engine-mlx` owns physical MLX KV state, prefix snapshots, n-gram
  acceleration, MTP decode, and runner telemetry.
- Existing MTP and n-gram rows show that effective throughput can improve on
  repetition-heavy workloads, but those wins are workload-sensitive and must be
  separated from raw kernel speed.
- `.internal/planning/quantum-op.md` already defines quantum-inspired work as
  offline classical policy search and rejects public quantum runtime framing.
- TurboQuant KV compression remains experimental and off by default; it may
  provide memory/locality value only after quality, fallback, and performance
  gates pass.

## Plan

### Phase 1: Telemetry Baseline

Create or extend an internal benchmark view that records the runtime causes of a
performance row:

- route identity and support tier;
- prompt/decode shape, model family, quantization, and host;
- prefix-cache hits, misses, stores, evictions, warmup tokens, and reused tokens;
- n-gram or MTP drafted tokens, accepted tokens, rejected tokens, backoff events,
  and accept rate;
- TTFT, TPOT, e2e tok/s, decode tok/s, and client-wall totals where relevant;
- fallback counts and blocked reasons for experimental paths.

The first output can be a planning/report artifact. It does not need to change
public benchmark schemas until the fields prove stable.

### Phase 2: Cache-Aware Request Grouping

Evaluate a conservative scheduler policy that prefers batches with compatible
cache and route locality when it does not violate existing fairness and memory
pressure rules.

Candidate grouping signals:

- same model and selected backend;
- compatible execution plan and route metadata;
- shared live prefix or retained prefix-cache eligibility;
- same speculation mode and bounded decode shape;
- memory pressure state that permits prefill rather than forcing decode-only
  progress.

The policy must preserve decode-first behavior and existing memory-blocked
recovery semantics unless a separate scheduler ADR changes them.

### Phase 3: Speculation Policy Hardening

For n-gram and MTP paths, add policy evidence before changing defaults:

- direct same-policy baseline rows for each candidate;
- accept/reject counters grouped by model family and prompt class;
- deterministic replay or token-exact checks where the benchmark contract
  requires them;
- fallback/backoff accounting when speculation is disabled mid-request;
- clear separation between effective throughput and raw model-kernel speed.

Candidate output should be a small static preset, such as a per-family threshold
or prompt-class backoff rule.

### Phase 4: Offline Policy Search Integration

Use the existing offline-policy-search direction only after Phase 1 telemetry
exists for the target.

Allowed search targets:

- prefix-cache retention capacity and eviction scoring;
- n-gram draft window, confidence threshold, and backoff settings;
- MTP depth or enablement when the model bundle supports it;
- benchmark matrix minimization for regression coverage;
- TurboQuant hot-window or eligible-layer policy after TurboQuant gates are
  satisfied.

Every search must emit reproducible artifacts and may promote only a static
candidate rule through the owning runtime PRD.

### Phase 5: Claim Gate and Documentation

Before a public performance claim changes, require:

- completed repeated rows with cooling or an explicit reason a shorter smoke is
  sufficient;
- matched baseline and candidate route metadata;
- artifact path and command;
- host, model source, quantization, prompt/decode shape, and git state;
- summary of quality, deterministic replay, fallback, and blocked-reason results;
- explicit caveat when the win depends on repetition-heavy or coding-shaped
  workloads.

## Acceptance Criteria

- Internal artifacts can explain whether a speedup came from prefix reuse,
  speculative acceptance, scheduling locality, graph/memory policy, or another
  named source.
- Cache-aware grouping improves or preserves TTFT/TPOT/e2e metrics on at least
  one repeated local workload without regressing a matched baseline beyond the
  benchmark noise band.
- Speculation policy changes include accept/reject/backoff telemetry and a direct
  baseline for the same model, prompt, decode shape, and route.
- Offline search outputs never change runtime defaults without a companion
  implementation slice and validation.
- Public docs do not use quantum or hardware-manufacturing language as a
  performance explanation.

## Validation

Use narrow checks while implementing each slice:

```bash
cargo test -p ax-engine-core
cargo test -p ax-engine-mlx
cargo test -p ax-engine-bench
cargo test -p ax-engine-microbench
bash scripts/check-offline-policy-search-artifacts.sh
bash scripts/check-bench-doctor.sh
bash scripts/check-scripts.sh
```

Before public benchmark or README updates:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --quiet --no-fail-fast
```

Model-dependent validation must record `AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR` or the
Hugging Face cache inputs used, and must keep local model artifacts out of git.
