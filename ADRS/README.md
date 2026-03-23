# Architecture Decision Records

These ADRs capture the architectural decisions that shaped AX Engine's
performance work.

## Included

- `ADR-0001-metal-function-constants-for-specialization.md`
  - **Implemented.** `ComputePipeline::from_source_with_constants()` shipped in
    `crates/ax-metal/src/pipeline.rs`. Used for the TG=256 matvec pilot (which
    regressed, but proved the infrastructure works). NR2 kernels use hardcoded
    constants instead (simpler for the shipped pattern).

- `ADR-0002-profile-guided-routing-with-runtime-heuristics.md`
  - **Implemented.** Profile system shipped in `crates/ax-metal/src/profile.rs`.
    16 model profiles in `perfs/`. Runtime heuristics remain authoritative;
    profiles provide tunable inputs. Env vars override profiles for experiments.

- `ADR-0003-benchmark-gated-selective-decode-fusion.md`
  - **Proposed.** Records the v2 decision to keep the current imperative,
    pipelined decode architecture and pursue local decode fusion plus
    hazard-scoped synchronization instead of blanket rewrites.

- `ADR-0004-prefill-profiles-should-configure-routing-not-copy-llama.cpp.md`
  - **Accepted.** Extends the profile system to prefill routing while
    explicitly rejecting blind import of unsupported llama.cpp kernel params.
- `ADR-0005-prefill-profiles-should-be-structural-guidance.md`
  - **Proposed.** Confirms schema-constrained, evidence-driven prefill profile
    mapping as the safe way to "learn from llama.cpp."
- `ADR-0006-defaults-vs-v2-prefill-kernel.md`
  - **Accepted.** Keeps Q4_K v2 prefill on opt-in only after three-model A/B
    showed repeatable regressions.

## Lessons learned

- Function constants are useful infrastructure but NR2 didn't need them —
  the kernel structure change (2 rows/SG) mattered more than parameterization.
- Profiles are valuable for per-model routing (e.g., Gemma3 split-K threshold
  vs Qwen3) but should NOT override runtime shape-dependent decisions.
- The three-tier precedence (env var > profile > runtime heuristic) works well
  for systematic experimentation (PRD-0004 sweep).
- The current AX-vs-MLX gap should not be read as “AX forgot to use Metal.”
  AX already uses shared UMA buffers, Metal 3.1 compile options, function
  constants, simdgroup kernels, and pipelined decode. The remaining gap is more
  about synchronization and fusion strategy than baseline Apple-feature usage.
