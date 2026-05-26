# ADR-004: Cache-Local Speculative Serving

**Status**: Accepted
**Date**: 2026-05-26

---

## Context

The project is evaluating whether ideas associated with faster hardware, such as
3D chip stacking, and quantum-inspired optimization can guide AX Engine
performance work.

The useful hardware analogy is locality: reduce data motion, keep reusable state
near compute, and avoid repeated work. AX Engine can express that in software
through scheduling, KV/cache policy, prefix reuse, graph reuse, and speculative
decode. It cannot manufacture new silicon, bypass Apple MLX/Metal kernels, or
make unmeasured "quantum" performance claims.

AX Engine already draws a clear line between MLX kernel ownership and AX-owned
runtime behavior:

- MLX owns tensor kernels for matmul, attention, normalization, RoPE, and related
  device work.
- AX owns request lifecycle, scheduling, KV/cache state, prefix reuse,
  speculative decoding, benchmark evidence, and route metadata.
- Existing internal planning uses "Quantum OP" only as an internal name for
  offline policy search, not as a runtime or product claim.

## Decision

Treat cache-local speculative serving as the authoritative performance direction
for this class of work.

AX may implement performance improvements that make runtime state more local and
reused:

- cache-aware request scheduling and route affinity;
- prefix-cache retention and physical MLX snapshot reuse;
- n-gram and MTP speculative decoding with explicit accept/reject accounting;
- bounded static policy presets selected from offline evidence;
- MLX graph, stream, and memory-policy reuse that preserves the MLX kernel
  boundary.

AX must not present this work as quantum computing, quantum ML inference, or a
hardware-level 3D manufacturing equivalent. "Quantum-inspired" techniques are
allowed only as offline classical policy search that emits reproducible
artifacts and promotes small static runtime rules after normal evidence gates.

New raw MLX/Metal custom-kernel development is disabled for this performance
track. Existing developed and active kernel paths may remain, be maintained, and
receive correctness or integration fixes. This track may use microbenchmarks to
measure scheduler, cache, speculation, sampler, existing-kernel, or MLX
call-boundary overhead, but it must not propose new custom kernels or replacing
Apple-maintained MLX/Metal kernels as an implementation route.

## Design Rules

- Keep the serving runtime deterministic. Do not add request-time annealing,
  live optimizers, stochastic policy search, or hidden adaptive state outside
  explicit cache and session state.
- Promote only static, bounded policies such as per-family speculation
  thresholds, cache-retention rules, or route-affinity heuristics.
- Keep every performance claim route-explicit: repo-owned MLX runtime,
  `mlx_lm_delegated`, `llama_cpp`, or external reference.
- Require direct same-policy baselines for speculative decode claims. Effective
  throughput wins must not be reported as raw kernel speedups.
- Treat quality, deterministic replay, fallback accounting, and route identity as
  hard constraints, not soft objective terms.
- Keep scheduler logic independent from runner-local physical buffer details. The
  scheduler may consume route metadata and cache signals, but it must not depend
  on MLX pointer identity or compressed-buffer internals.
- Do not use this ADR to justify new raw MLX/Metal custom-kernel development.
  Existing developed kernel paths can be kept and maintained, but new kernel work
  is out of scope even when a benchmark shows a bottleneck. Prefer MLX
  graph/stream policy, cache layout, scheduling, or speculation changes first.
- Keep public language neutral: use "cache-local serving", "speculative
  decoding", "prefix reuse", "offline policy search", or "autotuning". Reserve
  "Quantum OP" for internal planning only.

## Validation

Use the narrowest checks for the touched surface:

- scheduler and KV policy: `cargo test -p ax-engine-core`
- MLX runtime cache/speculation behavior: `cargo test -p ax-engine-mlx`
- benchmark artifact or claim gates: `cargo test -p ax-engine-bench` and
  `bash scripts/check-bench-doctor.sh`
- offline policy search artifacts:
  `bash scripts/check-offline-policy-search-artifacts.sh`
- script changes: `bash scripts/check-scripts.sh`
- workspace hygiene before public performance updates:
  `cargo fmt --check`,
  `cargo clippy --all-targets --all-features -- -D warnings`, and
  `cargo test --quiet --no-fail-fast`

Model-dependent evidence must record the model artifact source, route metadata,
host, command, git state, prompt/decode shape, and completed-row status.

## Consequences

- Performance work should start from telemetry and benchmark artifacts, not from
  broad algorithmic labels.
- Cache-aware batching, prefix reuse, and speculative decoding become first-class
  runtime policy surfaces, but still require explicit acceptance criteria before
  changing defaults.
- Offline search can reduce guesswork, but it cannot promote runtime behavior by
  itself. A winning search result becomes an implementation candidate, not a
  shipped feature.
- Negative search results are acceptable evidence when they show that a policy
  does not beat the baseline under correctness and quality constraints.
- AX keeps its current product boundary: high-throughput Apple Silicon serving
  above MLX, not new replacement work for MLX kernels, Metal kernels, or a
  quantum runtime.
- Existing developed kernel paths are not removed by this ADR. They remain
  governed by their current contracts and can receive maintenance, safety, and
  integration fixes.

## Rejected Alternatives

### Build a quantum-inspired inference runtime

Rejected. Current transformer inference bottlenecks are dominated by tensor
kernels, memory bandwidth, KV state, scheduling, and decode policy. Quantum or
quantum-inspired inference language would add product confusion without evidence.

### Add request-time policy search to serving

Rejected. Live search would add latency, non-determinism, and hard-to-audit
behavior. Search belongs offline; serving consumes static rules.

### Treat 3D chip manufacturing as a direct software architecture

Rejected. AX cannot reproduce hardware stacking in software. The useful lesson is
locality and reuse, expressed through cache, scheduler, and speculation policy.

### Add new MLX/Metal custom kernels as part of this performance track

Rejected. New custom-kernel development is too risky for the cache-local
speculative serving track. It expands correctness, portability, and maintenance
risk while competing with the safer runtime-policy work that AX already owns.
Existing developed and active kernel paths stay in place. Future new-kernel
experiments, if ever reopened, require a separate ADR and must not be bundled
into this work.
