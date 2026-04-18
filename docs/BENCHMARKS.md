# Benchmarks

AX Engine v4 treats benchmarking as a core engineering discipline, not an
afterthought.

## What Exists Today

The repository already contains canonical workload manifests under:

- `benchmarks/manifests/scenario/`
- `benchmarks/manifests/replay/`

These represent the initial benchmark shapes for:

- short chat
- coding
- long context
- concurrent scheduling
- live shared prefix reuse
- retained prefix reuse after cleanup
- replay and churn

The checked-in native scenario set now also includes one dense Qwen target and
one dense Gemma target, and `scripts/check-bench-native.sh` validates that
both families run through the new engine core with benchmark-visible prefill
and decode route evidence. That smoke path now also emits the repo-owned Metal
build report first into the default `build/metal` directory, or the explicit
`AX_ENGINE_METAL_BUILD_DIR` override if set, and when compiled native
artifacts are present it fails closed unless the benchmark trace and aggregate
route both show `metal_dispatch_completed`.

The repo now also carries a checked-in frozen native Tier 2 scenario matrix at
`benchmarks/manifests/matrix/frozen_native_dense_phase7.json`, and
`scripts/check-bench-matrix.sh` validates that the whole matrix executes as one
roll-up through `ax-bench matrix`.

The checked-in native replay set is also validated end to end through
`scripts/check-bench-replay.sh`, covering live-share reuse, retained-cache
reuse after cleanup, mixed live-plus-retained paths, full-prefix decode
branching, and memory-blocked prefix recovery.

AX Engine v4 benchmark commands now fail closed on pre-M4 hosts.
Native AX benchmark claims are scoped to Macs with Apple M4-or-newer CPU and
GPU. M3 and older Macs are out of scope and should not be treated as
decision-grade benchmark hosts for AX results.
`ax-bench doctor` now exposes that readiness boundary directly, so unsupported
hosts can still inspect the exact block reason or override state without
guessing from a failed benchmark invocation.

The repo now also carries compatibility example manifests under those same
`scenario/` and `replay/` directories. The delegated stepwise
`llama.cpp /completion` path still has the broadest compatibility benchmark
coverage, including submit/cancel replay and delegated prompt-cache reuse. The
checked-in scenario set now also includes blocking compatibility examples for
server-backed `vLLM`, `mistral.rs`, and MLX routes. Update their `server_url`
before running them directly, or use `scripts/check-bench-preview.sh` for the
repo-owned `llama.cpp` smoke path.

## Benchmark Philosophy

AX v4 benchmark work is built around:

- reproducibility
- explicit workload manifests
- correctness and determinism gates
- route-aware reporting
- regression comparison over anecdotes

## Current State

The benchmark assets and CLI are in early bring-up.

That means:

- the benchmark manifests are already part of the repo contract
- the CLI surface exists and runs scenario / replay workloads through the
  current deterministic engine bring-up path
- native benchmark interpretation now assumes an M4-or-newer Apple Silicon host
- `ax-bench doctor` now reports whether the local machine is fully ready for
  native AX bring-up, only temporarily allowed for CI/development bring-up via
  override, or not ready because the host or Metal toolchain is outside the
  native contract
- `ax-bench scenario` now also supports a delegated compatibility benchmark
  path through the SDK-owned backend contract
- server-backed `vLLM`, `mistral.rs`, and MLX scenario manifests currently run
  through a blocking compatibility runtime and must stay single-request
  (`shape.concurrency=1`) without delegated prefix-reuse requirements
- `llama.cpp /completion` scenario and replay manifests still run through the
  preview stepwise lifecycle exposed by the SDK-backed thin access layers,
  which is the current delegated path for multi-request replay and prompt-cache
  coverage
- execution artifacts now contain observed route and runtime-derived metrics
- outputs are still exploratory and not yet decision-grade performance numbers
- metadata-backed full-block prefix reuse now retains prompt prefixes across
  request cleanup through a hash-addressed retained block cache
- retained cached blocks deduplicate identical full-block content and evict
  through LRU-style block metadata when capacity is needed
- live request sharing and retained prompt-prefix reuse are both resolved
  through full-block metadata rather than raw prompt scans
- replay coverage now includes live request sharing, retained prompt-prefix
  reuse after cleanup, and mixed live-plus-retained route aggregation
- replay manifests can now also pin `runtime.kv_total_blocks` so deterministic
  workloads can force memory pressure and validate blocked-request recovery
- route artifacts now expose branch-oriented crossover decisions such as reused
  prompt tokens and whether a reused request continued via prefill tail or
  decode
- route artifacts and step traces now also preserve blocked-prefix-reuse
  evidence when a request hits prefix reuse before capacity is available for
  execution
- execution artifacts now also include `trace.json`, which records step-level
  scheduler, request-selection, prefix-reuse details, and native
  `metal_dispatch` summaries when the bring-up runner executes real on-device
  Metal work; those route/summary artifacts now distinguish between
  `metal_complete_model_forward_supported` and `metal_real_model_forward`
  so bring-up review can tell runtime capability from completed execution, and
  the step trace now also carries direct decode-logit resolution / real-forward
  completion evidence alongside the runtime summary, plus multilayer prefix
  native-vs-CPU dispatch counts for partial-forward coverage analysis, together
  with model-side RMSNorm+QKV / o_proj+FFN token counts and final logits
  projection + vocab-scan coverage for decode review
- metrics now also record `memory_blocked_steps` and
  `memory_blocked_request_events`, so replay artifacts can distinguish healthy
  blocked-request recovery from allocator churn failures
- benchmark manifests and execution artifacts now carry explicit backend
  resolution metadata such as `selected_backend`, `support_tier`, and
  `resolution_policy`, so native preview results do not get confused with later
  compatibility paths
- delegated benchmark runs now also record backend-adapter identity and a
  distinct compatibility tool mode, so delegated compatibility measurements do
  not get compared as if they were native scheduler traces; the current
  server-backed blocking path records `compatibility_blocking_runtime`, while
  the stepwise `llama.cpp /completion` path records
  `compatibility_stepwise_runtime`
- `environment.json` now records structured run provenance including manifest
  fingerprint, subcommand, output root, cwd, start/finish timing, host model,
  SOC, memory capacity, OS version/build, and kernel release, and compare
  rejects key environment mismatches before producing regressions
- benchmark runtime metadata now also carries the SDK-owned host and
  Metal-toolchain diagnostics, so artifact review can distinguish support-tier
  intent from actual local tool availability
- compare-time environment validation now also rejects host-override or
  Metal-toolchain drift, so native benchmark regressions do not silently mix
  different bring-up readiness states
- route and environment artifacts now also record explicit
  `prefix_reuse_provenance`, and delegated runs now also surface
  `backend_reported_cached_prompt_tokens`, so compare can fail closed if
  delegated prompt-cache evidence is accidentally lined up against AX-native
  prefix reuse or against a different prompt-cache hit depth
- compare artifacts now also carry resolved runtime identity such as
  `tool_mode`, `selected_backend`, `support_tier`, `resolution_policy`, and
  backend-adapter identity, so compatibility compares do not get mislabeled as
  native engine bring-up compares
- engine step metrics now also carry measured CPU and runner timing, so
  `cpu_time_per_token_us`, `runner_time_per_token_us`, and step traces are
  grounded in actual bring-up execution timing rather than only synthetic step
  counts
- route artifacts and per-step traces now carry bound execution-plan identities
  such as `phase1.<model>.dense_prefill` and `phase1.<model>.paged_decode`,
  and their static route labels now come from the same execution-plan binding
  source instead of scheduler-local stubs; deterministic runner output now
  preserves that bound route identity instead of synthesizing its own fallback
  labels, which makes bring-up replay routes explainable step by step
- aggregate `routes.json` now fails closed on multi-route runs by marking
  `execution_plan` / `attention_route` as mixed and recording route-variant
  counts, instead of silently latching the first executed route
- mixed prefill/decode queues are now emitted as route-homogeneous steps, and
  the engine performs a same-step fallback replan when a preferred prefill
  cohort is fully blocked on memory
- replay workloads can now force full prompt identity when needed, so direct
  decode-after-reuse behavior is covered as a first-class benchmark path
- delegated benchmark execution now uses one shared SDK session for
  multi-request `llama.cpp /completion` scenario and replay runs, so delegated
  step traces reflect aggregated session progress instead of a per-request
  round-robin workaround
- blocking compatibility benchmarking for delegated server-backed adapters
  such as `vLLM`, `mistral.rs`, and MLX-backed routes is currently limited to
  scenario manifests with `shape.concurrency=1` and
  `checks.require_prefix_reuse=false`; replay, multi-request, and shared-prefix
  compatibility coverage still belongs to the stepwise `llama.cpp` path
- compatibility replay benchmarks can now also measure delegated prompt-cache
  reuse when `llama.cpp` reports cached prompt tokens; those artifacts label the
  route as `delegated_prompt_cache`, record `delegated_cached_tokens`, and now
  label that reuse source explicitly as
  `backend_reported_cached_prompt_tokens`; `routes.json`,
  `environment.json`, and compare summaries now also surface the reported
  cached-token count directly
- compatibility `shared_prefix` scenario shapes can now execute through the
  delegated `/completion` path when the backend reports prompt-cache progress,
  but those results must still be read as delegated prompt-cache evidence
  rather than AX-native scheduler / KV prefix sharing
- `ax-bench` still emits `contract_failure.json` plus `summary.md` for other
  unsupported delegated workload shapes, so the rejected boundary remains
  auditable without inventing execution metrics; the failure artifact carries a
  stable `code`, `recommended_action`, and `tool_mode`
- `ax-bench baseline` can now snapshot one successful benchmark artifact
  directory into a named trusted baseline that copies the execution artifacts
  forward, emits `trusted_baseline.json` plus `trusted_baseline.md`, and fails
  closed if that baseline name already exists
- `ax-bench matrix` can now execute the checked-in frozen native scenario
  matrix under one result directory, emit `matrix.json` plus `summary.md`, and
  preserve per-member result directories for later compare or baseline work
- `ax-bench matrix-compare` can now compare two successful matrix result
  directories from the same frozen manifest fingerprint, emit
  `matrix_regression.json` plus `summary.md`, and preserve per-member compare
  directories so frozen-matrix regressions stay auditable
- compare output now also carries trusted-baseline identity when the baseline
  side came from one of those named snapshots, so regression review stays tied
  to a deliberately frozen reference instead of an arbitrary previous run
- delegated step traces should still be read as compatibility request cadence
  plus backend-reported prompt-cache evidence, not as native scheduler, KV, or
  runner evidence
- real Metal execution now surfaces in step traces through `metal_dispatch`;
  deeper accelerator-backend provenance is still being implemented

## Intended Outcome

The benchmark system is meant to support:

- engine regression detection
- replay validation
- scheduler and KV evaluation
- later performance hardening on the dense path
