# Quantum OP PRD

Status: Partially implemented; TurboQuant runtime promotion blocked
Date: 2026-05-14
Owner: AX Engine
Companions: TURBOQUANT-PROMOTION-PRD.md, MLX-RUNTIME-PERFORMANCE-PRD.md, KV-SCHEDULER-REVAMP-PRD.md, WEIGHT-QUANT-AND-SPECULATION-PRD.md

## 1. Summary

Quantum OP means **Quantum-inspired Optimization Policy**. It is an internal
planning name for offline, classical optimization of AX Engine runtime policy.
It does not imply quantum hardware, QML inference, quantum sampling, or a public
product claim.

This PRD defines how AX Engine may use quantum-inspired, annealing-style, and
other classical offline search methods to tune runtime policies without adding
runtime annealing or unverifiable optimizer behavior to the product path.

The accepted public framing remains "offline policy search" or "autotuning lab".
The search process explores discrete policy choices with validated artifacts.
The serving runtime remains deterministic, repo-owned MLX claims remain
artifact-backed, and any promoted policy must pass the same correctness,
quality, fallback, and performance gates as a hand-written policy.

The first implementation target is TurboQuant KV compression policy search:
choosing bounded combinations of hot-window size, K/V preset, eligible layer
mask, and fallback behavior for supported MLX model families. Later targets may
include n-gram speculation policy, prefix cache retention, and MoE locality, but
only after the telemetry needed to score those policies is checked in.

Current status as of 2026-05-14:

- the offline policy-search schema, builder, validator, checked-artifact gate,
  diagnostic TurboQuant KV policy harness, canonical result path, and docs are
  implemented;
- the validator now fails closed on duplicate policy identities, invalid search
  spaces, invalid output writes, missing confirmation evidence, missing
  quality/replay evidence flags, and inconsistent candidate-win evidence;
- one checked-in TurboQuant KV policy diagnostic artifact exists for
  `gemma-4-e2b-it-4bit`;
- the result is still diagnostic only and does not prove a runtime speedup;
- runner-side prefix-cache reuse was fixed for iterative-chat workloads where
  `ax-engine-core` and the MLX runner had different views of reusable prefixes.
  The fix is validated by checked-in before/after artifacts for Gemma 4 E2B
  4-bit and Qwen 3.5 9B 4-bit, plus checked-in post-fix positive evidence for
  Qwen 3.6 35B-A3B 4-bit (`ttft_growth_ratio=0.88`, 10/10 prefix-cache hits).
  GLM-4.7-Flash MLA now has checked-in default-path positive evidence: with the
  MLA prefill chunk defaulted to 16 and no opt-in restore override, the
  canonical warm-extend corpus passes 5/5 and records a real snapshot hit; the
  10-turn multi-turn chat artifact improves from the earlier blocked
  `ttft_growth_ratio=1.74` / 0 hits to `ttft_growth_ratio=0.905` / 10 hits.
  `AX_DISABLE_MLA_PREFIX_RESTORE=1` remains as a defensive kill switch if a
  future workload exposes residual drift. This remains separate from
  TurboQuant promotion;
- README performance provenance was refreshed to
  `benchmarks/results/mlx-inference/2026-05-14-ax-direct-ngram-r3/` and the
  README artifact checker validates 280 metrics plus three narrative claims;
- prefix-reuse and multi-turn KV evidence artifacts now record selected
  environment flags that can alter optimization behavior. The multi-turn
  harness distinguishes boolean fastpath flags from raw numeric/string tuning
  env such as `AX_MLX_PREFIX_CACHE_MAX_BYTES`, so artifacts do not mislabel
  numeric policy values as disabled boolean switches;
- Python extension guardrails were repaired: the embedding smoke test now uses
  the current `support_tier` constructor contract, is self-contained under the
  documented `python3 -m unittest discover -s python/tests -v` command, and the
  checked-in ABI3 extension was rebuilt from `maturin develop --release`;
- the MLA prefix-cache path was promoted from opt-in investigation to default
  restore behavior after the GLM-4.7-Flash warm-extend drift was avoided by
  chunk-aligned MLA prefill. Follow-up guardrails ensure constructor JIT warm-up
  uses the same effective chunk shape as runtime prefill;
- TurboQuant promotion artifacts now expose a `runtime_truth` surface that
  labels the real decode path, fused successes, Metal successes, fallback
  counts, fallback reason, and blocked-reason counters. The builder records this
  surface and the readiness checker reports it even for artifacts that fail the
  quality gate, so false "fused" claims and blocked runner gates are easier to
  diagnose. The quality gate now also requires the blocked-reason counters, so
  promotion artifacts cannot omit the real-runner gate surface;
- TurboQuant public/runtime promotion remains blocked because no artifact passes
  the stricter long-context fused-path quality gate with the full real-runner
  truth surface.

## 2. Naming and Best-Practice Position

Use `quantum-op.md` as the internal filename because it captures the source of
the idea: quantum-inspired optimization. In implementation, CLI flags,
artifacts, code modules, and public docs should use neutral names such as
`offline_policy_search`, `policy_search`, `autotune`, or `policy_lab`.

Best-practice principles:

- keep the runtime boring: deterministic, stateless between requests except for
  explicit cache state, and free of live optimizers;
- search offline, serve static presets;
- make quality and determinism hard constraints, not weighted objective terms;
- keep every candidate reproducible with seed, repo commit, model manifest
  digest, prompt shape, and route metadata;
- prefer small bounded search spaces before clever optimizers;
- treat negative results as useful evidence;
- require a companion PRD gate before any runtime or documentation promotion;
- never let "quantum-inspired" language outrun measured AX-owned MLX evidence.

Evidence ladder:

1. **Theory note**: explains why a discrete policy space may be worth searching.
2. **Synthetic oracle**: proves the candidate can be evaluated deterministically.
3. **Diagnostic artifact**: compares candidate policies without runtime defaults.
4. **Repeated candidate artifact**: repeats the best candidate with cooling and a
   matched baseline.
5. **Companion PRD review**: checks domain-specific promotion gates.
6. **Runtime preset**: ships only a small static rule, never the search loop.

Stop rules:

- stop when route identity is not `selected_backend=mlx`;
- stop when deterministic replay fails;
- stop when fallback accounting is missing;
- stop when a candidate win is smaller than benchmark noise;
- stop when the result only works by changing quality-sensitive semantics that
  the companion PRD has not approved.

## 3. Product Value

AX Engine already owns runtime policy above MLX kernels: scheduling, KV/cache
state, prefix reuse, n-gram acceleration, and benchmark evidence. Several
valuable optimization decisions are discrete and model-specific:

- which KV cache layers are eligible for compression;
- how much recent KV history should stay full precision;
- when n-gram speculation should draft, shrink, or back off;
- how much prefix cache capacity should favor short hot prefixes versus long
  expensive prefixes;
- when MoE expert locality work is worth promotion.

Offline search can reduce guesswork in these decisions. The user-facing value is
not that AX Engine "uses quantum". The value is better model-specific runtime
presets with proof: higher throughput or capacity, no hidden quality regression,
explicit fallback accounting, and reproducible artifacts.

## 4. Current Contract

Quantum OP inherits the active runtime and benchmark boundaries:

- Only `selected_backend=mlx` rows can support repo-owned MLX runtime
  performance claims.
- Delegated `mlx_lm_delegated` and `llama_cpp` rows may be compatibility or
  external-baseline evidence, not AX-owned MLX throughput evidence.
- TurboQuant remains experimental until ADR 0016 and
  `TURBOQUANT-PROMOTION-PRD.md` promotion gates pass.
- Runtime quantization, custom kernels, and eval-boundary changes remain governed
  by ADR 0017 / `MLX-RUNTIME-PERFORMANCE-PRD.md`.
- Weight-side rotation and sub-4-bit quantization remain closed or deferred under
  `WEIGHT-QUANT-AND-SPECULATION-PRD.md`; offline search does not reopen them.

## 5. Goals

G1. Define a stable `ax.offline_policy_search.v1` artifact schema that records
the search space, objective, constraints, candidate policies, raw measurements,
best candidate, and promotion decision.

G2. Add a validator that fails closed when the artifact mixes route boundaries,
omits baseline rows, hides fallbacks, lacks deterministic replay evidence, or
claims promotion without the companion PRD gates.

G3. Implement the first diagnostic search target for TurboQuant KV policy without
changing default runtime behavior.

G4. Keep search algorithms swappable and boring: grid search, random search,
simulated annealing, evolutionary search, or Bayesian optimization may be used,
but all must emit the same artifact shape.

G5. Promote only small static policy presets or model-specific rules. Do not add
per-token annealing, live optimizer state, or request-time search to the serving
path.

G6. Make negative results first-class. A search that concludes "no candidate
beats baseline under quality constraints" is a successful artifact.

G7. Keep the internal "Quantum OP" name isolated from public product claims. If
the artifact is used outside `.internal/`, it should be described as offline
policy search.

## 6. Non-Goals

- No quantum hardware integration.
- No runtime quantum annealing, QML circuit, Gaussian boson sampling, or quantum
  softmax/sampling replacement.
- No request-time policy optimization.
- No public claim that AX Engine is a quantum runtime.
- No runtime default change from search output alone.
- No promotion based on synthetic fixtures or microbenchmarks alone.
- No in-house checkpoint quantization pipeline.
- No automatic README performance update from a diagnostic search artifact.
- No search over sampling semantics that can change user-visible correctness
  without an explicit quality gate.

## 7. Search Targets

The search targets are candidate-discovery surfaces. They do not replace the
runtime implementation tracks below, and they cannot promote a runtime change by
themselves.

### T1. TurboQuant KV Policy Search

Status: First target.

Owned by: this PRD for search orchestration; `TURBOQUANT-PROMOTION-PRD.md` for
runtime promotion.

Candidate variables:

- `kv_preset`: `disabled`, `TurboQuantK8V4`, `TurboQuantK4V4`,
  `TurboQuantK3V4Research`;
- `hot_window_tokens`: bounded set such as `128`, `256`, `512`, `1024`;
- `eligible_layer_mask`: `full_attention_only`,
  `full_attention_excluding_sliding`, `model_supported_only`;
- `fallback_policy`: `fail_closed`, `fallback_with_accounting`;
- `quality_profile`: profile name from the TurboQuant quality gate mapping.

Hard constraints:

- `selected_backend=mlx`;
- no compression of GatedDelta recurrent state in the first target;
- unsupported layers fail closed or are explicitly counted as fallback;
- quality gate passes for any promoted candidate;
- no hidden full-history dequant path may be counted as TurboQuant acceleration.

Scoring dimensions:

- decode tok/s;
- TTFT;
- full-precision versus compressed KV read bytes;
- fallback count and fallback tokens;
- quality gate result;
- deterministic replay result;
- long-context capacity or memory-pressure improvement.

### T2. N-Gram Speculation Policy Search

Status: Deferred until telemetry is present.

Candidate variables:

- n-gram length;
- draft window size;
- accept-rate threshold;
- fallback threshold;
- prompt-class threshold;
- per-model and per-prompt-length enablement.

Entry condition:

- route metadata exposes accept-rate and rejected-token counters by prompt class;
- direct same-policy baseline rows exist for the same model, prompt, and decode
  shape.

### T3. Prefix Cache Retention Search

Status: Partially unblocked for evidence collection; still deferred as an
offline-search target.

The Phase C iterative-chat bug fix proved that runner-side physical snapshots
can remove pathological TTFT growth for FA and sliding-window models when the
runner probes its own snapshot map instead of relying only on scheduler
annotation. GLM-4.7-Flash MLA was then moved from opt-in investigation to
default-path restore after chunk-aligned MLA prefill avoided the reproduced
warm-extend fp drift. The checked-in GLM evidence now covers both canonical
warm-extend equivalence and 10-turn iterative chat with physical hits.

This improves the evidence base for prefix-cache policy work, but it is not yet
a retention-policy search. The remaining entry condition is clean physical-hit,
blocked-reason, and eviction telemetry across FA, sliding-window, and MLA-style
attention, plus cross-family retention experiments that distinguish retention
policy from the already-fixed restore mechanics.

Candidate variables:

- retained prefix cache capacity;
- eviction score weights;
- physical snapshot retention priority;
- warmup-cost threshold;
- long-prefix versus short-prefix priority.

Entry condition:

- artifacts separately report logical prefix hit, physical snapshot hit,
  physical snapshot miss, warmup tokens, store count, eviction count, and
  deterministic replay result.

### T4. MoE Locality Policy Search

Status: Future target.

Candidate variables:

- expert prefetch threshold;
- token grouping policy;
- shared-expert fast-path threshold;
- expert cache retention priority;
- router/dispatch fusion candidate flag.

Entry condition:

- model-family artifact reports per-stage or per-expert timing and shows MoE
  locality as a material bottleneck.

### T5. Benchmark Matrix Search

Status: Low-risk supporting target.

Objective:

- choose the smallest model/prompt/decode/policy matrix that catches known
  regressions while minimizing benchmark wall time.

This target may be implemented earlier than T2-T4 because it does not affect
runtime behavior.

## 8. Runtime Implementation Tracks

The following tracks translate the search targets into concrete implementation
work. Each track must keep runtime behavior deterministic and must use the
artifact contract in this PRD only as candidate-selection evidence. Promotion
still belongs to the owning companion PRD.

### R1. TurboQuant / KV Cache Compression

Priority: High.

Current state:

- ADR 0016 and `TURBOQUANT-PROMOTION-PRD.md` already define the experiment and
  promotion gates.
- CPU/reference codec, compressed-block layout, quality gate, readiness, and
  fused-decode launch metadata already exist as contract layers.
- Runtime default remains full precision unless an experimental mode is selected.

Implementation sequence:

1. Keep shadow/accounting paths available for diagnostic artifacts.
2. Complete real-path fallback accounting for every eligible and ineligible
   layer.
3. Emit candidate-policy metadata in `ax.offline_policy_search.v1` artifacts.
4. Promote only a bounded model/preset pair after the companion TurboQuant gates
   pass.

Do not:

- compress GatedDelta recurrent state in the first promoted path;
- count a full-history dequant path as TurboQuant acceleration;
- expose an aggressive preset as public behavior without model-specific quality
  evidence.

### R2. Fused Compressed Decode Attention

Priority: Very high.

This is the critical TurboQuant speedup gate. Storage compression alone can save
memory while regressing decode if the runtime dequantizes the full cold history
before attention.

Implementation sequence:

1. Treat the current fused launch descriptor and Metal prototype as kernel
   evidence, not production support.
2. Add a real-runner selection point that can report whether an eligible decode
   step selected `fused_compressed_decode`, `full_precision_hot_window`, or a
   named fallback.
3. Validate kernel output against the CPU all-head oracle for the same compressed
   buffer and query shape.
4. Add long-context repeated benchmarks for one conservative shape, initially
   `K8V4`, `head_dim=128`, full-attention layers only.
5. Promote only if fused compressed decode is selected on the real runner path
   with zero unexplained fallbacks and a measured long-context capacity or decode
   win.

Do not:

- wire a kernel into generation before oracle parity and fallback reporting are
  artifact-validated;
- generalize beyond `K8V4` or `head_dim=128` before the conservative path passes.

### R3. KV Cache Policy: Hot FP16 Window and Compressed Cold History

Priority: High.

This is the policy layer above R1/R2. It decides how recent tokens stay full
precision and which cold tokens can use compressed storage.

Implementation sequence:

1. Use offline policy search to evaluate bounded `hot_window_tokens` choices.
2. Keep hot-window-only plans independent from compressed buffer validation.
3. Track cold tokens, hot tokens, saved bytes, fallback tokens, and quality
   profile in artifacts.
4. Convert a winning search result into a static preset only after repeated
   evidence confirms it.

Initial candidate set:

- `hot_window_tokens`: `128`, `256`, `512`, `1024`;
- `eligible_layer_mask`: `full_attention_only` first;
- `kv_preset`: `disabled` baseline and `TurboQuantK8V4` first.

Do not:

- tune hot-window size per request in production;
- let the scheduler depend on runner-local compressed-storage details.

### R4. Adaptive Speculative Decoding

Priority: High.

AX already ships n-gram acceleration with confidence thresholding, prompt-class
telemetry, and EMA/backoff behavior. The next step is adaptive policy, not MTP,
unless ADR 0022 D3 prerequisites change.

Implementation sequence:

1. Keep direct same-policy rows available for every speculative benchmark.
2. Extend telemetry so accepted, rejected, drafted, disabled, and backoff tokens
   can be grouped by prompt class and model family.
3. Use offline policy search to evaluate bounded n-gram length, draft window,
   confidence threshold, and backoff settings.
4. Promote only small static defaults such as per-family confidence thresholds or
   prompt-class backoff rules.

Do not:

- add MTP scaffolding before a supported model ships an MTP head and n-gram
  artifacts show the regression class MTP would solve;
- make speculative behavior depend on non-deterministic request-time search.

### R5. MoE Routing and Expert Locality

Priority: Medium-high.

This is promising for Qwen3.6, Qwen Coder Next, GLM MoE, and Gemma MoE variants,
but it needs profiling evidence before runtime changes.

Implementation sequence:

1. Add or reuse model-family artifacts that separate router, expert gather,
   expert matmul, shared expert, and dispatch overhead.
2. Classify whether the bottleneck is expert locality, projection bandwidth,
   dispatch, or something else.
3. Only then search over expert prefetch thresholds, token grouping policy,
   shared-expert fast path threshold, and cache retention priority.
4. Promote model-family-specific policies only when per-stage artifacts show a
   repeated win.

Do not:

- implement a generic MoE scheduler rewrite before per-stage evidence exists;
- apply MoE locality policies to dense or non-MoE models;
- mix MoE locality conclusions with TurboQuant or n-gram artifacts.

### Implementation Order

The recommended order is:

1. W1 schema and validator for `ax.offline_policy_search.v1` (**done**).
2. R1/R3 diagnostic TurboQuant KV policy search, no runtime default changes
   (**done for diagnostic enumeration**).
3. Phase C runner-side prefix-cache bug fix for iterative chat (**done for
   Gemma 4 E2B 4-bit, Qwen 3.5 9B 4-bit, Qwen 3.6 35B-A3B 4-bit, and
   GLM-4.7-Flash MLA default-path positive evidence; still kept out of
   TurboQuant promotion**).
4. README/Python provenance and smoke-test guardrails (**done**).
5. R2 conservative fused compressed decode real-runner gate (**active
   blocker; Phase 1 truth surface is done, Phase 2 real-runner kernel gate
   remains**).
6. R4 adaptive n-gram policy search once telemetry is complete (**deferred**).
7. R5 MoE locality profiling, then search only if profiling identifies a real
   locality bottleneck.

This order keeps the fastest path toward long-context value while avoiding a
broad rewrite of scheduler, KV ownership, speculative decoding, and MoE routing
in one PR.

## 9. Artifact Contract

All searches emit one JSON artifact under:

`benchmarks/results/offline-policy-search/<date>/<target>-<model>.json`

The schema name is `ax.offline_policy_search.v1`.

Required top-level fields:

```json
{
  "schema": "ax.offline_policy_search.v1",
  "target": "turboquant_kv_policy",
  "status": "diagnostic_only",
  "created_at": "2026-05-14T00:00:00Z",
  "repo": {
    "commit": "required",
    "dirty": false
  },
  "model": {
    "id": "required",
    "family": "required",
    "artifacts_dir": "required",
    "manifest_digest": "required"
  },
  "route": {
    "selected_backend": "mlx",
    "support_tier": "repo_owned_runtime"
  },
  "search": {
    "algorithm": "simulated_annealing",
    "seed": 42,
    "budget": {
      "max_candidates": 64,
      "max_wall_time_seconds": 7200
    },
    "space": {}
  },
  "objective": {
    "maximize": ["decode_tok_s", "kv_saved_bytes"],
    "minimize": ["ttft_ms", "fallback_tokens"],
    "hard_constraints": [
      "quality_gate_pass",
      "deterministic_replay_pass",
      "selected_backend_mlx"
    ]
  },
  "baseline": {},
  "candidates": [],
  "best_candidate": {},
  "decision": {
    "classification": "diagnostic_only",
    "reason": "promotion gates not yet satisfied"
  }
}
```

Allowed `decision.classification` values:

- `diagnostic_only`;
- `negative_result`;
- `candidate_win_needs_repeat`;
- `promotion_ready_for_companion_prd_review`;
- `rejected_quality`;
- `rejected_fallbacks`;
- `rejected_route_boundary`;
- `rejected_noise`.

The artifact must keep raw candidate rows. Summaries without candidate rows are
not valid evidence.

## 10. Validator Rules

`scripts/check_offline_policy_search_artifact.py` should fail if:

- `schema` is not `ax.offline_policy_search.v1`;
- `route.selected_backend` is not `mlx` for runtime policy search targets;
- a delegated route is included in AX-owned throughput comparisons;
- the baseline row is missing;
- candidate rows omit prompt shape, generation tokens, seed, policy, or route
  metadata;
- fallback count or fallback tokens are missing for compression/speculation
  targets;
- quality and deterministic replay evidence are missing for any candidate marked
  beyond `diagnostic_only`;
- `promotion_ready_for_companion_prd_review` is claimed without companion gate
  references;
- repeated/cooled measurement evidence is missing for promotion-ready claims;
- the artifact was produced from a dirty repo without listing changed files.

## 11. Search Algorithm Policy

The algorithm is an implementation detail, not a product claim.

Allowed initial algorithms:

- exhaustive grid for small spaces;
- random search for medium spaces;
- simulated annealing for discrete policy spaces;
- evolutionary or cross-entropy search when candidate evaluation is noisy;
- Bayesian optimization only when the parameter space and noise model justify
  the dependency cost.

Best-practice defaults:

- start with grid or random search before simulated annealing;
- use fixed seeds;
- record rejected candidates and rejection reasons;
- prefer fewer, repeated, cooled candidates over a wide noisy sweep;
- treat non-determinism as a blocker, not as optimizer variance;
- compare against the current static policy and direct same-policy baseline.

Quantum-inspired language is acceptable only as internal background rationale
for the optimizer class. It must not appear in public performance claims.

## 12. Workstreams

### W1. Schema and Validator

Status: Implemented.

Deliverables:

- document `ax.offline_policy_search.v1`;
- implement `scripts/check_offline_policy_search_artifact.py`;
- add minimal valid and invalid fixture tests;
- add this PRD to the planning index.

Exit criteria:

- validator accepts a minimal diagnostic artifact;
- validator rejects missing baseline, wrong route, hidden fallback, and invalid
  promotion claims.
- validator rejects duplicate candidate policy IDs, invalid search-space
  dimensions, missing confirmation evidence, missing quality/replay evidence
  flags for non-diagnostic decisions, and candidate-win claims whose referenced
  candidate fails quality, replay, or fallback checks.

### W2. TurboQuant Diagnostic Search Harness

Status: Implemented for diagnostic enumeration.

Deliverables:

- add a harness that enumerates or samples T1 candidate policies;
- run only in diagnostic mode;
- write artifacts under `benchmarks/results/offline-policy-search/`;
- reuse TurboQuant readiness and quality metadata instead of duplicating it.

Exit criteria:

- one checked-in diagnostic artifact for a supported model;
- no runtime default behavior changes;
- no public README performance claim changes.

### W3. Repeated Candidate Confirmation

Status: Active blocker for TurboQuant promotion; diagnostic tooling is present,
but the real fused-path repeated measurement artifact is still missing.

Deliverables:

- rerun the best candidate and baseline with repeated, cooled measurements;
- classify noisy, negative, and candidate-win outcomes;
- hand off any promotion-ready candidate to the companion PRD.
- keep builder and validator writes atomic so failed validation cannot leave an
  invalid final artifact.

Exit criteria:

- `candidate_win_needs_repeat` can be upgraded or rejected with repeat evidence.
- the TurboQuant promotion readiness probe no longer reports `no passing
  long-context fused-path quality artifact was found` or `no passing
  long-context fused-path performance promotion artifact was found`.

### W4. N-Gram Policy Search

Status: Deferred.

Deliverables:

- prompt-class accept-rate telemetry;
- diagnostic search over n-gram policy variables;
- no runtime default change until repeated evidence supports it.

### W5. Prefix and MoE Search Targets

Status: Future, with prefix-cache evidence partially improved.

Deliverables:

- implement only after required telemetry exists;
- keep each target in a separate artifact family;
- avoid mixing prefix-cache, speculation, and MoE conclusions in one result.

## 13. Remaining Milestones

### TurboQuant KV Runtime Promotion

For the TurboQuant KV optimization path, **four blocking milestones remain**
before AX Engine can claim the TurboQuant optimization is finished:

1. **M1. Long-context fused-path performance artifact**
   Produce a real, passing artifact for an eligible model with
   `candidate_mode=turboquant-fused-experimental`, `preset=k8v4`,
   `decode_path=fused_compressed_decode`, at least one fused decode success,
   zero fused decode fallbacks, `context_tokens >= 8192`, and
   `generation_tokens >= 128`.

2. **M2. Repeated quality and replay confirmation**
   Rerun the chosen candidate and the matched baseline with repeated, cooled
   measurements. The artifact must include deterministic replay, quality gate,
   fallback accounting, medians, relative delta, and noise-band evidence.

3. **M3. Companion PRD promotion review**
   Hand the confirmed candidate to `TURBOQUANT-PROMOTION-PRD.md` and verify all
   domain-specific gates. This is where a diagnostic candidate can become a
   promotion-ready static preset or be rejected as noisy, negative, or outside
   scope.

4. **M4. Runtime and documentation promotion**
   Only after M1-M3 pass, add or enable a small static runtime preset or
   model-family rule, update public wording, and keep public performance claims
   tied to the promotion artifact.

### Non-Blocking Follow-Up Tracks

The following tracks remain, but they are **not blocking** the TurboQuant KV
promotion milestone:

- W4 adaptive n-gram policy search remains deferred until prompt-class
  accept-rate telemetry is complete.
- W5 prefix policy search remains future work. The iterative-chat snapshot probe
  fix makes physical-prefix evidence healthier for FA and sliding-window models,
  while GLM MLA evidence stays correctness-blocked. A retention-policy search
  still needs cross-family hit/miss/blocked-reason/eviction artifacts.
- W5 MoE policy search remains future work until per-stage telemetry shows a
  real expert-locality bottleneck.

### Overall Remaining Milestones

In milestone-count terms:

- **4 blocking milestones remain** for TurboQuant runtime promotion: M1 fused
  long-context artifact, M2 repeated quality/replay confirmation, M3 companion
  PRD promotion review, and M4 runtime/docs promotion.
- **3 non-blocking optimization tracks remain** after or alongside that:
  adaptive n-gram policy search, prefix-cache retention policy search, and MoE
  locality policy search.
- **0 cleanup milestones remain** for the already-completed guardrail work in
  this cycle: README provenance, prefix-reuse and multi-turn evidence
  provenance, Python smoke self-containment, checked-in ABI3 extension parity,
  and MLA prefill restore/warm-up chunk alignment are closed.

Current readiness probe result:

```json
{
  "can_make_public_support_claim": false,
  "public_docs_should_remain_experimental": true,
  "blockers": [
    "no passing long-context fused-path quality artifact was found"
  ]
}
```

## 14. Promotion Rules

Offline search cannot promote runtime behavior by itself. Promotion requires:

1. a valid `ax.offline_policy_search.v1` artifact;
2. companion PRD gates satisfied for the target domain;
3. deterministic replay pass;
4. quality gate pass where model outputs may change;
5. repeated, cooled before/after benchmark evidence;
6. explicit fallback accounting;
7. route boundary validation;
8. a small static policy preset or model-specific rule;
9. review of public docs wording before any claim update.

If any condition fails, the result remains diagnostic or negative evidence.

## 15. Risks and Mitigations

Risk: search finds noisy wins.
Mitigation: require repeated, cooled confirmation before promotion.

Risk: optimizer hides correctness loss behind throughput.
Mitigation: quality and deterministic replay are hard constraints, not objective
weights.

Risk: search expands runtime complexity.
Mitigation: only static presets or simple model-specific rules can be promoted.

Risk: quantum-inspired framing creates product confusion.
Mitigation: use "Quantum OP" only in `.internal/` planning; use "offline policy
search" in implementation and public docs.

Risk: search duplicates companion PRD gates.
Mitigation: this PRD owns candidate discovery; companion PRDs own promotion.

## 16. Active Reading Path

Read in this order:

1. this PRD for the offline search contract;
2. `TURBOQUANT-PROMOTION-PRD.md` for T1 promotion gates;
3. `MLX-RUNTIME-PERFORMANCE-PRD.md` for runtime evidence policy;
4. `KV-SCHEDULER-REVAMP-PRD.md` for prefix-cache and scheduler boundaries;
5. `WEIGHT-QUANT-AND-SPECULATION-PRD.md` for weight-side and MTP boundaries.
