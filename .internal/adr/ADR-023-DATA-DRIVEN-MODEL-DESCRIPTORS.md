# ADR-023: Data-Driven Model Descriptors Without a Runtime Graph DSL

| Field | Value |
| --- | --- |
| Status | Accepted |
| Decision date | 2026-08-12 |
| Owners | AX Engine maintainers |
| Scope | Model-family metadata, manifest evolution, load-time planning, and runtime dispatch |
| Related | [Architecture](../../docs/ARCHITECTURE.md), [Supported Models](../../docs/SUPPORTED-MODELS.md), ADR-020, ADR-022 |

## Context

AX already has the right major layers: source Hugging Face configuration is
normalized by the converter, `NativeModelManifest` stores portable structure,
`ArchitectureSpec` derives typed capabilities, and Rust/MLX implementations own
the numerical graph. The remaining model-family knowledge is spread across the
converter, manifest validation, model configuration, weight loading, and
forward dispatch as repeated string matches and compatibility defaults.

Peer inference engines also combine model configuration with registered code
implementations. Their configuration identifies shapes and capabilities; it
does not eliminate typed model code, tensor contracts, cache rules, or kernels.
AX should adopt the useful data-driven boundary without creating a second
interpreter whose behavior is difficult to audit, optimize, and certify.

## Decision

### D1 — Use a layered source of truth

Model behavior is resolved in this order:

1. Source configuration is compatibility input to conversion, not a runtime
   authority.
2. The native manifest records explicit, portable model structure.
3. An engine-owned family descriptor records canonical identity, artifact
   ownership, typed defaults, dispatch routes, and support policy.
4. Load-time validation produces typed runtime state.
5. Rust/MLX code executes numerical operators, cache behavior, and kernels.

Explicit manifest values override compatibility defaults. Artifact-provided
metadata cannot promote certification, enable a performance path, or weaken a
fail-closed runtime requirement.

### D2 — Registration and primary-runner admission are different

A known artifact may be auxiliary. `gemma4_assistant`, for example, is a known
assistant/MTP sidecar but cannot be loaded as a standalone primary runner.
Every registry row therefore carries typed MLX admission. Unknown and
auxiliary-only families fail closed at the primary runner boundary.

Support tier is also independent of admission. The quality metadata API may
return a conservative fallback for an unknown label; that does not authorize
the MLX runner to load it.

### D3 — Data describes semantics; it does not execute them

AX will not introduce a YAML/JSON expression language, arbitrary operator
graph, function-name dispatch, or model-controlled kernel selection. External
data may describe bounded schema fields such as layer kinds, dimensions,
tensor roles, routing semantics, and multimodal geometry. Compiled code owns
their interpretation and validates every combination.

Configuration is parsed and normalized at conversion or load time. Decode and
prefill hot paths operate on resolved enums and structs, not repeated file
reads, string matching, or dynamic maps.

### D4 — Preserve manifest and artifact compatibility

The current `ax.native_model.v1` schema is widely cached. AX will add a
v1-to-canonical in-memory normalizer before emitting any v2 schema. A version
change must include old-fixture loading, canonical-plan equivalence, round-trip,
and malformed-input tests. The schema constant must not simply be flipped.

### D5 — Certification and capacity policy remain engine-owned

Descriptor fields can express structural eligibility but cannot assert
numerical certification or benchmark evidence from untrusted model files.
Optimized paths still require engine-owned evidence gates.

Capacity policy follows the same rule. A future resolved plan must preserve
expert-streaming requirements: streamed expert tensors are not eagerly
materialized, and streamed layers do not use compiled MoE decode closures.

## Phased implementation plan

### Phase 1 — Centralize primary admission (implemented by this decision)

- Add typed `MlxRunnerAdmission` to every architecture registration.
- Mark all current standalone families `Primary` and
  `gemma4_assistant` `AuxiliaryOnly`.
- Make the MLX runner query the core registry instead of maintaining a second
  family allowlist.
- Keep unknown families fail-closed.
- Add registry uniqueness and auxiliary-admission regression tests.
- Preserve the `ArchitectureRegistration { ... }` source spelling until the
  smoke matrix stops source-parsing it; expose `FamilyDescriptor` as the
  forward-compatible vocabulary.

Exit criterion: core registry tests, MLX primary/unknown admission tests, and
the offline model smoke matrix all pass with no numerical-path changes.

### Phase 2 — Centralize stable family semantics

Add typed descriptor semantics for:

- dense activation (`SwiGlu`, `GeGlu`, `ReluSquared`);
- linear mixer (`None`, Qwen gated-delta, Nemotron Mamba-2);
- attention contract (standard, interleaved sliding, MLA, DeepSeek V4 sparse,
  hybrid mixer);
- MoE router and packed-weight layout;
- RMSNorm/query-scale compatibility defaults;
- think-token fallback and multimodal capability.

Migrate stable branches from core architecture derivation, MLX model config,
and manifest validation. Explicit manifest fields always win. Numerical
defaults require golden and real-model equivalence evidence; they are not
treated as cosmetic cleanup.

### Phase 3 — Resolve an immutable model plan at load time

Build a `ResolvedModelPlan` immediately after manifest admission and
validation. It contains generation kind, architecture capabilities,
layer-forward route, and typed per-layer attention/FFN/cache/mixer plans.

Forward paths dispatch on the resolved plan rather than rescanning the registry
or matching `model_family`. The family string remains for identity,
diagnostics, certification, and deliberately documented specialization.

This phase starts only after overlapping work in the model forward and weight
loader files has landed, so it does not merge unrelated performance changes.

### Phase 4 — Normalize v1, then evolve the native manifest

- Add a canonical in-memory representation and v1 adapter.
- Move legacy family-derived defaults into that adapter.
- Add golden v1 fixtures and equivalent v1/v2 resolved-plan tests.
- Introduce explicit per-layer descriptors in v2 only after compatibility is
  proven.
- Reject inconsistent or unsupported field combinations with typed errors.

No runtime migration depends on rereading the original Hugging Face
`config.json`.

### Phase 5 — Table-drive the converter compatibility boundary

Add typed converter-only profiles for upstream model-type aliases, nested
configuration layout, canonical family, tensor mapping, and prefix policy.
Config-dependent decisions remain Rust functions rather than expressions.

Preserve known canonicalization exceptions, including Qwen MoE aliases,
DeepSeek V3.2, Gemma unified packaging, conditional Nemotron Embed detection,
and the dedicated Whisper runtime. Extend the native manifest with missing
vision, audio/Whisper, and terminal-token metadata before removing remaining
runtime source-config reads.

### Phase 6 — Enforce the boundary and retire fallbacks

- Add CI checks against new unreviewed `model_family` string branches in
  execution paths, with explicit exceptions for specialization,
  certification, and performance policy.
- Add converter goldens, manifest property/fuzz tests, resolved-plan snapshots,
  KV rollback/preemption coverage, and MLX numerical equivalence tests.
- Require real-weight certification before deleting compatibility fallbacks or
  promoting optimized defaults.
- Replace the smoke script's source parser with a stable machine-readable
  registry export, then complete the `FamilyDescriptor` naming migration.

## Consequences

### Positive

- New model support becomes primarily descriptor, manifest, and test work when
  existing primitives are sufficient.
- Family admission, ownership, route selection, and support policy each have a
  single typed source.
- Hot paths become easier to optimize because decisions are resolved once.
- Cached artifacts and specialized implementations remain compatible during
  migration.

### Accepted costs

- Architecture-specific Rust remains necessary for genuinely new math, tensor
  layout, state-space mixers, multimodal towers, and kernels.
- Descriptor growth requires schema and invariant discipline.
- Removing all family branches is intentionally gradual because several encode
  numerical correctness rather than mere selection.

## Alternatives considered

| Alternative | Outcome |
| --- | --- |
| Put most execution logic in YAML/JSON | Rejected: creates an untyped runtime graph interpreter and weakens validation/certification |
| Keep family matches wherever convenient | Rejected: duplicates support knowledge and makes onboarding inconsistent |
| Treat every registered artifact as standalone | Rejected: auxiliary assistants and future sidecars have different ownership |
| Switch directly from manifest v1 to v2 | Rejected: invalidates cached artifacts and hides compatibility regressions |
| Generate all runtime code from external config | Rejected: poor fit for novel operators, kernels, cache state, and streamed experts |
