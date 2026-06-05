# ADR: Gemma 4 Assistant MTP Backend

## Status

**Proposed.**

## Context

AX Engine has a working MTP path for Qwen3.6-style sidecars. That path is built
around a recurrent `MtpWeights` head loaded from `mtp.safetensors`; it uses a
single MTP transformer layer and Qwen3NextAttention-specific query/gate layout.

Gemma 4 Assistant MTP is a different architecture. Public documentation
describes official assistant checkpoints that use target KV sharing, constant
position IDs, and input projection from target embedding plus hidden state.
vLLM also treats Gemma 4 assistants as a specialized MTP path rather than a
generic draft-model path.

This ADR records the implementation direction for Gemma 4 Assistant support.

## Decision

AX Engine will implement Gemma 4 Assistant as a separate speculative drafter
backend, not as an extension of the existing Qwen3-Next `MtpWeights` sidecar.

The backend will:

- load assistant artifacts through an explicit `gemma4_assistant` contract;
- validate target/assistant pairing before attach;
- default to speculative depth 1;
- use target-owned verification and correction;
- expose Gemma-specific route telemetry;
- remain opt-in until benchmark artifacts prove a default-route win.

## Alternatives Considered

| Alternative | Pros | Cons |
|---|---|---|
| **A1: Separate `Gemma4Assistant` backend** (chosen) | Correctly models the assistant architecture; fail-closed validation; avoids corrupting Qwen MTP path | Adds a new loader and forward path |
| A2: Convert Gemma assistant weights into `mtp.safetensors` | Reuses existing MTP code | Tensor semantics are incompatible; high risk of silent wrong outputs |
| A3: Generic draft-model speculation | Flexible across model families | Ignores Gemma 4 assistant KV-sharing contract; likely slower and less correct |
| A4: Wait for upstream MLX support only | Lowest local maintenance | AX cannot expose Gemma 4 assistant MTP in the repo-owned runtime |
| A5: Use n-gram only for Gemma 4 | Already implemented | Misses official assistant drafter speed path |

## Rationale

The existing MTP implementation is intentionally architecture-specific. It
infers Qwen-style MTP head dimensions from `q_proj`, expects `mtp.pre_fc_*`
tensors, and runs a Qwen3NextAttention gate. Gemma 4 Assistant checkpoints do
not share that contract. A separate backend keeps the boundary explicit and
makes attach failures observable.

Depth 1 is selected as the default because upstream guidance recommends a small
starting depth, and AX has no fresh benchmark evidence yet for larger Gemma 4
assistant depths.

Target-owned verification is required because speculative decoding preserves
quality only when the target model remains the authority for accepted output
tokens.

## Consequences

### Positive

- Avoids unsafe reuse of Qwen MTP internals.
- Makes Gemma 4 Assistant support auditable through dedicated telemetry.
- Allows incremental implementation: loader first, depth-1 draft second,
  benchmark-gated depth expansion later.
- Keeps current Qwen3.6 MTP benchmark path stable.

### Negative

- Adds a second speculative backend and more route-selection complexity.
- Requires assistant-specific weight loading and KV-sharing support.
- Requires new benchmark and packaging fixtures.

### Risks

- Incorrect KV-sharing can produce plausible but wrong drafts.
- Pair validation may initially be too strict for community-converted artifacts.
- Assistant overhead may exceed accepted-token benefit on small targets or
  creative prompts.
- Multimodal target state may expose shape/layout cases not covered by text-only
  smoke tests.

## Guardrails

- Backend must be opt-in until benchmark evidence supports promotion.
- Attach must fail closed on unknown model type, tokenizer mismatch, or unsupported
  KV layout.
- Route metadata must record whether assistant MTP was active.
- Default depth remains 1 until depth sweeps prove higher depth is beneficial.
- README performance claims require fresh artifacts with route metadata.

## Validation

Implementation is not complete until:

- loader and pair-validation unit tests pass;
- route telemetry tests pass;
- a local Gemma 4 E2B plus assistant smoke test produces valid text;
- benchmark artifacts show assistant-MTP rows with nonzero draft and accepted
  tokens;
- no baseline Gemma 4 target regression is introduced.

## Follow-ups

- Decide whether exact-pair mode should be the default for community artifacts.
- Add benchmark harness support for `ax_engine_gemma4_assistant_mtp`.
- Evaluate n-gram plus Gemma 4 Assistant stacking after pure assistant MTP is
  stable.
- Document supported assistant artifact layouts in `docs/SUPPORTED-MODELS.md`
  only after implementation and smoke validation.
