# Gemma 4 Assistant MTP Technical Specification

## Overview

This spec describes how to add Gemma 4 Assistant speculative decoding to
`ax-engine-mlx` without conflating it with the existing Qwen3-Next MTP sidecar
path.

Status: the opt-in depth-1 runtime path is implemented. Larger depths, default
route promotion, and performance publication remain benchmark-gated.

The existing MTP implementation loads a single recurrent MTP layer from
`mtp.safetensors` and applies it through `mtp_head_forward`. Gemma 4 Assistant
uses a separate assistant model architecture with target KV sharing, constant
position IDs, and assistant inputs derived from target embedding plus target
hidden state. Therefore the implementation adds a new backend rather than
extending `MtpWeights`.

## Current Runtime Boundary

Relevant current files:

| File | Role |
|---|---|
| `crates/ax-engine-mlx/src/weights.rs` | Loads model weights and existing Qwen-style `MtpWeights`. |
| `crates/ax-engine-mlx/src/mtp.rs` | Runs the Qwen3-Next recurrent MTP head. |
| `crates/ax-engine-mlx/src/runner.rs` | Owns speculative decode state, verification, acceptance, telemetry, and n-gram stacking. |
| `crates/ax-engine-mlx/src/model/mod.rs` | Runs target model forward and exposes hidden-state handling. |
| `crates/ax-engine-mlx/src/generate.rs` | Prefill helpers that already return MTP history hidden state. |

The new implementation should reuse `runner.rs` verification and acceptance
logic where possible, but not the `MtpWeights` tensor contract.

## Proposed Types

Add an architecture-neutral draft backend enum:

```rust
enum SpeculativeDraftBackend {
    QwenNextMtp(MtpWeights),
    Gemma4Assistant(Gemma4AssistantWeights),
}
```

If minimizing churn is preferable, keep `ModelWeights::mtp` unchanged and add a
new optional field:

```rust
pub struct ModelWeights {
    pub mtp: Option<MtpWeights>,
    pub gemma4_assistant_mtp: Option<Gemma4AssistantWeights>,
    // existing fields...
}
```

The second form is lower risk for the first implementation because it avoids
touching all existing MTP call sites.

## Gemma4AssistantWeights

Initial fields should mirror the assistant graph, not the target graph:

```rust
pub struct Gemma4AssistantWeights {
    pub model_id: String,
    pub target_pair: Gemma4AssistantPair,
    pub token_embedding: QuantizedWeight,
    pub input_projection: QuantizedWeight,
    pub layers: Vec<Gemma4AssistantLayerWeights>,
    pub norm: MlxArray,
    pub lm_head: QuantizedWeight,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub assistant_hidden_size: usize,
    pub max_depth: usize,
}
```

Exact field names should be adjusted after inspecting the converted MLX
assistant artifact. The important requirement is that the loader names the
assistant contract explicitly as `gemma4_assistant`, not generic `mtp`.

## Artifact Layout

Support a sibling assistant directory rather than embedding assistant weights in
the target directory:

```text
target/
  config.json
  model.safetensors
  tokenizer.json
  ax_gemma4_assistant_mtp.json

assistant/
  config.json
  model.safetensors
  tokenizer.json
```

`ax_gemma4_assistant_mtp.json`:

```json
{
  "schema_version": "ax.gemma4_assistant_mtp.v1",
  "backend": "gemma4_assistant",
  "target_model_id": "gemma-4-e2b-it",
  "assistant_model_id": "gemma-4-e2b-it-assistant",
  "assistant_path": "../gemma-4-e2b-it-assistant-bf16",
  "max_depth": 1,
  "pairing": "exact"
}
```

This keeps model packaging explicit and avoids accidentally treating a target
checkpoint as an assistant.

`pairing` accepts `"exact"` or `"compatible"`. `"exact"` requires a canonical
target/assistant model-id pair. `"compatible"` allows noncanonical model ids
after tokenizer, vocabulary, and architecture checks still pass.

## Loading

### Phase 1: Config detection

Add a loader function:

```rust
fn load_gemma4_assistant_mtp(
    target_root: &Path,
    target_manifest: &NativeModelManifest,
) -> Option<Gemma4AssistantWeights>
```

It should:

1. Return `None` unless target model family is `gemma4`.
2. Read `ax_gemma4_assistant_mtp.json` if present.
3. Resolve assistant path inside allowed filesystem roots.
4. Parse assistant `config.json`.
5. Require `model_type = "gemma4_assistant"` or a separately implemented
   successor.
6. Validate tokenizer/vocab compatibility.
7. Validate target/assistant pair when `pairing` requires an exact pair.
8. Load assistant tensors.

### Phase 2: Fail-closed validation

Represent disable reasons as an enum:

```rust
enum Gemma4AssistantMtpDisableReason {
    NotGemma4Target,
    MissingConfig,
    InvalidConfig,
    UnsupportedAssistantModelType,
    TokenizerMismatch,
    VocabMismatch,
    PairMismatch,
    UnsupportedKvSharingLayout,
    WeightLoadFailed,
}
```

Store the last reason in route telemetry so users can distinguish "not
configured" from "configured but rejected".

## Assistant Forward

Add a separate module, for example:

```text
crates/ax-engine-mlx/src/gemma4_assistant_mtp.rs
```

Primary API:

```rust
pub fn gemma4_assistant_draft_tokens(
    assistant: &Gemma4AssistantWeights,
    target_weights: &ModelWeights,
    target_cfg: &ModelConfig,
    target_hidden: &MlxArray,
    last_token: u32,
    shared_cache: &Gemma4SharedKvView,
    max_depth: usize,
    sampling: &MlxSamplingParams,
    rng: &mut Xorshift64,
) -> Gemma4AssistantDraft
```

Return:

```rust
pub struct Gemma4AssistantDraft {
    pub tokens: Vec<u32>,
    pub log_probs: Vec<f32>,
    pub distributions: Vec<TokenDistribution>,
    pub draft_forward_wall_us: u32,
}
```

The first implementation may cap `max_depth` to 1 while keeping the return type
vector-based for future expansion.

## KV Sharing

Gemma 4 Assistant requires a target-KV view rather than independent assistant
prefill. Add a narrow view type instead of exposing all cache internals:

```rust
pub struct Gemma4SharedKvView<'a> {
    pub cache: &'a MlxKVCache,
    pub logical_seq_len: usize,
}
```

The assistant forward must:

- use the target cache as read-only input;
- not mutate target cache during draft generation;
- use constant assistant position ID semantics from the assistant architecture;
- fail closed if a target layer layout cannot provide the required KV view.

## Decode Integration

### Phase 1: Pure Gemma 4 Assistant MTP

Integrate in `run_mtp_decode` only when:

- `weights.gemma4_assistant_mtp.is_some()`;
- target family is `gemma4`;
- existing Qwen `weights.mtp` is `None` or the request explicitly chooses the
  Gemma backend;
- `AX_MLX_GEMMA4_ASSISTANT_MTP` is not `0`.

Draft flow:

1. Run target step or use existing skip-state to obtain primary logits and target
   hidden.
2. Sample primary token from target.
3. Run Gemma 4 Assistant drafter from target hidden plus primary/last token.
4. Verify `[primary_token] ++ assistant_draft` through the existing target
   verify path.
5. Accept/reject using existing `mtp_accept_count` semantics where distributions
   are available.
6. Commit accepted target KV only from the target verify pass.

### Phase 2: N-gram stacking

Defer n-gram stacking until pure assistant MTP has benchmark evidence. If later
enabled, n-gram should be attempted first, then assistant fills the remaining
tail, matching the existing AX hybrid policy.

## Sampling and Acceptance

For stochastic target sampling:

- assistant token selection should use the request sampling parameters unless
  assistant docs or benchmarks justify a separate draft sampler;
- store assistant draft log-probs for probability-ratio acceptance;
- compute target probabilities from verify logits using the existing
  `compute_mtp_target_probs` path;
- sample residual correction from target minus draft distribution when rejected.

For greedy target sampling:

- accept assistant drafts only when target argmax matches, unless an existing
  mathematically equivalent greedy shortcut is proven.

## Telemetry

Add a `Gemma4AssistantMtpTelemetry` struct and merge it into route decisions.

Fields:

```rust
struct Gemma4AssistantMtpTelemetry {
    enabled: bool,
    attach_failed: bool,
    disable_reason: Gemma4AssistantMtpDisableReason,
    depth: u32,
    draft_tokens: u32,
    accepted_tokens: u32,
    rejected_tokens: u32,
    corrections: u32,
    draft_forward_wall_us: u32,
    verify_forward_wall_us: u32,
    verify_eval_wall_us: u32,
}
```

Route keys should use the `ax_mlx_gemma4_assistant_mtp_*` prefix.

## Environment Controls

Use cached env helpers with `OnceLock`, following existing runner conventions:

- `AX_MLX_GEMMA4_ASSISTANT_MTP=0`
- `AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH=<n>`
- `AX_MLX_GEMMA4_ASSISTANT_MTP_REQUIRE_EXACT_PAIR=1`
- `AX_MLX_GEMMA4_ASSISTANT_MTP_DEBUG=1`

Invalid values should fall back to safe defaults.

## Tests

Focused unit tests:

- config absent returns disabled reason `MissingConfig`;
- non-Gemma target does not attempt attach;
- unsupported assistant `model_type` fails closed;
- tokenizer/vocab mismatch fails closed;
- exact pair mismatch fails closed when exact-pair mode is enabled;
- route telemetry emits enabled/disabled state;
- depth env cap clamps to supported assistant depth;
- stochastic acceptance uses finite draft log-probs;
- greedy target path falls back to argmax verification.

Smoke tests when artifacts are available:

```bash
cargo test -p ax-engine-mlx gemma4_assistant_mtp --quiet
```

Benchmark gate:

```bash
python3 scripts/bench_mlx_inference_stack.py \
  --model gemma-4-e2b-it-4bit \
  --ax-gemma4-assistant-mtp \
  --prompt-tokens 128 512 2048 \
  --generation-tokens 128 \
  --repetitions 5 \
  --cooldown 15
```

The produced AX row is labeled `ax_engine_gemma4_assistant_mtp` and must carry
nonzero `ax_mlx_gemma4_assistant_mtp_draft_tokens` before it can be used as
assistant-MTP evidence.

## Rollout Plan

### Phase 0: Documentation and fixture inventory

- Add this PRD, tech spec, and ADR.
- Inventory available local Gemma 4 assistant artifacts.
- Add fixture config files for parser tests without committing model weights.

### Phase 1: Loader and validation

- Parse assistant config.
- Implement pair validation.
- Emit attach telemetry.
- No decode behavior change yet.

### Phase 2: Depth-1 assistant draft

- Implement assistant forward for one token.
- Verify through target.
- Keep backend opt-in.

### Phase 3: Sampling correctness and telemetry

- Add draft log-probs and correction distribution.
- Add route counters and timing.
- Add smoke tests.

### Phase 4: Benchmark and default evaluation

- Run E2B/E4B and one larger Gemma 4 target.
- Compare baseline, n-gram, assistant MTP, and any hybrid variants.
- Promote only if artifacts show reliable speedup without correctness risk.

## Risks

| Risk | Mitigation |
|---|---|
| Assistant KV-sharing semantics are subtly wrong | Fail closed until a KV view is explicitly implemented and smoke-tested. |
| Generic draft-model code path changes output distribution | Use target-owned verification and probability-ratio acceptance. |
| Assistant overhead exceeds accepted-token savings | Start depth 1 and require benchmark gates. |
| Multimodal target hidden state differs from text-only assumptions | Scope assistant drafting to post-prefill text decode and validate hidden shape. |
| Pair mismatch gives low accept rate or invalid logits | Exact-pair validation and route disable reasons. |

## Open Implementation Questions

- Whether MLX assistant artifacts expose tensor names compatible with the target
  `ModelWeights` loader or require a separate loader.
- Whether assistant layers can reuse existing Gemma4 layer-forward code with a
  different config, or need a smaller dedicated forward.
- Whether 26B-A4B assistant MoE-like structure needs special routing kernels.
- Whether larger speculative depths should be assistant-recurrent or generated
  through repeated assistant calls.
