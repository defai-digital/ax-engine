# Gemma 4 Assistant MTP PRD

## Status

**Proposed.** This document defines product requirements for supporting Gemma 4
Assistant speculative decoding in AX Engine's MLX runtime.

## Background

AX Engine already supports Gemma 4 target inference in the repo-owned MLX
runtime. The current MTP implementation, however, is a Qwen3-Next-style
recurrent MTP head loaded from `mtp.safetensors` plus `mtplx_runtime.json`.
It expects `mtp.pre_fc_*`, `mtp.fc`, and `mtp.layers.0.*` tensors and executes a
Qwen3NextAttention-style query plus gate projection.

Gemma 4 Assistant checkpoints use a different contract. Public guidance from
Google, Hugging Face, and vLLM describes Gemma 4 MTP as a specialized assistant
drafter path:

- Google describes a lightweight drafter paired with a heavier Gemma 4 target,
  where the target verifies suggested tokens in parallel.
- Hugging Face describes Gemma 4 Assistant as a small text-only model for
  speculative decoding with Gemma 4 targets.
- Hugging Face documents Gemma 4 Assistant architecture differences: full KV
  sharing with the target, constant position IDs, and inputs built from the
  target last-token embedding plus target hidden state.
- vLLM treats Gemma 4 assistant checkpoints as Gemma 4 MTP, not as generic draft
  models, and recommends starting with a small speculative depth such as 1.

References:

- Google Gemma 4 MTP announcement:
  https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
- Hugging Face Gemma 4 Assistant docs:
  https://huggingface.co/docs/transformers/model_doc/gemma4_assistant
- Hugging Face Gemma 4 assistant model card:
  https://huggingface.co/google/gemma-4-26B-A4B-it-assistant
- vLLM MTP documentation:
  https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/mtp.md

## Best-Practice Review

The implementation should follow these practices:

1. Treat Gemma 4 Assistant as a specialized MTP drafter backend, not as a
   generic draft model and not as a Qwen3-Next `MtpWeights` sidecar.
2. Validate target and assistant pairing fail-closed. A Gemma 4 E2B assistant
   must not silently attach to an incompatible target.
3. Start with `num_speculative_tokens = 1` as the default and make larger depths
   benchmark-gated.
4. Reuse target KV where the architecture requires it; do not allocate an
   independent assistant KV history that changes the assistant semantics.
5. Keep the target model as the only source of accepted output tokens. The
   assistant proposes; target verification accepts or rejects.
6. Preserve request sampling semantics. Speculative decode must not change the
   target output distribution for the same sampling configuration.
7. Keep multimodal boundaries explicit. The assistant is text-only; the target
   may be multimodal, but draft generation starts from the target text-side
   hidden state after multimodal prefill.
8. Expose route telemetry for attach success, disable reasons, draft depth,
   accept rate, correction count, and per-step overhead.
9. Keep kill switches for the whole backend and for depth expansion.
10. Require measured evidence before promoting Gemma 4 Assistant MTP to a default
    route for any model size or quantization.

## Problem

Without native Gemma 4 Assistant support, AX can run Gemma 4 target inference but
cannot use the official Gemma 4 assistant drafter family in the MLX runtime. The
current MTP path is intentionally tuned for Qwen3.6 sidecars and will either fail
to load Gemma 4 assistant weights or, worse, produce misleading behavior if a
future conversion reshapes incompatible tensors into the existing names.

## Goals

- **G1**: Add a Gemma 4 Assistant drafter backend that is separate from the
  existing Qwen3-Next `MtpWeights` backend.
- **G2**: Support the official Gemma 4 IT assistant families: E2B, E4B,
  26B-A4B, and 31B, with room for 12B/unified assistant variants behind an
  explicit compatibility gate.
- **G3**: Use target verification for every assistant draft token, preserving
  target-model output quality and sampling semantics.
- **G4**: Reuse the existing speculative verification, correction, and telemetry
  concepts where they are architecture-neutral.
- **G5**: Make attach failures understandable through route metadata rather than
  silently falling back.
- **G6**: Provide benchmark gates before enabling Gemma 4 Assistant MTP by
  default.

## Non-goals

- Retrofitting Gemma 4 Assistant weights into the Qwen3.6 `mtp.safetensors`
  layout.
- Implementing EAGLE, DFlash, or generic draft-model speculation.
- Supporting arbitrary cross-family assistant and target pairs.
- Changing Gemma 4 target model graph correctness.
- Enabling optimistic acceptance by default.
- Publishing performance claims before fresh AX benchmark artifacts exist.

## User Impact

| User / workflow | Impact |
|---|---|
| Gemma 4 local inference users | Lower decode latency when an assistant model is available and accepted. |
| Agentic coding users | Better responsiveness on long text/code generation if accept rate offsets assistant overhead. |
| Multimodal Gemma 4 users | Text generation after multimodal prefill can benefit, while assistant remains text-only. |
| Benchmark maintainers | New route metadata can separate baseline Gemma 4, n-gram, and assistant-MTP rows. |
| Model-packaging users | Clear pair validation prevents incompatible assistant attachment. |

## Product Requirements

### R1: Separate backend identity

Add a `gemma4_assistant` speculative backend identity that is independent from
`qwen3-next-mtp`. Telemetry and route metadata must distinguish:

- no MTP available;
- Qwen-style sidecar MTP;
- Gemma 4 Assistant MTP;
- n-gram only;
- hybrid n-gram plus MTP, if supported later.

### R2: Pair validation

The runtime must validate:

- target model family is `gemma4`;
- assistant model type is `gemma4_assistant` or explicitly supported successor;
- tokenizer and vocabulary are compatible;
- target and assistant model-size pairing is known;
- assistant checkpoint has text-only drafter weights, not a target model.

If validation fails, disable assistant MTP and emit a route decision explaining
the reason.

### R3: Conservative default depth

Default assistant speculative depth is 1. Depth > 1 requires an env/config knob
and benchmark evidence.

### R4: Target-owned verification

Every assistant draft token must be verified by the Gemma 4 target. The output
stream may contain accepted assistant tokens, but the acceptance decision and
correction token must come from the target distribution.

### R5: KV-sharing semantics

Assistant forward must consume the target KV-sharing contract instead of building
an independent prefill cache. The implementation must make any unsupported cache
layout fail-closed.

### R6: Sampling correctness

For stochastic sampling, use probability-ratio acceptance or an equivalent
mathematically correct speculative sampling path. Greedy mode may use argmax
verification only when target sampling is greedy.

### R7: Telemetry

Expose route metadata:

- `ax_mlx_gemma4_assistant_mtp_enabled`
- `ax_mlx_gemma4_assistant_mtp_attach_failed`
- `ax_mlx_gemma4_assistant_mtp_disable_reason`
- `ax_mlx_gemma4_assistant_mtp_depth`
- `ax_mlx_gemma4_assistant_mtp_draft_tokens`
- `ax_mlx_gemma4_assistant_mtp_accepted_tokens`
- `ax_mlx_gemma4_assistant_mtp_accept_rate_x1000`
- `ax_mlx_gemma4_assistant_mtp_verify_forward_wall_us`
- `ax_mlx_gemma4_assistant_mtp_draft_forward_wall_us`

### R8: Kill switches

Support:

- `AX_MLX_GEMMA4_ASSISTANT_MTP=0` to disable the backend.
- `AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH=<n>` to cap depth.
- `AX_MLX_GEMMA4_ASSISTANT_MTP_REQUIRE_EXACT_PAIR=1` to reject noncanonical
  assistant-target pairs.

### R9: Benchmark publication gate

Do not update README performance claims until benchmark artifacts include:

- baseline Gemma 4 target decode;
- Gemma 4 target plus assistant MTP;
- at least E2B and one larger target;
- sampled prompts that include chat, code, tool/JSON, and long continuation;
- route metadata proving assistant MTP was active.

## Acceptance Criteria

- Focused unit tests for pair validation and route metadata.
- Focused unit tests for assistant attach disable reasons.
- Golden-shape tests for assistant config parsing.
- `cargo fmt --check`.
- `cargo test -p ax-engine-mlx --quiet` or narrower tests covering the touched
  code during development.
- `cargo clippy -p ax-engine-mlx --all-targets -- -D warnings` before merge.
- A smoke test on a local Gemma 4 E2B target plus assistant artifact when model
  artifacts are available.
- Fresh benchmark artifact before any default-route or README promotion.

## Open Questions

- Should E2B/E4B assistants be allowed as drafters for larger Gemma 4 targets, or
  only exact official assistant-target pairs?
- Should assistant depth adapt from measured accept rate, or remain a fixed
  depth until benchmarks prove adaptive value?
- Should n-gram be stacked before Gemma 4 Assistant MTP, or should the first
  implementation keep assistant MTP pure?
- How should 12B/unified assistant support be represented if its config differs
  from `gemma4_assistant`?
