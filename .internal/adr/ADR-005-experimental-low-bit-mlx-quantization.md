# ADR-005: Experimental Low-Bit MLX Quantization

**Status**: Accepted
**Date**: 2026-05-27

---

## Context

AX Engine currently supports repo-owned MLX inference using pre-converted MLX
model artifacts and a manifest contract. The manifest validator accepts affine
quantization bits 4, 5, 6, and 8. The current runtime therefore rejects 2-bit and
3-bit artifacts even though upstream MLX and mlx-lm can represent and execute
some low-bit affine quantized layers.

The immediate pressure comes from Gemma E2B/A4B and Qwen A3B/linear-attention
decode performance. Recent AX artifacts show lower bit widths correlate with
higher direct decode throughput, but n-gram and MTP performance are not pure
weight-bandwidth measurements:

- direct mode runs one target-model step per token and is the cleanest low-bit
  signal;
- n-gram mode uses a CPU token-table drafter and then verifies with the target
  model, so low-bit weights reduce verification/fallback cost but do not improve
  the drafter itself;
- MTP has multiple independently quantizable surfaces: main model, MTP sidecar,
  draft-only LM head, and optional KV compression.

Upstream comparison:

- MLX Metal quantized kernels instantiate 2, 3, 4, 5, 6, and 8-bit affine
  variants.
- MLX `QuantizedLinear` passes `bits`, `group_size`, and `mode` directly to
  `mx.quantized_matmul`.
- mlx-lm keeps affine defaults at 4-bit group size 64, but exposes mixed recipes
  that use 2-bit or 3-bit on lower-risk layers and higher bits on selected
  sensitive layers and `lm_head`.
- MTPLX source uses 3-bit in optional TurboQuant value-cache paths and supports
  configurable draft-head quantization, but its default MTP profile uses a 4-bit
  draft-only LM head. MTPLX is therefore evidence for component experiments, not
  proof that all MTP or main-model weights should become 3-bit.
- Rapid-MLX reinforces that speculative decoding must be model and workload
  gated. Its SuffixDecoding path is default-off, greedy-only, single-request,
  profile-gated away from hybrid linear-attention models, and promoted through a
  workload tier matrix. Its DFlash path is also alias-gated and rejects MoE and
  unvalidated low-bit main models.
- Rapid-MLX's MTP support uses an extracted quantized sidecar plus runtime
  validation instead of treating MTP as part of the base model. That matches AX's
  need to label MTP sidecar, main model, draft head, and KV/cache compression
  independently.

## Decision

AX will add low-bit affine quantization only as an experimental, evidence-gated
model-artifact path.

3-bit may be admitted first, behind an explicit experimental gate, because MLX
and mlx-lm already provide execution and conversion surfaces for it. 2-bit remains
behind a stricter second gate and should start with mixed recipes only.

The current implementation session prioritizes direct mode and n-gram mode. MTP
remains in the ADR as a comparison and labeling boundary, but MTP sidecar,
draft-head, and KV-compression work is deferred until direct and n-gram evidence
exists.

AX must not add custom MLX/Metal kernels for this work. The runtime consumes MLX
quantized matmul support through the existing MLX/MLX-C boundary and validates
that model manifests describe supported packed tensor layouts.

AX must keep direct, n-gram, and MTP claims separate:

- direct mode can claim direct target-model decode throughput only;
- n-gram mode can claim effective throughput only when draft acceptance occurs and
  same-policy gates pass;
- MTP mode must identify which component uses lower bits;
- TurboQuant KV/cache experiments must stay separate from model weight
  quantization.

## Design Rules

- Keep production validation fail-closed. 2-bit and 3-bit are rejected unless an
  explicit experimental gate is active.
- Prefer mlx-lm mixed recipes over all-low-bit conversion for first experiments.
- Start with `mixed_3_4` for 3-bit and `mixed_2_6` only after 3-bit has passed
  direct and quality gates.
- Preserve the manifest as the source of truth for per-tensor quantization bits,
  group size, and mode.
- Do not infer quantization from model names such as "Speed", "Quality", or
  "3bit"; inspect manifest tensor metadata and runtime config.
- Require direct same-policy baselines before promoting any n-gram or MTP
  effective-throughput claim.
- For this session, implement and benchmark direct/n-gram slices first; do not
  spend implementation time on MTP beyond preserving labels and avoiding
  misleading claims.
- Require workload-tier evidence before promoting n-gram acceleration. At
  minimum, separate free-form chat, structured JSON/tool, and code-edit overlap
  prompts, and keep no-draft/no-accept rows labeled as fallback.
- Keep output quality and token divergence review in the benchmark artifact path.
- Keep public docs free of experimental 2-bit/3-bit support claims until the PRD
  acceptance criteria pass.

## Direct And N-gram Best Practices

Direct mode is the canonical baseline for this decision. AX must not use an
n-gram row to prove a low-bit model-weight win, because n-gram adds a draft source
and changes the effective decoding path.

- Direct rows must run with n-gram disabled and must report cold TTFT, warm
  prefix/state-cache TTFT when applicable, decode tok/s, e2e tok/s, route
  metadata, fallback counters, prefill step size, and token IDs.
- Direct low-bit promotion requires the matching 4-bit row, the same prompt and
  seed, no unsupported fallback route, and output divergence review.
- Cache reuse must be reported as a separate mechanism from weight-bit speedup.
  A warm-prefix TTFT win is valuable, but it cannot be described as a 3-bit decode
  win unless decode throughput also improves.
- N-gram rows require a matching direct same-policy baseline. The accepted draft
  tokens must explain the effective-throughput improvement.
- N-gram no-draft, zero-accept, complete-miss, or cooldown-heavy rows are fallback
  evidence, not acceleration evidence.
- N-gram default changes require workload-tier evidence. Free-form chat,
  structured JSON/tool-loop, code edit/input-output overlap, and long repeated
  context must be reported separately.
- Qwen linear-attention rows stay conservative: bounded draft length, repeated
  n-gram support before drafting, cooldown after misses, and direct fallback until
  generated output creates a usable draft source.
- Stochastic sampling rows cannot be treated as token-exact speculative wins
  unless the verifier preserves the requested sampling distribution.
- The direct/n-gram completion gate is
  `scripts/check_direct_ngram_outperformance.py`. It must pass without fallback
  or sweep-skip allowances before the direct/ngram objective can be treated as
  complete.
- A no-draft n-gram fallback row may be useful fallback-floor evidence, but it is
  not acceleration evidence and must not be used to satisfy accepted-draft
  n-gram promotion.

## Validation

Use narrow validation first:

```bash
cargo test -p ax-engine-core
cargo test -p ax-engine-mlx
cargo test -p ax-engine-bench
```

Model-dependent validation must record:

- source model and mlx-lm conversion recipe;
- manifest quantization summary;
- prompt/decode shape, seed, host, and git state;
- route metadata and fallback counters;
- direct baseline row;
- n-gram acceptance status when n-gram is enabled;
- MTP component identity when MTP is enabled.

Before any public performance documentation changes:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --quiet --no-fail-fast
```

## Consequences

- AX can test a likely direct-decode bandwidth win without claiming production
  support prematurely.
- The implementation can stay small at first: manifest validation, loader tests,
  benchmark artifact plumbing, and conversion documentation.
- n-gram and MTP claims remain honest because low-bit main weights are not
  confused with drafter quality, MTP sidecar quality, or KV compression.
- Direct and n-gram benchmark artifacts become more verbose, but the extra route,
  cache, and workload-tier fields prevent misleading speedup claims.
- 2-bit remains available for future investigation but cannot ride on a 3-bit
  implementation without separate evidence.
- AX accepts that some 3-bit artifacts may be slower than 4-bit if unpack overhead
  or quality/acceptance loss dominates bandwidth savings.

## Rejected Alternatives

### Enable 2-bit and 3-bit as normal supported bits immediately

Rejected. MLX can execute these bit widths, but AX does not yet have model-family
quality, token-divergence, n-gram acceptance, or MTP component evidence.

### Implement AX-owned low-bit Metal kernels

Rejected. Upstream MLX already owns quantized matmul kernels for these bit widths.
Custom kernels would expand maintenance and correctness risk without being needed
for the first experiment.

### Treat MTPLX 3-bit usage as proof for AX defaults

Rejected. MTPLX's 3-bit usage is component-specific and often optional. Its
default MTP profile still uses a 4-bit draft-only LM head, and local artifacts
must be inspected rather than inferred from profile names.

### Treat Rapid-MLX speculative gates as unnecessary for AX

Rejected. Rapid-MLX's SuffixDecoding and DFlash code paths both show that draft
success is workload-sensitive and can regress on unsupported model families or
quantization shapes. AX already has direct-mode and n-gram telemetry, but low-bit
promotion still needs explicit workload tiers and component labels.

### Optimize n-gram by focusing on low-bit quantization first

Rejected. N-gram's drafter is not model-weight bound. Low-bit weights may reduce
verification and fallback cost, but acceptance rate and input-output overlap
remain the dominant variables.

### Start with all-3-bit or all-2-bit conversion

Rejected. mlx-lm's low-bit recipes are mixed for a reason: selected projections
and `lm_head` receive higher precision. AX should follow the upstream conversion
shape before testing more aggressive variants.
