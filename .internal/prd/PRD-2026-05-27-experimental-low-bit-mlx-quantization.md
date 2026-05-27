# PRD: Experimental Low-Bit MLX Quantization

**Status**: Active
**Date**: 2026-05-27
**Current ADR**: `.internal/adr/ADR-005-experimental-low-bit-mlx-quantization.md`
**Scope**: `crates/ax-engine-core`, `crates/ax-engine-mlx`,
`crates/ax-engine-bench`, `scripts`, MLX model manifests, benchmark artifacts
**Current implementation focus**: direct mode and n-gram mode. MTP remains a
labeling and comparison boundary for this PRD, not a session implementation
target.

---

## Problem

AX Engine currently accepts affine quantized MLX model artifacts only at 4, 5, 6,
and 8 bits. MLX and mlx-lm already have upstream support for lower affine bit
widths, including 2-bit and 3-bit quantized matmul paths and mlx-lm mixed
quantization recipes.

The project needs an evidence-backed path to test whether 3-bit, and possibly
2-bit, can improve decode throughput for Gemma E2B/A4B and Qwen A3B/linear
attention models without weakening correctness, benchmark labeling, or model
quality claims. The decision must separate direct decode, n-gram effective
throughput, and MTP draft-head or sidecar quantization because each path has a
different bottleneck.

## Goals

- Enable a guarded experimental path for 3-bit affine MLX model artifacts.
- Compare AX behavior against the existing MLX and mlx-lm low-bit contracts
  before changing any runtime defaults.
- Measure direct decode first, because direct mode is the cleanest signal for
  model-weight bandwidth wins.
- Measure n-gram separately, because n-gram speed depends on draft acceptance and
  workload overlap, not just target-model matmul speed.
- Keep MTP draft-head and sidecar impacts separated from main-model quantization,
  but defer MTP implementation work until direct and n-gram gates complete.
- Keep 2-bit behind stricter quality and promotion gates until 3-bit has passed
  direct and n-gram checks.

## Non-Goals

- No public support claim for 2-bit or 3-bit models before benchmark and quality
  gates pass.
- No new MLX or Metal kernels in AX for this work. AX should rely on upstream MLX
  quantized matmul support.
- No default conversion to 2-bit or 3-bit in SDK, server, or benchmark scripts.
- No claim that 3-bit improves n-gram acceptance. It can only reduce target-model
  verification/fallback cost unless the generated token stream remains stable.
- No MTP benchmark promotion that conflates main-model quantization, MTP sidecar
  quantization, draft-only LM head quantization, and TurboQuant KV compression.
- No MTP implementation work in the current direct/n-gram session.

## Current Evidence

- AX manifest validation currently rejects affine quantization bits outside
  `[4, 5, 6, 8]`.
- AX packed-column validation already derives packed columns from the manifest
  bit width, so the validator is the first known blocker for 2-bit and 3-bit
  artifacts.
- MLX's Metal quantized kernels instantiate 2, 3, 4, 5, 6, and 8-bit affine
  groups at group sizes 32, 64, and 128.
- MLX's Python `QuantizedLinear` forwards `bits`, `group_size`, and `mode`
  directly into `mx.quantized_matmul`.
- mlx-lm exposes mixed quantization recipes:
  - `mixed_2_6`: low-bit layers use 2-bit, selected sensitive layers use 6-bit;
  - `mixed_3_4`: low-bit layers use 3-bit, selected sensitive layers and
    `lm_head` use 4-bit;
  - `mixed_3_6`: low-bit layers use 3-bit, selected sensitive layers use 6-bit;
  - `mixed_4_6`: low-bit layers use 4-bit, selected sensitive layers use 6-bit.
- mlx-lm selects higher bits for `v_proj`, `v_a_proj`, `v_b_proj`, `down_proj`,
  and `lm_head` on selected layers. AX should not start with an all-3-bit recipe
  unless upstream evidence shows it is safe for the target model family.
- Recent AX direct-vs-n-gram artifacts show that direct decode improves as bits
  drop from 8 to 6 to 5 to 4, while n-gram uplift is workload and acceptance
  dependent. This supports direct-mode-first validation.
- The 2026-05-27 direct/ngram audit is not complete. Gemma E2B completed rows
  beat `mlx_lm`, but Qwen 3.6 27B 4-bit and 8-bit direct rows still trail
  `mlx_lm`, Qwen 5-bit has a marginal prompt-512 direct gap, and Qwen random
  n-gram rows are no-draft direct fallback rather than accepted-draft
  acceleration.
- The same sweep has missing model-artifact rows for Gemma E2B 8-bit, Gemma E4B
  4-bit, Gemma 26B A4B 4-bit, and Gemma 31B 4-bit. These rows cannot be treated
  as passed until local artifacts exist and the benchmark gate covers them.
- MTPLX uses 3-bit in optional TurboQuant value-cache paths and supports lower
  draft-only LM-head bits, but its default public MTP profiles use 4-bit
  draft-only LM heads. MTPLX should not be treated as proof that 3-bit main
  weights or MTP sidecars are production-safe by default.
- Rapid-MLX uses explicit feature gates for speculative modes. SuffixDecoding is
  default-off, workload-tiered, single-request only, greedy-only, and disabled
  for hybrid linear-attention models in its profile layer. Its DFlash path is
  also alias-gated and rejects MoE, 4-bit, and unvalidated aliases. AX should use
  the same promotion posture for low-bit n-gram and MTP rows: local evidence per
  model, workload, and component before default enablement.
- Rapid-MLX's Qwen3-Next MTP integration keeps MTP weights in a quantized
  sidecar, applies mlx-lm-compatible norm shifts during extraction, and validates
  the injected MTP head at runtime. This supports AX's component labeling rule:
  main-model bit width, MTP sidecar bit width, draft LM-head bit width, and KV
  compression must be reported independently.

## Comparison With MLX And mlx-lm

AX should align with upstream behavior at the boundaries it consumes:

| Surface | Upstream behavior | AX implication |
|---|---|---|
| MLX Metal quantized kernel | 2/3/4/5/6/8-bit affine kernels are instantiated. | AX can rely on MLX for execution and should not add custom kernels. |
| MLX Python quantized layer | `QuantizedLinear.__call__` passes `bits`, `group_size`, and `mode` to `mx.quantized_matmul`. | AX manifests must represent the same quantization fields precisely. |
| mlx-lm plain quantization | Default affine quantization remains 4-bit at group size 64. | AX defaults should remain unchanged. |
| mlx-lm mixed recipes | 2-bit and 3-bit are used selectively with higher-bit sensitive layers. | AX should validate mixed manifests and benchmark mixed recipes first. |
| MTPLX draft paths | 3-bit appears in optional KV/value cache and supported draft-head mechanisms; default draft LM head is 4-bit. | AX must label MTP 3-bit experiments by component, not as a blanket MTP default. |
| Rapid-MLX SuffixDecoding | Default-off, workload-tiered, greedy-only, single-request, and profile-gated away from hybrid linear-attention models. | AX n-gram low-bit promotion needs acceptance telemetry and workload-tier gates, not only faster verifier weights. |
| Rapid-MLX DFlash | Alias-gated; rejects MoE and low-bit main models; reported speedups are workload-sensitive and some aliases are explicitly rejected. | AX should not generalize drafter wins across Gemma E2B, Qwen A3B, n-gram, and MTP without per-mode artifacts. |
| Rapid-MLX MTP | Extracts MTP weights to a sidecar, quantizes to match the target MLX config, and injects runtime support with validation. | AX MTP experiments should preserve sidecar provenance and emit component-specific quantization labels. |

## Direct And N-gram Improvement Focus

Rapid-MLX does not provide a direct-mode silver bullet for AX. Its own
architecture notes point at GPU compute, cache reuse, and prefill behavior as the
important bottlenecks, not scheduler rewrites. For AX, the direct-mode path should
therefore prioritize:

- low-bit direct decode baselines with n-gram disabled;
- prefix/cache snapshot hit-rate and TTFT reporting for repeated prompts;
- prefill-step and chunking comparisons at 128, 512, and 2048 prompt tokens;
- output-token divergence review before claiming any 3-bit direct win.

For n-gram, Rapid-MLX's SuffixDecoding is most useful as a gating model:

- keep acceleration claims workload-tiered instead of global;
- require non-zero accepted draft tokens and a same-policy direct baseline;
- separate no-draft, cooldown, complete-miss, and accepted-draft rows;
- add chat, structured JSON/tool, and code-edit overlap prompts to the benchmark
  suite before changing defaults.

## Best Practices

### Direct Mode

Direct mode is the root performance contract. Every optimization claim must first
show whether the target model itself became faster, independent of speculative
drafting.

- Use direct greedy decode with n-gram disabled as the baseline for every model,
  quantization recipe, prompt length, and generation length.
- Keep weight-bit wins separate from cache wins. Report cold TTFT, warm prefix
  TTFT, restored token count, prefix/state-cache hit type, decode tok/s, and e2e
  tok/s as separate fields.
- Prefer upstream MLX/mlx-lm-compatible quantization layouts over AX-owned kernel
  work. Start with mixed 3-bit recipes and validate packed shape handling before
  benchmarking.
- Treat prefill step size and chunking as workload parameters, not hidden
  defaults. Record the configured prefill step size in artifacts and compare at
  least one smaller and one larger value before recommending a default change.
- Do not promote a direct result when route metadata shows CPU fallback,
  unsupported linear-attention fallback, cache poisoning, or missing token-output
  capture.

### N-gram Mode

N-gram acceleration is a draft-source quality problem first and a verifier-cost
problem second. Lower-bit target weights can reduce the cost of verification and
fallback, but they do not make the n-gram table more predictive.

- Require a matching direct same-policy row before every n-gram row.
- Promote only rows with meaningful accepted draft tokens. No-draft, zero-accept,
  complete-miss, or cooldown-heavy rows must stay labeled as fallback or neutral.
- Keep workload tiers explicit: free-form chat, structured JSON/tool-loop, code
  edit/input-output overlap, and long repeated context.
- Keep greedy/speculative semantics fail-closed. Do not treat stochastic sampling
  rows as token-exact n-gram wins unless the verifier path preserves the requested
  sampling distribution.
- For Qwen linear-attention models, keep conservative gating: repeated evidence
  before drafting, bounded draft length, cooldown after misses, and direct
  fallback until generated output creates a usable draft source.
- Report n-gram speed as effective throughput only when the accepted-draft token
  count explains the win over the direct baseline.

### Completion Gate

The direct/n-gram completion gate is:

```bash
python3 scripts/check_direct_ngram_outperformance.py \
  benchmarks/results/mlx-inference/2026-05-27-ax-direct-ngram-all-models
```

The gate is strict by default:

- every completed direct row must beat the matching `mlx_lm` decode row;
- every n-gram row must beat the matching `mlx_lm` decode row;
- every n-gram row must be `ngram_acceleration_effective_throughput` with
  `ngram_verified_bonus_tokens`;
- `sweep_results.json` must contain no skipped or failed rows.

Diagnostic flags are allowed only for investigation:

- `--allow-ngram-fallback` checks fallback-floor throughput but does not prove
  n-gram acceleration;
- `--allow-sweep-skips` ignores missing local artifacts but does not prove the
  full model matrix.

## Implementation Plan

The detailed checklist lives in
`.internal/planning/LOW-BIT-MLX-QUANTIZATION-IMPLEMENTATION-2026-05-27.md`.

### Phase 1: Manifest Gate

Allow 3-bit affine quantization only under an explicit experimental gate. Keep
2-bit rejected unless a second gate is present.

Required behavior:

- default production validation still rejects 2-bit and 3-bit;
- experimental validation accepts 3-bit affine tensors with supported group sizes;
- tests cover global and per-tensor quantization configs;
- error messages distinguish unsupported production bits from gated experimental
  bits.

### Phase 2: Artifact Intake

Use mlx-lm to create or import mixed 3-bit artifacts for Gemma and Qwen.

Required behavior:

- prefer `mixed_3_4` as the first 3-bit recipe;
- record source repo, mlx-lm command, group size, recipe, and generated manifest;
- do not check model artifacts into git;
- reject manifests whose tensor shapes or packed columns do not match AX
  expectations.

### Phase 3: Direct Decode Benchmark

Run direct same-policy greedy rows before n-gram. MTP remains deferred for this
session.

Required behavior:

- same model family, prompt hash, generation length, seed, host, and git state as
  the 4-bit baseline;
- direct decode only, with n-gram disabled;
- cold TTFT, warm TTFT where applicable, decode tok/s, e2e tok/s, route metadata,
  cache metadata, prefill step size, and fallback counters recorded;
- generated token IDs captured for quality and divergence review.

### Phase 4: N-gram Benchmark

Run n-gram only after direct mode shows a plausible win.

Required behavior:

- direct same-policy baseline present for each row;
- `ax_decode_claim_mode` and `ax_decode_claim_status` present;
- accepted/rejected draft counts recorded;
- rows with no draft or no acceptance are reported as fallback, not acceleration;
- compare random-token contract and at least one input-output-overlap prompt suite;
- add a Rapid-MLX-style workload tier report for chat, structured JSON/tool, and
  code-edit overlap prompts before any default-enable recommendation.

### Phase 5: Deferred MTP Component Benchmarks

Run MTP only after the current direct/n-gram session completes. When it resumes,
keep it as separate component experiments:

- 3-bit main model with existing MTP sidecar;
- 3-bit draft-only LM head if the runtime supports installing it;
- 3-bit or mixed-bit MTP sidecar only if the manifest and loader can represent it;
- TurboQuant KV compression separately, because it is a cache format experiment,
  not a weight-quantization experiment.
- sidecar extraction provenance separately, including norm-shift handling,
  quantization recipe, and runtime MTP validation status when the sidecar is not
  generated by AX.

### Phase 6: 2-bit Risk Gate

Only evaluate 2-bit after 3-bit passes direct decode and n-gram acceptance gates.

Required behavior:

- use `mixed_2_6`, not all-2-bit, as the first experiment;
- require token-output divergence review against direct 4-bit and 3-bit rows;
- keep 2-bit out of public docs and default server behavior unless a follow-up ADR
  explicitly promotes it.

## Acceptance Criteria

- AX can load a gated 3-bit affine MLX artifact that matches MLX/mlx-lm packed
  shape expectations.
- Production validation remains fail-closed for ungated 2-bit and 3-bit artifacts.
- Direct decode artifacts compare 3-bit against the matching 4-bit baseline for
  Gemma E2B and at least one Qwen A3B/linear-attention target.
- Direct artifacts separate cold TTFT, warm cache TTFT, decode tok/s, e2e tok/s,
  prefill step size, route metadata, and token-output divergence.
- N-gram artifacts show workload tier, accepted draft tokens, rejected draft
  tokens, no-draft reasons, cooldowns, and whether the win is true effective
  throughput or direct fallback cost reduction.
- MTP remains deferred; no direct/n-gram acceptance criterion depends on MTP
  sidecar or draft-head work.
- No public README/PERFORMANCE claim is updated from a smoke-only or
  component-mixed artifact.

## Validation

Narrow checks for implementation slices:

```bash
cargo test -p ax-engine-core
cargo test -p ax-engine-mlx
cargo test -p ax-engine-bench
bash scripts/check-bench-doctor.sh
```

Before any public benchmark update:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --quiet --no-fail-fast
```

Model-dependent validation must record model artifact source, mlx-lm conversion
command, manifest path, host, prompt/decode shape, route metadata, git state, and
completed-row status.
