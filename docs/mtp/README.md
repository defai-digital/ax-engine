# MTP Docs

This is the MTP-specific documentation hub. Use it when the question is about
`ax-engine download-mtp`, MTP benchmark lanes, sidecar or assistant package
validation, or MTP tuning reports.

## Read This First

- Use the 6-bit `download-mtp` packages for practical AX Engine guidance.
- Keep 4-bit rows clearly labeled as comparison evidence for peer MTP engines
  that publish 4-bit results.
- Publish MTP rows in MTP mode only. Do not promote `mtp-ngram` rows in the
  current MTP matrix.
- Current Qwen linear-attention publication rows must explicitly record the
  validated exact-verifier profile (`--ax-qwen-linear-mtp-exact`). The profile
  is an arithmetic/checkpoint contract, not optimistic acceptance.
- Production runners automatically select that contract per loaded Qwen3.5 /
  Qwen3.6 artifact when its MTP depth and dense or affine 4/6/8-bit tensor
  layout are certified. This includes mixed 4/8-bit AXQ packages with group
  sizes 32/64. The selection is runner-scoped, so one resident model cannot
  change another model's arithmetic. Set
  `AX_MLX_QWEN_LINEAR_MTP_EXACT=0` as a kill switch; an ineligible or
  explicitly disabled Qwen linear-MTP package safely uses direct decode
  instead of the slow singleton-replay MTP path.
- Certified depth-one Qwen exact profiles, including the current
  `AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP` package, default the draft confidence gate
  to `0`. (A separate AXQ-4bit sibling for this base is not published: protection
  floors made it redundant with the 6bit pack.) A depth-one sidecar has no
  low-confidence tail to prune, while the generic `0.90` gate still pays for a
  full-vocabulary draft softmax and can discard the only proposal. Explicit
  `AX_MLX_MTP_DRAFT_MIN_CONFIDENCE`, speculation profiles, and the adaptive
  controller retain precedence.
- The policy mechanism covers every model-based drafter, but the numeric
  optimization does not. Qwen multi-depth and GLM heads retain the calibrated
  global/profile/adaptive policy; Gemma 4 assistant MTP retains its independent
  first/deep gates. An ineligible Qwen linear head or an invalid artifact with
  conflicting drafters fails closed to direct decode.
- Direct rows are allowed only as same-package denominators for AX MTP
  acceleration charts, not as cross-model speed evidence.

## Runtime Model Policy

The runner resolves one immutable MTP policy from validated, loaded components.
It does not trust model names or an artifact-provided confidence threshold.
Operator overrides keep the documented precedence, but model-owned defaults can
only come from a runtime-certified policy:

| Route code | Loaded drafter | Confidence policy | Route |
| ---: | --- | --- | --- |
| 0 | None | None | Direct |
| 1 | Qwen dense/recurrent | Calibrated Qwen resolver | MTP |
| 2 | Qwen linear, exact, depth 1 | Model default `0` | MTP |
| 3 | Qwen linear, exact, depth 2-3 | Calibrated Qwen resolver | MTP |
| 4 | Qwen linear without exact capability | None | Direct fallback |
| 5 | GLM | Calibrated GLM resolver; no Qwen override | MTP |
| 6 | Gemma 4 assistant | Independent first/deep calibrated gates | Assistant MTP |
| 7 | Conflicting attached drafters | None | Direct fallback |

Route telemetry exposes this decision through
`ax_mlx_mtp_model_policy`, `ax_mlx_mtp_model_policy_depth`,
`ax_mlx_mtp_model_policy_route_safe`, and
`ax_mlx_mtp_model_policy_active`. A certified model default is separately
reported by `ax_mlx_mtp_model_gate_default_present` and
`ax_mlx_mtp_model_gate_default_x1000`. The existing
`ax_mlx_qwen_linear_mtp_depth_one_gate_zero_model_default` key remains for
backward-compatible Qwen benchmark checks.

## Where To Go

| Need | Read |
| --- | --- |
| Download or prepare an MTP package | [Supported Models: MTP Downloads](../SUPPORTED-MODELS.md#mtp-downloads), [CLI](../CLI.md#ax-engine) |
| Read headline MTP result tables | [Performance Results: MTP](../PERFORMANCE-RESULTS.md#session-mode-mtp-generation), [Qwen3.6 MTP peer benchmark](qwen36-peer-comparison.md), [Performance: MTP Mode](../PERFORMANCE.md#mtp-mode) |
| Reproduce or review MTP benchmarks | [Benchmarks: MTP Matrix](../BENCHMARKS.md#mtp-matrix), [Benchmark Design](../BENCH-DESIGN.md) |
| Tune the MTP draft confidence gate | [MTP draft gate throughput](draft-gate-throughput.md) |
| Review Gemma assistant-MTP depth work | [Gemma 4 assistant MTP multi-depth drafting](gemma4-assistant-multi-depth.md) |
| Review Qwen3.6 peer-engine MTP results | [Qwen3.6 MTP peer benchmark](qwen36-peer-comparison.md) |
| Review archived Qwen3.6 AX-only multi-suite MTP results | [Qwen3.6 AX-only multi-suite MTP results](qwen36-matrix-refresh.md) |
| AX Engine native MTP vs Youssofal MTPLX bundle | [AX MTP vs Youssofal MTPLX-Optimized](ax-mtp-vs-youssofal.md) |
| Review tree-draft investigation history | [Tree draft phase A](tree-draft-phase-a.md) |

## Publication Lanes

### Recommended 6-bit Lane

This is the practical AX Engine lane. Prepare packages with:

```text
ax-engine download-mtp qwen3.6-27b-6bit
ax-engine download-mtp qwen3.6-35b-a3b
ax-engine download-mtp gemma-4-12b
ax-engine download-mtp gemma-4-26b
ax-engine download-mtp gemma-4-31b
```

Artifacts should live under `benchmarks/results/speculative/mtp-6bit/` and record the exact
prepared model path, model snapshot, sidecar or assistant provenance, route
identity, sampler, prompt suite, repetitions, cooldown, prefill, decode, TTFT,
and MTP accept rate. Qwen rows must also record
`ax_qwen_linear_mtp_exact=true` and
`ax_qwen_linear_mtp_exact_explicit_enable=true`; the publication runners reject
ambient or missing profile selection.

### 4-bit Comparison Lane

This lane exists to align with peer MTP-engine benchmark publications that use
4-bit models. It is not the recommended AX Engine deployment setting. Keep
artifacts in clearly labeled comparison directories and keep
[Performance Results](../PERFORMANCE-RESULTS.md) / [Performance](../PERFORMANCE.md)
explicit that 6-bit remains the recommended practical lane.

### Out Of Scope

- `mtp-ngram` rows in current MTP publication
- Qwen3-Coder-Next, 5-bit, 8-bit, FFN-only, or GGUF rows in the recommended
  6-bit matrix
- direct rows used as a cross-model speed leaderboard
