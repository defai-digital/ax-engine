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
- Direct rows are allowed only as same-package denominators for AX MTP
  acceleration charts, not as cross-model speed evidence.

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
and MTP accept rate.

### 4-bit Comparison Lane

This lane exists to align with peer MTP-engine benchmark publications that use
4-bit models. It is not the recommended AX Engine deployment setting. Keep
artifacts in clearly labeled comparison directories and keep the README or
PERFORMANCE text explicit that 6-bit remains the recommended practical lane.

### Out Of Scope

- `mtp-ngram` rows in current MTP publication
- Qwen3-Coder-Next, 5-bit, 8-bit, FFN-only, or GGUF rows in the recommended
  6-bit matrix
- direct rows used as a cross-model speed leaderboard
