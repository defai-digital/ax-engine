# MLX Prefill PRD Closure

Date: 2026-05-14
PRD: `.internal/planning/MLX-PREFILL-IMPROVEMENT-PRD.md`
Status: W0-W4 closed for the current evidence cycle

## Summary

This report closes the current W0-W4 MLX prefill improvement cycle with the
strongest claim each workstream can support from checked-in artifacts. It does
not invent missing benchmark coverage. Where the artifact says "boundary" or
"diagnostic only", the closure records that boundary as the completed outcome.

## W0: README Weak-Row Refresh

Status: complete.

Evidence:

- Public README source: `benchmarks/results/mlx-inference/2026-05-13-ax-direct-ngram-r2/`
- Classification report: `.internal/reports/W0-gemma-sliding-window-classification-2026-05-13.md`
- Validator: `python3 scripts/check_readme_performance_artifacts.py`

Closure:

- README performance artifact validation passes: 280 metrics validated.
- All 14 supported Gemma direct-prefill rows classify as neutral against the
  prior public baseline; no row regresses beyond the PRD 5% threshold.
- The earlier sliding-window verify weakness is no longer representative of the
  current public table.

## W1: Long-Context Prefill Campaign

Status: closed as bounded evidence, not a broad family-wide win.

Evidence:

- `benchmarks/results/mlx-inference/2026-05-07-real-p1/qwen3-4b-4bit-prefill-scaling/prefill-scaling.md`
- Schema: `ax.mlx_prefill_scaling.v1`
- Model: `mlx-community/Qwen3-4B-4bit`
- Contexts: 1k, 2k, 4k, 8k

Closure:

| Context tok | AX/MLX prefill |
|---:|---:|
| 1,024 | 1.159x |
| 2,048 | 0.982x |
| 4,096 | 0.913x |
| 8,192 | 0.840x |

This closes W1 for the current cycle as a long-context boundary: AX wins at 1k,
is near parity at 2k, and is below `mlx_lm` at 4k/8k on the checked-in Qwen
artifact. The repo must not claim a Gemma/Qwen/GLM-wide long-context win until
matching Gemma and GLM `ax.mlx_prefill_scaling.v1` artifacts are captured.

## W2: Hot-Prefix Physical Reuse

Status: positive physical-reuse claim for Qwen warm-repeat; no long-prompt TTFT
claim.

Evidence:

- `.internal/reports/W2-hot-prefix-qwen-enterprise-2026-05-13.md`
- `benchmarks/results/mlx-inference/2026-05-13-hot-prefix-w2/equivalence-gate/warm_repeat/qwen3-5-9b-2026-05-13.json`
- `benchmarks/results/mlx-inference/2026-05-13-hot-prefix-w2/equivalence-gate/warm_extend/qwen3-5-9b-2026-05-13.json`
- Gate: `bash scripts/check-prefix-reuse-equivalence.sh`

Closure:

- `warm_repeat` verdict: PASS, 5/5 prompts matching exactly.
- Warm physical snapshot hits: 5.
- Warm physical reused tokens: 176.
- Warmup substitution on claimed hit path: 0 tokens.
- Warm physical misses: 0.
- `warm_extend` remains safe diagnostic recompute fallback, not a physical-hit
  claim.

This is a positive product claim for physical prefix snapshot reuse on the
Qwen3.5 warm-repeat path. It is not a published TTFT improvement claim for a
long production prompt workload.

## W3: Concurrent Prefill Boundary

Status: complete boundary, no continuous-batching claim.

Evidence:

- `benchmarks/results/mlx-inference/2026-05-07-real-p2/qwen3-4b-4bit-p2-latency/p2-latency.md`
- `benchmarks/results/mlx-inference/2026-05-07-real-p2/qwen3-4b-4bit-p2-latency/concurrent-prefill.json`
- Schema: `ax.mlx_concurrent_prefill.v1`

Closure:

| Requests | Median request TTFT ms | Median wall ms | Overlap classification |
|---:|---:|---:|---|
| 1 | 2,208.9 | 2,374.5 | serialized |
| 2 | 3,900.0 | 4,202.9 | partial_overlap |
| 4 | 8,318.7 | 8,746.2 | serialized |

The current server path has validated concurrency evidence, but the 4-request
row is serialized. W3 is therefore finished as a negative boundary: no public
continuous-batching claim is allowed from this artifact.

## W4: Packed Projection Path

Status: diagnostic implementation complete; production/public claim blocked on
matched artifact win.

Evidence:

- `.internal/reports/W4-prefill-forward-trigger-2026-05-13.md`
- `scripts/bench_mlx_inference_stack.py --ax-compare-linear-attention-projection-pack`
- `scripts/render_mlx_forward_profile_report.py` `Pack Comparison`
- Loader env gate: `AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS=1`

Closure:

- Row-order oracle and manifest validation are in place.
- Loader-time split-to-packed projection packing is opt-in only.
- The benchmark harness can emit matched split and packed direct AX rows.
- The forward-profile renderer reports packed/split prefill and projection
  deltas with diagnostic verdicts.
- Public promotion remains blocked unless a repeated, cooled artifact shows a
  matched packed/split win.

## Final Claim State

| Workstream | Current claim |
|---|---|
| W0 | Public README weak-row refresh is closed and validator-backed |
| W1 | Long-context Qwen boundary is documented; no broad family-wide win |
| W2 | Positive Qwen warm-repeat physical prefix snapshot reuse claim |
| W3 | Concurrent prefill boundary is serialized at 4 requests; no continuous-batching claim |
| W4 | Packed projection path is diagnostic/opt-in only; no public performance claim |

This is the highest-confidence closure available from the checked-in evidence.
