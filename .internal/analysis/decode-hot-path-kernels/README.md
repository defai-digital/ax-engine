# Decode Hot-Path Kernel Candidate Registry

This registry tracks decode hot-path kernel and graph-fusion candidates governed
by `PRD-2026-07-03-decode-hot-path-kernel-strategy.md` and ADR-034.

Do not promote a runtime route from prose, intuition, or a standalone
microbenchmark. Start a candidate directory only after a profile identifies the
target stage.

## Candidate Directory Contract

```text
YYYY-MM-DD-<candidate-id>/
  candidate.json
  profile.md
  mechanism.md
  correctness.md
  microbench.json
  e2e-summary.md
  promotion-decision.md
```

Validate candidates with:

```bash
python3 scripts/check_decode_hot_path_kernel_admission.py
```

The checker allows this registry to be empty. Once a `candidate.json` exists,
it fails closed if a promoted/default-on candidate lacks profile evidence,
mechanism, correctness oracle, microbench, real-graph A/B, rollback, route
counters, kill switch, or promotion threshold evidence.

## Post-v6.7.0 Release Boundary

`v6.7.0` is the working release anchor. Custom kernels or custom decode
fastpaths introduced after that tag stay experimental/default-off until they
pass the admission checker as promoted candidates. This registry is the source
of truth for carrying forward those experiments, including no-go decisions.

## First Queue

| Order | Candidate | Class | Initial status |
|---|---|---|---|
| 1 | Per-layer decode compile expansion / MLX compile exposure | `graph_compile` | `no_go` (2026-07-04 E2B profile+A/B; see recorded experiments) |
| 2 | `paged_decode_attention` production validation | `phase1_metal` | needs long-context A/B |
| 3 | Quantized projection feasibility for layout-specific weights | `phase1_metal` / `mlx_sidecar` | needs MLX baseline proof |
| 4 | TurboQuant hot-tail merge on GPU | `kv_turboquant` | needs CPU merge/readback attribution |
| 5 | Top-k/top-p sampling GPU path | `sampling` | needs CPU sync/logits telemetry |

## Recorded Post-v6.7.0 Experiments

| Candidate | Class | Status |
|---|---|---|
| Qwen dense FFN standalone gate/up matvec | `mlx_sidecar` | `no_go` |
| Qwen dense FFN fused gate/up+SwiGLU | `mlx_sidecar` | `not_promoted` |
| Per-layer / sub-layer decode compile on E2B | `graph_compile` | `no_go` |

## NO-GO Carry-Forward

- Do not reopen standalone residual + RMSNorm sidecar work without a new
  mechanism and same-commit E2E evidence.
- Do not promote generic quantized projection replacements without beating the
  current MLX `quantized_matmul` or `gather_qmm` path in real graph.
- Do not reopen the Qwen dense FFN standalone gate/up matvec route unless a new
  graph-level mechanism changes the current cost model.
- Do not cite microbenchmarks as promotion evidence.
- Do not re-enable `AX_MLX_DENSE_FFN_COMPILE` or `AX_MLX_MOE_LAYER_COMPILE` by
  default without a new mechanism: the 2026-07-04 E2B thermal-bracketed A/B shows
  them within the ~2% same-config noise floor (whole-layer compile is already
  default-on and captures the fusable dispatch overhead).
