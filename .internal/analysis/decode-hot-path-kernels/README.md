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

## First Queue

| Order | Candidate | Class | Initial status |
|---|---|---|---|
| 1 | Per-layer decode compile expansion / MLX compile exposure | `graph_compile` | needs fresh profile |
| 2 | `paged_decode_attention` production validation | `phase1_metal` | needs long-context A/B |
| 3 | Quantized projection feasibility for layout-specific weights | `phase1_metal` / `mlx_sidecar` | needs MLX baseline proof |
| 4 | TurboQuant hot-tail merge on GPU | `kv_turboquant` | needs CPU merge/readback attribution |
| 5 | Top-k/top-p sampling GPU path | `sampling` | needs CPU sync/logits telemetry |

## NO-GO Carry-Forward

- Do not reopen standalone residual + RMSNorm sidecar work without a new
  mechanism and same-commit E2E evidence.
- Do not promote generic quantized projection replacements without beating the
  current MLX `quantized_matmul` or `gather_qmm` path in real graph.
- Do not cite microbenchmarks as promotion evidence.
