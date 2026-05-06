# TurboQuant Quality Gate Artifact

Status: Internal
Date: 2026-05-06

TurboQuant production promotion requires a long-context, model-level quality
artifact. The artifact is validated by:

```text
python3 scripts/check_turboquant_quality_artifact.py <artifact.json>
```

The validator is intentionally fail-closed. A passing artifact must include:

- schema version `ax.turboquant_quality_gate.v1`
- model id, family, revision, and `head_dim=128`, `head_dim=256`, or
  `head_dim=512`
- workload manifest, prompt hash, at least 8192 context tokens, and at least
  128 generation tokens
- MLX full-precision baseline metadata
- MLX fused compressed decode candidate metadata for `k8v4`
- K8/V4 route metadata showing eligible layers, cold candidate token-layers,
  positive estimated saved KiB, runtime compressed slot writes, and
  `production_ready=0`
- decode quality metrics passing the `reference_k8v4` gate:
  - max absolute difference <= 0.04
  - mean absolute difference <= 0.02
  - minimum cosine similarity >= 0.998
- decode throughput ratio to full-precision baseline >= 0.85
- positive KV saved KiB
- baseline and candidate artifact paths with SHA-256 provenance

This artifact can satisfy only the long-context quality gate. Runtime route
metadata may still report both remaining production blockers because it is
emitted before the external artifact is accepted. The artifact decision must
keep `public_support_docs_approved=false`; public support documentation remains
a separate production blocker.

## Builder

Use the builder after producing:

- a full-precision AX MLX benchmark artifact,
- a TurboQuant fused compressed decode candidate benchmark artifact, and
- a quality metrics JSON file with `max_abs_diff`, `mean_abs_diff`, and
  `min_cosine_similarity`.

Produce the quality metrics JSON from same-shaped decode output vectors:

```text
scripts/run-turboquant-quality-artifact.sh \
  --model-dir .internal/models/gemma-4-e2b-it-4bit \
  --context-tokens 8192 \
  --generation-tokens 256 \
  --repetitions 1 \
  --model-revision phase1-canonical
```

Use `--dry-run` first to verify inferred model metadata, output paths, and the
exact command sequence without loading a model.

The runner performs the full real-model pipeline:

```text
bench_mlx_inference_stack.py --capture-output-token-ids  # baseline
bench_mlx_inference_stack.py --capture-output-token-ids \
  --experimental-mlx-kv-compression turboquant-fused-experimental
build_turboquant_decode_outputs.py  # baseline and candidate token vectors
build_turboquant_quality_metrics.py
build_turboquant_quality_artifact.py
check_turboquant_promotion_readiness.py --artifact quality-gate.json
```

The decomposed commands are still available for debugging:

```text
python3 scripts/build_turboquant_decode_outputs.py \
  --benchmark benchmarks/results/turboquant/baseline.json \
  --context-tokens 8192 \
  --generation-tokens 256 \
  --compression-mode disabled \
  --output benchmarks/results/turboquant/baseline-decode-outputs.json

python3 scripts/build_turboquant_decode_outputs.py \
  --benchmark benchmarks/results/turboquant/candidate.json \
  --context-tokens 8192 \
  --generation-tokens 256 \
  --compression-mode turboquant-fused-experimental \
  --output benchmarks/results/turboquant/candidate-decode-outputs.json

python3 scripts/build_turboquant_quality_metrics.py \
  --baseline-outputs benchmarks/results/turboquant/baseline-decode-outputs.json \
  --candidate-outputs benchmarks/results/turboquant/candidate-decode-outputs.json \
  --output benchmarks/results/turboquant/quality-metrics.json

python3 scripts/build_turboquant_quality_artifact.py \
  --baseline-benchmark benchmarks/results/turboquant/baseline.json \
  --candidate-benchmark benchmarks/results/turboquant/candidate.json \
  --quality-metrics benchmarks/results/turboquant/quality-metrics.json \
  --output benchmarks/results/turboquant/quality-gate.json \
  --model-id gemma-4-e2b-it-4bit \
  --model-family gemma4 \
  --model-revision phase1-canonical \
  --head-dim 256
```

The metrics builder rejects empty vectors, non-finite values, and shape
mismatches.

The builder validates the produced artifact before writing it. Candidate rows
that are only `full_precision_shadow` or `cpu_oracle_compressed_decode`
evidence fail validation and cannot be promoted as fused compressed decode
evidence.

The initial promotion gate is intentionally narrower than the benchmark
telemetry surface: it accepts only K8/V4 `turboquant-fused-experimental`
artifacts with `head_dim=128`, `head_dim=256`, or `head_dim=512`,
`fused_compressed_decode`, positive fused decode successes, and zero fused
decode fallbacks. If the local model set cannot produce a passing long-context
fused-path quality artifact, the correct outcome is a blocked readiness report,
not a public support claim.

Check the current promotion boundary without running a model:

```text
python3 scripts/check_turboquant_promotion_readiness.py \
  --output benchmarks/results/turboquant/<date>/promotion-readiness.json
```

Run the lightweight CLI pipeline smoke without loading a model:

```text
bash scripts/check-turboquant-quality-gate.sh
```

The smoke builds synthetic metrics, compiles and validates a quality artifact,
then verifies that a `full_precision_shadow` candidate fails promotion.
