# Serving Invariants

AX Engine commits to six engine-layer invariants that govern how the runtime
behaves under serving-shaped load. They are first-class contracts: each one
is enforced by tests, exercised by a stress fixture, and surfaced as
benchmark telemetry. Releases that change runtime behavior in the affected
area must also update the corresponding invariant artifact.

## I-1 — Long prefill must not block decode

A long-prefill request entering the queue must not degrade the inter-token
latency (ITL) or time-to-first-token (TTFT) of in-flight decode requests
beyond a documented bound. The bound is measured in `ax-engine-bench`, not
asserted in prose.

- **Fixture**: `long_prefill_vs_decode` in `crates/ax-engine-bench/src/workloads/`.
- **Telemetry**: `foreground_ttft`, `foreground_itl` latency channels on
  the `ax.serving_workload.report.v1` artifact.
- **Reproducer**:
  ```bash
  cargo run -p ax-engine-bench --bin ax-engine-bench -- serving-stress \
    --workload long_prefill_vs_decode \
    --mlx-model-artifacts-dir "$AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR" \
    --json --output-path /tmp/long_prefill_vs_decode.json
  ```

## I-2 — KV cache restore is fail-closed and versioned

Any persisted or cross-request cache material (prefix cache, KV blocks,
n-gram tables) carries a canonical key that isolates `(model_id,
weight_revision, quantization, block_size, cache_layout, format_version)`.
Restore validates token payload, layer count, shape, and dtype before reuse.
Any mismatch falls back to recomputation; partial trust is forbidden.

- **Reference**: `crates/ax-engine-mlx/src/disk_prefix_cache.rs`. The
  canonical key encoding lives in `canonical_key_bytes` at module scope.
- **Fixture**: `post_restart_cache_safety` — exercises every deviation
  class (model, policy, layout, block size, token payload, token hash,
  payload corruption, truncated file) and asserts each is rejected.
- **Telemetry**: `post_restart_cache` block on the workload report, with
  per-class rejection counters.
- **Note**: this fixture is session-free; it runs without
  `AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR`.

## I-3 — MLX / `mlx-c` stream and thread ownership is an explicit contract

FFI calls into `mlx-c` carry assumptions about which stream is active and
which thread owns the in-flight evaluation. AX Engine documents and audits
those assumptions in `crates/mlx-sys/src/stream.rs` and `docs/MLX-BACKEND.md`,
and adds regression probes for the known cross-thread `eval` /
`clear_cache` / buffer-read failure class.

A type-level migration (for example a `!Send` wrapper or owner-token model)
is **out of scope** for the current invariant set. Such a change requires a
separate ADR justified by AX-specific evidence.

## I-4 — Memory pressure is a scheduler-visible signal

Memory pressure (host RSS, GPU resident-set, KV block pool occupancy) is
observable as a graded `Normal | Soft | Hard` signal on every benchmark
artifact and in route telemetry. This is **observe-mode only**: AX Engine
does not change admission, throttling, or eviction based on the signal.
Any policy that does so requires a follow-up ADR.

- **Module**: `ax_engine_core::mempressure` exposes `PressureLevel`,
  `PressureObservation`, `PressureThresholds`, and the host/device snapshot
  inputs.
- **Default thresholds**: `Soft` at 75% of the configured budget, `Hard`
  at 90%. Deployments with unusually small or large unified memory may
  construct custom thresholds via `PressureThresholds::new`.
- **Bench adapter**: `crates/ax-engine-bench/src/harness/pressure_observer.rs`
  records the observation onto a workload report as
  `ax_mempressure_*` decisions.

## I-5 — Benchmarks cover agent workload shapes

`ax-engine-bench` ships fixtures that mirror coding-agent and long-session
workloads: long-session prefix reuse, partial prefix hit, repeated
tool-output-shaped prompts, cancellation during prefill, post-restart cache
safety, and short-request TTFT under concurrent long requests. Single-request
throughput against `mlx_lm.benchmark` baseline is necessary but no longer
sufficient.

Fixtures live under `crates/ax-engine-bench/src/workloads/` and are driven
end-to-end by `scripts/run-serving-stress.sh`:

```bash
AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR=/path/to/model \
  bash scripts/run-serving-stress.sh \
  --output-dir target/serving-stress/$(date -u +%Y%m%dT%H%M%SZ)
```

Each fixture writes `ax.serving_workload.report.v1` JSON to the output
directory; the script aggregates per-fixture status into `summary.json`.

## I-6 — Speculative and n-gram acceleration claims are correctness-mode gated

AX Engine's current n-gram mode is a deterministic draft path verified by
target-model argmax. It is a valid greedy/exact acceleration contract, but
it is not the same as probability-correct stochastic acceleration. Benchmark
artifacts and release claims must distinguish:

- `direct_greedy_exact_baseline` / `direct_sampling_not_distribution_exact`
- `ngram_greedy_exact_candidate` / `ngram_sampling_not_distribution_exact`
- `ngram_no_observed_draft_path`
- `ngram_no_draft_direct_fallback`
- `ngram_no_accept_fallback`
- `ngram_acceleration_effective_throughput`

The bench script `scripts/bench_mlx_inference_stack.py` emits two separate
fields per row: `ax_decode_claim_status` (fallback / promotion state) and
`ax_decode_claim_mode` (correctness mode). The same-policy greedy gate in
`crates/ax-engine-bench/src/harness/ngram_claim_gate.rs` decides whether
two paired rows can be promoted as a single greedy-exact claim — see
`docs/NGRAM-ACCELERATION.md` for the full taxonomy.

Sampling-mode rows may report throughput and acceptance telemetry, but
**must not be labeled distribution-exact**. The promotion gate in
`scripts/bench_mlx_inference_stack.py::assert_no_distribution_exact_promotion_under_sampling`
fails closed if a row tries to claim otherwise.

## Reproducing the full invariant suite

```bash
# Lightweight invariant probes (run in CI, no model required).
bash scripts/check-mlx-telemetry.sh

# Full agent-workload suite (requires an MLX model artifact directory).
AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR=/path/to/model \
  bash scripts/run-serving-stress.sh
```

Artifacts conform to `ax.serving_workload.report.v1` for per-fixture
reports and `ax.serving_stress.summary.v1` for the aggregated summary.
