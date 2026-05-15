# F3 M5 — Disk Prefix Cache Runtime/Docs Promotion Review (2026-05-14)

PRD: `MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`
Prior findings:
- `MLX-F3-DISK-PREFIX-CACHE-M4-FINDINGS-2026-05-14.md`
- `MLX-F3-DISK-PREFIX-CACHE-M3C-STRESS-FINDINGS-2026-05-14.md`

## 1. Status: **M5 DOCS LANDED; PROMOTION IS OPT-IN, NOT DEFAULT.**

The disk-durable prefix cache now has public runtime documentation and
performance-boundary documentation. The correct promotion level is:

- **Enabled by operator opt-in** via `AX_MLX_PREFIX_CACHE_DIR`.
- **Disabled by default**.
- **Validated for cross-restart correctness** on Gemma 4 E2B,
  Qwen3.5-9B, and GLM-4.7-Flash.
- **Validated for short cache-primitive stress** with four worker
  processes and tight entry-budget eviction.
- **Not yet promoted as a production-serving latency / availability
  claim** without a full AX-serving soak artifact.

## 2. Public docs updated

| Path | Promotion content |
|---|---|
| `docs/KV-CACHE.md` | Added the opt-in disk-durable prefix-cache contract, env flags, file integrity contract, telemetry keys, checked-in evidence, rerun commands, and serving-soak boundary. |
| `docs/PERFORMANCE.md` | Added disk-prefix-cache evidence table and clarified that the artifacts support cross-restart correctness / primitive safety, not broad serving throughput. |
| `docs/README.md` | Updated the docs index so `KV-CACHE.md` is the entry point for disk-durable prefix-cache behavior. |

## 3. Remaining blocker

Only one blocker remains before closing F3 completely:

- full AX-serving soak / promotion decision.

The serving benchmark harness now preserves
`response.route.crossover_decisions` into each observation, aggregates numeric
route counters in `summary.route_decisions`, and the artifact checker can fail
closed with `--require-route-decision-min KEY=MIN`. The final soak should use
that gate, for example
`--require-route-decision-min ax_mlx_prefix_cache_disk_hits=1`, so the artifact
proves it exercised the disk-cache route and not only a generic long-context
serving path.

The repo also has a deterministic shared-prefix token-corpus builder,
`scripts/build_serving_shared_prefix_corpus.py`, for this final soak. Use it to
generate an 8K+ token shared-prefix corpus, run at least one full warmup corpus
pass with `--input-kind tokens`, and store the corpus, serving JSON artifact,
and rendered report under the same `benchmarks/results/serving/<run-id>/`
directory.

This should be a serving artifact, not another cache-primitive unit stress:
run the AX server path with the disk cache enabled, multiple processes or
workers where applicable, shared prompts, latency/queueing telemetry, memory
pressure, disk hit rate, and zero-failure evidence. If the product decision is
to keep the disk cache as opt-in experimental until real deployment demand,
record that as the final F3 closure decision instead of inventing a broad
production-serving claim.
