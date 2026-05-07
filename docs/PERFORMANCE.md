# Performance

This page explains how to interpret the public performance tables in the root
`README.md`. The README intentionally keeps only the current result tables; this
page carries methodology, artifact provenance, and caveats.

## Current Result Set

The current public table was refreshed on 2026-05-07 on:

- Apple M5 Max
- 128 GB memory
- macOS 26.4.1

Benchmark shape:

- random-token prompts from the `mlx_lm.benchmark` seed-0 contract
- batch size 1
- `prefill_step_size=2048`
- 128 generated tokens
- temperature 0
- 3 timed trials plus 1 AX warmup

AX rows in the README were refreshed from
`benchmarks/results/mlx-inference/2026-05-07-v4.4.1-readme-refresh/`.
Most reference `mlx_lm` and `mlx_swift_lm` rows were reused from matching
checked-in artifacts, so the table is primarily an AX-only refresh against
stable references. Qwen 3.5 `mlx_lm` was rerun fresh after investigation showed
the older reused 512-token row was an outlier for current comparisons. Qwen 3.5
AX rows were then rerun after rebuilding the release server binary.

`ax direct baseline` is the direct same-policy comparison against `mlx_lm`; the
benchmark starts the AX server with n-gram acceleration disabled for this row.
`ax default n-gram` reports observed effective throughput from AX's default
n-gram acceleration policy. It is not raw model-kernel decode speed.

## Interpretation

The 2026-05-07 v4.4.1 AX refresh is not a universal direct-decode win. Direct
AX is intentionally measured with n-gram acceleration disabled, and it remains
behind `mlx_lm` on the Gemma 4 E2B direct rows. The Qwen 3.5 direct rows are
around parity in repeated runs and moved across the baseline after the release
server was rebuilt. The artifact telemetry shows direct rows are pure
direct-pipeline runs (`ax_mlx_direct_pipeline_steps=381`,
`ax_mlx_ngram_decode_steps=0` across three trials), so any direct-row gap is the
same-policy baseline rather than a failed n-gram path.

N-gram acceleration is the default AX user path and is the better headline row
for throughput expectations. In the same Gemma 4 E2B rows it accepts every
draft token in the random-token contract (`ax_ngram_accept_rate_micros=1000000`)
and reports 421-572 tok/s decode throughput, well ahead of both reference
runtimes. Qwen 3.5 at 512 prompt tokens is the row where the default n-gram
path remains slightly below the fresh `mlx_lm` reference: it records zero draft
attempts before falling back to the direct pipeline after the no-draft probe
window, while the direct row is +0.9% in the current artifact. The benchmark
harness now labels this condition as
`ngram_no_draft_direct_fallback` so artifacts do not describe a no-draft
fallback as n-gram effective throughput.

The earlier larger Qwen 3.5 512-token gap was a benchmark-comparison artifact:
the AX-only README refresh reused an older 2026-05-05 `mlx_lm` row at 101.3
tok/s, while a fresh same-turn `mlx_lm` rerun on 2026-05-07 reports 93.4 tok/s
for the same prompt/decode contract. The current README row uses the fresh
reference for Qwen 3.5, so the direct AX row is now reported as +0.9% rather
than a stale-reference -8% to -12% regression.

Qwen-family linear-attention n-gram rows use a rollback-safe branch/recompute
path for SSM state. Acceleration is prompt/output-pattern dependent. Benchmark
JSON artifacts include fixed-schema n-gram telemetry fields, and the README
table uses median AX runner timing plus output-token count.

## Gemma Direct-Decode Follow-Ups

A follow-up Gemma 4 E2B 4-bit direct-decode refresh after the sliding-window
decode KV view change is recorded in:

```text
benchmarks/results/mlx-inference/2026-05-06/gemma-4-e2b-it-4bit-ax-engine-sliding-window-direct.json
```

It improves AX direct decode versus r4, but AX direct remains below `mlx_lm`.

An opt-in true rotating-backing-store experiment is recorded in:

```text
benchmarks/results/mlx-inference/2026-05-06/gemma-4-e2b-it-4bit-ax-engine-rotating-direct.json
```

It is not the default because it did not produce a consistent direct-decode win.

An opt-in `mlx_lm`-style direct decode cache-clear cadence experiment is
available with:

```text
AX_MLX_DIRECT_CLEAR_CACHE_CADENCE=256
```

The artifact is:

```text
benchmarks/results/mlx-inference/2026-05-06/gemma-4-e2b-it-4bit-ax-engine-clear-cache-direct.json
```

It is also not the default because it did not improve the retained-window
direct-decode result.

## Model Notes

Qwen Coder Next uses MLX affine 4-bit globally, with 8-bit overrides for router
and shared-expert gate tensors.

Qwen 3.6 35B A3B includes the Unsloth UD-MLX 4-bit checkpoint plus
MLX-community 5/6/8-bit checkpoints. The 5/6-bit checkpoints are affine 5/6-bit
globally with 8-bit router and shared-expert gate overrides; the 8-bit
checkpoint is affine 8-bit throughout.

Gemma 4 26B A4B is the public Gemma 4 MoE MLX model. Its checkpoint uses affine
4-bit globally, with 8-bit overrides for dense MLP and router projections. The
`mlx_swift_lm` reference row loads this model via the `MLXVLM` factory, which is
required for MoE architectures, rather than the standard Swift path. Prompt
tokenization is otherwise identical to all other Swift rows.

For the remaining Gemma 4 26B A4B direct-decode gap, rerun AX direct rows with
`--ax-gemma4-moe-profile`. Those diagnostic rows record per-section MoE decode
counters under `ax_mlx_gemma4_moe_profile` and intentionally insert eval
barriers, so compare their bottleneck split rather than their headline tok/s.

Gemma 4 E2B 5/6/8-bit checkpoints use affine quantization at their respective
bit depths globally. These rows verify AX Engine's higher-bit quantization
support; the 4-bit row is the primary decode performance reference.

Gemma 4 E4B benchmark rows are pending. The model manifest and scenario
manifest are present. See
`benchmarks/results/mlx-inference/2026-05-04/README.md` for the run command.

## Reproduction

To reproduce the benchmark procedure on an Apple Silicon host, use:

```text
scripts/reproduce-mlx-inference-benchmark.sh \
  --model-dir /path/to/mlx-model \
  --run-label local-refresh
```

The script records doctor output, command logs, prompt artifacts, and raw JSON
results under `benchmarks/community-results/local/` by default.

Before submitting or comparing external rows, read:

- `docs/BENCHMARKS.md`
- `benchmarks/community-results/README.md`

Community runs reproduce the procedure, not identical numbers. Compare rows
only after checking model artifact identity, prompt-token hashes, generated
token count, repetitions, reference runtime, AX decode policy, host class, and
thermal context.

## Workload Contracts

The README throughput tables are MLX model-inference comparisons. Workload
contracts are a separate `ax-engine-bench` surface for checked-in scenario,
replay, matrix, route, correctness, determinism, and regression gates.

Canonical manifests live under:

- `benchmarks/manifests/scenario/`
- `benchmarks/manifests/replay/`
- `benchmarks/manifests/matrix/`

See `docs/BENCHMARKS.md` for the evidence split, methodology, and
prompt-provenance requirements.
