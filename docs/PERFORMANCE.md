# Performance

This page explains how to interpret the public performance tables in the root
`README.md`. The README intentionally keeps only the current result tables; this
page carries methodology, artifact provenance, and caveats.

## Current Result Set

The current public table was refreshed on 2026-05-06 on:

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
`benchmarks/results/mlx-inference/2026-05-06-ax-rework-r4/`. Reference
`mlx_lm` and `mlx_swift_lm` rows were reused from matching checked-in artifacts,
so the table is an AX-only rework refresh against stable references.

`ax engine` is the direct same-policy comparison against `mlx_lm`.
`ax engine + n-gram accel` reports observed effective throughput from AX's
n-gram acceleration policy. It is not raw model-kernel decode speed.

## Interpretation

The 2026-05-06 r4 AX refresh is not a universal direct-decode win. Direct AX
decode is positive only for Qwen 3.6 UD 4-bit and the 512-token Qwen 3.6 8-bit
shape. GLM 4.7, Qwen Coder Next, and Gemma remain below the matching `mlx_lm`
decode baseline in direct mode.

N-gram acceleration is also not universally effective. It remains strong for
Gemma, Qwen Coder Next, GLM 4.7, and some Qwen 3.6 shapes, but Qwen 3.5 and
Qwen 3.6 5-bit are negative for both prompt shapes.

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
