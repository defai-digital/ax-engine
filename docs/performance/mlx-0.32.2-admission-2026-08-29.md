# MLX 0.32.2 admission — 2026-08-29

## Decision

The PyPI MLX 0.32.2 wheel is admitted as AX Engine's pinned runtime. The exact
pin is recorded in `mlx.version` and enforced by `mlx-sys` at link time. This
runtime admission does not by itself issue a Gemma checkpoint or MTP Tier 2
certificate; those claims still require immutable model and clean-build
evidence through the AXQuant certification contract.

## Runtime and host class

- Apple M2 Ultra, 192 GB unified memory
- macOS 26.6.2
- Python 3.12 validation environment
- `mlx==0.32.2`, `mlx-metal==0.32.2`, PyPI-wheel provenance
- `mlx-lm==0.31.3`
- AX Engine 7.2.0, runtime implementation commit `cd21cdb4`
- wheel `libmlx.dylib` reports `LC_BUILD_VERSION minos 26.2`

No machine hostname or machine identifier is part of this record.

## M2 Ultra qmm admission probe

The probe uses FP16 activations and a four-bit, group-size-64 quantized weight
at `M=2048`, `K=4096`, `N=16384`. Throughput is calculated as
`2*M*K*N / elapsed`.

| Trial | Median latency | Throughput |
| ---: | ---: | ---: |
| 1 | 17.1011 ms | 16.07 TFLOP/s |
| 2 | 16.9997 ms | 16.17 TFLOP/s |
| 3 | 17.0556 ms | 16.12 TFLOP/s |
| **Median** | **17.0556 ms** | **16.12 TFLOP/s** |

Result: **pass for the M2 Ultra factory class**. The approximately 56 TFLOP/s
M5 Max threshold is hardware-specific and must not be applied to this host
class. `scripts/check-mlx-version.sh` resolved the exact wheel version and the
wheel-bundled dylib. The release server links `@rpath/libmlx.dylib`, with its
RPATH bound to the admitted MLX 0.32.2 environment.

## Compatibility and Gemma exactness findings

MLX 0.32.2 adds the `force_fused` argument to the C++
`scaled_dot_product_attention` API. AX passes `false` at existing shim call
sites, preserving the prior dispatch policy. See the
[MLX 0.32.2 release](https://github.com/ml-explore/mlx/releases/tag/v0.32.2)
and [upstream API change](https://github.com/ml-explore/mlx/pull/4185).

Gemma 4 assistant MTP required four additional correctness constraints:

- Recurrent assistant draft depths reuse one absolute RoPE position while the
  target KV view is frozen. Advancing the position by draft depth misaligns
  the assistant from its candidate-generation contract.
- Gemma's dense MoE branch keeps singleton arithmetic during exact verify;
  the invariant multi-row projection remains available to the router and
  expert paths.
- The formal direct/MTP profile presents ordered sliding-KV views to both arms.
  A bounded rotating ring may select a different BF16 SDPA reduction when a
  verify chunk straddles the 1024-token sliding window.
- The skinny multi-row qmm verifier is opt-in for the Gemma language head and
  is enabled only by the admitted exact profile.

## Validation

- `cargo test -p ax-engine-mlx`: 1,392 passed, 3 ignored
- MLX 0.32.2 release build and provenance check: passed
- `cargo fmt --all -- --check`: passed
- Clippy with warnings denied: passed after allowing three baseline lints in
  unchanged code (`manual_range_contains`, `too_many_arguments`, `map_entry`)
- Gemma benchmark, MLX version, standalone release, and wheel-minimum script
  tests: 50 passed, 4 skipped

A pre-commit matched-environment diagnostic used the real
`AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP` pack, four long-code prompts,
1,000 output tokens, two warmups, and three measured trials per prompt. All
12 measured direct/MTP token sequences matched; MTP accepted 7,719 of 8,556
assistant drafts (90.22%) and measured 92.45 versus 68.47 token-weighted decode
tok/s (1.350x). Because that build was from a tracked-dirty tree, those numbers
are diagnostic only and are not certification evidence.

## Clean Gemma diagnostic follow-up

A clean-tree follow-up at Engine commit `0d779e3a` used the same 26B 6-bit
pack, four long-code prompts, 1,000 output tokens, two warmups, and five measured
repetitions per prompt. All 20 measured direct/MTP token sequences matched and
every arm emitted 1,000 tokens. The assistant was validated at recurrent depth
2 in all measured trials, proposed 14,260 tokens, accepted 12,865 (90.22%), and
reported no n-gram proposals or accepts. Aggregate decode measured 90.40 versus
66.78 tok/s (1.354x); every paired trial exceeded 1.29x. The build record is
clean and resolves MLX through the pinned 0.32.2 wheel RPATH. See the
[raw diagnostic and limitations](../../benchmarks/results/gemma4-assistant-mtp/2026-08-30-mlx0322-26b6-clean-nonidle-diagnostic/README.md).

This follow-up is still diagnostic, not certification evidence. Periodic host
synchronization and macOS indexing prevented the publication load gates from
passing: the direct measurement window observed one-minute load 25.479 to
31.857, and the MTP window observed 34.191 to 25.310. The raw artifacts retain
those conditions and the run is labeled `clean-nonidle-diagnostic`.

A separate revision-bound AXQuant integration smoke used immutable revision
`a279773d3eecc75d317ec7049bc80bd4a1ec4da2`, chat-template tokenization, the
Gemma exact profile, and a 893-token prompt that reached 1,022 logical KV
positions. Its 128-token direct/MTP outputs matched, composite integrity passed,
MTP accepted 85 of 88 assistant drafts, n-gram counters remained zero, and the
single trial measured 1.402x. That smoke proves the certification integration
path but is too small and was not run on an idle host.

An idle-gated, revision-bound rerun remains required before a Gemma MTP
acceleration certificate. The wider Gemma Tier 2 model and workload matrix also
remains open.
