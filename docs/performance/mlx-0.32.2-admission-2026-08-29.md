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
are diagnostic only and are not certification evidence. A clean,
revision-bound rerun is required before a Gemma MTP acceleration certificate.
