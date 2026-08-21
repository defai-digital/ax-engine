# MLX 0.32.1 admission — 2026-08-21

## Decision

The PyPI MLX 0.32.1 wheel is admitted as AX Engine's pinned runtime. The pin is
exact in `mlx.version` and is enforced by `mlx-sys` at link time. This decision
does not promote the separate MTP multirow experiment described below.

## Runtime and host class

- Apple M5 Max, 128 GB unified memory
- macOS 26.6.1, `applegpu_g17s`
- Python 3.13 validation environment
- `mlx==0.32.1`, `mlx-metal==0.32.1`, PyPI-wheel provenance
- wheel `libmlx.dylib` reports `LC_BUILD_VERSION minos 26.2`
- Qwen serving check: AX Engine 7.1.5 working-tree release-server build

No machine hostname is part of the published evidence.

## M5 qmm gate

The admission probe uses FP16 activations and a four-bit, group-size-64
quantized weight at `M=2048`, `K=4096`, `N=16384`. Throughput is calculated as
`2*M*K*N / elapsed`; the M5 Max gate is approximately 56 TFLOP/s.

| Trial | Median latency | Throughput |
| ---: | ---: | ---: |
| 1 | 4.9112 ms | 55.97 TFLOP/s |
| 2 | 4.8745 ms | 56.39 TFLOP/s |
| 3 | 4.8777 ms | 56.35 TFLOP/s |
| **Median** | **4.8777 ms** | **56.35 TFLOP/s** |

Result: **pass**. `scripts/check-mlx-version.sh` also resolved the exact 0.32.1
wheel and wheel-bundled dylib without an override.

## Compatibility findings

MLX 0.32.1 changed two behaviors that required explicit AX coverage:

- Dequantizing a strided slice of an over-allocated packed KV buffer produced
  incorrect values. AX now materializes packed values, scales, and biases with
  `contiguous` before dequantization. The 16 focused KV-cache tests pass after
  this fix.
- Shapeless compiled linear closures are now shape-polymorphic across the
  decode and prefill shapes covered by the existing guard. The regression test
  now requires equality with the imperative oracle rather than requiring the
  old divergence. MoE graph compilation remains independently gated by its
  stream-registry and real-weight performance criteria.

The repository's final focused MLX library run passed 1,370 tests with three
ignored, including the hybrid KV/recurrent-state writeback and stale
private-feed latch regressions. Formatting, Clippy with warnings denied, the
MLX provenance check, and 30 targeted Python/release-script tests pass on the
final tree.

## Qwen 3.8 serving experiment

Learning from oMLX's multirow behavior, AX can now experimentally suspend
depth-one Qwen linear MTP when at least two compatible greedy rows form, then
transfer each row's complete committed target cache into direct tensor-batched
decode. `AX_MLX_MTP_MULTIROW_BATCH=1` opts in; the default remains off.

On the same M5 class, four 155-token prompts with fixed 256-token outputs gave:

| Mode | Concurrency | Output throughput | Relative to feature-off |
| --- | ---: | ---: | ---: |
| Feature-off MTP | 2 | 24.296 tok/s | 1.00x |
| MTP-to-direct batch, trial 1 | 2 | 42.566 tok/s | 1.752x |
| MTP-to-direct batch, trial 2 | 2 | 44.047 tok/s | 1.813x |
| Feature-off MTP | 4 | 24.314 tok/s | 1.00x |
| MTP-to-direct batch, trial 1 | 4 | 74.422 tok/s | 3.061x |
| MTP-to-direct batch, trial 2 | 4 | 77.405 tok/s | 3.184x |

All requests completed at exactly 256 output tokens and route telemetry proved
the feature reached every reported candidate row. However, only 31 of 32
multirow output streams matched the feature-off greedy token hash. That single
divergence fails AX's numerical promotion bar, so the route remains
diagnostic/default-off and still requires ordinary batched-decode structural
and numerical admission. Raw artifacts are in
[`benchmarks/results/serving/2026-08-21-mlx0321-mtp-multirow/`](../../benchmarks/results/serving/2026-08-21-mlx0321-mtp-multirow/).

The relevant upstream release notes are the
[MLX v0.32.1 release](https://github.com/ml-explore/mlx/releases/tag/v0.32.1).
