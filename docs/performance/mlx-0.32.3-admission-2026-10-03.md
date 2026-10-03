# MLX 0.32.3 runtime admission — 2026-10-03

AX Engine pins the published PyPI `mlx==0.32.3` and `mlx-metal==0.32.3`
wheels in `mlx.version`, following the existing exact-version and wheel
provenance gates. This is runtime dependency admission. It does not issue a
model certificate, promote MTP defaults, or extend existing throughput claims
to the new runtime. Historical results retain their original runtime identity.

## Compatibility changes

MLX 0.32.3 adds an optional `global_scale` before `sorted_indices` in the C++
`gather_qmm` signature. AX passes `std::nullopt`, preserving existing affine
and MXFP4 pack semantics and the public C ABI. The MXFP4 regression exercises
both sorted and unsorted expert indices against dequantized dense gathering.
See the [upstream change](https://github.com/ml-explore/mlx/pull/4458).

The upstream [precise sigmoid change](https://github.com/ml-explore/mlx/pull/4461)
also changes FP16 intermediate arithmetic. The initial workspace run detected
an exactness failure in AX's fused linear-attention post-input kernel. Its
FP16 specialization now calculates the exponential and reciprocal in FP32,
then stores the FP16 sigmoid before multiplication. The existing FP16 and
BF16 rounding tests cover ordinary decode and short target-verifier paths;
their exactness assertions are unchanged.

## Same-host runtime probes

The reproducible probe is `scripts/probe_mlx_runtime_upgrade.py`. Each host
ran separate Python 3.12 environments for the old and candidate wheels with
identical seeded NumPy inputs. The input hashes match across runs. Seven
synthetic checks pass under both versions, and all retained output arrays are
bit-identical between versions on each host: affine 4/6/8-bit qmm, sorted and
unsorted BF16 MXFP4 gather-qmm, compiled FP32 sigmoid, and GQA SDPA.

The large qmm probe uses FP16, four-bit affine weights, group size 64,
`M=2048`, `K=4096`, `N=16384`, five warmups and three trials of ten iterations.
Each trial records every synchronized latency; the table uses the median of
trial medians. These are dependency-admission probes, not model throughput.

| Host class | Memory | macOS | 0.32.2 median | 0.32.3 median | Latency change |
| --- | --- | --- | ---: | ---: | ---: |
| Mac mini M4 Pro | 64 GiB | 27.0 | 40.4527 ms | 40.4446 ms | -0.020% |
| Mac Studio M2 Ultra | 192 GiB | 27.0.1 | 16.9020 ms | 16.9287 ms | +0.158% |

Both hosts resolve the same candidate `libmlx.dylib` SHA-256:
`ebc8a5f5465afb79d3d9c8f5b0929de1a9b277b7179db8672f92da77ae110a1e`.
The macOS-26 wheel's MLX and JACCL dylibs report `minos 26.2`. That field is
informational; wheel provenance, asset hashes and runtime checks remain the
admission contract. No M5-specific throughput threshold is applied to M4 Pro
or M2 Ultra.

[Raw reports, output arrays and comparison](../../benchmarks/results/mlx-runtime-admission/2026-10-03/)
retain runtime versions, Python versions, hardware, input hashes and all
latencies. The probe source hash, tested runtime source hashes and repository
base revision are recorded beside them. The rebuilt AX test executables also
pass the FP16/BF16 post-input exactness tests and the sorted/unsorted MXFP4
gather test on both target host classes; their hashes and raw logs are retained.

Reproduce independently under each wheel:

```bash
python3.12 -m venv /tmp/ax-mlx-probe
/tmp/ax-mlx-probe/bin/python -m pip install mlx==0.32.3 numpy
/tmp/ax-mlx-probe/bin/python scripts/probe_mlx_runtime_upgrade.py \
  --output /tmp/ax-mlx-0.32.3.json
```

## Validation and boundaries

Final validation used Rust 1.97.1 and Python 3.12 on an Apple M2 Ultra
development workstation with explicit wheel include/library paths:

- `bash scripts/cargo-pinned.sh fmt --check`: passed.
- `bash scripts/cargo-pinned.sh clippy --all-targets --all-features -- -D warnings`: passed without lint exemptions.
- `bash scripts/cargo-pinned.sh test --quiet --no-fail-fast`: 4,044 passed, 61 ignored, zero failures.
- `maturin develop --profile release-pyext`: rebuilt and installed successfully.
- `pytest python/tests -q`: 252 passed, 54 skipped, 164 subtests passed.
- `bash scripts/check-scripts.sh`: passed, including standalone release fixtures.
- `bash scripts/check-mlx-version.sh`: passed for the pinned published wheel.
- Actual wheel runtime staging through `scripts/prepare-mlx-release-runtime.sh`: passed.
- Both Qwen 3.8 qualification dry runs and the primary-claim checker: passed.

The local software results do not qualify model performance on either SKU.
The [structured validation summary](../../benchmarks/results/mlx-runtime-admission/2026-10-03/validation.json)
retains the commands and gate outcomes. The Python warning concerns the
optional FastAPI/Starlette test client, not MLX.

Real-weight 27B and Flash Next generation, MTP-S/P/D campaigns, long-context
qualification and endurance are not certified by these synthetic probes.
Their existing gates remain open; this admission publishes no model speedup
or default-promotion claim. Hardware-gated model tests retain their existing
skip conditions and the always-on qualification checks run in dry-run mode.
