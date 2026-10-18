# MLX 0.32.4 runtime admission — 2026-10-16

AX Engine pins the published PyPI `mlx==0.32.4` and `mlx-metal==0.32.4`
wheels through `mlx.version`. This admits a runtime dependency. Model
certificates, MTP-S/P/D, pack admission and default-path promotion remain
separate decisions. Historical benchmark rows retain their recorded runtimes.

## Compatibility and scope

The [upstream release](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)
includes Metal synchronization, foreign-buffer ownership, attention dispatch
and quantized arithmetic fixes. AX's existing C++ shims compile unchanged;
the rebuilt Rust workspace passes against the new headers and dylib. Consumers
must rebuild native binaries and Python extensions together with the runtime.
The allocator API changed upstream, so replacing a dylib under an old binary
is not an upgrade procedure.

The optional `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS` remains unset by AX
(default zero upstream). This admission adds no residency, expert-stream,
GDN-prefill or MTP defaults. The separate Flash Next reference harness keeps
its own pinned runtime and evidence contract.

## Numerical checks

`scripts/probe_mlx_runtime_upgrade.py` now records 23 checks, covering the
original affine qmm, sorted/unsorted MXFP4 gather, compiled sigmoid and GQA
checks, plus BF16 MXFP4 at 1/2/4/16/32/128 rows, D256/D512 attention at
1/4/5/32 query tokens, and foreign NumPy-buffer preservation. MXFP4 dot
products and attention use independent NumPy float64 references. The original
seven check tolerances are unchanged.

Both published versions pass all 23 checks on M2 Ultra 192 GB and M5 Max
128 GB, macOS 27.0.1. Same-host input hashes match. The comparison validates
probe identity, hardware, Python/NumPy versions, input hashes, check thresholds and the baseline
output digest before reporting drift. Drift is not used as an accuracy oracle.

On M2 Ultra, the 32-row and 128-row BF16 MXFP4 outputs differ across versions
by at most 0.00390625; both pass the independent reference bound of 0.004.
The original small affine 4/6-bit checks differ by at most 0.000244140625.
Generic affine checks test the runtime API and do not restore retired Qwen
3.8 affine catalog variants.

The upstream [split-K correction](https://github.com/ml-explore/mlx/pull/4641)
keeps partial sums in FP32. For the probe's K=2560, N=1024 BF16 case,
measured peak-device-memory increases above the active baseline were:

| Rows | MLX 0.32.3 | MLX 0.32.4 | Increase |
| --- | ---: | ---: | ---: |
| 32 | 1,114,112 bytes | 2,162,688 bytes | 1,048,576 bytes |
| 128 | 1,310,720 bytes | 2,359,296 bytes | 1,048,576 bytes |

These are operation-level device-allocation measurements, not model RSS or
expert-paging budgets. Real-model memory and near-tie token behavior require
fresh evidence under the new runtime.

### Reference correction found during admission

The first M5 run failed the existing FP32 GQA check on both versions with
identical error 0.000018641352653503418 against the GPU-matmul reference.
Against independent float64 CPU arithmetic, the retained attention outputs
were accurate to approximately 3.5e-9. The reference was affected by GPU
matmul arithmetic, rather than an attention regression. The probe now uses
float64 CPU arithmetic for this reference and retains the original 2e-6
threshold. Both versions then pass at 3.725290298461914e-9 against the stored
float32 reference. Initial failed reports and arrays, plus the exact probe
correction patch, are retained under `initial-gpu-reference/` in the evidence.

## Qmm admission measurement

The M5 Max 128 GB comparison uses FP16 affine four-bit weights, group size 64,
M=2048, K=4096, N=16384, five warmups, then three trials of ten synchronized
calls. The result is the median of trial medians:

| Runtime | Median latency |
| --- | ---: |
| MLX 0.32.3 | 4.912167 ms |
| MLX 0.32.4 | 4.922479 ms |

The candidate is 0.210% slower in this bounded comparison. This provides no
model speedup claim and does not qualify either product SKU. M2 Ultra runs
use `--skip-timing` for local software correctness checks.

## Reproduction and evidence

Install each version into a separate Python environment, using the same
Python and NumPy versions within each host comparison:

```bash
/path/to/old/python scripts/probe_mlx_runtime_upgrade.py \
  --skip-timing --output /tmp/mlx-old.json
/path/to/candidate/python scripts/probe_mlx_runtime_upgrade.py \
  --skip-timing --output /tmp/mlx-candidate.json --compare-to /tmp/mlx-old.json
```

Omit `--skip-timing` for the qmm admission measurement on a designated benchmark
system. Reports bind source, input, runtime-asset and output hashes; adjacent
NPZ files retain output arrays. A failed numerical check produces a report
and a nonzero exit. Comparisons require the same source and host, and reject
changed thresholds, mismatched arrays or altered baseline files.

[Raw reports, arrays and validation](../../benchmarks/results/mlx-runtime-admission/2026-10-16/)
contain all checked outputs and synchronized qmm samples. Both candidate
runtime dylibs report macOS `minos 26.2`; the wheel tag is `macosx_26_0_arm64`.
Runtime staging reports the effective floor and preserves the official assets.

## Validation and boundaries

Validation uses pinned Rust 1.97.1 and a separate Python 3.12 environment:

- Rust formatting and all-target/all-feature Clippy: passed.
- Rust workspace tests: 4,029 passed, 67 ignored, zero failures.
- Script gate, including eight evidence-comparison regressions: passed.
- Exact-version/provenance gate and actual release-runtime staging: passed.
- Both Qwen qualification dry runs and primary-claims checker: passed.
- Python extension rebuilt with `maturin develop` (`release-pyext`): passed.
- Python suite: 264 passed, 54 skipped, 261 subtests passed.
- Metal contract gate: passed its unavailable-toolchain branch; standalone
  Metal asset compilation was skipped because this machine lacks the complete
  Metal toolchain. Existing Rust tests still exercise MLX runtime Metal kernels.

Real-weight 27B/Flash Next qualification, MTP-S/P/D, long-context model
qualification and endurance are not established by these tests. No M4 Pro
64 GB qualification run is claimed. Existing hardware-gated tests retain
their skip conditions; no model weights were added to CI.
