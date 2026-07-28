# MLX main (973e27f) admission snapshot — 2026-07-28

Source-built MLX main passes the M5 Max qmm admission bar and shows a
strictly better steady-state submission profile than the pinned 0.32.0
wheel. Recorded here so the runtime pin decision can be made without
re-running the experiments.

## Admission (qmm 4-bit TFLOP/s, M5 Max)

| stack | qmm 4-bit | verdict |
|---|---|---|
| wheel 0.32.0 (pinned) | 57.0 | admitted (baseline) |
| main @ 973e27f (source) | 57.1 | passes the same bar |

## Steady-state profile (gemma-4-26B server TTFT, 12 varied reps)

| stack | mean | band |
|---|---|---|
| wheel 0.32.0 | ~219.9 ms | 203.6-227.1 (+/-12 ms) |
| main @ 973e27f | ~209.0 ms | 205.0-213.0 (+/-4 ms) |

~5% faster with 3x tighter variance — consistent with the upstream
Metal residency work (mlx#3539 area) that the steady-state eval-wall
investigation attributed the one-way degradation to. S1 dual-model
numbers are exactly wheel-parity (qwen 19.43 tok/s / gemma 8215 ms):
the delta lives in the repeated-submission domain, not kernels.

## Build recipe (drop-in, no shim changes)

```sh
git clone https://github.com/ml-explore/mlx.git && cd mlx  # 973e27f
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_METAL=ON \
  -DBUILD_SHARED_LIBS=ON -DMLX_BUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=$PWD/install
cmake --build build -j && cmake --install build
# then: MLX_LIB_DIR=$PWD/install/lib MLX_INCLUDE_DIR=$PWD/install/include
```

## Pin decision (open)

Switching the production runtime from the wheel to a source pin
requires updating the `mlx.version` consumers (mlx-sys/build.rs, the
runtime version check in mlx-sys/src/error.rs, batched-decode
certification metadata, RELEASING.md) and is a deployment-policy call.
Until then, the eval-wall closure lands automatically with the next
published wheel that includes the residency work; this snapshot is the
evidence either way.
