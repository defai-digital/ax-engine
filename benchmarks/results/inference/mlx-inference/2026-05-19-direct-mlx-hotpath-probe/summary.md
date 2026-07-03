# Direct MLX Hotpath Probe — Gemma 4 E2B p2048 FFN Shapes

Date: 2026-05-19

Artifacts:

- `direct-mlx-hotpath-gemma4-e2b-p2048-geglu.json`
- `direct-mlx-hotpath-gemma4-e2b-p2048-geglu-down.json`
- `direct-mlx-hotpath-gemma4-e2b-p2048-qffn.json`

Command:

```bash
cargo run --release -p mlx-sys --bin direct-mlx-hotpath-probe -- \
  --rows 2048 \
  --cols 6144 \
  --warmup 5 \
  --iterations 30 \
  --json-out benchmarks/results/mlx-inference/2026-05-19-direct-mlx-hotpath-probe/direct-mlx-hotpath-gemma4-e2b-p2048-geglu.json

cargo run --release -p mlx-sys --bin direct-mlx-hotpath-probe -- \
  --candidate gelu_approx_mul_matmul \
  --rows 2048 \
  --cols 6144 \
  --down-cols 1536 \
  --warmup 5 \
  --iterations 30 \
  --json-out benchmarks/results/mlx-inference/2026-05-19-direct-mlx-hotpath-probe/direct-mlx-hotpath-gemma4-e2b-p2048-geglu-down.json

cargo run --release -p mlx-sys --bin direct-mlx-hotpath-probe -- \
  --candidate gelu_approx_quantized_ffn \
  --rows 2048 \
  --cols 6144 \
  --down-cols 1536 \
  --group-size 64 \
  --bits 4 \
  --warmup 5 \
  --iterations 30 \
  --json-out benchmarks/results/mlx-inference/2026-05-19-direct-mlx-hotpath-probe/direct-mlx-hotpath-gemma4-e2b-p2048-qffn.json
```

Validation:

```bash
python3 scripts/check_direct_mlx_hotpath_probe_artifact.py \
  benchmarks/results/mlx-inference/2026-05-19-direct-mlx-hotpath-probe/direct-mlx-hotpath-gemma4-e2b-p2048-geglu.json

python3 scripts/check_direct_mlx_hotpath_probe_artifact.py \
  benchmarks/results/mlx-inference/2026-05-19-direct-mlx-hotpath-probe/direct-mlx-hotpath-gemma4-e2b-p2048-geglu.json \
  --min-speedup 1.05

python3 scripts/check_direct_mlx_hotpath_probe_artifact.py \
  benchmarks/results/mlx-inference/2026-05-19-direct-mlx-hotpath-probe/direct-mlx-hotpath-gemma4-e2b-p2048-geglu-down.json

python3 scripts/check_direct_mlx_hotpath_probe_artifact.py \
  benchmarks/results/mlx-inference/2026-05-19-direct-mlx-hotpath-probe/direct-mlx-hotpath-gemma4-e2b-p2048-geglu-down.json \
  --min-speedup 1.05

python3 scripts/check_direct_mlx_hotpath_probe_artifact.py \
  benchmarks/results/mlx-inference/2026-05-19-direct-mlx-hotpath-probe/direct-mlx-hotpath-gemma4-e2b-p2048-qffn.json

python3 scripts/check_direct_mlx_hotpath_probe_artifact.py \
  benchmarks/results/mlx-inference/2026-05-19-direct-mlx-hotpath-probe/direct-mlx-hotpath-gemma4-e2b-p2048-qffn.json \
  --min-speedup 1.05
```

The schema/correctness gate passes for all artifacts. The promotion speedup
gate fails for all artifacts.

Activation-only results:

| Measurement | Median us | Rust op-count median |
|---|---:|---:|
| Portable `gelu_approx + multiply` | 2256.7085 | 10 |
| Direct C++ `gelu_approx_mul` | 2255.9165 | 1 |
| Direct C++ speedup ratio | 1.00035x | n/a |

Activation + down-projection results:

| Measurement | Median us | Rust op-count median |
|---|---:|---:|
| Portable `matmul(gelu_approx(gate) * up, down)` | 3099.2705 | 11 |
| Direct C++ `gelu_approx_mul_matmul` | 3092.0630 | 1 |
| Direct C++ speedup ratio | 1.00233x | n/a |

Full quantized FFN results:

| Measurement | Median us | Rust op-count median |
|---|---:|---:|
| Portable packed gate/up QMM + GEGLU + down QMM | 4816.8330 | 5 |
| Direct C++ `gelu_approx_quantized_ffn` | 4806.4165 | 1 |
| Direct C++ speedup ratio | 1.00217x | n/a |

Correctness:

- Activation-only: `max_abs_error = 0.0`, `shape = [2048, 6144]`
- Activation + down projection: `max_abs_error = 0.0`, `shape = [2048, 1536]`
- Full quantized FFN: `max_abs_error = 0.0`, `shape = [2048, 1536]`

Decision:

The activation-only, activation+down-projection, and full quantized FFN direct
C++ candidates materially reduce Rust-side dispatch count, but none produces a
meaningful real-shape wall-time speedup. None should be promoted as the Phase 3
production hotpath. The best direct-MLX bypass candidate in this PRD is now a
no-go under the real-shape promotion gate; further performance work should move
to higher-level scheduling, cache, and model-layout changes instead of adding
more one-off direct C++ shims.

Note:

The artifacts record `git.dirty = true` because unrelated script changes were
present in the working tree. Those changes do not affect the `mlx-sys` probe
binary used for these measurements.
