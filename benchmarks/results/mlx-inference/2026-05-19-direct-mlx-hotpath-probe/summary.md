# Direct MLX Hotpath Probe — Gemma 4 E2B p2048 GEGLU Shape

Date: 2026-05-19

Artifact:

- `direct-mlx-hotpath-gemma4-e2b-p2048-geglu.json`

Command:

```bash
cargo run --release -p mlx-sys --bin direct-mlx-hotpath-probe -- \
  --rows 2048 \
  --cols 6144 \
  --warmup 5 \
  --iterations 30 \
  --json-out benchmarks/results/mlx-inference/2026-05-19-direct-mlx-hotpath-probe/direct-mlx-hotpath-gemma4-e2b-p2048-geglu.json
```

Validation:

```bash
python3 scripts/check_direct_mlx_hotpath_probe_artifact.py \
  benchmarks/results/mlx-inference/2026-05-19-direct-mlx-hotpath-probe/direct-mlx-hotpath-gemma4-e2b-p2048-geglu.json

python3 scripts/check_direct_mlx_hotpath_probe_artifact.py \
  benchmarks/results/mlx-inference/2026-05-19-direct-mlx-hotpath-probe/direct-mlx-hotpath-gemma4-e2b-p2048-geglu.json \
  --min-speedup 1.05
```

The schema/correctness gate passes. The promotion speedup gate fails.

Results:

| Measurement | Median us | Rust op-count median |
|---|---:|---:|
| Portable `gelu_approx + multiply` | 2256.7085 | 10 |
| Direct C++ `gelu_approx_mul` | 2255.9165 | 1 |
| Direct C++ speedup ratio | 1.00035x | n/a |

Correctness:

- `max_abs_error = 0.0`
- `shape = [2048, 6144]`

Decision:

The existing activation-only direct C++ shim materially reduces Rust-side
dispatch count, but it does not produce a meaningful real-shape wall-time
speedup. It should not be promoted as the Phase 3 production hotpath. Any
future Phase 3 work must target a larger FFN subgraph and clear the PRD's
real-shape speedup gate before production wiring.

Note:

The artifact records `git.dirty = true` because unrelated script changes were
present in the working tree. Those changes do not affect the `mlx-sys` probe
binary used for this measurement.
