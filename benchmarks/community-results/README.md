# Community Benchmark Results

This directory documents the submission contract for community benchmark
bundles. Local reproduction runs are written under `local/` by default and are
not required to be committed.

## Reproduce A Run

Use the wrapper script with a local MLX model artifact directory:

```text
scripts/reproduce-mlx-inference-benchmark.sh \
  --model-dir /path/to/mlx-model \
  --run-label qwen3-5-9b-m4-max
```

The wrapper runs:

- `ax-engine-bench doctor` in text and JSON modes.
- `cargo build -p ax-engine-server --release`.
- `scripts/bench_mlx_inference_stack.py` with the required `mlx_lm.benchmark`
  primary baseline.

The output bundle includes:

- `result.json`
- `prompts/`
- `doctor.json`
- `doctor.txt`
- `command.txt`
- `command.log`
- `environment.txt`

## Submission Rules

Submit raw artifacts, not screenshots or hand-copied tables. A useful community
result must include:

- Apple Silicon chip, unified memory, macOS version, and AX Engine commit.
- Model source and revision, or a local manifest path when the artifact is not
  publicly downloadable.
- The unmodified `result.json` and prompt-token artifacts.
- The command log and doctor output.
- Whether the row is direct same-policy AX, AX n-gram effective throughput,
  `mlx_lm`, or admitted `mlx-swift-lm`.

Do not compare rows when the model artifact, prompt-token shape, generation
length, repetition count, reference runtime, or host class is missing. Different
M4/M5 chips, memory sizes, and thermal states can reproduce the procedure while
producing different throughput.

## Review Checklist

Before accepting a community result into public tables:

- Confirm `mlx_lm.benchmark` rows are present.
- Confirm prompt-token SHA-256 values match across engines.
- Confirm AX rows record `selected_backend`, `support_tier`, and decode policy.
- Confirm n-gram acceleration rows are labeled as effective throughput.
- Keep external results separate from repo-owned release claims unless the same
  host and artifact contract were used.
