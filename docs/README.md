# AX Engine Public Docs

This directory is reserved for public-facing documentation.

It should contain materials such as:

- getting started
- JavaScript usage
- supported models
- CLI usage
- benchmark interpretation and workload-contract guidance
- integration guidance

Current public docs:

- `ARCHITECTURE.md`
- `GETTING-STARTED.md`
- `JAVASCRIPT.md`
- `CLI.md`
- `PYTHON.md`
- `SERVER.md`
- `SUPPORTED-MODELS.md`
- `BENCHMARKS.md`

Start with `BENCHMARKS.md` before interpreting any benchmark number. It defines
the project split between `ax-engine-bench` workload-contract artifacts,
`scripts/bench_mlx_inference_stack.py` MLX model-inference comparisons with a
required matching `mlx_lm.benchmark` primary baseline, optional
`mlx-swift-lm` secondary baseline adapter rows, and delegated non-MLX route
checks.

It should not contain:

- PRDs
- ADRs
- implementation plans
- internal rewrite notes
- engineering best-practice memos
