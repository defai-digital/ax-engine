# AX Engine Public Docs

This directory is the public documentation hub for AX Engine.

AX Engine has three runtime paths, and most docs are organized around keeping
those paths explicit:

- repo-owned MLX runtime for supported Apple Silicon model families
- delegated `mlx_lm.server` compatibility for unsupported MLX text models
- delegated `llama.cpp` compatibility for GGUF and non-MLX local inference

Start here:

- `GETTING-STARTED.md`: installation, first commands, and runtime-path choice
- `SUPPORTED-MODELS.md`: model support tiers and what each tier means
- `SERVER.md`: local HTTP server routes and backend behavior
- `CLI.md`: `ax-engine-bench` and server command surfaces
- `BENCHMARKS.md`: how to interpret performance and workload-contract evidence
- `PYTHON.md`: Python binding scope and examples
- `JAVASCRIPT.md`: preview HTTP client usage
- `ARCHITECTURE.md`: crate boundaries and dependency rules

Start with `BENCHMARKS.md` before interpreting any benchmark number. It defines
the project split between `ax-engine-bench` workload-contract artifacts,
`scripts/bench_mlx_inference_stack.py` repo-owned MLX model-inference
comparisons with a required matching `mlx_lm.benchmark` primary baseline, optional
`mlx-swift-lm` secondary baseline adapter rows, delegated `mlx_lm_delegated`
compatibility checks, and delegated non-MLX route checks.

Public docs should not contain:

- PRDs
- ADRs
- implementation plans
- internal rewrite notes
- engineering best-practice memos
