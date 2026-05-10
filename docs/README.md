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
- `FAQ.md`: common runtime-path and performance-boundary questions
- `PERFORMANCE.md`: current performance-table methodology, interpretation, and
  artifact provenance
- `API-COMPATIBILITY.md`: current OpenAI-compatible endpoint contract and
  compatibility boundaries
- `SERVER.md`: local HTTP server routes and backend behavior
- `CLI.md`: `ax-engine-bench` and server command surfaces
- `MANAGER.md`: `ax-engine-manager` web quick start, support bundles, and
  release gate
- `BENCHMARKS.md`: how to interpret performance and workload-contract evidence
- `PYTHON.md`: Python binding scope, examples, and LangChain integration
- `JAVASCRIPT.md`: TypeScript/JS SDK usage and LangChain integration
- `GO.md`: Go SDK usage and examples
- `RUBY.md`: Ruby SDK usage and examples
- `MOJO.md`: Mojo SDK usage and examples
- `ARCHITECTURE.md`: crate boundaries and dependency rules

Start with `PERFORMANCE.md` for the current public result-table context, then
use `BENCHMARKS.md` for the benchmark evidence taxonomy. `BENCHMARKS.md`
defines the project split between `ax-engine-bench` workload-contract artifacts,
`scripts/bench_mlx_inference_stack.py` repo-owned MLX model-inference
comparisons with a required matching `mlx_lm.benchmark` primary baseline, optional
`mlx-swift-lm` secondary baseline adapter rows, delegated `mlx_lm_delegated`
compatibility checks, and delegated non-MLX route checks.

Experimental MLX KV compression, including `turboquant-shadow` and
`turboquant-fused-experimental`, is off by default and is not a production
support claim. Treat TurboQuant results as evidence artifacts only until a
long-context, model-level quality artifact and decode-throughput promotion gate
both pass.

Public docs should not contain:

- PRDs
- ADRs
- implementation plans
- internal rewrite notes
- engineering best-practice memos
