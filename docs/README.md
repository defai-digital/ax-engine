# AX Engine Public Docs

This directory is the public documentation hub for AX Engine.

AX Engine has three runtime paths, and most docs are organized around keeping
those paths explicit:

- repo-owned MLX runtime for supported Apple Silicon model families
- delegated `mlx_lm.server` compatibility for unsupported MLX text models
- delegated `llama.cpp` compatibility for GGUF and non-MLX local inference

Start here:

- `GETTING-STARTED.md`: installation, first commands, and runtime-path choice
- `SUPPORTED-MODELS.md`: supported LLM model paths: direct support,
  `mlx_lm_delegated`, `llama_cpp`, and unsupported requests
- `FAQ.md`: hardware support, model-stack guidance, runtime paths,
  limitations, and performance-boundary questions
- `performance/README.md`: performance and benchmark docs map, public claim
  boundaries, and promotion rules
- `PERFORMANCE.md`: current performance results, result tables, artifact
  summaries, and interpretation
- `LONG-CONTEXT.md`: long-context evidence, prefix-reuse boundaries, and
  current cold-prefill/concurrency limitations
- `KV-CACHE.md`: logical KV ledger, MLX physical snapshots, opt-in
  disk-durable prefix cache, and memory-pressure invariants
- `API-COMPATIBILITY.md`: current OpenAI-compatible endpoint contract and
  compatibility boundaries
- `SERVER.md`: local HTTP server routes and backend behavior
- `CLI.md`: `ax-engine-bench` and server command surfaces
- `sdk/README.md`: SDK docs hub for Rust, Python, JavaScript/TypeScript, Go,
  Ruby, and Mojo
- `ROADMAP.md`: current serving runtime direction and evidence gates for future
  claims
- `BENCHMARKS.md`: benchmark methodology, test setup, commands, evidence
  contracts, and reproduction details
- `SERVING-BENCHMARKS.md`: online serving benchmark contract for prompt-mix,
  concurrency, request-rate, latency percentile, throughput, and SLO-goodput
  evidence
- `ARCHITECTURE.md`: crate boundaries and dependency rules

Start with `performance/README.md` when you need the performance docs map or
public claim-boundary policy. Use `PERFORMANCE.md` when you want the current
public results in table form. Use `BENCHMARKS.md` when you need to understand
how the tests are set up, which commands produce the artifacts, and how
evidence is classified. Use `LONG-CONTEXT.md` when the claim is about long
prompts, long-running sessions, or prefix reuse. Use `SERVING-BENCHMARKS.md`
when the claim is market-style online serving behavior over a prompt mix.
`BENCHMARKS.md` defines the project split between `ax-engine-bench`
workload-contract artifacts,
`scripts/bench_mlx_inference_stack.py` repo-owned MLX model-inference
comparisons with a required matching `mlx_lm.benchmark` primary baseline,
optional `mlx-swift-lm` secondary baseline adapter rows, delegated
`mlx_lm_delegated` compatibility checks, delegated non-MLX route checks, and
serving latency artifacts.

Experimental MLX KV compression, including `turboquant-shadow` and
`turboquant-fused-experimental`, is off by default and is not a production
support claim. Treat TurboQuant results as evidence artifacts only until a
long-context, model-level quality artifact and decode-throughput promotion gate
both pass.

Public docs should not contain:

- PRDs
- ADRs
- tech specs
- implementation plans
- internal rewrite notes
- engineering best-practice memos

Internal planning records live under `.internal/adr`, `.internal/prd`, and
`.internal/tech-spec`.
