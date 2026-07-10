# AX Engine Docs

This is the public documentation hub for AX Engine. The docs are organized by
task first, then by runtime area, so readers do not have to know the repository
layout before choosing the right page.

AX Engine is direct-first. Keep the path explicit when reading, benchmarking,
or writing claims:

- repo-owned MLX runtime for supported Apple Silicon model families
- explicit delegated `mlx_lm.server` compatibility for migration and validation
- explicit delegated `llama.cpp` compatibility for GGUF/non-MLX checks and
  external reference rows

## Start By Task

| Need | Start here | Then read |
| --- | --- | --- |
| Install AX Engine with pip and run the first request | [Getting Started](GETTING-STARTED.md) | [CLI](CLI.md), [Server](SERVER.md) |
| Choose, download, or prepare a model | [Supported Models](SUPPORTED-MODELS.md) | [CLI](CLI.md#ax-engine) |
| Decide whether a model family should be supported | [Model Support Policy](MODEL-SUPPORT-POLICY.md) | [Supported Models](SUPPORTED-MODELS.md) |
| Use MTP or understand 4-bit vs 6-bit MTP results | [MTP Docs](mtp/README.md) | [Qwen3.6 MTP Peer Benchmark](mtp/qwen36-peer-comparison.md), [Performance](PERFORMANCE.md#mtp-mode), [Benchmarks](BENCHMARKS.md#mtp-matrix) |
| Interpret public performance numbers | [Performance Docs Map](performance/README.md) | [Performance](PERFORMANCE.md), [Benchmarks](BENCHMARKS.md) |
| Run or review benchmarks | [Benchmarks](BENCHMARKS.md) | [Benchmark Design](BENCH-DESIGN.md), [Serving Benchmarks](SERVING-BENCHMARKS.md) |
| Serve OpenAI-compatible or Ollama-shaped APIs | [Server](SERVER.md) | [API Compatibility](API-COMPATIBILITY.md) |
| Use a language SDK | [SDK Docs](sdk/README.md) | The SDK page for your language |
| Debug long context, prefix reuse, or KV behavior | [Long Context](LONG-CONTEXT.md) | [KV Cache](KV-CACHE.md), [Scheduler](SCHEDULER.md) |
| Understand crate boundaries before changing code | [Architecture](ARCHITECTURE.md) | [Scheduler](SCHEDULER.md), [KV Cache](KV-CACHE.md) |

## Recommended Paths

### New User

1. [Getting Started](GETTING-STARTED.md)
2. [Supported Models](SUPPORTED-MODELS.md)
3. [Server](SERVER.md)
4. [SDK Docs](sdk/README.md), if you are integrating from an application

### MTP User Or Reviewer

1. [Supported Models: MTP Downloads](SUPPORTED-MODELS.md#mtp-downloads)
2. [MTP Docs](mtp/README.md)
3. [Performance: MTP Mode](PERFORMANCE.md#mtp-mode)
4. [Benchmarks: MTP Matrix](BENCHMARKS.md#mtp-matrix)

The practical AX Engine recommendation is the 6-bit `download-mtp` lane. The
4-bit lane is kept as clearly labeled comparison evidence because peer MTP
engines often publish 4-bit benchmark rows.

### Benchmark Reviewer

1. [Performance Docs Map](performance/README.md)
2. [Performance](PERFORMANCE.md)
3. [Benchmarks](BENCHMARKS.md)
4. [Benchmark Design](BENCH-DESIGN.md)

Use this path when deciding whether a result is current public evidence,
diagnostic history, or out of scope for a claim.

### API Or SDK Integrator

1. [Server](SERVER.md)
2. [API Compatibility](API-COMPATIBILITY.md)
3. [SDK Docs](sdk/README.md)
4. [Getting Started](GETTING-STARTED.md#installation), for install and release
   channel details

## Docs By Area

### Setup And Models

- [Getting Started](GETTING-STARTED.md): pip-first installation, optional
  Homebrew installs, source builds, first commands, and runtime-path choice
- [Supported Models](SUPPORTED-MODELS.md): direct support, delegated paths,
  aliases, `download-mtp`, and unsupported requests
- [Model Support Policy](MODEL-SUPPORT-POLICY.md): direct-support promotion,
  six-month activity rule, compatibility-adapter boundary, and EOL handling
- [FAQ](FAQ.md): hardware support, model-stack guidance, runtime paths,
  limitations, and performance-boundary questions
- [CLI](CLI.md): `ax-engine`, `ax-engine-server`, and `ax-engine-bench`
  command surfaces

### Serving And SDKs

- [Server](SERVER.md): local HTTP server routes, streaming, auth, embeddings,
  and backend behavior
- [API Compatibility](API-COMPATIBILITY.md): OpenAI-compatible endpoint
  contract and compatibility boundaries
- [SDK Docs](sdk/README.md): Rust, Python, JavaScript/TypeScript, Go, Ruby, and
  Mojo SDKs

### Performance And Benchmarks

- [Performance Docs Map](performance/README.md): performance navigation,
  public claim boundaries, and promotion rules
- [Performance](PERFORMANCE.md): current public result tables, artifact
  summaries, MTP mode, and interpretation
- [Benchmarks](BENCHMARKS.md): benchmark methodology, commands, evidence
  contracts, and reproduction details
- [DiffusionGemma](DIFFUSIONGEMMA.md): experimental block-diffusion support,
  first-block telemetry, and non-autoregressive benchmark boundaries
- [Benchmark Design](BENCH-DESIGN.md): workload-contract CLI and artifact
  design
- [Serving Benchmarks](SERVING-BENCHMARKS.md): online serving prompt-mix,
  concurrency, latency, throughput, and SLO-goodput evidence
- [Long Context](LONG-CONTEXT.md): long-context evidence, prefix-reuse
  boundaries, and current cold-prefill/concurrency limits
- [Embedding Cold-Start](EMBEDDING_COLDSTART.md): `AX_MMAP_WEIGHTS`
  measurement guide and default-on criteria for the mmap weight loader
- [MTP Docs](mtp/README.md): MTP-specific navigation, validation notes, and
  tuning reports
- [Qwen3.6 MTP Peer Benchmark](mtp/qwen36-peer-comparison.md): full AX Engine,
  MTPLX, and lightning-mlx peer result table with fairness limitations
- [N-gram Acceleration](NGRAM-ACCELERATION.md): n-gram acceleration claim
  taxonomy and evidence rules

### Runtime Architecture

- [Architecture](ARCHITECTURE.md): crate boundaries and dependency rules
- [Scheduler](SCHEDULER.md): batching, routing, and execution planning
- [KV Cache](KV-CACHE.md): logical KV ledger, MLX snapshots, disk-durable
  prefix cache, and memory-pressure invariants
- [Roadmap](ROADMAP.md): serving runtime direction and evidence gates for
  future claims

### Focused Reports

Use these as historical or diagnostic context unless a current result page links
to a fresh artifact.

- [MTP Draft Gate Throughput](mtp/draft-gate-throughput.md)
- [Gemma 4 Assistant MTP Multi-Depth](mtp/gemma4-assistant-multi-depth.md)
- [Tree Draft Phase A](mtp/tree-draft-phase-a.md)
- [Performance Decode Gap](performance/decode-gap.md)
- [Performance MoE Bandwidth Gap](performance/moe-bandwidth-gap.md)
- [Performance MoE Fused Downprojection](performance/moe-fused-downproj.md)

## What Stays Out Of Public Docs

Public docs should not contain PRDs, ADRs, tech specs, implementation plans,
internal rewrite notes, or engineering best-practice memos. Internal planning
records live under `.internal/adr`, `.internal/prd`, and `.internal/tech-spec`.
