# AX Engine Docs

Task-first documentation for AX Engine: install, serve, integrate, measure, and
extend the Mac-first Apple Silicon inference runtime.

## How To Read These Docs

| Surface | What it is | Start here if you want… |
| --- | --- | --- |
| Root [README](../README.md) | Short product entry point | Install, first request, headline MTP numbers |
| This hub | Navigation by task and area | The right deep page without guessing layout |
| [Performance Results](PERFORMANCE-RESULTS.md) | Full public tables and charts | Numbers, charts, session-mode evidence |
| [Performance](PERFORMANCE.md) | Interpretation and claim context | What a row does *and does not* prove |
| [Benchmarks](BENCHMARKS.md) | How to run and classify evidence | Reproduction commands and artifact contracts |

AX Engine is **direct-first**. Keep the path explicit when reading or claiming:

- **Repo-owned MLX runtime** — supported Apple Silicon model families AX owns
- **Delegated `mlx_lm.server`** — explicit migration / validation only
- **Delegated `llama.cpp`** — GGUF / non-MLX checks and external reference rows

## Start By Task

| Need | Start here | Then read |
| --- | --- | --- |
| Install and run the first request | [Getting Started](GETTING-STARTED.md) | [CLI](CLI.md), [Server](SERVER.md) |
| Choose, download, or prepare a model | [Supported Models](SUPPORTED-MODELS.md) | [CLI](CLI.md#ax-engine), [FAQ](FAQ.md) |
| Decide whether a family should be supported | [Model Support Policy](MODEL-SUPPORT-POLICY.md) | [Supported Models](SUPPORTED-MODELS.md) |
| Use MTP or compare 4-bit vs 6-bit rows | [MTP Docs](mtp/README.md) | [Performance Results: MTP](PERFORMANCE-RESULTS.md#session-mode-mtp-generation), [Benchmarks: MTP](BENCHMARKS.md#mtp-matrix) |
| Interpret public performance numbers | [Performance Docs Map](performance/README.md) | [Performance Results](PERFORMANCE-RESULTS.md), [Performance](PERFORMANCE.md) |
| Reproduce or review benchmarks | [Benchmarks](BENCHMARKS.md) | [Benchmark Design](BENCH-DESIGN.md), [Serving Benchmarks](SERVING-BENCHMARKS.md) |
| Serve OpenAI / Ollama-shaped APIs | [Server](SERVER.md) | [API Compatibility](API-COMPATIBILITY.md) |
| Integrate from an app or agent | [SDK Docs](sdk/README.md) | [Server](SERVER.md), [Local Engine Clients](LOCAL-ENGINE-CLIENTS.md) |
| Debug long context, prefix reuse, or KV | [Long Context](LONG-CONTEXT.md) | [KV Cache](KV-CACHE.md), [Scheduler](SCHEDULER.md) |
| Change code safely | [Architecture](ARCHITECTURE.md) | [Scheduler](SCHEDULER.md), [KV Cache](KV-CACHE.md) |

## Recommended Paths

### New user

1. [Getting Started](GETTING-STARTED.md)
2. [Supported Models](SUPPORTED-MODELS.md)
3. [Server](SERVER.md)
4. [SDK Docs](sdk/README.md) (if integrating from an application)

### MTP user or reviewer

1. [Supported Models: MTP Downloads](SUPPORTED-MODELS.md#mtp-downloads)
2. [MTP Docs](mtp/README.md)
3. [Performance Results: MTP](PERFORMANCE-RESULTS.md#session-mode-mtp-generation)
4. [Benchmarks: MTP Matrix](BENCHMARKS.md#mtp-matrix)

Prefer the **6-bit** `download-mtp` lane for practical AX usage. Keep **4-bit**
rows as labeled comparison evidence for peer engines that publish 4-bit results.

### Benchmark reviewer

1. [Performance Docs Map](performance/README.md)
2. [Performance Results](PERFORMANCE-RESULTS.md)
3. [Performance](PERFORMANCE.md)
4. [Benchmarks](BENCHMARKS.md)
5. [Benchmark Design](BENCH-DESIGN.md)

Use this path to decide whether a result is current public evidence, diagnostic
history, or out of scope for a claim.

### API or SDK integrator

1. [Server](SERVER.md)
2. [API Compatibility](API-COMPATIBILITY.md)
3. [SDK Docs](sdk/README.md)
4. [Getting Started](GETTING-STARTED.md#installation) for install channels

## Docs By Area

### Setup and models

- [Getting Started](GETTING-STARTED.md) — pip-first install, Homebrew, source builds, first commands
- [Supported Models](SUPPORTED-MODELS.md) — direct support, aliases, `download-mtp`, unsupported requests
- [Model Support Policy](MODEL-SUPPORT-POLICY.md) — promotion, six-month activity rule, EOL
- [FAQ](FAQ.md) — hardware, model stack, runtime paths, limitations
- [CLI](CLI.md) — `ax-engine`, `ax-engine-server`, `ax-engine-bench`

### Serving and SDKs

- [Server](SERVER.md) — HTTP routes, streaming, auth, embeddings, backends
- [API Compatibility](API-COMPATIBILITY.md) — OpenAI-compatible contract and boundaries
- [SDK Docs](sdk/README.md) — Rust, Python, JS/TS, Go, Ruby, Swift, Mojo
- [Local Engine Clients](LOCAL-ENGINE-CLIENTS.md) — in-process vs sidecar HTTP for first-party apps
- [LAN Discovery](LAN-DISCOVERY.md) — local network discovery notes
- [Minisign](MINISIGN.md) — release signature verification

### Performance and benchmarks

- [Performance Docs Map](performance/README.md) — navigation, claim boundaries, promotion rules
- [Performance Results](PERFORMANCE-RESULTS.md) — full tables and charts (MTP, direct, embeddings)
- [Performance](PERFORMANCE.md) — interpretation, long-context notes, MTP mode policy
- [Benchmarks](BENCHMARKS.md) — methodology, commands, evidence classification
- [Benchmark Design](BENCH-DESIGN.md) — workload-contract CLI and artifact design
- [Serving Benchmarks](SERVING-BENCHMARKS.md) — online concurrency, latency, SLO goodput
- [Serving Invariants](SERVING-INVARIANTS.md) — serving correctness invariants
- [Long Context](LONG-CONTEXT.md) — prefix reuse, cold prefill, concurrency limits
- [Embeddings](EMBEDDINGS.md) — embedding API, pooling, fair ingest methodology
- [Embedding Cold-Start](EMBEDDING_COLDSTART.md) — `AX_MMAP_WEIGHTS` measurement guide
- [MTP Docs](mtp/README.md) — MTP navigation, validation, tuning reports
- [Qwen3.6 MTP Peer Benchmark](mtp/qwen36-peer-comparison.md) — AX / MTPLX / lightning-mlx peer table
- [N-gram Acceleration](NGRAM-ACCELERATION.md) — n-gram claim taxonomy
- [DiffusionGemma](DIFFUSIONGEMMA.md) — experimental block-diffusion path

### Runtime architecture

- [Architecture](ARCHITECTURE.md) — crate boundaries and dependency rules
- [Scheduler](SCHEDULER.md) — batching, routing, execution planning
- [KV Cache](KV-CACHE.md) — logical ledger, MLX snapshots, disk prefix cache
- [MLX Backend](MLX-BACKEND.md) — MLX runner and acceleration notes
- [KV weak-surfaces design](designs/kv-weak-surfaces-2026-07-14.md) — MLA / multi-prefill / FA pool plan
- [Roadmap](ROADMAP.md) — serving direction and evidence gates

### Focused reports (diagnostic / historical)

Treat these as context unless a current results page links a fresh artifact.

- [MTP Draft Gate Throughput](mtp/draft-gate-throughput.md)
- [Gemma 4 Assistant MTP Multi-Depth](mtp/gemma4-assistant-multi-depth.md)
- [Tree Draft Phase A](mtp/tree-draft-phase-a.md)
- [AX MTP vs Youssofal MTPLX](mtp/ax-mtp-vs-youssofal.md)
- [Qwen3.6 AX-only multi-suite archive](mtp/qwen36-matrix-refresh.md)
- [Performance Decode Gap](performance/decode-gap.md)
- [Performance MoE Bandwidth Gap](performance/moe-bandwidth-gap.md)
- [Performance MoE Fused Downprojection](performance/moe-fused-downproj.md)

## What Stays Out Of Public Docs

Public docs should not contain PRDs, ADRs, tech specs, implementation plans,
internal rewrite notes, or engineering best-practice memos. Internal planning
lives under `.internal/adr`, `.internal/prd`, and `.internal/tech-spec`.
