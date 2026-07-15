# AX Engine

AX Engine is a Mac-first LLM inference runtime for Apple Silicon. For
direct-support Gemma, Qwen, and GLM families it owns the MLX graph path,
KV/runtime behavior, local server, model packaging, and benchmark contract —
not just a thin wrapper around `mlx_lm`.

**Requires macOS 26 (Tahoe)+ on Apple Silicon (M2 Max or newer).**

## Why AX Engine

- **First-class MTP** — prepare packages with one command (`ax-engine download-mtp`)
- **Simple local serving** — install the wheel, download a model, run the printed
  `ax-engine serve` command for OpenAI-compatible endpoints
- **Repo-owned runtime** — direct Gemma / Qwen / GLM paths run in AX's MLX runtime
  by default; `mlx-lm` and `llama.cpp` are explicit compatibility adapters
- **Evidence-backed peer benchmarks** — MTP vs MTPLX / lightning-mlx; direct vs
  mlx-lm / llama.cpp Metal, with checked-in artifacts

## Quick Start

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -U "ax-engine[download]>=6.9.0,<7"
ax-engine doctor
```

Download a Gemma 4 12B MTP package and serve it:

```bash
ax-engine download-mtp gemma-4-12b-4bit
# then run the `ax-engine serve ...` command printed by the downloader
```

Send a request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"gemma-4-12b-it","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

Interactive UI (models, downloads, chat):

```bash
ax-engine tui
```

Serve a direct model in one command:

```bash
ax-engine serve qwen36-35b --download --port 8080
```

Full install options (Homebrew, source builds, troubleshooting):
[Getting Started](docs/GETTING-STARTED.md).

## Models

Primary families run on AX-owned MLX graphs. Secondary families ship as
preview direct support. Use `ax-engine download` / `download-mtp` for
pre-sanitized packages.

| Family | Role | Notes |
| --- | --- | --- |
| Gemma 4 | Primary chat | Affine 4/5/6-bit; assistant-MTP |
| Qwen 3.6 | Primary agent | Fused sidecar MTP |
| Qwen3-Coder-Next | Primary coding agent | Coding-first architecture |
| Qwen 3 / 3.5, GLM 4.7 Flash | Supported direct | Dense / linear-attn / Flash MLA |
| Llama, Mistral, GPT-OSS | Secondary preview | One-command download/serve |

Aliases, hardware sizing, and MTP targets:
[Supported Models](docs/SUPPORTED-MODELS.md) ·
[Hardware FAQ](docs/FAQ.md#what-hardware-does-ax-engine-support) ·
[CLI](docs/CLI.md).

## Performance

Why people try AX Engine on Apple Silicon: **faster local decode** against the
engines they already know. Results are **session-separated** — do not mix MTP
rows with direct rows, or either with embeddings.

| Session | Peers | Headline metric |
| --- | --- | --- |
| **MTP generation** | AX Engine · [MTPLX](https://github.com/youssofal/MTPLX) · [lightning-mlx](https://github.com/samuelfaj/lightning-mlx) | MTP decode tok/s |
| **Direct generation** | AX Engine · [mlx-lm](https://github.com/ml-explore/mlx-lm) · [llama.cpp](https://github.com/ggml-org/llama.cpp) Metal | Decode / prefill / TTFT |
| Embeddings | AX · mlx-lm / mlx-embeddings | Ingest tok/s (see full results) |

Host baseline for the published local rows: **Apple M5 Max · 128 GB · macOS 26.x**.
AX Engine **v6.9.0**; `mlx-lm` **0.31.3**; `llama.cpp` **b9910** / ggml **0.15.3**;
MTPLX **2.0.1**. Full methodology, accept rates, and artifacts:
[Performance Results](docs/PERFORMANCE-RESULTS.md) ·
[Benchmarks](docs/BENCHMARKS.md) ·
[Claim boundaries](docs/performance/README.md).

> [!IMPORTANT]
> Prefill/TTFT peer rows require the **same resolved `libmlx`** on both sides.
> Some Homebrew or low-deployment-target MLX builds omit M5 GEMM paths and look
> ~3–4× slower. See the admission note in
> [Performance Results](docs/PERFORMANCE-RESULTS.md).

### MTP: AX Engine vs MTPLX vs lightning-mlx

Qwen3.6 MTP peer comparison (decode is the closest fair metric across engines).
27B 4-bit uses the **same** AX sidecar on all three; 35B-A3B rows are
production-configuration packages. Stitched peer session (not one interleaved
physical run). Full fairness notes:
[Qwen3.6 MTP peer comparison](docs/mtp/qwen36-peer-comparison.md).

<img src="docs/assets/perf-mtp-peer-comparison-apples-to-apples.svg" alt="Qwen3.6 MTP peer comparison: AX Engine, MTPLX, and lightning-mlx decode throughput">

| Target | AX Engine | MTPLX | lightning-mlx | Readout |
| --- | ---: | ---: | ---: | --- |
| Qwen3.6 27B 4-bit | **63.0** tok/s | 58.5 tok/s | 55.7 tok/s | Same sidecar; AX leads |
| Qwen3.6 27B 6-bit | 41.8 tok/s | — | — | No official peer 27B 6-bit MTP artifact |
| Qwen3.6 35B-A3B 4-bit | **172.4** tok/s | 137.9 tok/s | 116.2 tok/s | AX leads production-config row |
| Qwen3.6 35B-A3B 6-bit | **141.2** tok/s | 119.0 tok/s | 96.3 tok/s | AX leads production-config row |

**Same-package AX MTP acceleration** (exact sampled MTP, not a peer
leaderboard): all 15 target/suite rows speed up decode by **1.41×–2.66×** over
AX direct on the same 6-bit package, with 100% MTP step coverage.

<img src="docs/assets/perf-mtp-6bit-ax-acceleration.svg" alt="AX Engine 6-bit exact sampled-MTP acceleration vs same-package direct">

### Direct: AX Engine vs mlx-lm vs llama.cpp

Non-speculative autoregressive generation. Charts are a **v6.9.0 AX-only
snapshot** overlaid with **retained** historical `mlx-lm` and `llama.cpp` Metal
rows (cross-run distribution view, not a same-session peer matrix). Exact AX
per-model numbers and caveats:
[Performance Results: Direct](docs/PERFORMANCE-RESULTS.md#session-mode-direct-generation).

**Gemma 4** (decode / prefill / TTFT across model+quant rows × prompt depths):

<img width="100%" src="docs/assets/perf-gemma4-decode-box-whisker.svg" alt="Gemma 4 direct decode: AX Engine vs mlx-lm vs llama.cpp">

<img width="100%" src="docs/assets/perf-gemma4-prefill-box-whisker.svg" alt="Gemma 4 direct prefill: AX Engine vs mlx-lm vs llama.cpp">

<img width="100%" src="docs/assets/perf-gemma4-ttft-box-whisker.svg" alt="Gemma 4 direct TTFT: AX Engine vs mlx-lm vs llama.cpp">

**Qwen 3.6:**

<img width="100%" src="docs/assets/perf-qwen-decode-box-whisker.svg" alt="Qwen 3.6 direct decode: AX Engine vs mlx-lm vs llama.cpp">

<img width="100%" src="docs/assets/perf-qwen-prefill-box-whisker.svg" alt="Qwen 3.6 direct prefill: AX Engine vs mlx-lm vs llama.cpp">

<img width="100%" src="docs/assets/perf-qwen-ttft-box-whisker.svg" alt="Qwen 3.6 direct TTFT: AX Engine vs mlx-lm vs llama.cpp">

**How to read these charts**

- **Decode** is the main interactive-generation metric (tok/s; higher is better).
- **Prefill** and **TTFT** are cold-prompt cost; mixed across the historical
  composite — AX does **not** claim a matrix-wide prefill/TTFT lead on current
  HEAD.
- `llama.cpp` rows are shape-compatible GGUF Metal references, not prompt-hash
  parity with the MLX artifacts.
- Gemma 4 12B is a separate case study (native `gemma4_unified`; `mlx-lm` cannot
  load it) in
  [Performance Results](docs/PERFORMANCE-RESULTS.md#gemma-4-12b-retained-v682-case-study).

More detail (AX-only tables, embeddings, DiffusionGemma, archives):
**[Performance Results](docs/PERFORMANCE-RESULTS.md)** ·
[Performance interpretation](docs/PERFORMANCE.md).

## SDKs

Most clients use the OpenAI-compatible HTTP server. Python also has an
in-process session API.

| SDK | Docs |
| --- | --- |
| Rust | [docs/sdk/rust.md](docs/sdk/rust.md) |
| Python | [docs/sdk/python.md](docs/sdk/python.md) |
| JavaScript / TypeScript | [docs/sdk/javascript.md](docs/sdk/javascript.md) |
| Go | [docs/sdk/go.md](docs/sdk/go.md) |
| Ruby | [docs/sdk/ruby.md](docs/sdk/ruby.md) |
| Swift | [docs/sdk/swift.md](docs/sdk/swift.md) |
| Mojo | [docs/sdk/mojo.md](docs/sdk/mojo.md) |

## Server

```bash
ax-engine serve qwen36-35b --download --port 8080
curl http://127.0.0.1:8080/v1/runtime
```

Auth, streaming, embeddings, Ollama-shaped routes, and delegated backends:
[Server](docs/SERVER.md) · [API Compatibility](docs/API-COMPATIBILITY.md).

## Documentation

Full task map: [docs/README.md](docs/README.md).

| Need | Read |
| --- | --- |
| Install and first request | [Getting Started](docs/GETTING-STARTED.md) |
| Choose or download a model | [Supported Models](docs/SUPPORTED-MODELS.md) |
| Hardware / stack FAQ | [FAQ](docs/FAQ.md) |
| MTP packages and results | [MTP Docs](docs/mtp/README.md) |
| Full performance tables | [Performance Results](docs/PERFORMANCE-RESULTS.md) |
| Claim boundaries | [Performance Docs Map](docs/performance/README.md) |
| Reproduce benchmarks | [Benchmarks](docs/BENCHMARKS.md) |
| Server / API | [Server](docs/SERVER.md) · [API Compatibility](docs/API-COMPATIBILITY.md) |
| SDKs | [SDK Docs](docs/sdk/README.md) |
| Architecture / KV / scheduler | [Architecture](docs/ARCHITECTURE.md) |

## Development

```bash
cargo build --workspace
cargo test --quiet
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
maturin develop
python -m unittest discover -s python/tests -v
```

Crate layout, CI gates, and conventions: [AGENTS.md](AGENTS.md) ·
[Architecture](docs/ARCHITECTURE.md).

## Limitations

- Qwen3.5 long-prompt prefill can trail upstream MLX references on longer prompts
- Use pre-sanitized MLX community weights (or convert with `mlx_lm.convert` first)
- N-gram acceleration rows are workload-dependent, not raw kernel speedups

Details: [FAQ limitations](docs/FAQ.md#what-are-the-current-limitations).

## Contributing

Issue tickets, wishlist items, reproducible benchmarks, and docs feedback are
welcome. Unsolicited code PRs for runtime/kernel/scheduler/performance paths
are generally not accepted — open an issue first. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Community

- Website: [automatosx.com](https://automatosx.com)
- Discord: [Join us](https://discord.gg/aDhhburqJg)
- Email: [enquiry@defai.digital](mailto:enquiry@defai.digital)

## Acknowledgments

Thanks to **[Samuel Faj](https://www.samuelfaj.com/en/)**
([lightning-mlx](https://github.com/samuelfaj/lightning-mlx)) for support during
MTP peer-comparison work. Also see **[Remote Code](https://www.remotecode.io/)**.

## License

Apache License, Version 2.0. See [LICENSE](LICENSE).

Copyright (c) 2026 [DEFAI Private Limited](https://defai.digital)
