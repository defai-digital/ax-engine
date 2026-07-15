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
- **Evidence-backed benchmarks** — public claims are tied to checked-in artifacts
  with route identity, sampler, accept rate, and provenance

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

AX Engine publishes session-separated results: **MTP generation**, **direct
generation**, and **embeddings**. Do not compare rows across modes unless a
doc says they share a same-artifact denominator.

**Headline (exact sampled MTP, 6-bit packages):** all 15 target/suite rows
accelerate decode by **1.41×–2.66×** over same-package direct, with 100% MTP
step coverage.

<img src="docs/assets/perf-mtp-6bit-ax-acceleration.svg" alt="AX Engine 6-bit exact sampled-MTP acceleration">

**Qwen3.6 MTP peer decode (production-config rows):**

| Target | AX Engine | MTPLX | lightning-mlx |
| --- | ---: | ---: | ---: |
| 27B 4-bit | 63.0 tok/s | 58.5 tok/s | 55.7 tok/s |
| 35B-A3B 4-bit | 172.4 tok/s | 137.9 tok/s | 116.2 tok/s |
| 35B-A3B 6-bit | 141.2 tok/s | 119.0 tok/s | 96.3 tok/s |

Full tables, charts, methodology, embeddings, and historical archives:

- **[Performance Results](docs/PERFORMANCE-RESULTS.md)** — all session-mode tables and charts
- [Performance interpretation](docs/PERFORMANCE.md) · [Claim boundaries](docs/performance/README.md)
- [Benchmarks](docs/BENCHMARKS.md) · [MTP peer comparison](docs/mtp/qwen36-peer-comparison.md)

> [!IMPORTANT]
> Prefill/TTFT comparisons require the same resolved `libmlx` on both sides.
> Some Homebrew / low-deployment-target MLX builds omit M5 GEMM paths and look
> ~3–4× slower. See the MLX admission note in
> [Performance Results](docs/PERFORMANCE-RESULTS.md).

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

| Need | Read |
| --- | --- |
| Install and first request | [Getting Started](docs/GETTING-STARTED.md) |
| Choose or download a model | [Supported Models](docs/SUPPORTED-MODELS.md) |
| MTP packages and results | [MTP Docs](docs/mtp/README.md) |
| Full performance tables | [Performance Results](docs/PERFORMANCE-RESULTS.md) |
| Reproduce benchmarks | [Benchmarks](docs/BENCHMARKS.md) |
| Architecture / KV / scheduler | [Architecture](docs/ARCHITECTURE.md) |
| Docs hub | [docs/README.md](docs/README.md) |

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
