# AX Engine

AX Engine is a Rust inference engine for running GGUF language models locally on Apple Silicon M3 and newer, with Metal-backed prefill and decode.

It is built for one specific target: fast, correct local inference on modern Macs without a Python runtime or a C++ build chain.

> AX Engine is currently optimized for Apple Silicon M3 and later. M1 and M2 are not supported targets.

## What It Is

- Metal-native GGUF inference for macOS on Apple Silicon
- A llama.cpp-style CLI: `ax-llama`
- A `libllama`-compatible shim via `ax-shim`
- Benchmarking and soak-test tools via `ax-bench`
- A Rust codebase organized around explicit backends, model configs, tokenizer/gguf loading, KV management, and sampling

## What It Is Not

- A general-purpose multi-platform inference runtime
- A cloud serving stack
- A Python-first project
- A full OpenAI-compatible server

If you need a production HTTP server, routing, request validation, or adapter logic, that should sit above AX Engine.

## Why Use It

- Local-first: models run on your machine, not a remote API
- Focused architecture: the project is optimized around Apple Silicon GPU execution instead of trying to be portable first
- Clean integration surface: `ax-core` is usable as a library, `ax-cli` is usable as a tool, and `ax-shim` is usable for llama.cpp-style integrations
- Modern sampling and stopping surface: `top_k`, `top_p`, `min_p`, `min_keep`, `logit_bias`, allowed/banned-token masks, repetition/presence/frequency penalties, `top_logprobs`, stop strings, and stop token IDs
- Strong development ergonomics: pure Rust workspace, normal `cargo` workflows, no custom toolchain beyond Xcode and Rust

## Current Capabilities

- Metal GPU inference for prefill and decode
- Pipelined GPU decode paths where supported
- CPU and hybrid backend modes
- Speculative decoding with a draft model
- Library chat rendering helpers for common model families
- Multiple supported model architectures, including:
  - LLaMA / LLaMA 3
  - Qwen 3
  - Gemma 3
  - Phi 3 / Phi 4
  - Falcon
  - Mistral / Mixtral
  - StarCoder 2
  - GLM variants

## Requirements

- macOS on Apple Silicon M3 or newer
- Xcode with Metal tooling available
- Rust 1.88 or newer

## Quick Start

Build the workspace:

```bash
cargo build --workspace --release
```

Run a prompt against a GGUF model:

```bash
./target/release/ax-llama \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --chat \
  --prompt "Explain speculative decoding in plain English."
```

Start an interactive chat session:

```bash
./target/release/ax-llama \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --interactive \
  --chat
```

Capture top logprobs while generating:

```bash
./target/release/ax-llama \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --chat \
  --prompt "Write one sentence about Rust." \
  --top-logprobs 5
```

Stop on a string or token ID without leaking the stop text:

```bash
./target/release/ax-llama \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --prompt "Write a list with END at the end." \
  --stop "END" \
  --stop-token-id 2
```

Run speculative decoding:

```bash
./target/release/ax-llama \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --speculative-draft ./models/small-draft.gguf \
  --speculative-k 4 \
  --experimental \
  --prompt "Summarize the purpose of KV cache reuse."
```

For a step-by-step setup guide, see [QUICKSTART.md](./QUICKSTART.md).

## Common Commands

Run a normal prompt:

```bash
./target/release/ax-llama \
  --model ./models/<model>.gguf \
  --prompt "Hello"
```

Use a larger context window:

```bash
./target/release/ax-llama \
  --model ./models/<model>.gguf \
  --ctx-size 8192 \
  --prompt "Summarize this document..."
```

Tune sampling:

```bash
./target/release/ax-llama \
  --model ./models/<model>.gguf \
  --prompt "Write a short story about a lighthouse." \
  --temp 0.9 \
  --top-p 0.95 \
  --min-p 0.05 \
  --min-keep 2 \
  --repeat-penalty 1.1 \
  --repeat-last-n 128
```

Bias a specific token up or down:

```bash
./target/release/ax-llama \
  --model ./models/<model>.gguf \
  --prompt "Answer in one word." \
  --logit-bias 42=2.0 \
  --logit-bias 7=-5.0
```

Restrict generation to a known token subset:

```bash
./target/release/ax-llama \
  --model ./models/<model>.gguf \
  --prompt "Answer with yes or no." \
  --allow-token-id 1234 \
  --allow-token-id 5678
```

Ban specific tokens outright:

```bash
./target/release/ax-llama \
  --model ./models/<model>.gguf \
  --prompt "Answer briefly." \
  --ban-token-id 2
```

Use explicit stop controls:

```bash
./target/release/ax-llama \
  --model ./models/<model>.gguf \
  --prompt "Return one line and then STOP." \
  --stop "STOP"
```

Print detailed runtime metrics:

```bash
./target/release/ax-llama \
  --model ./models/<model>.gguf \
  --prompt "Hello" \
  --verbose
```

## Benchmarking

Build release binaries first:

```bash
cargo build --workspace --release
```

Then run:

```bash
./target/release/ax-bench bench --model ./models/<model>.gguf
./target/release/ax-bench profile --model ./models/<model>.gguf
./target/release/ax-bench soak --model ./models/<model>.gguf --smoke
```

## Workspace Layout

```text
crates/ax-core   core inference engine
crates/ax-metal  Metal backend and shaders
crates/ax-cli    ax-llama CLI
crates/ax-shim   llama.cpp-compatible C API shim
crates/ax-bench  benchmarking and soak tooling
```

## Development

Core development commands:

```bash
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --tests -- -D warnings
cargo fmt --all -- --check
```

Useful targeted commands:

```bash
cargo test -p ax-core
cargo test -p ax-core sampling::
cargo run -p ax-cli -- --help
```

## Environment Variables

| Variable | Values | Meaning |
|---|---|---|
| `AX_CPU_ONLY` | `1` | Force CPU-only backend |
| `AX_HYBRID_DECODE` | `cpu` | Use CPU decode with GPU prefill |
| `AX_METAL_F16_KV_CACHE` | `auto`, `on`, `off` | KV cache precision policy |
| `AX_METAL_BARRIERS` | `0`, `1` | Toggle Metal buffer barriers |
| `AX_DEBUG_LOGITS` | `1` | Dump top logits during generation |
| `RUST_LOG` | standard tracing filter | Enable debug logging |

## Project Status

AX Engine is under active development. The current focus is:

- strong local inference performance on supported Apple Silicon targets
- explicit backend behavior
- stable library surfaces for prompt rendering, sampling, and decode
- performance and correctness tooling in-tree

There are still deliberate gaps relative to larger serving stacks, especially around server/runtime concerns such as request orchestration, continuous batching at the serving layer, and OpenAI-compatible HTTP semantics.

## License

MIT. Copyright (c) 2026 [DEFAI Private Limited](https://www.defai.digital)
