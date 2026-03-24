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

## Performance (v1.3)

Benchmarked on Apple M3 Max, Q4_K_M quantization, 256 prompt + 256 decode tokens:

| Model | Quant | Prefill tok/s | vs llama.cpp | Decode tok/s | vs llama.cpp |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | Q4_K_M | 656 | **93%** | 58.6 | **93%** |
| Meta-Llama-3.1-8B-Instruct | Q4_K_M | 664 | **89%** | 61.3 | **92%** |
| gemma-3-12b-it | Q4_K_M | 417 | **88%** | 37.7 | **93%** |

## Supported Models

**Minimum model sizes:**
- Generative / chat / code models: **6B+ parameters**
- Embedding models: **90M+ parameters**

All models must be in **GGUF format**. Recommended quantization: **Q4_K_M**. Also supported: Q6_K, Q8_0.

> MoE (Mixture-of-Experts) models such as Qwen3-30B-A3B and Qwen3-235B-A22B are not yet supported. Only dense architectures and Mixtral-style MoE are supported.

**Supported architectures and models:**

| Architecture | Tested Models | Quant | Sizes |
|---|---|---|---|
| LLaMA 3 | Meta-Llama-3.1-8B-Instruct, Meta-Llama-3-70B-Instruct | Q4_K_M, Q8_0 | 8B, 70B |
| Qwen 3 (dense) | Qwen3-8B, Qwen3-14B, Qwen3-32B | Q4_K_M | 8B, 14B, 32B |
| Qwen 3.5 | Qwen3.5-27B | Q4_K_M | 27B |
| Gemma 3 | gemma-3-12b-it, gemma-3-27b-it | Q4_K_M | 12B, 27B |
| Phi 3 / Phi 4 | Phi-3-small-8k-instruct, Phi-3-medium-14b-instruct, Phi-4-14B | Q4_K_M | 7B, 14B |
| Falcon | falcon-7b-instruct, falcon-40b-instruct | Q4_K_M | 7B, 40B |
| Mistral | Mistral-7B-Instruct-v0.3, Mistral-Nemo-Instruct-2407, Mistral-Small-24B-Instruct-2501 | Q4_K_M | 7B, 12B, 22B |
| Mixtral (MoE) | Mixtral-8x7B-Instruct-v0.1, Mixtral-8x22B-Instruct-v0.1 | Q4_K_M | 8x7B, 8x22B |
| StarCoder 2 | starcoder2-7b, starcoder2-15b | Q4_K_M | 7B, 15B |
| GLM-4 / GLM-5 | GLM-4-9B-Chat, glm-4-32b-0414, GLM-5-9B-Chat, GLM-5-32B-Chat | Q4_K_M | 9B, 32B |
| DeepSeek (distill) | DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-Coder-V2-Lite-Instruct | Q4_K_M | 7B, 8B, 33B |
| CodeLlama | CodeLlama-7b-Instruct-hf | Q4_K_M | 7B |

**Memory requirements** (approximate, Q4_K_M weights + KV cache at 4K context):

| Model Size | Weights | KV Cache | Total | Minimum Mac RAM |
|---|---:|---:|---:|---|
| 7–8B | ~4.5 GB | ~0.5 GB | ~5 GB | 16 GB |
| 12–14B | ~8 GB | ~1 GB | ~9 GB | 16 GB |
| 22–32B | ~18 GB | ~2 GB | ~20 GB | 32 GB |
| 70B | ~40 GB | ~4 GB | ~44 GB | 64 GB |
| 8x7B (MoE) | ~26 GB | ~2 GB | ~28 GB | 36 GB |
| 8x22B (MoE) | ~80 GB | ~5 GB | ~85 GB | 128 GB |

## Current Capabilities

- Metal GPU inference for prefill and decode
- Pipelined GPU decode paths where supported
- CPU and hybrid backend modes
- Speculative decoding with a draft model
- Library chat rendering helpers for common model families
- Blocked-layout batch matmul (llama.cpp kernel_mul_mm architecture)
- Concurrent Metal dispatch with smart barrier placement
- Precompiled Metal shaders (.metallib) for fast startup

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

Current speculative limitation:

- `--top-logprobs`, stop controls, token masks, and `--logit-bias` are not supported with speculative decoding

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
