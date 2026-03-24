# AX Engine

AX Engine is a focused Rust inference engine for serious local LLM workloads on Apple Silicon M3 and newer.

It is built for one specific target: fast, correct, Mac-native GGUF inference without a Python runtime or a C++ build chain. AX Engine is not trying to be a general-purpose serving platform. It is designed for local execution, explicit backend control, and clean integration into local AI systems.

> AX Engine is currently optimized for Apple Silicon M3 and later. M1 and M2 are not supported targets.

## What It Is

- Metal-native GGUF inference for macOS on Apple Silicon
- A Rust library (`ax-core`) with explicit backends, model configs, KV management, and sampling — designed to be embedded into larger systems
- A llama.cpp-style CLI via `ax-llama`
- A `libllama`-compatible shim via `ax-shim`
- Benchmarking and soak-test tools via `ax-bench`

## What It Is Not

- A general-purpose multi-platform inference runtime
- A cloud serving stack or OpenAI-compatible HTTP server
- A Python-first project
- A claim to universal performance superiority over llama.cpp

AX Engine is probably not the right fit if you need:

- Broad cross-platform model support beyond Apple Silicon
- Cloud-scale multi-tenant serving
- The widest possible quantization ecosystem
- Immediate parity across every model that llama.cpp supports

If you need production HTTP serving, request orchestration, or API adapters, those concerns should sit above AX Engine.

## Why Use It

- **Local-first execution**: models run on your machine, not a remote API
- **Apple Silicon focus**: optimized for Mac GPU execution instead of portability-first tradeoffs
- **Explicit backend control**: you decide what runs on GPU vs CPU, with per-model KV budgets and graceful OOM handling (returns errors, never kills the process)
- **Multi-model safe**: each model gets its own Metal command queue, KV cache, and backend instance — designed for running multiple models in the same process without shared global state conflicts
- **Clean integration surface**: usable as a library, CLI, or drop-in `libllama` compatibility shim
- **Rust-native**: pure Rust workspace, no Python runtime, no C++ build chain, normal `cargo` workflows

## Current Optimization Focus

AX Engine is currently most relevant for:

- Dense GGUF models in the 14B–70B range on Apple Silicon
- Model families with optimization depth: **Qwen 3 dense**, **LLaMA 3**, **Gemma 3**
- Workflows that benefit from explicit backend control and long-running local inference
- Mac-native deployments where stable, isolated execution matters more than broad platform coverage

## Performance (v1.3.2)

Benchmarked on Apple M3 Max, Q4_K_M quantization, 512 prompt + 128 decode tokens.

AX Engine **matches or beats llama.cpp on decode** across all tested models. Pipelined decode with f16 KV cache.

| Model | Quant | Prefill tok/s | vs llama.cpp | Decode tok/s | vs llama.cpp |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | Q4_K_M | 655 | **87%** | 58.8 | **103%** |
| Qwen3-14B | Q4_K_M | 354 | **90%** | 35.7 | **102%** |
| Qwen3-32B | Q4_K_M | 142 | **84%** | 16.0 | **99%** |
| Meta-Llama-3.1-8B-Instruct | Q4_K_M | 640 | **85%** | 60.2 | **100%** |
| gemma-3-12b-it | Q4_K_M | 420 | **86%** | 39.5 | **102%** |

Results measured with cooldown between runs to avoid thermal throttling.

Areas where we expect the most improvement:

- Prefill throughput (closing the 85–87% gap via kernel tuning)
- Larger dense models (32B–70B), where decode already approaches or exceeds llama.cpp
- Model-specific kernel fusions for Qwen3 QKV bias and other architecture-specific operations

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


**Memory requirements** (approximate, Q4_K_M weights + KV cache at 4K context):

| Model Size | Weights | KV Cache | Total | Minimum Mac RAM |
|---|---:|---:|---:|---|
| 7–8B | ~4.5 GB | ~0.5 GB | ~5 GB | 16 GB |
| 12–14B | ~8 GB | ~1 GB | ~9 GB | 16 GB |
| 22–32B | ~18 GB | ~2 GB | ~20 GB | 32 GB |
| 70B | ~40 GB | ~4 GB | ~44 GB | 64 GB |

## Current Capabilities

**Inference**
- Metal GPU inference for prefill and decode
- Pipelined GPU decode with double-buffered command buffers
- CPU and hybrid backend modes with explicit selection
- Speculative decoding with a draft model

**GPU Optimization**
- Blocked-layout batch matmul (llama.cpp `kernel_mul_mm` architecture)
- Model-specific fused kernels (QKV bias + QK norm + RoPE + KV append in a single dispatch)
- Concurrent Metal dispatch with smart barrier placement
- Precompiled Metal shaders (`.metallib`) for fast startup
- simdgroup-matrix Flash Attention (HD=64 and HD=128)

**Multi-Model Foundation**
- Per-model Metal command queues and KV caches — no shared mutable state between models
- Graceful OOM handling — buffer allocation failures return errors, never crash the process
- Explicit backend per model — run one model on GPU, another on CPU, in the same process
- Thread-safe throughout (`Arc`/`Mutex`, no `Rc`/`RefCell`)

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

## Roadmap

**Now** (v1.x):
- Steady performance progress toward llama.cpp parity across supported model families
- Model-specific kernel fusions for Qwen3, GLM, StarCoder2
- Stable `ax-core` library surface for embedding into local AI toolchains

**Next** (v2.x):
- Multi-model resource isolation: per-model KV budgets, memory admission control, graceful degradation under memory pressure
- Per-model kernel profiles (currently process-global) — enabling correct dispatch when multiple models share a process
- Paged KV prefix cache for long-context and multi-turn workloads

**Later**:
- Model-level scheduler for multi-model GPU sharing (prefill/decode priority, latency-sensitive vs throughput policies)
- Q8 KV quantization for reduced memory footprint
- Fused QKV weights (3 matmuls to 1)

## License

MIT. Copyright (c) 2026 [DEFAI Private Limited](https://www.defai.digital)
