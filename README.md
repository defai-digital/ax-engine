# AX Engine

**Run LLMs on your Mac. No cloud. No compromise.**

AX Engine is a purpose-built Rust inference engine that runs large language models natively on Apple Silicon M3 and later (M3, M3 Pro, M3 Max, M3 Ultra, M4, and newer), using Metal GPU compute shaders for maximum throughput. It loads standard GGUF models and delivers performance competitive with llama.cpp — while offering a clean, modular Rust codebase designed for correctness and extensibility.

> **Important:** AX Engine is optimized for and tested on Apple Silicon **M3 or later**. M1 and M2 chips are not officially supported and may produce incorrect results, degraded performance, or unexpected behavior due to differences in Metal GPU architecture and capabilities.

## Why AX Engine?

**On-device AI matters.** Cloud inference means latency, cost, and data leaving your machine. M3+ chips have powerful GPU compute sitting idle in every MacBook and Mac Studio. AX Engine puts it to work.

### What makes it different

- **Metal-native from the ground up** — not a CUDA port. Custom compute shaders written specifically for Apple's unified memory architecture, eliminating unnecessary copies between CPU and GPU.

- **Correctness by design** — single-owner KV cache eliminates an entire class of state synchronization bugs. The type system prevents invalid backend combinations at compile time, not runtime.

- **Drop-in llama.cpp compatibility** — produces a `libllama.dylib` that matches the llama.h C API. Swap it into existing toolchains without changing a line of application code.

- **Pipelined GPU execution** — double-buffered Metal command buffers overlap compute and submission, keeping the GPU saturated instead of waiting on round-trips.

- **Speculative decoding** — a small draft model proposes tokens on CPU while the main model verifies on GPU in parallel, accelerating generation without sacrificing quality.

- **Pure Rust, minimal dependencies** — no C++ build chain, no Python, no CUDA toolkit. `cargo build` and you're running.

### Who is it for?

- **Application developers** building local-first AI features on macOS
- **Researchers** who need fast iteration on M3+ Macs without cloud dependency
- **Product teams** evaluating on-device inference for privacy-sensitive workloads
- **Anyone** who wants to run 8B+ parameter models at full speed on a MacBook

## Features

- **Metal GPU inference** — custom compute shaders for matmul, attention, dequantization, and elementwise ops
- **Pipelined decode** — double-buffered Metal command buffers for overlapped GPU execution
- **Speculative decoding** — draft-model acceleration with KV rollback on rejection
- **NEON fused matvec** — hand-tuned ARM NEON kernels for CPU decode path (Q8_0, Q6_K)
- **llama.cpp compatibility** — drop-in `libllama.dylib` shim via C API (`ax-shim`)
- **Multiple architectures** — LLaMA/LLaMA-3, Qwen3, Gemma 3, Phi-3/Phi-4, Falcon, Mistral, Mixtral (MoE), StarCoder2, GLM-4/5
- **Broad model support** — tuned performance profiles for 50+ models across 15 families
- **Built-in benchmarking** — throughput, layer profiling, and stability soak tests

## Supported Models

AX Engine includes tuned decode performance profiles for the following model families. Any GGUF model using a supported architecture (LLaMA, Qwen3, Gemma3) can run; models listed below have optimized kernel parameters.

| Family | Models | Sizes |
|--------|--------|-------|
| **LLaMA 3** | llama3-1b, 3b, 8b, 70b | 1B – 70B |
| **Code Llama** | codellama-8b | 8B |
| **Qwen3** | qwen3-1b, 2b, 4b, 8b, 14b, 32b, 72b | 1B – 72B |
| **Gemma 3** | gemma3-2b, 4b, 9b, 27b | 2B – 27B |
| **GLM-4** | glm4-9b, glm4-32b | 9B – 32B |
| **GLM-5** | glm5-9b, glm5-32b | 9B – 32B |
| **Phi (Microsoft)** | phi3-mini-4b, phi3-small-7b, phi3-medium-14b, phi4-14b | 4B – 14B |
| **Mistral** | mistral-8b, mistral-nemo-12b, mistral-small-22b | 8B – 22B |
| **Mixtral (MoE)** | mixtral-8x7b, mixtral-8x22b | 47B – 141B (active ~13B – 39B) |
| **Falcon** | falcon-7b, falcon-40b | 7B – 40B |
| **DeepSeek** | deepseek-qwen-2b, 8b, deepseek-llama-8b, deepseek-coder-7b, 33b | 2B – 33B |
| **StarCoder 2** | starcoder2-3b, 7b, 15b | 3B – 15B |

Performance profiles are in the [`perfs/`](perfs/) directory. To use a custom profile, place a JSON file matching the model name.

## Requirements

- macOS on Apple Silicon **M3 or later** (M1/M2 not supported)
- Xcode (for Metal compiler)
- Rust 1.88+ (edition 2024)

## Quick Start

```bash
# Build
cargo build --workspace --release

# Run inference
./target/aarch64-apple-darwin/release/ax-llama \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  -p "Hello, world"

# Interactive chat
./target/aarch64-apple-darwin/release/ax-llama \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --interactive --chat

# Speculative decoding (draft + target)
./target/aarch64-apple-darwin/release/ax-llama \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --speculative-draft ./models/small-draft.gguf \
  --speculative-k 4 \
  -p "Explain quantum computing"
```

See [QUICKSTART.md](QUICKSTART.md) for a detailed setup guide.

## Crate Structure

```
ax-cli   → ax-core → ax-metal
ax-shim  → ax-core
ax-bench → ax-core
```

| Crate | Description |
|-------|-------------|
| **ax-core** | Inference engine — GGUF loader, tokenizer, backends, KV cache, model forward passes, sampling |
| **ax-metal** | Metal GPU backend — device management, buffer pools, compute pipeline compilation, shader dispatch |
| **ax-cli** | `ax-llama` binary — llama.cpp-compatible CLI with single-prompt and interactive REPL modes |
| **ax-shim** | C API shim producing `libllama.dylib` — drop-in binary compatibility with llama.h |
| **ax-bench** | Benchmarking binary — `bench`, `profile`, and `soak` subcommands via Criterion |

## Backend Modes

| Config | Prefill | Decode | Notes |
|--------|---------|--------|-------|
| `Metal` | GPU | GPU | Full Metal acceleration |
| `Hybrid` (default) | GPU | GPU | Same as Metal with hybrid dispatch |
| `HybridCpuDecode` | GPU | CPU | GPU prefill, NEON CPU decode |
| `Cpu` | CPU | CPU | Fallback; set `AX_CPU_ONLY=1` |

## Benchmarking

```bash
# Full benchmark suite
./target/aarch64-apple-darwin/release/ax-bench bench \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf

# Layer-level profiling
./target/aarch64-apple-darwin/release/ax-bench profile \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf

# Stability soak test
./target/aarch64-apple-darwin/release/ax-bench soak \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf --smoke
```

## Development

```bash
cargo check --workspace            # compile check
cargo test --workspace             # run all tests
cargo clippy --workspace --tests -- -D warnings  # lint
cargo fmt --all -- --check         # format check
```

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `AX_CPU_ONLY` | `1` | Force CPU-only backend |
| `AX_DEBUG_LOGITS` | `1` | Dump top-5 logits each step |
| `AX_METAL_F16_KV_CACHE` | `auto`/`on`/`off` | f16 KV cache policy (auto: f16 when max_seq_len >= 256) |
| `AX_HYBRID_DECODE` | `cpu` | Select HybridCpuDecode backend |
| `AX_METAL_BARRIERS` | `0`/`1` | Toggle Metal buffer barriers (default: on) |

## Architecture Decisions

Key v2 design changes from v1:

- **Single-owner KV cache** — `ModelKv` enum eliminates the split-brain CPU/GPU sync bug from v1
- **Bias folding** — QKV bias applied once per batch instead of per-token (Qwen3 hot-path fix)
- **Explicit backend types** — `HybridCpuDecodeBackend` is its own type, not a hidden bool flag
- **`use_gpu_decode()` oracle** — single source of truth for GPU vs CPU decode dispatch

See the source code and commit history for full ADR details.

## License

MIT — Copyright (c) 2026 [DEFAI Private Limited](https://www.defai.digital)
