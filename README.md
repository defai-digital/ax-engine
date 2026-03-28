# AX Engine

A native-first LLM inference engine for Apple Silicon M3+, built around one
idea:

**for supported local transformer workloads, a domain-specific fused runtime can
extract more value from Apple GPUs and Apple UMA than a general-purpose graph
executor.**

> Requires macOS on Apple Silicon M3 or newer, Xcode, and Rust 1.88+.

## Why Have AX Engine

`llama.cpp` already provides excellent GGUF coverage. MLX and MLX-based engines
already align well with Apple's high-level stack. AX exists for a narrower and
more opinionated reason:

- to own a **native Apple-silicon performance path** for supported transformer
  model families
- to optimize both the **execution path** and the **memory path**
- to take Apple-only decisions that a portable engine cannot prioritize as
  aggressively

AX is therefore not trying to be:

- a generic tensor graph executor
- a universal GGUF engine
- a thin wrapper around `llama.cpp`

For supported native models, AX runs through its own Metal runtime.
For unsupported models, AX can route to `llama.cpp` as a compatibility
fallback so coverage does not collapse into crashes or dead ends.

For a research-level implementation comparison, see
[AX Engine vs llama.cpp](./docs/ax-engine-vs-llama-cpp.md).

## Design Philosophy

Most inference engines execute a model as a graph of small ops:

- matmul
- add
- norm
- RoPE
- KV updates
- elementwise activation

That is flexible, but it creates overhead on Apple GPUs:

- more dispatches
- more pipeline and resource binding churn
- more intermediate memory traffic
- fewer opportunities to reuse data in registers or threadgroup memory

AX takes the opposite stance:

- it is **transformer-specific**, not a general graph runtime
- it prefers **selective fusion** where the fused path is structurally faster
- it treats **Apple UMA** as a first-class optimization surface, not just a
  background hardware fact

The project is benchmark-gated on purpose. AX does not assume that every
possible fusion or every experimental kernel should ship by default. If a path
is not yet stable across real models, it stays opt-in.

## What AX Optimizes

AX's native edge is built on two structural surfaces.

### 1. Execution Path

AX reduces work on the hot path by replacing split operator sequences with
transformer-specific fused kernels and model-aware scheduling.

Examples of the current fused path include:

- QKV split + bias + QK norm + RoPE + KV append
- residual add + RMSNorm
- selected FFN-side pair kernels
- pipelined decode with model-aware execution planning

This is why some families benefit more than others. Llama 3 currently benefits
the most because its decode path lines up well with AX's fusion depth. Qwen3
improved materially once the post-QKV fused decode path shipped, but decode-side
fusion is still the main near-term performance priority. AX is not trying to
"fuse everything"; it is trying to fuse where dispatch savings, occupancy, and
memory traffic all still make sense together.

### 2. Memory Path

AX is also designed around Apple Silicon's unified memory architecture.

The most important upgrade in this direction is **mmap-backed no-copy Metal
buffer aliasing** for model weights where Metal accepts the alias. That matters
strategically because it is not just a shader trick. It improves the runtime's
memory path itself:

- less copy overhead on model load
- better alignment with long-lived resident models
- a stronger base for future KV and prompt-cache work
- more direct use of Apple Silicon's shared address-space advantages

This is one of the clearest reasons AX exists as a native Apple-silicon engine
instead of just another frontend around someone else's runtime.

## Native-First, Compatibility-Backed

AX should be read as:

- **native-first** for supported model families
- **compatibility-backed** for unsupported coverage

That means:

- native AX is the product path
- `llama.cpp` fallback is the safety net
- benchmark claims should be made on AX's supported native set, not on fallback
  coverage

Use routing when you want coverage. Use AX native when you want to test or ship
the Apple-specific path.

## Current Optimization Posture

AX's current optimization posture is deliberate rather than random:

- Llama 3 benefits strongly from the current fusion depth
- Qwen3 decode improved after the fused post-QKV path landed
- deeper decode-side fusion remains the highest-value next step, especially for
  dense transformer decode
- FFN-side decode prototypes remain benchmark-gated and are not enabled by
  default if they regress common workloads
- profile-driven tuning exists today and is expected to become more
  regime-sensitive over time

This is the pattern AX wants: performance should be explainable by the runtime's
structure, not by isolated one-off benchmark wins.

## Fusion Architecture

A standard transformer layer has roughly 15-20 logical operations. A generic
executor may dispatch many of them independently. AX fuses selected regions into
fewer, larger kernels where the fused path is worth owning.

**Attention preparation** (4-9 ops fused into 1 dispatch):

```text
QKV split + bias add + per-head QK norm + RoPE + KV cache append
```

A single kernel reads the fused QKV projection output, splits it, adds
per-component bias when needed, normalizes Q and K heads for model families
that require it, applies rotary position embedding, and writes K/V directly to
the KV cache.

**Residual + normalization** (2 ops fused into 1 dispatch):

```text
hidden += projection_output; norm_out = RMSNorm(hidden)
```

The residual addition at the end of one sub-block is fused with the RMSNorm at
the start of the next, reducing an extra pass over the same data.

**Selected FFN pair kernels**:

```text
gate = W_gate @ x; up = W_up @ x
```

Where the shape and the measured results justify it, AX computes multiple
projections from the same input load instead of paying that bandwidth cost
twice.

For a 32-layer model, this kind of structural fusion can remove on the order of
100-200 Metal dispatches per forward pass compared with a more split execution
strategy.

## Prefill Kernel Strategy

Prefill throughput depends heavily on the batched matmul path. AX uses a
blocked-layout batch matmul design with bounded shader specialization:

- tile family tuned for Apple GPU execution
- blocked data layout to improve cache behavior
- vectorized tile loading
- function constants for profitable hot-path specialization such as full-tile
  vs edge-tile handling
- quantized coverage for common GGUF families including Q4_K, Q5_K, Q6_K,
  Q8_0, and Q4_0

The goal is not to copy `llama.cpp` parameter-for-parameter. The goal is to
adopt similar structural wins only where AX's own runtime and buffer contracts
actually match.

## Tuning Model

AX uses JSON-driven kernel and routing profiles today:

```text
perfs/qwen3-8b.json    perfs/gemma3-12b.json    perfs/llama3-70b.json
```

These profiles currently control things like:

- threadgroup sizing
- tile choices
- kernel variants
- attention strategy

This is useful, but it is not the end state. The direction is toward more
regime-sensitive tuning across:

- architecture
- hidden size
- head dimension
- quant family
- short vs long prefill
- short vs long decode
- memory-pressure mode

## Performance

AX Engine vs llama.cpp on Apple M3 Max (March 2026). Values over 100% mean AX was faster.

| Model | AX prefill | llama.cpp prefill | AX vs llama.cpp | AX decode | llama.cpp decode | AX vs llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | 675.8 tok/s | 639.2 tok/s | **105.7%** | 61.2 tok/s | 47.1 tok/s | **129.9%** |
| Llama 3 70B | 55.7 tok/s | 66.8 tok/s | 83.4% | 6.5 tok/s | 6.3 tok/s | **103.2%** |
| Qwen3 8B | 727.4 tok/s | 736.7 tok/s | 98.7% | 62.4 tok/s | 60.3 tok/s | **103.5%** |
| Qwen3 14B | 391.6 tok/s | 408.2 tok/s | 95.9% | 37.1 tok/s | 35.6 tok/s | **104.2%** |
| Qwen3 32B | 126.7 tok/s | 150.7 tok/s | 84.1% | 15.6 tok/s | 14.9 tok/s | **104.7%** |
| Qwen3.5 9B | 240.5 tok/s | 732.2 tok/s | 32.8% | 26.8 tok/s | 48.9 tok/s | 54.8% |
| Gemma 3 12B | 420.5 tok/s | 463.3 tok/s | 90.8% | 39.6 tok/s | 35.8 tok/s | **110.5%** |
| Gemma 3 27B | 155.7 tok/s | 170.2 tok/s | 91.5% | 17.8 tok/s | 15.1 tok/s | **117.9%** |

LLaMA 3 benefits most from AX's fusion depth (10 dispatches/layer vs ~20 in a
per-op executor). That is the cleanest example of AX's design philosophy
showing up in measured results.

Qwen3 benefits less, but its decode path improved once the fused post-QKV route
(`qkv=fused`) shipped. That is the kind of movement AX wants to see: a runtime
change with a clear structural explanation, not a random benchmark spike.

Qwen3.5 9B is the first hybrid attention+SSM (Mamba-2/GDN) model supported. Its
decode improved from 12.9 to 26.8 tok/s via GPU-unified decode, reaching ~55%
of `llama.cpp` on decode. Prefill now measures 240.5 tok/s (~33% of
`llama.cpp`) after re-enabling the GPU-unified prefill path. The remaining gap
still traces to sequential host orchestration and hybrid recurrent state
handling (10,496 decode cmd_buf submissions, 1,024 decode barriers, and 165
prefill cmd_buf submissions for the 512/128 throughput benchmark, vs 128 decode
submissions for the pure-transformer pipelined path).

Prefill uses fused Q4K/Q5K/Q6K dequant batch matmul kernels with native
token-major layout. GPU attention KV is f16 by default for all models.
Synthetic throughput benches mask stop tokens during decode measurement so
long-context rows like `Qwen3 32B` do not collapse to `0 tok/s` when EOS wins
the first sampled step.

These numbers should be read as a pattern, not a blanket claim:

- AX wins most clearly where its fused native path is already deep
- AX improves when a real fused path replaces a split one
- AX still leaves performance on the table where decode fusion, specialization,
  or resource policy is not yet deep enough

See [BENCHMARKING.md](./BENCHMARKING.md) for methodology.

## Supported Models

All models must be in **GGUF format**. Recommended quantization: **Q4_K_M**. Also supported: Q5_K, Q6_K, Q8_0.

| Architecture | Models | Sizes |
|---|---|---|
| LLaMA 3 | Meta-Llama-3.1-8B-Instruct, Meta-Llama-3-70B-Instruct | 8B, 70B |
| Qwen 3 (dense) | Qwen3-8B, Qwen3-14B, Qwen3-32B | 8B, 14B, 32B |
| Qwen 3.5 (hybrid) | Qwen3.5-9B, Qwen3.5-27B | 9B, 27B |
| Gemma 3 | gemma-3-12b-it, gemma-3-27b-it | 12B, 27B |

Minimum: 6B+ parameters for generative models. MoE models are not supported.

**Memory requirements** (Q4_K_M, 4K context):

| Model Size | Total | Minimum Mac RAM |
|---|---:|---|
| 7-8B | ~5 GB | 16 GB |
| 12-14B | ~9 GB | 16 GB |
| 22-32B | ~20 GB | 32 GB |
| 70B | ~44 GB | 64 GB |

## Capabilities

**Inference**: Metal GPU prefill and decode, pipelined double-buffered decode, CPU and hybrid backends, speculative decoding with draft models.

**GPU kernels**: Blocked-layout batch matmul (Q4_K/Q5_K/Q6_K/Q8_0/Q4_0), simdgroup-matrix Flash Attention (HD=64/128), split-K decode attention, fused QKV+bias+QKnorm+RoPE+KV-append, fused residual+RMSNorm, gate+up pair kernel, concurrent dispatch with smart barriers.

**Multi-model**: Per-model Metal command queues and KV caches, graceful OOM handling, explicit GPU/CPU backend per model, thread-safe (`Arc`/`Mutex`).

**Integration**: core runtime (`ax-engine-core`), high-level Rust SDK facade (`ax-engine-sdk`), llama.cpp-compatible CLI (`ax-engine`), basic OpenAI-compatible inference server built on the Rust SDK (`ax-engine-server`), drop-in `libllama` shim (`ax-engine-shim`), Python bindings built on the Rust SDK (`ax-engine-py`), Node.js / Next.js HTTP client SDK (`packages/ax-engine-js`), benchmarking tools (`ax-engine-bench`).

## Quick Start

```bash
cargo build --workspace --release

# Single prompt
./target/release/ax-engine \
  --model ./models/Qwen3-8B-Q4_K_M.gguf \
  --chat --prompt "Explain speculative decoding."

# Interactive
./target/release/ax-engine \
  --model ./models/Qwen3-8B-Q4_K_M.gguf \
  --interactive --chat

# Benchmark
./target/release/ax-engine-bench bench --model ./models/Qwen3-8B-Q4_K_M.gguf

# Basic inference server
./target/release/ax-engine-server \
  --model ./models/Qwen3-8B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 3000
```

The server exposes `GET /healthz`, `GET /v1/models`, `POST /v1/completions`, and
`POST /v1/chat/completions`. It is intentionally basic: one loaded model, one
request at a time, OpenAI-compatible JSON plus SSE streaming, and it now shares
the same high-level Rust SDK path used by `ax-engine-py`. For multi-model
routing, continuous batching, and production serving concerns, keep using
`ax-serving`.

Inference routing is also built into the repo surfaces now. Set
`AX_ROUTING=auto` to keep native AX for supported models and fall back to
`llama.cpp` for unsupported GGUF architectures:

```bash
AX_ROUTING=auto \
./target/release/ax-engine \
  --model ./models/Mistral-7B-Instruct.Q4_K_M.gguf \
  --prompt "Explain unified memory."
```

The CLI prints a one-line routing summary, and `ax-engine-server`,
`ax-engine-sdk`, and `ax-engine-py` use the same routing behavior.

Server documentation:

- [docs/AX Engine Server Overview](./docs/ax-engine-server.md)
- [docs/AX Engine Server API](./docs/ax-engine-server-api.md)
- [docs/Inference Routing](./docs/routing.md)
- [docs/JavaScript SDK](./docs/js-sdk.md)
- [docs/Rust SDK](./docs/rust-sdk.md)
- [docs/Index](./docs/README.md)

For Node.js and Next.js software, use the JavaScript SDK over the built-in
server instead of trying to bind Metal/Rust directly into a JS runtime:

```js
import { AxEngineClient } from "@defai.digital/ax-engine-js";

const client = new AxEngineClient({
  baseURL: "http://127.0.0.1:3000",
  defaultModel: "Qwen3-8B-Q4_K_M",
});

const response = await client.chat.completions.create({
  messages: [{ role: "user", content: "Summarize AX Engine in one sentence." }],
});

console.log(response.choices[0].message.content);
```

The JS SDK also exposes a client-side `responses` compatibility layer for apps
that prefer that API shape, while the transport still runs over the existing AX
server endpoints.

See [QUICKSTART.md](./QUICKSTART.md) for setup details and [BENCHMARKING.md](./BENCHMARKING.md) for benchmark methodology.

## Workspace Layout

```
crates/ax-engine-core    Core inference engine (backends, models, KV, sampling)
crates/ax-engine-sdk     High-level Rust SDK facade
crates/ax-engine-metal   Metal GPU backend (device, shaders, dispatch, profiles)
crates/ax-engine-cli     ax-engine CLI
crates/ax-engine-server  Basic OpenAI-compatible inference server
crates/ax-engine-shim    llama.cpp-compatible C API shim (libllama.dylib)
crates/ax-engine-bench   Benchmarking, profiling, and soak testing
packages/ax-engine-js    Node.js / Next.js client SDK for ax-engine-server
```

## Development

```bash
cargo check --workspace                        # compile check
cargo test --workspace                         # all tests
cargo clippy --workspace --tests -- -D warnings  # lint
cargo fmt --all -- --check                     # format check
```

## License

MIT. Copyright (c) 2026 [DEFAI Private Limited](https://www.defai.digital)
