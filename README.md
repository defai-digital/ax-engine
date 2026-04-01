# AX Engine

A native-first LLM inference engine for Apple Silicon M3+, built around one
idea:

**for supported local transformer workloads, a domain-specific fused runtime can
extract more value from Apple GPUs and Apple UMA than a general-purpose graph
executor.**

> Requires macOS on Apple Silicon M3 or newer, Xcode, and Rust 1.88+.

## Why Have AX Engine

The local inference ecosystem already has world-class engines.
[llama.cpp](https://github.com/ggerganov/llama.cpp) is a state-of-the-art
portable runtime — its GGUF format, quantization kernels, and cross-platform
coverage are industry-leading work that we respect and learn from.
[vLLM](https://github.com/vllm-project/vllm) and
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) dominate
server-side GPU inference. [MLX](https://github.com/ml-explore/mlx) and
MLX-based engines align well with Apple's high-level stack.

AX Engine is not trying to compete with any of them on their home turf.

We are a small team that noticed a narrow gap: on **Apple Silicon Macs**,
for a **curated set of transformer model families**, a purpose-built runtime
that makes Apple-only decisions can extract more from the hardware than a
portable engine can. That is the entire thesis — a niche bet on a niche
platform:

- own a **native Apple-silicon performance path** for supported models
- optimize both the **execution path** and the **memory path** on Apple UMA
- take Apple-only decisions (fused Metal dispatches, UMA buffer contracts,
  single-owner KV, model-aware execution plans) that a portable engine
  cannot prioritize as aggressively

AX is therefore not trying to be:

- a generic tensor graph executor
- a universal GGUF engine
- a replacement for llama.cpp, vLLM, or TensorRT-LLM

For supported native models, AX runs through its own Metal runtime.
For unsupported models, AX routes to `llama.cpp` as a compatibility
fallback — because llama.cpp is already excellent at what it does, and
there is no reason to reimplement coverage that already exists.

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

- QKV split + bias + QK norm + RoPE + KV append (up to 9 ops in 1 dispatch)
- residual add + RMSNorm (layer handoff without an extra memory pass)
- fused activation + down projection matvec (SiLU/GELU + Wd in 1 kernel)
- pipelined double-buffered decode with model-aware execution planning
- concurrent Metal dispatch with SmartBarrier conflict detection

AX is not trying to "fuse everything" — it fuses where dispatch savings,
occupancy, and memory traffic all make sense together. The benefit varies by
model family: LLaMA 3 and Qwen3 benefit strongly from the current fusion
depth, Qwen3.5 benefits from the hybrid attention + SSM pipeline, and
Gemma 3 benefits from the per-head QK norm fusion path.

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

## Current Optimization Posture

AX's current optimization posture is deliberate rather than random:

- **Dense transformer decode** benefits strongly from the current fusion depth.
  LLaMA 3, Qwen3, and Gemma 3 all use the shared GPU decode layer encoder
  with fused QKV + attention + residual handoff.
- **Qwen3.5 hybrid attention + SSM** is natively supported. The recurrent
  (Mamba-2) layers use GPU-resident state with pipelined batch prefill and
  fused GDN (gated delta net) kernels.
- **Concurrent Metal dispatch** with per-dispatch SmartBarrier conflict
  detection overlaps independent GPU work (Q/K/V matvecs, gate/up
  projections) while inserting barriers only at data-hazard boundaries.
- **Split-K decode attention** distributes the KV-scan across multiple
  threadgroups for long contexts, with a lightweight reduce step.
- **Speculative decoding** is supported via `--speculative-draft`: a small
  draft model runs K steps, the target model batch-verifies, and rejected
  tokens roll back via KV truncation.
- FFN-side decode prototypes remain benchmark-gated and are not enabled by
  default if they regress common workloads.
- Profile-driven tuning with per-model heuristics and decode regime routing
  exists today and is expected to become more regime-sensitive over time.

Performance should be explainable by the runtime's structure, not by isolated
one-off benchmark wins. Benchmark claims are made on AX's supported native
set, not on llama.cpp fallback coverage.

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

**Fused activation + down projection**:

```text
output = W_down @ (SiLU(gate) * up)
```

The activation function and element-wise multiply are fused into the
down-projection matvec, eliminating the intermediate buffer write between
the FFN activation and the final projection.

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
- quantized coverage for common GGUF families: Q4_K, Q5_K, Q6_K, Q8_0

The goal is not to copy `llama.cpp` parameter-for-parameter. The goal is to
adopt similar structural wins only where AX's own runtime and buffer contracts
actually match.

## Kernel Provenance

AX adopts industry best practices at the individual kernel level — including
quantized dequantization strategies that originated in `llama.cpp` — while
providing structural advantages above the kernel layer through fused
scheduling, execution planning, and UMA-aware memory paths that a
general-purpose graph executor cannot achieve.

This is a deliberate stance, not an accident. At the Metal shader level, there
are a limited number of efficient ways to dequantize Q4_K data and multiply it
on Apple GPUs. The blocked tile geometry and nibble-extraction patterns that
`llama.cpp` established are effectively the known-good solutions for this
hardware. Reinventing them differently would not produce faster code.

AX is transparent about this. Of the 200+ Metal kernel entry points in the
shader set:

| Category | Total | Directly attributed | Structurally influenced | AX-original |
|---|---:|---:|---:|---:|
| Dequant (batch + matvec) | 100+ | 9 | ~20 | ~70+ |
| Attention | 30 | 2 | — | 28 |
| Elementwise + GDN | 63 | 4 | — | 59 |
| General matmul | 3 | — | — | 3 |
| **Total** | **~200** | **15** | **~20** | **~165** |

**Directly attributed** means the kernel source contains an explicit comment
referencing the `llama.cpp` kernel it was ported from or modeled on (for
example, `dequant_batch_q4_k_blocked` references `kernel_mul_mm_q4_K_f32`).

**Structurally influenced** means the kernel uses the same tile geometry or
data layout strategy established by `llama.cpp`, but the implementation is
written independently for AX's buffer contracts and dispatch model.

**AX-original** means the kernel has no `llama.cpp` counterpart. This includes
all fused operator kernels (QKV+bias+QKnorm+RoPE+KV-append, residual+RMSNorm,
activation+gating), all attention kernels, all general matmul kernels, and
most elementwise kernels.

The convergence is concentrated in one area: **quantized weight
dequantization**, where the physics of Apple GPU memory bandwidth leaves little
room for alternative approaches. The differentiation is concentrated in two
areas: **fused operator kernels** that `llama.cpp`'s graph executor dispatches
as separate operations, and the **runtime orchestration layer** (execution
plans, smart barriers, model-aware scheduling) that decides how those kernels
are composed and submitted.

## Relationship with mistral.rs

Early in AX Engine's development, we studied
[mistral.rs](https://github.com/ericlbuehler/mistral.rs) as a reference for
Rust-based inference on Apple Silicon. Several initial design ideas — including
the use of Candle-style Metal kernels and the general approach to GGUF weight
loading in Rust — were informed by reading the mistral.rs codebase.

As AX Engine's architecture matured, we found that the two projects'
design goals diverged significantly:

- **Execution model**: mistral.rs follows Candle's op-at-a-time graph
  execution, dispatching each operation as an independent Metal kernel.
  AX Engine fuses multiple operations into single dispatches and uses
  execution plans to control the entire forward pass as a single command
  buffer.
- **Memory model**: mistral.rs allocates intermediate tensors through Candle's
  general-purpose allocator. AX Engine pre-allocates a fixed set of scratch
  buffers and reuses them across layers, eliminating per-dispatch allocation
  overhead.
- **Kernel ownership**: mistral.rs relies on Candle's Metal kernel library.
  AX Engine owns its entire shader set (181 kernel entry points), with
  architecture-specific fused kernels that have no Candle or mistral.rs
  counterpart.
- **Tuning surface**: mistral.rs uses Candle's fixed dispatch parameters.
  AX Engine uses JSON-driven kernel profiles with per-model heuristics,
  regime-aware decode routing, and runtime A/B testing infrastructure.

AX Engine has since fully decoupled from mistral.rs's design. No code is
shared between the two projects, and the runtime architectures are
fundamentally different. We acknowledge mistral.rs as an early reference
point and recommend it as an excellent project for users who need broader
model coverage or Candle ecosystem integration.

## Native Coverage and llama.cpp Routing

AX Engine is built by a small team. We cannot match the breadth of
`llama.cpp`, which supports hundreds of model architectures through a large
contributor community. Instead, AX focuses native support on a curated set
of model families where we can deliver measurable performance advantages
over general-purpose engines on Apple Silicon.

**Natively supported architectures** (full Metal GPU path):

| Architecture | Models | Forward pass |
|---|---|---|
| `llama` | LLaMA 3 / 3.1 (8B, 70B) | `LlamaForward` |
| `qwen2` / `qwen3` | Qwen 2/3 dense (8B, 14B, 32B) | `Qwen3Forward` |
| `qwen2moe` / `qwen3moe` | Qwen 2/3 MoE | `Qwen3MoeForward` |
| `qwen35` | Qwen 3.5 hybrid attention + SSM dense (9B, 27B) | `Qwen35Forward` |
| `qwen35moe` | Qwen 3.5 hybrid attention + SSM MoE (35B-A3B) | `Qwen35Forward` |
| `gemma` / `gemma2` / `gemma3` | Gemma 2/3 (4B, 12B, 27B) | `Gemma3Forward` |

Each native architecture has its own hand-written forward pass, fused Metal
kernels, and model-specific tuning profiles. Adding a new architecture
means implementing `ForwardPass`, writing any needed fused kernels, and
registering it in `arch_registry.rs` — not just wiring up a graph.

**For everything else**, AX routes to `llama.cpp` automatically. The SDK's
routing layer (`ax-engine-sdk/src/routing.rs`) inspects the GGUF file's
`general.architecture` metadata and quantization types at load time:

- If the architecture and quant types are natively supported, the model
  loads through AX's own Metal runtime.
- If not, the SDK spawns a `llama-server` subprocess and proxies requests
  to it over HTTP, providing the same `Model` / `Session` / `TextStream`
  API to the caller.

This means the CLI, server, and Python bindings all handle unsupported
models gracefully — the user gets `llama.cpp`-quality inference instead of
an error, while supported models get AX's native performance path. The
routing decision is logged at startup:

```
Routing: native | arch=qwen3 | source=global
Routing: llama_cpp | arch=mixtral | source=global | reason=architecture 'mixtral' is not natively supported
```

The routing preference can be controlled via `AX_ROUTING=auto` (default:
try native, fall back to llama.cpp), `AX_ROUTING=native` (fail if not
supported), or `AX_ROUTING=llama_cpp` (always use llama.cpp).

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

Apple M3 Max, April 2026. P=512 prefill, 128-token decode, f16 KV cache.
AX values are median of 5 iterations (2 warmup). llama.cpp values are
3-sample medians with 20s cooldown. AX% over 100% means AX was faster.

| Model | Quant | AX Prefill | AX Decode | llama Prefill | llama Decode | Prefill % | Decode % |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3 4B | Q6_K | **1,347** tok/s | **87.8** tok/s | 1,369 tok/s | 83.1 tok/s | 98% | **106%** |
| Qwen3.5 4B | Q8_0 | 1,057 tok/s | 50.9 tok/s | 1,340 tok/s | 56.5 tok/s | 79% | **90%** |
| Qwen3.5 9B | Q4_K_M | 585 tok/s | 48.6 tok/s | 733 tok/s | 49.0 tok/s | 80% | 99% |
| Gemma 3 12B | Q4_K_M | 432 tok/s | 44.6 tok/s | 495 tok/s | 40.2 tok/s | 87% | **111%** |
| Qwen3.5 27B | Q4_K_M | 191 tok/s | 17.4 tok/s | 209 tok/s | 17.6 tok/s | 91% | 99% |
| Qwen3 32B | Q4_K_M | 156 tok/s | 17.6 tok/s | 160 tok/s | 16.2 tok/s | 97% | **109%** |
| Qwen3.5 35B-A3B | Q4_K_M | 195 tok/s | 7.3 tok/s | 1,194 tok/s | 55.3 tok/s | 16% | 13% |

Qwen3.5 35B-A3B uses `mul_mat_id` with 64×32 blocked half-precision tiles
for unified MoE expert dispatch. Prefill improved 38× from the initial
baseline (5.1 → 195 tok/s) via batched-by-expert routing, single-CB
encoding, compact expert grid, and half-precision simdgroup compute.
Decode remains CPU-bound (MoE guard prevents GPU unified decode path).

AX decode exceeds llama.cpp on models where the fused decode path (concurrent
Metal dispatch, fused QKV+RoPE+KV-append, fused residual+RMSNorm) provides
structural advantages.

Prefill uses config-driven kernel selection across all supported quant types
(Q4_K, Q5_K, Q6_K, Q8_0) with f16-input full-tile kernels (64x64, 64x32,
tail, small-N variants), blocked layout, and pair (gate+up fused) batch
dispatch. GPU attention KV is f16 by default for all models.

See [BENCHMARKING.md](./BENCHMARKING.md) for methodology.

## Supported Models

All models must be in **GGUF format**. Recommended quantization: **Q4_K_M**. Also supported: Q5_K, Q6_K, Q8_0. Legacy formats (Q4_0, Q5_0) are not supported — use K-quant equivalents.

| Architecture | Models | Sizes |
|---|---|---|
| LLaMA 3 | Meta-Llama-3.1-8B-Instruct, Meta-Llama-3-70B-Instruct | 8B, 70B |
| Qwen 3 (dense) | Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen3-32B | 4B, 8B, 14B, 32B |
| Qwen 3 MoE | Qwen3-30B-A3B (experimental) | 30B-A3B |
| Qwen 3.5 (hybrid dense) | Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B | 4B, 9B, 27B |
| Qwen 3.5 MoE | Qwen3.5-35B-A3B | 35B-A3B |
| Gemma 3 | gemma-3-12b-it, gemma-3-27b-it | 12B, 27B |

Unsupported architectures automatically route to llama.cpp when
`AX_ROUTING=auto` (see [Native Coverage](#native-coverage-and-llamacpp-routing)).

**Memory requirements** (Q4_K_M, 4K context):

| Model Size | Total | Minimum Mac RAM |
|---|---:|---|
| 7-8B | ~5 GB | 16 GB |
| 12-14B | ~9 GB | 16 GB |
| 22-32B | ~20 GB | 32 GB |
| 70B | ~44 GB | 64 GB |

## Capabilities

**Inference**: Metal GPU prefill and decode, pipelined double-buffered decode, CPU and hybrid backends, speculative decoding with draft models.

**GPU kernels**: Blocked-layout batch matmul (Q4_K/Q5_K/Q6_K/Q8_0) with config-driven kernel selection (full-tile 64x64/64x32, tail, small-N, f16-input, pair, fused-SiLU variants), simdgroup-matrix Flash Attention (HD=64/128), split-K decode attention, fused QKV+bias+QKnorm+RoPE+KV-append, fused residual+RMSNorm, NR2/ILP4 decode matvec, gate+up pair kernel, concurrent dispatch with smart barriers.

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

# Benchmark (single engine)
./target/release/ax-engine-bench bench --model ./models/Qwen3-8B-Q4_K_M.gguf

# Benchmark (AX vs llama.cpp comparison)
./benchmarks/run_apple_to_apple.py --model ./models/Qwen3-8B-Q4_K_M.gguf

# Benchmark (AX only, fast iteration)
./benchmarks/run_apple_to_apple.py --model ./models/Qwen3-8B-Q4_K_M.gguf --ax-only

# Basic inference server
./target/release/ax-engine-server \
  --model ./models/Qwen3-8B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 3000
```

## Script Helpers

Frequently used benchmark helpers live under `./scripts`:

- `scripts/autotune.sh` and `scripts/autotune_quick.sh` for model-specific tuning.
- `scripts/qwen35_prefill_*_ab.sh` for Qwen3.5 prefill state-path sweeps.
- `scripts/bench_*` for quick cross-script comparisons (`_decode_barriers_ab.sh`, `_speculative_pairs.sh`, `cross_validate_defaults.sh`).
- Benchmark outputs are generated via `benchmarks/run_apple_to_apple.py`.

Legacy scripts kept for reference:

- `scripts/bench_prefill_v2_ab.sh` (legacy v2 prefill toggle matrix).
- `scripts/bench_llama_serial_median.sh` (legacy llama-bench serial median baseline; prefer `benchmarks/run_apple_to_apple.py --ax-only`).

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

**v3.0 onwards**: [Mozilla Public License 2.0 (MPL-2.0)](./LICENSE).
Copyright (c) 2026 [DEFAI Private Limited](https://defai.digital).

Starting with AX Engine v3.0, this project is licensed under the
MPL-2.0. This means you are free to use, modify, and distribute AX
Engine — including in proprietary products — but any modifications to
MPL-licensed source files must be shared under the same license. We
chose MPL-2.0 to keep AX Engine open and accessible while encouraging
improvements to flow back to the community.

Prior releases (v2.x and earlier) were licensed under MIT
