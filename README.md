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

If you need production HTTP serving, request orchestration, API adapters, or more advanced LLM serving features, use [ax-serving](https://github.com/defai-digital/ax-serving). Those concerns should sit above AX Engine.

## Why Use It

- **Local-first execution**: models run on your machine, not a remote API
- **Apple Silicon focus**: optimized for Mac GPU execution instead of portability-first tradeoffs
- **Explicit backend control**: you decide what runs on GPU vs CPU, with per-model KV budgets and graceful OOM handling (returns errors, never kills the process)
- **Multi-model safe**: each model gets its own Metal command queue, KV cache, and backend instance — designed for running multiple models in the same process without shared global state conflicts
- **Clean integration surface**: usable as a library, CLI, or drop-in `libllama` compatibility shim
- **Rust-native**: pure Rust workspace, no Python runtime, no C++ build chain, normal `cargo` workflows

## When To Choose AX Engine

Use AX Engine when your application needs one or more of these:

- **Multiple local models in one runtime**: for example, an embedding model, a chat model, and a coder model running under one process with isolated queues, KV caches, and backend instances
- **Explicit placement control**: for example, keep a latency-sensitive chat model on GPU while running a secondary model on CPU
- **A Rust-native inference layer**: you want to embed local model execution directly into a Rust system instead of orchestrating Python or external model services
- **Stable local Mac orchestration**: you care about predictable ownership of memory, queues, fallback behavior, and failure handling in a long-running desktop or local-server app
- **Focused Apple Silicon optimization**: you are willing to optimize for a narrower model set in exchange for tighter control over the hot path

You probably do not need AX Engine if your setup is simpler:

- a single local model at a time is enough
- a separate model process per task is acceptable
- broad model compatibility matters more than runtime control
- the simplest path to decent Apple Silicon performance matters more than in-process multi-model orchestration

## Engine Comparison

The table below is intentionally practical. It answers "which engine shape fits my app?" more than "which project is universally best?"

| Engine | Best When | Strengths | Tradeoffs | Typical Multi-Model Shape |
|---|---|---|---|---|
| **AX Engine** | You need a Rust-native local inference layer with explicit control over backend placement, memory behavior, and more than one model role in one app | In-process multi-model design, per-model queues/KV/backend instances, explicit GPU vs CPU placement, good fit for embedding/chat/coder combinations inside one runtime | Narrower model scope, more tuning work, not trying to be the broadest ecosystem | One process can own multiple models directly |
| **llama.cpp** | You want the broadest GGUF compatibility and a strong default single-model local server/toolchain | Huge model ecosystem, mature tooling, strong Apple Silicon support, easy single-model serving with concurrent requests | Main shape is one loaded model per server process; multi-model setups usually mean multiple processes/services | Usually one process per model, plus concurrent users on each model |
| **mlx-lm / MLX** | You want the fastest path to a solid Apple Silicon baseline with the MLX stack | Apple-focused stack, good default local performance, simple Python workflow, strong single-model experience | Python/MLX-oriented workflow, default server/tooling is centered on one model at a time, multi-model orchestration is usually custom code or multiple processes | Usually one process or custom orchestrator per model role |

Rule of thumb:

- choose **AX Engine** when your product needs one runtime to coordinate multiple model roles with explicit control
- choose **llama.cpp** when you want the most general GGUF path and can tolerate one model service per role
- choose **mlx-lm / MLX** when Apple Silicon is the whole target and a Python-first single-model workflow is acceptable

## Current Optimization Focus

AX Engine is currently most relevant for:

- Dense GGUF models in the 14B–70B range on Apple Silicon
- Model families with optimization depth: **Qwen 3 dense**, **LLaMA 3**, **Gemma 3**
- Workflows that benefit from explicit backend control and long-running local inference
- Mac-native deployments where stable, isolated execution matters more than broad platform coverage

## Performance

Performance depends heavily on model family, quantization, context length, kernel routing, and benchmark configuration.

AX Engine is optimized for Apple Silicon throughput and low-overhead local inference, but cross-engine claims should only be made from apples-to-apples runs. In particular, `llama.cpp` comparisons must record whether Flash Attention was enabled, because that materially changes the result on supported models.

For the current benchmarking methodology, command lines, and reporting rules, see [BENCHMARKING.md](./BENCHMARKING.md).

Current local snapshot, measured on March 25, 2026 on Apple M3 Max with Q4_K_M GGUFs, `512` prompt tokens, `128` decode tokens, AX deterministic `5` samples with `500ms` cooldown, and `llama.cpp` `llama-bench` with `-r 5 -fa 1`. Table values below use medians. `AX vs llama.cpp` over `100%` means AX was faster.

| Model | AX prefill | llama.cpp prefill | AX vs llama.cpp | AX decode | llama.cpp decode | AX vs llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 3 12B | 418.5 tok/s | 477.7 tok/s | 87.6% | 39.3 tok/s | 39.1 tok/s | 100.5% |
| Gemma 3 27B | 161.3 tok/s | 191.3 tok/s | 84.3% | 17.7 tok/s | 14.6 tok/s | 121.2% |
| Llama 3 8B | 642.0 tok/s | 771.4 tok/s | 83.2% | 58.1 tok/s | 64.8 tok/s | 89.7% |
| Llama 3 70B | 70.1 tok/s | 71.4 tok/s | 98.2% | 8.0 tok/s | 7.1 tok/s | 112.7% |
| Qwen3 8B | 631.4 tok/s | 664.8 tok/s | 95.0% | 55.1 tok/s | 59.8 tok/s | 92.1% |
| Qwen3 14B | 251.0 tok/s | 334.4 tok/s | 75.1% | 18.1 tok/s | 21.8 tok/s | 82.9% |
| Qwen3 32B | 126.3 tok/s | 129.4 tok/s | 97.6% | 13.1 tok/s | 12.0 tok/s | 109.2% |

For the 70B row, the model file is mixed-quant and contains an active `Q5_K` tensor. The published 70B result therefore records only the meaningful post-fix path with `AX_METAL_EXPERIMENTAL_Q5K_PREFILL=1`; the default shipped behavior still falls back on prefill and is documented as a caveat, not a headline comparison row.

Areas where we expect the most improvement:

- Prefill throughput on dense models
- Decode throughput on dense models where AX still trails on some model families
- Model-specific kernel fusions for architecture-specific operations

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

Mixed-quant caveat:

- some `Q4_K_M` or `Q5_K_M` GGUFs can contain active `Q5_K` layer tensors
- AX defaults `Q5_K` to decode-only baseline
- an experimental GPU prefill path exists behind `AX_METAL_EXPERIMENTAL_Q5K_PREFILL=1`
- under that experimental path, AX currently auto-selects the small-`N` `Q5_K`
  prefill route only when the mapped model is predominantly `Q5_K` and the
  prompt batch is small (`<= 32` tokens)
- validation-only override knobs also exist:
  - `AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT=base`
  - `AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT=small`
- when that happens, AX will tell you explicitly in runtime output, for example:
  - `PrefillPlan: mode=serial reason=unsupported_quant:blk.10.attn_v.weight:Q5K`
  - `Support: Q5_K support defaults to decode-only baseline...`
  - `PrefillPlan: mode=gpu_batch ... q5k_prefill=experimental`
  - `PrefillPlan: mode=gpu_batch ... q5k_prefill=experimental_small_n`

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

### Python Bindings

An experimental direct Python binding is available in
[`crates/ax-engine-py`](/Users/akiralam/code/ax-engine-v2/crates/ax-engine-py/README.md).
It builds a local `ax_engine` module on top of `ax-core`; it does not use
`ax-serving`.

Install it into a virtual environment with `maturin`:

```bash
python3 -m venv /tmp/ax-engine-py-venv
env -u CONDA_PREFIX \
  VIRTUAL_ENV=/tmp/ax-engine-py-venv \
  PATH=/tmp/ax-engine-py-venv/bin:$PATH \
  maturin develop --manifest-path crates/ax-engine-py/Cargo.toml
```

Basic usage:

```python
import ax_engine

model = ax_engine.Model.load("./models/Qwen3-8B-Q4_K_M.gguf", backend="auto")
session = model.session()
reply = session.generate("Explain KV cache briefly.", max_tokens=48, temperature=0.7)
print(reply)
session.close()
model.close()
```

Runnable examples:

- [`examples/python/basic.py`](/Users/akiralam/code/ax-engine-v2/examples/python/basic.py)
- [`examples/python/smoke.py`](/Users/akiralam/code/ax-engine-v2/examples/python/smoke.py)

Repeatable smoke test:

```bash
./scripts/test_ax_engine_py_smoke.sh
```

To include a tiny generation check:

```bash
AX_ENGINE_PY_GENERATE=1 ./scripts/test_ax_engine_py_smoke.sh
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

For a step-by-step setup guide, see [QUICKSTART.md](./QUICKSTART.md). For benchmark procedure and AX-vs-`llama.cpp` comparisons, see [BENCHMARKING.md](./BENCHMARKING.md).

If you need machine-readable benchmark artifacts, `ax-bench` now supports JSON
output for:

- `bench`
- `profile`
- `speculative`
- `soak`

Examples:

```bash
./target/release/ax-bench bench \
  --model ./models/<model>.gguf \
  --json \
  --json-output /tmp/ax-bench.json

./target/release/ax-bench profile \
  --model ./models/<model>.gguf \
  --json \
  --json-output /tmp/ax-profile.json

./target/release/ax-bench soak \
  --model ./models/<model>.gguf \
  --smoke \
  --json \
  --json-output /tmp/ax-soak.json
```

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
- Steady performance progress toward better AX-vs-`llama.cpp` parity across supported model families
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
