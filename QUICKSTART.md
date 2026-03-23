# AX Engine — Quick Start

Get from zero to running inference in under 5 minutes.

## 1. Prerequisites

- **macOS** on Apple Silicon **M3 or later** (M1/M2 not supported)
- **Xcode** installed (provides the Metal compiler)
- **Rust 1.88+** — install via [rustup](https://rustup.rs):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

## 2. Clone and Build

```bash
git clone https://github.com/defai-digital/ax-engine.git
cd ax-engine
cargo build --workspace --release
```

The release binary lands at `./target/aarch64-apple-darwin/release/ax-llama`.

## 3. Get a Model

Download any GGUF-format model. For example, LLaMA 3.1 8B Instruct (Q8_0):

```bash
mkdir -p models
# Place your .gguf file in ./models/
# e.g. models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
```

Supported architectures: **LLaMA/LLaMA-3**, **Qwen3**, **Gemma 3**.

## 4. Run Inference

### Single prompt

```bash
./target/aarch64-apple-darwin/release/ax-llama \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  -p "What is the capital of France?"
```

### Chat mode (applies model chat template)

```bash
./target/aarch64-apple-darwin/release/ax-llama \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  -p "Explain Rust ownership in simple terms" \
  --chat
```

### Interactive REPL

```bash
./target/aarch64-apple-darwin/release/ax-llama \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --interactive --chat
```

Type your messages at the prompt. Use `/reset` to clear context.

## 5. CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | (required) | Path to GGUF model file |
| `-p, --prompt` | | Input prompt text |
| `-n, --n-predict` | `-1` (infinite) | Max tokens to generate |
| `-c, --ctx-size` | `4096` | Context window size |
| `-t, --threads` | `0` (auto) | Number of CPU threads |
| `--temp` | `0.8` | Sampling temperature |
| `--top-k` | `40` | Top-K sampling (0 = disabled) |
| `--top-p` | `0.9` | Top-P nucleus sampling (1.0 = disabled) |
| `--seed` | `-1` (random) | Random seed for reproducibility |
| `--repeat-penalty` | `1.0` | Repetition penalty |
| `--chat` | off | Wrap prompt in model chat template |
| `--interactive` | off | Multi-turn REPL mode |
| `--verbose` | off | Print timing metrics |

## 6. Speculative Decoding

Use a small draft model to accelerate generation:

```bash
./target/aarch64-apple-darwin/release/ax-llama \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --speculative-draft ./models/small-draft-model.gguf \
  --speculative-k 4 \
  -p "Write a haiku about Metal shaders"
```

The draft model runs on CPU while the target model verifies on GPU. Mismatched tokens are rejected and the KV cache rolls back automatically.

## 7. Benchmarking

```bash
# Token throughput benchmark
./target/aarch64-apple-darwin/release/ax-bench bench \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf

# Per-layer profiling
./target/aarch64-apple-darwin/release/ax-bench profile \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf

# Stability soak test (short smoke run)
./target/aarch64-apple-darwin/release/ax-bench soak \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf --smoke
```

## 8. Backend Selection

AX Engine defaults to Metal GPU for both prefill and decode. Override with environment variables:

```bash
# Force CPU-only
AX_CPU_ONLY=1 ./target/aarch64-apple-darwin/release/ax-llama \
  -m ./models/model.gguf -p "Hello"

# GPU prefill + CPU decode (NEON-optimized)
AX_HYBRID_DECODE=cpu ./target/aarch64-apple-darwin/release/ax-llama \
  -m ./models/model.gguf -p "Hello"

# f16 KV cache (reduces memory, auto-enabled for long contexts)
AX_METAL_F16_KV_CACHE=on ./target/aarch64-apple-darwin/release/ax-llama \
  -m ./models/model.gguf -p "Hello"
```

## 9. C API (libllama.dylib)

The `ax-shim` crate builds a drop-in `libllama.dylib` compatible with llama.h:

```bash
cargo build -p ax-shim --release
# Output: target/aarch64-apple-darwin/release/libllama.dylib
```

Use it with any application expecting the llama.cpp C API.

## 10. Running Tests

```bash
cargo test --workspace             # all tests
cargo test -p ax-core test_name    # single test
cargo clippy --workspace --tests -- -D warnings  # lint
cargo fmt --all -- --check         # format check
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `compile_error!` on non-Apple platform | AX Engine only supports `aarch64-apple-darwin` (M3+) |
| Incorrect output or GPU errors on M1/M2 | M1 and M2 are not officially supported and may produce abnormal results — M3 or later required |
| Metal shader compilation fails | Ensure Xcode is installed: `xcode-select --install` |
| Out of memory on large models | Try `AX_METAL_F16_KV_CACHE=on` or reduce `-c` context size |
| Slow decode throughput | Ensure you're using `--release` build, not debug |
