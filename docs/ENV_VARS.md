# AX Engine Environment Variables and Advanced Usage

## Core Environment Variables

### Backend Control
- `AX_CPU_ONLY=1`: Force CPU-only backend (no Metal)
- `AX_HYBRID_DECODE=cpu`: Use CPU for decode stage (hybrid prefill+CPU decode)

### KV Cache
- `AX_METAL_F16_KV_CACHE=on|off|auto`: Control f16 KV cache (default: auto, uses f16 for seq_len >= 256)

### Metal Dispatch
- `AX_METAL_CONCURRENT_DECODE=1`: Enable concurrent decode (default on)
- `AX_METAL_BARRIERS=0`: Disable Metal buffer barriers
- `AX_METAL_RESIDENCY=1`: Keep weights resident in GPU memory (macOS 15+)

### Debugging
- `AX_DEBUG_LOGITS=1`: Print top-5 logits at each decode step
- `RUST_LOG=ax_core=debug,ax_cli=info`: Set tracing level

## Advantages of AX Engine

**1. Apple Silicon Native Optimization**
- Fused kernels reduce Metal dispatches (100-200 fewer per forward pass)
- UMA-aware memory path with mmap-backed no-copy Metal buffers
- Model-specific forward passes and execution plans

**2. Performance Edge on Supported Models**
- Better decode on several models vs llama.cpp (see README performance table)
- Specialized for Llama, Qwen3.5, Gemma families
- Pipelined double-buffered decode

**3. Clean Architecture**
- Single-owner KV cache (`ModelKv`)
- Explicit backend selection
- Rust SDK for easy integration

See `README.md` for benchmarks, `QUICKSTART.md` for basic usage, and `BEST-PRACTICES.md` for use cases and recommendations.

## Common Flags

See `./target/release/ax-engine --help` for full CLI.

Key flags:
- `--model <path>` - GGUF model
- `--chat` - Use chat template
- `--interactive` - REPL mode
- `--ctx-size N` - Context length
- `--temp`, `--top-p`, `--top-k` - Sampling
- `--speculative-draft <path>` - Speculative decoding (experimental)

## Troubleshooting

See QUICKSTART.md section 12.
