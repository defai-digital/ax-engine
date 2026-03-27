# AX Engine

A Rust inference engine for LLM workloads on Apple Silicon M3+, built around one idea: **domain-specific kernel fusion beats general-purpose graph execution for local inference**.

> Requires macOS on Apple Silicon M3 or newer, Xcode, and Rust 1.88+.

## Design Philosophy

Most inference engines are general tensor graph executors. They represent a forward pass as a DAG of individual operations (matmul, norm, add, RoPE, ...) and dispatch each as a separate GPU kernel. This is flexible but leaves performance on the table: every dispatch has overhead (pipeline binding, buffer binding, encoder calls), and adjacent operations that share data cannot reuse it in registers or threadgroup memory.

AX Engine takes a different approach. Because it only needs to run transformer inference (not arbitrary computation), it fuses operations across graph boundaries into domain-specific Metal kernels that a general executor cannot express.

The runtime also leans into Apple UMA directly: mmap-backed model weights are
now aliased into Metal buffers without an extra copy when Metal accepts the
alias, with a safe fallback to copied buffers when it does not.

The Metal API cleanup track is now fully shipped as well: command buffers
prefer unretained references, shared buffers opt out of automatic hazard
tracking, and the remaining single-vs-batch elementwise duplication has been
collapsed behind function-constant-specialized pipelines.

The next development phase follows that same idea more aggressively. The
remaining high-value wins are no longer generic matmul cleanups; they are
deeper decode fusions and prefill attention/runtime work.

The current FFN-side decode prototypes are both benchmark-gated:
- `AX_METAL_DECODE_PAIR_MATVEC=1` for gate+up pair matvec
- `AX_METAL_DECODE_FUSED_SILU_DOWN=1` for fused SiLU*Up + Down projection

Neither is shipped by default yet because the first real-model results
regressed decode on common `Q4_K`/`Q5_K` workloads.

The matmul-parity phase is therefore considered closed. Ongoing performance
work is now a fusion-architecture follow-on, not another round of isolated
kernel parity cleanup.

### Fusion Architecture

A standard transformer layer has ~15-20 logical operations. A general graph executor dispatches each as a separate Metal kernel. AX fuses them into fewer, larger kernels:

**Attention preparation** (4-9 ops fused into 1 dispatch):
```
QKV split + bias add + per-head QK norm + RoPE + KV cache append
```
A single kernel reads the fused QKV projection output, splits it, adds per-component bias (Qwen3), normalizes Q and K heads (Gemma3), applies rotary position embedding, and writes K/V directly to the KV cache. llama.cpp does each of these as a separate dispatch because its graph IR cannot express cross-op fusion.

**Residual + normalization** (2 ops fused into 1 dispatch):
```
hidden += projection_output; norm_out = RMSNorm(hidden)
```
The residual addition at the end of one sub-block is fused with the RMSNorm at the start of the next. The data is read once, updated in-place, and normalized in one pass.

**FFN gate+up pair kernel** (2 matmuls fused into 1 dispatch):
```
gate = W_gate @ x; up = W_up @ x  // same input, one B-tile load
```
Both projections share the same input vector; the pair kernel loads it once and computes two independent outputs, halving input bandwidth.

For a 32-layer model, these fusions reduce total Metal dispatches per forward pass by **~100-200 dispatches** compared to llama.cpp. At 10-15 microseconds per dispatch, this saves 1-3ms per token -- significant when decode tokens take 15-25ms total.

### Blocked-Layout Batch Matmul

Prefill throughput depends on the batch (multi-token) matmul kernel. AX uses the same blocked threadgroup layout as llama.cpp's `kernel_mul_mm`:

- Tile: BM=64, BN=32, BK=32, TG=128 (4 simdgroups)
- Blocked stride-8 layout: each `simdgroup_load` 8x8 fragment hits 1 cache line (vs 8 with row-major)
- Vectorized B-tile load: `float2x4 -> half2x4` single transaction per thread
- `BLOCKED_BC_OUT` function constant: compile-time elimination of boundary checks for full tiles
- Covered types: Q4_K, Q5_K, Q6_K, Q8_0, Q4_0

The inner loop achieves 1.33 MACs/load (8 multiply-accumulate ops per 6 simdgroup loads), matching llama.cpp's compute density.

### Per-Model Kernel Profiles

Dispatch parameters (threadgroup size, tile dimensions, kernel variants, attention strategy) are driven by JSON profiles loaded at model initialization. Each model family and size has its own profile:

```
perfs/qwen3-8b.json    perfs/gemma3-12b.json    perfs/llama3-70b.json
```

This enables per-model tuning without recompilation and drives the `ax-bench` auto-tuning workflow.

## Performance

AX Engine vs llama.cpp on Apple M3 Max (March 2026). Values over 100% mean AX was faster.

| Model | AX prefill | llama.cpp prefill | AX vs llama.cpp | AX decode | llama.cpp decode | AX vs llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | 675.8 tok/s | 639.2 tok/s | **105.7%** | 61.2 tok/s | 47.1 tok/s | **129.9%** |
| Llama 3 70B | 55.7 tok/s | 66.8 tok/s | 83.4% | 6.5 tok/s | 6.3 tok/s | **103.2%** |
| Qwen3 8B | 727.4 tok/s | 736.7 tok/s | 98.7% | 62.4 tok/s | 60.3 tok/s | **103.5%** |
| Qwen3 14B | 391.6 tok/s | 408.2 tok/s | 95.9% | 37.1 tok/s | 35.6 tok/s | **104.2%** |
| Qwen3 32B | 126.7 tok/s | 150.7 tok/s | 84.1% | 15.6 tok/s | 14.9 tok/s | **104.7%** |
| Gemma 3 12B | 420.5 tok/s | 463.3 tok/s | 90.8% | 39.6 tok/s | 35.8 tok/s | **110.5%** |
| Gemma 3 27B | 155.7 tok/s | 170.2 tok/s | 91.5% | 17.8 tok/s | 15.1 tok/s | **117.9%** |

LLaMA 3 benefits most from AX's fusion depth (10 dispatches/layer vs ~20 in a
per-op executor). Qwen3 decode now uses the shipped fused post-QKV path
(`qkv=fused`) on the common route, and `Qwen3 14B` is now back near parity on
both prefill and decode after restoring its shipped prefill route to
`f16_io=off`. Synthetic throughput benches now mask stop tokens during decode
measurement so long-context rows like `Qwen3 32B` do not collapse to `0 tok/s`
when EOS wins the first sampled step. The remaining FFN decode fusions are
still experimental: both the gate+up pair matvec and the fused SiLU+Down path
are opt-in only until they show a stable real-model decode win. See
[BENCHMARKING.md](./BENCHMARKING.md) for methodology.

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

**Integration**: Rust library (`ax-core`), llama.cpp-compatible CLI (`ax-llama`), drop-in `libllama` shim (`ax-shim`), experimental Python bindings (`ax-engine-py`), benchmarking tools (`ax-bench`).

## Quick Start

```bash
cargo build --workspace --release

# Single prompt
./target/release/ax-llama \
  --model ./models/Qwen3-8B-Q4_K_M.gguf \
  --chat --prompt "Explain speculative decoding."

# Interactive
./target/release/ax-llama \
  --model ./models/Qwen3-8B-Q4_K_M.gguf \
  --interactive --chat

# Benchmark
./target/release/ax-bench bench --model ./models/Qwen3-8B-Q4_K_M.gguf
```

See [QUICKSTART.md](./QUICKSTART.md) for setup details and [BENCHMARKING.md](./BENCHMARKING.md) for benchmark methodology.

## Workspace Layout

```
crates/ax-core    Core inference engine (backends, models, KV, sampling)
crates/ax-metal   Metal GPU backend (device, shaders, dispatch, profiles)
crates/ax-cli     ax-llama CLI
crates/ax-shim    llama.cpp-compatible C API shim (libllama.dylib)
crates/ax-bench   Benchmarking, profiling, and soak testing
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
