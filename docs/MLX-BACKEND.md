# ax-engine MLX Backend

## Current Direction

AX Engine v4 uses two user-facing inference paths: MLX mode for repo-owned MLX execution, and llama.cpp for non-MLX inference.
MLX mode is the repo-owned Mac-local inference path, implemented through
`ax-engine-mlx` and selected explicitly with `--mlx` or Python `mlx=True`.

Non-MLX inference bypasses AX-owned execution and routes to `llama.cpp`.
Direct MLX adapters, `mlx_lm` wrapper paths, `vLLM`, and `mistral.rs` are not
shipping peer inference routes.

## Backend Design

MLX mode uses a direct Rust ↔ MLX C++ integration via the official `mlx-c` C API.
This mirrors the SwiftLM lesson that high-throughput Mac inference needs direct
MLX tensor execution, explicit GPU queue control, and an AX-owned scheduler layer
for batching and prefix reuse rather than a delegated subprocess wrapper.

## Architecture

```
ax-engine-sdk (Rust)
  └── MlxRunner (ExecutionRunner)
        ├── ax-engine-mlx (Rust)
        │     ├── generate.rs  — chunked prefill (512 tok) + decode loop
        │     ├── model.rs     — Qwen3 transformer (attn + FFN + norm)
        │     ├── kv_cache.rs  — KV cache as MLX arrays
        │     └── weights.rs   — NativeTensorSpec → MlxArray loader
        └── mlx-sys (Rust FFI)
              ├── bindgen over /opt/homebrew/include/mlx/c/mlx.h
              └── safe wrappers: MlxArray, MlxStream, ops, fast, transforms
```

### Layer responsibilities

| Layer | What it does |
|---|---|
| `mlx-sys` | Unsafe FFI + safe RAII wrappers. No logic. |
| `ax-engine-mlx` | Model graph, inference loop, KV cache. All MLX ops live here. |
| `ax-engine-sdk` | Session routing: MLX mode → MlxRunner; non-MLX mode → llama.cpp. |

## Key design decisions learnt from SwiftLM

### Chunked prefill
Process prompt in 512-token windows (configurable). Between each chunk call
`mlx_async_eval` to drain the GPU command queue. Prevents Metal's 5-second watchdog
from firing on long prompts. Matches SwiftLM's default `--prefill-size 512`.

### Single GPU command buffer per layer
Fuse attention + FFN ops into one `async_eval` fence per transformer layer during
decode. Avoids 150+ round-trips between CPU and GPU per token (matches ax-engine's
recent `fuse FFN into single GPU command buffer` commit pattern).

### Native quantized matmul
`mlx_quantized_matmul` handles Q4_K_M / Q8_0 weights directly in the GPU kernel,
no dequantization to f16 needed. Maps directly to `NativeQuantizedTensorSource` in
the model manifest.

### Custom Metal kernel injection
Existing `phase1_dense_path.metal` kernels (paged attention, KV cache ops) are
registered via `mlx_fast_metal_kernel_new` and called inside the MLX compute graph.
This is the exact mechanism SwiftLM uses for its DFlash tape-replay kernels.

### KV cache as MLX arrays
Keys and values grow along the sequence dimension with `mlx_concatenate_axis`.
Prefix reuse: on a cache hit the existing KV arrays are reused for the shared prefix,
decode resumes from the split point. No block-table indirection needed at the MLX
layer (paged dispatch remains in the MLX Metal path for phase1 models).

## Implementation phases

### Phase 1 — mlx-sys crate (this PR)
- `bindgen` over mlx-c headers, linked against `/opt/homebrew/lib/libmlxc.dylib`
- Safe `MlxArray` (RAII, `Drop` calls `mlx_array_free`)
- Core ops: matmul, add, multiply, softmax, concat, reshape, transpose, astype, take
- Fast ops: rms_norm, rope, scaled_dot_product_attention
- Transforms: eval, async_eval
- IO: load_safetensors
- Metal: metal_kernel registration + call

### Phase 2 — ax-engine-mlx crate (this PR)
- Weight loader: reads `NativeTensorSpec` offsets from safetensors → `MlxArray`
- Quantized weight binding: Q4_K_M → `mlx_quantized_matmul`
- Qwen3 dense model graph: embed → 36 × (RMSNorm + GQA + SwiGLU) → RMSNorm → lm_head
- KV cache: grow-on-append, trim on prefix hit
- Chunked prefill loop
- Decode loop (single token, returns f32 logits slice)
- `MlxRunner` implementing `ExecutionRunner`

### Phase 3 — integration (this PR)
- Add both crates to workspace
- `SelectedBackend::Mlx` variant in `ax-engine-sdk`
- `--mlx` flag in `ax-engine-server`
- Python binding: `mlx=True` routes to MLX mode when model artifacts are configured

### Phase 4 — future
- Extend model coverage beyond Qwen3 dense (Llama, Mistral, MoE)
- TurboKV 3-bit KV cache compression (PolarQuant + QJL)
- Speculative decoding (DFlash-style draft model)
- Vision-language model support

## Success metrics

| Metric | Target |
|---|---|
| Streaming latency | First token < 200 ms for 512-token prompt |
| Throughput | ≥ mlx_lm baseline tok/s for Qwen3.5 9B Q4 |
| Delegated-wrapper elimination | Zero subprocess/HTTP overhead on MLX mode |
| logprob access | Returned on every decode step |
| Prefix reuse | KV cache reused across requests with shared prefix |
| Custom kernels | phase1 Metal kernels callable from MLX compute graph |

## File map

```
crates/mlx-sys/
  Cargo.toml
  build.rs
  src/lib.rs, array.rs, stream.rs, ops.rs, fast.rs, transforms.rs, io.rs, metal.rs

crates/ax-engine-mlx/
  Cargo.toml
  src/lib.rs, weights.rs, model.rs, kv_cache.rs, generate.rs, runner.rs, sampling.rs

docs/MLX-BACKEND.md  (this file)
```
