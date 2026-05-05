# ax-engine MLX Backend

## Current Direction

AX Engine uses two user-facing inference paths: MLX mode for repo-owned MLX execution, and llama.cpp for non-MLX inference.
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
        │     ├── generate.rs    — chunked prefill (512 tok) + decode loop
        │     ├── model.rs       — Qwen3 transformer (attn + FFN + norm)
        │     ├── kv_cache.rs    — chunked KV cache with slice_update growth
        │     ├── speculative.rs — n-gram self-speculative decode + EMA gating
        │     └── weights.rs     — NativeTensorSpec → MlxArray loader
        └── mlx-sys (Rust FFI)
              ├── bindgen over /opt/homebrew/include/mlx/c/mlx.h
              └── safe wrappers: MlxArray, MlxStream, ops, fast, transforms
```

### Layer responsibilities

| Layer | What it does |
|---|---|
| `mlx-sys` | Unsafe FFI + safe RAII wrappers. No logic. |
| `ax-engine-mlx` | Model graph, inference loop, KV cache, speculative decode. All MLX ops live here. |
| `ax-engine-sdk` | Session routing: MLX mode → MlxRunner; non-MLX mode → llama.cpp. |

## Key design decisions

### Graph compilation

`mlx_enable_compile()` is called once at runner startup before warmup.  This
enables MLX's compute-graph caching — equivalent to `mx.compile()` in Python
mlx_lm — so Metal shader compilation and graph dispatch are reused across decode
steps with the same shape.  Without this, every token step pays a fresh graph
build cost (~10–15% throughput regression vs mlx_lm).

### Dedicated GPU stream

A new GPU stream is created and set as the thread default (mirrors mlx_lm's
`mx.new_stream(mx.default_device())`).  This avoids implicit cross-stream
synchronization on the shared default stream.

### Wired weight memory

`mlx_set_wired_limit(recommendedMaxWorkingSetSize)` is called at startup to
wire model weights into GPU memory, preventing Metal from paging them between
requests.

### Chunked prefill

Process prompt in 512-token windows (configurable).  Between each chunk,
`mlx_async_eval` drains the GPU command queue without blocking the CPU.
Prevents Metal's 5-second watchdog from firing on long prompts.  Matches
SwiftLM's default `--prefill-size 512`.

### Chunked KV cache

Keys and values are stored in pre-allocated backing buffers sized to the next
256-token boundary.  New tokens are written with `mlx_slice_update` (no data
copy of existing entries).  When the buffer is full, a larger buffer is
allocated and the old data is copied once via `slice_update`.

This avoids the O(n) full-array `mlx_concatenate_axis` cost that the original
naive implementation paid on every append.  Speculative rollback (`trim_to`) is
O(1) — only the logical sequence-length pointer changes; the backing buffer
retains its data.

After each decode step, all backing buffers are evaluated alongside the output
token (`mlx_eval([token, k0, v0, k1, v1, ...])`).  This materialises the
`slice_update` chain into a flat buffer, preventing computation-graph depth
from growing linearly with sequence length.  Mirrors mlx_lm's `mx.eval(y, cache)`.

### N-gram speculative decode

Self-speculative decoding with a bigram/trigram n-gram table (no second model
required).  Up to 4 draft tokens per step; verified in one causal forward pass
over `[last_token, D1, D2, …, D_n]`.  EMA accept-rate gating (α=0.1, threshold
0.5) disables speculation for 8 steps after the EMA drops below threshold,
letting the n-gram table recover before re-enabling.

Dense/full-attention models use the chunked KV cache's O(1) `trim_to` rollback.
Linear-attention models such as Qwen3.5 use a rollback-safe branch path instead:
verification mutates a cloned cache, all-accepted drafts commit that branch, and
rejected drafts recompute `[last_token + accepted_drafts]` from the original
cache so recurrent state stays aligned with the logical sequence.

Speculative throughput claims must be reproduced through
`scripts/bench_mlx_inference_stack.py --ax-both-modes` before they are used in
release notes or architecture decisions. Measured results on Gemma4-e2b-it-4bit
(Apple M5 Max, 128 GB, batch=1, 3 trials) are recorded in `README.md`:
1.83x mlx_lm at 128-token prompt and 1.89x at 512-token prompt.
Any prior unattributed `~1.96x mlx_lm` rows are investigation notes only; they
do not carry model, host, random-token prompt/decode shape, reference identity,
or AX decode mode provenance.

### Batch contract

MLX mode is single-request optimised for the current milestone.  `MlxRunner::run`
processes batch items serially; the states mutex is released before GPU work so
unrelated request-state access is never blocked by a long prefill.  True
multi-item MLX batching (shared K/V across requests) is deferred.

### Custom Metal kernels

The `mlx-sys` crate exposes `metal.rs` — safe wrappers around `mlx_fast_metal_kernel_new`
and `mlx_fast_metal_kernel_call` for injecting custom Metal shaders into the MLX
compute graph.  The `phase1_dense_path.metal` kernel and registration code exist
in `metal/` and `crates/ax-engine-core/src/metal/`.

**These kernels are not active in the current MLX model execution path.**  The
model forward pass in `model.rs` uses only MLX fast ops (`rms_norm`, `rope`,
`scaled_dot_product_attention`, `quantized_matmul`).  Custom-kernel work requires
profiling evidence on the production decode path before a kernel can be selected
and wired in.  Any custom-kernel PR must include before/after benchmark rows on
the canonical scenario matrix.

## Runtime controls

| Flag | Default | Description |
|---|---|---|
| `--mlx` | false | Route to MLX mode |
| `--mlx-model-artifacts-dir <path>` | — | Path to safetensors artifacts dir |
| `--no-speculative-decode` | false | Disable n-gram speculation (greedy baseline) |

`AX_NO_SPEC=1` environment variable also disables speculation (for debugging
without restarting with a flag).

## Benchmarking

MLX backend throughput is benchmarked through the MLX inference-stack harness,
not through `ax-engine-bench` scenario/replay manifests and not through
delegated llama.cpp manifests:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/Qwen3.5-9B-MLX-4bit \
  --prompt-tokens 256,512,2048 \
  --generation-tokens 128 \
  --ax-both-modes
```

The canonical reference is `mlx_lm.benchmark`, and the harness now treats it as
required. A run that cannot produce the matching `mlx_lm.benchmark` baseline
fails instead of emitting AX-only numbers. `mlx-swift-lm` numbers are valid only
when they come from a named `BenchmarkHelpers` / `MLXLMCommon` generation
adapter that reads the harness-emitted random-token prompt JSON. SwiftLM
application server measurements are retired and should not be used as a
baseline for this backend.

Use `ax-engine-bench scenario`, `replay`, `matrix`, `compare`, `matrix-compare`,
`baseline`, and `autotune` for workload-contract evidence: route identity,
determinism, prefix reuse, trace shape, regression comparison, trusted baseline
snapshots, and bounded manifest-knob exploration.

Interpretation rule:

- `bench_mlx_inference_stack.py` supports AX MLX model-inference claims only
  when the required matching `mlx_lm.benchmark` baseline is present.
- `ax-engine-bench` supports workload-contract and regression claims.
- llama.cpp manifests support delegated non-MLX route-contract claims only.

## Implementation phases

### Phase 1 — mlx-sys crate
- `bindgen` over mlx-c headers, linked against `/opt/homebrew/lib/libmlxc.dylib`
- Safe `MlxArray` (RAII, `Drop` calls `mlx_array_free`)
- Core ops: matmul, add, multiply, softmax, reshape, transpose, astype, take,
  slice, slice_update, zeros, as_strided, repeat_axis, concatenate, argmax
- Fast ops: rms_norm, rope, scaled_dot_product_attention, dequantize, quantized_matmul
- Transforms: eval, async_eval, enable_compile, clear_cache, set_wired_limit
- IO: load_safetensors

### Phase 2 — ax-engine-mlx crate
- Weight loader: reads `NativeTensorSpec` offsets from safetensors → `MlxArray`
- Quantized weight binding: Q4_K_M → `mlx_quantized_matmul`
- Model graph: Qwen3 dense (GQA + SwiGLU), Qwen3.5 MoE (linear attention + MoE FFN + attn_output_gate), Gemma4 (per-layer embeddings, per-layer input gating, sliding-window + full attention, KV sharing, logit softcapping)
- Chunked KV cache with slice_update growth and O(1) speculative rollback
- N-gram speculative decode with EMA gating
- Chunked prefill loop
- `MlxRunner` implementing `ExecutionRunner`

### Phase 3 — integration
- Both crates in workspace
- `SelectedBackend::Mlx` variant in `ax-engine-sdk`
- `--mlx` flag in `ax-engine-server`
- `--no-speculative-decode` flag for greedy baseline benchmarks
- Python binding: `mlx=True` routes to MLX mode

### Phase 4 — future
- Prompt-prefix reuse (LRU cache for shared prefixes across requests)
- KV quantization and sliding-window cache layouts
- Custom Metal kernel integration (after profiling confirms a hot-path target)
- Multi-item batch execution with shared K/V primitives
- Expand model coverage to additional dense and MoE architectures

## File map

```
crates/mlx-sys/
  Cargo.toml
  build.rs
  src/lib.rs, array.rs, stream.rs, ops.rs, fast.rs, transforms.rs, io.rs, metal.rs

crates/ax-engine-mlx/
  Cargo.toml
  src/lib.rs, weights.rs, model.rs, kv_cache.rs, generate.rs, speculative.rs, runner.rs

docs/MLX-BACKEND.md  (this file)
```
