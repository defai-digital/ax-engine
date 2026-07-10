# AX Engine Repo-Owned MLX Runtime

## Current Direction

AX Engine uses three labeled user-facing inference paths:

- `mlx`: repo-owned MLX execution through `ax-engine-mlx`, selected explicitly
  with `--mlx` or Python `mlx=True`
- `mlx_lm_delegated`: explicit upstream `mlx-lm` text compatibility through a
  user-provided `mlx_lm.server`
- `llama_cpp`: delegated GGUF/non-MLX inference

`mlx_lm_delegated` is a compatibility route, not the repo-owned MLX runtime. It
must stay outside repo-owned MLX performance claims. `vLLM` and `mistral.rs`
are not shipping peer inference routes.

## Backend Design

The repo-owned MLX runtime uses a direct Rust ↔ MLX C++ integration through the
repo-owned `ax_shim` C ABI over `libmlx`. This mirrors the SwiftLM lesson that high-throughput Mac
inference needs direct MLX tensor execution, explicit GPU queue control, and an
AX-owned scheduler layer for batching and prefix reuse rather than a delegated
subprocess wrapper.

## Architecture

```text
ax-engine-sdk (Rust)
  └── MlxRunner (ExecutionRunner)
        ├── ax-engine-mlx (Rust)
        │     ├── generate.rs    — chunked prefill (2048 tok) + decode loop
        │     ├── model.rs       — Qwen3 transformer (attn + FFN + norm)
        │     ├── kv_cache.rs    — chunked KV cache with slice_update growth
        │     ├── ngram_accel.rs — n-gram acceleration + EMA gating
        │     └── weights.rs     — NativeTensorSpec → MlxArray loader
        └── mlx-sys (Rust FFI)
              ├── bindgen over crates/mlx-sys/native/ax_shim.h
              └── safe wrappers: MlxArray, MlxStream, ops, fast, transforms
```

### Layer responsibilities

| Layer | What it does |
|---|---|
| `mlx-sys` | Unsafe FFI + safe RAII wrappers. No logic. |
| `ax-engine-mlx` | Model graph, inference loop, KV cache, n-gram acceleration. All MLX ops live here. |
| `ax-engine-sdk` | Session routing: `mlx` → MlxRunner; `mlx_lm_delegated` → `mlx_lm.server`; `llama_cpp` → llama.cpp. |

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

Process prompt in 2048-token windows by default (configurable). Between each
chunk, `mlx_async_eval` drains the GPU command queue without blocking the CPU.
The default matches `mlx_lm.generate`'s `prefill_step_size`; MLA warm-extend
requests use a smaller restore-safe chunk, and Qwen GatedDelta prefill clamps
to the long Metal specialization capacity.

### Chunked KV cache

Keys and values are stored in pre-allocated backing buffers sized to the next
256-token boundary.  New tokens are written with `mlx_slice_update` (no data
copy of existing entries).  When the buffer is full, a larger buffer is
allocated and the old data is copied once via `slice_update`.

This avoids the O(n) full-array `mlx_concatenate_axis` cost that the original
naive implementation paid on every append.  Draft rollback (`trim_to`) is
O(1) — only the logical sequence-length pointer changes; the backing buffer
retains its data.

After each decode step, all backing buffers are evaluated alongside the output
token (`mlx_eval([token, k0, v0, k1, v1, ...])`).  This materialises the
`slice_update` chain into a flat buffer, preventing computation-graph depth
from growing linearly with sequence length.  Mirrors mlx_lm's `mx.eval(y, cache)`.

### N-gram acceleration

Runtime acceleration with a bigram/trigram n-gram table (no second model
required).  Up to 4 draft tokens per step; verified in one causal forward pass
over `[last_token, D1, D2, …, D_n]`.  EMA accept-rate gating (α=0.1, threshold
0.5) disables n-gram probing for 8 steps after the EMA drops below threshold,
letting the n-gram table recover before re-enabling.

The intended high-value product case is coding-shaped generation with repeated
local structure: completions, edits, structured diffs, imports, indentation,
repeated identifiers, JSON/tool payloads, tests, and config files. Those outputs
often have local token regularity, so AX can draft short spans cheaply and rely
on the target model to verify them. This is opportunistic acceleration, not a
universal coding guarantee: novel code, high-entropy explanations, very short
answers, or low-acceptance outputs can fall back toward direct decode. Random-token
benchmarks are useful for repeatable comparison, but real coding workloads are
the user-facing case where repeated local structure should be expected most
often.

Dense/full-attention models use the chunked KV cache's O(1) `trim_to` rollback.
Linear-attention models such as Qwen3.5 use a rollback-safe branch path instead:
verification mutates a cloned cache, all-accepted drafts commit that branch, and
rejected drafts recompute `[last_token + accepted_drafts]` from the original
cache so recurrent state stays aligned with the logical sequence.

N-gram acceleration throughput claims must be reproduced through
`scripts/bench_mlx_inference_stack.py --ax-compare-policies` before they are used in
release notes or architecture decisions. Measured results on Gemma4-e2b-it-4bit
(Apple M5 Max, 128 GB, batch=1, 3 trials) are recorded in `README.md`:
1.83x mlx_lm at 128-token prompt and 1.89x at 512-token prompt.
Any prior unattributed `~1.96x mlx_lm` rows are investigation notes only; they
do not carry model, host, random-token prompt/decode shape, reference identity,
or AX decode mode provenance.

### Batch contract

The scheduler can submit multiple requests in one engine step. `MlxRunner`
contains an opt-in continuous dense-decode path behind
`AX_MLX_BATCHED_DECODE`; unsupported or uncertified rows remain on the per-item
path. Eligibility is resolved once at model load into typed structural
capabilities plus a separate numerical certification. Dense full-attention
Qwen 3 without MTP, diffusion, sliding attention, MoE, MLA, layer gating, or KV
compression is structurally supported, but it is not numerically certified:
real-weight sequential-oracle probes diverge for some prompt lengths because
MLX batched and single-row numerical paths can select different greedy tokens.
Production routing therefore fails closed even when `AX_MLX_BATCHED_DECODE=1`.
Diagnostics may additionally set
`AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED=1`; this override is not a production
setting. Promotion requires an artifact-, quantization-, MLX-version-, prompt-,
and fastpath-aware equivalence matrix plus KV/preemption and throughput gates.

### Custom Metal kernels

The `mlx-sys` crate exposes `metal.rs` — safe wrappers around `mlx_fast_metal_kernel_new`
and `mlx_fast_metal_kernel_call` for injecting custom Metal shaders into the MLX
compute graph.  The `phase1_dense_path.metal` kernel and registration code exist
in `metal/` and `crates/ax-engine-core/src/metal/`.

**These kernels are not active in the current MLX model execution path.**  The
model forward pass in `model.rs` uses only MLX fast ops (`rms_norm`, `rope`,
`scaled_dot_product_attention`, `quantized_matmul`).  Custom-kernel work requires
profiling evidence on the production decode path before a kernel can be selected
and wired in.  Open an issue with before/after benchmark rows on the canonical
scenario matrix before proposing any custom-kernel code change; maintainers may
invite a scoped PR after the evidence and validation gate are agreed.

## Runtime controls

| Flag | Default | Description |
|---|---|---|
| `--mlx` | false | Route to the repo-owned MLX runtime |
| `--mlx-model-artifacts-dir <path>` | — | Path to safetensors artifacts dir |
| `--disable-ngram-acceleration` | false | Disable n-gram acceleration for direct comparison runs |

Use `--disable-ngram-acceleration` when a direct same-policy comparison row is
required. AX Engine does not expose deprecated decode-mode aliases.

## Benchmarking

Repo-owned MLX runtime throughput is benchmarked through the MLX
inference-stack harness, not through `ax-engine-bench` scenario/replay manifests and not through
delegated llama.cpp manifests:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 256,512,2048 \
  --generation-tokens 128 \
  --ax-compare-policies
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

`ax-engine-bench doctor` also emits structured `performance_advice`. This advice
does not replace benchmark rows; it turns the current MLX contract into local
next steps, including that N-gram acceleration is enabled by default, batch=1 is
the supported MLX performance shape today, `mlx-swift-lm` is a baseline adapter
rather than a hybrid prefill/decode path, and model-specific quantization claims
must be checked with `--mlx-model-artifacts-dir`.

Interpretation rule:

- `bench_mlx_inference_stack.py` supports repo-owned MLX model-inference claims only
  when the required matching `mlx_lm.benchmark` baseline is present.
- `ax-engine-bench` supports workload-contract and regression claims.
- llama.cpp manifests support delegated non-MLX route-contract claims only.
- `mlx_lm_delegated` checks support delegated MLX text route-contract claims
  only; they are not repo-owned MLX throughput evidence.

## Implementation phases

### Phase 1 — mlx-sys crate

- `bindgen` over `crates/mlx-sys/native/ax_shim.h`, linked against Homebrew `libmlx`
- Safe `MlxArray` (RAII, `Drop` calls `mlx_array_free`)
- Core ops: matmul, add, multiply, softmax, reshape, transpose, astype, take,
  slice, slice_update, zeros, as_strided, repeat_axis, concatenate, argmax
- Fast ops: rms_norm, rope, scaled_dot_product_attention, dequantize, quantized_matmul
- Transforms: eval, async_eval, enable_compile, clear_cache, set_wired_limit
- IO: load_safetensors

### Phase 2 — ax-engine-mlx crate

- Weight loader: reads `NativeTensorSpec` offsets from safetensors → `MlxArray`
- Quantized weight binding: Q4_K_M → `mlx_quantized_matmul`
- Model graph: Qwen3 dense (GQA + SwiGLU), Qwen3.5 MoE (linear attention + MoE FFN + attn_output_gate), Gemma4 (per-layer embeddings, per-layer input gating, sliding-window + full attention, KV sharing, logit softcapping, sorted SwitchGLU expert gather for large MoE prefill batches)
- Chunked KV cache with slice_update growth and O(1) n-gram branch rollback
- N-gram acceleration with EMA gating
- Chunked prefill loop
- `MlxRunner` implementing `ExecutionRunner`

### Phase 3 — integration

- Both crates in workspace
- `SelectedBackend::Mlx` variant in `ax-engine-sdk`
- `SelectedBackend::MlxLmDelegated` for explicit upstream `mlx-lm` text
  compatibility through `mlx_lm.server`
- `--mlx` flag in `ax-engine-server`
- `--disable-ngram-acceleration` flag for direct comparison benchmarks
- Python binding: `mlx=True` routes to the repo-owned MLX runtime

### Phase 4 — future

- Prompt-prefix reuse (LRU cache for shared prefixes across requests)
- KV quantization and sliding-window cache layouts
- Custom Metal kernel integration (after profiling confirms a hot-path target)
- Multi-item batch execution with shared K/V primitives
- Expand model coverage to additional dense and MoE architectures

## MLX Stream / Thread Ownership Contract (I-3)

`mlx_stream` is the FFI handle for an MLX GPU compute stream. Upstream MLX
0.31 treats GPU streams as **thread-local** in two distinct
ways that AX Engine call sites must respect:

1. **One default stream per device per OS thread.**
   `mlx_default_gpu_stream_new` lazily creates the calling thread's
   default; another thread calling the same function gets its own,
   separate default. Two threads never share the same default by
   construction.

2. **Stream-index registration is thread-local at creation time.**
   `mlx_stream_new_device` registers the Metal command encoder for the
   stream index on the thread that called it. Passing the resulting
   handle to another thread and calling `mlx_set_default_stream` there
   does *not* register that index's encoder on the new thread.
   Subsequent MLX ops on the new thread fall back to that thread's own
   default — silently bypassing the dedicated stream's ordering
   guarantees.

`MlxStream` in `crates/mlx-sys/src/stream.rs` declares `Send + Sync`. The
type-level claim is correct in the narrow sense that the Rust struct is a
non-thread-affine pointer pair; **the thread affinity lives inside MLX,
not inside this struct.** Call sites that send a stream across threads are
responsible for ensuring the receiving thread has registered an encoder
for that stream's index.

### What `&MlxStream` does NOT promise across threads

- `set_as_default()` on thread B does not make a stream created on
  thread A use thread B's command encoder. Ops dispatched on thread B
  go through thread B's default.
- `mlx_eval`, `mlx_clear_cache`, and host buffer reads (`array.data()`
  and similar) do not take an explicit stream argument; they inherit the
  calling thread's current default. Mixing these calls across threads
  without explicit per-thread stream pinning produces undefined ordering
  relative to in-flight evaluation.

### Audit and probe coverage

- Module docs in `crates/mlx-sys/src/stream.rs` describe the contract
  inline so readers reaching the FFI boundary see the constraint before
  using `MlxStream`.
- A regression probe in `crates/mlx-sys/src/stream.rs` (test
  `cached_default_stream_wrappers_are_thread_local`) spawns a worker
  thread and asserts the default stream wrapper differs across OS threads,
  verifying the one-default-per-thread property holds against the live MLX
  runtime. The probe is self-contained and runs without an MLX model.
- **Call-site inventory.** The only runtime owner of a non-default stream
  is `MlxRunner`. It constructs exactly one stream at
  `crates/ax-engine-mlx/src/runner/mod.rs` (`MlxStream::new_gpu()` followed by
  `set_as_default()`) on the thread that builds the runner, stores it as a
  liveness-only `_stream` field, and never re-dispatches or sends it across
  threads. The server passes `EngineSessionConfig` into
  `NativeGenerationService`; the named generation worker constructs, executes,
  and drops the `EngineSession` on that same OS thread. Replacement-model cache
  clearing runs there before session construction. A service regression test
  asserts that construction and command execution observe the same thread ID.
  The only other spawned threads in `ax-engine-mlx` perform disk-prefix-cache
  I/O (`disk_prefix_cache.rs`). The handle therefore never legitimately crosses
  a thread at runtime today.
- **`!Send` blast radius.** With a single owner and a single construction
  site, a future `!Send`/owner-token migration touches just that one field
  and constructor. The cost is in re-proving the single-threaded execution
  assumption at the type level, not in chasing scattered call sites.

### Why no type-level migration yet

ADR-007 defers a `!Send` wrapper or owner-token migration to a follow-up
ADR. The current contract is documented and probed; a stricter type
model would close the gap structurally but at the cost of a workspace-
wide refactor. The audit captures the cost so a future ADR can argue
the tradeoff on AX-specific evidence rather than upstream comments.

## File map

```text
crates/mlx-sys/
  Cargo.toml
  build.rs
  src/lib.rs, array.rs, stream.rs, ops.rs, fast.rs, transforms.rs, io.rs, metal.rs

crates/ax-engine-mlx/
  Cargo.toml
  src/lib.rs, weights.rs, model.rs, kv_cache.rs, generate.rs, ngram_accel.rs, runner.rs

docs/MLX-BACKEND.md  (this file)
```
