# W2 stream-contract source read (mlx-c 0.6.0 / MLX 0.31.2)

Date: 2026-05-14

This artifact closes the source-read pre-work for github issue #22 after the
W2.a `geglu` compile spike and W1.3 C5 `as_strided` fusion spike both hit MLX
runtime behavior that was not visible from the Rust wrappers.

## Source pins

| Component | Version / ref | Evidence |
| --- | --- | --- |
| Homebrew `mlx-c` | `0.6.0_2` | `brew info --json=v2 mlx-c` |
| Homebrew runtime dependency | `mlx 0.31.2` | `brew info --json=v2 mlx-c` |
| `mlx-c` source | `v0.6.0`, commit `0726ca922fc902c4c61ef9c27d94132be418e945` | `/tmp/ax-mlx-c-v0.6.0` |
| `mlx` source | `v0.31.2`, commit `68cf2fddd8de5edd8ab3d926391772b2e2cedad8` | `/tmp/ax-mlx-v0.31.2` |

## Findings

### F1. `mlx-c` compile/apply is a thin wrapper around MLX core

`mlx-c` does not add a separate stream abstraction for closures:

- `mlx/c/compile.cpp:11-20`: `mlx_compile` calls
  `mlx::core::compile(mlx_closure_get_(fun), shapeless)`.
- `mlx/c/closure.cpp:96-108`: `mlx_closure_apply` calls the stored C++ closure
  and returns `1` after `mlx_error(e.what())` if the closure throws.

So the stream behavior observed by Rust is the upstream MLX core stream
contract, not an extra mlx-c policy layer.

### F2. Default streams are thread-local

`mlx/stream.cpp:17-25` stores default streams in a `static thread_local`
container, with the upstream comment: "Each device has its own default stream in
each thread."

`mlx/stream.cpp:40-49` lazily creates a new default stream for the current
thread when no default exists. `mlx/stream.cpp:52-58` only assigns
`default_stream_storage(s.device) = s`; it does not register any Metal encoder
for an existing stream index on that thread.

This invalidates the older wording that `MlxStream::set_as_default()` is a
process-wide default.

### F3. Metal command encoders are also thread-local

`mlx/backend/metal/device.cpp:819-820` stores command encoders in a
`static thread_local std::unordered_map<int, CommandEncoder>`.

`mlx/backend/metal/eval.cpp:14-19` registers a GPU stream's command encoder only
when `gpu::new_stream(s)` runs. That happens from `mlx/stream.cpp:66-74` during
`new_stream`, on the thread creating the stream.

If evaluation later asks for a stream index that was not registered in the
current thread, `mlx/backend/metal/device.cpp:809-815` throws:

```text
There is no Stream(gpu, N) in current thread.
```

That is the exact failure class observed by the W2.a compiled `geglu` spike.

### F4. Compiled-function cache entries are thread-local and stream-sensitive

`mlx/compile.cpp:346-367` records the current `default_stream(default_device())`
inside compiler cache entries and skips cache entries whose stream does not
match the current default stream.

`mlx/compile.cpp:393-395` stores the compiler cache itself in a
`static thread_local CompilerCache`.

The supported shape is therefore to compile and apply on a thread whose MLX
default stream and Metal encoder registry are already coherent for that thread.

## Decision

Do not retry W2.a or C5 by "re-registering" a runner-created stream from
per-request tokio worker threads. With the public mlx-c 0.6 API, calling
`mlx_set_default_stream` on a worker thread can change that thread's default
stream pointer, but it does not register the matching Metal command encoder for
that existing stream index. The same failure should be expected.

The viable retry paths are:

1. Route compiled MLX closure work through a dedicated MLX executor thread that
   owns both compile and apply for the relevant stream.
2. Add or consume an upstream C API that exposes MLX's `ThreadLocalStream` path
   or otherwise registers an existing stream index's encoder on the current
   thread.
3. Plumb explicit `MlxStream` values into closure bodies only as part of a
   design that also guarantees the apply thread has a registered encoder for the
   same stream. Passing `Some(&stream)` alone is not enough if the apply thread
   never registered that stream index.

Until one of those exists, the production hot path should remain imperative and
the W2.a/C5 helpers should stay as spike artifacts plus regression tests.
