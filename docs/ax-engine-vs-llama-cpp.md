# AX Engine vs llama.cpp

Research-level implementation comparison for users evaluating why AX Engine
exists and how it differs from `llama.cpp`.

---

## 1. Executive Summary

The short answer is:

- AX Engine is **not** a wrapper around `llama.cpp` on supported native paths.
- AX Engine **does** use `llama.cpp` as a compatibility fallback for
  unsupported models.
- The two projects are built around different implementation philosophies.

`llama.cpp` is a broad, portable GGUF inference platform built around a
general `ggml` graph and backend scheduler.

AX Engine is a narrower Apple-Silicon-native runtime built around:

- architecture-specific forward implementations
- transformer-specific fused execution paths
- Apple-UMA-first memory handling
- explicit support tiers and truthful fallback

That means AX should not be evaluated as "llama.cpp, but wrapped in Rust."
It should be evaluated as:

> a native-first Apple-Silicon runtime for a narrower supported model set,
> with `llama.cpp` used as a compatibility rail rather than the native engine.

---

## 2. Implementation Comparison at a Glance

| Dimension | AX Engine | llama.cpp |
|---|---|---|
| Core abstraction | architecture-specific `ForwardPass`, direct model-family code | `llm_graph_context` builds graphs, `ggml_backend_sched` schedules them |
| Native model support | explicit support matrix, narrower native allowlist | broad coverage across many architectures |
| Execution model | `ExecutionPlan`, `PrefillOp`, fused decode/prefill paths | `llama_batch` / `llama_ubatch` + graph reuse + backend scheduler |
| Backend strategy | Apple-Silicon-only Metal/CPU/hybrid runtime | cross-platform `ggml` backends: CPU, Metal, CUDA, Vulkan, SYCL, HIP, and more |
| How work reaches Mac GPU | model-aware runtime plan picks kernel path, dispatch config, and command-buffer shape before Metal submission | graph builder + backend scheduler decide backend splits and kernel submission shape before Metal execution |
| Hardware adaptation on Mac | queries Metal device/pipeline capabilities, then chooses fused paths, threadgroup sizes, barriers, and pipeline mode | queries Metal device/backend capabilities, then chooses backend buffers, graph splits, offload, and kernel threadgroup shapes |
| GPU core/thread control model | indirect: AX does not assign work to GPU cores manually; Metal schedules threadgroups on Apple GPU hardware | indirect: `llama.cpp` does not assign work to GPU cores manually; Metal schedules threadgroups on Apple GPU hardware |
| Memory and loading | GGUF mmap + no-copy Metal aliasing + pointer-keyed weight buffer cache | GGUF mmap/mlock + backend buffer mapping + layer/device placement |
| KV and session model | single-owner `ModelKv`, cleaner invariants, narrower session semantics | richer KV/session/state surface with seq ops, state I/O, unified KV modes |
| Server and integration | basic built-in server + SDK surfaces + `llama-server` fallback subprocess | mature `llama-server` with broader API and serving features |
| Routing philosophy | native-first, compatibility-backed | no fallback concept; it is itself the broad coverage engine |

This table is the clearest short version of the comparison.
The rest of this document expands each row and explains why the differences
matter.

---

## 3. How to Read This Comparison

The deepest difference is this:

- `llama.cpp` is primarily a **graph-and-scheduler inference system**
- AX Engine is primarily a **model-family-aware fused runtime**

That leads to different tradeoffs.

`llama.cpp` optimizes for:

- broad architecture coverage
- many hardware backends
- a shared execution substrate across model families
- mature serving and session behavior

AX Engine optimizes for:

- Apple Silicon only
- a narrower set of supported transformer families
- direct control of Metal execution structure
- direct exploitation of Apple UMA memory behavior

---

## 4. Core Runtime Architecture

This section expands the first four rows of the table:

- core abstraction
- native model support
- execution model
- backend strategy
- GPU execution path on Mac

### 4.1 AX Engine

AX Engine centers its runtime around architecture-specific forward paths.

The clearest evidence is in the `ForwardPass` trait and architecture registry:

- [`ForwardPass`](../crates/ax-engine-core/src/model/forward.rs)
- [`arch_registry.rs`](../crates/ax-engine-core/src/model/arch_registry.rs)

In AX:

- each supported architecture maps to an explicit forward implementation
- native support is a deliberate allowlist
- the runtime decides prefill and decode strategy through explicit execution
  plans
- unsupported models are not silently treated as "native enough"

This means AX is not trying to execute "any GGUF graph." It is trying to own
the fast path for a narrower set of families such as Llama, Qwen, and Gemma
variants that it supports natively.

### 4.2 llama.cpp

`llama.cpp` centers much more of its execution around `ggml` graph
construction and backend scheduling.

The key files are:

- [`src/llama-graph.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp)
- [`ggml/src/ggml-backend.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp)
- [`src/llama-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp)

In `llama.cpp`:

- inputs are packed into `llama_batch` and `llama_ubatch`
- a graph is built for the current batch
- `ggml_backend_sched` assigns nodes to backends, splits graphs, inserts
  copies, and executes the resulting backend splits

This architecture is more general. It is one reason `llama.cpp` supports so
many architectures and backends. It is also one reason AX can legitimately
claim a different design philosophy.

---

## 5. Execution Model Comparison

### 5.1 AX Engine: execution plans and fused regions

AX does not only "run Metal kernels." It pre-decides meaningful runtime
structure:

- decode mode selection
- barrier policy
- split vs fused QKV path
- prefill route
- attention route
- activation route
- residual handoff route

See:

- [`execution_plan.rs`](../crates/ax-engine-core/src/model/execution_plan.rs)
- [`decode.rs`](../crates/ax-engine-core/src/model/decode.rs)
- [`prefill_schedule.rs`](../crates/ax-engine-core/src/model/prefill_schedule.rs)

The prefill path is especially revealing. AX builds a flat pre-resolved op
schedule of `PrefillOp` entries rather than computing many decisions inline
during encoding. That is closer to a domain-specific command plan than a
generic graph executor.

This is why AX can say things like:

- "Qwen3 decode improved after the post-QKV fused path landed"
- "Llama 3 benefits more because fusion depth is higher"

Those are runtime-structure claims, not just benchmark slogans.

### 5.2 llama.cpp: graph reuse, batching, scheduler-driven execution

`llama.cpp` also does significant planning, but the planning unit is
different.

Key observations from the official source:

- `llama_batch_allocr` validates and reshapes incoming sequence batches
- `llama_ubatch` expresses batch topology and sequence relationships
- `llm_graph_context` builds the computation graph for embeddings, attention,
  FFN, MoE, and outputs
- `ggml_backend_sched` splits the graph across backend-support boundaries and
  manages copies and execution

This is a stronger fit for:

- broad model coverage
- multi-backend routing
- generalized graph reuse
- serving features built on a mature batch/session substrate

But it is less opinionated than AX about owning a narrow fused path for a
specific Apple runtime shape.

---

## 5.3 How both engines actually use the Mac GPU

One common misunderstanding is that a local inference engine directly manages
"GPU cores" or "GPU threads" in the same way a CPU runtime manages OS
threads.

That is not how either AX Engine or `llama.cpp` works on Apple Silicon.

The more accurate description is:

- the engine queries Metal device and pipeline capabilities
- the engine chooses kernels, threadgroup sizes, buffer strategy, and command
  shape
- Metal and the Apple GPU driver then schedule those threadgroups onto the
  physical GPU hardware

So the real control point is not:

- "how many GPU cores were detected, and how many threads did we manually pin?"

It is:

- "what work shape did we ask Metal to execute?"

### AX Engine on Mac GPU

AX reads device properties such as:

- `maxThreadsPerThreadgroup`
- `hasUnifiedMemory`
- `recommendedMaxWorkingSetSize`

and pipeline properties such as:

- `maxTotalThreadsPerThreadgroup`
- `threadExecutionWidth`

See:

- [`crates/ax-engine-metal/src/device.rs`](../crates/ax-engine-metal/src/device.rs)
- [`crates/ax-engine-metal/src/pipeline.rs`](../crates/ax-engine-metal/src/pipeline.rs)

AX then uses its own runtime planning layers to decide:

- which fused vs split path to run
- which decode mode to run
- what dispatch configuration to use
- whether to use sequential, single-command-buffer, concurrent, or pipelined
  submission

See:

- [`crates/ax-engine-core/src/model/execution_plan.rs`](../crates/ax-engine-core/src/model/execution_plan.rs)
- [`crates/ax-engine-metal/src/dispatch.rs`](../crates/ax-engine-metal/src/dispatch.rs)

That means AX's extra leverage is not that it can manually assign work to GPU
cores. Its leverage is that it controls more of the runtime shape before work
reaches Metal.

### llama.cpp on Mac GPU

`llama.cpp` also detects Metal device properties and backend capabilities.

From the official Metal backend source, it queries or uses:

- `MTLCopyAllDevices()` / `MTLCreateSystemDefaultDevice()`
- `supportsFamily(...)`
- `hasUnifiedMemory`
- `recommendedMaxWorkingSetSize`
- `threadExecutionWidth`
- `maxTotalThreadsPerThreadgroup`

See:

- [`ggml-metal-context.m`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-context.m)
- [`ggml-metal-device.m`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m)
- [`ggml-metal.metal`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal)

But `llama.cpp` reaches the GPU through a different control structure:

- `llama_batch` / `llama_ubatch`
- graph construction
- backend scheduling
- backend split execution

The graph/scheduler stack decides:

- what gets offloaded
- which backend owns which graph region
- where copies or split boundaries are needed
- what kernel submission shape is appropriate

After that, Metal schedules the actual GPU execution.

### Why this difference matters

This is the key distinction:

- AX controls more at the level of **model-aware runtime planning**
- `llama.cpp` controls more at the level of **graph and backend scheduling**

Both are adaptive.
Neither directly micromanages Apple GPU cores.

What changes between them is where the optimization intelligence lives.

---

## 6. Model Support and Fallback Philosophy

### 6.1 AX Engine

AX native support is intentionally explicit.

In practice that means:

- AX checks support before full native loading
- unsupported architectures are not treated as native
- unsupported quant families are surfaced clearly
- fallback can route to `llama.cpp`, but that route stays visible

See:

- [`arch_registry.rs`](../crates/ax-engine-core/src/model/arch_registry.rs)
- [`routing.rs`](../crates/ax-engine-sdk/src/routing.rs)

This is a narrower, more opinionated policy.

### 6.2 llama.cpp

`llama.cpp` is far broader by design.

The official project README presents it as a GGUF inference platform with wide
architecture coverage, wide backend coverage, and a large downstream ecosystem:

- [llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/README.md)

This difference matters because it explains why AX should not try to "win on
coverage." Coverage is `llama.cpp` territory, while AX has to justify itself
through stronger native specialization on a narrower set.

---

## 7. Memory Path and Model Loading

This section expands the table row for memory and loading.

### 7.1 What both systems do

Both AX and `llama.cpp` use mmap-backed GGUF loading.

That is important because people sometimes assume AX is unique simply because
it uses mmap. It is not.

### 7.2 AX Engine memory path

AX's model file layer is clearly designed around zero-copy GGUF access:

- [`mmap.rs`](../crates/ax-engine-core/src/gguf/mmap.rs)

The strategic step is what happens next in the Metal backend:

- AX attempts **no-copy Metal buffer aliasing** from mmap-backed model data
- if Metal accepts the alias, AX avoids an extra weight copy
- AX caches Metal weight buffers by mmap pointer identity

See:

- [`backend/metal.rs`](../crates/ax-engine-core/src/backend/metal.rs)

This is a major part of AX's Apple-UMA-first thesis. It means AX is trying to
optimize not only math kernels, but also:

- model-load overhead
- buffer ownership
- resource reuse
- long-lived resident-model behavior

### 7.3 llama.cpp memory path

`llama.cpp` also has a mature mmap and backend-buffer path:

- `use_mmap`
- `use_mlock`
- backend-specific buffer-type selection
- layer/device placement
- a `buffer_from_host_ptr` path where supported

See:

- [`src/llama-model.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)

So the honest conclusion is:

- AX has a very important memory-path direction
- but `llama.cpp` is not "memory naive"
- the real difference is that AX treats the Apple UMA path as a core product
  thesis, while `llama.cpp` must express it inside a more general multi-backend
  system

---

## 8. KV Cache and Session Semantics

This section expands the table row for KV and session model.

### 8.1 AX Engine

AX's current KV design is cleaner and narrower.

The key abstraction is:

- [`ModelKv`](../crates/ax-engine-core/src/kv/mod.rs)

AX moved to a single-owner KV design to avoid CPU/GPU split-brain errors.

Strengths:

- simpler invariants
- clearer ownership
- easier reasoning about native CPU vs GPU paths

Current limitations:

- GPU KV snapshot/restore is not yet supported
- long-lived session primitives are still earlier than `llama.cpp`

### 8.2 llama.cpp

`llama.cpp` has a much richer KV/session feature surface today:

- sequence remove/copy/keep/add/div
- shifting and reuse behavior
- state save/load
- unified/non-unified KV modes
- server slot integration

See:

- [`src/llama-kv-cache.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp)

This is one of the biggest areas where `llama.cpp` is currently more mature.

---

## 9. Tuning and Optimization Governance

This section expands the table row for execution policy and optimization
governance.

### 9.1 AX Engine

AX has explicit kernel-profile infrastructure:

- [`crates/ax-engine-metal/src/profile.rs`](../crates/ax-engine-metal/src/profile.rs)

It uses JSON-driven profile selection and explicit runtime heuristics for:

- matvec parameters
- prefill strategy
- decode attention thresholds
- fused and experimental path gates

This fits AX's philosophy:

- benchmark-gated
- model-aware
- Apple-focused
- increasingly regime-sensitive

### 9.2 llama.cpp

`llama.cpp` also contains substantial backend-level optimization logic, but it
is shaped by portability and backend-general abstractions.

That means:

- it can scale across more backends
- it can reuse more shared machinery
- but it is naturally more constrained in making engine-wide Apple-only
  decisions

This is the core strategic gap AX is trying to exploit.

---

## 10. Serving and Integration Surface

This section expands the table rows for server, integration, and routing.

### 10.1 AX Engine

AX now has several integration surfaces:

- Rust SDK
- Python binding
- JS SDK
- basic HTTP inference server

The built-in server is intentionally small:

- [`ax-engine-server/src/api.rs`](../crates/ax-engine-server/src/api.rs)

Unsupported model coverage is provided by spawning `llama-server` as a
subprocess:

- [`llama_cpp_process.rs`](../crates/ax-engine-sdk/src/llama_cpp_process.rs)

This is a compatibility strategy, not AX's native runtime identity.

### 10.2 llama.cpp

`llama.cpp` already has a mature serving surface:

- `/v1/completions`
- `/v1/chat/completions`
- `/v1/responses`
- slot save/load endpoints
- parallel serving controls

See:

- [`tools/server/server.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp)

This is another area where `llama.cpp` is clearly ahead in breadth and
maturity.

---

## 11. What AX Is Stronger At Today

AX is structurally stronger when all of the following are true:

- the model family is in AX's native support set
- the hot path is genuinely fused in AX
- Apple Silicon behavior matters more than cross-platform portability
- the runtime benefit comes from both execution path and memory path

That is why AX's strongest claims today are not:

- "we support more models"
- "we have a bigger server"
- "we replace llama.cpp everywhere"

They are closer to:

- "for a narrower native model set, we can own a more aggressive Apple path"
- "we can make execution-path and memory-path decisions a portable engine
  cannot prioritize as aggressively"

---

## 12. What llama.cpp Is Stronger At Today

`llama.cpp` is clearly stronger today in:

- model and architecture coverage
- server maturity
- batch/session/state management
- multi-backend maturity
- operational completeness for a wide range of deployments

Users should understand this clearly.

If the primary need is:

- broad GGUF compatibility
- a mature OpenAI-compatible server
- advanced session/state behavior
- hardware portability

then `llama.cpp` is usually the more natural default.

---

## 13. Why AX Still Has a Reason to Exist

AX has a reason to exist only if the following remain true:

1. Native AX support is not a wrapper path.
2. AX has measurable native advantage on its supported Apple-Silicon target
   set.
3. AX is more willing than `llama.cpp` to make Apple-only optimization
   decisions.

If those stop being true, then AX becomes difficult to justify.

If they remain true, then AX is not redundant with `llama.cpp`. It becomes a
specialized native engine with a different mission.

---

## 14. Practical Guidance for Users

Use AX Engine when:

- you are on Apple Silicon
- your model is in AX's native supported set
- you care about native Apple-specific execution behavior
- you want a runtime that is explicitly pursuing fused execution plus UMA-first
  memory handling

Use `llama.cpp` when:

- you need maximum GGUF coverage
- you need a mature, feature-rich local server today
- you need broader hardware/backend portability
- you need richer session/KV/state operations today

Use AX with fallback when:

- you want AX's native path where available
- but you do not want unsupported models to fail hard

That hybrid mode is a product convenience. It should not be confused with AX's
native identity.

---

## 15. Final Assessment

AX Engine and `llama.cpp` overlap, but they are not implementation-equivalent.

If a reader only remembers one thing from this document, it should be the
table in Section 2:

- `llama.cpp` is the broader graph-and-scheduler coverage engine
- AX Engine is the narrower fused native runtime for Apple Silicon

The fairest summary is:

- `llama.cpp` is the broader and more mature general GGUF inference platform
- AX Engine is the narrower and more opinionated Apple-native runtime

The most important technical difference is not just language choice
(`Rust` vs `C/C++`), but architectural intent:

- `llama.cpp` generalizes through graph and backend scheduling
- AX specializes through architecture-specific forward paths, fused execution,
  and UMA-aware memory strategy

That is the real reason AX can exist without being dismissed as a mere wrapper.

---

## 16. Evidence and Further Reading

### AX Engine Source References

- [`ForwardPass` trait](../crates/ax-engine-core/src/model/forward.rs)
- [`Architecture registry and support detection`](../crates/ax-engine-core/src/model/arch_registry.rs)
- [`GGUF mmap layer`](../crates/ax-engine-core/src/gguf/mmap.rs)
- [`Single-owner KV`](../crates/ax-engine-core/src/kv/mod.rs)
- [`Decode and prefill execution planning`](../crates/ax-engine-core/src/model/execution_plan.rs)
- [`Prefill graph-IR schedule`](../crates/ax-engine-core/src/model/prefill_schedule.rs)
- [`Metal backend and no-copy weight aliasing`](../crates/ax-engine-core/src/backend/metal.rs)
- [`Kernel profile system`](../crates/ax-engine-metal/src/profile.rs)
- [`Basic HTTP server`](../crates/ax-engine-server/src/api.rs)
- [`llama.cpp fallback subprocess integration`](../crates/ax-engine-sdk/src/llama_cpp_process.rs)

### llama.cpp Official References

- [llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/README.md)
- [llama-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp)
- [llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)
- [llama-kv-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp)
- [llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp)
- [ggml-backend.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp)
- [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp)
