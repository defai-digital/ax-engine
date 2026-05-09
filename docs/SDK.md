# SDK Design

`ax-engine-sdk` is the runtime-facing contract layer between callers (the HTTP
server, Python bindings, CLI tools) and the execution core. It owns backend
resolution, session lifecycle, and all public request/response types.

## Position in the Stack

```
ax-engine-server  ax-engine-py  ax-engine-bench
        \               |           /
         \              |          /
          ax-engine-sdk  ← you are here
          /                      \
ax-engine-mlx            ax-engine-core
```

`ax-engine-sdk` never contains GPU or model-forward-pass logic. Those live in
`ax-engine-mlx` (behind the `mlx-native` feature flag) and `ax-engine-core`.
The SDK's job is to decide which backend runs, build a session around it, and
translate caller requests into backend-appropriate calls.

---

## Backend Resolution

Before any inference can run, the SDK selects a backend. The selection is
explicit: the caller specifies a `PreviewBackendRequest`, and
`resolve_preview_backend()` validates it into a `PreviewBackendResolution`.

### Backends

| `SelectedBackend` | Description |
|---|---|
| `Mlx` | Repo-owned MLX-native path via `ax-engine-mlx` and `ax-engine-core` |
| `MlxLmDelegated` | Delegated HTTP to a running `mlx_lm.server` process |
| `LlamaCpp` | Delegated HTTP to a running llama.cpp server, or CLI subprocess |

### Support tiers

| `SupportTier` | Capabilities |
|---|---|
| `MlxCertified` | Full: generation, streaming, deterministic mode, prefix reuse, benchmarks |
| `MlxPreview` | Same as certified; long-context and benchmark metrics at preview level |
| `MlxLmDelegated` | Generation and streaming only; no determinism or prefix reuse |
| `LlamaCpp` | Generation and streaming only; no determinism or prefix reuse |
| `Unsupported` | Session creation fails |

### Resolution policy

`BackendPolicy.resolution_policy` controls which fallbacks are allowed:

| `ResolutionPolicy` | Meaning |
|---|---|
| `MlxOnly` | Only `SelectedBackend::Mlx` is accepted — llama.cpp and mlx_lm are rejected |
| `PreferMlx` | MLX when available, no automatic fallback (caller must configure explicitly) |
| `AllowMlxLmDelegated` | Permits `MlxLmDelegated` resolution |
| `AllowLlamaCpp` | Permits `LlamaCpp` resolution |

The default is `MlxOnly`. A `ResolvedBackend` with a non-MLX backend but an
`MlxOnly` policy fails `validate_against()` with a typed `BackendContractError`.

### Contract rules (validated by `ResolvedBackend::validate_against`)

- MLX backends must not carry a `fallback_reason`.
- Non-MLX backends must carry a non-empty `fallback_reason`.
- Each backend must match its expected tier: `Mlx` → `MlxCertified|MlxPreview`,
  `MlxLmDelegated` → `MlxLmDelegated`, `LlamaCpp` → `LlamaCpp`.

---

## Session Configuration

`EngineSessionConfig` is the single configuration struct passed to
`EngineSession::new()`. The defaults target the repo-owned MLX path:

```
kv_config:                KvManagerConfig { block_size=16, total_blocks=1024 }
deterministic:            true
max_batch_tokens:         2048
backend_policy:           MlxOnly
resolved_backend:         MlxPreview
mlx_disable_ngram_acceleration: false
mlx_kv_compression:       disabled
```

`PreviewSessionConfigRequest` is the high-level builder form used by the
server and Python bindings. `EngineSessionConfig::from_preview_request()` runs
backend resolution and produces a validated `EngineSessionConfig`.

`ResolvedSessionConfigRequest` is the serializable version used for persisting
resolved config across process boundaries (e.g. bench replay).

### Artifact discovery

Model weights (`mlx_model_artifacts_dir`) are found in priority order:
1. `with_mlx_model_artifacts_dir()` call on the config.
2. `AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR` environment variable.

Runtime Metal kernel artifacts (`mlx_runtime_artifacts_dir`) are found in priority order:
1. `AX_ENGINE_METAL_BUILD_DIR` environment variable.
2. Repo auto-detect: walks ancestor directories looking for
   `metal/phase1-kernels.json` + `build/metal/build_report.json` with a valid
   `MetalKernelAssets` contract.

---

## EngineSession: Stateful API

`EngineSession` is the primary stateful session. It wraps an `EngineCore`
(for the MLX path) or manages delegated backend connections.

### Creation

```rust
let config = EngineSessionConfig::from_preview_request(request)?;
let mut session = EngineSession::new(config)?;
```

`new()` validates the config, builds the `EngineCore` (MLX path), or
establishes backend connectivity (delegated paths).

### Convenience methods (one-shot)

```rust
// Blocking: submit + run to completion + return response
let response = session.generate(request)?;

// With an explicit request_id (must be non-zero)
let response = session.generate_with_request_id(42, request)?;
```

These are the simplest entry points. They submit one request, drive the step
loop until the request terminates, and return a `GenerateResponse`. Not
suitable for concurrent requests.

### Streaming (iterator-style)

```rust
let mut stream = session.stream_generate(request)?;
// GenerateStream implements an iterator over GenerateStreamEvent
while let Some(event) = stream.next()? {
    match event {
        GenerateStreamEvent::Request(r) => { /* initial metadata */ }
        GenerateStreamEvent::Step(s)    => { /* delta tokens each step */ }
        GenerateStreamEvent::Response(r) => { /* final response */ }
    }
}
let response = stream.into_response()?;
```

`GenerateStream<'_>` borrows the session. A `Request` event fires once at the
start. `Step` events fire once per engine step, carrying `delta_tokens` and
optional `delta_text`. A `Response` event fires when the request terminates
with the full `GenerateResponse`.

### Lifecycle API (concurrent requests, MLX only)

For running multiple requests concurrently on the MLX path:

```rust
// 1. Submit one or more requests
let id1 = session.submit_generate(request1)?;
let id2 = session.submit_generate(request2)?;

// 2. Drive the engine step by step
loop {
    let report = session.step_report()?;
    // inspect report.requests for per-request state
    if all_done { break; }
}

// 3. Attach a stream to an already-submitted request
let mut stream = session.stream_request(id1)?;

// 4. Cancel
session.cancel_request(id2)?;
```

`submit_generate` admits the request to the engine and returns its ID.
`step_report()` calls `EngineCore::step()` once and returns an `EngineStepReport`
covering all active requests. `stream_request()` attaches a `GenerateStream`
to an already-submitted request. Only the MLX path supports this pattern;
calling `step` or `stream_request` with a delegated backend returns
`EngineSessionError::LlamaCppDoesNotSupportLifecycle`.

### Embeddings (MLX only)

```rust
let embedding = session.embed(&token_ids, EmbeddingPooling::Mean, true)?;
let batch = session.embed_batch(&[tokens1, tokens2], EmbeddingPooling::Last, false)?;
```

`normalize=true` returns an L2-normalized unit vector suitable for cosine or
dot-product similarity. Delegated backends return
`EngineSessionError::EmbeddingNotSupported`.

---

## StatelessGenerateContext: Stateless API

`StatelessGenerateContext` is a lightweight wrapper for callers that need
per-request stateless invocations without sharing session state. Useful for
delegated backends in multi-threaded server contexts.

```rust
let ctx = StatelessGenerateContext::new(config)?;
let response = ctx.generate_with_request_id(id, request)?;
```

For the MLX path, `generate_with_request_id` internally creates a full
`EngineSession`, runs it, and drops it — no state is preserved between calls.
For delegated backends the context holds a reusable `RuntimeReport` and
dispatches directly to the backend.

Streaming via `StatelessGenerateContext` is only supported for delegated
backends (`supports_stateless_streaming()` returns false for MLX). The
`stream_state_with_request_id` + `next_stream_event` pair implements an
explicit pull-based iterator for callers (like the HTTP SSE adapter) that
drive their own event loop.

---

## Request and Response Types

### GenerateRequest

```rust
pub struct GenerateRequest {
    pub model_id: String,
    pub input_tokens: Vec<u32>,       // required for MLX path
    pub input_text: Option<String>,   // optional; delegated backends may use this
    pub max_output_tokens: u32,
    pub sampling: GenerateSampling,
    pub stop_sequences: Vec<String>,
    pub metadata: Option<String>,
}
```

The MLX path requires `input_tokens` to be non-empty. Delegated backends
accept `input_text` as an alternative when tokenization happens server-side.

`GenerateSampling` carries `temperature`, `top_p`, `seed`, and `repetition_penalty`.
When `session.config.deterministic = true`, `temperature` is forced to 0.0
and `seed` is ignored.

### GenerateResponse

```rust
pub struct GenerateResponse {
    pub request_id: u64,
    pub model_id: String,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub output_token_logprobs: Vec<Option<f32>>,
    pub status: GenerateStatus,
    pub finish_reason: Option<GenerateFinishReason>,
    pub step_count: u64,
    pub ttft_step: Option<u64>,   // step number of first output token
    pub route: GenerateRouteReport,
    pub runtime: RuntimeReport,
    // token counts for text-only delegated backends:
    pub prompt_token_count: Option<u32>,
    pub output_token_count: Option<u32>,
}
```

`ttft_step` records the engine step at which the first output token was
produced. For prefill-heavy requests this is greater than 1.

`route` contains the scheduler and KV routing decisions for the request.
`runtime` records which backend ran and its capability level.

### GenerateStreamEvent

Three event variants in order:

```
Request  → emitted once at stream start; carries initial SessionRequestReport + RuntimeReport
Step     → emitted once per engine step; carries delta_tokens and optional delta_text
Response → emitted once at termination; carries the complete GenerateResponse
```

The step loop may emit multiple Step events before any output token appears
(during chunked prefill). `delta_tokens` is empty for pure-prefill steps.

---

## GenerateRouteReport

`GenerateRouteReport` is the public view of `RouteMetadata`. It appears in
both `GenerateResponse.route` and `GenerateStreamStepEvent.step.route`.

```rust
pub struct GenerateRouteReport {
    pub execution_plan: Option<String>,
    pub attention_route: Option<String>,
    pub kv_mode: Option<String>,
    pub prefix_cache_path: Option<String>,
    pub barrier_mode: Option<String>,
    pub crossover_decisions: BTreeMap<String, u32>,
}
```

`crossover_decisions` holds all numeric telemetry values from the scheduler and
KV manager, keyed by the `ROUTE_DECISION_AX_*` constants from `ax-engine-core`.
Helper methods:

| Method | Key read |
|---|---|
| `kv_compression_active()` | `ax_mlx_kv_compression_status` |
| `kv_compression_preset_code()` | `ax_mlx_kv_compression_preset` |
| `kv_capacity_kib()` | `ax_mlx_kv_capacity_kib` |
| `linear_state_layers()` | `ax_mlx_kv_linear_state_layers` |

The full set of available keys is defined by the constant arrays in
`crates/ax-engine-core/src/scheduler.rs`:
`ROUTE_DECISION_AX_MLX_KV_KEYS`, `ROUTE_DECISION_AX_MLX_MODEL_KEYS`,
`ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEYS`,
`ROUTE_DECISION_AX_SCHEDULER_TOKEN_BUDGET_KEYS`.

---

## Error Types

| Type | Crate | When thrown |
|---|---|---|
| `EngineSessionError` | sdk | Session creation, request validation, step failure |
| `PreviewSessionConfigError` | sdk | Backend resolution failure during config build |
| `BackendContractError` | sdk | Policy/tier mismatch in `validate_against` |
| `LlamaCppBackendError` | sdk | llama.cpp HTTP or subprocess failure |
| `MlxLmBackendError` | sdk | mlx_lm HTTP delegation failure |
| `EngineCoreError` | core | Internal scheduler, KV, or runner error |

`EngineSessionError` wraps `EngineCoreError` via `From`. Callers do not need
to import `ax-engine-core` to handle session errors.

---

## Backend Capability Matrix

| Operation | Mlx | MlxLmDelegated | LlamaCpp |
|---|---|---|---|
| `generate` | Yes | Yes | Yes |
| `stream_generate` | Yes | Yes | Yes |
| `submit_generate` + `step_report` | Yes | No | Yes (no `step`) |
| `stream_request` (attach to submitted) | Yes | No | No |
| `embed` / `embed_batch` | Yes | No | No |
| Deterministic mode | Yes | No | No |
| Prefix reuse | Yes | No | No |
| Per-step delta tokens | Yes | Yes (stream) | Yes (stream) |

`StatelessGenerateContext::supports_stateless_streaming()` returns `true` for
`LlamaCpp` and `MlxLmDelegated`, `false` for `Mlx`.
