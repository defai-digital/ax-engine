# Local Engine Client Integration

Status: Active  
Scope: product integration contract for AX Studio, AX Code, and other first-party clients  
Last reviewed: 2026-07-14

AX Engine supports **two first-class client execution models**. Products must pick
one default and may optionally expose the other. They must **not** invent a third
wire protocol for chat.

## Dual client model

| Backend | Mechanism | Primary wire format | Default for |
|---|---|---|---|
| **In-process** | Link `ax-engine-sdk` (or language binding) and own `EngineSession` | Native SDK types; product may present OpenAI-shaped façades | AX Studio (`mlx` provider on macOS) |
| **Sidecar HTTP** | Run `ax-engine serve` / `ax-engine-server` as a local process | OpenAI-compatible HTTP `/v1/*` (SSE streaming) | AX Code (`ax-engine` provider) |

Optional gRPC (`--grpc-bind-address`) is an **adapter**, not the cross-product
baseline. Go / JS / Ruby / Swift SDKs that talk to a running server use HTTP
unless a product explicitly opts into gRPC.

### Non-goals

- Forcing every product onto the same process model
- Inserting an intermediate Go service between a Rust host and a Rust engine
- Making gRPC the only supported control plane
- Claiming full OpenAI/Ollama parity (see [API-COMPATIBILITY.md](./API-COMPATIBILITY.md))

## Shared lifecycle phases

All first-party clients should report engine readiness with this closed set of
**phases**. Product-specific detail lives in `blockers[]` / `reason` strings.

| Phase | Meaning |
|---|---|
| `unavailable` | Host/platform cannot run the backend (e.g. non–Apple Silicon for MLX) |
| `missing_dependency` | Binary, SDK, Metal toolchain, or other runtime dependency missing |
| `missing_model` | Eligible host, but model weights/artifacts are not prepared |
| `starting` | Spawning server, loading model, or warming session |
| `ready` | Can accept chat/generate work |
| `degraded` | Running, but a capability is limited (e.g. toolcall not advertised) |
| `error` | Hard failure after an attempt (crash loop, health failed, load failed) |

Mapping rules:

1. Prefer the **most severe** applicable phase: `error` > `unavailable` >
   `missing_dependency` > `missing_model` > `starting` > `degraded` > `ready`.
2. `ready` requires the backend-specific health signal:
   - In-process: session/model loaded (or load is known-instant and probe OK)
   - Sidecar: process alive **and** `/v1/models` (or product health probe) OK
3. Capability limits without total failure → `degraded`, not `error`.

## Wire contracts

### Chat / generate (sidecar)

Baseline: OpenAI-compatible routes documented in
[API-COMPATIBILITY.md](./API-COMPATIBILITY.md) and [SERVER.md](./SERVER.md).

Minimum product surface:

- `GET /v1/models` (or `/models` relative to a `/v1` base URL)
- `POST /v1/chat/completions` (streaming and non-streaming as product needs)

### Chat / generate (in-process)

Use `ax-engine-sdk` session APIs (`GenerateRequest` / stream events). Products
**may** re-encode results as OpenAI chat-completion JSON/SSE for UI SDKs, as long
as the OpenAI shape is treated as a façade over the native session.

### Control plane (product-owned)

Products own install/prepare/start/stop UX. Recommended operations:

| Operation | Sidecar | In-process |
|---|---|---|
| Probe host | binary version + platform checks | SDK host/Metal reports |
| Ensure runtime | install/locate binary, spawn `serve` | ensure worker thread / SDK init |
| Prepare model | download/cache artifacts | same cache/download ownership |
| Load | `POST …/model/load` or serve with model path | `EngineSession` / load command |
| Health | process + `/v1/models` | list loaded / probe |
| Stop | SIGTERM managed pid | unload / drop session |

## Product defaults (normative)

| Product | Default backend | Optional backend | Notes |
|---|---|---|---|
| **AX Studio** | `in_process` | `sidecar_http` (future/advanced) | Desktop Tauri host already links the SDK; HTTP base URL on `mlx` is compatibility-only today |
| **AX Code** | `sidecar_http` | — | Agent runtime manages `ax-engine` binary + `serve`; OpenAI-compatible client |
| **Other HTTP SDKs** | `sidecar_http` | — | Connect to a user- or product-managed server |

## Versioning

- Sidecar clients should pin a **minimum engine binary version** and fail closed
  with `missing_dependency` (or `error`) when below floor.
- In-process clients should pin SDK crates by **git rev / crate version** and
  document the validated engine commit in their manifest comments.
- Bumping either pin is a deliberate integration change, not a silent float.

## Implementation checklist for new clients

1. Choose default backend (`in_process` vs `sidecar_http`).
2. Implement phase mapping with the table above.
3. Use `/v1` for any out-of-process chat path.
4. Keep product-specific install UX outside ax-engine core crates.
5. Document the choice next to the provider id (`mlx`, `ax-engine`, etc.).
