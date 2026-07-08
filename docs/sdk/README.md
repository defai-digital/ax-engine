# SDKs

AX Engine SDKs are thin client or binding layers over the same runtime contract.
Use this folder as the entry point for SDK-specific setup, examples, and
integration notes.

Python is a first-class supported package and the primary pip-distributed
runtime surface. JavaScript and TypeScript are the HTTP/OpenAI/SSE client SDK
for applications that connect to a running AX Engine server.

| SDK | Package / path | Use when |
|---|---|---|
| [Rust](rust.md) | `crates/ax-engine-sdk` | You need the runtime-facing session, backend-resolution, and request/response contract used by the server and bindings |
| [Python](python.md) | `python/ax_engine` | You want the in-process Python session API, model downloads, or LangChain integration |
| [JavaScript / TypeScript](javascript.md) | `sdk/javascript` / `@ax-engine/sdk` | You want HTTP, OpenAI-compatible, SSE, or LangChain clients from JS/TS |
| [Go](go.md) | `sdk/go/axengine` | You want the stdlib-only Go HTTP client |
| [Ruby](ruby.md) | `sdk/ruby` / `ax-engine-sdk` | You want the stdlib Ruby HTTP client or langchain-rb integration |
| [Mojo](mojo.md) | `sdk/mojo/ax_engine.mojo` | You want Mojo bindings through Python interop |

Most SDKs talk to a running AX Engine server at `http://127.0.0.1:8080`.
Python can also run the SDK-owned in-process session directly.

Related docs:

- [Getting Started](../GETTING-STARTED.md)
- [Server](../SERVER.md)
- [OpenAI-compatible API contract](../API-COMPATIBILITY.md)
- [CLI](../CLI.md)
