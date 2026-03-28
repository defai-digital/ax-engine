# Docs

This folder contains focused documentation for integration surfaces that sit
above the core AX Engine runtime.

## Research

- [AX Engine vs llama.cpp](./ax-engine-vs-llama-cpp.md)

## Server

- [AX Engine Server Overview](./ax-engine-server.md)
- [AX Engine Server API](./ax-engine-server-api.md)
- [Inference Routing](./routing.md)

## SDKs

- [JavaScript SDK](./js-sdk.md)
- [Rust SDK](./rust-sdk.md)
- `ax-engine-py` is the Python binding built on top of the Rust SDK facade

## Scope

`ax-engine-server` is the basic HTTP inference surface that ships inside this
repository so other software can consume AX Engine without linking Rust
directly. It is intentionally smaller in scope than `ax-serving`, and now uses
the same `ax-engine-sdk` application-facing layer as the Python binding. For
Node.js and Next.js, the intended consumption path is the JavaScript SDK over
that HTTP surface.
