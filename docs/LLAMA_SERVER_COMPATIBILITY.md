# Llama-Server Compatibility Plan

This document defines how `ax-engine-server` should approach compatibility with
`llama-server` in a maintainable way.

## Goal

Make it cheap for existing `llama-server` users to switch to AX Engine for
single-node Apple-Silicon deployment, without turning AX Engine into a router,
cluster manager, or enterprise control plane.

## Compatibility Principles

1. Match the request and response contracts that downstream clients depend on.
2. Prefer explicit non-support over partial emulation that silently changes
   semantics.
3. Keep the runtime and the server thinly separated. Do not leak routing or
   fleet concerns into `ax-engine`.
4. Add automated compatibility checks for every newly supported endpoint.
5. When a feature is accepted for CLI compatibility but not truly implemented,
   warn clearly at startup.

## Scope

### In Scope for `ax-engine-server`

- single-model, single-node HTTP serving
- request/response compatibility for the most-used `llama-server` and
  OpenAI-compatible endpoints
- SSE streaming compatibility
- model info, health, tokenize, detokenize, template application
- slot inspection for the active single-node runtime
- metrics suitable for local observability

### Out of Scope for `ax-engine-server`

- router mode / multi-model orchestration
- multi-node request routing
- enterprise authz, tenancy, quotas, billing
- governed rollout or sovereign control-plane concerns

Those belong in `ax-serving`.

## Current Support Matrix

### Implemented

- `GET /health`
- `GET /healthz`
- `GET /v1/health`
- `GET /models`
- `GET /v1/models`
- `GET /props`
- `GET /slots`
- `GET /metrics` when `--metrics` is enabled
- `POST /completion`
- `POST /infill`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /tokenize`
- `POST /detokenize`
- `POST /apply-template`
- `POST /slots/{id_slot}?action=erase`
- `POST /slots/{id_slot}?action=save`
- `POST /slots/{id_slot}?action=restore`
- single-slot request fields `cache_prompt` and `id_slot`

### Accepted but Not Fully Implemented

- `--parallel`
- `--cont-batching`
- `--jinja`
- `--chat-template`
- `--chat-template-file`
- raw llama.cpp KV blob persistence format
- automatic multi-slot assignment heuristics

### Explicitly Not Implemented Yet

- embeddings
- reranking
- tool calling
- multimodal inputs
- continuous batching
- parallel multi-request decode

## Milestones

### Milestone 1: Contract Parity

Status: in progress, partially complete

Acceptance criteria:

- existing clients that use `health`, `models`, `completion`,
  `chat/completions`, `responses`, `tokenize`, `detokenize`, and
  `apply-template` can switch with no payload rewrites
- OAI-style error envelopes are returned consistently
- SSE streaming emits stable event shapes

### Milestone 2: Stateful Slot Parity

Status: in progress, partially complete

Acceptance criteria:

- `save`, `restore`, and `erase` operate on meaningful runtime state
- prompt-cache semantics are documented
- slot IDs remain stable for the process lifetime
- persisted snapshots are safe to load only under the configured save root

### Milestone 3: Throughput Parity

Acceptance criteria:

- `--parallel` supports real concurrent request processing
- `--cont-batching` performs actual request coalescing
- `/slots?fail_on_no_slot=1` reflects real busy state under load

### Milestone 4: Advanced Feature Parity

Acceptance criteria:

- tool calling / Jinja mode support
- embeddings and reranking support where the runtime exposes the required
  primitives
- richer `/v1/responses` features beyond the current text-first surface

## Validation Strategy

### Required Checks

- `cargo check -p ax-engine-server`
- `cargo test -p ax-engine-server`
- `crates/ax-engine-server/scripts/compat_smoke.sh <base-url> [model-id]`

### Manual Checks Before Declaring a Compatibility Improvement

- verify both JSON and SSE variants when an endpoint supports streaming
- verify `/infill` returns explicit `not_supported_error` on models without native
  FIM tokens
- verify both direct string prompts and token-array prompts where supported
- verify a repeated request preserves contract shape and reports `cache_n` when
  the backend can expose cached prompt reuse directly
- verify save → erase → restore works under `--slot-save-path`
- verify that unsupported features return explicit `501` / `not_supported_error`
  or `not_implemented_error`
- verify that docs and README statements match the code

## Engineering Rules

- Do not claim support for router mode from `ax-engine-server`
- Do not introduce fake slot save/restore semantics
- Do not silently alias unsupported advanced features to degraded behavior
- Keep slot persistence rooted under `--slot-save-path`; do not allow arbitrary
  filesystem writes from API input
- Keep compatibility tests close to the server crate
- Prefer additive compatibility shims over invasive runtime changes unless the
  runtime change clearly improves both correctness and maintainability
