# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- Multi-model serving: `POST /v1/model/load` accepts `load_mode=add` to keep
  multiple models resident (scoped to Qwen 3.6 27B/35B and Gemma 4
  12B/26B/31B), with per-request `model` routing across the OpenAI, gRPC,
  Ollama, and Anthropic surfaces and `POST /v1/model/unload` to retire a
  retained model. Load/unload preflight runs synchronously before admission
  drain.
- Memory-aware load admission: loads whose projected peak resident set
  exceeds the Metal working-set budget are rejected with
  `422 insufficient_memory` before any drain. The estimate combines on-disk
  safetensors bytes with each model's worst-case KV pool derived from
  manifest attention geometry (sliding-window layers bounded at their ring
  window, hybrid linear-attention and KV-shared layers charge no per-token
  cache), so it scales with `--total-blocks` and with the number of resident
  models; `AX_SERVER_LOAD_MEMORY_PREFLIGHT=off` disables the check.
- `POST /v1/model/load` accepts `make_default` (default `true`;
  `load_mode=add` only) so a model can be added without changing what
  requests that omit `model` resolve to; load and unload responses report
  the resulting `default_model_id`. The Go and Swift typed clients and the
  JavaScript type declarations expose both fields (Ruby and JavaScript
  request bodies already pass through unknown fields).
- `/health` and `/v1/discovery` list every loaded model id (`models`)
  alongside the default `model_id` in multi-model serving.
- `response_format: json_schema` (non-streaming): OpenAI request shape
  accepted; output validated server-side against a documented schema subset
  (`502 invalid_output` on mismatch); schemas using keywords outside that
  subset are rejected up front with `400 unsupported_json_schema` rather than
  silently partially validated. Post-hoc validation, not constrained
  decoding. (Unenforceable *values* of supported keywords: see Fixed.)
- Streaming reasoning: native Qwen ChatML and Gemma 4 chat streams emit
  incremental `delta.reasoning_content` when the `reasoning` opt-in is set
  (previously rejected for all streaming requests).
- `usage.prompt_tokens_details.cached_tokens`: OpenAI responses report
  per-request prefix-cache reuse in the standard prompt-caching shape.
- Per-model `/metrics`: engine-step series now carry a `model` label (plus
  unlabeled aggregates), fixing last-writer-wins gauge interleaving under
  multi-model serving.
- `--model-idle-timeout-secs` / `AX_ENGINE_MODEL_IDLE_TIMEOUT_SECS`: opt-in
  idle eviction of non-default resident models for multi-model serving.
- MLX toolchain pinning: the admitted MLX version lives in `mlx.version`
  (repo root); `mlx-sys` now resolves the repo `.venv` even when it is not
  activated, refuses linking Homebrew's MLX formula (deployment-target
  truncation silently disables NAX kernels; `AX_MLX_ALLOW_HOMEBREW=1` for
  bring-up), and fails the build on version drift
  (`AX_MLX_VERSION_OVERRIDE=1` to experiment). Install scripts, the wheel
  build, and CI coverage install the pinned version, and
  `scripts/check-mlx-version.sh` verifies the toolchain (pin, wheel dylib,
  `LC_BUILD_VERSION` ≥ 26.2) without compiling.
- TUI Chat: markdown rendering for assistant replies (headings, bold/italic,
  inline + fenced code blocks, lists, blockquotes, links), with reasoning
  models' `<think>` blocks shown as a dimmed "Thinking" section.
- TUI Chat: live `~tok/s` + token estimate in the title while streaming and a
  per-reply `TTFT · elapsed · ~tokens · ~tok/s` line after each answer
  (client-side estimates; the SSE stream carries no usage chunk).
- TUI Chat: readline-style prompt history (↑/↓, draft stash/restore),
  bracketed paste into the composer, `Ctrl+Y` copy last reply, `Ctrl+R`
  regenerate, `Ctrl+L` clear transcript, and `/clear` `/copy` `/retry`
  `/help` slash commands. Composer column math is display-width aware
  (CJK/emoji).
- TUI: if the server process exits after binding, the Chat screen now drops
  the stale ready state and returns to the no-server card with a warning.

### Changed

- Client `stop` sequences are now enforced on the native MLX backend
  (previously rejected with `400 unsupported_parameter`): OpenAI semantics,
  server-side over decoded text, on chat, completions, Ollama, and Anthropic
  surfaces; streaming stops end the stream early and cancel the generation.
  The Anthropic surface reports the matched `stop_sequence`.
- Native Qwen ChatML and Gemma 4 chat streams with `tools` now emit
  incremental tool-call deltas (live content, one `delta.tool_calls`
  fragment per completed call with stream-wide 0-based `index`,
  `finish_reason:"tool_calls"`) instead of buffering the entire generation
  into a single chunk. GLM 4.x / GPT-OSS keep the buffered fallback.

### Fixed

- `response_format: json_schema` fail-closed on unenforceable *values* of
  supported keywords (for example string `minimum`, non-array `required`,
  draft-04 boolean `exclusiveMinimum`, non-string `type`), not only on
  unknown keywords — previously those schemas were accepted and the broken
  constraints were silently skipped (`400 unsupported_json_schema`).
- Native MLX streams cancel in-flight work when abandoned: Rust
  `GenerateStream` Drop / iterator error, and Python `stream_generate`
  iterator drop / mid-stream error, so a discarded stream cannot keep
  co-decoding or holding KV with later session calls.
- OpenAI native SSE detokenize failures stop the stream after the error and
  `[DONE]` frames (no further content or second terminal after a failed
  decode).

## [6.8.2] - 2026-07-09

### Added

- gRPC bearer-token authentication reusing `--api-key`.
- DiffusionGemma exponential temperature schedule and self-conditioning skip.
- Server: opt-in CLI flags (with env-var fallbacks) for concurrency, request
  body size, and request-timeout limits; a global request-rate limiter;
  idle-SSE and max-stream-duration deadlines; and gRPC request metrics on
  `/metrics`. All default to today's behavior when unset — see
  `docs/SERVER.md`'s "Resource Limits & Rate Limiting" section.
- `SECURITY.md`, `CODE_OF_CONDUCT.md`, and GitHub issue/PR templates.

### Changed

- OpenAI-compatible endpoints now reject unsupported non-default sampling
  params (`n`, `frequency_penalty`, `presence_penalty`, `logit_bias`) instead
  of silently ignoring them.
- Malformed `AX_NGRAM_CONFIDENCE_THRESHOLD`, `AX_NGRAM_SPECULATIVE_ACCEPT_THRESHOLD`,
  and `AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION` values now warn and fall back to
  defaults instead of panicking.
- pip is now the primary documented install path; PyPI metadata enriched and
  stale packaging artifacts removed.
- JavaScript SDK moved to `sdk/javascript`.
- `sdk/go` and `sdk/swift` now carry an explicit version marker, checked by
  CI's version-consistency gate alongside the other SDK/package manifests.
- The Mojo SDK is now labeled experimental in `docs/sdk/README.md` (a thin
  Python-interop shim with no test suite, not run in CI) rather than
  presented as a peer of the other client SDKs.

### Fixed

- GEGLU Metal kernel bit-exactness vs the imperative `gelu_approx` reference
  (restores per-step bf16/f16 rounding while keeping saturation clamps that
  prevent fast-math tanh NaN).
- Dense-FFN compile cache refresh no longer permanently disables the decode
  fast path.
- Unbounded scheduler retry recursion on KV-blocked batches.
- A poisoned mutex in the tokenizer cache or delegated-HTTP-agent cache no
  longer permanently cascade-fails subsequent requests; both now recover the
  last-known-good state instead of propagating the poison.

## [6.8.1] - 2026-07-08

### Fixed

- Metal runtime assets are now bundled in the PyPI wheel, and `doctor`
  accepts the bundled assets, so pip installs work without a local Metal
  toolchain (documented fallback for toolchain installs).

## [6.8.0] - 2026-07-07

### Added

- Linear-attention prefix snapshots captured at aligned prefill boundaries
  for Qwen 3.6 hybrids, mirrored to the disk prefix-cache tier.
- Largest-aligned-prefix snapshot store for sliding-window models.
- GPU top-p sampling; Gemma 4 assistant-MTP drafts are now verified.

### Changed

- OpenAI shim no longer exposes internal exception details in error responses.
- Benchmark publication is gated on recorded run conditions (load averages,
  stability summaries, condition metadata) with strict artifact validation.

### Fixed

- DiffusionGemma multi-block KV drop and self-conditioning dtype leak.
- Prefix-cache generation-counter eviction bug and unbounded mask cache growth.
- Qwen 3.6 decode compile, Qwen 3 embedding correctness, and Qwen 3.6
  think-token ids in MTP paths.
- n-gram-ON sessions no longer scrambled sliding KV for rollback-free
  requests (run()-latch bug found and fixed during ring rollout).

### Performance

- Bounded-rollback rotating KV rings extended to all serving classes on
  sliding-window models: n-gram-ON, sampled, and Gemma 4 assistant-MTP.
- Sliding-layer KV views trimmed on multi-token forwards (+23% Gemma E2B
  8k-token prefill).
- MTP verify-cache clone skipped on optimistic accept.
- Faster Qwen embedding ingest.

## [6.7.1] - 2026-07-04

### Added

- Batched dense decode plumbing wired end-to-end into the MLX runner
  (batched KV cache, attention mask, token assembly, ragged positions,
  continuous batched-decode session) with an E2E serving harness.
- Qwen dense FFN matvec fastpath and decode hot-path admission gate.
- Open-TQ-Metal K4/V4 TurboQuant support classifier.
- TUI usability: colors, breadcrumbs, validation, filtering; presets pass
  through to server launch.

### Fixed

- GLM MTP drafts are verified before accept.
- MTP runtime model resolution.
- Embedding post-processing deduplicated; template-injection hole closed.
- Hardened server and MTP routing paths.

## [6.7.0] - 2026-07-03

### Added

- Apples-to-apples Qwen 3.6 MTP peer benchmark vs MTPLX with degeneracy
  gate, MTP provenance, and fairness disclosures.
- Native runtime sharing and stream decoding in the Python runtime.

### Changed

- Optimistic MTP verify promoted to default-ON.
- Internal planning files and build artifacts removed from the repository.

### Fixed

- mlx-sys shim hardened: error-slot hygiene, closure `Sync` soundness, MLX
  version guard, RAII-guarded closure trampoline vectors, and fixes for UB
  and missing error handling in the C++ shim layer.
- Compiled MTP draft panic (token_offset deferred to the static RoPE branch).
- DiffusionGemma KV concat buffer output divergence.

### Performance

- DiffusionGemma denoise stops exactly at convergence (+5-10% first-block
  decode).
- lm_head projection skipped on non-final prefill chunks; KV cache arrays
  materialised alongside cache-only hidden eval.
- Dynamic-RoPE binding enables compiled-closure reuse across MTP decode steps.
- f32 cast folded into compiled embedding closures; faster embedding output
  construction.

## [6.6.0] - 2026-06-29

### Added

- GLM 4.7 Flash promoted to direct support, with native GLM 4.x tool calling
  and built-in MTP-head speculative decoding.
- GPT-OSS model family with per-head attention sinks.
- EmbeddingGemma-300m embedding support (Gemma3 bidirectional encoder) with
  batched-embedding profiler and fair benchmarks.
- Interactive model downloader and serve launcher (`ax-engine tui`,
  Textual-based) with live download progress.
- Qwen3.5-9B 4-bit downloadable preset and Qwen 3.6 27B server preset.

### Changed

- Gemma 4 MTP gate lowered to 0.85; n-gram stacking enabled by default.
- Unsupported MLX model families and delegated chat tool requests are
  rejected explicitly.
- Dropped Qwen-AgentWorld-35B-A3B support.

### Fixed

- Removed the 512-token OpenAI output cap that truncated chat/coding
  responses.
- Qwen MoE decode regression from an unguarded compile path.
- OpenAI tool-call parser ordering and invalid tool-name handling; bare
  Gemma tool calls and GLM tool calls with no arguments now parse.
- DiffusionGemma denoise cache alignment, restored self-conditioning, and
  per-request RNG seeding.
- mlx-sys closure-callback vector leaks and missing null-ctx guards.

### Performance

- MoE decode +40%: MLX buffer cache is no longer disabled by default.
- Gemma 4 and Qwen direct-mode decode optimizations (compile promotion,
  Metal kernel scaffolds); embedding-path packed projection and FFN compile
  optimizations.

## [6.5.2] - 2026-06-19

### Performance

- DiffusionGemma Phase 2 denoise optimizations, including skipping the
  self-conditioning matmul on converged steps; multi-block fix.

## [6.5.1] - 2026-06-18

### Changed

- DiffusionGemma GPU-sampling benchmark added; README benchmark section
  clarified.

## [6.5.0] - 2026-06-18

### Added

- DiffusionGemma direct decode support: manifest generation, Gemma4
  turn-based chat template routing, decode telemetry, and benchmark
  integration.
- Packed GEGLU Metal kernel for Gemma 4 MoE expert decode.

### Changed

- Qwen MTP gate lowered per workload with sticky auto-optimistic; adaptive
  MTP depth initialization for qwen3_next.

### Performance

- DiffusionGemma denoise optimizations: GPU matmul self-conditioning, cached
  embed table, argmax rejection; stochastic MTP draft fused into a
  single-eval lazy GPU graph.

---

Earlier history (v0.5 through v6.4.6) is tracked in git tags and commit
history.
