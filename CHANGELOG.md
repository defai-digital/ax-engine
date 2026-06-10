# Changelog

All notable changes to AX Engine are documented here. This project follows
[Semantic Versioning](https://semver.org/). Releases prior to `v6.0.0` are
tracked via Git tags and GitHub Releases.

## [Unreleased]

### Added

- **Inline MP4/WebM video on chat (ffmpeg)** — `POST /v1/chat/completions`
  `video_url` content parts now accept inline base64 MP4/WebM in addition to
  animated GIF when `ffmpeg` is available on the server `PATH`. Containers
  are routed by magic bytes; ffmpeg pipes PNG frames with `showinfo`
  timestamps keyed by frame number. Extraction is resource-bounded: frames
  are downscaled to at most 1600 px on the longest side, the decoded stream
  is capped at 512 MiB, and only the ≤ 32 uniformly sampled frames are
  PNG-decoded. Without `ffmpeg`, MP4/WebM still report a clear unsupported
  error and `/v1/generate` keeps accepting pre-extracted frame tensors.

### Fixed

- **Anthropic Messages no-op feature payloads** — `tools: []`,
  `tool_choice: {"type": "none"}` / `{"type": "auto"}`, and
  `thinking: {"type": "disabled"}` are valid Anthropic payloads that use no
  unsupported feature and are now accepted instead of being rejected with
  400 (`tools: []` regressed in v6.2.5; the others were rejected since the
  endpoint was introduced). Non-empty `tools`, enabled `thinking`, and
  malformed values are still rejected.
- **Non-positive or subnormal `image_std` rejected at config load** — image
  normalization validation now rejects negative, zero, and subnormal-for-f32
  std channels (previously only zero or non-finite) in both the server and
  the Python SDK; a subnormal value such as `1e-40` passed a naive `> 0`
  check but still overflowed the per-pixel division to inf. The per-pixel
  epsilon clamps — which silently rewrote a bad std to ~1.2e-7 (Rust) or
  1e-12 (Python) — are removed in favor of the load-time guarantee.
- **Gemma 4 resize fallback divisibility (hardening, no output change)** —
  the single-axis resize fallback floors the aspect ratio again, keeping the
  fallback dimension a multiple of `patch_size * pooling_kernel_size` by
  construction rather than only via the `max_side` clamp (which still binds
  for every reachable input today, so resize outputs are unchanged). The
  Python parity test now pins the same vectors and the divisibility contract
  as the Rust test.

## [6.2.5] - 2026-06-10

Adds API key authentication, a Prometheus metrics endpoint, agentic
response contracts (logprobs, reasoning, tool calls, JSON validation), a
CLI model manager, and hardens multimodal serving against misconfigured
checkpoints.

### Added

- **API key authentication** — `--api-key` flag or `AX_ENGINE_API_KEY`
  environment variable requires `Authorization: Bearer <key>` on all
  `/v1/*` routes. `/health` and `/healthz` stay unauthenticated for
  readiness probes. Token comparison uses constant-time equality to avoid
  timing side channels.
- **Prometheus metrics** (`GET /metrics`) — HTTP request counters (total,
  in-flight, 2xx/4xx/5xx) and engine-step gauges (scheduled requests and
  tokens, KV block usage, prefix-cache hits). Engine-step gauges are
  read-only snapshots from real generation steps; the scrape path never
  advances the engine. Requires the API key when auth is enabled.
- **Logprobs** (completions and chat) — when the engine observed
  sampled-token logprobs, non-streaming responses carry them in OpenAI-shaped
  `logprobs` blocks. Logprobs are all-or-nothing: partially observed values
  are omitted entirely to keep arrays aligned. Streaming logprobs are
  rejected with `400 unsupported_parameter` for now.
- **Reasoning output** (chat) — opt-in via the `reasoning` field. Known
  model-family thinking patterns are split into
  `message.reasoning_content`: Qwen ` THOUGHT…` text markers and Gemma 4
  thinking channels (extracted token-level during native decode). Unknown
  formats are left in `content` untouched.
- **Tool call extraction** (chat) — experimental. When `tools` are present,
  explicit `ARGS…` spans in model output are parsed into
  `message.tool_calls`. Bare JSON is never reinterpreted as a tool call.
  `/v1/models` continues to report `openai_tool_calling_supported: false`
  until streaming deltas and continuation handling land.
- **JSON object validation** (`response_format: json_object`) — non-streaming
  responses are validated server-side; output that is not a JSON object
  returns `502 invalid_output`. This is post-hoc validation, not constrained
  decoding.
- **CLI model manager** — `ax-engine models list`, `info`, and `rm`
  subcommands for inspecting and cleaning up downloaded model artifacts.

### Fixed

- **Image normalization division by zero** — config loading now rejects a
  zero or non-finite `image_std` channel when `do_normalize` is set (server
  returns `MediaError::Config`, Python SDK raises `ValueError`) instead of
  producing inf/NaN pixels.
- **Anthropic request validation** — `json_value_is_present` now checks for
  presence (non-null) rather than truthiness, so `thinking: false` or
  `tools: 0` correctly trigger the unsupported-feature rejection.
- **Resize extreme aspect ratios** — single-axis fallback in `resize_target`
  now uses `.max(unit)` to guarantee at least one patch unit per dimension,
  preventing zero-dimension targets on extreme aspect ratios (e.g., 1×10000).

## [6.2.4] - 2026-06-10

Adds an Anthropic-compatible Messages endpoint, accepts MP3 audio in
multimodal chat, and hardens Gemma 4 input validation.

### Added

- **Anthropic Messages endpoint** (`POST /v1/messages`) — translates
  Anthropic-style `system`, `messages`, `max_tokens`, `temperature`,
  `top_p`, `top_k`, and `stop_sequences` into the internal OpenAI chat
  pipeline. Content blocks, `model` validation, and usage tracking are
  supported. Streaming, tool use, and extended thinking are rejected with
  clear error messages. Works with native MLX, MLX-LM delegated, and
  llama.cpp delegated backends.
- **MP3 audio in multimodal chat** — `/v1/chat/completions` now accepts MP3
  inline audio in addition to WAV. The container is sniffed from magic bytes
  (RIFF → hound WAV decoder, ID3/MPEG sync → symphonia MP3 decoder). MP3
  decoding stops at the model's `audio_seq_length` frame cap to cap memory
  use. AAC/OGG/FLAC remain unsupported; send pre-computed tensors via
  `/v1/generate` for those. The Python SDK preprocessing helper stays WAV-only.

### Changed

- **Gemma 4 video timestamp validation** — `video_timestamp_token_ids` now
  validates that every entry is a non-negative integer (rejects booleans,
  floats, and negative values) with per-element error messages identifying
  the exact video, frame, and index.
- **Serving contract documentation** — `docs/SERVER.md` updated to reflect
  WAV/MP3 audio support, magic-byte sniffing, decode cap, and fixed
  soft-token budgets.

## [6.2.2] - 2026-06-09

Patch release that fixes a critical multimodal attention bug in Gemma 4
where vision tokens lost intra-image bidirectionality on full-attention
layers, improves OpenAI API output postprocessing, and adds idempotent
PyPI publish.

### Fixed
- **Gemma 4 media block overlay** — multimodal PrefixLM mask was previously
  applied only to sliding-window layers, leaving full-attention layers with
  a plain causal mask. Vision tokens larger than the sliding window were
  silently dropped, losing intra-image bidirectionality on every global
  layer. Now the bidirectional vision-block overlay is applied to both
  full-attention and sliding-window layers, matching the reference
  implementation. Memoized per unique window size for efficiency.
- **Gemma 4 channel output markers** — thinking-channel framing stripped
  from chat responses to prevent leaking internal model markers.
- **OpenAI unusual prompt output postprocessing** — handles edge cases
  where model output contains unexpected prompt echoes or structural anomalies.
- **OpenAI response formatting** — standardized postprocessing for
  consistent API output.

### Added
- **Tokenizer token lookup** — exposed for debugging and inspection of
  tokenized inputs.
- **Idempotent PyPI release publish** — re-publishing the same version
  no longer fails, enabling safe retry of interrupted releases.

## [6.2.1] - 2026-06-09

Patch release focused on multimodal peer benchmark methodology hardening,
Gemma 4 benchmark artifact sanitization, and a new Homebrew CLI entrypoint.

### Added
- **Homebrew CLI entrypoint** — `ax-engine` installable and runnable via Homebrew.
- **Cold peer benchmark** — refreshed Gemma 4 multimodal cold peer benchmark
  with documented llama peer launch contract.

### Changed
- **Atomic multimodal prefill scheduling** — prefill no longer split across
  scheduling boundaries, eliminating race conditions in peer benchmarks.
- Peer benchmark methodology hardened with stricter validation and
  reproducibility guarantees.
- Gemma 4 peer chart styling adjusted for clarity.

### Fixed
- **Benchmark preview smoke binary selection** — smoke test now selects the
  correct `ax-engine-bench` binary.
- **`cargo run` disambiguation** — `ax-engine-bench` cargo run commands now
  unambiguous in workspace.
- **Gemma 4 benchmark artifact paths** — sanitized to prevent path traversal
  in artifact naming.
- **Llama peer slot reuse** — rejected in multimodal benchmarks to prevent
  stale state contamination.
- **Llama audio cap peer row** — skipped due to instability.
- **Multimodal peer fairness** — hardened scheduling to ensure equitable
  resource allocation across peers.

## [6.2.0] - 2026-06-09

Completes the **Gemma 4 12B multimodal story** (image, audio, video) with
golden-validated preprocessing, introduces **speculation profile presets** for
workload-tuned MTP gating, and ships benchmark hardening and download UX improvements.

### Added
- **Gemma 4 multimodal chat** — inline video (GIF), image, and audio input in
  chat conversations.
- **Golden-validated preprocessing** — Python SDK preprocessing matches the
  reference implementation for audio and video vectors.
- **Video fidelity** — 70-token frames, 32-frame cap, mm:ss timestamp formatting.
- **Runtime smoke tests** — image TTFT and end-to-end multimodal probes.
- **Speculation profiles** — four presets (`auto`, `coding`, `agentic`,
  `chatbot`) with calibrated MTP draft-confidence gates. Gemma 4 gates
  calibrated from 12B ablation data. CLI flag `--speculation-profile` with
  programmatic SDK override.
- **Hugging Face Hub** snapshot download support.
- **Same-artifact direct-vs-MTP parity harness** for Gemma 4 12B.

### Changed
- **`--force` flag** now invalidates stale manifests in download destination.
- **Bundled benchmark binary** preferred; fails loudly on missing manifest.
- Multimodal benchmark modality set validation prevents invalid configurations.
- Qwen MTP improvement chart added to README.
- Gemma 4 MTP public artifacts and phase 4 results published.
- README announcement flow productized.

### Fixed
- **Embed mutex poison recovery** — embedding pipeline no longer panics on
  poisoned mutex.
- **EWMA clamp** — exponential moving average clamped to prevent numerical drift.
- **SSE role emission** — role field now correctly emitted in streaming responses.
- Multimodal benchmark artifact validation for missing or malformed outputs.
- Gemma4 multimodal QA probe false-positive content match.
- Multimodal config-loading divergences from the reference implementation.
- Shifted MTP norm sidecar validation.
- Bench doctor smoke status check.

## [6.1.3] - 2026-06-09

Introduces speculation profile presets (`auto`, `coding`, `agentic`,
`chatbot`) that bundle per-knob MTP and n-gram speculative-decode
configuration into named postures.

### Added
- **Speculation profiles** — four presets (`auto`, `coding`, `agentic`,
  `chatbot`) with calibrated MTP draft-confidence gates and n-gram policies.
- **CLI flag:** `--speculation-profile` / `-s` server flag with hidden alias
  `--spec`.
- **SDK wiring:** `EngineSessionConfig::mlx_speculation_profile` threaded
  through preview/resolved request structs.
- **Safe process-level override:** `AtomicU8` checked before env var,
  compatible with `unsafe_code = "forbid"`.
- **Precedence:** explicit per-knob env > profile preset > built-in default.
- **Telemetry:** route metadata emits `ax_mlx_speculation_profile` with
  `ResolutionSource` tracking.
- **Gemma 4 12B MTP phase 4 results** published.

### Preset Behavior
- **`auto`** (default) — temperature-driven: low temp defers to built-in
  defaults (never lowers Gemma 0.999 gate); high temp raises gates to protect
  sampling diversity.
- **`coding`** — optimized for code generation.
- **`agentic`** — optimized for agentic/tool-use workloads.
- **`chatbot`** — raises Qwen gate; prefers utility gate for n-gram policy.

## [6.1.2] - 2026-06-09

Patch release that establishes a fair same-artifact benchmark harness for
Gemma 4 12B MTP vs direct decode, adds GPU-exact draft confidence mode,
and fixes three latent bugs.

### Added
- **Same-artifact direct-vs-MTP parity harness** — direct decode and MTP now
  run in the same prompt-suite harness with identical artifacts. Survival
  taxonomy classifies each profile (keep-default/keep-opt-in/retest/reject/
  remove-claim) with artifact-parity gate and route-draft validation.
- **GPU-exact draft confidence mode** — opt-in
  `AX_MLX_GEMMA4_ASSISTANT_MTP_CONFIDENCE_MODE=gpu-exact` computes argmax +
  softmax confidence on-device. Default stays `exact-cpu`.
- **Per-workload gate guidance table** — README documents draft confidence
  gate as a speed knob with starting-point recommendations for coding,
  agentic, and chatbot workloads.
- **Affine quantization telemetry** — records bit composition per row
  (min/max bits, 4-bit and 8-bit tensor counts).
- **`unit_test` QA prompt** — verifies model produces ≥3 `def test_()` functions.
- **`json_invoice_nested` QA prompt** — verifies correct invoice total computation.

### Fixed
- **Embed mutex poison recovery** — extended to `embed_compile_stats`,
  `embed_compile_cache`, and `embed_batch_compile_cache` (5 call sites),
  eliminating cascade-panic risk in the embed JIT path.
- **EWMA clamp** — `accept_rate_ewma` and `mtp_only_accept_rate_ewma` clamped
  to `[0.0, 1.0]` before `u32` cast, preventing >1000 telemetry emissions.
- **SSE role emission** — `role:"assistant"` now always emitted in at least
  one SSE chunk, fixing zero-token completions that violated the OpenAI
  streaming API spec.

## [6.1.1] - 2026-06-08

Small patch release adding Hugging Face Hub download support and MTP norm
sidecar validation.

### Added
- **Hugging Face Hub download** — MLX model snapshots can now be downloaded
  directly via Hugging Face Hub.

### Changed
- **Shifted MTP norm sidecar validation** — validates shifted MTP norm
  sidecar files to catch malformed or mismatched artifacts early.

## [6.1.0] - 2026-06-08

First-class support for **Google Gemma 4 12B (unified)** with assistant
speculative decoding (MTP depth-2), multimodal preprocessing infrastructure,
and a ~34% GEMV decode kernel speedup.

### Added
- **Gemma 4 12B (unified) model support.** Full text inference with preset
  aliases for quick model selection. Multimodal preprocessing routes (image,
  audio, video) are wired and validated; full multimodal chat arrives in v6.2.0.
- **Assistant MTP depth-2 drafting.** Delivers 1.10–1.20× decode speedup over
  direct decode. Ships with 4-bit-FFN artifacts for fair benchmarking against
  llama.cpp Metal.
- **Benchmark suite.** Gemma 4 12B text benchmarks: direct, MTP, and MTP+n-gram
  vs llama.cpp Metal. New q4km GEMV throughput microbench for kernel diagnostics.
- **Multimodal infrastructure.** Unified media preprocessing pipeline, input
  validation for Gemma4 multimodal payloads, tokenized media routes, and video
  frame range handling.
- **Memory-bandwidth utilization table** in README for Gemma 4 12B.

### Changed
- **~34% faster `decode_projection_q4km` GEMV kernel** via vectorization and
  per-lane scale reuse.
- README reorganized: TOC, install-first flow, collapsed repro blocks.
- Gemma 4 12B benchmark artifacts and charts refreshed (4-bit-FFN direct +
  depth-2 MTP).
- Bandwidth chart refreshed with corrected height and "Higher is better" annotation.

### Fixed
- **Streaming lock scope** — tightened lock holding to prevent concurrent
  decode corruption.
- **LangChain SSE parser** — choices guard now correctly accumulates multi-line
  `data:` events per SSE spec.
- **Mutex poison recovery** — engine no longer panics on poisoned mutex after
  thread failure.
- Image zero-dimension guard prevents panic on empty image inputs.
- Environment variable typo in model download path resolution.
- CLI flag parser edge case for composite boolean flags.
- Gemma4 12B audio soft-token count and `k_eq_v` default corrected.
- Benchmark chart annotation overflow and axis label rendering.

## [6.0.1] - 2026-06-08

Release-pipeline fixes only. The engine, server, and SDK code is byte-for-byte
identical to `6.0.0`; the benchmark numbers in the README carry over unchanged.

### Fixed
- **PyPI wheel smoke test:** `python/tests/test_cli.py` prepended the source
  `python/` dir to `sys.path` unconditionally at import time, so during
  `unittest discover` the un-built source package shadowed the installed wheel
  and the `AX_ENGINE_RUN_INSTALLED_TESTS=1` smoke tests failed with
  `ModuleNotFoundError: ax_engine._ax_engine`. The insert is now guarded the
  same way as `test_embedding_smoke.py`. This was blocking the PyPI publish.
- **Homebrew tap push:** `actions/checkout` installs a global github.com
  credential helper (`persist-credentials: true`) that overrode the
  `HOMEBREW_TAP_TOKEN` embedded in the tap clone URL, so the tap push
  authenticated as the default Actions identity and was denied (403). The Brew
  workflow now checks out with `persist-credentials: false`.

### Changed
- **Brew release workflow no longer rebuilds the binary.** It previously built
  and uploaded its own tarball with `--clobber`, overwriting the minisign-signed
  artifact from `scripts/publish-github-release.sh` and breaking the signature
  and `.sha256`. It now reads the authoritative checksum from the published
  `.sha256` sidecar and points the formula at the signed asset, never rebuilding
  or clobbering it (and runs on `ubuntu-latest`).

## [6.0.0] - 2026-06-07

`v6.0.0` is a milestone release centred on the v6 MTP n-gram utility gating
work, a CLI-first local serving workflow, and a batch of latent correctness
fixes across the MLX backend, server, and SDKs. All workspace crates and the
Python/JS/Ruby SDKs are aligned to `6.0.0`.

### Added
- `ax-engine serve` local CLI, now the primary documented entry point for
  running a local server, with model download options.

### Changed
- MTP n-gram utility gating finalised for v6 (gaps closed across the n-gram
  acceleration path).
- `--mtp-depth-max` now defaults per model instead of a flat `1`
  (Qwen3.6 27B → 3, Qwen3.6 35B-A3B → 1). Pass the flag explicitly to override.
- QA now fails on Unicode replacement characters (`U+FFFD`) in model output,
  catching encoding regressions in CI.
- Refreshed Qwen3.6 fair and Gemma 4 assistant MTP benchmark documentation and
  charts; AX pure-MTP accept rows are reported at the accept-maximizing `0.98`
  draft-confidence gate with min–max ranges.

### Fixed
Six latent correctness bugs (see commit `82e270e4`):
- **RoPE (llama3):** `build_llama3_rope_freqs` used the inverse-frequency
  convention instead of the wavelength divisor `mlx_fast_rope` expects,
  corrupting RoPE for any `rope_scaling_type="llama3"` model (LLaMA-3/4). Now
  matches mlx-lm's Llama3RoPE divisor convention.
- **MTP draft gate:** a fully-accepted gated draft left stale KV rows, inflating
  `cache.seq_len` above `mtp_decode_count` and degrading later draft acceptance.
  The gated-out tail is now trimmed so the cache matches the returned count.
- **Prefix cache:** a cache hit handed every request the producer's greedy
  prefill token, so temperature/top-p (and repetition-penalty) requests received
  a deterministic greedy first token when the cache was warm. Token reuse is now
  gated on the consumer being greedy with no repetition penalty.
- **Server `/v1/models`:** the OpenAI text endpoints were reported as
  unsupported on the native MLX backend even though they are served. They are
  now advertised correctly.
- **Server streaming:** native MLX OpenAI streaming decoded each step's tokens
  independently, corrupting multi-byte UTF-8 (CJK/emoji) split across step
  boundaries. An incremental decoder now emits only completed codepoints.
- **Python langchain:** SSE parsing kept only the last `data:` line per event;
  it now accumulates and joins per the SSE spec.

### Notes
- No public Rust/Python API removals. The notable behavioural change to be aware
  of when upgrading is the model-aware `--mtp-depth-max` default.
