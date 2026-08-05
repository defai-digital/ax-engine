# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning.

## [Unreleased]

## [6.13.1] - 2026-08-05

### Added

- Model downloads accept Hugging Face links and pinned revisions everywhere:
  `ax-engine download`, the Python CLI/SDK (`ax_engine.download_model()`),
  and the TUI all parse bare `owner/repo`, `owner/repo@revision`, full
  `https://huggingface.co/owner/repo` links (`hf.co` included), and
  `/tree/<revision>` URLs through one shared parser
  (`ax-engine-core/src/repo_ref.rs`, mirrored in
  `python/ax_engine/_repo_ref.py`); file links (`/blob/...`, `/resolve/...`)
  and non-Hugging-Face hosts fail with an actionable message instead of a
  hub-library error. The TUI gains a download-by-link prompt (`d` on the
  Models screen) with bracketed-paste support that queues arbitrary repos
  through the same download queue as catalog picks.
- Download robustness: the helper verifies free disk space against the repo
  size before fetching, copies `--dest` atomically (temp dir + swap, so an
  interrupted copy is never mistaken for a complete model), only treats a
  pre-existing `--dest` as done when its contents actually validate, and
  emits its final `ax.download_model.v1` summary as the structured
  completion contract the TUI now prefers over log scraping.
  `ax_engine.download_model()` delegates to the same bundled helper (with a
  legacy fallback), so manifest semantics can no longer diverge between
  entry points.

### Fixed

- Model manifests now fail closed on missing or mismatched safetensors
  metadata, overlapping tensor ranges, invalid scalar roles, inconsistent
  attention geometry, and incorrect dense, MoE, or quantized projection
  shapes. Python and standalone download paths admit cached manifests only
  after native-loader validation, closing the remaining structural-check
  drift. Manifest replacement is atomic and cannot overwrite a shared Hub
  blob through a symlink.
- Model downloads now validate every cached snapshot entry before use,
  materialize only links into the requested snapshot or its Hub blob store,
  protect unrelated destinations during forced and concurrent installs, and
  emit one terminal structured-progress record on every failure path. Rust,
  Python, and standalone reference parsing also agree on URLs, revisions,
  option-looking values, control characters, and invalid ports.
- Download cancellation now terminates the complete child process group, and
  server rate limits reject non-finite values while keeping fractional request
  rates usable with a one-request minimum burst.
- Download destinations staged via the atomic-swap path keep umask-derived
  permissions instead of `mkdtemp`'s 0700 (a `--dest` written by one user is
  readable by the services that consume it), a symlinked `--dest` directory
  is followed to its target instead of being rejected when empty or silently
  replaced by a real directory under `--force`, manifest regeneration
  preserves a snapshot's existing manifest when no generator is available
  instead of unlinking it first, a backup that cannot be removed after a
  successful install is reported as a warning instead of failing the
  download, and every failure path that strands the previous destination in
  a uniquely named backup now says where it went.
- Revisions are percent-decoded exactly once across every entry point:
  the helper no longer re-decodes values embedded in the reference or
  pre-normalized by `main()`, the Rust CLI, Python CLI, and SDK re-escape
  literal `%` when invoking the helper, and stored provenance revisions are
  validated without a second decode — so pinned revisions containing a
  literal percent sign resolve, match, and report consistently.
- `ax-engine` CLI options that take a value (`--dest`, `--output`, `--host`,
  …) reject option-shaped values instead of consuming the next flag, so
  `--dest --force` is an error rather than a download into a directory
  literally named `--force`.
- TUI modals size themselves from post-wrap line counts, so a pasted long
  link or parser error in the download-by-link prompt no longer pushes the
  confirm/cancel chips out of the popup, and the `d` shortcut works on the
  Precision/Confirm wizard stages even when a stale family-filter flag is
  set.
- The manifest validator, the MLX packed-QKV slicer, and the Nemotron-H
  shape checks now agree that the packed KV section width is
  `kv_head_count × attention_head_dim` (the base geometry) on layers with a
  wider per-layer head dim, and MoE router sidecars
  (`ffn_gate_inp_correction_bias`, `ffn_gate_inp_expert_scale`) are
  shape-checked and counted as MoE evidence even when `ffn_gate_inp` is
  absent.
- `docs/RELEASING.md` documents the fresh-notarization-ticket recovery for
  the publisher's final `-R=notarized` verification: standalone Mach-Os
  cannot staple tickets and `codesign` only consults locally registered
  ones, so a just-accepted ticket fails the check until a one-time
  Gatekeeper install assessment registers it. The v6.13.0 publish hit this;
  the publisher itself deliberately keeps `codesign` as the only fail-closed
  gate.
- Qwen3.5/Qwen3.6 hybrid models (`qwen3_5` family, including
  Qwen3.6-35B-A3B) no longer receive the optimistic
  `MLX_MAX_MB_PER_BUFFER=1024` / `MLX_MAX_OPS_PER_BUFFER=1000`
  command-buffer raise. On the server path the raise cost prefill
  throughput and caused a one-way per-request prefill degradation (M5 Max
  6-bit MTP matrix: 35B-A3B prefill 971 → 517 tok/s from the pre-caps
  v6.9.0 build to v6.12.1, decode flat; M3 Max interleaved A/B: caps-off
  prefill flat at ~970 tok/s vs caps-on wobble/decline, decode identical
  ~45.5–46 tok/s). The family now keeps MLX defaults, following the
  existing Gemma/unlimited-OCR exclusions; `qwen3_next` (Coder-Next)
  retains the raise its greedy-decode evidence was measured on despite a
  ~5–6% sampled-prefill cost. Explicit `MLX_MAX_*_PER_BUFFER` values still
  win and `AX_MLX_AUTO_BUFFER_CAPS=0` remains the global kill switch.

## [6.13.0] - 2026-08-03

### Added

- Per-layer KV-cache quantization (AXQ-021): `NativeModelManifest` gains an
  optional `kv_cache_quantization` table (`layer_bits` / `layer_group_sizes` /
  `basis`, one entry per layer, bits from 4/6/8/16 with 16 meaning full
  precision, group sizes 32/64/128, validated against `layer_count`).
  `generate-manifest` lifts the table best-effort from a sibling
  `axquant_runtime.json` (`kv_cache` block, schema `axquant.runtime.v1`),
  skipping — never failing — on absent, malformed, or inconsistent input.
  Layers named by the table store K/V as affine-packed buffers in `MlxKVCache`
  (quantize-on-append for the new token slice, dequantize-on-read into the
  same dense views every attention path already consumes), with
  `usage_snapshot` reporting packed bytes (4-bit: 320 vs 1024 B/token dense).
  Quantized layers never take native paged decode or repage,
  rotating/protected-prefix rings demote the layer back to dense storage, and
  cache serialization keeps the dense wire format with re-quantization on
  first append. Only full-attention contiguous layers participate; MLA,
  linear-attention, and GLM caches are unaffected. `AX_KV_QUANT=0` disables
  the feature at injection; the default is on when a manifest table is
  present.
- AXQuant vision sidecars load with provenance verification at weight-load
  time: `vision.safetensors` is merged into the weight map (main-file tensors
  win over sidecar duplicates) only after its
  `axquant_vision_sidecar_manifest.json` passes strict checks on schema
  version, role, output path/size/SHA-256 binding, and tensor count. A missing
  manifest, a tampered sidecar, a stale manifest without the sidecar, or a
  wrong role fails closed with `WeightLoadError::VisionSidecarInvalid` instead
  of silently serving a model without vision weights.
- `ax-engine-bench doctor` now warns (advice-level, never readiness-failing)
  when AXQuant plan assignments or execution records use affine bit widths the
  runtime cannot load unconditionally: widths outside the supported set
  (7, 9-15) name the supported affine bits (4/5/6/8), and 2-bit/3-bit widths
  name the `AX_ENGINE_2BIT_EXPERIMENTAL=1` / `AX_ENGINE_3BIT_EXPERIMENTAL=1`
  gates required to load them. Warnings reuse the shared core bit constants
  and never affect `metadata_valid`.
- MTP sidecar bit-width inference now prefers a structured
  `mtp_sidecar_bits` field in the MTP runtime config over the free-text
  `mtp_sidecar` heuristic. Malformed structured values (wrong type or outside
  the supported set) log a warning and fall through to the heuristic;
  heuristic guesses are debug-logged with a hint to declare
  `mtp_sidecar_bits` explicitly.
- KV-cache quantization and the 2-bit experimental gate now surface in route
  telemetry: `ax_mlx_kv_quantized_layers` reports the per-request peak count
  of layers actually holding quantized storage (so ring demotions and
  `AX_KV_QUANT=0` are observable, not just manifest intent) and joins the
  canonical KV counter set; `ax_mlx_experimental_2bit_gate` records the
  `AX_ENGINE_2BIT_EXPERIMENTAL` load-time state alongside the existing
  3-bit gate counter.
- `docs/ROADMAP.md` gains a v7.0.0 readiness-gate table from the 2026-08-03
  multi-model / multimodal / AXQuant maturity review.

### Fixed

- `docs/SERVER.md` no longer advertises a vision-feature cache
  (`AX_MLX_VISION_FEATURE_CACHE`) that is not wired into the serving path;
  the multimodal caching section now states the real behavior (full prefill
  recompute unless `AX_MLX_MULTIMODAL_PREFIX_REUSE=1`). Multi-model unload
  and idle eviction are now documented as soft-parking the retired
  generation (weights stay resident for fast same-id reload; no TTL or cap),
  including the memory-preflight visibility caveat.
- The MLX eval-site baseline accepts the vision-sidecar loader's reviewed
  load-path `eval`, restoring the `Scripts and Bench Smoke` CI gate.

## [6.12.1] - 2026-08-02

### Added

- `ax-engine-bench doctor` now recognizes AXQuant artifact metadata and reports
  actual mixed-precision widths, measured BPW, evidence kind, source provenance,
  and quantizer execution coverage. It validates the artifact's SHA-256 metadata
  bindings and plan lineage, fails readiness for malformed metadata or any
  failed/fallback module, and warns without blocking when evidence is marked for
  development rather than release certification.
- `AX_ENGINE_2BIT_EXPERIMENTAL=1`: admits affine-quantized MLX artifacts at
  2-bit under the same experimental contract as the existing 3-bit gate —
  rejected by default, no quality or correctness guarantee, MLX affine
  kernels execute the width natively. Manifest validation errors name the
  gate. Verified end to end with an AXQuant mixed 2/4/8/BF16 MiniCPM5-1B
  development artifact passing doctor and MLX-LM generation.
- `AX_MLX_MTP_ASYNC_DRAFT` (default off): the greedy zero-gate MTP draft is
  scheduled with `async_eval` and the speculative verifier chains directly
  on the lazy draft-token arrays, so the verify graph builds while the
  draft head's GPU forward is still running and one eval batch
  materialises both. Exactness-preserving by construction (the identical
  lazy graph is evaluated; only the synchronization point moves) and
  verified on real Qwen3.6 27B artifacts: byte-identical greedy output on
  M3 and M5, draft wall 4.2 -> 0.2 ms/cycle, and the formal-protocol
  depth-1 MTP speedup on Apple M5 Max improving from 1.097x to 1.191x.
  Engages only under the exact profile with the confidence gate disabled,
  non-stochastic drafting, and skip-state off; every other flow
  materialises the draft at the next cycle start unchanged.
- The Qwen linear-attention exact MTP profile now serves draft depths 2-3
  through the lazy committed-prefix checkpoint instead of falling back to
  per-cycle singleton replay. The invariant-projection contract was already
  validated for a 1-4 token verifier; the runner's eligibility gate was the
  only depth-one restriction. Full accepts adopt the verify cache, complete
  misses restore the committed-prefix checkpoint at any depth, and partial
  accepts recompute only the short committed prefix. Measured on Apple M5
  Max (Qwen3.6 27B mixed-precision, greedy, exact profile): depth 2 repeats
  exact output with zero divergence at 2.59 emitted tokens/cycle, but
  verify-eval scales ~linearly with verifier length (35 ms/cycle at 2
  tokens, 47 ms at 3), so depth 1 remains the fastest configuration until
  the multi-token verify path approaches bandwidth-bound scaling.
- `mtp_norm_layout` in `mtplx_runtime.json`: a fused MTP sidecar can now
  declare whether its 1-D RMSNorm tensors are raw HF zero-centred deltas
  (`"raw_hf_delta"`, loader applies the `+1.0` lift to every norm) or
  already-shifted MLX multipliers (`"mlx_multiplier"`, loader leaves them
  unchanged), so third-party byte-preserving converters (e.g. AXQuant) load
  correctly without statistical guessing. Both prepare scripts declare
  `mlx_multiplier` in the contracts they write; absent or unknown values keep
  auto-detection.

### Fixed

- `ax-engine-bench doctor` now validates the native model manifest and every
  referenced tensor file before reporting model artifacts as ready. A malformed
  manifest, a missing referenced shard, or a directory merely named with a
  `.safetensors` suffix can no longer produce a false-ready result.
- AXQuant doctor validation now recomputes the plan's Python-compatible
  canonical JSON digest, requires exact plan-to-execution module coverage,
  rejects malformed or duplicate execution records and manifest bindings,
  verifies embedded runtime and BPW metadata, and reuses one artifact snapshot
  for both readiness and performance advice. This closes cases where internally
  inconsistent AXQuant metadata could previously pass readiness or produce
  contradictory doctor output.
- MTP sidecar norm auto-correction now decides once per sidecar instead of
  per tensor. Raw HF deltas are not uniformly small (Qwen 3.6's raw
  `q_norm`/`k_norm`/`mtp.norm` deltas have mean-abs 0.21–1.27 while the raw
  input layernorm sits at 0.08), so the old per-tensor `mean_abs < 0.15`
  test lifted exactly one of the seven norms and left the sidecar in a
  silently mixed state that collapsed draft acceptance to 0/40 — while the
  load-time warning claimed the correction had been applied. A single
  sub-threshold norm now marks the whole sidecar raw and every norm is
  shifted together.
- Qwen fused-sidecar MTP: the draft gate no longer jumps to the chatbot
  threshold for high-temperature sampling (the diversity regime now defers
  to the 0.90 default, and the high-temperature Auto profile resolves to the
  default gate directly). The pinned gate roughly doubled empty-draft MTP
  steps and cost up to 26% MTP decode on Qwen3.6 27B 6-bit; the refreshed
  2026-07-29 publication campaign confirms recovery to the published band
  on every regressed row with Gemma assistant-MTP rows unchanged.
- Release publisher: export `COPYFILE_DISABLE=1` so macOS bsdtar does not
  embed AppleDouble `._*` members (extended attributes such as
  `com.apple.provenance`) in the release archive, which failed the signed
  manifest's archive-member verification during upload re-verification.

### Changed

- Refreshed the published 6-bit MTP acceleration matrix (README and
  Performance Results) from the 2026-07-29 campaign on the fixed runtime:
  15/15 rows accelerate 1.28x-2.64x, artifacts under
  `benchmarks/results/speculative/mtp-6bit/2026-07-29-v6.12.1-m5max-supported-mtp-ax-only/`.
  The prior v6.9.0 matrix's higher Qwen3.6 35B-A3B bound benefited from the
  since-removed incorrect MTP skip-state path (see the Performance Results
  provenance note).

## [6.12.0] - 2026-07-29

### Added

- S1 multi-model prefix reuse: the exact-prompt warmup's KV snapshot now
  restores on the replayed request (block-aligned warm, right-sized
  prefix-cache budget in the tracked S1 target), taking the Gemma 4 12B
  13.8k-token leg from ~8.2 s to ~0.4 s TTFT and the official
  single-process-vs-mlxcel campaign to 4/4 locked gates at a 5.0x
  throughput ratio (artifacts under
  `benchmarks/results/serving/s1-peer-flip/`).
- `AX_MLX_FUSED_PREFILL_ATTENTION=1` (default off): fused offset-0 prefill
  attention for Gemma-family layers (rms_norm -> QKV qmm -> QK/V norms ->
  rope -> causal fast SDPA -> o-proj in one shim call), measured -9.7%
  TTFT at p128 on Gemma 4 12B and +6-8% cold prefill on 26B, plus a
  two-stage fused pair for offset/sliding/ring chunks.
- `mlx.version` accepts `git:<sha>@<version>` for admitted MLX source
  builds alongside wheel semvers; provenance enforcement and release
  packaging documentation updated accordingly.
- `AX_SERVER_WORKER_RECYCLE_AFTER_TICKS`: opt-in idle-time engine-session
  rebuild for long-lived deployments, bounding the per-process Metal
  steady-state accumulation until the upstream MLX residency fix ships.
- `AX_MLX_SIBLING_PREFILL_ROTATION=0|1` hard override for the sibling
  prefill-rotation hint (A/B isolation), and richer prefix-cache /
  prefill diagnostics under `AX_MLX_PREFILL_TIME_DEBUG`.

### Fixed

- Host SoC detection no longer reports `unknown Apple Silicon` (refusing
  to start MLX backends) in minimal-`PATH` environments: detection now
  falls back to the absolute `/usr/sbin/sysctl` and
  `/usr/sbin/system_profiler` locations, fixing `ax-engine serve` on
  hosts where `ax-engine doctor` already recognized the chip
  (issue #73, Apple M3 Pro / `Mac15,6`).
- Rotated-ring KV corruption panics: the prefill rotation decision now
  latches per request (the process-wide sibling hint could flip
  mid-prompt and hand a rotated ring an ordered append), and the
  adaptive prefill quantum clamps to the operator `--prefill-chunk`
  (quanta larger than the ring capacity panicked at kv_cache.rs:2065).
- Multi-model fair prefill quantum now honors `--prefill-chunk` instead
  of the pinned 256-token default, and the single-stream engine burst
  releases the arbiter turn per prefill quantum under sibling load.
- The third GEGLU JIT kernel (`ax_moe_fused_activation_unsort`) now uses
  the branchless saturation form like its siblings (divergent guards
  serialize vectorized loads).
- Warm-extend prefix restores snap to the cold prefill-chunk grid so
  extension chunks replay the cold trail shape-for-shape; the remaining
  short-prompt warm_extend variance is root-caused as in-session
  recompute non-determinism and documented in PERFORMANCE-RESULTS.

- Native direct-mode support for **Nemotron-H** (`nemotron_h`, Nemotron 3 Nano):
  hybrid Mamba-2 / GQA attention / ReLU² MoE residual mixers driven by
  `hybrid_override_pattern`, convert mapping under `backbone.layers.*.mixer.*`,
  and ChatML chat templating for Nemotron hub ids.
- Native direct-mode support for **Unlimited-OCR** (`unlimited_ocr` /
  DeepSeek-OCR lineage): dual vision (SAM-ViT-B + CLIP-L) + SWA MoE language
  tower, MXFP8 dense/expert packs, causal full-prompt prefill plus bounded R-SWA
  decode, DeepSeek-V2 `rms_norm_eps=1e-6` default, and an image+text smoke
  binary. The public Python request helper accepts a local image/Pillow image,
  reproduces the released 1024px global plus dynamic 640px crop grid, expands
  the exact soft-token span, and exposes bounded no-repeat n-gram decoding
  through `Session`.
- All 25 public AutomatosX model packs
  (https://huggingface.co/AutomatosX/models) are first-class managed downloads:
  `ax-engine download` / `serve` aliases for Qwen 3.5/3.6, Gemma 4,
  Qwen3-Coder-Next, DiffusionGemma, EmbeddingGemma, and Qwen3-Embedding. The
  TUI shows only these reviewed snapshots, distinguishes QAT/OptiQ/DWQ recipes,
  and downloads bundled MTP/assistant artifacts directly without a separate
  packaging step. Multi-model `load_mode=add` accepts supported `AX-`-branded
  model ids while keeping manifest signatures authoritative.
- Multi-model allowlist extends to Qwen3-Coder-Next and to embedding
  co-residency targets (EmbeddingGemma 300M, Qwen3-Embedding 0.6B/4B/8B), so
  one process can serve `/v1/chat/completions` and `/v1/embeddings` side by
  side, routed by `model`.
- `qwen3-coder-next` server preset (`--preset qwen3-coder-next`, model_type
  `qwen3_next`).
- Embedding repos download through the standard `ax-engine download` /
  `download_model()` flow; the AX-ready manifest check replaces the previous
  name-based rejection.

### Changed

- Hub chat-template fidelity audit for every distinct `chat_template.jinja`
  under the primary model hub: DiffusionGemma no longer pre-fills Gemma IT's
  empty thought channel; Llama 4 uses `<|header_start|>` / `<|eot|>` (not Llama
  3 markers), injects tool schemas into the first user turn, wraps tool_calls in
  `<|python_start|>…<|python_end|>`, and frames tool results as `ipython`;
  Ministral folds system into the *last* `[INST]`, emits ` content</s>` for
  assistants, prefixes `[AVAILABLE_TOOLS]` on the last user turn, and replays
  history as `[TOOL_CALLS]` / `[TOOL_RESULTS]` (9-char call ids); Devstral/Mistral
  append `</s>` after assistant turns; Qwen3.6 tools match Qwen3.5 function-XML +
  JSON schemas (Coder-Next keeps XML declarations). Intentional deltas remain for
  GPT-OSS final-channel prefill and Llama 3 omitting the default knowledge-date
  system preamble.
- Gemma 4 chat prompt rendering aligns with Google's 2026-07-09 canonical
  chat template: multi-turn tool loops keep tool_call, tool_response, and the
  follow-up answer in one model turn; prior assistant thinking channels are
  stripped from prefill history; optional `reasoning` enables official
  thinking (`<|think|>` + open thought channel). Default remains thinking-off
  with an empty thought prefill for short OpenAI-compatible answers.
- Managed downloads are restricted to the curated AutomatosX catalog:
  `ax-engine download --list`, the TUI catalog, and `download <alias>` cover
  the `ax-*` aliases only. Legacy mlx-community aliases (`qwen36-35b`,
  `gemma4-12b`, `gpt-oss-20b`, …) remain serve aliases for already-downloaded
  artifacts, and raw `org/repo` ids stay an explicit download escape hatch.
- The TUI Models wizard is three steps (family → size → confirm): AutomatosX
  snapshots bundle their MTP/assistant extras, so the separate speed-up
  download step is gone and bundled MTP is reported on the confirm summary.

- `mlx_lm.server` migration surface: `/chat/completions` aliases
  `/v1/chat/completions`, `max_completion_tokens` takes precedence over
  `max_tokens`, and `stream_options.include_usage` emits the OpenAI-standard
  final empty-`choices` usage chunk (including delegated mlx-lm streams whose
  usage arrives after the terminal choice).
- Stateless `POST /v1/responses` subset for text/message input, function-call
  history, native model-family function tools, structured text formats,
  reasoning, and OpenAI-shaped output/usage. Delegated tools, persisted state,
  streaming Responses events, hosted prompts, background mode, built-in tools,
  and MCP fail closed.
- `/v1/embeddings` accepts string and string-batch input using the loaded
  model tokenizer (including configured EOS), in addition to token arrays.
  Unsupported base64 encoding and dimension projection fail closed.

- Multi-model serving: `POST /v1/model/load` accepts `load_mode=add` to keep
  multiple models resident (scoped to Qwen 3.5 9B, Qwen 3.6 27B/35B, and Gemma 4
  12B/26B/31B), with per-request `model` routing across the OpenAI, gRPC,
  Ollama, and Anthropic surfaces and `POST /v1/model/unload` to retire a
  retained model. Load/unload preflight runs synchronously before admission
  drain.
- Memory-aware load admission: loads whose projected peak resident set
  exceeds the Metal working-set budget are rejected with
  `422 insufficient_memory` before any drain. The estimate combines on-disk
  safetensors bytes with each model's worst-case KV pool derived from
  manifest attention geometry (every KV-backed layer at the configured
  pool — sliding-window rings bound KV per request, not per pool, so
  sliding layers differ only in head dim; hybrid linear-attention and
  KV-shared layers charge no per-token cache), so it scales with
  `--total-blocks` and with the number of resident models;
  `AX_SERVER_LOAD_MEMORY_PREFLIGHT=off` disables the check.
- `POST /v1/model/load` accepts `make_default` (default `true`;
  `load_mode=add` only) so a model can be added without changing what
  requests that omit `model` resolve to; load and unload responses report
  the resulting `default_model_id`. The Go and Swift typed clients and the
  JavaScript type declarations expose both fields (Ruby and JavaScript
  request bodies already pass through unknown fields).
- `/health` and `/v1/discovery` list every loaded model id (`models`)
  alongside the default `model_id` in multi-model serving.
- SDK typed-contract catch-up: tool calling (request `tools`/`tool_choice`/
  `response_format`, response `tool_calls`, streamed tool-call deltas),
  `reasoning_content`, `usage.prompt_tokens_details.cached_tokens`, `/health`
  `models` + `runtime`, full `/v1/models` cards, and `GET /v1/runtime`
  clients in Go (`Runtime()`) and Swift (`runtime()`); JavaScript adds
  per-call `AbortSignal` support and complete type declarations. Swift's
  typed request defers free-form `tools` fields (Foundation snake-case key
  rewriting would corrupt arbitrary schema keys — documented in
  `docs/sdk/swift.md`) while fully typing tool-call responses and echo.
- Swift SDK fix: `step(model:)` built the query with `appendingPathComponent`,
  percent-encoding the `?` — multi-model step always returned 404. The query
  is now attached via `URLComponents`, with a regression test.
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

- After merging Nemotron-H and Unlimited-OCR direct-mode: standard MoE
  decode now honors `moe.sigmoid_routing` (aligned with the MTP path),
  convert unsupported-type diagnostics list `nemotron_h` /
  `unlimited_ocr` / related families, and `probe_mlx_model_support.py`
  accepts `unlimited_ocr`.
- Unevaluated-array readbacks in `mlx-sys` now fail with a recorded error
  instead of a SIGSEGV: the shim checks `is_available()` before
  `data<T>()`, `MlxArray::is_evaled` exposes the precondition, and
  `first_u32_unchecked` panics with the reason instead of silently
  returning token id 0 on a failed read. Manifest validators guarding
  decode-path invariants (diffusion contract, GLM router numeric-group
  invariants, interleaved layer_types shapes) gained direct unit tests,
  MTP decode carries emission-accounting debug asserts, and `make_array`
  normalizes bool payload bytes (undefined behavior for values >1).
- MLX panic containment for serving: the generation worker loop now runs
  under `catch_unwind`, and the new `release-server` profile (inherits
  release, `panic = "unwind"`) is the documented build for serving
  binaries. A runtime MLX failure — the FFI reports errors by panicking,
  reachable on the decode hot path under memory pressure — now retires
  only the affected model's worker (in-flight requests fail unavailable,
  sibling models keep serving, `POST /v1/model/load` recovers) instead of
  aborting the whole server process. The plain `release` profile keeps
  fail-fast `panic = "abort"` for bench/CLI binaries.
- `pyproject.toml` now pins `profile = "release-pyext"` in `[tool.maturin]`,
  so every build frontend (`pip install .`, plain `maturin build --release`,
  PEP-517 builds from the sdist) produces the unwind-capable extension.
  Previously only `scripts/build-pypi-wheel.sh` passed the profile, and
  other paths silently inherited the workspace `panic = "abort"` release
  profile — a reachable Rust panic would SIGABRT the host Python process
  instead of raising `PanicException`.
- MTP skip-state (`AX_MLX_MTP_SKIP_STATE`) is now **off by default** and
  fixed to never emit literal token id 0. The path — which only engaged
  when the draft gate left a cycle without pending drafts — had three
  defects: the greedy primary was committed through `sample_logit_row`'s
  argmax shortcut with a placeholder `0`, emitting "!" tokens (fixed: the
  skip capture now carries the row argmax inside its existing async_eval
  batch); a correctly-computed greedy primary still duplicates the
  previous tail by construction; and every capture-cycle tail was emitted
  without ever being forwarded, leaving it out of the KV history for both
  greedy and sampled requests. Benchmark workloads draft nearly every
  cycle and never entered the path, so headline MTP numbers are
  unaffected; flat/short prompts on the native greedy path (or OpenAI
  requests with an explicit `repetition_penalty` of 1.0) hit it every
  cycle. Enabling the flag now logs a warning.
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
