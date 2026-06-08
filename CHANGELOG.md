# Changelog

All notable changes to AX Engine are documented here. This project follows
[Semantic Versioning](https://semver.org/). Releases prior to `v6.0.0` are
tracked via Git tags and GitHub Releases.

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
