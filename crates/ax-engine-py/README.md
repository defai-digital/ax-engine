# ax-engine-py

Preview Python bindings for AX Engine v4.

This crate exposes the `ax_engine._ax_engine` extension module and stays aligned
with the v4 SDK contract rather than inventing a separate Python-only runtime
surface.

Current preview scope:

- SDK-backed `Session`
- fail-closed host validation for pre-M4 Macs
- runtime metadata reporting
- native token-based `generate(...)`
- text/chat convenience helpers over the same SDK-backed request contract
- stepwise request control via `submit(...)`, `step()`, `snapshot(...)`, and
  `cancel(...)`
- SDK-backed in-process `stream_generate(...)` lifecycle events emitted through
  a native incremental iterator
- first compatibility preview path via
  `support_tier="compatibility"` plus `compat_server_url` for the delegated
  server-backed adapter, with `compat_backend="vllm"` or
  `compat_backend="mistral_rs"` or `compat_backend="mlx"` available for
  server-backed compatibility targets,
  and `compat_cli_path` / `compat_model_path` retained as CLI fallbacks for
  blocking `generate(...)` through `llama.cpp` or direct `mlx-lm`; the
  server-backed path also supports iterator-style `stream_generate(...)` plus
  preview stepwise request control

Current non-goals:

- text tokenization and decoding
- automatic model-aware chat templating
- transport-level remote streaming
- broad compatibility lifecycle parity across multiple delegated adapters

For Phase 1, stepwise request-control methods now support native preview plus
the server-backed `llama.cpp`, `vLLM`, `mistral.rs`, and explicit MLX
compatibility paths, and one compatibility session can now hold multiple active
delegated requests while `step()` aggregates progress across them. The `mlx`
path also supports a blocking direct `mlx_lm.generate` fallback, while the CLI
paths remain narrower text-prompt bring-up routes rather than full streamed
runtime integrations.

Build from the repository root with `maturin`:

```bash
maturin develop
```

For a repo-owned integration smoke check that bootstraps a temporary virtual
environment, installs `maturin`, builds the extension, runs the checked-in
Python examples, and then runs both the installed-package preview tests and the
wrapper tests, use:

```bash
bash scripts/check-python-preview.sh
```

To verify the extension-module packaging path without installing it, build from
the repository root with:

```bash
cargo build -p ax-engine-py --no-default-features --features python-extension
```
