# ax-engine-py

Preview Python bindings for AX Engine v4.

This crate exposes the `ax_engine._ax_engine` extension module and stays aligned
with the v4 SDK contract rather than inventing a separate Python-only runtime
surface.

Current preview scope:

- SDK-backed `Session`
- fail-closed host validation for pre-M4 Macs
- runtime metadata reporting
- `mlx=True` selects the repo-owned MLX runtime
- non-MLX inference routes to `llama.cpp`
- text/chat convenience helpers over the same SDK-backed request contract
- stepwise request control via `submit(...)`, `step()`, `snapshot(...)`, and
  `cancel(...)`
- SDK-backed in-process `stream_generate(...)` lifecycle events emitted through
  a native incremental iterator
- llama.cpp compatibility via `llama_model_path`, `llama_cli_path`, or
  `llama_server_url`
- `native_mode=True` is retired and returns an error

Current non-goals:

- text tokenization and decoding
- automatic model-aware chat templating
- transport-level remote streaming
- broad compatibility lifecycle parity across multiple delegated adapters

For Phase 1, stepwise request-control methods support MLX preview plus the
server-backed `llama.cpp` compatibility path. One compatibility session can now
hold multiple active delegated requests while `step()` aggregates progress
across them.

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
