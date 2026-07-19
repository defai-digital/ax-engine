# AX Engine

High-performance local inference engine for Apple Silicon — Python bindings.

## Installation

This package is the **Python SDK** for AX Engine. For the primary end-user
CLI / server install on macOS Apple Silicon, use **Homebrew** instead:

```bash
brew tap defai-digital/ax-engine
brew install defai-digital/ax-engine/ax-engine
ax-engine doctor
```

See the
[Getting Started installation guide](https://github.com/defai-digital/ax-engine/blob/main/docs/GETTING-STARTED.md#installation)
for brew trust, linkage, and troubleshooting details.

### Python (pip) — SDK and library use

```bash
python3 -m pip install "ax-engine[download]>=6.9.0,<7"
```

Requires macOS 26+, Apple Silicon (M2 Max or newer), Python 3.10+.
The current macOS arm64 wheel also includes the `ax-engine` orchestration CLI
plus bundled `ax-engine-server` and `ax-engine-bench` binaries when you need a
wheel-only install. If your package index only shows older `ax-engine`
versions, those wheels may not expose the top-level CLI commands shown below.

Verify the installed command surface:

```bash
ax-engine doctor
ax-engine-server --help
```

## Quick start

```python
import ax_engine

session = ax_engine.Session(mlx=True, mlx_model_artifacts_dir="/path/to/model")
result = session.generate([token_id, ...], max_output_tokens=128)
print(result.output_tokens)
```

Or use the OpenAI-compatible shim:

```bash
python -m ax_engine.openai_server \
    --model-id my-model \
    --mlx-model-artifacts-dir /path/to/model \
    --tokenizer /path/to/tokenizer.json \
    --port 31418
```

Then point any OpenAI client at `http://127.0.0.1:31418`.

## Optional dependencies

Install the OpenAI shim or image/audio helpers with the matching extra:

```bash
python3 -m pip install "ax-engine[openai]>=6.9.0,<7"
python3 -m pip install "ax-engine[multimodal]>=6.9.0,<7"
```

## Requirements

- macOS 26 (Tahoe) or later
- Apple Silicon — M2 Max / M2 Ultra / M3 / M4 family (32 GB RAM minimum)
- Python 3.10+

## Project

[github.com/defai-digital/ax-engine](https://github.com/defai-digital/ax-engine)
