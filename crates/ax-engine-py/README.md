# AX Engine

High-performance local inference engine for Apple Silicon — Python bindings.

## Installation

This package README describes the current `6.4.x` Python package. If your
package index only shows older `ax-engine` versions, those wheels may not expose
the top-level `ax-engine` CLI commands shown below.

### Python (pip)

```bash
python3 -m pip install "ax-engine[download]>=6.4.1,<7"
```

Requires macOS 26+, Apple Silicon (M2 Max or newer), Python 3.10+.
The current macOS arm64 wheel includes the `ax-engine` orchestration CLI plus
bundled `ax-engine-server` and `ax-engine-bench` binaries.

Verify the installed command surface:

```bash
ax-engine doctor
ax-engine-server --help
```

Homebrew, source-build, and release-archive details are documented in the
[Getting Started installation guide](https://github.com/defai-digital/ax-engine/blob/main/docs/GETTING-STARTED.md#installation).

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
    --port 8080
```

Then point any OpenAI client at `http://127.0.0.1:8080`.

## Optional dependencies

Install the OpenAI shim or image/audio helpers with the matching extra:

```bash
python3 -m pip install "ax-engine[openai]>=6.4.1,<7"
python3 -m pip install "ax-engine[multimodal]>=6.4.1,<7"
```

## Requirements

- macOS 26 (Tahoe) or later
- Apple Silicon — M2 Max / M2 Ultra / M3 / M4 family (32 GB RAM minimum)
- Python 3.10+

## Project

[github.com/defai-digital/ax-engine](https://github.com/defai-digital/ax-engine)
