# AX Engine

High-performance local inference engine for Apple Silicon — Python bindings.

## Installation

### Python (pip)

```bash
pip install ax-engine
```

Requires macOS 14+, Apple Silicon (M2 Max or newer), Python 3.10+.

### Command-line tools (Homebrew)

To install the `ax-engine-server` HTTP adapter and `ax-engine-bench` CLI:

```bash
brew install defai-digital/ax-engine/ax-engine
```

Then verify:

```bash
ax-engine-server --help
ax-engine-bench doctor
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
    --port 8080
```

Then point any OpenAI client at `http://127.0.0.1:8080`.

## Optional dependencies

```bash
pip install "ax-engine[openai]"   # FastAPI + uvicorn for the OpenAI shim
pip install "ax-engine[download]" # mlx-lm for model downloading helpers
```

## Requirements

- macOS 14 (Sonoma) or later
- Apple Silicon — M2 Max / M2 Ultra / M3 / M4 family (32 GB RAM minimum)
- Python 3.10+

## Source

[github.com/defai-digital/ax-engine](https://github.com/defai-digital/ax-engine)
