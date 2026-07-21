# AX Engine

High-performance local inference engine for Apple Silicon — Python bindings.

## Installation

### Python SDK (pip)

Use pip when your application imports `ax_engine` or needs AX Engine's optional
Python integrations. Install the wheel in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade "ax-engine[download]>=6.11.0,<7"
ax-engine doctor
```

Requires macOS 26+, Apple Silicon (M2 Max or newer), Python 3.10+.
The current macOS arm64 wheel also includes the `ax-engine` orchestration CLI
plus bundled `ax-engine-server` and `ax-engine-bench` binaries when you need a
wheel-only install.

Verify the installed command surface:

```bash
ax-engine doctor
ax-engine-server --help
```

### Native CLI and server (Homebrew, primary)

For end users running the TUI, CLI, server, or benchmark tools, Homebrew is the
primary installation channel:

```bash
brew tap defai-digital/ax-engine
brew trust --formula \
  defai-digital/ax-engine/ax-engine \
  defai-digital/ax-engine/mlx \
  defai-digital/ax-engine/mlx-c
brew install defai-digital/ax-engine/ax-engine
ax-engine doctor
```

If both channels are installed, an active virtual environment normally shadows
Homebrew's commands. Use `which -a ax-engine` to identify every copy and avoid
mixing versions in one shell. See the
[Getting Started installation guide](https://github.com/defai-digital/ax-engine/blob/main/docs/GETTING-STARTED.md#installation)
for requirements, linkage details, and troubleshooting.

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
python3 -m pip install --upgrade "ax-engine[openai]>=6.11.0,<7"
python3 -m pip install --upgrade "ax-engine[multimodal]>=6.11.0,<7"
```

## Requirements

- macOS 26 (Tahoe) or later
- Apple Silicon — M2 Max / M2 Ultra / M3 / M4 family (32 GB RAM minimum)
- Python 3.10+

## Project

[github.com/defai-digital/ax-engine](https://github.com/defai-digital/ax-engine)
