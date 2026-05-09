# Python

`python/ax_engine` is the preview Python access layer for AX Engine.

It is intentionally thin:

- it wraps `ax-engine-sdk`, not `ax-engine-core`
- it exposes runtime metadata from the SDK contract
- `mlx=True` selects the repo-owned MLX runtime
- `support_tier="mlx_lm_delegated"` selects explicit upstream `mlx-lm` text
  compatibility through `mlx_lm.server`
- non-MLX inference routes to `llama.cpp`
- llama.cpp preview reuses the same SDK contract instead of inventing a
  Python-only adapter

## Current Scope

The current preview package provides:

- `Session(...)`
- fail-closed host validation for pre-M4 Macs
- `Session.runtime()`
- `Session.generate(...)`
- `Session.generate_text(...)`
- `Session.submit(...)`, `Session.step()`, `Session.snapshot(...)`, and
  `Session.cancel(...)`
- `Session.submit_text(...)`
- `Session.stream_generate(...)` for preview in-process lifecycle streaming
- `Session.stream_text(...)`
- text-first chat convenience through `ChatMessage`,
  `Session.chat(...)`, `Session.submit_chat(...)`, and
  `Session.stream_chat(...)`
- backend-resolution metadata such as `selected_backend`, `support_tier`, and
  `resolution_policy`, plus host and Metal-toolchain diagnostics
- explicit MLX artifact selection through `mlx_model_artifacts_dir`
- delegated `mlx-lm` text compatibility through `mlx_lm_server_url`
- llama.cpp wiring through `llama_cli_path`,
  `llama_model_path`, or `llama_server_url`

The current preview package does not yet provide:

- text tokenization or decoding helpers
- automatic model-aware chat templating
- transport-level streaming APIs

## Install

From the repository root:

```text
maturin develop
```

The preview package requires a local Apple M4-or-newer host.
Constructing `Session(...)` on an M3-or-older Mac now fails closed instead of
pretending native or delegated support exists.

This builds the Rust extension and installs the `ax_engine` package into the
active Python environment.

If you want the repo-owned MLX runtime to use explicit validated local model
artifacts, pass that path directly into `Session(...)`:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    mlx=True,
    mlx_model_artifacts_dir="/absolute/path/to/mlx-model-artifacts",
) as session:
    runtime = session.runtime()

print(runtime.mlx_model)
```

If you want a repo-owned smoke check that bootstraps a temporary virtualenv,
installs `maturin`, builds the extension, runs the checked-in examples, and
then runs both the installed-package preview tests and the Python wrapper
tests, including server-backed llama.cpp generate / stream coverage, use:

```text
bash scripts/check-python-preview.sh
```

## Example

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    mlx=True,
    mlx_model_artifacts_dir="/absolute/path/to/mlx-model-artifacts",
) as session:
    runtime = session.runtime()
    result = session.generate(input_text="Hello from default MLX path", max_output_tokens=2)

print(runtime)
print(result.output_text)
```

MLX preview sessions currently accept pre-tokenized `input_tokens`:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    mlx=True,
    mlx_model_artifacts_dir="/absolute/path/to/mlx-model-artifacts",
) as session:
    result = session.generate([1, 2, 3], max_output_tokens=32)

print(result.runtime.selected_backend)
print(result.output_tokens)
```

If you want the repo-owned MLX runtime:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    mlx=True,
    mlx_model_artifacts_dir="/absolute/path/to/mlx-model-artifacts",
) as session:
    result = session.generate(input_text="Hello from MLX", max_output_tokens=32)

print(result.prompt_text)
print(result.output_text)
```

If you want an unsupported MLX text model to stay behind AX Engine Python
surfaces, run `mlx_lm.server` yourself and choose the explicit delegated route:

```python
import ax_engine

with ax_engine.Session(
    model_id="local-mlx-model",
    support_tier="mlx_lm_delegated",
    mlx_lm_server_url="http://127.0.0.1:8090",
) as session:
    result = session.generate_text("Hello from mlx-lm", max_output_tokens=32)

print(result.runtime.selected_backend)
print(result.output_text)
```

`mlx_lm_delegated` is text-only delegated compatibility. The Python blocking
generate path forwards text prompts to upstream `mlx_lm.server`; token-array
prompts and stepwise lifecycle calls still fail closed. Streaming text surfaces
forward delegated text deltas through AX envelopes and remain compatibility
evidence, not repo-owned MLX throughput. Use `mlx=True` when the goal is
repo-owned MLX inference.

If you pass a local GGUF model, AX routes the request to delegated `llama.cpp`:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    llama_model_path="/absolute/path/to/model.gguf",
) as session:
    result = session.generate(input_text="Hello from llama.cpp", max_output_tokens=32)

print(result.prompt_text)
print(result.output_text)
```

For callers who want the thin Python wrapper to own the text prompt wiring
directly, `generate_text(...)` is a convenience alias over the llama.cpp-backed
llama.cpp request contract:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    llama_server_url="http://127.0.0.1:8081",
    delegated_http_connect_timeout_secs=30,
    delegated_http_read_timeout_secs=300,
    delegated_http_write_timeout_secs=300,
) as session:
    result = session.generate_text("Hello from AX text helper", max_output_tokens=32)

print(result.prompt_text)
print(result.output_text)
```

For the repo-owned MLX runtime, enable `mlx=True` and point the model artifact
path at the local MLX-ready model directory:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    mlx=True,
    mlx_model_artifacts_dir="/absolute/path/to/mlx-model-artifacts",
) as session:
    result = session.generate(input_text="Hello from MLX", max_output_tokens=32)

print(result.output_text)
```

Use `mlx=True` for repo-owned MLX inference, `support_tier="mlx_lm_delegated"`
for explicit upstream MLX text compatibility, or configure a llama.cpp target
for non-MLX inference.

Delegated HTTP backends default to a 30 second connect timeout and 300 second
read/write timeouts. Set `delegated_http_connect_timeout_secs`,
`delegated_http_read_timeout_secs`, and `delegated_http_write_timeout_secs`
when calling a remote or slow-starting upstream server; zero values are rejected
at session construction.

Validation failures continue to raise `ValueError`. Runtime failures are exposed
through `ax_engine.EngineError`, with narrower subclasses for
`EngineBackendError` (delegated process/HTTP/protocol failures),
`EngineInferenceError` (engine/runtime inference failures), and
`EngineStateError` (closed or concurrently streaming sessions).

If llama.cpp is backed by `llama_server_url`, the same SDK-owned contract
also supports iterator-style `stream_generate(...)`:

For llama.cpp-backed server routes, AX requests delegated stream usage metadata
when the upstream supports it so final streamed responses can report
prompt/completion token counts without local heuristics.

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    llama_server_url="http://127.0.0.1:8081",
) as session:
    for event in session.stream_generate([1, 2, 3], max_output_tokens=2):
        print(event.event, event.delta_tokens, event.response)
```

The wrapper now also exposes a thin text-first chat convenience layer.
It intentionally keeps the prompt rendering simple and explicit by flattening
messages into `role: content` lines plus a trailing `assistant:` line, rather
than claiming model-aware chat-template fidelity:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    llama_server_url="http://127.0.0.1:8081",
) as session:
    result = session.chat(
        [
            ax_engine.ChatMessage(role="system", content="You are AX."),
            {"role": "user", "content": "Say hello."},
        ],
        max_output_tokens=32,
    )

print(result.prompt_text)
print(result.output_text)
```

After `maturin develop`, you can also run the checked-in example:

```text
python examples/python/basic.py
python examples/python/chat.py
```

The preview package also exposes the underlying request lifecycle controls that
`generate(...)` uses internally:

If you want the repo-owned MLX lifecycle path specifically, opt in with
`mlx=True`:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    mlx=True,
    mlx_model_artifacts_dir="/absolute/path/to/mlx-model-artifacts",
) as session:
    request_id = session.submit([1, 2, 3], max_output_tokens=2)
    while True:
        request = session.snapshot(request_id)
        assert request is not None
        if request.state in {"finished", "cancelled", "failed"}:
            break
        step = session.step()
        print("scheduled_requests:", step.scheduled_requests)

print(request.output_tokens)
```

After `maturin develop`, you can run the stepwise example:

```text
python examples/python/stepwise.py
```

For callers that want a thin in-process streaming convenience surface without
dropping down to manual `submit/step/snapshot`, the wrapper also exposes
`stream_generate(...)`, `stream_text(...)`, and `stream_chat(...)`.
That stream is now translated directly from SDK-owned MLX events rather than
re-implementing the lifecycle in Python, and events are yielded incrementally
rather than buffered into one terminal batch:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    mlx=True,
    mlx_model_artifacts_dir="/absolute/path/to/mlx-model-artifacts",
) as session:
    for event in session.stream_generate([1, 2, 3], max_output_tokens=2):
        print(event.event, event.delta_tokens, event.response)
```

After `maturin develop`, you can run the streaming example:

```text
python examples/python/streaming.py
```

The response objects are intentionally simple preview dataclasses so evaluation
and automation code can inspect:

- generated token ids
- optional prompt/output text for llama.cpp-backed one-shot requests
- finish status
- step and TTFT metadata
- route metadata
- MLX `step()` payloads use the same SDK-owned lifecycle shape as other
  repo-owned runtime events
- backend-resolution metadata plus host / Metal-toolchain diagnostics from the
  SDK

`generate(...)` remains the blocking convenience API.
The request lifecycle methods and `stream_generate(...)` are thin SDK-backed
preview surfaces for bring-up, evaluation, and future transport work; they are
not text decoding or transport-level streaming APIs.
Those blocking extension calls now release the Python GIL while the Rust SDK
does native or delegated work, so Python-side threads can continue to make
progress around long-running generate, step, cancel, or stream polling calls.
For Phase 1, stepwise request-control methods (`submit`, `step`, `snapshot`,
`cancel`) support MLX preview plus the server-backed llama.cpp preview path
selected with `llama_server_url`. The `mlx_lm_delegated` route remains
text-generation-only for Python request control: blocking and streaming text
surfaces are supported, but stepwise lifecycle control and token-array prompts
still fail closed because AX does not own token IDs or KV state on this route.
That delegated stepwise path is still intentionally narrow, but one session can
now hold multiple active llama.cpp requests at once and `step()` aggregates
progress across all currently active delegated requests through the same
SDK-owned lifecycle shape.

## Design Rule

This package should remain a thin SDK-backed access layer.
It must not become a second runtime contract with Python-only backend
resolution, scheduler behavior, or model lifecycle rules.
