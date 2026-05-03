# Python

`python/ax_engine` is the preview Python access layer for AX Engine v4.

It is intentionally thin:

- it wraps `ax-engine-sdk`, not `ax-engine-core`
- it exposes runtime metadata from the SDK contract
- shipped local defaults now use MLX-backed compatibility unless you pass a
  `.gguf` model path or opt into native mode
- compatibility preview reuses the same SDK contract instead of inventing a
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
- explicit native bring-up artifact selection through
  `native_runtime_artifacts_dir` and `native_model_artifacts_dir` when you want
  to point the SDK-owned native path at validated local Metal and model assets
- shipped default compatibility wiring via
  `Session(..., compat_model_path=...)` for MLX-backed local model directories,
  `.gguf` routing to `llama.cpp`, plus
  `compat_backend="vllm"`, `compat_backend="mistral_rs"`, or
  `compat_backend="mlx"` for MLX-backed compatibility targets, plus
  `compat_cli_path` / `compat_model_path` for blocking `generate(...)` requests
  backed by `llama.cpp` or direct `mlx-lm`; the same `compat_server_url` path
  now also supports preview stepwise request control and `stream_generate(...)`
  and `mlx=True` now means "try MLX first, then fall back to llama.cpp"

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

If you want to force the native bring-up path to use explicit validated local
artifacts instead of SDK defaults or environment auto-detect, pass those paths
directly into `Session(...)`:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_5_9b_q4",
    native_mode=True,
    native_runtime_artifacts_dir="/absolute/path/to/build/metal",
    native_model_artifacts_dir="/absolute/path/to/native-model-artifacts",
) as session:
    runtime = session.runtime()

print(runtime.native_runtime)
print(runtime.native_model)
```

If you want a repo-owned smoke check that bootstraps a temporary virtualenv,
installs `maturin`, builds the extension, runs the checked-in examples, and
then runs both the installed-package preview tests and the Python wrapper
tests, including server-backed compatibility generate / stream coverage, use:

```text
bash scripts/check-python-preview.sh
```

## Example

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    compat_cli_path="python3",
    compat_model_path="/absolute/path/to/mlx-model",
) as session:
    runtime = session.runtime()
    result = session.generate(input_text="Hello from default MLX path", max_output_tokens=2)

print(runtime)
print(result.output_text)
```

Native preview sessions still accept pre-tokenized `input_tokens`, but they are
now opt-in through `native_mode=True` and currently allowlisted only for
`qwen3_5_9b_q4`:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_5_9b_q4",
    native_mode=True,
    native_runtime_artifacts_dir="/absolute/path/to/build/metal",
    native_model_artifacts_dir="/absolute/path/to/native-model-artifacts",
) as session:
    result = session.generate([1, 2, 3], max_output_tokens=32)

print(result.runtime.selected_backend)
print(result.output_tokens)
```

If you want the local default MLX path:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    compat_cli_path="python3",
    compat_model_path="/absolute/path/to/mlx-model",
) as session:
    result = session.generate(input_text="Hello from MLX", max_output_tokens=32)

print(result.prompt_text)
print(result.output_text)
```

If you pass a local GGUF model, AX routes the request to `llama.cpp` bypass:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    compat_model_path="/absolute/path/to/model.gguf",
) as session:
    result = session.generate(input_text="Hello from compatibility", max_output_tokens=32)

print(result.prompt_text)
print(result.output_text)
```

For callers who want the thin Python wrapper to own the text prompt wiring
directly, `generate_text(...)` is now a convenience alias over the same
compatibility-backed request contract:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    compat_backend="vllm",
    compat_server_url="http://127.0.0.1:8081",
) as session:
    result = session.generate_text("Hello from AX text helper", max_output_tokens=32)

print(result.prompt_text)
print(result.output_text)
```

For a direct local `mlx-lm` path, enable `mlx=True`, point the primary target at
the local MLX model directory, and also provide a `llama.cpp` fallback:

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    mlx=True,
    compat_cli_path="python3",
    compat_model_path="/absolute/path/to/mlx-model",
    llama_fallback_cli_path="llama-cli",
    llama_fallback_model_path="/absolute/path/to/model.gguf",
) as session:
    result = session.generate(input_text="Hello from MLX", max_output_tokens=32)

print(result.output_text)
```

That direct MLX CLI fallback is a blocking one-shot path. AX forwards the
request text as a raw prompt and disables the tokenizer chat template so the
CLI route does not silently rewrite the prompt shape. Because `mlx_lm.generate`
does not report authoritative token usage, AX keeps prompt/completion token
counts absent instead of fabricating heuristic values.

If compatibility is backed by `compat_server_url`, the same SDK-owned contract
also supports iterator-style `stream_generate(...)`:

For MLX-backed OpenAI-compatible servers, AX also requests delegated stream
usage metadata when the upstream supports it so final streamed responses can
report prompt/completion token counts without local heuristics.

```python
import ax_engine

with ax_engine.Session(
    model_id="qwen3_dense",
    compat_backend="mistral_rs",
    compat_server_url="http://127.0.0.1:8081",
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
    compat_backend="mistral_rs",
    compat_server_url="http://127.0.0.1:8081",
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

If you want the AX-native lifecycle path specifically, opt in with
`native_mode=True`:

```python
import ax_engine

with ax_engine.Session(model_id="qwen3_5_9b_q4", native_mode=True) as session:
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
That stream is now translated directly from SDK-owned native events rather than
re-implementing the lifecycle in Python, and events are yielded incrementally
rather than buffered into one terminal batch:

```python
import ax_engine

with ax_engine.Session(model_id="qwen3_5_9b_q4", native_mode=True) as session:
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
- optional prompt/output text for compatibility-backed one-shot requests
- finish status
- step and TTFT metadata
- route metadata
- native `step()` payloads now also surface a compact `metal_dispatch`
  summary when the SDK session is running the AX-native Metal bring-up runner,
  so Python-side evaluation code can inspect command-buffer completion,
  pipeline/archive state, arena sizing, checksum/validation evidence, and
  whether that step used model-conditioned inputs / real model tensor buffers,
  whether the active runtime/source combination supports complete
  model-forward execution, and whether that specific step resolved decode
  logits / completed the real-model forward path, plus how many multilayer
  prefix-attention dispatches already ran natively vs. CPU-reference fallback,
  how many tokens already passed the model-side RMSNorm+QKV projection /
  o_proj+FFN continuation stages, and how many decode tokens / vocab rows were
  covered by the final logits projection path
- backend-resolution metadata plus host / Metal-toolchain diagnostics from the
  SDK, including native-runner provenance through the optional
  `native_runtime` section when AX native is active

`generate(...)` remains the blocking convenience API.
The request lifecycle methods and `stream_generate(...)` are thin SDK-backed
preview surfaces for bring-up, evaluation, and future transport work; they are
not text decoding or transport-level streaming APIs.
Those blocking extension calls now release the Python GIL while the Rust SDK
does native or delegated work, so Python-side threads can continue to make
progress around long-running generate, step, cancel, or stream polling calls.
For Phase 1, stepwise request-control methods (`submit`, `step`, `snapshot`,
`cancel`) now support native preview plus the server-backed compatibility
preview path selected with `compat_server_url`. That delegated stepwise path is
still intentionally narrow, but one session can now hold multiple active
compatibility requests at once and `step()` aggregates progress across all
currently active delegated requests through the same SDK-owned lifecycle shape
as the native preview surface. `llama.cpp`, `vLLM`, and `mistral.rs` are the
main reviewed compatibility references, and `compat_backend="mlx"` now
supports both an explicit server-backed MLX target and a blocking direct
`mlx-lm` CLI path. The CLI path remains a narrower blocking fallback rather
than a full streamed/session-native runtime.

## Design Rule

This package should remain a thin SDK-backed access layer.
It must not become a second runtime contract with Python-only backend
resolution, scheduler behavior, or model lifecycle rules.
