# ax-engine-py

Direct Python bindings for `ax-engine` on Apple Silicon.

This crate exposes a Python module named `ax_engine` built with PyO3. It is a
local, in-process binding built on the high-level `ax-engine-sdk` facade.

## Status

Current surface:

- `Model.load(path, backend=None)`
- `Model.tokenize(text, add_special=False)`
- `Model.decode(token_ids)`
- `Model.session(ctx_size=None, seed=None)`
- `Model.close()`
- `Session.generate(...)`
- `Session.stream(...)`
- `Session.chat(...)`
- `Session.reset()`
- `Session.close()`

Current limitations:

- Apple Silicon macOS only
- no JS binding
- `stream()` currently yields buffered chunks after generation rather than live
  token-by-token streaming
- no wheel publishing pipeline yet

## Build

Use `maturin` from the repository root:

```bash
maturin develop --manifest-path crates/ax-engine-py/Cargo.toml
```

If your shell already exports `CONDA_PREFIX`, unset it when targeting a venv:

```bash
env -u CONDA_PREFIX \
  VIRTUAL_ENV=/tmp/ax-engine-py-venv \
  PATH=/tmp/ax-engine-py-venv/bin:$PATH \
  maturin develop --manifest-path crates/ax-engine-py/Cargo.toml
```

One-command smoke path from the repo root:

```bash
./scripts/test_ax_engine_py_smoke.sh
```

Optional generation smoke:

```bash
AX_ENGINE_PY_GENERATE=1 ./scripts/test_ax_engine_py_smoke.sh
```

## Example

```python
import ax_engine

with ax_engine.Model.load("./models/Qwen3-8B-Q4_K_M.gguf", backend="auto") as model:
    with model.session() as session:
        print(model.architecture)
        print(model.vocab_size)

        reply = session.generate(
            "Explain KV cache in one short sentence.",
            max_tokens=64,
            temperature=0.7,
        )
        print(reply)

        session.reset()

        reply = session.chat(
            [{"role": "user", "content": "Say hello in five words."}],
            max_tokens=16,
            temperature=0.0,
        )
        print(reply)
```
