# Rust SDK

`ax-engine-sdk` is the high-level Rust integration facade for AX Engine.

Use it when you want:

- a stable Rust API instead of depending directly on `ax-engine-core` internals
- model loading, tokenization, sessions, chat rendering, and streaming through one crate
- a foundation for higher-level integrations such as Python bindings

## Scope

`ax-engine-core` remains the runtime and internal engine crate.

`ax-engine-sdk` is the intended Rust application-facing surface.

## Current Surface

- `Model::load(...)`
- `Model::info()`
- `Model::tokenize(...)`
- `Model::decode(...)`
- `Model::supports_infill()`
- `Model::render_infill_prompt(...)`
- `Model::session(...)`
- `Session::generate(...)`
- `Session::stream(...)`
- `Session::infill(...)`
- `Session::infill_with_stats(...)`
- `Session::stream_infill(...)`
- `Session::stream_infill_with_stats(...)`
- `Session::chat(...)`
- `Session::stream_chat(...)`
- `Session::reset()`

## Example

```rust
use ax_engine_sdk::{GenerationOptions, LoadOptions, Model, SessionOptions};

fn main() -> anyhow::Result<()> {
    let model = Model::load(
        "./models/Qwen3-8B-Q4_K_M.gguf",
        LoadOptions::default(),
    )?;

    let session = model.session(SessionOptions::default())?;
    let output = session.generate(
        "Explain KV cache in one short sentence.",
        GenerationOptions::default(),
    )?;

    println!("{}", output.text);
    Ok(())
}
```

## Infill

```rust
use ax_engine_sdk::{GenerationOptions, LoadOptions, Model, SessionOptions};

fn main() -> anyhow::Result<()> {
    let model = Model::load(
        "./models/Qwen3.5-9B-Q4_K_M.gguf",
        LoadOptions::default(),
    )?;
    anyhow::ensure!(model.supports_infill(), "model does not support infill");

    let session = model.session(SessionOptions::default())?;
    let (output, cache_stats) = session.infill_with_stats(
        "fn add(a: i32, b: i32) {\n",
        "\n}\n",
        GenerationOptions::default().max_tokens(32),
    )?;

    println!("{}", output.text);
    println!("cached tokens: {}", cache_stats.cached_tokens);
    Ok(())
}
```

## Relationship to Python

`ax-engine-py` builds on `ax-engine-sdk` instead of re-implementing model
load and session glue directly over `ax-engine-core`.
