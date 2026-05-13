//! Optional tokenizer integration for the embedding path.
//!
//! Enabled by the `tokenizer` feature. Wraps HuggingFace's `tokenizers`
//! crate so a Rust caller can go from `&str` directly to `Vec<u32>` token
//! IDs without bringing their own tokenizer. The tokenization is plain
//! CPU work and runs independently of the GPU forward pass, so callers
//! that want to overlap tokenize + embed can run them on separate threads
//! or `tokio` tasks (see `crates/ax-engine-bench/examples/embed_rust_bench.rs`).

use std::path::Path;

use thiserror::Error;
use tokenizers::Tokenizer;

/// Errors from the tokenizer wrapper. Mirrors `tokenizers::Error`
/// without leaking that type through the public API (callers shouldn't
/// have to depend on `tokenizers` directly).
#[derive(Debug, Error)]
pub enum EngineTokenizerError {
    #[error("tokenizer file not found: {0}")]
    NotFound(String),
    #[error("failed to load tokenizer.json: {0}")]
    Load(String),
    #[error("tokenization failed: {0}")]
    Encode(String),
}

/// Thin wrapper over `tokenizers::Tokenizer`. Cloned cheaply (the
/// inner tokenizer is `Arc`-shared internally by the upstream crate).
#[derive(Clone)]
pub struct EngineTokenizer {
    inner: Tokenizer,
    eos_token_id: Option<u32>,
}

impl std::fmt::Debug for EngineTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineTokenizer")
            .field("eos_token_id", &self.eos_token_id)
            .finish()
    }
}

impl EngineTokenizer {
    /// Load from a `tokenizer.json` file. `model_dir` is the model
    /// artifacts directory; this looks for `tokenizer.json` inside it.
    /// The model's `config.json` is read opportunistically to extract
    /// `eos_token_id` so callers that want Qwen3-Embedding's EOS-append
    /// convention get it without a separate config lookup.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, EngineTokenizerError> {
        let tok_path = model_dir.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(EngineTokenizerError::NotFound(
                tok_path.display().to_string(),
            ));
        }
        let inner = Tokenizer::from_file(&tok_path)
            .map_err(|e| EngineTokenizerError::Load(e.to_string()))?;

        let eos_token_id = std::fs::read_to_string(model_dir.join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.get("eos_token_id").and_then(|x| x.as_u64()))
            .map(|id| id as u32);

        Ok(Self {
            inner,
            eos_token_id,
        })
    }

    /// The model's EOS token id if known, taken from `config.json`.
    /// Qwen3-Embedding inputs are conventionally terminated with this
    /// token before pooling on the last position.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Encode a single string. When `add_eos` is true and the model has
    /// an EOS id, it is appended (Qwen3-Embedding convention).
    pub fn encode(
        &self,
        text: &str,
        add_eos: bool,
    ) -> Result<Vec<u32>, EngineTokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| EngineTokenizerError::Encode(e.to_string()))?;
        let mut ids = encoding.get_ids().to_vec();
        if add_eos {
            if let Some(eos) = self.eos_token_id {
                ids.push(eos);
            }
        }
        Ok(ids)
    }

    /// Encode a batch. Uses the upstream crate's `encode_batch` so
    /// individual sequence tokenization is parallelised internally.
    /// Returns one `Vec<u32>` per input string, in input order.
    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_eos: bool,
    ) -> Result<Vec<Vec<u32>>, EngineTokenizerError> {
        let owned: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();
        let encodings = self
            .inner
            .encode_batch(owned, false)
            .map_err(|e| EngineTokenizerError::Encode(e.to_string()))?;
        let mut out = Vec::with_capacity(encodings.len());
        for enc in encodings {
            let mut ids = enc.get_ids().to_vec();
            if add_eos {
                if let Some(eos) = self.eos_token_id {
                    ids.push(eos);
                }
            }
            out.push(ids);
        }
        Ok(out)
    }
}
