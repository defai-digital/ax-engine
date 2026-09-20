//! Optional tokenizer integration for the embedding path.
//!
//! Enabled by the `tokenizer` feature. Wraps HuggingFace's `tokenizers`
//! crate so a Rust caller can go from `&str` directly to `Vec<u32>` token
//! IDs without bringing their own tokenizer. The tokenization is plain
//! CPU work and runs independently of the GPU forward pass, so callers
//! that want to overlap tokenize + embed can run them on separate threads
//! or `tokio` tasks (see `crates/ax-engine-bench/examples/embed_rust_bench.rs`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

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
    #[error("token decode failed: {0}")]
    Decode(String),
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

/// `eos_token_id` in HF configs is a number or a list of numbers (the first
/// entry is the canonical EOS); some multimodal packs nest it under
/// `text_config`. A present-but-unusable field yields `None`.
fn parse_eos_token_id(config: &serde_json::Value) -> Option<u32> {
    fn from_value(value: &serde_json::Value) -> Option<u32> {
        match value {
            serde_json::Value::Number(number) => {
                number.as_u64().and_then(|id| u32::try_from(id).ok())
            }
            serde_json::Value::Array(items) => items.first().and_then(from_value),
            _ => None,
        }
    }
    config.get("eos_token_id").and_then(from_value).or_else(|| {
        config
            .get("text_config")
            .and_then(|text| text.get("eos_token_id"))
            .and_then(from_value)
    })
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
            .and_then(|v| parse_eos_token_id(&v));

        Ok(Self {
            inner,
            eos_token_id,
        })
    }

    /// Eagerly populate the process-level tokenizer cache for `model_dir`.
    ///
    /// Call this at model load so first-request TTFT under a fresh process does
    /// not include the multi-tens-of-milliseconds `tokenizer.json` parse.
    pub fn prewarm_model_dir(model_dir: &Path) -> Result<(), EngineTokenizerError> {
        Self::from_model_dir_cached(model_dir).map(|_| ())
    }

    /// Like [`Self::from_model_dir`], but backed by a process-level cache
    /// keyed by the model directory and the `tokenizer.json` modification
    /// time. Parsing a large `tokenizer.json` costs tens to hundreds of
    /// milliseconds; request-scoped callers (the HTTP server loads a
    /// tokenizer per request at several sites) should use this so only the
    /// first request per model pays the parse. The returned value is a cheap
    /// clone (the inner tokenizer is `Arc`-shared).
    pub fn from_model_dir_cached(model_dir: &Path) -> Result<Self, EngineTokenizerError> {
        // The EOS id comes from config.json, so both files' mtimes identify
        // the cached value; a config-only rewrite must not serve a stale EOS.
        type CacheKey = (PathBuf, Option<SystemTime>, Option<SystemTime>);
        static CACHE: Mutex<Option<HashMap<CacheKey, EngineTokenizer>>> = Mutex::new(None);

        let canonical = model_dir
            .canonicalize()
            .unwrap_or_else(|_| model_dir.to_path_buf());
        let mtime = std::fs::metadata(canonical.join("tokenizer.json"))
            .and_then(|m| m.modified())
            .ok();
        let config_mtime = std::fs::metadata(canonical.join("config.json"))
            .and_then(|m| m.modified())
            .ok();
        let cache_key = (canonical, mtime, config_mtime);

        // A panic elsewhere while holding this lock must not permanently
        // poison the shared tokenizer cache for every later request; recover
        // the last-known-good map instead of propagating the poison.
        if let Some(hit) = CACHE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_or_insert_with(HashMap::new)
            .get(&cache_key)
        {
            return Ok(hit.clone());
        }

        // Parse outside the lock so concurrent requests for other models
        // are not serialized behind a slow load.
        let loaded = Self::from_model_dir(model_dir)?;
        let mut guard = CACHE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let cache = guard.get_or_insert_with(HashMap::new);
        // Re-check under the lock: another thread may have inserted the same
        // key while we parsed, or a newer mtime for this path. Retain+insert
        // must be atomic under the lock so two racers cannot leave two
        // entries for the same directory.
        if let Some(hit) = cache.get(&cache_key) {
            return Ok(hit.clone());
        }
        // Stale entries for the same directory (older mtime) are dropped so
        // a hot-swapped model directory does not grow the map unboundedly.
        // A slower load that snapshotted an older mtime must not clobber an
        // entry a faster racer inserted for a newer file: keep the newest.
        let newer_exists = cache.iter().any(|((path, tok_mtime, cfg_mtime), _)| {
            path == &cache_key.0 && (tok_mtime, cfg_mtime) > (&cache_key.1, &cache_key.2)
        });
        if newer_exists {
            return Ok(loaded);
        }
        cache.retain(|(path, _, _), _| path != &cache_key.0);
        cache.insert(cache_key, loaded.clone());
        Ok(loaded)
    }

    /// The model's EOS token id if known, taken from `config.json`.
    /// Qwen3-Embedding inputs are conventionally terminated with this
    /// token before pooling on the last position.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Encode a single string. When `add_eos` is true and the model has
    /// an EOS id, it is appended (Qwen3-Embedding convention).
    pub fn encode(&self, text: &str, add_eos: bool) -> Result<Vec<u32>, EngineTokenizerError> {
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

    /// Encode a single string using the tokenizer's own special-token handling.
    /// This mirrors compatibility endpoints such as llama.cpp's `/tokenize`,
    /// where `add_special` is not the same as appending the model EOS token.
    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, EngineTokenizerError> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| EngineTokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Look up the surface string for a single token id. Used to render a
    /// special placeholder token (e.g. the Gemma image soft token) into a
    /// prompt so that it re-encodes back to exactly that id.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Look up the id of a single token by its surface string. Used to find
    /// model-specific control tokens (e.g. the Gemma 4 channel markers) in
    /// generated output without hardcoding ids.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
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

    /// Decode token ids back to text. When `skip_special_tokens` is true,
    /// tokenizer-defined special tokens are omitted from the decoded string.
    pub fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, EngineTokenizerError> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| EngineTokenizerError::Decode(e.to_string()))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    #[test]
    fn eos_token_id_accepts_scalar_array_and_text_config_forms() {
        use super::parse_eos_token_id;
        let scalar = serde_json::json!({"eos_token_id": 151645});
        assert_eq!(parse_eos_token_id(&scalar), Some(151645));
        let array = serde_json::json!({"eos_token_id": [151645, 151643]});
        assert_eq!(parse_eos_token_id(&array), Some(151645));
        let nested = serde_json::json!({"text_config": {"eos_token_id": [1, 106]}});
        assert_eq!(parse_eos_token_id(&nested), Some(1));
        let unusable = serde_json::json!({"eos_token_id": "eos"});
        assert_eq!(parse_eos_token_id(&unusable), None);
        assert_eq!(parse_eos_token_id(&serde_json::json!({})), None);
    }

    use super::*;
    use std::path::Path;

    #[test]
    fn prewarm_model_dir_reports_missing_tokenizer() {
        let err = EngineTokenizer::prewarm_model_dir(Path::new("/no/such/ax-engine-model-dir"))
            .expect_err("missing model dir must fail closed");
        assert!(
            matches!(err, EngineTokenizerError::NotFound(_)),
            "expected NotFound, got {err:?}"
        );
    }
}
