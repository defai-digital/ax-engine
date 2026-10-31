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

/// One full tokenization paired with per-token byte offsets into the same
/// text: `(token ids, [(start, end), ...])`. The offsets are what maps
/// rendered-prompt byte positions to token indices for semantic cut points.
pub type EncodedTokenByteOffsets = (Vec<u32>, Vec<(usize, usize)>);

/// Thin wrapper over `tokenizers::Tokenizer`. Cloned cheaply (the
/// inner tokenizer is `Arc`-shared internally by the upstream crate).
#[derive(Clone)]
pub struct EngineTokenizer {
    inner: Tokenizer,
    eos_token_id: Option<u32>,
    eos_token_ids: Vec<u32>,
    embedding_special_tokens: bool,
}

impl std::fmt::Debug for EngineTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineTokenizer")
            .field("eos_token_id", &self.eos_token_id)
            .field("eos_token_ids", &self.eos_token_ids)
            .finish()
    }
}

/// `eos_token_id` in HF configs is a number or a list of numbers (the first
/// entry is the canonical EOS); some multimodal packs nest it under
/// `text_config`. Collect every usable id, in config order, deduplicated.
fn parse_eos_token_ids(config: &serde_json::Value) -> Vec<u32> {
    fn collect(value: Option<&serde_json::Value>, ids: &mut Vec<u32>) {
        match value {
            Some(serde_json::Value::Number(number)) => {
                if let Some(id) = number.as_u64().and_then(|id| u32::try_from(id).ok())
                    && !ids.contains(&id)
                {
                    ids.push(id);
                }
            }
            Some(serde_json::Value::Array(items)) => {
                for item in items {
                    collect(Some(item), ids);
                }
            }
            _ => {}
        }
    }
    let mut ids = Vec::new();
    collect(config.get("eos_token_id"), &mut ids);
    if let Some(text_config) = config.get("text_config") {
        collect(text_config.get("eos_token_id"), &mut ids);
    }
    ids
}

/// The canonical EOS id from `config.json` only: the first usable
/// top-level id, falling back to `text_config` when the top-level field
/// is missing or unusable. A present-but-unusable field yields `None`
/// when the fallback is also missing or unusable.
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

/// Pre-tokenizer split regex shipped by the Qwen2-era tokenizers: a word is a
/// run of `\p{L}` only, so a combining mark ends the word.
const LEGACY_QWEN2_PRETOKENIZER_SPLIT: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
);

/// The regex the Qwen3.5-generation models (and the official Qwen3.5 / 3.6 /
/// 3.8 `tokenizer.json`) were trained with: letters and their combining marks
/// (`\p{M}`) stay in one word. Latin, CJK and code tokenize identically;
/// Devanagari, Thai and vowelled Arabic no longer split at every mark.
const QWEN3_PRETOKENIZER_SPLIT: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|",
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
);

/// Whether `config.json` describes a Qwen3.5-generation (or later) text model.
/// Plain `qwen3` / `qwen2` and the earlier `qwen3_next` keep their own regex.
fn is_qwen3_pretokenizer_family(config: &serde_json::Value) -> bool {
    let model_types = [
        config.get("model_type"),
        config
            .get("text_config")
            .and_then(|text| text.get("model_type")),
    ];
    model_types
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .any(|model_type| model_type.starts_with("qwen3_5") || model_type.starts_with("qwen4"))
}

/// Swap the legacy Qwen2 split regex for the Qwen3 one inside raw
/// `tokenizer.json` text. Matches the whole quoted JSON string, so any other
/// regex (including a tokenizer that already carries the Qwen3 one) is left
/// untouched. Returns `None` when nothing needed rewriting.
fn restore_qwen3_pretokenizer(tokenizer_json: &str) -> Option<String> {
    let legacy = serde_json::to_string(LEGACY_QWEN2_PRETOKENIZER_SPLIT).ok()?;
    if !tokenizer_json.contains(&legacy) {
        return None;
    }
    let current = serde_json::to_string(QWEN3_PRETOKENIZER_SPLIT).ok()?;
    Some(tokenizer_json.replace(&legacy, &current))
}

/// Load `tokenizer.json`, restoring the Qwen3 pre-tokenizer regex when a
/// Qwen3.5-generation pack shipped the legacy one (the model was trained on
/// the Qwen3 regex, so the legacy split changes token ids for scripts with
/// combining marks). A rewrite that fails to load falls back to the file.
fn load_tokenizer(
    tok_path: &Path,
    config: Option<&serde_json::Value>,
) -> Result<Tokenizer, EngineTokenizerError> {
    if config.is_some_and(is_qwen3_pretokenizer_family)
        && let Ok(text) = std::fs::read_to_string(tok_path)
        && let Some(fixed) = restore_qwen3_pretokenizer(&text)
        && let Ok(tokenizer) = Tokenizer::from_bytes(fixed.as_bytes())
    {
        return Ok(tokenizer);
    }
    Tokenizer::from_file(tok_path).map_err(|e| EngineTokenizerError::Load(e.to_string()))
}

impl EngineTokenizer {
    /// Load from a `tokenizer.json` file. `model_dir` is the model
    /// artifacts directory; this looks for `tokenizer.json` inside it.
    /// EOS ids are read opportunistically from `config.json` and
    /// `generation_config.json`; the canonical config EOS supports conventions
    /// such as Qwen3-Embedding's EOS-append without a separate config lookup.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, EngineTokenizerError> {
        let tok_path = model_dir.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(EngineTokenizerError::NotFound(
                tok_path.display().to_string(),
            ));
        }
        let read_config = |name: &str| {
            std::fs::read_to_string(model_dir.join(name))
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        };
        let config = read_config("config.json");
        let generation_config = read_config("generation_config.json");
        let inner = load_tokenizer(&tok_path, config.as_ref())?;
        // Canonical EOS stays config.json-only for encode/embedding;
        // generation stopping additionally merges generation_config ids.
        let eos_token_id = config.as_ref().and_then(parse_eos_token_id);
        let mut eos_token_ids = config.as_ref().map(parse_eos_token_ids).unwrap_or_default();
        if let Some(generation_config) = generation_config.as_ref() {
            for id in parse_eos_token_ids(generation_config) {
                if !eos_token_ids.contains(&id) {
                    eos_token_ids.push(id);
                }
            }
        }
        let embedding_special_tokens = config
            .as_ref()
            .and_then(|config| config.get("model_type"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|kind| {
                matches!(
                    kind,
                    "gemma3_text" | "embeddinggemma" | "embedding_gemma2" | "embedding_gemma2_text"
                )
            });

        Ok(Self {
            inner,
            eos_token_id,
            eos_token_ids,
            embedding_special_tokens,
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
    /// keyed by the model directory and modification times of the tokenizer
    /// and config files. Parsing a large `tokenizer.json` costs tens to hundreds of
    /// milliseconds; request-scoped callers (the HTTP server loads a
    /// tokenizer per request at several sites) should use this so only the
    /// first request per model pays the parse. The returned value is a cheap
    /// clone (the inner tokenizer is `Arc`-shared).
    pub fn from_model_dir_cached(model_dir: &Path) -> Result<Self, EngineTokenizerError> {
        // The EOS ids come from config.json and generation_config.json, so
        // all three files' mtimes identify the cached value; a config-only
        // rewrite must not serve a stale EOS.
        type CacheKey = (
            PathBuf,
            Option<SystemTime>,
            Option<SystemTime>,
            Option<SystemTime>,
        );
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
        let generation_config_mtime = std::fs::metadata(canonical.join("generation_config.json"))
            .and_then(|m| m.modified())
            .ok();
        let cache_key = (canonical, mtime, config_mtime, generation_config_mtime);

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
        let newer_exists = cache
            .iter()
            .any(|((path, tok_mtime, cfg_mtime, gen_mtime), _)| {
                path == &cache_key.0
                    && (tok_mtime, cfg_mtime, gen_mtime)
                        > (&cache_key.1, &cache_key.2, &cache_key.3)
            });
        if newer_exists {
            return Ok(loaded);
        }
        cache.retain(|(path, _, _, _), _| path != &cache_key.0);
        cache.insert(cache_key, loaded.clone());
        Ok(loaded)
    }

    /// The model's canonical EOS token id if known, taken from
    /// `config.json` only. Qwen3-Embedding inputs are conventionally
    /// terminated with this token before pooling on the last position.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Every configured EOS token id, in config order and deduplicated,
    /// merging `config.json` and `generation_config.json`. Generation
    /// should stop on any of these, not only the canonical one.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Encode for the model's embedding contract: Gemma uses its tokenizer
    /// postprocessor (including BOS); Qwen uses plain text followed by EOS.
    pub fn encode_embedding(&self, text: &str) -> Result<Vec<u32>, EngineTokenizerError> {
        if self.embedding_special_tokens {
            self.encode_with_special_tokens(text, true)
        } else {
            self.encode(text, true)
        }
    }

    /// Batched equivalent of [`Self::encode_embedding`].
    pub fn encode_embedding_batch(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<u32>>, EngineTokenizerError> {
        if !self.embedding_special_tokens {
            return self.encode_batch(texts, true);
        }
        self.inner
            .encode_batch(texts.to_vec(), true)
            .map(|encodings| {
                encodings
                    .into_iter()
                    .map(|encoding| encoding.get_ids().to_vec())
                    .collect()
            })
            .map_err(|error| EngineTokenizerError::Encode(error.to_string()))
    }

    /// Apply embedding special tokens after a document has been chunked.
    /// `add_eos=None` selects the model default; an explicit value overrides it.
    pub fn prepare_embedding_chunk(
        &self,
        mut ids: Vec<u32>,
        add_eos: Option<bool>,
    ) -> Result<Vec<u32>, EngineTokenizerError> {
        if self.embedding_special_tokens {
            let encoding = tokenizers::Encoding::from_tokens(
                ids.into_iter()
                    .map(|id| tokenizers::Token::new(id, String::new(), (0, 0)))
                    .collect(),
                0,
            );
            ids = self
                .inner
                .post_process(encoding, None, true)
                .map_err(|error| EngineTokenizerError::Encode(error.to_string()))?
                .get_ids()
                .to_vec();
        }
        if let Some(eos) = self.eos_token_id() {
            match add_eos {
                Some(false) if self.embedding_special_tokens && ids.last() == Some(&eos) => {
                    ids.pop();
                }
                Some(true) if ids.last() != Some(&eos) => ids.push(eos),
                None if !self.embedding_special_tokens => ids.push(eos),
                _ => {}
            }
        }
        Ok(ids)
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
            if let Some(eos) = self.eos_token_id() {
                ids.push(eos);
            }
        }
        Ok(ids)
    }

    /// Encode `text` once and return token byte offsets into `text`.
    ///
    /// Callers that need prompt positions (semantic snapshot cut points)
    /// must map through this full tokenization instead of summing
    /// per-message encode lengths: template headers, special tokens and
    /// whitespace merges make per-message lengths non-additive.
    ///
    /// `Tokenizer::encode` already returns **byte** offsets; the char-index
    /// variant is the separate `encode_char_offsets` API. Re-deriving byte
    /// offsets from a char table here would shift every span in multi-byte
    /// text and place the cut on the wrong token (or drop the hint).
    pub fn encode_with_byte_offsets(
        &self,
        text: &str,
    ) -> Result<EncodedTokenByteOffsets, EngineTokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| EngineTokenizerError::Encode(e.to_string()))?;
        let ids = encoding.get_ids().to_vec();
        let text_len = text.len();
        let byte_offsets = encoding
            .get_offsets()
            .iter()
            .map(|(start, end)| ((*start).min(text_len), (*end).min(text_len)))
            .collect();
        Ok((ids, byte_offsets))
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
                if let Some(eos) = self.eos_token_id() {
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
    fn embedding_tokenization_preserves_gemma_bos_and_qwen_eos_for_each_chunk() {
        use tokenizers::processors::template::TemplateProcessing;

        let mut inner =
            Tokenizer::from_bytes(tokenizer_json_with_split(QWEN3_PRETOKENIZER_SPLIT).as_bytes())
                .expect("valid embedding tokenizer fixture");
        inner.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<bos> $A")
                .expect("valid embedding tokenizer fixture")
                .special_tokens(vec![("<bos>", 1)])
                .build()
                .expect("valid embedding tokenizer fixture"),
        ));
        let mut tokenizer = EngineTokenizer {
            inner,
            eos_token_id: Some(2),
            eos_token_ids: vec![2],
            embedding_special_tokens: true,
        };
        assert_eq!(
            tokenizer
                .encode_embedding("hello")
                .expect("valid embedding tokenizer fixture"),
            vec![1, 0]
        );
        assert_eq!(
            tokenizer
                .encode_embedding_batch(&["hello", "world"])
                .expect("valid embedding tokenizer fixture"),
            vec![vec![1, 0], vec![1, 0]]
        );
        for chunk in [vec![3, 4], vec![5]] {
            let mut expected = vec![1];
            expected.extend(&chunk);
            assert_eq!(
                tokenizer
                    .prepare_embedding_chunk(chunk.clone(), None)
                    .expect("valid embedding tokenizer fixture"),
                expected
            );
            expected.push(2);
            assert_eq!(
                tokenizer
                    .prepare_embedding_chunk(chunk, Some(true))
                    .expect("valid embedding tokenizer fixture"),
                expected
            );
        }
        assert_eq!(
            tokenizer
                .encode("hello", true)
                .expect("valid embedding tokenizer fixture"),
            vec![0, 2]
        );
        tokenizer.inner.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<bos> $A <eos>")
                .expect("Gemma template")
                .special_tokens(vec![("<bos>", 1), ("<eos>", 2)])
                .build()
                .expect("Gemma postprocessor"),
        ));
        assert_eq!(
            tokenizer.encode_embedding("hello").expect("Gemma text"),
            vec![1, 0, 2]
        );
        assert_eq!(
            tokenizer
                .encode_embedding_batch(&["hello"])
                .expect("Gemma batch"),
            vec![vec![1, 0, 2]]
        );
        for add_eos in [None, Some(true)] {
            assert_eq!(
                tokenizer
                    .prepare_embedding_chunk(vec![3], add_eos)
                    .expect("Gemma chunk"),
                vec![1, 3, 2]
            );
        }
        assert_eq!(
            tokenizer
                .prepare_embedding_chunk(vec![3], Some(false))
                .expect("Gemma EOS override"),
            vec![1, 3]
        );
        tokenizer.embedding_special_tokens = false;
        assert_eq!(
            tokenizer
                .encode_embedding("hello")
                .expect("valid embedding tokenizer fixture"),
            vec![0, 2]
        );
        assert_eq!(
            tokenizer
                .prepare_embedding_chunk(vec![3], None)
                .expect("valid embedding tokenizer fixture"),
            vec![3, 2]
        );
        assert_eq!(
            tokenizer
                .prepare_embedding_chunk(vec![3], Some(false))
                .expect("valid embedding tokenizer fixture"),
            vec![3]
        );
    }

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

    #[test]
    fn eos_token_ids_collects_all_scalar_array_and_text_config_forms() {
        use super::parse_eos_token_ids;
        assert_eq!(
            parse_eos_token_ids(&serde_json::json!({"eos_token_id": 128001})),
            vec![128001]
        );
        // Llama 3 style: <|end_of_text|> plus <|eot_id|> must all stop.
        assert_eq!(
            parse_eos_token_ids(&serde_json::json!({"eos_token_id": [128001, 128008, 128009]})),
            vec![128001, 128008, 128009]
        );
        assert_eq!(
            parse_eos_token_ids(&serde_json::json!({"text_config": {"eos_token_id": [1, 106]}})),
            vec![1, 106]
        );
        // Duplicates across top level and text_config collapse, order kept.
        assert_eq!(
            parse_eos_token_ids(&serde_json::json!({
                "eos_token_id": [1, 106, 1],
                "text_config": {"eos_token_id": [106, 50]}
            })),
            vec![1, 106, 50]
        );
        assert!(parse_eos_token_ids(&serde_json::json!({"eos_token_id": "eos"})).is_empty());
        assert!(parse_eos_token_ids(&serde_json::json!({})).is_empty());
    }

    #[test]
    fn from_model_dir_merges_generation_config_eos_and_cache_tracks_it() {
        let dir = std::env::temp_dir().join(format!(
            "ax-tokenizer-eos-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(
            dir.join("tokenizer.json"),
            tokenizer_json_with_split(QWEN3_PRETOKENIZER_SPLIT),
        )
        .expect("write tokenizer");
        std::fs::write(
            dir.join("config.json"),
            r#"{"eos_token_id":[128001,128008]}"#,
        )
        .expect("write config");

        let loaded = EngineTokenizer::from_model_dir(&dir).expect("load tokenizer");
        assert_eq!(loaded.eos_token_id(), Some(128001));
        assert_eq!(loaded.eos_token_ids(), &[128001, 128008]);

        // Adding generation_config.json must be visible through the cache.
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"eos_token_id":[128009,128008]}"#,
        )
        .expect("write generation config");
        let merged = EngineTokenizer::from_model_dir_cached(&dir).expect("cached load");
        assert_eq!(merged.eos_token_id(), Some(128001));
        assert_eq!(merged.eos_token_ids(), &[128001, 128008, 128009]);

        // Rewriting generation_config.json must not serve the stale entry.
        let gen_path = dir.join("generation_config.json");
        let previous_mtime = std::fs::metadata(&gen_path)
            .and_then(|meta| meta.modified())
            .expect("generation config mtime");
        for _ in 0..500 {
            std::fs::write(&gen_path, r#"{"eos_token_id":128010}"#)
                .expect("rewrite generation config");
            let mtime = std::fs::metadata(&gen_path)
                .and_then(|meta| meta.modified())
                .expect("generation config mtime");
            if mtime != previous_mtime {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let refreshed = EngineTokenizer::from_model_dir_cached(&dir).expect("cached reload");
        assert_eq!(refreshed.eos_token_ids(), &[128001, 128008, 128010]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generation_config_only_eos_does_not_change_canonical_eos_or_encode() {
        let dir = std::env::temp_dir().join(format!(
            "ax-tokenizer-gen-only-eos-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(
            dir.join("tokenizer.json"),
            tokenizer_json_with_split(QWEN3_PRETOKENIZER_SPLIT),
        )
        .expect("write tokenizer");
        // config.json carries no EOS at all; the only EOS lives in
        // generation_config.json.
        std::fs::write(dir.join("config.json"), r#"{}"#).expect("write config");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"eos_token_id":128009}"#,
        )
        .expect("write generation config");

        let loaded = EngineTokenizer::from_model_dir(&dir).expect("load tokenizer");
        // Canonical EOS stays config.json-only: absent there, so None, and
        // encode(..., true) must not append the generation-only id.
        assert_eq!(loaded.eos_token_id(), None);
        assert_eq!(
            loaded.encode("hello", true).expect("encode with eos"),
            vec![0]
        );
        assert_eq!(
            loaded.encode("hello", false).expect("encode without eos"),
            vec![0]
        );
        // Generation stopping still sees the generation_config EOS.
        assert_eq!(loaded.eos_token_ids(), &[128009]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    use super::*;
    use std::path::Path;

    /// Minimal `tokenizer.json` whose pre-tokenizer is the given split regex;
    /// the word-level model is irrelevant to the split boundaries under test.
    fn tokenizer_json_with_split(regex: &str) -> String {
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated", "invert": false},
                    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
                ]
            },
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}
        })
        .to_string()
    }

    fn split_count(tokenizer: &Tokenizer, text: &str) -> usize {
        use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};
        let mut pretokenized = PreTokenizedString::from(text);
        tokenizer
            .get_pre_tokenizer()
            .expect("pre-tokenizer")
            .pre_tokenize(&mut pretokenized)
            .expect("pre-tokenize");
        pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .len()
    }

    #[test]
    fn qwen3_regex_keeps_combining_marks_in_one_word() {
        let legacy_json = tokenizer_json_with_split(LEGACY_QWEN2_PRETOKENIZER_SPLIT);
        let fixed_json = restore_qwen3_pretokenizer(&legacy_json).expect("legacy regex rewritten");
        assert!(!fixed_json.contains(
            &serde_json::to_string(LEGACY_QWEN2_PRETOKENIZER_SPLIT).expect("json string")
        ));

        let legacy = Tokenizer::from_bytes(legacy_json.as_bytes()).expect("legacy tokenizer");
        let fixed = Tokenizer::from_bytes(fixed_json.as_bytes()).expect("fixed tokenizer");

        // Hindi and Thai words carry Mn/Mc marks: the legacy regex splits at
        // every mark, the Qwen3 regex keeps each word whole.
        let hindi = "नमस्ते दुनिया";
        assert_eq!(split_count(&fixed, hindi), 2);
        assert!(split_count(&legacy, hindi) > 2);
        let thai = "สวัสดีครับ";
        assert_eq!(split_count(&fixed, thai), 1);
        assert!(split_count(&legacy, thai) > 1);

        // Latin, CJK, digits and code split identically under both.
        for text in [
            "Hello, world! It's 2026.",
            "你好，世界",
            "fn main() { println!(\"hi\"); }",
            "line one\n\nline two",
        ] {
            assert_eq!(
                split_count(&fixed, text),
                split_count(&legacy, text),
                "split boundaries changed for {text:?}"
            );
        }
    }

    #[test]
    fn restore_leaves_other_regexes_and_already_fixed_tokenizers_alone() {
        assert!(restore_qwen3_pretokenizer("{}").is_none());
        let other = tokenizer_json_with_split(r"\w+");
        assert!(restore_qwen3_pretokenizer(&other).is_none());
        let already = tokenizer_json_with_split(QWEN3_PRETOKENIZER_SPLIT);
        assert!(restore_qwen3_pretokenizer(&already).is_none());
    }

    #[test]
    fn byte_offsets_stay_bytes_for_cjk_and_emoji_before_the_assistant_turn() {
        // A CJK/emoji history turn before the assistant marker is the case
        // that broke when byte offsets were remapped through a char table:
        // every span shifted forward, so the cut landed on an earlier token.
        let json = tokenizer_json_with_split(QWEN3_PRETOKENIZER_SPLIT);
        let raw = Tokenizer::from_bytes(json.as_bytes()).expect("tokenizer");
        let tokenizer = EngineTokenizer {
            inner: raw.clone(),
            eos_token_id: None,
            eos_token_ids: Vec::new(),
            embedding_special_tokens: false,
        };
        let text = "用户：请用三句话解释哈希表。🙂\n<|im_start|>assistant\n好的。<|im_end|>\n<|im_start|>user\n再来一个例子。<|im_start|>assistant\n";
        assert!(
            text.len() > text.chars().count() + 8,
            "the fixture must carry multi-byte characters"
        );

        let (ids, offsets) = tokenizer
            .encode_with_byte_offsets(text)
            .expect("byte offsets");
        assert_eq!(ids, tokenizer.encode(text, false).expect("plain encode"));
        for (start, end) in &offsets {
            assert!(start <= end, "descending offset {start}..{end}");
            assert!(
                text.get(*start..*end).is_some(),
                "offset {start}..{end} is not a byte range of the text"
            );
        }

        // Oracle: converting the tokenizer's own *char* offsets once gives
        // the same spans as using its byte offsets directly.
        let char_encoded = raw
            .encode_char_offsets(text, false)
            .expect("char-offset encode");
        let mut char_to_byte: Vec<usize> = text.char_indices().map(|(byte, _)| byte).collect();
        char_to_byte.push(text.len());
        let expected: Vec<(usize, usize)> = char_encoded
            .get_offsets()
            .iter()
            .map(|(start, end)| (char_to_byte[*start], char_to_byte[*end]))
            .collect();
        assert_eq!(offsets, expected);

        // The assistant turn's first byte maps to the token consuming it.
        // Which token that is depends on the vocabulary: the small test
        // fixture keeps `<|im_start|>` whole, while a real vocabulary splits
        // it into `<|`, `im_start`, `|>`. Both are correct cuts, so assert
        // the invariant the caller relies on (the header starts the chosen
        // token) instead of one vocabulary's segmentation.
        let turn_start = text.find("<|im_start|>assistant").expect("assistant turn");
        let cut = offsets
            .iter()
            .position(|(_, end)| *end > turn_start)
            .expect("a token covers the turn start");
        assert!(offsets[cut].0 <= turn_start);
        assert!(offsets[cut].1 > turn_start);
        assert_eq!(
            offsets[cut].0,
            turn_start,
            "the cut token must start at the assistant header, got {:?}",
            &text[offsets[cut].0..offsets[cut].1]
        );
        assert!(
            text[offsets[cut].0..].starts_with("<|"),
            "the cut token must open the assistant header, got {:?}",
            &text[offsets[cut].0..offsets[cut].1]
        );
    }

    #[test]
    fn pretokenizer_family_gate_is_limited_to_qwen3_5_and_later() {
        let family = |config: serde_json::Value| is_qwen3_pretokenizer_family(&config);
        assert!(family(serde_json::json!({"model_type": "qwen3_5"})));
        assert!(family(serde_json::json!({
            "model_type": "qwen3_5_moe",
            "text_config": {"model_type": "qwen3_5_moe_text"}
        })));
        assert!(family(
            serde_json::json!({"text_config": {"model_type": "qwen3_5_text"}})
        ));
        assert!(!family(serde_json::json!({"model_type": "qwen3_next"})));
        assert!(family(serde_json::json!({"model_type": "qwen4_exp"})));
        assert!(!family(serde_json::json!({"model_type": "qwen3"})));
        assert!(!family(serde_json::json!({"model_type": "qwen2"})));
        assert!(!family(serde_json::json!({"model_type": "gemma4"})));
        assert!(!family(serde_json::json!({})));
    }

    #[test]
    fn load_tokenizer_restores_regex_only_for_qwen3_5_family() {
        let dir = std::env::temp_dir().join(format!(
            "ax-tokenizer-regex-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let tok_path = dir.join("tokenizer.json");
        std::fs::write(
            &tok_path,
            tokenizer_json_with_split(LEGACY_QWEN2_PRETOKENIZER_SPLIT),
        )
        .expect("write tokenizer");
        let hindi = "नमस्ते दुनिया";

        let qwen35 = load_tokenizer(
            &tok_path,
            Some(&serde_json::json!({"model_type": "qwen3_5"})),
        )
        .expect("load");
        assert_eq!(split_count(&qwen35, hindi), 2);

        let qwen3 = load_tokenizer(&tok_path, Some(&serde_json::json!({"model_type": "qwen3"})))
            .expect("load");
        assert!(split_count(&qwen3, hindi) > 2);

        let unknown = load_tokenizer(&tok_path, None).expect("load");
        assert!(split_count(&unknown, hindi) > 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

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
