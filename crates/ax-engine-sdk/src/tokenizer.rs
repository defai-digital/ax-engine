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
        let config = std::fs::read_to_string(model_dir.join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        let inner = load_tokenizer(&tok_path, config.as_ref())?;
        let eos_token_id = config.as_ref().and_then(parse_eos_token_id);

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
