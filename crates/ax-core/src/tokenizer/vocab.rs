use std::collections::HashMap;

use crate::gguf::GgufHeader;

/// Token type flags from GGUF (matches llama.cpp token types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TokenType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl TokenType {
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }
}

/// Vocabulary loaded from GGUF metadata.
#[derive(Debug, Clone)]
pub struct Vocab {
    /// Token ID → token string.
    pub tokens: Vec<String>,
    /// Token ID → score (used for BPE merge priority).
    pub scores: Vec<f32>,
    /// Token ID → token type.
    pub types: Vec<TokenType>,
    /// Token string → token ID (for encoding).
    pub token_to_id: HashMap<String, u32>,
    /// Ordered merge list for GPT-2 BPE (pair → priority rank).
    /// Loaded from `tokenizer.ggml.merges`. Lower rank = higher priority.
    /// Key is `"a\x01b"` (the two sub-tokens joined by ASCII SOH) — avoids a
    /// heap-allocated tuple and allows zero-alloc lookup with a scratch buffer.
    pub merge_ranks: Option<HashMap<String, u32>>,
    /// BOS token ID.
    pub bos_id: u32,
    /// EOS token ID.
    pub eos_id: u32,
    /// Unknown token ID.
    pub unk_id: u32,
    /// Whether to automatically prepend BOS.
    pub add_bos: bool,
    /// Whether to automatically append EOS.
    pub add_eos: bool,
    /// Whether to prepend a space before tokenization (SentencePiece `add_dummy_prefix`).
    pub add_space_prefix: bool,
    /// Tokenizer model type string (e.g. "llama", "gpt2").
    pub model_type: String,
    /// End-of-turn token ID (e.g. `<end_of_turn>` in Gemma3).
    /// Used as an additional stop token during generation.
    pub eot_id: Option<u32>,
}

impl Vocab {
    /// Extract vocabulary from GGUF header metadata.
    pub fn from_gguf(header: &GgufHeader) -> anyhow::Result<Self> {
        let tokens: Vec<String> = header
            .get_str_array("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow::anyhow!("missing tokenizer.ggml.tokens in GGUF metadata"))?
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let n_vocab = tokens.len();

        let scores = header
            .get_f32_array("tokenizer.ggml.scores")
            .unwrap_or_else(|| {
                // If no scores, default to 0.0 for all tokens
                vec![0.0; n_vocab]
            });

        let types: Vec<TokenType> = header
            .get_i32_array("tokenizer.ggml.token_type")
            .map(|arr| arr.into_iter().map(TokenType::from_i32).collect())
            .unwrap_or_else(|| vec![TokenType::Normal; n_vocab]);

        // Build reverse lookup
        let mut token_to_id = HashMap::with_capacity(n_vocab);
        for (id, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), id as u32);
        }

        let bos_id = header.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_id = header.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);
        let unk_id = header
            .get_u32("tokenizer.ggml.unknown_token_id")
            .unwrap_or(0);
        let add_bos = header
            .get_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(true);
        let add_eos = header
            .get_bool("tokenizer.ggml.add_eos_token")
            .unwrap_or(false);
        // SentencePiece add_dummy_prefix: prepend space before tokenization.
        // Defaults to true for SPM ("llama") models, false otherwise.
        let model_type = header
            .get_str("tokenizer.ggml.model")
            .unwrap_or("llama")
            .to_string();
        let add_space_prefix = header
            .get_bool("tokenizer.ggml.add_space_prefix")
            .unwrap_or(model_type == "llama");

        // Load ordered merge list for GPT-2 BPE (if present).
        // Each entry is "token_a token_b" separated by a space.
        let merge_ranks = header.get_str_array("tokenizer.ggml.merges").map(|arr| {
            let mut ranks = HashMap::with_capacity(arr.len());
            for (rank, entry) in arr.iter().enumerate() {
                if let Some((a, b)) = entry.split_once(' ') {
                    let mut key = String::with_capacity(a.len() + 1 + b.len());
                    key.push_str(a);
                    key.push('\x01');
                    key.push_str(b);
                    ranks.insert(key, rank as u32);
                }
            }
            ranks
        });

        if let Some(ref ranks) = merge_ranks {
            tracing::info!(n_merges = ranks.len(), "loaded GPT-2 merge list from GGUF");
        }

        // End-of-turn token: try GGUF metadata first, then vocab lookup.
        let eot_id = header
            .get_u32("tokenizer.ggml.eot_token_id")
            .or_else(|| token_to_id.get("<end_of_turn>").copied());

        if scores.len() != n_vocab {
            anyhow::bail!(
                "tokenizer.ggml.scores length ({}) != tokens length ({n_vocab})",
                scores.len()
            );
        }

        tracing::info!(
            n_vocab = n_vocab,
            model_type = %model_type,
            bos_id = bos_id,
            eos_id = eos_id,
            eot_id = ?eot_id,
            add_bos = add_bos,
            add_space_prefix = add_space_prefix,
            "vocabulary loaded from GGUF"
        );

        Ok(Vocab {
            tokens,
            scores,
            types,
            token_to_id,
            merge_ranks,
            bos_id,
            eos_id,
            unk_id,
            add_bos,
            add_eos,
            add_space_prefix,
            model_type,
            eot_id,
        })
    }

    /// Number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Look up token string by ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    /// Look up token ID by string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get the score for a token ID.
    pub fn score(&self, id: u32) -> f32 {
        self.scores.get(id as usize).copied().unwrap_or(0.0)
    }

    /// Check if a token is a special/control token.
    pub fn is_special(&self, id: u32) -> bool {
        matches!(
            self.types.get(id as usize),
            Some(TokenType::Control) | Some(TokenType::Unknown)
        )
    }

    /// Check if a token is a byte-level fallback token (e.g. "<0x41>").
    pub fn is_byte_token(&self, id: u32) -> bool {
        matches!(self.types.get(id as usize), Some(TokenType::Byte))
    }

    /// Get the byte value for a byte-level fallback token.
    /// Returns None if the token is not a byte token (gated on TokenType::Byte).
    pub fn byte_value(&self, id: u32) -> Option<u8> {
        if !self.is_byte_token(id) {
            return None;
        }
        let tok = self.id_to_token(id)?;
        // Byte tokens are formatted as "<0xNN>"
        if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
            u8::from_str_radix(&tok[3..5], 16).ok()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vocab() -> Vocab {
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "hello".to_string(),
            "world".to_string(),
            "<0x41>".to_string(),
        ];
        let n = tokens.len();
        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        Vocab {
            tokens,
            scores: vec![0.0; n],
            types: vec![
                TokenType::Unknown,
                TokenType::Control,
                TokenType::Control,
                TokenType::Normal,
                TokenType::Normal,
                TokenType::Byte,
            ],
            token_to_id,
            merge_ranks: None,
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            add_bos: true,
            add_eos: false,
            add_space_prefix: true,
            model_type: "llama".to_string(),
            eot_id: None,
        }
    }

    #[test]
    fn test_vocab_lookup() {
        let vocab = make_test_vocab();
        assert_eq!(vocab.len(), 6);
        assert_eq!(vocab.id_to_token(3), Some("hello"));
        assert_eq!(vocab.token_to_id("world"), Some(4));
        assert_eq!(vocab.token_to_id("nonexistent"), None);
    }

    #[test]
    fn test_vocab_special() {
        let vocab = make_test_vocab();
        assert!(vocab.is_special(0)); // <unk>
        assert!(vocab.is_special(1)); // <s>
        assert!(vocab.is_special(2)); // </s>
        assert!(!vocab.is_special(3)); // hello
    }

    #[test]
    fn test_vocab_byte_token() {
        let vocab = make_test_vocab();
        assert!(vocab.is_byte_token(5));
        assert_eq!(vocab.byte_value(5), Some(0x41)); // 'A'
        assert!(!vocab.is_byte_token(3));
        assert_eq!(vocab.byte_value(3), None);
    }
}
