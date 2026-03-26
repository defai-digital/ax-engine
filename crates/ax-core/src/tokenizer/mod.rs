pub mod bpe;
pub mod vocab;

pub use vocab::Vocab;

use crate::chat::gguf_chat_template;
use crate::gguf::GgufHeader;

/// Tokenizer backed by GGUF-embedded vocabulary.
///
/// Supports SentencePiece BPE (LLaMA-family) tokenization.
/// Extracts vocab, scores, and special tokens from GGUF metadata.
pub struct Tokenizer {
    vocab: Vocab,
    chat_template: Option<String>,
}

impl Tokenizer {
    /// Create a tokenizer from GGUF header metadata.
    pub fn from_gguf(header: &GgufHeader) -> anyhow::Result<Self> {
        let vocab = Vocab::from_gguf(header)?;
        Ok(Self {
            vocab,
            chat_template: gguf_chat_template(header).map(str::to_string),
        })
    }

    /// Create a tokenizer from a pre-built vocabulary.
    pub fn from_vocab(vocab: Vocab) -> Self {
        Self {
            vocab,
            chat_template: None,
        }
    }

    /// Encode text to token IDs.
    ///
    /// Recognizes special/control tokens (e.g. `<start_of_turn>`, `<end_of_turn>`)
    /// as single tokens rather than splitting them through BPE. This is required
    /// for chat templates to work correctly.
    ///
    /// If `add_special` is true, prepends BOS and/or appends EOS based on
    /// the model's configuration.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        self.encode_with_options(text, add_special, true)
    }

    /// Encode text to token IDs with explicit special-token parsing control.
    ///
    /// When `parse_special` is false, control token strings such as
    /// `<start_of_turn>` are treated as ordinary text and passed through
    /// the model tokenizer instead of being recognized as single token IDs.
    pub fn encode_with_options(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Vec<u32> {
        let mut tokens = if parse_special {
            self.encode_with_special_tokens(text)
        } else {
            let mut vocab = self.vocab.clone();
            let special_ids: Vec<u32> = vocab
                .types
                .iter()
                .enumerate()
                .filter_map(|(id, ty)| match ty {
                    vocab::TokenType::Control
                    | vocab::TokenType::Unknown
                    | vocab::TokenType::Unused => Some(id as u32),
                    _ => None,
                })
                .collect();
            vocab
                .token_to_id
                .retain(|_, &mut id| !special_ids.contains(&id));
            bpe::bpe_encode(&vocab, text)
        };

        if add_special {
            if self.vocab.add_bos {
                tokens.insert(0, self.vocab.bos_id);
            }
            if self.vocab.add_eos {
                tokens.push(self.vocab.eos_id);
            }
        }

        tokens
    }

    /// Encode text, recognizing control tokens as single tokens.
    ///
    /// Splits input at special token boundaries, BPE-encodes text fragments,
    /// and inserts special token IDs directly for recognized control tokens.
    fn encode_with_special_tokens(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Fast path: if no angle brackets in text, no special tokens possible.
        if !text.contains('<') {
            return bpe::bpe_encode(&self.vocab, text);
        }

        // Build list of control token strings sorted longest-first for greedy matching.
        // Filter out <unused*> placeholder tokens for performance.
        let mut special_tokens: Vec<(&str, u32)> = Vec::new();
        for (id, tok_str) in self.vocab.tokens.iter().enumerate() {
            let id = id as u32;
            if self.vocab.is_special(id) && tok_str.len() > 2 && !tok_str.starts_with("<unused") {
                special_tokens.push((tok_str.as_str(), id));
            }
        }

        if special_tokens.is_empty() {
            return bpe::bpe_encode(&self.vocab, text);
        }

        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut result = Vec::new();
        let mut remaining = text;
        // Space prefix only applies to the first text segment at the start of input.
        let mut emitted_any = false;

        while !remaining.is_empty() {
            // Try to match a special token at the current position.
            let mut matched = false;
            for &(tok_str, tok_id) in &special_tokens {
                if remaining.starts_with(tok_str) {
                    result.push(tok_id);
                    remaining = &remaining[tok_str.len()..];
                    matched = true;
                    emitted_any = true;
                    break;
                }
            }

            if !matched {
                // Find the earliest occurrence of any special token.
                let mut next_pos = remaining.len();
                for &(tok_str, _) in &special_tokens {
                    if let Some(pos) = remaining.find(tok_str) {
                        next_pos = next_pos.min(pos);
                    }
                }

                // BPE encode the text segment before the next special token.
                let segment = &remaining[..next_pos];
                if !segment.is_empty() {
                    if !emitted_any {
                        // First segment at start of input: use normal BPE (with space prefix).
                        result.extend(bpe::bpe_encode(&self.vocab, segment));
                    } else {
                        // Fragment after a special token: skip space prefix.
                        result.extend(bpe::bpe_encode_fragment(&self.vocab, segment));
                    }
                    emitted_any = true;
                }
                remaining = &remaining[next_pos..];
            }
        }

        result
    }

    /// Decode a sequence of token IDs to a string.
    ///
    /// Handles SentencePiece `▁` → space conversion, byte token reassembly,
    /// and special token skipping.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut output = String::new();
        let mut pending_bytes: Vec<u8> = Vec::new();

        for &id in tokens {
            // Skip special tokens in output
            if self.vocab.is_special(id) {
                // Flush any pending bytes before special tokens
                flush_bytes(&mut pending_bytes, &mut output);
                continue;
            }

            if let Some(byte_val) = self.vocab.byte_value(id) {
                // Byte-level token: accumulate for UTF-8 reassembly
                pending_bytes.push(byte_val);
            } else {
                // Regular token: flush pending bytes first, then append
                flush_bytes(&mut pending_bytes, &mut output);
                if let Some(text) = self.vocab.id_to_token(id) {
                    if self.vocab.model_type == "gpt2" {
                        output.push_str(&gpt2_decode(text));
                    } else {
                        // Replace SentencePiece space marker with actual space
                        output.push_str(&text.replace('▁', " "));
                    }
                }
            }
        }

        // Flush remaining bytes
        flush_bytes(&mut pending_bytes, &mut output);

        output
    }

    /// Decode a single token ID to its text representation (raw vocab lookup).
    pub fn decode_token(&self, id: u32) -> Option<&str> {
        self.vocab.id_to_token(id)
    }

    /// Render a single token for streaming output.
    ///
    /// Handles SentencePiece `▁` → space conversion, GPT-2 byte decoding,
    /// and byte tokens like `<0x0A>` → actual byte values. Returns `None`
    /// for special tokens (BOS/EOS) that shouldn't be printed.
    pub fn render_token(&self, id: u32) -> Option<String> {
        if self.vocab.is_special(id) {
            return None;
        }
        if let Some(byte_val) = self.vocab.byte_value(id) {
            return Some(String::from_utf8_lossy(&[byte_val]).into_owned());
        }
        self.vocab.id_to_token(id).map(|s| {
            if self.vocab.model_type == "gpt2" {
                gpt2_decode(s)
            } else {
                s.replace('▁', " ")
            }
        })
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get the BOS token ID.
    pub fn bos_id(&self) -> u32 {
        self.vocab.bos_id
    }

    /// Get the EOS token ID.
    pub fn eos_id(&self) -> u32 {
        self.vocab.eos_id
    }

    /// Check if a token is an end-of-sequence token (EOS or end-of-turn).
    pub fn is_eos(&self, id: u32) -> bool {
        id == self.vocab.eos_id || self.vocab.eot_id == Some(id)
    }

    /// Get a reference to the underlying vocabulary.
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// Return the raw GGUF chat template string if present.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }
}

/// Decode a GPT-2 byte-encoded token string to its raw bytes.
///
/// GPT-2 BPE maps all 256 byte values to visible Unicode characters:
/// printable ASCII (0x21-0x7E) and Latin-1 (0xA1-0xAC, 0xAE-0xFF) map to themselves,
/// all other bytes (0x00-0x20, 0x7F-0xA0, 0xAD) map to U+0100 onwards.
/// E.g. space (0x20) → Ġ (U+0120), newline (0x0A) → Ċ (U+010A).
fn gpt2_decode(encoded: &str) -> String {
    // Build the non-printable byte list matching GPT-2's bytes_to_unicode() order.
    let non_printable: Vec<u8> = (0u8..=255)
        .filter(|&b| {
            !((0x21..=0x7E).contains(&b)
                || (0xA1..=0xAC).contains(&b)
                || (0xAE..=0xFF).contains(&b))
        })
        .collect();

    let bytes: Vec<u8> = encoded
        .chars()
        .map(|c| {
            let cp = c as u32;
            if (0x21..=0x7E).contains(&cp)
                || (0xA1..=0xAC).contains(&cp)
                || (0xAE..=0xFF).contains(&cp)
            {
                cp as u8
            } else if cp >= 0x100 {
                let idx = (cp - 0x100) as usize;
                non_printable.get(idx).copied().unwrap_or(b'?')
            } else {
                // Shouldn't happen for valid GPT-2 tokens
                b'?'
            }
        })
        .collect();

    String::from_utf8_lossy(&bytes).into_owned()
}

/// Flush accumulated byte tokens as UTF-8 (or lossy replacement).
fn flush_bytes(pending: &mut Vec<u8>, output: &mut String) {
    if pending.is_empty() {
        return;
    }
    match std::str::from_utf8(pending) {
        Ok(s) => output.push_str(s),
        Err(_) => output.push_str(&String::from_utf8_lossy(pending)),
    }
    pending.clear();
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::tokenizer::vocab::TokenType;

    fn make_test_tokenizer() -> Tokenizer {
        let token_defs: Vec<(&str, f32, TokenType)> = vec![
            ("<unk>", 0.0, TokenType::Unknown),  // 0
            ("<s>", 0.0, TokenType::Control),    // 1 (BOS)
            ("</s>", 0.0, TokenType::Control),   // 2 (EOS)
            ("▁", -10.0, TokenType::Normal),     // 3 (space replacement)
            ("H", -5.0, TokenType::Normal),      // 4
            ("e", -5.0, TokenType::Normal),      // 5
            ("l", -5.0, TokenType::Normal),      // 6
            ("o", -5.0, TokenType::Normal),      // 7
            ("He", -3.0, TokenType::Normal),     // 8
            ("ll", -3.0, TokenType::Normal),     // 9
            ("lo", -3.0, TokenType::Normal),     // 10
            ("Hel", -2.0, TokenType::Normal),    // 11
            ("Hello", -1.0, TokenType::Normal),  // 12
            ("▁w", -2.0, TokenType::Normal),     // 13
            ("or", -3.0, TokenType::Normal),     // 14
            ("orld", -1.5, TokenType::Normal),   // 15
            ("▁world", -0.5, TokenType::Normal), // 16
            ("w", -5.0, TokenType::Normal),      // 17
            ("r", -5.0, TokenType::Normal),      // 18
            ("d", -5.0, TokenType::Normal),      // 19
            ("<0x41>", 0.0, TokenType::Byte),    // 20 ('A')
            ("<0x42>", 0.0, TokenType::Byte),    // 21 ('B')
        ];

        let tokens: Vec<String> = token_defs.iter().map(|(t, _, _)| t.to_string()).collect();
        let scores: Vec<f32> = token_defs.iter().map(|(_, s, _)| *s).collect();
        let types: Vec<TokenType> = token_defs.iter().map(|(_, _, t)| *t).collect();

        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let vocab = Vocab {
            tokens,
            scores,
            types,
            token_to_id,
            merge_ranks: None,
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            add_bos: true,
            add_eos: false,
            add_space_prefix: false,
            model_type: "llama".to_string(),
            eot_id: None,
        };

        Tokenizer::from_vocab(vocab)
    }

    #[test]
    fn test_encode_basic() {
        let tok = make_test_tokenizer();
        // "Hello" should merge to single token 12
        let ids = tok.encode("Hello", false);
        assert_eq!(ids, vec![12]);
    }

    #[test]
    fn test_encode_with_bos() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("Hello", true);
        assert_eq!(ids[0], 1); // BOS
        assert_eq!(ids[1], 12); // Hello
    }

    #[test]
    fn test_decode_basic() {
        let tok = make_test_tokenizer();
        let text = tok.decode(&[12]); // "Hello"
        assert_eq!(text, "Hello");
    }

    #[test]
    fn test_decode_skips_special() {
        let tok = make_test_tokenizer();
        // BOS + "Hello" + EOS → should skip BOS and EOS
        let text = tok.decode(&[1, 12, 2]);
        assert_eq!(text, "Hello");
    }

    #[test]
    fn test_decode_byte_tokens() {
        let tok = make_test_tokenizer();
        // Byte tokens for 'A' and 'B'
        let text = tok.decode(&[20, 21]);
        assert_eq!(text, "AB");
    }

    #[test]
    fn test_decode_mixed() {
        let tok = make_test_tokenizer();
        // "Hello" + byte 'A'
        let text = tok.decode(&[12, 20]);
        assert_eq!(text, "HelloA");
    }

    #[test]
    fn test_encode_empty() {
        let tok = make_test_tokenizer();
        assert!(tok.encode("", false).is_empty());
    }

    #[test]
    fn test_encode_with_bos_empty() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("", true);
        assert_eq!(ids, vec![1]); // BOS only
    }

    #[test]
    fn test_roundtrip() {
        let tok = make_test_tokenizer();
        let text = "Hello";
        let ids = tok.encode(text, false);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_vocab_size() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.vocab_size(), 22);
    }

    #[test]
    fn test_special_token_ids() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.bos_id(), 1);
        assert_eq!(tok.eos_id(), 2);
        assert!(tok.is_eos(2));
        assert!(!tok.is_eos(3));
    }

    #[test]
    fn test_render_token_space_replacement() {
        let tok = make_test_tokenizer();
        // Token 16 = "▁world" → " world"
        let text = tok.render_token(16).unwrap();
        assert_eq!(text, " world");
    }

    #[test]
    fn test_render_token_byte() {
        let tok = make_test_tokenizer();
        // Token 20 = "<0x41>" → 'A'
        let text = tok.render_token(20).unwrap();
        assert_eq!(text, "A");
    }

    #[test]
    fn test_render_token_special_returns_none() {
        let tok = make_test_tokenizer();
        assert!(tok.render_token(1).is_none()); // BOS
        assert!(tok.render_token(2).is_none()); // EOS
    }

    #[test]
    fn test_decode_with_space_tokens() {
        let tok = make_test_tokenizer();
        // "Hello world" = token 12 ("Hello") + token 16 ("▁world")
        let text = tok.decode(&[12, 16]);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_encode_space_to_spiece_marker() {
        let tok = make_test_tokenizer();
        // "Hello world": space between words becomes ▁, producing ▁w (token 13)
        // instead of a bare space token. BPE greedily merges what it can.
        let ids = tok.encode("Hello world", false);
        // Verify: first token is "Hello", and ▁w (13) appears (space merged with 'w')
        assert_eq!(ids[0], 12); // "Hello"
        assert_eq!(ids[1], 13); // "▁w" — space became ▁ and merged with w
        // No bare space character should appear
        assert!(
            !ids.contains(&3),
            "bare ▁ token should not appear — it should merge with next char"
        );
    }

    #[test]
    fn test_encode_with_space_prefix() {
        // Build a tokenizer with add_space_prefix=true and full merge chain
        let token_defs: Vec<(&str, f32, TokenType)> = vec![
            ("<unk>", 0.0, TokenType::Unknown),  // 0
            ("<s>", 0.0, TokenType::Control),    // 1 (BOS)
            ("</s>", 0.0, TokenType::Control),   // 2 (EOS)
            ("▁", -10.0, TokenType::Normal),     // 3
            ("H", -5.0, TokenType::Normal),      // 4
            ("e", -5.0, TokenType::Normal),      // 5
            ("l", -5.0, TokenType::Normal),      // 6
            ("o", -5.0, TokenType::Normal),      // 7
            ("He", -3.0, TokenType::Normal),     // 8
            ("ll", -3.0, TokenType::Normal),     // 9
            ("lo", -3.0, TokenType::Normal),     // 10
            ("Hel", -2.0, TokenType::Normal),    // 11
            ("Hello", -1.0, TokenType::Normal),  // 12
            ("▁H", -2.0, TokenType::Normal),     // 13
            ("▁He", -1.5, TokenType::Normal),    // 14
            ("▁Hel", -1.0, TokenType::Normal),   // 15
            ("▁Hell", -0.8, TokenType::Normal),  // 16
            ("Hell", -0.9, TokenType::Normal),   // 17
            ("▁Hello", -0.5, TokenType::Normal), // 18
        ];

        let tokens: Vec<String> = token_defs.iter().map(|(t, _, _)| t.to_string()).collect();
        let scores: Vec<f32> = token_defs.iter().map(|(_, s, _)| *s).collect();
        let types: Vec<TokenType> = token_defs.iter().map(|(_, _, t)| *t).collect();

        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let vocab = Vocab {
            tokens,
            scores,
            types,
            token_to_id,
            merge_ranks: None,
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            add_bos: false,
            add_eos: false,
            add_space_prefix: true,
            model_type: "llama".to_string(),
            eot_id: None,
        };

        let tok = Tokenizer::from_vocab(vocab);
        // With add_space_prefix, "Hello" → " Hello" → "▁Hello" → token 18
        let ids = tok.encode("Hello", false);
        assert_eq!(ids, vec![18]); // "▁Hello"
    }

    /// Build a tokenizer with Gemma3-like special tokens for testing chat templates.
    fn make_gemma_tokenizer() -> Tokenizer {
        let token_defs: Vec<(&str, f32, TokenType)> = vec![
            ("<pad>", 0.0, TokenType::Control),           // 0
            ("<eos>", 0.0, TokenType::Control),           // 1
            ("<bos>", 0.0, TokenType::Control),           // 2
            ("<unk>", 0.0, TokenType::Control),           // 3
            ("u", -5.0, TokenType::Normal),               // 4
            ("s", -5.0, TokenType::Normal),               // 5
            ("e", -5.0, TokenType::Normal),               // 6
            ("r", -5.0, TokenType::Normal),               // 7
            ("us", -3.0, TokenType::Normal),              // 8
            ("er", -3.0, TokenType::Normal),              // 9
            ("user", -1.0, TokenType::Normal),            // 10
            ("\n", -5.0, TokenType::Normal),              // 11
            ("H", -5.0, TokenType::Normal),               // 12
            ("i", -5.0, TokenType::Normal),               // 13
            ("Hi", -2.0, TokenType::Normal),              // 14
            ("m", -5.0, TokenType::Normal),               // 15
            ("o", -5.0, TokenType::Normal),               // 16
            ("d", -5.0, TokenType::Normal),               // 17
            ("l", -5.0, TokenType::Normal),               // 18
            ("mod", -2.0, TokenType::Normal),             // 19
            ("model", -1.0, TokenType::Normal),           // 20
            ("<start_of_turn>", 0.0, TokenType::Control), // 21
            ("<end_of_turn>", 0.0, TokenType::Control),   // 22
        ];

        let tokens: Vec<String> = token_defs.iter().map(|(t, _, _)| t.to_string()).collect();
        let scores: Vec<f32> = token_defs.iter().map(|(_, s, _)| *s).collect();
        let types: Vec<TokenType> = token_defs.iter().map(|(_, _, t)| *t).collect();

        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let vocab = Vocab {
            tokens,
            scores,
            types,
            token_to_id,
            merge_ranks: None,
            bos_id: 2,
            eos_id: 1,
            unk_id: 3,
            add_bos: false,
            add_eos: false,
            add_space_prefix: false,
            model_type: "llama".to_string(),
            eot_id: Some(22),
        };

        Tokenizer::from_vocab(vocab)
    }

    #[test]
    fn test_encode_special_token_recognized() {
        let tok = make_gemma_tokenizer();
        // <start_of_turn> should be recognized as a single token (ID 21)
        let ids = tok.encode("<start_of_turn>", false);
        assert_eq!(ids, vec![21]);
    }

    #[test]
    fn test_encode_special_token_with_text() {
        let tok = make_gemma_tokenizer();
        // <start_of_turn>user should be [21, 10]
        let ids = tok.encode("<start_of_turn>user", false);
        assert_eq!(ids, vec![21, 10]);
    }

    #[test]
    fn test_encode_without_parsing_special_tokens_treats_control_text_literally() {
        let tok = make_gemma_tokenizer();
        let ids = tok.encode_with_options("<start_of_turn>user", false, false);
        assert!(!ids.is_empty());
        assert_ne!(ids[0], 21);
        assert!(!ids.contains(&21));
    }

    #[test]
    fn test_encode_chat_template() {
        let tok = make_gemma_tokenizer();
        // Simulate a Gemma3 chat template:
        // <start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n
        let template = "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n";
        let ids = tok.encode(template, false);
        // Verify special tokens are recognized as single IDs
        assert_eq!(ids[0], 21); // <start_of_turn>
        assert_eq!(ids[1], 10); // user
        assert_eq!(ids[2], 11); // \n
        assert_eq!(ids[3], 14); // Hi
        assert_eq!(ids[4], 22); // <end_of_turn>
        assert_eq!(ids[5], 11); // \n
        assert_eq!(ids[6], 21); // <start_of_turn>
        // "model" is split into chars (m,o,d,e,l) since test vocab lacks
        // intermediate merge tokens — the important thing is the special
        // tokens above are recognized correctly.
        let last = *ids.last().unwrap();
        assert_eq!(last, 11); // \n at the end
    }

    #[test]
    fn test_encode_with_bos_and_special_tokens() {
        let tok = make_gemma_tokenizer();
        // <bos><start_of_turn>user\nHi should start with BOS when add_special=false
        // because <bos> is recognized as special token ID 2
        let ids = tok.encode("<bos><start_of_turn>user\nHi", false);
        assert_eq!(ids, vec![2, 21, 10, 11, 14]);
    }

    #[test]
    fn test_encode_no_special_tokens_fast_path() {
        let tok = make_gemma_tokenizer();
        // Text without angle brackets should use fast path
        let ids = tok.encode("Hi", false);
        assert_eq!(ids, vec![14]);
    }

    #[test]
    fn test_is_eos_with_eot() {
        let tok = make_gemma_tokenizer();
        assert!(tok.is_eos(1)); // <eos>
        assert!(tok.is_eos(22)); // <end_of_turn> (eot_id)
        assert!(!tok.is_eos(21)); // <start_of_turn> is NOT eos
    }
}
