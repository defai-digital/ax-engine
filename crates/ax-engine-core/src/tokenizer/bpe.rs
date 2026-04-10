use std::collections::HashMap;

use super::vocab::Vocab;

/// BPE encoding dispatcher. Uses merge-based BPE for GPT-2 models,
/// score-based BPE for SentencePiece models.
pub fn bpe_encode(vocab: &Vocab, text: &str) -> Vec<u32> {
    if vocab.merge_ranks.is_some() {
        gpt2_bpe_encode(vocab, text)
    } else {
        spm_bpe_encode_ex(vocab, text, vocab.add_space_prefix)
    }
}

/// Encode a text fragment without the automatic space prefix.
///
/// Used for encoding text segments between special tokens, where the
/// SentencePiece `add_dummy_prefix` should not apply.
pub fn bpe_encode_fragment(vocab: &Vocab, text: &str) -> Vec<u32> {
    if vocab.merge_ranks.is_some() {
        gpt2_bpe_encode(vocab, text)
    } else {
        spm_bpe_encode_ex(vocab, text, false)
    }
}

// ---------------------------------------------------------------------------
// GPT-2 merge-based BPE
// ---------------------------------------------------------------------------

/// GPT-2's bytes_to_unicode mapping: maps all 256 byte values to unique
/// visible Unicode characters. This is used to convert raw bytes to tokens
/// that can participate in BPE merging.
fn byte_to_gpt2_char(b: u8) -> char {
    let cp: u32 = match b {
        // Printable ASCII and Latin-1 supplement map to themselves
        0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => b as u32,
        // All other bytes (0x00-0x20, 0x7F-0xA0, 0xAD) map to U+0100+
        _ => {
            // Count how many "non-printable" bytes come before this one
            let idx = (0u8..=255)
                .filter(|&x| {
                    !((0x21..=0x7E).contains(&x)
                        || (0xA1..=0xAC).contains(&x)
                        || (0xAE..=0xFF).contains(&x))
                })
                .position(|x| x == b)
                .unwrap_or(0);
            0x100 + idx as u32
        }
    };
    char::from_u32(cp).unwrap_or('?')
}

/// Build the byte-to-GPT2-char lookup table (cached on first call).
fn gpt2_byte_table() -> &'static [char; 256] {
    static TABLE: std::sync::OnceLock<[char; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = ['\0'; 256];
        for b in 0u8..=255 {
            table[b as usize] = byte_to_gpt2_char(b);
        }
        table
    })
}

/// Encode text using GPT-2 merge-based BPE.
///
/// 1. Convert input bytes to GPT-2's unicode representation
/// 2. Split on whitespace boundaries (each word gets its own BPE)
/// 3. Apply merge-based BPE using the ordered merge list
fn gpt2_bpe_encode(vocab: &Vocab, text: &str) -> Vec<u32> {
    if text.is_empty() {
        return Vec::new();
    }

    let merge_ranks = vocab.merge_ranks.as_ref().unwrap();
    let byte_table = gpt2_byte_table();

    // Convert raw bytes to GPT-2 unicode characters
    let gpt2_text: String = text.bytes().map(|b| byte_table[b as usize]).collect();

    // GPT-2 tokenizes each "word" (whitespace-delimited chunk) separately.
    // Spaces attach to the following word as a prefix, matching the GPT-2 regex:
    // r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    // For simplicity, we split on space boundaries keeping space with the next word.
    let words = split_gpt2_words(&gpt2_text);

    let mut result = Vec::new();
    for word in &words {
        if word.is_empty() {
            continue;
        }
        let word_tokens = gpt2_bpe_word(vocab, merge_ranks, word);
        result.extend(word_tokens);
    }

    result
}

/// Split GPT-2 encoded text into words for BPE.
/// Whitespace characters (space, newline, tab in GPT-2 encoding) attach to
/// the following word, matching the GPT-2 tokenizer's word boundary logic.
fn split_gpt2_words(text: &str) -> Vec<String> {
    let space_char = byte_to_gpt2_char(b' '); // Ġ (U+0120)
    let newline_char = byte_to_gpt2_char(b'\n'); // Ċ (U+010A)
    let tab_char = byte_to_gpt2_char(b'\t'); // ĉ (U+0109)
    let mut words = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if (ch == space_char || ch == newline_char || ch == tab_char) && !current.is_empty() {
            words.push(current);
            current = String::new();
        }
        current.push(ch);
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// Apply merge-based BPE to a single word.
fn gpt2_bpe_word(vocab: &Vocab, merge_ranks: &HashMap<String, u32>, word: &str) -> Vec<u32> {
    // Initialize: each character is a separate symbol
    let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

    if symbols.len() <= 1 {
        // Single character — look up directly
        return symbols
            .iter()
            .map(|s| vocab.token_to_id(s).unwrap_or(vocab.unk_id))
            .collect();
    }

    // Scratch buffer for pair lookup — reused every iteration to avoid allocation.
    let mut pair_key = String::with_capacity(64);

    loop {
        if symbols.len() < 2 {
            break;
        }

        // Find the pair with the lowest merge rank (highest priority)
        let mut best_rank = u32::MAX;
        let mut best_idx = usize::MAX;

        for i in 0..symbols.len() - 1 {
            pair_key.clear();
            pair_key.push_str(&symbols[i]);
            pair_key.push('\x01');
            pair_key.push_str(&symbols[i + 1]);
            if let Some(&rank) = merge_ranks.get(&pair_key)
                && rank < best_rank
            {
                best_rank = rank;
                best_idx = i;
            }
        }

        if best_idx == usize::MAX {
            break;
        }

        // Merge the best pair
        let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
        symbols[best_idx] = merged;
        symbols.remove(best_idx + 1);
    }

    // Convert symbol strings to token IDs
    symbols
        .iter()
        .map(|s| vocab.token_to_id(s).unwrap_or(vocab.unk_id))
        .collect()
}

// ---------------------------------------------------------------------------
// SentencePiece score-based BPE
// ---------------------------------------------------------------------------

/// SentencePiece-style BPE encoding.
///
/// This implements the "score-based" BPE used by LLaMA/SentencePiece models:
/// 1. Start with each UTF-8 byte or character as an individual token
/// 2. Repeatedly merge the pair with the highest score
/// 3. Stop when no more merges are possible
fn spm_bpe_encode_ex(vocab: &Vocab, text: &str, add_space_prefix: bool) -> Vec<u32> {
    if text.is_empty() {
        return Vec::new();
    }

    // Step 1: SentencePiece normalization (SPM / "llama" model type).
    // SentencePiece uses ▁ (U+2581) to represent word boundaries. All spaces
    // must be replaced with ▁ so that BPE merges produce tokens like "▁the",
    // "▁world" that match the model's vocabulary. Without this, spaces stay as
    // token 270 (" ") and words never merge with their ▁ prefix.
    let has_spiece_marker =
        vocab.model_type == "llama" || vocab.token_to_id.contains_key("\u{2581}");
    let text = if has_spiece_marker {
        let mut s = if add_space_prefix {
            format!(" {text}")
        } else {
            text.to_string()
        };
        s = s.replace(' ', "\u{2581}");
        s
    } else {
        text.to_string()
    };

    let mut symbols: Vec<Symbol> = Vec::new();

    for ch in text.chars() {
        let ch_str = ch.to_string();
        if let Some(id) = vocab.token_to_id(&ch_str) {
            symbols.push(Symbol {
                token_id: id,
                text: ch_str,
            });
        } else {
            // Fall back to byte-level tokens
            for byte in ch_str.as_bytes() {
                let byte_tok = format!("<0x{byte:02X}>");
                let id = vocab.token_to_id(&byte_tok).unwrap_or(vocab.unk_id);
                symbols.push(Symbol {
                    token_id: id,
                    text: byte_tok,
                });
            }
        }
    }

    // Step 2: Iteratively merge the best pair until no merges remain.
    // This is O(n² × m) where n = number of symbols and m = iterations.
    // For typical prompt lengths (<4K chars) this is fast enough.
    loop {
        if symbols.len() < 2 {
            break;
        }

        // Find the best merge: the adjacent pair whose merged token has the highest score.
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = usize::MAX;
        let mut best_id = 0u32;

        for i in 0..symbols.len() - 1 {
            let merged_text = format!("{}{}", symbols[i].text, symbols[i + 1].text);
            if let Some(id) = vocab.token_to_id(&merged_text) {
                let score = vocab.score(id);
                if score > best_score {
                    best_score = score;
                    best_idx = i;
                    best_id = id;
                }
            }
        }

        // No more merges possible
        if best_idx == usize::MAX {
            break;
        }

        // Apply the merge: combine symbols[best_idx] and symbols[best_idx+1]
        let merged_text = format!("{}{}", symbols[best_idx].text, symbols[best_idx + 1].text);
        symbols[best_idx] = Symbol {
            token_id: best_id,
            text: merged_text,
        };
        symbols.remove(best_idx + 1);
    }

    symbols.into_iter().map(|s| s.token_id).collect()
}

/// A symbol in the BPE merge process.
struct Symbol {
    token_id: u32,
    text: String,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::tokenizer::vocab::TokenType;

    /// Build a minimal test vocab for BPE testing.
    fn make_bpe_vocab() -> Vocab {
        // Tokens: individual chars + some merged tokens with scores
        let token_defs: Vec<(&str, f32, TokenType)> = vec![
            ("<unk>", 0.0, TokenType::Unknown), // 0
            ("<s>", 0.0, TokenType::Control),   // 1
            ("</s>", 0.0, TokenType::Control),  // 2
            ("h", -1.0, TokenType::Normal),     // 3
            ("e", -2.0, TokenType::Normal),     // 4
            ("l", -3.0, TokenType::Normal),     // 5
            ("o", -4.0, TokenType::Normal),     // 6
            ("he", -0.5, TokenType::Normal),    // 7  (merge h+e, higher score)
            ("ll", -0.6, TokenType::Normal),    // 8  (merge l+l)
            ("lo", -0.7, TokenType::Normal),    // 9  (merge l+o)
            ("hel", -0.3, TokenType::Normal),   // 10 (merge he+l)
            ("hell", -0.2, TokenType::Normal),  // 11 (merge hel+l)
            ("hello", -0.1, TokenType::Normal), // 12 (merge hell+o)
        ];

        let tokens: Vec<String> = token_defs.iter().map(|(t, _, _)| t.to_string()).collect();
        let scores: Vec<f32> = token_defs.iter().map(|(_, s, _)| *s).collect();
        let types: Vec<TokenType> = token_defs.iter().map(|(_, _, t)| *t).collect();

        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        Vocab {
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
        }
    }

    #[test]
    fn test_bpe_empty() {
        let vocab = make_bpe_vocab();
        assert!(bpe_encode(&vocab, "").is_empty());
    }

    #[test]
    fn test_bpe_single_char() {
        let vocab = make_bpe_vocab();
        let tokens = bpe_encode(&vocab, "h");
        assert_eq!(tokens, vec![3]); // "h"
    }

    #[test]
    fn test_bpe_merges_to_single_token() {
        let vocab = make_bpe_vocab();
        // "hello" should merge all the way to token 12
        let tokens = bpe_encode(&vocab, "hello");
        assert_eq!(tokens, vec![12]); // "hello"
    }

    #[test]
    fn test_bpe_partial_merge() {
        let vocab = make_bpe_vocab();
        // "he" should merge to token 7
        let tokens = bpe_encode(&vocab, "he");
        assert_eq!(tokens, vec![7]); // "he"
    }

    #[test]
    fn test_bpe_no_merge_possible() {
        let vocab = make_bpe_vocab();
        // "oh" — "o" and "h" individually, no "oh" token exists
        let tokens = bpe_encode(&vocab, "oh");
        assert_eq!(tokens, vec![6, 3]); // "o", "h"
    }

    #[test]
    fn test_bpe_byte_fallback() {
        let mut vocab = make_bpe_vocab();
        // Add a byte token for 'A' (0x41)
        let id = vocab.tokens.len() as u32;
        vocab.tokens.push("<0x41>".to_string());
        vocab.scores.push(0.0);
        vocab.types.push(TokenType::Byte);
        vocab.token_to_id.insert("<0x41>".to_string(), id);

        // 'A' is not in vocab as a char, so should fall back to byte token
        let tokens = bpe_encode(&vocab, "A");
        assert_eq!(tokens, vec![id]);
    }

    // -----------------------------------------------------------------------
    // GPT-2 merge-based BPE tests
    // -----------------------------------------------------------------------

    /// Build a GPT-2-style vocab with merge ranks for testing.
    ///
    /// Merge order matters: for "hello" = [h,e,l,l,o], we need merges in an
    /// order where l+o happens before l+l, so that the chain produces:
    ///   [h,e,l,l,o] → [h,e,l,lo] → [he,l,lo] → [hel,lo] → [hello]
    fn make_gpt2_vocab() -> Vocab {
        let space_char = "\u{0120}"; // Ġ = GPT-2's encoding of space byte

        // Build tokens list — all tokens that appear in vocab
        let space_w = format!("{space_char}w");
        let space_world = format!("{space_char}world");
        let space_str = space_char.to_string();

        let mut tokens: Vec<String> = vec![
            "<unk>", "<s>", "</s>", "h", "e", "l", "o", "w", "r", "d", // single chars (3-9)
            "lo", "he", "hel", "hello", // hello merges (10-13)
            "or", "ld", "wor", "orld", "world", // world merges (14-18)
        ]
        .into_iter()
        .map(String::from)
        .collect();
        tokens.push(space_str.clone()); // 19: Ġ
        tokens.push(space_w.clone()); // 20: Ġw
        tokens.push(space_world.clone()); // 21: Ġworld

        let n = tokens.len();
        let types = vec![TokenType::Normal; n];
        let scores = vec![0.0f32; n];

        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        // Merge list: order determines priority (lower rank = higher priority).
        // Keys use "\x01" separator: "a\x01b" represents the pair (a, b).
        // For "hello": l+o first, then h+e, then he+l, then hel+lo
        let mut merge_ranks = HashMap::new();
        merge_ranks.insert("l\x01o".into(), 0); // l+o → lo
        merge_ranks.insert("h\x01e".into(), 1); // h+e → he
        merge_ranks.insert("he\x01l".into(), 2); // he+l → hel
        merge_ranks.insert("hel\x01lo".into(), 3); // hel+lo → hello
        merge_ranks.insert("o\x01r".into(), 4); // o+r → or
        merge_ranks.insert("l\x01d".into(), 5); // l+d → ld
        merge_ranks.insert("w\x01or".into(), 6); // w+or → wor
        merge_ranks.insert("or\x01ld".into(), 7); // or+ld → orld
        merge_ranks.insert("wor\x01ld".into(), 8); // wor+ld → world
        merge_ranks.insert(format!("{space_str}\x01world"), 9); // Ġ+world → Ġworld

        Vocab {
            tokens,
            scores,
            types,
            token_to_id,
            merge_ranks: Some(merge_ranks),
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            add_bos: false,
            add_eos: false,
            add_space_prefix: false,
            model_type: "gpt2".to_string(),
            eot_id: None,
        }
    }

    #[test]
    fn test_gpt2_bpe_empty() {
        let vocab = make_gpt2_vocab();
        assert!(bpe_encode(&vocab, "").is_empty());
    }

    #[test]
    fn test_gpt2_bpe_single_char() {
        let vocab = make_gpt2_vocab();
        let tokens = bpe_encode(&vocab, "h");
        // "h" = token ID 3
        assert_eq!(tokens, vec![vocab.token_to_id("h").unwrap()]);
    }

    #[test]
    fn test_gpt2_bpe_hello() {
        let vocab = make_gpt2_vocab();
        // "hello" → merge chain: l+o→lo, h+e→he, he+l→hel, hel+lo→hello
        let tokens = bpe_encode(&vocab, "hello");
        assert_eq!(tokens, vec![vocab.token_to_id("hello").unwrap()]);
    }

    #[test]
    fn test_gpt2_bpe_with_space() {
        let vocab = make_gpt2_vocab();
        // "hello world" → two words: "hello" + "Ġworld"
        let tokens = bpe_encode(&vocab, "hello world");
        let hello_id = vocab.token_to_id("hello").unwrap();
        let space_world_id = vocab.token_to_id("\u{0120}world").unwrap();
        assert_eq!(tokens, vec![hello_id, space_world_id]);
    }

    #[test]
    fn test_gpt2_byte_mapping() {
        // Verify the byte-to-char mapping for key values
        assert_eq!(byte_to_gpt2_char(b'A'), 'A'); // Printable ASCII
        assert_eq!(byte_to_gpt2_char(b' '), '\u{0120}'); // Space → Ġ
        assert_eq!(byte_to_gpt2_char(b'\n'), '\u{010A}'); // Newline → Ċ
    }
}
