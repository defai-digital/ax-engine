//! Per-step allowed-token masks: the byte string of every token, a byte trie
//! over them, and a walk of that trie with the grammar matcher.
//!
//! Inside a free string (no escape or UTF-8 sequence pending) every token
//! made only of unescaped printable characters is allowed outright, so the
//! walk only visits the remaining tokens. That keeps string content, the
//! common case, cheap.

use serde_json::Value;

use super::json::{JsonGrammar, JsonMatcher};

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
struct Edge {
    byte: u8,
    child: u32,
    next: u32,
}

#[derive(Clone, Copy, Debug)]
struct TrieNode {
    first_edge: u32,
    first_token: u32,
}

#[derive(Clone, Copy, Debug)]
struct TokenLink {
    token: u32,
    next: u32,
}

#[derive(Clone, Debug, Default)]
struct Trie {
    nodes: Vec<TrieNode>,
    edges: Vec<Edge>,
    tokens: Vec<TokenLink>,
}

impl Trie {
    fn new() -> Self {
        Self {
            nodes: vec![TrieNode {
                first_edge: NONE,
                first_token: NONE,
            }],
            edges: Vec::new(),
            tokens: Vec::new(),
        }
    }

    fn insert(&mut self, bytes: &[u8], token: u32) {
        let mut node = 0usize;
        for &byte in bytes {
            let mut edge = self.nodes[node].first_edge;
            while edge != NONE && self.edges[edge as usize].byte != byte {
                edge = self.edges[edge as usize].next;
            }
            node = if edge == NONE {
                let child = self.nodes.len() as u32;
                self.nodes.push(TrieNode {
                    first_edge: NONE,
                    first_token: NONE,
                });
                self.edges.push(Edge {
                    byte,
                    child,
                    next: self.nodes[node].first_edge,
                });
                self.nodes[node].first_edge = (self.edges.len() - 1) as u32;
                child as usize
            } else {
                self.edges[edge as usize].child as usize
            };
        }
        self.tokens.push(TokenLink {
            token,
            next: self.nodes[node].first_token,
        });
        self.nodes[node].first_token = (self.tokens.len() - 1) as u32;
    }
}

/// Byte strings of a model's tokens, indexed for mask computation.
#[derive(Clone, Debug)]
pub struct TokenVocab {
    vocab_size: usize,
    /// Bytes of each token; empty for tokens a mask never admits.
    bytes: Vec<Box<[u8]>>,
    full: Trie,
    /// Tokens that are not plain string text.
    special_text: Trie,
    /// Plain string text tokens as a bitset, and as `(chars, token)` sorted
    /// by character count for length-limited strings.
    plain: Vec<u64>,
    plain_by_chars: Vec<(u32, u32)>,
}

/// Reusable matcher states for the trie walk.
#[derive(Debug, Default)]
pub struct MaskWorkspace {
    states: Vec<JsonMatcher>,
}

fn byte_level_decoder_table() -> [Option<u8>; 512] {
    // Inverse of the GPT-2 `bytes_to_unicode` map.
    let mut table = [None; 512];
    let mut extra = 0u32;
    for byte in 0u32..256 {
        let printable = (33..=126).contains(&byte)
            || (161..=172).contains(&byte)
            || (174..=255).contains(&byte);
        let code = if printable {
            byte
        } else {
            extra += 1;
            255 + extra
        };
        table[code as usize] = Some(byte as u8);
    }
    table
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decoding {
    ByteLevel,
    /// SentencePiece style: `▁` is a space, `<0xNN>` a raw byte.
    Metaspace {
        byte_fallback: bool,
    },
}

fn decoder_kind(decoder: &Value) -> Option<Decoding> {
    let kind = decoder.get("type").and_then(Value::as_str)?;
    match kind {
        "ByteLevel" => Some(Decoding::ByteLevel),
        "Metaspace" => Some(Decoding::Metaspace {
            byte_fallback: false,
        }),
        "Sequence" => {
            let parts = decoder.get("decoders").and_then(Value::as_array)?;
            if parts
                .iter()
                .any(|part| decoder_kind(part) == Some(Decoding::ByteLevel))
            {
                return Some(Decoding::ByteLevel);
            }
            let types: Vec<&str> = parts
                .iter()
                .filter_map(|part| part.get("type").and_then(Value::as_str))
                .collect();
            let replaces_metaspace = parts.iter().any(|part| {
                part.get("type").and_then(Value::as_str) == Some("Replace")
                    && part
                        .get("pattern")
                        .and_then(|pattern| pattern.get("String"))
                        .and_then(Value::as_str)
                        == Some("\u{2581}")
            }) || types.contains(&"Metaspace");
            replaces_metaspace.then_some(Decoding::Metaspace {
                byte_fallback: types.contains(&"ByteFallback"),
            })
        }
        _ => None,
    }
}

fn token_bytes(piece: &str, decoding: Decoding, table: &[Option<u8>; 512]) -> Option<Vec<u8>> {
    match decoding {
        Decoding::ByteLevel => piece
            .chars()
            .map(|ch| table.get(ch as usize).copied().flatten())
            .collect(),
        Decoding::Metaspace { byte_fallback } => {
            if byte_fallback && piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>')
            {
                return u8::from_str_radix(&piece[3..5], 16)
                    .ok()
                    .map(|byte| vec![byte]);
            }
            Some(piece.replace('\u{2581}', " ").into_bytes())
        }
    }
}

/// Whether a token is plain JSON string text: valid UTF-8 with no quote,
/// backslash or control character. Returns its character count.
fn plain_text_chars(bytes: &[u8]) -> Option<u32> {
    let text = std::str::from_utf8(bytes).ok()?;
    if text.is_empty()
        || text
            .bytes()
            .any(|byte| byte < 0x20 || byte == b'"' || byte == b'\\')
    {
        return None;
    }
    Some(text.chars().count() as u32)
}

impl TokenVocab {
    /// Build from an HF `tokenizer.json`. Added tokens (special or not) are
    /// left out, so a mask never admits a control token such as `<think>`.
    /// `vocab_size` is the logits width.
    pub fn from_tokenizer_json(tokenizer: &Value, vocab_size: usize) -> Result<Self, String> {
        let decoding = tokenizer
            .get("decoder")
            .and_then(decoder_kind)
            .ok_or_else(|| "tokenizer decoder is not ByteLevel or Metaspace".to_string())?;
        let table = byte_level_decoder_table();
        let added: std::collections::HashSet<u64> = tokenizer
            .get("added_tokens")
            .and_then(Value::as_array)
            .map(|tokens| {
                tokens
                    .iter()
                    .filter_map(|token| token.get("id").and_then(Value::as_u64))
                    .collect()
            })
            .unwrap_or_default();
        let vocab = tokenizer
            .get("model")
            .and_then(|model| model.get("vocab"))
            .ok_or_else(|| "tokenizer has no model.vocab".to_string())?;
        let mut pieces: Vec<(u32, String)> = Vec::new();
        match vocab {
            Value::Object(map) => {
                for (piece, id) in map {
                    if let Some(id) = id.as_u64() {
                        pieces.push((id as u32, piece.clone()));
                    }
                }
            }
            Value::Array(entries) => {
                for (id, entry) in entries.iter().enumerate() {
                    if let Some(piece) = entry.get(0).and_then(Value::as_str) {
                        pieces.push((id as u32, piece.to_string()));
                    }
                }
            }
            _ => return Err("tokenizer model.vocab is neither a map nor a list".to_string()),
        }
        let mut tokens = Vec::with_capacity(pieces.len());
        for (id, piece) in pieces {
            if added.contains(&u64::from(id)) || id as usize >= vocab_size {
                continue;
            }
            if let Some(bytes) = token_bytes(&piece, decoding, &table)
                && !bytes.is_empty()
            {
                tokens.push((id, bytes));
            }
        }
        Ok(Self::from_token_bytes(tokens, vocab_size))
    }

    /// Build from explicit `(token, bytes)` pairs.
    pub fn from_token_bytes(tokens: Vec<(u32, Vec<u8>)>, vocab_size: usize) -> Self {
        let mut full = Trie::new();
        let mut special_text = Trie::new();
        let mut plain = vec![0u64; vocab_size.div_ceil(64)];
        let mut plain_by_chars = Vec::new();
        let mut by_token: Vec<Box<[u8]>> = vec![Box::default(); vocab_size];
        for (token, bytes) in tokens {
            if token as usize >= vocab_size || bytes.is_empty() {
                continue;
            }
            full.insert(&bytes, token);
            match plain_text_chars(&bytes) {
                Some(chars) => {
                    plain[token as usize / 64] |= 1u64 << (token % 64);
                    plain_by_chars.push((chars, token));
                }
                None => special_text.insert(&bytes, token),
            }
            by_token[token as usize] = bytes.into_boxed_slice();
        }
        plain_by_chars.sort_unstable();
        Self {
            vocab_size,
            bytes: by_token,
            full,
            special_text,
            plain,
            plain_by_chars,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The token's bytes; `None` for tokens a mask never admits.
    pub fn token_bytes(&self, token: u32) -> Option<&[u8]> {
        self.bytes
            .get(token as usize)
            .map(|bytes| &bytes[..])
            .filter(|bytes| !bytes.is_empty())
    }

    /// Fill `out` (one bit per token) with the tokens the grammar allows
    /// next. `eos` tokens are allowed once the value is complete. Never
    /// returns an empty mask: a dead end allows `eos` so generation stops.
    pub fn allowed_tokens(
        &self,
        grammar: &JsonGrammar,
        matcher: &JsonMatcher,
        eos: &[u32],
        workspace: &mut MaskWorkspace,
        out: &mut Vec<u64>,
    ) {
        out.clear();
        out.resize(self.vocab_size.div_ceil(64), 0);
        let trie = match matcher.free_text_remaining(grammar) {
            Some(u32::MAX) => {
                for (word, plain) in out.iter_mut().zip(&self.plain) {
                    *word |= plain;
                }
                &self.special_text
            }
            Some(remaining) => {
                for &(chars, token) in &self.plain_by_chars {
                    if chars > remaining {
                        break;
                    }
                    out[token as usize / 64] |= 1u64 << (token % 64);
                }
                &self.special_text
            }
            None => &self.full,
        };
        if workspace.states.is_empty() {
            workspace.states.push(matcher.clone());
        } else {
            workspace.states[0].clone_from(matcher);
        }
        walk(trie, grammar, 0, 0, workspace, out);
        if matcher.is_complete(grammar) || out.iter().all(|word| *word == 0) {
            for &token in eos {
                if let Some(word) = out.get_mut(token as usize / 64) {
                    *word |= 1u64 << (token % 64);
                }
            }
        }
    }
}

fn walk(
    trie: &Trie,
    grammar: &JsonGrammar,
    node: u32,
    depth: usize,
    workspace: &mut MaskWorkspace,
    out: &mut [u64],
) {
    let mut edge = trie.nodes[node as usize].first_edge;
    while edge != NONE {
        let Edge { byte, child, next } = trie.edges[edge as usize];
        edge = next;
        if workspace.states.len() <= depth + 1 {
            let parent = workspace.states[depth].clone();
            workspace.states.push(parent);
        } else {
            let (parents, children) = workspace.states.split_at_mut(depth + 1);
            children[0].clone_from(&parents[depth]);
        }
        if !workspace.states[depth + 1].advance(grammar, byte) {
            continue;
        }
        let mut link = trie.nodes[child as usize].first_token;
        while link != NONE {
            let TokenLink { token, next } = trie.tokens[link as usize];
            out[token as usize / 64] |= 1u64 << (token % 64);
            link = next;
        }
        if trie.nodes[child as usize].first_edge != NONE {
            walk(trie, grammar, child, depth + 1, workspace, out);
        }
    }
}

/// Whether `token` is set in a mask built by [`TokenVocab::allowed_tokens`].
pub fn mask_allows(mask: &[u64], token: u32) -> bool {
    mask.get(token as usize / 64)
        .is_some_and(|word| word & (1u64 << (token % 64)) != 0)
}

/// Set every disallowed logit to negative infinity.
pub fn apply_token_mask(logits: &mut [f32], mask: &[u64]) {
    for (index, logit) in logits.iter_mut().enumerate() {
        if !mask_allows(mask, index as u32) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn vocab() -> TokenVocab {
        let pieces: Vec<&[u8]> = vec![
            b"{",
            b"}",
            b"{\"",
            b"\":",
            b"\"",
            b"a",
            b"name",
            b" ",
            b"\n",
            b"1",
            b"12",
            b"x\"",
            b",\"",
            b"hello world",
            b"\\",
            b"\\n",
            b"\xe6\x97",
            b"\xa5",
            b"\xe6\x97\xa5",
            b"true",
        ];
        let tokens = pieces
            .into_iter()
            .enumerate()
            .map(|(id, bytes)| (id as u32, bytes.to_vec()))
            .collect();
        TokenVocab::from_token_bytes(tokens, 24)
    }

    fn allowed(vocab: &TokenVocab, grammar: &JsonGrammar, prefix: &str) -> Vec<u32> {
        let mut matcher = grammar.matcher();
        assert!(
            matcher.advance_bytes(grammar, prefix.as_bytes()),
            "{prefix}"
        );
        let mut out = Vec::new();
        vocab.allowed_tokens(
            grammar,
            &matcher,
            &[23],
            &mut MaskWorkspace::default(),
            &mut out,
        );
        (0..24).filter(|token| mask_allows(&out, *token)).collect()
    }

    #[test]
    fn masks_follow_structure_strings_and_completion() {
        let vocab = vocab();
        let grammar = JsonGrammar::any_object();
        // Start: only tokens beginning an object (or whitespace).
        assert_eq!(allowed(&vocab, &grammar, ""), vec![0, 2, 7, 8]);
        // Inside a key every plain token is allowed, plus quote-bearing ones
        // that close the key correctly.
        let in_key = allowed(&vocab, &grammar, "{\"");
        for token in [5, 6, 13, 16, 18, 3, 4, 11] {
            assert!(in_key.contains(&token), "{token} in {in_key:?}");
        }
        // Keys take no escapes, and a lone continuation byte is not text.
        assert!(!in_key.contains(&14) && !in_key.contains(&15));
        assert!(!in_key.contains(&17));
        // Complete: only EOS.
        assert_eq!(allowed(&vocab, &grammar, "{}"), vec![23]);
    }

    #[test]
    fn length_limited_strings_filter_plain_tokens_by_characters() {
        let vocab = vocab();
        let grammar = JsonGrammar::from_schema(&json!({"type": "string", "maxLength": 2})).unwrap();
        let tokens = allowed(&vocab, &grammar, "\"");
        assert!(tokens.contains(&5));
        assert!(!tokens.contains(&6), "name is 4 characters");
        assert!(!tokens.contains(&13));
        assert!(tokens.contains(&4), "closing quote");
    }

    #[test]
    fn byte_level_tokenizer_json_maps_pieces_to_bytes() {
        let tokenizer = json!({
            "decoder": {"type": "ByteLevel"},
            "model": {"vocab": {"{": 0, "\u{0120}x": 1, "\u{010a}": 2, "<|im_end|>": 3}},
            "added_tokens": [{"id": 3, "content": "<|im_end|>", "special": true}]
        });
        let vocab = TokenVocab::from_tokenizer_json(&tokenizer, 4).unwrap();
        let grammar = JsonGrammar::any_object();
        let mut out = Vec::new();
        vocab.allowed_tokens(
            &grammar,
            &grammar.matcher(),
            &[3],
            &mut MaskWorkspace::default(),
            &mut out,
        );
        assert!(mask_allows(&out, 0));
        assert!(mask_allows(&out, 2), "newline");
        assert!(!mask_allows(&out, 1));
        assert!(!mask_allows(&out, 3));
        let mut logits = vec![1.0; 4];
        apply_token_mask(&mut logits, &out);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[0], 1.0);
    }
}

/// Mask cost on a real vocabulary. Run with
/// `AX_ENGINE_GRAMMAR_TOKENIZER_JSON=<path> cargo test --release -p ax-engine-core real_vocab_mask_cost -- --ignored --nocapture`.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod real_vocab {
    use super::*;

    #[test]
    #[ignore = "needs AX_ENGINE_GRAMMAR_TOKENIZER_JSON"]
    fn real_vocab_mask_cost() {
        let path = std::env::var("AX_ENGINE_GRAMMAR_TOKENIZER_JSON").unwrap();
        let tokenizer: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let started = std::time::Instant::now();
        let vocab = TokenVocab::from_tokenizer_json(&tokenizer, 248_320).unwrap();
        eprintln!("build {:?}", started.elapsed());
        let grammar = JsonGrammar::from_schema(&serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "short": {"type": "string", "maxLength": 8},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["name", "age"]
        }))
        .unwrap();
        let mut workspace = MaskWorkspace::default();
        let mut out = Vec::new();
        for prefix in [
            "",
            "{",
            "{\"",
            "{\"name\": \"Al",
            "{\"short\": \"ab",
            "{\"age\": 4",
            "{\"tags\": [",
            "{\"x\": {\"y",
        ] {
            let mut matcher = grammar.matcher();
            assert!(matcher.advance_bytes(&grammar, prefix.as_bytes()));
            let started = std::time::Instant::now();
            for _ in 0..20 {
                vocab.allowed_tokens(&grammar, &matcher, &[248_046], &mut workspace, &mut out);
            }
            let allowed: u32 = out.iter().map(|word| word.count_ones()).sum();
            eprintln!(
                "{prefix:<20} {:>8.3} ms  allowed {allowed}",
                started.elapsed().as_secs_f64() * 50.0
            );
        }
    }
}
