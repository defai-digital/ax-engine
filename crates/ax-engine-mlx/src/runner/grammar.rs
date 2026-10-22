//! Grammar-constrained decoding state of one request (`response_format`
//! json_object / json_schema): the committed matcher, and cursors that walk
//! it forward along MTP drafts so every verify row gets its own mask.
//!
//! Think blocks are not constrained: think markers and the tokens between
//! them pass through, and the grammar applies to the answer after `</think>`.

use std::sync::Arc;

use ax_engine_core::grammar::{JsonGrammar, JsonMatcher, MaskWorkspace, TokenVocab};

#[derive(Clone, Debug)]
pub(super) struct RequestGrammar {
    grammar: Arc<JsonGrammar>,
    matcher: JsonMatcher,
    /// A committed token the matcher refused. Masking stops; the server's
    /// post-hoc validation reports the output.
    dead: bool,
}

/// Grammar position at one point of a (possibly speculative) token stream.
#[derive(Clone, Debug)]
pub(super) struct GrammarCursor<'a> {
    grammar: &'a JsonGrammar,
    vocab: &'a TokenVocab,
    think: (Option<u32>, Option<u32>),
    terminal: &'a [u32],
    matcher: JsonMatcher,
    in_think: bool,
    dead: bool,
}

impl RequestGrammar {
    pub(super) fn new(grammar: Arc<JsonGrammar>) -> Self {
        let matcher = grammar.matcher();
        Self {
            grammar,
            matcher,
            dead: false,
        }
    }

    pub(super) fn is_for(&self, grammar: &Arc<JsonGrammar>) -> bool {
        Arc::ptr_eq(&self.grammar, grammar)
    }

    pub(super) fn cursor<'a>(
        &'a self,
        vocab: &'a TokenVocab,
        think: (Option<u32>, Option<u32>),
        in_think: bool,
        terminal: &'a [u32],
    ) -> GrammarCursor<'a> {
        GrammarCursor {
            grammar: &self.grammar,
            vocab,
            think,
            terminal,
            matcher: self.matcher.clone(),
            in_think,
            dead: self.dead,
        }
    }

    /// Commit one generated token; `in_think` is the think state before it.
    pub(super) fn commit(
        &mut self,
        vocab: &TokenVocab,
        think: (Option<u32>, Option<u32>),
        in_think: bool,
        terminal: &[u32],
        token: u32,
    ) {
        let mut cursor = self.cursor(vocab, think, in_think, terminal);
        cursor.advance(token);
        let (matcher, dead) = (cursor.matcher, cursor.dead);
        if dead && !self.dead {
            tracing::warn!(
                token,
                "grammar-constrained request committed a token outside its grammar; \
                 constraint released for the rest of the request"
            );
        }
        self.matcher = matcher;
        self.dead = dead;
    }
}

impl GrammarCursor<'_> {
    pub(super) fn advance(&mut self, token: u32) {
        if self.think.0 == Some(token) {
            self.in_think = true;
            return;
        }
        if self.think.1 == Some(token) {
            self.in_think = false;
            return;
        }
        if self.in_think || self.dead || self.terminal.contains(&token) {
            return;
        }
        let accepted = self
            .vocab
            .token_bytes(token)
            .is_some_and(|bytes| self.matcher.advance_bytes(self.grammar, bytes));
        if !accepted {
            self.dead = true;
        }
    }

    /// Allowed tokens at this position; `None` leaves the row unconstrained
    /// (inside a think block, after a refused token, or when nothing at all
    /// would be allowed).
    pub(super) fn mask(&self) -> Option<Vec<u64>> {
        if self.in_think || self.dead {
            return None;
        }
        let mut out = Vec::new();
        self.vocab.allowed_tokens(
            self.grammar,
            &self.matcher,
            self.terminal,
            &mut MaskWorkspace::default(),
            &mut out,
        );
        out.iter().any(|word| *word != 0).then_some(out)
    }

    /// Masks for every verify row of an MTP window: row `i` follows
    /// `drafts[..i]`, so there are `drafts.len() + 1` rows.
    pub(super) fn row_masks(mut self, drafts: &[u32]) -> Vec<Option<Vec<u64>>> {
        let mut masks = Vec::with_capacity(drafts.len() + 1);
        masks.push(self.mask());
        for &token in drafts {
            self.advance(token);
            masks.push(self.mask());
        }
        masks
    }
}

/// Load the byte vocabulary of the model's `tokenizer.json` for masks.
pub(super) fn load_token_vocab(artifacts_root: &str, vocab_size: usize) -> Option<Arc<TokenVocab>> {
    let path = std::path::Path::new(artifacts_root).join("tokenizer.json");
    let built = std::fs::read(&path)
        .map_err(|error| error.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|error| error.to_string()))
        .and_then(|tokenizer| TokenVocab::from_tokenizer_json(&tokenizer, vocab_size));
    match built {
        Ok(vocab) => Some(Arc::new(vocab)),
        Err(error) => {
            tracing::warn!(
                path = %path.display(),
                %error,
                "grammar-constrained decoding unavailable; json responses keep post-hoc validation only"
            );
            None
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ax_engine_core::grammar::mask_allows;

    const THINK: (Option<u32>, Option<u32>) = (Some(5), Some(6));
    const EOS: &[u32] = &[7];

    fn vocab() -> TokenVocab {
        let pieces: [&[u8]; 5] = [b"{", b"}", b"x", b"\"a\":1", b" "];
        TokenVocab::from_token_bytes(
            pieces
                .iter()
                .enumerate()
                .map(|(id, bytes)| (id as u32, bytes.to_vec()))
                .collect(),
            8,
        )
    }

    #[test]
    fn row_masks_follow_drafts_and_skip_think_blocks() {
        let vocab = vocab();
        let grammar = RequestGrammar::new(Arc::new(JsonGrammar::any_object()));
        // Inside a think block nothing is masked until `</think>`.
        let masks = grammar
            .cursor(&vocab, THINK, true, EOS)
            .row_masks(&[2, 6, 0]);
        assert_eq!(masks.len(), 4);
        assert!(masks[0].is_none() && masks[1].is_none());
        let after_close = masks[2].as_ref().unwrap();
        assert!(mask_allows(after_close, 0) && !mask_allows(after_close, 2));
        // After `{`: a key or `}`; `x` and EOS are refused.
        let after_open = masks[3].as_ref().unwrap();
        assert!(mask_allows(after_open, 1) && mask_allows(after_open, 3));
        assert!(!mask_allows(after_open, 2) && !mask_allows(after_open, 7));
    }

    #[test]
    fn commits_complete_the_value_and_then_allow_only_eos() {
        let vocab = vocab();
        let mut grammar = RequestGrammar::new(Arc::new(JsonGrammar::any_object()));
        for token in [0, 3, 1] {
            grammar.commit(&vocab, THINK, false, EOS, token);
        }
        let mask = grammar.cursor(&vocab, THINK, false, EOS).mask().unwrap();
        assert_eq!(
            (0..8)
                .filter(|token| mask_allows(&mask, *token))
                .collect::<Vec<_>>(),
            vec![7]
        );
        // A refused token releases the constraint instead of trapping the
        // request with an empty mask.
        let mut broken = RequestGrammar::new(Arc::new(JsonGrammar::any_object()));
        broken.commit(&vocab, THINK, false, EOS, 2);
        assert!(broken.cursor(&vocab, THINK, false, EOS).mask().is_none());
    }
}
