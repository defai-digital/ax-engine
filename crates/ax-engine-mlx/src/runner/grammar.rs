//! Grammar-constrained decoding state of one request (`response_format`
//! json_object / json_schema): the committed matcher, and cursors that walk
//! it forward along MTP drafts so every verify row gets its own mask.
//!
//! Think blocks are not constrained: think markers and the tokens between
//! them pass through, and the grammar applies to the answer after `</think>`.

use std::collections::HashMap;
use std::sync::Arc;

use ax_engine_core::grammar::budget::{BudgetError, BudgetWorkspace};
use ax_engine_core::grammar::{JsonGrammar, JsonMatcher, MaskWorkspace, TokenVocab};
use parking_lot::Mutex;

/// Masks a request keeps by grammar position. Positions repeat a lot (every
/// step inside one long string, the same separators between array items),
/// so most masks are looked up rather than walked.
const MASK_CACHE_ENTRIES: usize = 64;

type MaskCache = Mutex<HashMap<JsonMatcher, Arc<[u64]>>>;

#[derive(Clone, Debug)]
pub(super) struct RequestGrammar {
    grammar: Arc<JsonGrammar>,
    matcher: JsonMatcher,
    masks: Arc<MaskCache>,
    budget: Arc<Mutex<BudgetWorkspace>>,
    budget_error: Arc<Mutex<Option<BudgetError>>>,
    remaining: u32,
    /// A committed token the matcher refused. Masking stops; the server's
    /// post-hoc validation reports the output.
    dead: bool,
}

/// Grammar position at one point of a (possibly speculative) token stream.
#[derive(Clone, Debug)]
pub(super) struct GrammarCursor<'a> {
    grammar: &'a JsonGrammar,
    masks: &'a MaskCache,
    budget: &'a Mutex<BudgetWorkspace>,
    budget_error: &'a Mutex<Option<BudgetError>>,
    remaining: u32,
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
            masks: Arc::default(),
            budget: Arc::default(),
            budget_error: Arc::default(),
            remaining: u32::MAX,
            dead: false,
        }
    }

    pub(super) fn is_for(&self, grammar: &Arc<JsonGrammar>) -> bool {
        Arc::ptr_eq(&self.grammar, grammar)
    }

    pub(super) fn set_remaining(&mut self, remaining: u32) {
        self.remaining = remaining;
    }

    pub(super) fn take_budget_error(&self) -> Option<BudgetError> {
        self.budget_error.lock().take()
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
            masks: &self.masks,
            budget: &self.budget,
            budget_error: &self.budget_error,
            remaining: self.remaining,
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
        self.remaining = self.remaining.saturating_sub(1);
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
    /// (inside a think block, after a refused token, or at a dead end, which
    /// the post-hoc check then reports).
    pub(super) fn mask(&self) -> Option<Vec<u64>> {
        if self.dead && self.grammar.completion_budget() {
            return Some(vec![0; self.vocab.vocab_size().div_ceil(64)]);
        }
        match self.mask_checked() {
            Ok(mask) => mask,
            Err(error) => {
                // An infeasible speculative suffix has zero target support;
                // its entering draft is rejected by the preceding mask.
                // Resource failure is not zero support: stop the request
                // before committing the speculative result.
                if error != BudgetError::Infeasible {
                    *self.budget_error.lock() = Some(error);
                }
                Some(vec![0; self.vocab.vocab_size().div_ceil(64)])
            }
        }
    }

    pub(super) fn mask_checked(&self) -> Result<Option<Vec<u64>>, BudgetError> {
        if self.dead && self.grammar.completion_budget() {
            return Err(BudgetError::InvalidState);
        }
        if self.grammar.completion_budget() && !self.dead {
            let mut workspace = self.budget.lock();
            if self.in_think {
                let end = self
                    .think
                    .1
                    .filter(|&t| (t as usize) < self.vocab.vocab_size())
                    .ok_or(BudgetError::MissingReasoningBoundary)?;
                // The close marker itself consumes one output token.
                workspace.mask(
                    self.vocab,
                    self.grammar,
                    &self.matcher,
                    self.terminal,
                    self.remaining.saturating_sub(1),
                )?;
                return match workspace.mask(
                    self.vocab,
                    self.grammar,
                    &self.matcher,
                    self.terminal,
                    self.remaining.saturating_sub(2),
                ) {
                    Ok(_) => {
                        // Reasoning is free text, but terminal ids would end
                        // the request before the required JSON answer exists.
                        let mut mask = vec![u64::MAX; self.vocab.vocab_size().div_ceil(64)];
                        for &token in self.terminal {
                            if let Some(word) = mask.get_mut(token as usize / 64) {
                                *word &= !(1u64 << (token % 64));
                            }
                        }
                        Ok(Some(mask))
                    }
                    Err(BudgetError::Infeasible) => {
                        let mut mask = vec![0; self.vocab.vocab_size().div_ceil(64)];
                        mask[end as usize / 64] |= 1u64 << (end % 64);
                        Ok(Some(mask))
                    }
                    Err(error) => Err(error),
                };
            }
            return workspace
                .mask(
                    self.vocab,
                    self.grammar,
                    &self.matcher,
                    self.terminal,
                    self.remaining,
                )
                .map(Some);
        }
        if self.in_think || self.dead {
            return Ok(None);
        }
        let key = self.matcher.mask_key();
        if let Some(mask) = self.masks.lock().get(&key) {
            return Ok(Some(mask.to_vec()));
        }
        let mut out = Vec::new();
        self.vocab.allowed_tokens(
            self.grammar,
            &self.matcher,
            self.terminal,
            &mut MaskWorkspace::default(),
            &mut out,
        );
        if out.iter().all(|word| *word == 0) {
            tracing::warn!(
                "grammar reached a position no token can continue; row left unconstrained"
            );
            return Ok(None);
        }
        let mut masks = self.masks.lock();
        if masks.len() >= MASK_CACHE_ENTRIES {
            masks.clear();
        }
        masks.insert(key, Arc::from(out.as_slice()));
        Ok(Some(out))
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

    #[test]
    fn completion_budget_reserves_reasoning_close_and_counts_each_verify_row() {
        let vocab = vocab();
        let compiled = Arc::new(JsonGrammar::any_object().with_completion_budget(true));
        let mut grammar = RequestGrammar::new(compiled);
        grammar.set_remaining(3);
        let close = grammar
            .cursor(&vocab, THINK, true, EOS)
            .mask_checked()
            .unwrap()
            .unwrap();
        assert!(mask_allows(&close, 6));
        assert!(!mask_allows(&close, 2));
        let masks = grammar
            .cursor(&vocab, THINK, true, EOS)
            .row_masks(&[6, 0, 1]);
        assert!(mask_allows(masks[1].as_ref().unwrap(), 0));
        assert!(mask_allows(masks[2].as_ref().unwrap(), 1));
        assert!(!mask_allows(masks[2].as_ref().unwrap(), 3));
        assert!(masks[3].as_ref().unwrap().iter().all(|&word| word == 0));
        assert!(grammar.take_budget_error().is_none());
        // A rejected speculative branch never advances the committed matcher.
        let fresh = grammar
            .cursor(&vocab, THINK, false, EOS)
            .mask_checked()
            .unwrap()
            .unwrap();
        assert!(mask_allows(&fresh, 0));
        assert!(!mask_allows(&fresh, 1));
    }

    #[test]
    fn completion_budget_invalid_drafts_are_zero_support_not_deleted() {
        let vocab = vocab();
        let mut grammar = RequestGrammar::new(Arc::new(
            JsonGrammar::any_object().with_completion_budget(true),
        ));
        grammar.set_remaining(3);
        for drafts in [&[2, 0][..], &[0, 2, 1][..]] {
            let masks = grammar.cursor(&vocab, THINK, false, EOS).row_masks(drafts);
            assert_eq!(masks.len(), drafts.len() + 1);
            let invalid = drafts.iter().position(|&token| token == 2).unwrap();
            assert!(!mask_allows(masks[invalid].as_ref().unwrap(), 2));
            assert!(
                masks[invalid + 1]
                    .as_ref()
                    .unwrap()
                    .iter()
                    .all(|&word| word == 0)
            );
        }
        assert!(grammar.take_budget_error().is_none());
        grammar.commit(&vocab, THINK, false, EOS, 0);
        grammar.set_remaining(1);
        let last = grammar
            .cursor(&vocab, THINK, false, EOS)
            .mask_checked()
            .unwrap()
            .unwrap();
        assert!(mask_allows(&last, 1));
        assert!(!mask_allows(&last, 3));
    }

    #[test]
    fn completion_budget_failure_is_explicit_and_request_local() {
        let vocab = vocab();
        let compiled = Arc::new(JsonGrammar::any_object().with_completion_budget(true));
        let mut short = RequestGrammar::new(Arc::clone(&compiled));
        let mut enough = RequestGrammar::new(compiled);
        short.set_remaining(1);
        enough.set_remaining(2);
        assert_eq!(
            short.cursor(&vocab, THINK, false, EOS).mask_checked(),
            Err(BudgetError::Infeasible)
        );
        assert!(
            enough
                .cursor(&vocab, THINK, false, EOS)
                .mask_checked()
                .is_ok()
        );
        short.commit(&vocab, THINK, false, EOS, 2);
        assert_eq!(
            short.cursor(&vocab, THINK, false, EOS).mask_checked(),
            Err(BudgetError::InvalidState)
        );
        assert!(
            enough
                .cursor(&vocab, THINK, false, EOS)
                .mask_checked()
                .is_ok()
        );
    }

    #[test]
    fn completion_budget_cannot_end_during_unconstrained_reasoning() {
        let vocab = vocab();
        let mut grammar = RequestGrammar::new(Arc::new(
            JsonGrammar::any_object().with_completion_budget(true),
        ));
        grammar.set_remaining(16);
        let mask = grammar
            .cursor(&vocab, THINK, true, EOS)
            .mask_checked()
            .unwrap()
            .unwrap();
        assert!(mask_allows(&mask, 2));
        assert!(mask_allows(&mask, 6));
        assert!(!mask_allows(&mask, 7));
    }
}
