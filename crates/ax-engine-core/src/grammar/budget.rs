//! Bounded proofs that a legal next token can finish the JSON answer in time.
//!
//! Search operates at tokenizer boundaries. It never appends a closing suffix
//! or substitutes values. A resource limit is an explicit error, not evidence
//! that a continuation is impossible.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use super::{JsonGrammar, JsonMatcher, MaskWorkspace, TokenVocab};

const MAX_STATES: usize = 4096;
const MAX_TOKEN_VISITS: usize = 4_000_000;
const MAX_SEARCH_DEPTH: u32 = 128;
const MAX_SEARCH_WORK: usize = 1_000_000;

struct SearchWork(usize);

impl SearchWork {
    fn charge(&mut self) -> Result<(), BudgetError> {
        self.0 = self.0.checked_sub(1).ok_or(BudgetError::SearchLimit)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BudgetError {
    #[error("grammar_budget cannot complete the JSON answer within the remaining output tokens")]
    Infeasible,
    #[error("grammar_budget completion proof exceeded its bounded search resources")]
    SearchLimit,
    #[error("grammar_budget requires a reasoning end token to reserve room for the JSON answer")]
    MissingReasoningBoundary,
    #[error("grammar_budget encountered an invalid committed JSON parser state")]
    InvalidState,
}

type Successors = Arc<Vec<(JsonMatcher, Vec<u32>)>>;

/// Request-owned transition cache. Remaining tokens are supplied to every
/// proof; a mask proved for a larger budget is never reused for a smaller one.
#[derive(Debug, Default)]
pub struct BudgetWorkspace {
    transitions: HashMap<JsonMatcher, Successors>,
    /// Length of an actual discovered completion, never a heuristic bound.
    witnesses: HashMap<JsonMatcher, u32>,
    token_visits: usize,
    terminal: Vec<u32>,
    scope: Option<(JsonGrammar, u64)>,
}

impl BudgetWorkspace {
    fn remember_witness(
        &mut self,
        mut state: JsonMatcher,
        mut length: u32,
        parents: &HashMap<JsonMatcher, JsonMatcher>,
        work: &mut SearchWork,
    ) -> Result<(), BudgetError> {
        loop {
            work.charge()?;
            if self.witnesses.len() < MAX_STATES || self.witnesses.contains_key(&state) {
                self.witnesses
                    .entry(state.clone())
                    .and_modify(|old| *old = (*old).min(length))
                    .or_insert(length);
            }
            let Some(parent) = parents.get(&state) else {
                break;
            };
            state = parent.clone();
            length += 1;
        }
        Ok(())
    }

    fn successors(
        &mut self,
        vocab: &TokenVocab,
        grammar: &JsonGrammar,
        state: &JsonMatcher,
        work: &mut SearchWork,
    ) -> Result<Successors, BudgetError> {
        work.charge()?;
        let state = state.mask_key();
        if let Some(found) = self.transitions.get(&state) {
            return Ok(Arc::clone(found));
        }
        if self.transitions.len() >= MAX_STATES {
            return Err(BudgetError::SearchLimit);
        }
        let mut mask = Vec::new();
        vocab.allowed_tokens(
            grammar,
            &state,
            &[],
            &mut MaskWorkspace::default(),
            &mut mask,
        );
        let mut groups = Vec::<(JsonMatcher, Vec<u32>)>::new();
        let mut indices = HashMap::new();
        for (word_index, &word) in mask.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                work.charge()?;
                self.token_visits += 1;
                if self.token_visits > MAX_TOKEN_VISITS {
                    return Err(BudgetError::SearchLimit);
                }
                let bit = remaining.trailing_zeros();
                remaining &= remaining - 1;
                let token = (word_index * 64 + bit as usize) as u32;
                if self.terminal.contains(&token) {
                    continue;
                }
                let Some(bytes) = vocab.token_bytes(token) else {
                    continue;
                };
                let mut next = state.clone();
                if !next.advance_bytes(grammar, bytes) {
                    continue;
                }
                let next = next.mask_key();
                let index = *indices.entry(next.clone()).or_insert_with(|| {
                    groups.push((next, Vec::new()));
                    groups.len() - 1
                });
                groups[index].1.push(token);
            }
        }
        let groups = Arc::new(groups);
        self.transitions.insert(state, Arc::clone(&groups));
        Ok(groups)
    }

    fn can_finish(
        &mut self,
        vocab: &TokenVocab,
        grammar: &JsonGrammar,
        start: &JsonMatcher,
        tokens: u32,
        work: &mut SearchWork,
    ) -> Result<bool, BudgetError> {
        if start.is_complete(grammar) {
            return Ok(true);
        }
        let mut seen = HashSet::from([start.clone()]);
        let mut parents = HashMap::new();
        let mut queue = VecDeque::from([(start.clone(), 0)]);
        let mut truncated = false;
        while let Some((state, depth)) = queue.pop_front() {
            work.charge()?;
            if let Some(&length) = self.witnesses.get(&state)
                && depth + length <= tokens.min(MAX_SEARCH_DEPTH)
            {
                self.remember_witness(state, length, &parents, work)?;
                return Ok(true);
            }
            if depth >= tokens {
                continue;
            }
            if depth >= MAX_SEARCH_DEPTH {
                truncated = true;
                continue;
            }
            for (next, _) in self.successors(vocab, grammar, &state, work)?.iter() {
                work.charge()?;
                if next.is_complete(grammar) {
                    parents.entry(next.clone()).or_insert_with(|| state.clone());
                    self.remember_witness(next.clone(), 0, &parents, work)?;
                    return Ok(true);
                }
                if seen.insert(next.clone()) {
                    if seen.len() > MAX_STATES {
                        return Err(BudgetError::SearchLimit);
                    }
                    parents.insert(next.clone(), state.clone());
                    queue.push_back((next.clone(), depth + 1));
                }
            }
        }
        if truncated {
            Err(BudgetError::SearchLimit)
        } else {
            Ok(false)
        }
    }

    /// Legal tokens with a proven complete JSON continuation within `remaining`
    /// draws, including this draw. A complete value does not require an extra
    /// EOS draw: finishing on the length boundary is valid JSON but still has
    /// the normal length finish reason. EOS is allowed only at a complete value.
    pub fn mask(
        &mut self,
        vocab: &TokenVocab,
        grammar: &JsonGrammar,
        matcher: &JsonMatcher,
        eos: &[u32],
        remaining: u32,
    ) -> Result<Vec<u64>, BudgetError> {
        if remaining == 0 {
            return Err(BudgetError::Infeasible);
        }
        if self.terminal != eos
            || self
                .scope
                .as_ref()
                .is_none_or(|(g, v)| g != grammar || *v != vocab.identity())
        {
            self.transitions.clear();
            self.witnesses.clear();
            self.token_visits = 0;
            self.terminal = eos.to_vec();
            self.scope = Some((grammar.clone(), vocab.identity()));
        }
        let mut work = SearchWork(MAX_SEARCH_WORK);
        let mut mask = vec![0; vocab.vocab_size().div_ceil(64)];
        let successors = self.successors(vocab, grammar, matcher, &mut work)?;
        for (next, tokens) in successors.iter() {
            work.charge()?;
            if self.can_finish(vocab, grammar, next, remaining - 1, &mut work)? {
                for &token in tokens {
                    work.charge()?;
                    // Special terminal ids must obey the parser's current state,
                    // even when their vocabulary spelling looks like JSON.
                    if !eos.contains(&token) {
                        mask[token as usize / 64] |= 1u64 << (token % 64);
                    }
                }
            }
        }
        if matcher.is_complete(grammar) {
            for &token in eos {
                if let Some(word) = mask.get_mut(token as usize / 64) {
                    *word |= 1u64 << (token % 64);
                }
            }
        }
        if mask.iter().all(|&word| word == 0) {
            Err(BudgetError::Infeasible)
        } else {
            Ok(mask)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::grammar::mask_allows;
    use serde_json::json;

    fn vocab(parts: &[&[u8]]) -> TokenVocab {
        TokenVocab::from_token_bytes(
            parts
                .iter()
                .enumerate()
                .map(|(i, b)| (i as u32, b.to_vec()))
                .collect(),
            parts.len(),
        )
    }

    #[test]
    fn proof_depth_limit_is_reported_separately_from_token_budget() {
        let grammar = JsonGrammar::from_schema(&json!({"const":"a".repeat(129)})).unwrap();
        let vocab = vocab(&[b"\"", b"a"]);
        let mut workspace = BudgetWorkspace::default();
        assert_eq!(
            workspace.mask(&vocab, &grammar, &grammar.matcher(), &[], 128),
            Err(BudgetError::Infeasible)
        );
        assert_eq!(
            workspace.mask(&vocab, &grammar, &grammar.matcher(), &[], 200),
            Err(BudgetError::SearchLimit)
        );
    }

    #[test]
    fn budget_counts_tokens_not_characters_and_keeps_length_terminal() {
        let grammar = JsonGrammar::from_schema(&json!({"const":{"a":true}})).unwrap();
        let vocab = vocab(&[b"{", b"\"a\":true}", b"{\"a\":true}", b" ", b"EOS"]);
        let mut workspace = BudgetWorkspace::default();
        let state = grammar.matcher();
        let one = workspace.mask(&vocab, &grammar, &state, &[4], 1).unwrap();
        assert!(!mask_allows(&one, 0));
        assert!(mask_allows(&one, 2));
        assert!(!mask_allows(&one, 3));
        assert!(!mask_allows(&one, 4));
        let two = workspace.mask(&vocab, &grammar, &state, &[4], 2).unwrap();
        assert!(mask_allows(&two, 0));
        assert!(mask_allows(&two, 3));
        // Cached transitions cannot widen a subsequent smaller-budget mask.
        assert_eq!(
            workspace.mask(&vocab, &grammar, &state, &[4], 1).unwrap(),
            one
        );
    }

    #[test]
    fn repeated_property_masks_keep_declared_value_constraints() {
        let grammar = JsonGrammar::from_schema(&json!({
            "type": "object", "properties": {"x": {"type": "integer"}},
            "required": ["x"]
        }))
        .unwrap();
        let vocab = vocab(&[b"2}", b"\"wrong\"}", b"false}", b"EOS"]);
        let mut state = grammar.matcher();
        assert!(state.advance_bytes(&grammar, br#"{"x":1,"x":"#));
        let mut plain = Vec::new();
        vocab.allowed_tokens(
            &grammar,
            &state,
            &[3],
            &mut MaskWorkspace::default(),
            &mut plain,
        );
        let bounded = BudgetWorkspace::default()
            .mask(&vocab, &grammar, &state, &[3], 1)
            .unwrap();
        for mask in [plain, bounded] {
            assert!(mask_allows(&mask, 0));
            for token in [1, 2, 3] {
                assert!(!mask_allows(&mask, token), "{token}");
            }
        }
    }

    #[test]
    fn unicode_and_nested_completion_use_real_token_boundaries() {
        let grammar = JsonGrammar::from_schema(&json!({"const":[{"x":"\u{6c34}"}]})).unwrap();
        let vocab = vocab(&[b"[{\"x\":\"", &[0xe6], &[0xb0, 0xb4], b"\"}]", b"EOS"]);
        let mut workspace = BudgetWorkspace::default();
        let mut state = grammar.matcher();
        assert_eq!(
            workspace.mask(&vocab, &grammar, &state, &[4], 3),
            Err(BudgetError::Infeasible)
        );
        for (i, remaining) in [(0, 4), (1, 3), (2, 2), (3, 1)] {
            let mask = workspace
                .mask(&vocab, &grammar, &state, &[4], remaining)
                .unwrap();
            assert!(mask_allows(&mask, i));
            assert!(state.advance_bytes(&grammar, vocab.token_bytes(i).unwrap()));
        }
        assert!(state.is_complete(&grammar));
        assert!(mask_allows(
            &workspace.mask(&vocab, &grammar, &state, &[4], 1).unwrap(),
            4
        ));
    }

    #[test]
    fn budget_mask_matches_exhaustive_small_language() {
        let grammar = JsonGrammar::from_schema(
            &json!({"type":"array","items":{"enum":[true,false]},"maxItems":2}),
        )
        .unwrap();
        let vocab = vocab(&[b"[", b"]", b",", b"true", b"false", b" "]);
        fn brute(
            vocab: &TokenVocab,
            grammar: &JsonGrammar,
            state: &JsonMatcher,
            left: u32,
        ) -> bool {
            state.is_complete(grammar)
                || (left > 0
                    && (0..vocab.vocab_size()).any(|token| {
                        let mut next = state.clone();
                        next.advance_bytes(grammar, vocab.token_bytes(token as u32).unwrap())
                            && brute(vocab, grammar, &next, left - 1)
                    }))
        }
        let mut workspace = BudgetWorkspace::default();
        for prefix in ["", "[", "[true", "[false,", "[true,false"] {
            let mut state = grammar.matcher();
            assert!(state.advance_bytes(&grammar, prefix.as_bytes()));
            // Reuse a generous proof, then shrink the budget, repeatedly.
            for left in [5, 1, 4, 2, 3, 5, 1] {
                let result = workspace.mask(&vocab, &grammar, &state, &[], left);
                assert_ne!(result, Err(BudgetError::SearchLimit));
                for token in 0..vocab.vocab_size() as u32 {
                    let mut next = state.clone();
                    let expected = next.advance_bytes(&grammar, vocab.token_bytes(token).unwrap())
                        && brute(&vocab, &grammar, &next, left - 1);
                    assert_eq!(
                        result.as_ref().is_ok_and(|mask| mask_allows(mask, token)),
                        expected,
                        "prefix={prefix}, left={left}, token={token}"
                    );
                }
            }
        }
    }

    #[test]
    fn resource_exhaustion_is_distinct_from_an_impossible_budget() {
        let grammar = JsonGrammar::from_schema(&json!({"const":true})).unwrap();
        let vocab = vocab(&[b"true"]);
        let mut workspace = BudgetWorkspace {
            token_visits: MAX_TOKEN_VISITS,
            scope: Some((grammar.clone(), vocab.identity())),
            ..Default::default()
        };
        assert_eq!(
            workspace.mask(&vocab, &grammar, &grammar.matcher(), &[], 1),
            Err(BudgetError::SearchLimit)
        );
    }

    #[test]
    fn terminal_spelling_cannot_supply_a_completion_witness() {
        let grammar = JsonGrammar::from_schema(&json!({"const":[true]})).unwrap();
        let vocab = vocab(&[b"[", b"true]", b"true", b"]"]);
        let mut workspace = BudgetWorkspace::default();
        assert!(
            workspace
                .mask(&vocab, &grammar, &grammar.matcher(), &[], 2)
                .is_ok()
        );
        assert_eq!(
            workspace.mask(&vocab, &grammar, &grammar.matcher(), &[1, 2], 8),
            Err(BudgetError::Infeasible)
        );
    }

    #[test]
    fn transition_cache_is_scoped_to_grammar_and_tokenizer() {
        let a = JsonGrammar::from_schema(&json!({"const":true})).unwrap();
        let b = JsonGrammar::from_schema(&json!({"const":false})).unwrap();
        let first = vocab(&[b"true", b"false"]);
        let second = vocab(&[b"false", b"true"]);
        let mut workspace = BudgetWorkspace::default();
        for (grammar, vocab, expected) in [(&a, &first, 0), (&b, &first, 1), (&b, &second, 0)] {
            let mask = workspace
                .mask(vocab, grammar, &grammar.matcher(), &[], 1)
                .unwrap();
            assert!(mask_allows(&mask, expected));
            assert!(!mask_allows(&mask, 1 - expected));
        }
    }

    #[test]
    fn warm_transition_search_still_has_a_work_ceiling() {
        let grammar = JsonGrammar::from_schema(&json!({"const":[true]})).unwrap();
        let vocab = vocab(&[b"[", b"true", b"]"]);
        let mut workspace = BudgetWorkspace::default();
        assert!(
            workspace
                .mask(&vocab, &grammar, &grammar.matcher(), &[], 3)
                .is_ok()
        );
        assert_eq!(
            workspace.can_finish(&vocab, &grammar, &grammar.matcher(), 3, &mut SearchWork(1)),
            Err(BudgetError::SearchLimit)
        );
    }

    #[test]
    fn no_extra_eos_draw_is_permitted_after_the_length_limit() {
        let grammar = JsonGrammar::from_schema(&json!({"const":true})).unwrap();
        let vocab = vocab(&[b"true", b"EOS"]);
        let mut state = grammar.matcher();
        let mut workspace = BudgetWorkspace::default();
        assert!(mask_allows(
            &workspace.mask(&vocab, &grammar, &state, &[1], 1).unwrap(),
            0
        ));
        assert!(state.advance_bytes(&grammar, b"true"));
        assert!(state.is_complete(&grammar));
        assert_eq!(
            workspace.mask(&vocab, &grammar, &state, &[1], 0),
            Err(BudgetError::Infeasible)
        );
    }
}
