//! Pure speculative-token commit rules. No tensor, runtime, or platform dependencies.
//!
//! Backend code owns verification and rollback. These functions define which
//! verified tokens survive and how many draft rows that rollback must retain.

/// Number of consecutive draft tokens matching the verifier's choices.
/// A verifier bonus row, when present, is not counted as a draft.
pub fn accepted_prefix_len(drafts: &[u32], predicted: &[u32]) -> usize {
    drafts
        .iter()
        .zip(predicted)
        .take_while(|(draft, target)| draft == target)
        .count()
}

/// Cut verified tokens at the output budget and first EOS, including that EOS.
/// Returns emitted tokens and the accepted draft rows retained in backend state.
pub fn limit_commit(
    tokens: &[u32],
    accepted_drafts: usize,
    budget: usize,
    eos: &[u32],
) -> (Vec<u32>, usize) {
    let end = tokens
        .iter()
        .take(budget)
        .position(|token| eos.contains(token))
        .map_or(tokens.len().min(budget), |index| index + 1);
    (tokens[..end].to_vec(), end.min(accepted_drafts))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GreedyCommit {
    pub accepted_drafts: usize,
    pub emitted: Vec<u32>,
    pub retained_draft_rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CommitError {
    #[error("a greedy verify needs one prediction per draft plus a bonus prediction")]
    PredictionCount,
}

/// The greedy verify walk shared by every backend: accepted drafts followed by
/// the target's own token at the first mismatch (or after the last draft).
/// `predicted[i]` is the target choice after row `i`, not the input at that row.
/// Fails closed unless there is exactly one prediction per draft plus a bonus.
pub fn greedy_verify(drafts: &[u32], predicted: &[u32]) -> Result<(usize, Vec<u32>), CommitError> {
    if drafts.len().checked_add(1) != Some(predicted.len()) {
        return Err(CommitError::PredictionCount);
    }
    let accepted_drafts = accepted_prefix_len(drafts, predicted);
    let mut verified = drafts[..accepted_drafts].to_vec();
    verified.push(predicted[accepted_drafts]);
    Ok((accepted_drafts, verified))
}

/// A checked host-side oracle for backend conformance tests and replay tools.
pub fn plan_greedy_commit(
    drafts: &[u32],
    predicted: &[u32],
    budget: usize,
    eos: &[u32],
) -> Result<GreedyCommit, CommitError> {
    let (accepted_drafts, verified) = greedy_verify(drafts, predicted)?;
    let (emitted, retained_draft_rows) = limit_commit(&verified, accepted_drafts, budget, eos);
    Ok(GreedyCommit {
        accepted_drafts,
        emitted,
        retained_draft_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// The loop-based rule `limit_commit` replaced, kept as an independent reference.
    fn reference_limit_commit(
        tokens: &[u32],
        accepted_drafts: usize,
        budget: usize,
        eos: &[u32],
    ) -> (Vec<u32>, usize) {
        let mut emitted = Vec::new();
        for &token in tokens {
            if emitted.len() >= budget {
                break;
            }
            emitted.push(token);
            if eos.contains(&token) {
                break;
            }
        }
        let retained = emitted.len().min(accepted_drafts);
        (emitted, retained)
    }

    proptest! {
        #[test]
        fn limit_commit_matches_the_reference_loop(
            tokens in proptest::collection::vec(0u32..12, 0..40),
            accepted in 0usize..40, budget in 0usize..40,
            eos in proptest::collection::vec(0u32..12, 0..3),
        ) {
            let actual = limit_commit(&tokens, accepted, budget, &eos);
            prop_assert!(actual.0.len() <= budget);
            prop_assert!(tokens.starts_with(&actual.0));
            prop_assert_eq!(actual, reference_limit_commit(&tokens, accepted, budget, &eos));
        }
    }
}
