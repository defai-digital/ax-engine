//! Request-owned n-gram lookup planning for Flash Next.
//!
//! Hash constants and head ranges come from checkpoint buffers. Planning is
//! immutable so failed row reads cannot advance the request's token history.

#[derive(Clone, Debug)]
pub struct NgramLayout {
    vocabulary: u32,
    eos: u32,
    multipliers: Vec<u64>,
    heads_per_order: usize,
    head_sizes: Vec<u64>,
    head_offsets: Vec<u64>,
}

impl NgramLayout {
    pub fn new(
        vocabulary: u32,
        eos: u32,
        multipliers: Vec<u64>,
        heads_per_order: usize,
        head_sizes: Vec<u64>,
        head_offsets: Vec<u64>,
        table_rows: u64,
    ) -> Result<Self, String> {
        if vocabulary == 0 || eos >= vocabulary || multipliers.len() < 2 || heads_per_order == 0 {
            return Err("invalid Flash Next n-gram vocabulary, EOS or order".to_string());
        }
        let heads = (multipliers.len() - 1)
            .checked_mul(heads_per_order)
            .ok_or("n-gram head count overflow")?;
        if head_sizes.len() != heads || head_offsets.len() != heads {
            return Err("n-gram head buffers do not match orders and heads_per_ngram".to_string());
        }
        // Checkpoint hashes use nonnegative signed 64-bit arithmetic.
        if multipliers.iter().any(|multiplier| {
            *multiplier == 0
                || multiplier
                    .checked_mul(u64::from(vocabulary - 1))
                    .is_none_or(|product| product > i64::MAX as u64)
        }) {
            return Err("n-gram multiplier exceeds the signed hash domain".to_string());
        }
        let mut end = 0;
        for (&size, &offset) in head_sizes.iter().zip(&head_offsets) {
            if size == 0 || offset != end {
                return Err("n-gram head ranges must be nonempty and contiguous".to_string());
            }
            end = offset
                .checked_add(size)
                .ok_or("n-gram head range overflow")?;
            if end > table_rows {
                return Err("n-gram head range exceeds table rows".to_string());
            }
        }
        Ok(Self {
            vocabulary,
            eos,
            multipliers,
            heads_per_order,
            head_sizes,
            head_offsets,
        })
    }

    pub fn initial_history(&self) -> NgramHistory {
        NgramHistory {
            recent: vec![self.eos; self.multipliers.len() - 1],
        }
    }

    /// Validate a decoded history against this layer's hash order and
    /// vocabulary. Called only from request-state restore, before the
    /// caller adopts a new owner for the whole snapshot; [`Self::plan`]
    /// re-derives the same bound independently on every forward call.
    pub(crate) fn validate_history(&self, history: &NgramHistory) -> Result<(), String> {
        if history.recent.len() + 1 != self.multipliers.len()
            || history.recent.iter().any(|token| *token >= self.vocabulary)
        {
            return Err(
                "qwen4_exp ple n-gram history length or token is invalid for this model"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// Return token-major row IDs and the history to adopt after a successful
    /// forward. `history` remains valid if any subsequent IO or graph fails.
    pub fn plan(&self, history: &NgramHistory, tokens: &[u32]) -> Result<NgramLookup, String> {
        if history.recent.len() + 1 != self.multipliers.len()
            || history.recent.iter().any(|token| *token >= self.vocabulary)
            || tokens.iter().any(|token| *token >= self.vocabulary)
        {
            return Err("invalid n-gram history or input token".to_string());
        }
        let count = tokens
            .len()
            .checked_mul(self.head_sizes.len())
            .ok_or("n-gram row count overflow")?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(count)
            .map_err(|_| "cannot allocate n-gram lookup rows")?;
        let mut next_history = history.clone();
        for &token in tokens {
            let mut hash = u64::from(token) * self.multipliers[0];
            for order in 1..self.multipliers.len() {
                hash ^= u64::from(next_history.recent[order - 1]) * self.multipliers[order];
                let start = (order - 1) * self.heads_per_order;
                for head in start..start + self.heads_per_order {
                    rows.push(self.head_offsets[head] + hash % self.head_sizes[head]);
                }
            }
            if token == self.eos {
                next_history.recent.fill(self.eos);
            } else {
                next_history.recent.rotate_right(1);
                next_history.recent[0] = token;
            }
        }
        Ok(NgramLookup { rows, next_history })
    }
}

/// Most recent token first; the immutable layout belongs to shared weights,
/// while each request owns this history and its checkpoints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NgramHistory {
    recent: Vec<u32>,
}

impl NgramHistory {
    /// Crate-private restore path for durable snapshots. Length and
    /// vocabulary legality against a specific layer's hash order is
    /// [`NgramLayout::validate_history`], checked afterward.
    pub(crate) fn from_recent(recent: Vec<u32>) -> Self {
        Self { recent }
    }

    pub(crate) fn recent(&self) -> &[u32] {
        &self.recent
    }
}

pub struct NgramLookup {
    pub rows: Vec<u64>,
    pub next_history: NgramHistory,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> NgramLayout {
        NgramLayout::new(
            8,
            7,
            vec![3, 5, 7],
            2,
            vec![11, 13, 17, 19],
            vec![0, 11, 24, 41],
            60,
        )
        .unwrap()
    }

    #[test]
    fn chunked_lookup_matches_whole_prompt_across_eos() {
        let layout = layout();
        let tokens = [1, 2, 3, 7, 4, 7, 7, 5];
        let initial = layout.initial_history();
        let whole = layout.plan(&initial, &tokens).unwrap();
        for split in 0..=tokens.len() {
            let first = layout.plan(&initial, &tokens[..split]).unwrap();
            let second = layout.plan(&first.next_history, &tokens[split..]).unwrap();
            assert_eq!([first.rows, second.rows].concat(), whole.rows);
            assert_eq!(second.next_history, whole.next_history);
        }
        let mut history = initial;
        let mut rows = Vec::new();
        for token in tokens {
            let step = layout.plan(&history, &[token]).unwrap();
            rows.extend(step.rows);
            history = step.next_history;
        }
        assert_eq!(rows, whole.rows);
        assert_eq!(history, whole.next_history);
    }

    #[test]
    fn checkpoint_hash_ranges_and_eos_boundary_are_exact() {
        let layout = layout();
        // First bigram: (1*3) XOR (7*5) = 32; trigram adds XOR (7*7) = 17.
        let first = layout.plan(&layout.initial_history(), &[1]).unwrap();
        assert_eq!(first.rows, vec![10, 17, 24, 58]);
        let terminated = layout.plan(&first.next_history, &[7]).unwrap();
        assert_eq!(terminated.next_history, layout.initial_history());
        assert_eq!(
            layout.plan(&terminated.next_history, &[1]).unwrap().rows,
            first.rows
        );
    }

    #[test]
    fn abandoned_lookup_and_fork_do_not_advance_the_original_history() {
        let layout = layout();
        let original = layout
            .plan(&layout.initial_history(), &[1, 2])
            .unwrap()
            .next_history;
        let checkpoint = original.clone();
        let failed_io_plan = layout.plan(&original, &[3, 4]).unwrap();
        drop(failed_io_plan);
        assert_eq!(original, checkpoint);
        let other_request = layout.plan(&checkpoint, &[6]).unwrap();
        assert_eq!(
            layout.plan(&original, &[5]).unwrap().rows,
            layout.plan(&checkpoint, &[5]).unwrap().rows
        );
        assert_ne!(other_request.next_history, original);
        assert!(layout.plan(&original, &[8]).is_err());
        assert_eq!(original, checkpoint);
    }

    #[test]
    fn invalid_hash_layouts_fail_before_lookup() {
        for (multipliers, sizes, offsets, rows) in [
            (vec![3], vec![11], vec![0], 11),
            (vec![u64::MAX, 5], vec![11], vec![0], 11),
            (vec![3, 5], vec![0], vec![0], 11),
            (vec![3, 5], vec![11], vec![1], 12),
            (vec![3, 5], vec![11], vec![0], 10),
        ] {
            assert!(NgramLayout::new(8, 7, multipliers, 1, sizes, offsets, rows).is_err());
        }
    }

    #[test]
    fn lookup_matches_pinned_transformers_oracle() {
        #[derive(serde::Deserialize)]
        struct Oracle {
            vocabulary: u32,
            eos: u32,
            multipliers: Vec<u64>,
            heads_per_order: usize,
            head_sizes: Vec<u64>,
            head_offsets: Vec<u64>,
            table_rows: u64,
            tokens: Vec<u32>,
            rows: Vec<Vec<u64>>,
        }
        let oracle: Oracle =
            serde_json::from_str(include_str!("../tests/fixtures/flash_next/ngram.json")).unwrap();
        let layout = NgramLayout::new(
            oracle.vocabulary,
            oracle.eos,
            oracle.multipliers,
            oracle.heads_per_order,
            oracle.head_sizes,
            oracle.head_offsets,
            oracle.table_rows,
        )
        .unwrap();
        let expected: Vec<u64> = oracle.rows.into_iter().flatten().collect();
        let result = layout
            .plan(&layout.initial_history(), &oracle.tokens)
            .unwrap();
        assert_eq!(result.rows, expected);
        for split in 0..=oracle.tokens.len() {
            let prefix = layout
                .plan(&layout.initial_history(), &oracle.tokens[..split])
                .unwrap();
            let suffix = layout
                .plan(&prefix.next_history, &oracle.tokens[split..])
                .unwrap();
            assert_eq!([prefix.rows, suffix.rows].concat(), expected);
        }
    }
}
