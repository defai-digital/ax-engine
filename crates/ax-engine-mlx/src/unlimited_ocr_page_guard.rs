//! Output page-count contract for Unlimited-OCR-2.
//!
//! The released model's generation keeps the output page count equal to the
//! prompt's: one `<PAGE>` marker per input page, EOS only after the last page,
//! and a per-page token cap. Without it, global-view multi-page requests can
//! keep emitting extra pages. The guard turns that contract into a per-step
//! allowed-token mask for the host sampler.

/// Per-page output cap (the released generation's default).
pub(crate) const MAX_PAGE_TOKENS: u32 = 20_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PageCountGuard {
    expected_pages: u32,
    page_token_id: u32,
    max_page_tokens: u32,
    scanned: usize,
    pages_out: u32,
    /// Tokens after the most recent `<PAGE>` marker (or since the start).
    page_len: u32,
}

/// What the next token may be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PageStep {
    /// Pages are still missing: anything but a terminal token.
    NoTerminal,
    /// Only a terminal token (too many pages, or the last page hit its cap).
    TerminalOnly,
    /// Only the next `<PAGE>` marker (a page hit its cap with pages missing).
    PageOnly,
    Unconstrained,
}

impl PageCountGuard {
    /// `None` when the prompt carries no `<PAGE>` marker.
    pub(crate) fn from_prompt(prompt: &[u32], page_token_id: u32) -> Option<Self> {
        let expected_pages = prompt.iter().filter(|&&t| t == page_token_id).count();
        (expected_pages > 0).then(|| Self {
            expected_pages: u32::try_from(expected_pages).unwrap_or(u32::MAX),
            page_token_id,
            max_page_tokens: MAX_PAGE_TOKENS,
            scanned: 0,
            pages_out: 0,
            page_len: 0,
        })
    }

    #[cfg(test)]
    fn with_max_page_tokens(mut self, max_page_tokens: u32) -> Self {
        self.max_page_tokens = max_page_tokens;
        self
    }

    /// Account for tokens appended to `generated` since the last call.
    pub(crate) fn observe(&mut self, generated: &[u32]) {
        if generated.len() < self.scanned {
            // The output was rewound (for example a rollback); rescan it.
            self.scanned = 0;
            self.pages_out = 0;
            self.page_len = 0;
        }
        for &token in &generated[self.scanned..] {
            if token == self.page_token_id {
                self.pages_out = self.pages_out.saturating_add(1);
                self.page_len = 0;
            } else {
                self.page_len = self.page_len.saturating_add(1);
            }
        }
        self.scanned = generated.len();
    }

    pub(crate) fn step(&self) -> PageStep {
        if self.pages_out > self.expected_pages {
            PageStep::TerminalOnly
        } else if self.max_page_tokens > 0 && self.page_len >= self.max_page_tokens {
            if self.pages_out < self.expected_pages {
                PageStep::PageOnly
            } else {
                PageStep::TerminalOnly
            }
        } else if self.pages_out < self.expected_pages {
            PageStep::NoTerminal
        } else {
            PageStep::Unconstrained
        }
    }

    /// Allowed-token bitset for the next sample (`None` = no constraint).
    /// Without terminal tokens the contract cannot be expressed; the guard
    /// then only forces page markers on capped pages.
    pub(crate) fn mask(&self, vocab_size: usize, terminal_token_ids: &[u32]) -> Option<Vec<u64>> {
        let words = vocab_size.div_ceil(64);
        let allow_only = |tokens: &[u32]| {
            let mut mask = vec![0_u64; words];
            for &token in tokens {
                if let Some(word) = mask.get_mut(token as usize / 64) {
                    *word |= 1_u64 << (token % 64);
                }
            }
            mask
        };
        match self.step() {
            PageStep::Unconstrained => None,
            PageStep::PageOnly => Some(allow_only(&[self.page_token_id])),
            PageStep::TerminalOnly if terminal_token_ids.is_empty() => None,
            PageStep::TerminalOnly => Some(allow_only(terminal_token_ids)),
            PageStep::NoTerminal if terminal_token_ids.is_empty() => None,
            PageStep::NoTerminal => {
                let mut mask = vec![u64::MAX; words];
                if !vocab_size.is_multiple_of(64)
                    && let Some(last) = mask.last_mut()
                {
                    *last = (1_u64 << (vocab_size % 64)) - 1;
                }
                for &token in terminal_token_ids {
                    if let Some(word) = mask.get_mut(token as usize / 64) {
                        *word &= !(1_u64 << (token % 64));
                    }
                }
                Some(mask)
            }
        }
    }
}

/// Intersection of two optional allowed-token masks.
pub(crate) fn intersect_masks(a: Option<Vec<u64>>, b: Option<Vec<u64>>) -> Option<Vec<u64>> {
    match (a, b) {
        (Some(mut a), Some(b)) => {
            a.truncate(b.len());
            for (word, other) in a.iter_mut().zip(&b) {
                *word &= other;
            }
            Some(a)
        }
        (a, None) => a,
        (None, b) => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_core::grammar::mask_allows;

    const PAGE: u32 = 7;
    const EOS: u32 = 1;
    const VOCAB: usize = 130;

    fn guard(pages: usize) -> PageCountGuard {
        let mut prompt = vec![0];
        for _ in 0..pages {
            prompt.extend([PAGE, 5]);
        }
        PageCountGuard::from_prompt(&prompt, PAGE).expect("prompt has pages")
    }

    #[test]
    fn prompt_without_page_markers_has_no_guard() {
        assert!(PageCountGuard::from_prompt(&[0, 5, 6], PAGE).is_none());
    }

    #[test]
    fn terminal_is_banned_until_every_page_has_started() {
        let mut g = guard(2);
        assert_eq!(g.step(), PageStep::NoTerminal);
        g.observe(&[PAGE, 20, 21]);
        assert_eq!(g.step(), PageStep::NoTerminal);
        let mask = g.mask(VOCAB, &[EOS]).expect("terminal banned");
        assert!(!mask_allows(&mask, EOS));
        assert!(mask_allows(&mask, PAGE) && mask_allows(&mask, 21));
        assert!(mask_allows(&mask, (VOCAB - 1) as u32));
        assert!(
            !mask_allows(&mask, VOCAB as u32),
            "no bits past the vocabulary"
        );
        g.observe(&[PAGE, 20, 21, PAGE, 22]);
        assert_eq!(g.step(), PageStep::Unconstrained);
        assert!(g.mask(VOCAB, &[EOS]).is_none());
    }

    #[test]
    fn an_extra_page_allows_only_terminal() {
        let mut g = guard(2);
        g.observe(&[PAGE, 20, PAGE, 21, PAGE]);
        assert_eq!(g.step(), PageStep::TerminalOnly);
        let mask = g.mask(VOCAB, &[EOS, 3]).expect("terminal only");
        assert!(mask_allows(&mask, EOS) && mask_allows(&mask, 3));
        assert!(!mask_allows(&mask, PAGE) && !mask_allows(&mask, 20));
    }

    #[test]
    fn capped_page_forces_next_marker_or_terminal() {
        let mut g = guard(2).with_max_page_tokens(3);
        g.observe(&[PAGE, 20, 21, 22]);
        assert_eq!(g.step(), PageStep::PageOnly);
        let mask = g.mask(VOCAB, &[EOS]).expect("page only");
        assert!(mask_allows(&mask, PAGE) && !mask_allows(&mask, EOS) && !mask_allows(&mask, 20));
        g.observe(&[PAGE, 20, 21, 22, PAGE, 30, 31, 32]);
        assert_eq!(g.step(), PageStep::TerminalOnly);
    }

    #[test]
    fn observe_is_incremental_and_handles_a_rewind() {
        let mut g = guard(1);
        let mut out = vec![PAGE, 20];
        g.observe(&out);
        out.push(PAGE);
        g.observe(&out);
        assert_eq!(g.step(), PageStep::TerminalOnly);
        out.truncate(2);
        g.observe(&out);
        assert_eq!(g.step(), PageStep::Unconstrained);
    }

    #[test]
    fn without_terminal_ids_only_capped_pages_are_constrained() {
        let mut g = guard(2).with_max_page_tokens(2);
        assert!(g.mask(VOCAB, &[]).is_none());
        g.observe(&[PAGE, 20, 21]);
        assert!(g.mask(VOCAB, &[]).is_some());
    }

    #[test]
    fn guarded_requests_stay_on_the_per_item_host_sampler() {
        let sampling = crate::sampling::MlxSamplingParams::greedy().with_page_guard_active(true);
        assert!(sampling.uses_logits_processors());
        // The batched paths do not install per-item token masks.
        assert!(crate::batched_sampling::batched_sampling_class(sampling, true).is_none());
    }

    #[test]
    fn masks_intersect() {
        let a = Some(vec![0b1110_u64]);
        let b = Some(vec![0b0111_u64]);
        assert_eq!(intersect_masks(a.clone(), b), Some(vec![0b0110]));
        assert_eq!(intersect_masks(a.clone(), None), a);
        assert_eq!(intersect_masks(None, None), None);
    }
}
