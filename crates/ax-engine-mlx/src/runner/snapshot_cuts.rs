//! Snapshot cut planning (`AX_MLX_PREFIX_SEMANTIC_CUTS`).
//!
//! Cold and warm prefill use the same plan so a restored snapshot replays
//! the exact same partition. Semantic cut points come from assistant-turn
//! token starts supplied by the request edge; the terminal anchor keeps a
//! safe block-aligned fallback even when no semantic start is usable.
//!
//! Every cut is a token index strictly below `prompt_len`, semantic starts
//! are at least [`SEMANTIC_CUT_MIN_SPACING_TOKENS`] apart, and at most
//! [`MAX_SEMANTIC_CUTS`] of the most recent semantic starts are kept. The
//! anchor is aligned to `lcm(block_size_tokens, SEMANTIC_CUT_MIN_SPACING_TOKENS)`.

/// Minimum distance between two semantic cut points.
pub(crate) const SEMANTIC_CUT_MIN_SPACING_TOKENS: usize = 256;

/// Most recent semantic cut points kept per request.
pub(crate) const MAX_SEMANTIC_CUTS: usize = 3;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SnapshotCutPlan {
    pub(crate) semantic_starts: Vec<u32>,
    /// Strictly increasing token indices, each `< prompt_len`.
    pub(crate) cuts: Vec<usize>,
    /// Execution contract this plan was produced under.
    pub(crate) execution_signature: String,
}

/// Stable execution boundaries. Retention may keep only the last three turns,
/// but execution must retain earlier boundaries when a conversation grows.
pub(crate) fn semantic_execution_cuts(
    start: usize,
    end: usize,
    prompt_len: usize,
    grid: usize,
    semantic_starts: &[u32],
) -> Vec<usize> {
    if grid == 0 || start >= end {
        return Vec::new();
    }
    let mut cuts = Vec::new();
    let mut last_semantic = None;
    for cut in semantic_starts.iter().map(|&n| n as usize) {
        if cut == 0
            || cut >= prompt_len
            || last_semantic.is_some_and(|last: usize| {
                cut < last.saturating_add(SEMANTIC_CUT_MIN_SPACING_TOKENS)
            })
        {
            continue;
        }
        last_semantic = Some(cut);
        if cut > start && cut <= end {
            cuts.push(cut);
        }
    }
    let first_grid = (start / grid + 1).saturating_mul(grid);
    for cut in (first_grid..=end.min(prompt_len.saturating_sub(1))).step_by(grid) {
        cuts.push(cut);
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd(a, b) * b
}

/// Alignment of the terminal anchor.
pub(crate) fn anchor_alignment(block_size_tokens: usize) -> usize {
    lcm(block_size_tokens, SEMANTIC_CUT_MIN_SPACING_TOKENS)
}

/// Build the plan for one prompt. `semantic_cut_tokens` are assistant-turn
/// token starts from the request's single full tokenization.
pub(crate) fn snapshot_cut_plan(
    prompt_len: usize,
    block_size_tokens: usize,
    semantic_cut_tokens: &[u32],
    semantic_cuts_enabled: bool,
    execution_signature: &str,
) -> SnapshotCutPlan {
    let mut semantic: Vec<usize> = Vec::new();
    if semantic_cuts_enabled && prompt_len > 1 {
        let max_cut = prompt_len - 1;
        let mut accepted: Vec<usize> = Vec::new();
        for start in semantic_cut_tokens.iter().map(|token| *token as usize) {
            if start == 0 || start > max_cut {
                continue;
            }
            if accepted
                .last()
                .is_some_and(|last| start < last.saturating_add(SEMANTIC_CUT_MIN_SPACING_TOKENS))
            {
                continue;
            }
            accepted.push(start);
        }
        // Keep the most recent (last) semantic cut points.
        let keep_from = accepted.len().saturating_sub(MAX_SEMANTIC_CUTS);
        semantic.extend_from_slice(&accepted[keep_from..]);
    }

    let alignment = anchor_alignment(block_size_tokens);
    let mut cuts = semantic;
    if alignment > 0 && prompt_len > 1 {
        let anchor = ((prompt_len - 1) / alignment) * alignment;
        if anchor > 0 && !cuts.contains(&anchor) {
            cuts.push(anchor);
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    SnapshotCutPlan {
        semantic_starts: if semantic_cuts_enabled {
            semantic_cut_tokens.to_vec()
        } else {
            Vec::new()
        },
        cuts,
        execution_signature: execution_signature.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_keeps_early_cuts_and_is_stable_across_turns_and_items() {
        let starts = [700, 1400, 1800, 2500, 2900];
        let old = semantic_execution_cuts(0, 1930, 1930, 1024, &starts);
        assert_eq!(old, vec![700, 1024, 1400, 1800]);
        let extended = semantic_execution_cuts(0, 3200, 3200, 1024, &starts);
        assert_eq!(
            old,
            extended
                .iter()
                .copied()
                .filter(|&n| n < 1930)
                .collect::<Vec<_>>()
        );
        let mut split = semantic_execution_cuts(0, 2048, 3200, 1024, &starts);
        split.extend(semantic_execution_cuts(2048, 3200, 3200, 1024, &starts));
        assert_eq!(split, extended);
    }

    #[test]
    fn snapshot_cut_plan_bounds_and_spacing() {
        let starts = [0u32, 100, 300, 700, 900, 5000];
        let plan = snapshot_cut_plan(4096, 32, &starts, true, "sig");
        assert!(plan.cuts.iter().all(|cut| *cut < 4096));
        assert!(plan.cuts.windows(2).all(|w| w[1] >= w[0] + 256));
        // 0 is not a cut; 100 is too close to the alignment anchor 3840?
        // 100 is retained, 300 rejected (spacing), 700/900 kept, 5000 out of
        // bounds, and the anchor 3840 is appended.
        assert_eq!(plan.cuts, vec![100, 700, 3840]);
        assert_eq!(plan.execution_signature, "sig");
    }

    #[test]
    fn snapshot_cut_plan_keeps_terminal_anchor_without_semantics() {
        let plan = snapshot_cut_plan(4096, 32, &[], true, "sig");
        assert_eq!(plan.cuts, vec![3840]);
        let disabled = snapshot_cut_plan(4096, 32, &[512], false, "sig");
        assert_eq!(disabled.cuts, vec![3840]);
    }

    #[test]
    fn snapshot_cut_plan_keeps_three_most_recent_semantic_cuts() {
        let starts = [512u32, 1024, 2048, 3072];
        let plan = snapshot_cut_plan(4096, 32, &starts, true, "sig");
        assert_eq!(plan.cuts, vec![1024, 2048, 3072, 3840]);
    }

    #[test]
    fn gdn_cut_resume_matches_same_partition() {
        // Cold and warm paths derive the plan from the same prompt and
        // block size; a resumed prefill therefore sees the identical
        // partition (the cut is where the full conv/recurrent state was
        // captured, never a trimmed one).
        let starts = [700u32, 1400, 2600];
        let cold = snapshot_cut_plan(4000, 64, &starts, true, "gdn-sig");
        let warm = snapshot_cut_plan(4000, 64, &starts, true, "gdn-sig");
        assert_eq!(cold, warm);
        assert!(cold.cuts.iter().all(|cut| *cut < 4000));
        // Partition prefix + cuts covers the prompt without gaps.
        let mut covered = 0usize;
        for cut in &cold.cuts {
            assert!(*cut > covered);
            covered = *cut;
        }
        assert!(covered > 0 && covered < 4000);
    }
}
