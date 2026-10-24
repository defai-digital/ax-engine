//! Value-priced prefix restore (`AX_MLX_PREFIX_RESTORE_VALUE`).
//!
//! The historical decision restores the longest available prefix. That is
//! not always the cheapest completion: extending an already-restored prefix
//! by a few tokens can cost more (materialization + repage) than recomputing
//! those tokens, and a short durable prefix can be dominated by the prefix
//! the request would restore anyway. This module prices each candidate
//! against that off-path baseline:
//!
//! `benefit(p) = [C(p) - C(b)] - [restore_us(p) - restore_us(b)]`
//!
//! where `C` is fresh prefill cost on the same artifact/execution path, `b` is
//! the prefix the request would restore without the value path (the scheduler
//! claim, or the unpriced probe result when there is no claim), and
//! `restore_us` covers read, verification, materialization and repage. Both
//! restores pay their own materialization, so only the *incremental* restore
//! cost is charged. That distinction decides whether the feature can ever fire
//! on a hybrid (linear-attention) model: their snapshot carries a large
//! fixed-size recurrent state, so charging the candidate the whole state while
//! pricing the baseline restore at zero rejects every option, including ones
//! that genuinely save hundreds of milliseconds of prefill. When the baseline
//! is not a restore at all (`b == 0`, nothing restorable) the incremental cost
//! is the whole restore cost, which is the conservative case.
//!
//! Only candidates strictly longer than `b` can win — restoring less than the
//! baseline would leave tokens to recompute that the baseline already covers.
//! A strictly-greater-than [`RESTORE_SAFETY_MARGIN_US`] benefit is required;
//! unknown cost inputs fail closed (recompute / baseline only). Ties prefer
//! the longer prefix.
//!
//! Pure decision logic plus a process-local prefill-rate EWMA. Never part
//! of correctness.

use parking_lot::Mutex;

use crate::disk_prefix_cache::RestoreCandidate;

/// Safety margin: restore must save strictly more than this to be chosen.
pub(crate) const RESTORE_SAFETY_MARGIN_US: u64 = 10_000;

/// EWMA smoothing for the observed fresh-prefill rate.
const PREFILL_EWMA_ALPHA: f64 = 0.2;

static PREFILL_US_PER_TOKEN: Mutex<Option<f64>> = Mutex::new(None);

/// Record one measured fresh prefill (`tokens` in `wall_us`).
pub(crate) fn record_prefill(tokens: usize, wall_us: u64) {
    if tokens == 0 || wall_us == 0 {
        return;
    }
    let observed = wall_us as f64 / tokens as f64;
    let mut rate = PREFILL_US_PER_TOKEN.lock();
    *rate = Some(match *rate {
        Some(current) => current * (1.0 - PREFILL_EWMA_ALPHA) + observed * PREFILL_EWMA_ALPHA,
        None => observed,
    });
}

/// Estimated fresh-prefill wall time for `tokens`, or `None` when no
/// prefill has been observed yet (the caller must then recompute).
pub(crate) fn estimated_prefill_us(tokens: u32) -> Option<u64> {
    let rate = (*PREFILL_US_PER_TOKEN.lock())?;
    if !rate.is_finite() || rate <= 0.0 {
        return None;
    }
    let us = rate * f64::from(tokens);
    (us < u64::MAX as f64).then_some(us as u64)
}

/// Pick the candidate index with the largest positive benefit over the
/// off-path baseline restore. Returns `None` when no candidate clears the
/// margin, when the prefill rate is unknown, or when a candidate's restore
/// cost cannot be priced — the caller then keeps the baseline.
///
/// `baseline_tokens` is the prefix the request restores without the value
/// path: the scheduler claim when one exists, otherwise the unpriced probe
/// result. Candidates must be *strictly* longer than it; a candidate at the
/// baseline saves nothing (the baseline already covers those tokens) and a
/// shorter one would leave a positional gap the caller can only replay in
/// full (linear state cannot be trimmed).
///
/// `min_tokens` is the shortest prefix the scheduler can accept (the claim).
/// With `baseline_tokens` derived from the claim the two floors coincide for
/// claimed requests; the explicit floor still rejects shorter candidates for
/// an unclaimed request whose baseline is a probe result.
///
/// `baseline_restore_us` is what the baseline restore itself costs. It is
/// charged against the candidate so that a hybrid model's fixed-size recurrent
/// state — which both restores pay — cannot veto a restore that saves more
/// prefill than the extra bytes cost. Pass `0` when the baseline is not a
/// restore (nothing restorable): the candidate then pays its full cost.
pub(crate) fn choose_restore_candidate(
    candidates: &[RestoreCandidate],
    baseline_tokens: u32,
    min_tokens: u32,
    baseline_restore_us: u64,
    prefill_us: impl Fn(u32) -> Option<u64>,
    restore_us: impl Fn(&RestoreCandidate) -> Option<u64>,
) -> Option<usize> {
    // Unknown baseline cost: nothing can be priced against it, so keep the
    // baseline instead of inventing a saving.
    let baseline_cost = prefill_us(baseline_tokens)?;
    let mut best: Option<(usize, i128)> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.token_count <= baseline_tokens || candidate.token_count < min_tokens {
            continue;
        }
        let Some(cost) = prefill_us(candidate.token_count) else {
            continue;
        };
        let Some(restore) = restore_us(candidate) else {
            continue;
        };
        // Prefill the caller saves by restoring `p` instead of the baseline,
        // minus the extra restore cost of the longer prefix. Both restores pay
        // their own materialization, so only the difference is decision
        // relevant.
        let prefill_saved = i128::from(cost).saturating_sub(i128::from(baseline_cost));
        let extra_restore = i128::from(restore).saturating_sub(i128::from(baseline_restore_us));
        let benefit = prefill_saved.saturating_sub(extra_restore);
        if benefit <= i128::from(RESTORE_SAFETY_MARGIN_US) {
            continue;
        }
        match best {
            Some((best_index, best_benefit))
                if benefit < best_benefit
                    || (benefit == best_benefit
                        && candidate.token_count <= candidates[best_index].token_count) => {}
            _ => best = Some((index, benefit)),
        }
    }
    best.map(|(index, _)| index)
}

/// Reclaimable capacity for a restore: only blocks with a single owner can
/// actually be released. Shared blocks are counted once even when several
/// entries reference them, and already-free blocks contribute nothing.
///
/// `blocks` is `(block_id, block_bytes, ref_count)`.
pub(crate) fn reclaimable_bytes(blocks: &[(u32, u64, u32)]) -> u64 {
    let mut seen = std::collections::HashSet::new();
    blocks
        .iter()
        .filter(|(id, _, ref_count)| *ref_count == 1 && seen.insert(*id))
        .map(|(_, bytes, _)| *bytes)
        .fold(0u64, u64::saturating_add)
}

/// Wall time to make one candidate usable.
///
/// An L1-resident candidate pays only rehydration/materialization: the bytes
/// are already in host RAM, so there is no durable read to amortise. A
/// durable candidate additionally pays the measured read cost, and is
/// unpriceable (`None`) until a throughput sample exists.
///
/// Materialization is still a placeholder (`materialized_bytes / 1000`)
/// until a real rehydration cost is measured; it is deliberately small so a
/// pricing error cannot promote a bad restore.
pub(crate) fn restore_cost_us(
    candidate: &RestoreCandidate,
    durable_read_us: Option<u64>,
) -> Option<u64> {
    let materialize_us = candidate.materialized_bytes / 1000;
    if candidate.resident_in_l1 {
        return Some(materialize_us);
    }
    Some(durable_read_us?.saturating_add(materialize_us))
}

/// Tokens the scheduler claimed as reused but a shorter restore did not
/// cover. A non-zero gap must be recomputed at the correct positions; it is
/// never skipped.
pub(crate) fn restore_gap_tokens(claimed: usize, restored: usize) -> u32 {
    u32::try_from(claimed.saturating_sub(restored)).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(tokens: u32) -> RestoreCandidate {
        RestoreCandidate {
            token_count: tokens,
            payload_bytes: u64::from(tokens) * 10,
            materialized_bytes: u64::from(tokens) * 10,
            estimated_prefill_us: u64::from(tokens) * 2,
            key: vec![1, 2, 3],
            resident_in_l1: false,
        }
    }

    #[test]
    fn restore_value_uses_off_path_baseline() {
        // Fresh prefill = 2 us/token, restore = 1 us/token.
        let prefill = |tokens: u32| Some(u64::from(tokens) * 2);
        let restore = |c: &RestoreCandidate| Some(u64::from(c.token_count));
        // 1000-token candidate over an 800-token baseline only saves
        // 400us - 1000us < 0, so the baseline wins.
        assert_eq!(
            choose_restore_candidate(&[candidate(1000)], 800, 0, 0, prefill, restore),
            None
        );
        // 2000 tokens over the same baseline saves 2400us - 2000us = 400us,
        // still under the 10_000us margin.
        assert_eq!(
            choose_restore_candidate(&[candidate(2000)], 800, 0, 0, prefill, restore),
            None
        );
        // 2000 tokens over a 40_000us margin-clearing rate is chosen.
        let prefill_fast = |tokens: u32| Some(u64::from(tokens) * 50);
        assert_eq!(
            choose_restore_candidate(&[candidate(2000)], 800, 0, 0, prefill_fast, restore),
            Some(0)
        );
        // Raising the baseline to the candidate length removes it: the
        // candidate saves nothing over what the request already restores.
        assert_eq!(
            choose_restore_candidate(&[candidate(2000)], 2000, 0, 0, prefill_fast, restore),
            None
        );
    }

    #[test]
    fn restore_reclaim_excludes_borrowed_blocks() {
        let blocks = [(1, 100, 1), (1, 100, 1), (2, 50, 2), (3, 80, 0)];
        assert_eq!(reclaimable_bytes(&blocks), 100);
    }

    #[test]
    fn restore_tie_prefers_longer_prefix() {
        let prefill = |tokens: u32| Some(u64::from(tokens) * 100);
        let candidates = [candidate(1000), candidate(2000)];
        // Benefit is identical only if restore cost cancels the extra
        // prefill; use a restore cost that scales so p=2000 wins on tie.
        let tie_restore = |c: &RestoreCandidate| Some(u64::from(c.token_count) * 100);
        assert_eq!(
            choose_restore_candidate(&candidates, 0, 0, 0, prefill, tie_restore),
            None,
            "equal benefit must not clear the strict margin"
        );
        let win = |c: &RestoreCandidate| Some(u64::from(c.token_count) * 100 - 20_000);
        assert_eq!(
            choose_restore_candidate(&candidates, 0, 0, 0, prefill, win),
            Some(1),
            "equal benefit prefers the longer prefix"
        );
    }

    #[test]
    fn value_path_chooses_a_cut_beyond_the_claim() {
        // A resident semantic cut sits 30 tokens past the scheduler claim and
        // is priced against the claim restore: prefill 4_000us/token, restore
        // 300us/token (payload materialization). 30 * 4_000 - 350 * 300 =
        // 15_000us clears the 10_000us margin, so the cut is chosen.
        let prefill = |tokens: u32| Some(u64::from(tokens) * 4_000);
        let restore = |c: &RestoreCandidate| Some(u64::from(c.token_count) * 300);
        assert_eq!(
            choose_restore_candidate(&[candidate(350)], 320, 320, 0, prefill, restore),
            Some(0)
        );
        // The claim itself is never a candidate: it is the baseline.
        assert_eq!(
            choose_restore_candidate(&[candidate(320)], 320, 320, 0, prefill, restore),
            None
        );
        // A cut too close to the claim cannot pay for its own materialization.
        assert_eq!(
            choose_restore_candidate(&[candidate(325)], 320, 320, 0, prefill, restore),
            None,
            "5 tokens saved (20_000us) minus 97_500us of materialization must lose"
        );
    }

    #[test]
    fn value_path_charges_only_the_incremental_restore() {
        // Hybrid models (linear attention) carry a large fixed-size recurrent
        // state, so every restore costs ~155_000us regardless of prefix length.
        // Both the baseline and the candidate pay it; charging the candidate
        // the whole amount while the baseline restore is free would reject a
        // cut that saves 63ms of prefill. Measured shape: prefill 9_076us/token,
        // 65.5us per extra token of payload.
        let prefill = |tokens: u32| Some(u64::from(tokens) * 9_076);
        let fixed = 154_600u64;
        let restore = |c: &RestoreCandidate| Some(fixed + u64::from(c.token_count) * 66);
        // Claim 400, cut 415: 15 tokens * 9_076 = 136_140us saved, 15 tokens of
        // extra payload = 990us. Clears the margin by a wide gap.
        assert_eq!(
            choose_restore_candidate(
                &[candidate(415)],
                400,
                400,
                restore(&candidate(400)).unwrap(),
                prefill,
                restore
            ),
            Some(0),
            "a 15-token cut that saves 136ms must be chosen"
        );
        // The same candidate priced against a free baseline restore loses: this
        // is the model the incremental charge replaces.
        assert_eq!(
            choose_restore_candidate(&[candidate(415)], 400, 400, 0, prefill, restore),
            None,
            "charging the whole fixed state to the candidate would veto the restore"
        );
    }

    #[test]
    fn value_path_prices_a_longer_disk_entry_over_the_claim() {
        // Durable entry one block past the claim: read + materialization is
        // priced, and the saved prefill still clears the margin.
        let prefill = |tokens: u32| Some(u64::from(tokens) * 4_000);
        let mut durable = candidate(384);
        durable.resident_in_l1 = false;
        durable.materialized_bytes = 0;
        assert_eq!(
            choose_restore_candidate(&[durable], 320, 320, 0, prefill, |_| Some(5_000)),
            Some(0)
        );
    }

    #[test]
    fn unknown_baseline_cost_keeps_the_baseline() {
        // The rate is known for the candidate but not for the baseline: the
        // saving cannot be computed, so the baseline is kept.
        let prefill = |tokens: u32| (tokens != 320).then_some(u64::from(tokens) * 4_000);
        let restore = |c: &RestoreCandidate| Some(u64::from(c.token_count));
        assert_eq!(
            choose_restore_candidate(&[candidate(350)], 320, 320, 0, prefill, restore),
            None
        );
    }

    #[test]
    fn short_restore_recomputes_scheduler_gap() {
        assert_eq!(restore_gap_tokens(2048, 2048), 0);
        assert_eq!(restore_gap_tokens(2048, 1536), 512);
        assert_eq!(restore_gap_tokens(0, 0), 0);
        // A restore longer than the claim is not a gap (the caller trims to
        // the claim); saturating arithmetic keeps this total.
        assert_eq!(restore_gap_tokens(512, 2048), 0);
    }

    #[test]
    fn resident_restore_skips_durable_read() {
        // An L1-resident candidate pays only materialization: its bytes are
        // already in host RAM, so a missing throughput sample must not make
        // it unpriceable (semantic cuts live in L1 with no disk layer).
        let mut resident = candidate(1000);
        resident.resident_in_l1 = true;
        resident.materialized_bytes = 4_000;
        assert_eq!(restore_cost_us(&resident, None), Some(4));
        // A durable candidate without a throughput sample is unpriceable.
        assert_eq!(restore_cost_us(&candidate(1000), None), None);
        // With a sample, durable = read + materialization.
        assert_eq!(restore_cost_us(&candidate(1000), Some(7)), Some(7 + 10));
    }

    #[test]
    fn shorter_candidate_never_beats_taking_the_scheduler_claim() {
        // Restoring 1000 of a 2000-token claim would leave a positional gap
        // that only a full replay can cover, which is worse than replaying
        // the claim itself. The claim is a hard floor even when the short
        // prefix is trivially cheap to restore.
        let prefill = |tokens: u32| Some(u64::from(tokens) * 10);
        let free = |_: &RestoreCandidate| Some(0);
        assert_eq!(
            choose_restore_candidate(&[candidate(1000)], 0, 2000, 0, prefill, free),
            None,
            "a prefix shorter than the scheduler claim must not be chosen"
        );
        // The claim itself (and anything longer) stays eligible.
        assert_eq!(
            choose_restore_candidate(&[candidate(2000)], 0, 2000, 0, prefill, free),
            Some(0)
        );
        // A candidate at the baseline saves nothing, even when it is exactly
        // the scheduler claim.
        assert_eq!(
            choose_restore_candidate(&[candidate(2000)], 2000, 2000, 0, prefill, free),
            None
        );
    }

    #[test]
    fn unknown_costs_fail_closed() {
        let unknown = |_: u32| None;
        let restore = |_: &RestoreCandidate| Some(0);
        assert_eq!(
            choose_restore_candidate(&[candidate(4096)], 0, 0, 0, unknown, restore),
            None
        );
        let prefill = |tokens: u32| Some(u64::from(tokens));
        assert_eq!(
            choose_restore_candidate(&[candidate(4096)], 0, 0, 0, prefill, |_| None),
            None
        );
    }
}
