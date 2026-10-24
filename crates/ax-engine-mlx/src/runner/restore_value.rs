//! Value-priced prefix restore (`AX_MLX_PREFIX_RESTORE_VALUE`).
//!
//! The historical decision restores the longest resident prefix. That is
//! not always the cheapest completion: extending an already-restored RAM
//! prefix by a few tokens can cost more (materialization + repage) than
//! recomputing those tokens, and a short durable prefix can be dominated by
//! the best RAM prefix. This module prices each candidate against the best
//! resident RAM prefix:
//!
//! `benefit(p) = C(p) - C(m) - restore_us(p)`
//!
//! where `C` is fresh prefill cost on the same artifact/execution path,
//! `m` is the best RAM-resident prefix, and `restore_us` covers read,
//! verification, materialization and repage. A strictly-greater-than
//! [`RESTORE_SAFETY_MARGIN_US`] benefit is required; unknown cost inputs
//! fail closed (recompute / RAM only). Ties prefer the longer prefix.
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
/// best resident RAM prefix. Returns `None` when no candidate clears the
/// margin, when the prefill rate is unknown, or when a candidate's restore
/// cost cannot be priced.
///
/// `min_tokens` is the shortest prefix this choice may use — the
/// scheduler claim. A shorter restore leaves a positional gap that the
/// caller can only replay in full (linear state cannot be trimmed), so it
/// is strictly worse than taking the claim; such candidates are never
/// chosen. The claim itself stays eligible: it is the reference restore
/// this choice has to justify replacing.
pub(crate) fn choose_restore_candidate(
    candidates: &[RestoreCandidate],
    ram_prefix_tokens: u32,
    min_tokens: u32,
    prefill_us: impl Fn(u32) -> Option<u64>,
    restore_us: impl Fn(&RestoreCandidate) -> Option<u64>,
) -> Option<usize> {
    let ram_cost = prefill_us(ram_prefix_tokens)?;
    let mut best: Option<(usize, i128)> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        // The RAM baseline is an *exclusive* floor: restoring a prefix the
        // cache would reuse anyway saves nothing. The scheduler claim is an
        // *inclusive* one: a candidate exactly at the claim must stay
        // eligible, otherwise the priced path can never choose the claim it
        // is supposed to compete against and returns no restore at all when
        // no longer candidate exists.
        if candidate.token_count <= ram_prefix_tokens || candidate.token_count < min_tokens {
            continue;
        }
        let Some(cost) = prefill_us(candidate.token_count) else {
            continue;
        };
        let Some(restore) = restore_us(candidate) else {
            continue;
        };
        // Extra prefill from the RAM prefix to the candidate, then restore.
        let benefit = i128::from(cost)
            .saturating_sub(i128::from(ram_cost))
            .saturating_sub(i128::from(restore));
        if benefit <= i128::from(RESTORE_SAFETY_MARGIN_US) {
            continue;
        }
        match best {
            Some((_, best_benefit))
                if benefit < best_benefit
                    || (benefit == best_benefit
                        && candidate.token_count
                            <= candidates[best.map(|(i, _)| i).unwrap_or(index)].token_count) => {}
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
    fn restore_value_uses_ram_baseline() {
        // Fresh prefill = 2 us/token, restore = 1 us/token.
        let prefill = |tokens: u32| Some(u64::from(tokens) * 2);
        let restore = |c: &RestoreCandidate| Some(u64::from(c.token_count));
        // 1000-token candidate over an 800-token RAM prefix only saves
        // 400us - 1000us < 0, so the RAM prefix wins.
        assert_eq!(
            choose_restore_candidate(&[candidate(1000)], 800, 0, prefill, restore),
            None
        );
        // 2000 tokens over the same baseline saves 2400us - 2000us = 400us,
        // still under the 10_000us margin.
        assert_eq!(
            choose_restore_candidate(&[candidate(2000)], 800, 0, prefill, restore),
            None
        );
        // A 20_000-token candidate saves 24_000us - 20_000us = 4_000us...
        // extend the rate so the margin is cleared and the same baseline
        // still matters: 100k tokens saves 100_000us with 100k restore us.
        let prefill_fast = |tokens: u32| Some(u64::from(tokens) * 50);
        assert_eq!(
            choose_restore_candidate(&[candidate(2000)], 800, 0, prefill_fast, restore),
            Some(0)
        );
        // Raising the RAM baseline to the candidate length removes it.
        assert_eq!(
            choose_restore_candidate(&[candidate(2000)], 2000, 0, prefill_fast, restore),
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
            choose_restore_candidate(&candidates, 0, 0, prefill, tie_restore),
            None,
            "equal benefit must not clear the strict margin"
        );
        let win = |c: &RestoreCandidate| Some(u64::from(c.token_count) * 100 - 20_000);
        assert_eq!(
            choose_restore_candidate(&candidates, 0, 0, prefill, win),
            Some(1),
            "equal benefit prefers the longer prefix"
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
            choose_restore_candidate(&[candidate(1000)], 0, 2000, prefill, free),
            None,
            "a prefix shorter than the scheduler claim must not be chosen"
        );
        // The claim itself (and anything longer) stays eligible.
        assert_eq!(
            choose_restore_candidate(&[candidate(2000)], 0, 2000, prefill, free),
            Some(0)
        );
        // The inclusive claim floor must not weaken the exclusive RAM floor:
        // a candidate that only ties a resident RAM prefix still loses, even
        // when it is exactly the scheduler claim.
        assert_eq!(
            choose_restore_candidate(&[candidate(2000)], 2000, 2000, prefill, free),
            None
        );
    }

    #[test]
    fn unknown_costs_fail_closed() {
        let unknown = |_: u32| None;
        let restore = |_: &RestoreCandidate| Some(0);
        assert_eq!(
            choose_restore_candidate(&[candidate(4096)], 0, 0, unknown, restore),
            None
        );
        let prefill = |tokens: u32| Some(u64::from(tokens));
        assert_eq!(
            choose_restore_candidate(&[candidate(4096)], 0, 0, prefill, |_| None),
            None
        );
    }
}
