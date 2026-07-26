//! Pure cohort planner for padded batched prefill.
//!
//! A priority-ordered prefill window can mix rows the padded batched path
//! supports with rows it must never receive (adopted-prefix rows resuming at
//! a non-zero KV offset, multimodal / custom-embedding rows, oversized
//! heads). This module partitions the window into execution cohorts without
//! reordering it, so the scheduler's priority and FIFO decisions survive the
//! split, and bounds the padded transient with an explicit `rows * max_len`
//! admission cost.
//!
//! The planner is a pure function over per-row classifications — no model,
//! GPU, or scheduler dependency — so every invariant is pinned by unit and
//! property tests:
//!
//! - **Order preserving.** Concatenating the members of the returned cohorts
//!   reproduces the input window order exactly: no row is dropped,
//!   duplicated, or hoisted past a row that preceded it.
//! - **Offset isolation.** A batched cohort contains only cold rows (zero KV
//!   offset). The padded path assumes a zero cache offset for every row, so
//!   an adopted-prefix row can never be handed a padded slot.
//! - **Multimodal atomicity.** Rows carrying multimodal or custom-embedding
//!   inputs stay on the single-sequence path.
//! - **Contiguity.** Cold rows batch only where they are contiguous in the
//!   window; a cold row separated by an incompatible row is prefilled
//!   sequentially rather than reordered into a distant batch.
//! - **Bounded padded cost.** Every batched cohort satisfies
//!   `rows * max_len <= max_padded_tokens` and `max_len <= max_padded_tokens
//!   / 2` (the oversized-head fallback), which bounds the true `[B, L, L]`
//!   mask transient — see [`padded_mask_bytes_upper_bound`].
//!
//! The execution side (a padded batched forward in the runner) is a separate
//! product decision; landing the planner first lets that work start from
//! verified cohort semantics. Execution-side tests must follow the
//! seed-determinism / cohort-parity methodology: never assert
//! "batched == single-request byte-identical" (half-precision padding jitter
//! makes it flaky); fix the cohort composition and vary only the seed, and
//! isolate per-request RNG seeding order.

/// How a contiguous group of prefill-window rows is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillCohortKind {
    /// Two or more cold text rows sharing one padded batched forward pass.
    BatchedCold,
    /// Rows prefilled one at a time on the offset-aware single-sequence
    /// path: adopted prefixes, multimodal rows, isolated or oversized cold
    /// rows, and length-incompatible rows on equal-length-only models.
    Sequential,
}

/// A planned cohort: its execution kind and the window indices of its
/// members, in window (priority) order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillCohort {
    pub kind: PrefillCohortKind,
    pub members: Vec<usize>,
}

/// Per-row classification consumed by [`plan_prefill_cohorts`]. Callers map
/// their scheduled prefill items to this shape; the planner never sees model
/// or request state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillRowClass {
    /// Tokens already resident in KV for this row (adopted prefix or warm
    /// extend). Any non-zero offset excludes the row from batching.
    pub kv_offset_tokens: u32,
    /// Row carries multimodal or custom-embedding inputs. Such rows are
    /// atomic and always take the single-sequence path.
    pub multimodal: bool,
    /// Prompt tokens scheduled for this row's prefill.
    pub prompt_len: u32,
}

impl PrefillRowClass {
    /// Eligible for a padded batched slot: zero KV offset, text-only, and a
    /// non-empty prompt.
    fn is_cold(&self) -> bool {
        self.kv_offset_tokens == 0 && !self.multimodal && self.prompt_len > 0
    }
}

/// Model-level batching capability, resolved by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedPrefillCapability {
    /// The model can run any batched prefill at all. When false the whole
    /// window is one sequential cohort.
    pub supports_batched: bool,
    /// The model can pad rows to a common length. When false a cohort may
    /// batch only rows that already share one exact length.
    pub supports_padding: bool,
}

/// Admission-cost limits for one batched cohort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaddedCostLimits {
    /// Cap on `rows * max_len` for a batched cohort; 0 disables the cap.
    /// This bounds the padded `[B, L, hidden]` forward and, because
    /// `max_len <= cap / 2` inside any admitted cohort, the `[B, L, L]`
    /// mask transient stays within `cap^2 / 2` elements.
    pub max_padded_tokens: u32,
    /// Cap on rows per batched cohort; 0 disables the cap.
    pub max_rows: u32,
}

/// Whether a batched window currently holding `count` rows padded to
/// `current_max_len` may admit one more row of `next_len` tokens under
/// `limits`. Unlike the incremental drain in MLXcel (which always admits the
/// head row and relies on a separate dispatch-time guard), an oversized head
/// is rejected here outright — [`plan_prefill_cohorts`] routes it to the
/// sequential path — so `max_len <= max_padded_tokens / 2` holds for every
/// admitted cohort by construction.
pub fn padded_window_admits(
    count: u32,
    current_max_len: u32,
    next_len: u32,
    limits: PaddedCostLimits,
) -> bool {
    if limits.max_rows != 0 && count >= limits.max_rows {
        return false;
    }
    if limits.max_padded_tokens == 0 {
        return true;
    }
    let new_max = current_max_len.max(next_len) as u64;
    // A batched cohort has at least two rows, so requiring 2 * len within
    // the budget for the FIRST row is exactly the oversized-head fallback.
    let projected_rows = u64::from(count.max(1)) + 1;
    projected_rows * new_max <= u64::from(limits.max_padded_tokens)
}

/// Partition a priority-ordered prefill window into execution cohorts.
/// See the module documentation for the guaranteed invariants.
pub fn plan_prefill_cohorts(
    rows: &[PrefillRowClass],
    capability: BatchedPrefillCapability,
    limits: PaddedCostLimits,
) -> Vec<PrefillCohort> {
    let n = rows.len();
    if n == 0 {
        return Vec::new();
    }
    if !capability.supports_batched {
        return vec![PrefillCohort {
            kind: PrefillCohortKind::Sequential,
            members: (0..n).collect(),
        }];
    }

    let mut cohorts: Vec<PrefillCohort> = Vec::new();
    // Sequential rows accumulated since the last batched flush, in window
    // order.
    let mut pending_sequential: Vec<usize> = Vec::new();
    let mut index = 0usize;
    while index < n {
        if !rows[index].is_cold() {
            pending_sequential.push(index);
            index += 1;
            continue;
        }

        // Split the maximal contiguous cold run into admission windows.
        let run_start = index;
        while index < n && rows[index].is_cold() {
            index += 1;
        }
        let mut window_start = run_start;
        while window_start < index {
            let mut window_end = window_start;
            let mut count = 0u32;
            let mut max_len = 0u32;
            let anchor_len = rows[window_start].prompt_len;
            while window_end < index {
                let len = rows[window_end].prompt_len;
                if !capability.supports_padding && len != anchor_len {
                    break;
                }
                if !padded_window_admits(count, max_len, len, limits) {
                    break;
                }
                count += 1;
                max_len = max_len.max(len);
                window_end += 1;
            }
            if window_end - window_start >= 2 {
                flush_sequential(&mut cohorts, &mut pending_sequential);
                cohorts.push(PrefillCohort {
                    kind: PrefillCohortKind::BatchedCold,
                    members: (window_start..window_end).collect(),
                });
                window_start = window_end;
            } else {
                // Isolated, oversized, or length-incompatible cold row:
                // sequential, and the window restarts at the next row.
                pending_sequential.push(window_start);
                window_start += 1;
            }
        }
    }
    flush_sequential(&mut cohorts, &mut pending_sequential);
    cohorts
}

fn flush_sequential(cohorts: &mut Vec<PrefillCohort>, pending: &mut Vec<usize>) {
    if pending.is_empty() {
        return;
    }
    cohorts.push(PrefillCohort {
        kind: PrefillCohortKind::Sequential,
        members: std::mem::take(pending),
    });
}

/// Default padded-token budget when the operator configures nothing:
/// `2 * max_batch_rows * prefill_chunk_tokens`, falling back to 512 tokens
/// per row when chunking is disabled. The factor of 2 is headroom for
/// padding slop: real "chunk-sized" prompts (chat template plus a nominal
/// body) land slightly over the chunk size, and a budget of exactly
/// `rows * chunk` would spill the last row of the motivating short-prompt
/// batch, staggering prefill and regressing p95 TTFT.
pub fn default_padded_token_budget(prefill_chunk_tokens: u32, max_batch_rows: u32) -> u32 {
    let per_row = if prefill_chunk_tokens == 0 {
        512
    } else {
        prefill_chunk_tokens
    };
    max_batch_rows
        .max(1)
        .saturating_mul(per_row)
        .saturating_mul(2)
}

/// Explicit upper bound, in bytes, of the padded `[B, L, L]` attention-mask
/// transient for any cohort admitted under `max_padded_tokens`: the mask is
/// `B * L^2 = (B * L) * L` elements, and every admitted cohort has
/// `B * L <= budget` with `L <= budget / 2`, so the mask stays within
/// `budget^2 / 2` elements. This is the number the execution side must keep
/// beside its steady-state KV budget: the transient is NOT modeled there and
/// an allocation failure aborts the process (uncatchable MLX C++ throw).
pub fn padded_mask_bytes_upper_bound(max_padded_tokens: u32, element_bytes: u32) -> u64 {
    let budget = u64::from(max_padded_tokens);
    budget * budget / 2 * u64::from(element_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cold(prompt_len: u32) -> PrefillRowClass {
        PrefillRowClass {
            kv_offset_tokens: 0,
            multimodal: false,
            prompt_len,
        }
    }

    fn adopted(prompt_len: u32) -> PrefillRowClass {
        PrefillRowClass {
            kv_offset_tokens: 64,
            multimodal: false,
            prompt_len,
        }
    }

    fn vlm(prompt_len: u32) -> PrefillRowClass {
        PrefillRowClass {
            kv_offset_tokens: 0,
            multimodal: true,
            prompt_len,
        }
    }

    const PAD_CAPABLE: BatchedPrefillCapability = BatchedPrefillCapability {
        supports_batched: true,
        supports_padding: true,
    };
    const EQUAL_LEN_ONLY: BatchedPrefillCapability = BatchedPrefillCapability {
        supports_batched: true,
        supports_padding: false,
    };
    const UNCAPPED: PaddedCostLimits = PaddedCostLimits {
        max_padded_tokens: 0,
        max_rows: 0,
    };

    fn dispatch_order(cohorts: &[PrefillCohort]) -> Vec<usize> {
        cohorts
            .iter()
            .flat_map(|cohort| cohort.members.iter().copied())
            .collect()
    }

    /// The fairness invariant: every input row exactly once, in window order.
    fn assert_order_preserved(row_count: usize, cohorts: &[PrefillCohort]) {
        assert_eq!(
            dispatch_order(cohorts),
            (0..row_count).collect::<Vec<_>>(),
            "cohorts must reproduce the window order exactly"
        );
    }

    fn assert_invariants(
        rows: &[PrefillRowClass],
        capability: BatchedPrefillCapability,
        limits: PaddedCostLimits,
        cohorts: &[PrefillCohort],
    ) {
        assert_order_preserved(rows.len(), cohorts);
        for cohort in cohorts {
            assert!(!cohort.members.is_empty(), "empty cohort");
            if cohort.kind != PrefillCohortKind::BatchedCold {
                continue;
            }
            assert!(capability.supports_batched);
            assert!(cohort.members.len() >= 2, "batched cohort needs >= 2 rows");
            // Contiguity: members are consecutive window indices.
            for pair in cohort.members.windows(2) {
                assert_eq!(pair[1], pair[0] + 1, "batched cohort must be contiguous");
            }
            let mut max_len = 0u64;
            for &member in &cohort.members {
                let row = rows[member];
                assert_eq!(row.kv_offset_tokens, 0, "offset isolation violated");
                assert!(!row.multimodal, "multimodal atomicity violated");
                assert!(row.prompt_len > 0);
                max_len = max_len.max(u64::from(row.prompt_len));
            }
            if !capability.supports_padding {
                let first = rows[cohort.members[0]].prompt_len;
                assert!(
                    cohort
                        .members
                        .iter()
                        .all(|&member| rows[member].prompt_len == first),
                    "equal-length-only cohort mixes lengths"
                );
            }
            if limits.max_rows != 0 {
                assert!(cohort.members.len() as u64 <= u64::from(limits.max_rows));
            }
            if limits.max_padded_tokens != 0 {
                let budget = u64::from(limits.max_padded_tokens);
                assert!(
                    cohort.members.len() as u64 * max_len <= budget,
                    "rows * max_len exceeds the padded budget"
                );
                assert!(
                    max_len * 2 <= budget,
                    "oversized head admitted into a batched cohort"
                );
            }
        }
    }

    #[test]
    fn empty_window_plans_nothing() {
        assert!(plan_prefill_cohorts(&[], PAD_CAPABLE, UNCAPPED).is_empty());
    }

    #[test]
    fn non_batching_model_gets_one_sequential_cohort() {
        let rows = vec![cold(8), cold(8), adopted(4)];
        let cohorts = plan_prefill_cohorts(
            &rows,
            BatchedPrefillCapability {
                supports_batched: false,
                supports_padding: false,
            },
            UNCAPPED,
        );
        assert_eq!(cohorts.len(), 1);
        assert_eq!(cohorts[0].kind, PrefillCohortKind::Sequential);
        assert_order_preserved(rows.len(), &cohorts);
    }

    #[test]
    fn contiguous_cold_rows_batch_and_incompatible_rows_split() {
        let rows = vec![cold(10), cold(20), adopted(5), cold(7), vlm(9), cold(3)];
        let cohorts = plan_prefill_cohorts(&rows, PAD_CAPABLE, UNCAPPED);
        assert_invariants(&rows, PAD_CAPABLE, UNCAPPED, &cohorts);
        assert_eq!(
            cohorts
                .iter()
                .map(|cohort| (cohort.kind, cohort.members.clone()))
                .collect::<Vec<_>>(),
            vec![
                (PrefillCohortKind::BatchedCold, vec![0, 1]),
                // The adopted row, the isolated cold row 3, the VLM row, and
                // the trailing isolated cold row drain sequentially in order.
                (PrefillCohortKind::Sequential, vec![2, 3, 4, 5]),
            ]
        );
    }

    #[test]
    fn cost_cap_splits_a_long_cold_run_without_reordering() {
        // Budget 40: rows of len 10 admit windows of 4 (4 * 10 = 40).
        let rows: Vec<_> = (0..9).map(|_| cold(10)).collect();
        let limits = PaddedCostLimits {
            max_padded_tokens: 40,
            max_rows: 0,
        };
        let cohorts = plan_prefill_cohorts(&rows, PAD_CAPABLE, limits);
        assert_invariants(&rows, PAD_CAPABLE, limits, &cohorts);
        assert_eq!(
            cohorts
                .iter()
                .map(|cohort| (cohort.kind, cohort.members.len()))
                .collect::<Vec<_>>(),
            vec![
                (PrefillCohortKind::BatchedCold, 4),
                (PrefillCohortKind::BatchedCold, 4),
                // The ninth row cannot pair with anything: sequential.
                (PrefillCohortKind::Sequential, 1),
            ]
        );
    }

    #[test]
    fn oversized_head_falls_back_to_sequential() {
        // Budget 40 means max_len must stay <= 20; the 30-token head can
        // never share a window, and must not stall the rows behind it.
        let rows = vec![cold(30), cold(10), cold(10)];
        let limits = PaddedCostLimits {
            max_padded_tokens: 40,
            max_rows: 0,
        };
        let cohorts = plan_prefill_cohorts(&rows, PAD_CAPABLE, limits);
        assert_invariants(&rows, PAD_CAPABLE, limits, &cohorts);
        assert_eq!(
            cohorts
                .iter()
                .map(|cohort| (cohort.kind, cohort.members.clone()))
                .collect::<Vec<_>>(),
            vec![
                (PrefillCohortKind::Sequential, vec![0]),
                (PrefillCohortKind::BatchedCold, vec![1, 2]),
            ]
        );
    }

    #[test]
    fn equal_length_only_model_batches_only_uniform_runs() {
        let rows = vec![cold(8), cold(8), cold(9), cold(9), cold(7)];
        let cohorts = plan_prefill_cohorts(&rows, EQUAL_LEN_ONLY, UNCAPPED);
        assert_invariants(&rows, EQUAL_LEN_ONLY, UNCAPPED, &cohorts);
        assert_eq!(
            cohorts
                .iter()
                .map(|cohort| (cohort.kind, cohort.members.clone()))
                .collect::<Vec<_>>(),
            vec![
                (PrefillCohortKind::BatchedCold, vec![0, 1]),
                (PrefillCohortKind::BatchedCold, vec![2, 3]),
                (PrefillCohortKind::Sequential, vec![4]),
            ]
        );
    }

    #[test]
    fn row_cap_bounds_batched_cohort_size() {
        let rows: Vec<_> = (0..7).map(|_| cold(4)).collect();
        let limits = PaddedCostLimits {
            max_padded_tokens: 0,
            max_rows: 3,
        };
        let cohorts = plan_prefill_cohorts(&rows, PAD_CAPABLE, limits);
        assert_invariants(&rows, PAD_CAPABLE, limits, &cohorts);
        assert_eq!(
            cohorts
                .iter()
                .map(|cohort| cohort.members.len())
                .collect::<Vec<_>>(),
            vec![3, 3, 1]
        );
    }

    #[test]
    fn zero_length_prompt_rows_never_batch() {
        let rows = vec![cold(0), cold(5), cold(5)];
        let cohorts = plan_prefill_cohorts(&rows, PAD_CAPABLE, UNCAPPED);
        assert_invariants(&rows, PAD_CAPABLE, UNCAPPED, &cohorts);
        assert_eq!(cohorts[0].kind, PrefillCohortKind::Sequential);
        assert_eq!(cohorts[0].members, vec![0]);
    }

    #[test]
    fn default_budget_follows_two_x_rows_times_chunk() {
        assert_eq!(default_padded_token_budget(512, 4), 4096);
        // Chunking disabled: 512 tokens per row.
        assert_eq!(default_padded_token_budget(0, 4), 4096);
        // Zero rows clamps to one.
        assert_eq!(default_padded_token_budget(256, 0), 512);
    }

    #[test]
    fn mask_upper_bound_matches_documented_derivation() {
        // Default budget 4096 at FP32: 4096^2 / 2 elements * 4 bytes = 32 MiB.
        assert_eq!(
            padded_mask_bytes_upper_bound(4096, 4),
            u64::from(4096u32) * 4096 / 2 * 4
        );
        assert_eq!(padded_mask_bytes_upper_bound(0, 4), 0);
    }

    /// Property sweep: random windows, capabilities, and limits must always
    /// satisfy every planner invariant — most importantly that the flattened
    /// cohort order is exactly the input order.
    #[test]
    fn property_random_windows_preserve_all_invariants() {
        use proptest::prelude::{Just, Strategy, any, prop_oneof};
        use proptest::test_runner::{Config, TestError, TestRunner};

        let row = prop_oneof![
            (1u32..64).prop_map(cold),
            (0u32..64).prop_map(adopted),
            (0u32..64).prop_map(vlm),
            Just(cold(0)),
        ];
        let strategy = (
            proptest::collection::vec(row, 0..24),
            any::<bool>(),
            any::<bool>(),
            0u32..96,
            0u32..6,
        );
        let mut runner = TestRunner::new(Config {
            cases: 512,
            failure_persistence: None,
            ..Config::default()
        });
        let result = runner.run(
            &strategy,
            |(rows, supports_batched, supports_padding, max_padded_tokens, max_rows)| {
                let capability = BatchedPrefillCapability {
                    supports_batched,
                    supports_padding,
                };
                let limits = PaddedCostLimits {
                    max_padded_tokens,
                    max_rows,
                };
                let cohorts = plan_prefill_cohorts(&rows, capability, limits);
                assert_invariants(&rows, capability, limits, &cohorts);
                Ok(())
            },
        );
        match result {
            Ok(()) => {}
            Err(TestError::Fail(reason, value)) => {
                panic!("planner invariant failed: {reason}\nminimal input: {value:?}")
            }
            Err(TestError::Abort(reason)) => panic!("property run aborted: {reason}"),
        }
    }
}
