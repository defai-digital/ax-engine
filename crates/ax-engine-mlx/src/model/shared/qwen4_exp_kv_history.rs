//! Append-only QSA key/value history for Flash Next (`qwen4_exp`).
//!
//! Every QSA layer keeps the K/V rows of the whole request as a
//! `[batch, tokens, kv_heads, head_dim]` tensor (token axis = 1). A decode
//! step appends one row to it. MLX arrays are immutable and the request state
//! contract is "a forward never mutates the cache and returns a staged next
//! state" (rollback, fork and durable snapshots all rely on it), so the old
//! `contiguous(concatenate([past, new], 1))` copied the entire history on
//! every step and decode slowed down as context grew.
//!
//! [`KvHistory`] stores the same logical tensor as an immutable sealed prefix
//! plus a short tail. A step copies only the tail (at most `seal_tokens` rows)
//! and every `seal_tokens` steps the tail is folded into the sealed prefix
//! once, so the full-history copy happens once per seal instead of once per
//! step. Selected attention does not flatten at all: [`SegmentRows::gather`]
//! reads the chosen rows straight from the segment that owns them.
//! [`KvHistory::flat`] still rebuilds the logical tensor for the dense kernels
//! and the durable snapshot writer.
//!
//! Segmentation is off below `min_tokens` rows, where a full copy is cheap,
//! and can be switched off with `AX_FLASH_NEXT_KV_SEGMENTS=0`.

use mlx_sys::{
    MlxArray, MlxDtype, astype, concatenate, contiguous, less, maximum, minimum, reshape, slice,
    subtract, take, where_cond,
};

/// Logical rows below which the history stays a single flat array.
const DEFAULT_MIN_TOKENS: usize = 4096;
/// Tail rows that trigger a seal into the immutable prefix.
const DEFAULT_SEAL_TOKENS: usize = 256;

/// Segmentation thresholds.
///
/// `min_tokens` keeps short contexts on the plain flat path; `seal_tokens`
/// bounds the rows a single step copies.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SegmentLimits {
    min_tokens: usize,
    seal_tokens: usize,
}

impl Default for SegmentLimits {
    fn default() -> Self {
        Self {
            min_tokens: DEFAULT_MIN_TOKENS,
            seal_tokens: DEFAULT_SEAL_TOKENS,
        }
    }
}

/// Immutable sealed prefix plus a short mutable-in-name-only tail. Logical
/// content is the sealed rows followed by the tail rows, both on axis 1.
#[derive(Clone)]
pub(crate) struct KvHistory {
    sealed: Option<MlxArray>,
    tail: Option<MlxArray>,
}

impl KvHistory {
    pub(crate) fn empty() -> Self {
        Self {
            sealed: None,
            tail: None,
        }
    }

    /// Restore a single-segment history from a durable snapshot.
    pub(crate) fn from_flat(array: MlxArray) -> Self {
        Self {
            sealed: Some(array),
            tail: None,
        }
    }

    pub(crate) fn is_some(&self) -> bool {
        self.sealed.is_some() || self.tail.is_some()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn segments(&self) -> (Option<&MlxArray>, Option<&MlxArray>) {
        (self.sealed.as_ref(), self.tail.as_ref())
    }

    /// Physical segments in logical order, sealed prefix first.
    pub(crate) fn arrays(&self) -> impl Iterator<Item = &MlxArray> + '_ {
        self.sealed.iter().chain(self.tail.iter())
    }

    /// First physical segment; carries the shape, dtype and batch that every
    /// segment of one history shares.
    pub(crate) fn first(&self) -> Option<&MlxArray> {
        self.sealed.as_ref().or(self.tail.as_ref())
    }

    /// Logical row count across all segments.
    pub(crate) fn token_count(&self) -> usize {
        self.arrays().map(rows).sum()
    }

    /// One array holding the whole logical history. `None` when empty.
    pub(crate) fn flat(&self) -> Option<MlxArray> {
        match (&self.sealed, &self.tail) {
            (None, None) => None,
            (Some(only), None) | (None, Some(only)) => Some(only.clone()),
            (Some(sealed), Some(tail)) => {
                #[cfg(test)]
                FLAT_CONCATS.with(|counter| counter.set(counter.get().saturating_add(1)));
                Some(contiguous(&concatenate(&[sealed, tail], 1, None), None))
            }
        }
    }

    /// One batch row per physical segment, for gathers straight from the
    /// segments. Each present segment is sliced exactly like the flat history
    /// would be, so a single-segment history emits the same ops as before.
    pub(crate) fn batch_slice(&self, batch_index: i32) -> SegmentRows {
        SegmentRows {
            sealed: self
                .sealed
                .as_ref()
                .map(|segment| batch_row(segment, batch_index)),
            tail: self
                .tail
                .as_ref()
                .map(|segment| batch_row(segment, batch_index)),
        }
    }

    /// Sealed-prefix rows, or `None` when this history is one segment. A
    /// history that only has a tail (no seal yet) is also one segment.
    pub(crate) fn sealed_rows(&self) -> Option<i32> {
        match (&self.sealed, &self.tail) {
            (Some(sealed), Some(_)) => rows_i32(sealed),
            _ => None,
        }
    }

    /// Stage `new` onto the history without mutating `self`.
    pub(crate) fn append(&self, new: &MlxArray) -> KvHistory {
        let limits = match segment_limits() {
            Some(limits) if self.token_count() + rows(new) >= limits.min_tokens => limits,
            _ => return self.append_flat(new),
        };
        self.append_segmented(new, limits)
    }

    /// One full-history copy: exactly the pre-segmentation behaviour.
    fn append_flat(&self, new: &MlxArray) -> KvHistory {
        let staged = match self.flat() {
            Some(past) => concatenate(&[&past, new], 1, None),
            None => new.clone(),
        };
        KvHistory {
            sealed: Some(contiguous(&staged, None)),
            tail: None,
        }
    }

    /// Copy the tail only; fold it into the sealed prefix once it is long
    /// enough. A chunk of at least `seal_tokens` rows seals immediately.
    fn append_segmented(&self, new: &MlxArray, limits: SegmentLimits) -> KvHistory {
        let tail = match &self.tail {
            Some(tail) => contiguous(&concatenate(&[tail, new], 1, None), None),
            None => contiguous(new, None),
        };
        if rows(&tail) >= limits.seal_tokens {
            let sealed = match &self.sealed {
                Some(sealed) => contiguous(&concatenate(&[sealed, &tail], 1, None), None),
                None => tail,
            };
            return KvHistory {
                sealed: Some(sealed),
                tail: None,
            };
        }
        KvHistory {
            sealed: self.sealed.clone(),
            tail: Some(tail),
        }
    }
}

/// Rows on the token axis; a history only ever holds 4-d tensors.
fn rows(array: &MlxArray) -> usize {
    array.shape().get(1).map_or(0, |rows| (*rows).max(0)) as usize
}

fn rows_i32(array: &MlxArray) -> Option<i32> {
    array.shape().get(1).copied()
}

/// Slice `batch_index` out of one physical segment: `[1, tokens, heads, dim]`.
fn batch_row(segment: &MlxArray, batch_index: i32) -> MlxArray {
    let shape = segment.shape();
    if batch_index == 0 && shape[0] == 1 {
        // The slice would be the identity: cloning is a refcount bump with
        // the same values and no op.
        return segment.clone();
    }
    slice(
        segment,
        &[batch_index, 0, 0, 0],
        &[batch_index + 1, shape[1], shape[2], shape[3]],
        &[1, 1, 1, 1],
        None,
    )
}

/// One batch row of a [`KvHistory`], still split across its segments.
pub(crate) struct SegmentRows {
    sealed: Option<MlxArray>,
    tail: Option<MlxArray>,
}

impl SegmentRows {
    /// Gather `idx` rows as `[1, idx_len, kv_heads, head_dim]`, the layout of
    /// `take(&rows, idx, 1, None)`. `None` for an empty row set.
    ///
    /// A single segment gathers exactly like the flat history. With both
    /// segments the rows are read from whichever owns them: a gather is a pure
    /// copy, so the result is bit-identical to gathering from the
    /// concatenation without materializing it.
    pub(crate) fn gather(&self, idx: &MlxArray) -> Option<MlxArray> {
        match (&self.sealed, &self.tail) {
            (Some(only), None) | (None, Some(only)) => Some(take(only, idx, 1, None)),
            (Some(sealed), Some(tail)) => {
                let sealed_rows = rows_i32(sealed)?;
                if sealed_rows <= 0 {
                    return Some(take(tail, idx, 1, None));
                }
                let split = SplitIndex::new(idx, sealed_rows)?;
                #[cfg(test)]
                SEGMENTED_GATHERS.with(|counter| counter.set(counter.get().saturating_add(1)));
                Some(split.merge(sealed, tail))
            }
            (None, None) => None,
        }
    }

    /// Sealed-prefix rows when this row set is split, `None` otherwise.
    fn split_rows(&self) -> Option<i32> {
        match (&self.sealed, &self.tail) {
            (Some(sealed), Some(_)) => rows_i32(sealed),
            _ => None,
        }
    }
}

/// Gather the same rows from a K/V row pair, computing the split index
/// arithmetic once for both.
///
/// Both histories advance in lockstep, so their segment boundaries agree; when
/// they do not, the pair falls back to two independent gathers (still
/// correct, just not shared).
pub(crate) fn gather_selected(
    keys: &SegmentRows,
    values: &SegmentRows,
    idx: &MlxArray,
) -> Option<(MlxArray, MlxArray)> {
    let shared = keys
        .split_rows()
        .filter(|&sealed_rows| sealed_rows > 0 && values.split_rows() == Some(sealed_rows));
    let Some(sealed_rows) = shared else {
        return Some((keys.gather(idx)?, values.gather(idx)?));
    };
    let split = SplitIndex::new(idx, sealed_rows)?;
    let (Some(k_sealed), Some(k_tail), Some(v_sealed), Some(v_tail)) = (
        keys.sealed.as_ref(),
        keys.tail.as_ref(),
        values.sealed.as_ref(),
        values.tail.as_ref(),
    ) else {
        return Some((keys.gather(idx)?, values.gather(idx)?));
    };
    #[cfg(test)]
    SEGMENTED_GATHERS.with(|counter| counter.set(counter.get().saturating_add(1)));
    Some((split.merge(k_sealed, k_tail), split.merge(v_sealed, v_tail)))
}

/// Per-row index arithmetic shared by a K/V gather over two segments.
///
/// `sealed` and `tail` are the clamped per-segment indices: every index is
/// kept inside the range of both segments, and the segment a row really
/// belongs to is chosen afterwards by `mask` (broadcast over heads and
/// head_dim). Indices are absolute positions in `[0, total)`.
struct SplitIndex {
    sealed: MlxArray,
    tail: MlxArray,
    mask: MlxArray,
}

impl SplitIndex {
    fn new(idx: &MlxArray, sealed_rows: i32) -> Option<Self> {
        let idx_len = *idx.shape().first()?;
        // Comparisons and subtraction need a signed index; the one-segment
        // path keeps whatever dtype the caller supplied.
        let signed = match idx.dtype() {
            MlxDtype::Int32 => None,
            _ => Some(astype(idx, MlxDtype::Int32, None)),
        };
        let idx = signed.as_ref().unwrap_or(idx);
        let boundary = i32_scalar(sealed_rows);
        let last_sealed = i32_scalar(sealed_rows - 1);
        let zero = i32_scalar(0);
        Some(Self {
            sealed: minimum(idx, &last_sealed, None),
            tail: maximum(&subtract(idx, &boundary, None), &zero, None),
            mask: reshape(&less(idx, &boundary, None), &[1, idx_len, 1, 1], None),
        })
    }

    fn merge(&self, sealed: &MlxArray, tail: &MlxArray) -> MlxArray {
        let from_sealed = take(sealed, &self.sealed, 1, None);
        let from_tail = take(tail, &self.tail, 1, None);
        where_cond(&self.mask, &from_sealed, &from_tail, None)
    }
}

fn i32_scalar(value: i32) -> MlxArray {
    MlxArray::from_raw_data(
        (&value as *const i32).cast(),
        std::mem::size_of::<i32>(),
        &[1],
        MlxDtype::Int32,
    )
}

#[cfg(test)]
thread_local! {
    /// Two-segment gathers executed on this thread.
    static SEGMENTED_GATHERS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Whole-history concatenations executed on this thread.
    static FLAT_CONCATS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn segmented_gathers_for_test() -> u64 {
    SEGMENTED_GATHERS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn flat_concats_for_test() -> u64 {
    FLAT_CONCATS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn reset_counters_for_test() {
    SEGMENTED_GATHERS.with(|counter| counter.set(0));
    FLAT_CONCATS.with(|counter| counter.set(0));
}

fn segment_limits() -> Option<SegmentLimits> {
    #[cfg(test)]
    if let Some(value) = KV_SEGMENT_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    crate::fastpath::flash_next_kv_segments_enabled().then(SegmentLimits::default)
}

#[cfg(test)]
thread_local! {
    static KV_SEGMENT_OVERRIDE: std::cell::Cell<Option<Option<SegmentLimits>>> =
        const { std::cell::Cell::new(None) };
}

/// Run `action` with segmentation forced on (`Some((min_tokens, seal_tokens))`)
/// or off (`None`) on this thread.
#[cfg(test)]
pub(crate) fn with_kv_segments<T>(limits: Option<(usize, usize)>, action: impl FnOnce() -> T) -> T {
    struct Restore(Option<Option<SegmentLimits>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            KV_SEGMENT_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let limits = limits.map(|(min_tokens, seal_tokens)| SegmentLimits {
        min_tokens,
        seal_tokens,
    });
    let _restore = Restore(KV_SEGMENT_OVERRIDE.with(|value| value.replace(Some(limits))));
    action()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use mlx_sys::{MlxDtype, astype, eval};

    fn row(values: &[f32], dtype: MlxDtype) -> MlxArray {
        let flat = MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values),
            &[1, values.len() as i32, 1, 1],
            MlxDtype::Float32,
        );
        astype(&flat, dtype, None)
    }

    fn concat_all(parts: &[MlxArray]) -> MlxArray {
        if parts.len() == 1 {
            return contiguous(&parts[0], None);
        }
        let refs: Vec<&MlxArray> = parts.iter().collect();
        contiguous(&concatenate(&refs, 1, None), None)
    }

    fn f32_bits(array: &MlxArray) -> Vec<u32> {
        let widened = contiguous(&astype(array, MlxDtype::Float32, None), None);
        eval(&[&widened]);
        widened
            .data_f32()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    fn segmented_history(rows: usize, dtype: MlxDtype) -> KvHistory {
        let mut history = KvHistory::empty();
        for step in 0..rows {
            history = history.append(&row(&[step as f32], dtype));
        }
        history
    }

    #[test]
    fn flat_mode_matches_plain_concatenation() {
        with_kv_segments(None, || {
            for dtype in [MlxDtype::Bfloat16, MlxDtype::Float32] {
                let mut history = KvHistory::empty();
                let mut parts: Vec<MlxArray> = Vec::new();
                let mut appended = 0usize;
                for count in [1usize, 2, 3, 7] {
                    let data: Vec<f32> = (0..count).map(|i| i as f32 + 0.25).collect();
                    let new = row(&data, dtype);
                    parts.push(new.clone());
                    history = history.append(&new);
                    appended += count;
                    assert_eq!(history.token_count(), appended);
                    assert!(history.segments().1.is_none(), "flat mode keeps no tail");
                    assert_eq!(
                        f32_bits(&history.flat().unwrap()),
                        f32_bits(&concat_all(&parts)),
                        "{dtype:?} after {count} rows"
                    );
                }
            }
        });
    }

    #[test]
    fn segmented_steps_keep_content_and_bound_the_tail() {
        with_kv_segments(Some((8, 4)), || {
            let dtype = MlxDtype::Float32;
            let mut history = KvHistory::empty();
            let mut parts: Vec<MlxArray> = Vec::new();
            let mut last_sealed = 0i32;
            for step in 0..40usize {
                let new = row(&[step as f32], dtype);
                parts.push(new.clone());
                history = history.append(&new);
                let (sealed, tail) = history.segments();
                let now_sealed = sealed.map_or(0, |array| array.shape()[1]);
                let tail_rows = tail.map_or(0, |array| array.shape()[1]);
                assert_eq!(history.token_count(), step + 1);
                assert_eq!(
                    now_sealed + tail_rows,
                    step as i32 + 1,
                    "segments must cover the logical history at step {step}"
                );
                assert!(tail_rows < 4, "tail {tail_rows} rows at step {step}");
                assert!(
                    now_sealed >= last_sealed,
                    "the sealed prefix must only grow at step {step}"
                );
                last_sealed = now_sealed;
                if tail.is_none() {
                    assert_eq!(
                        now_sealed,
                        step as i32 + 1,
                        "a seal must leave no tail at step {step}"
                    );
                }
                assert_eq!(
                    f32_bits(&history.flat().unwrap()),
                    f32_bits(&concat_all(&parts)),
                    "content at step {step}"
                );
            }
        });
    }

    #[test]
    fn sealed_prefix_costs_no_ops_on_a_tail_step() {
        with_kv_segments(Some((8, 4)), || {
            let dtype = MlxDtype::Float32;
            // 15 and 39 rows both end right after a seal, so the next append
            // is a pure tail step.
            let short = segmented_history(15, dtype);
            let long = segmented_history(39, dtype);
            assert!(short.segments().1.is_none());
            assert!(long.segments().1.is_none());
            assert_eq!(short.token_count(), 15);
            assert_eq!(long.token_count(), 39);

            let next = row(&[1.0], dtype);
            let start = mlx_sys::op_count_snapshot();
            let short_next = short.append(&next);
            let short_ops = mlx_sys::op_count_take(start);
            let start = mlx_sys::op_count_snapshot();
            let long_next = long.append(&next);
            let long_ops = mlx_sys::op_count_take(start);
            assert_eq!(
                short_ops, long_ops,
                "a tail step must not scale with the sealed prefix"
            );
            assert_eq!(long_ops, 1, "one contiguous copy of the new rows");
            assert_eq!(long_next.token_count(), 40);
            assert_eq!(short_next.token_count(), 16);

            with_kv_segments(None, || {
                let flat = segmented_history(39, dtype);
                let start = mlx_sys::op_count_snapshot();
                let flat_next = flat.append(&next);
                let flat_ops = mlx_sys::op_count_take(start);
                assert!(
                    long_ops < flat_ops,
                    "segmented={long_ops} must beat flat={flat_ops}"
                );
                assert_eq!(flat_next.token_count(), 40);
            });
        });
    }

    #[test]
    fn multi_row_appends_seal_at_the_boundary() {
        with_kv_segments(Some((8, 4)), || {
            let dtype = MlxDtype::Float32;
            let mut history = KvHistory::empty();
            let mut parts: Vec<MlxArray> = Vec::new();

            // A prefill chunk of at least `seal_tokens` rows seals at once.
            let prefill = row(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype);
            parts.push(prefill.clone());
            history = history.append(&prefill);
            assert_eq!(history.segments().0.map(|array| array.shape()[1]), Some(8));
            assert!(history.segments().1.is_none());
            assert_eq!(history.token_count(), 8);

            // Three rows that fill the tail without sealing.
            let first = row(&[8.0, 9.0, 10.0], dtype);
            parts.push(first.clone());
            history = history.append(&first);
            assert_eq!(history.segments().0.map(|array| array.shape()[1]), Some(8));
            assert_eq!(history.segments().1.map(|array| array.shape()[1]), Some(3));
            assert_eq!(history.token_count(), 11);

            // Three more rows cross the boundary and seal the whole tail.
            let second = row(&[11.0, 12.0, 13.0], dtype);
            parts.push(second.clone());
            history = history.append(&second);
            assert_eq!(history.segments().0.map(|array| array.shape()[1]), Some(14));
            assert!(history.segments().1.is_none());
            assert_eq!(history.token_count(), 14);
            assert_eq!(
                f32_bits(&history.flat().unwrap()),
                f32_bits(&concat_all(&parts))
            );
        });
    }

    #[test]
    fn switching_segmentation_off_midway_keeps_content() {
        let dtype = MlxDtype::Float32;
        let mut parts: Vec<MlxArray> = Vec::new();
        let mut history = KvHistory::empty();
        with_kv_segments(Some((4, 2)), || {
            for step in 0..4usize {
                let new = row(&[step as f32], dtype);
                parts.push(new.clone());
                history = history.append(&new);
            }
            assert!(
                history.segments().1.is_some(),
                "expected a live tail before switching segmentation off"
            );
            with_kv_segments(None, || {
                let new = row(&[9.0], dtype);
                parts.push(new.clone());
                history = history.append(&new);
            });
        });
        assert!(
            history.segments().1.is_none(),
            "flat mode collapses the tail"
        );
        assert_eq!(history.token_count(), 5);
        assert_eq!(
            f32_bits(&history.flat().unwrap()),
            f32_bits(&concat_all(&parts))
        );
    }

    #[test]
    fn clones_do_not_share_appended_rows() {
        let dtype = MlxDtype::Float32;
        with_kv_segments(Some((4, 2)), || {
            let mut base = KvHistory::empty();
            for step in 0..6usize {
                base = base.append(&row(&[step as f32], dtype));
            }
            let mut left = base.clone();
            let mut right = base.clone();
            for step in 0..3usize {
                left = left.append(&row(&[100.0 + step as f32], dtype));
                right = right.append(&row(&[200.0 + step as f32], dtype));
            }
            assert_eq!(left.token_count(), 9);
            assert_eq!(right.token_count(), 9);
            let base_bits = f32_bits(&base.flat().unwrap());
            let left_bits = f32_bits(&left.flat().unwrap());
            let right_bits = f32_bits(&right.flat().unwrap());
            assert_eq!(left_bits.len(), 9);
            assert_eq!(right_bits.len(), 9);
            assert_eq!(left_bits[..6], base_bits[..]);
            assert_eq!(right_bits[..6], base_bits[..]);
            assert_ne!(left_bits[6..], right_bits[6..]);
        });
    }

    #[test]
    fn from_flat_then_append_matches_incremental_history() {
        let dtype = MlxDtype::Float32;
        let parts: Vec<MlxArray> = (0..5).map(|i| row(&[i as f32], dtype)).collect();
        with_kv_segments(Some((8, 4)), || {
            let mut seeded = KvHistory::from_flat(concat_all(&parts));
            assert_eq!(seeded.token_count(), 5);
            assert!(seeded.segments().1.is_none());
            let mut incremental = KvHistory::empty();
            for part in &parts {
                incremental = incremental.append(part);
            }
            for extra in 0..5usize {
                let new = row(&[50.0 + extra as f32], dtype);
                seeded = seeded.append(&new);
                incremental = incremental.append(&new);
                assert_eq!(seeded.token_count(), incremental.token_count());
                assert_eq!(
                    f32_bits(&seeded.flat().unwrap()),
                    f32_bits(&incremental.flat().unwrap()),
                    "seeded vs incremental after extra {extra}"
                );
            }
        });
    }

    /// `[1, rows, 2, 8]` with a distinct value per element, so a gather that
    /// reads the wrong row cannot compare equal. `base` keeps the two segments
    /// of one logical history in disjoint value ranges.
    fn ramp(rows: i32, dtype: MlxDtype, base: f32) -> MlxArray {
        let data: Vec<f32> = (0..(rows as usize * 2 * 8))
            .map(|i| base + i as f32)
            .collect();
        let array = MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &[1, rows, 2, 8],
            MlxDtype::Float32,
        );
        astype(&array, dtype, None)
    }

    fn int32_index(values: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values),
            &[i32::try_from(values.len()).unwrap()],
            MlxDtype::Int32,
        )
    }

    fn assert_rows_match(actual: &MlxArray, expected: &MlxArray, label: &str) {
        assert_eq!(actual.shape(), expected.shape(), "{label}: shape");
        assert_eq!(f32_bits(actual), f32_bits(expected), "{label}: row bits");
    }

    /// Gather from the segments must be bit-identical to a gather from the
    /// concatenation, for every shape of index list.
    #[test]
    fn segment_gather_matches_take_on_concatenation() {
        let sealed_rows = 300i32;
        let tail_rows = 17i32;
        let total = sealed_rows + tail_rows;
        let cases: Vec<(&str, Vec<i32>)> = vec![
            ("ascending", (0..total).collect()),
            // 149 is coprime with 317, so this visits every row exactly once.
            ("shuffled", (0..total).map(|i| (i * 149) % total).collect()),
            (
                "repeated",
                vec![0, 0, 299, 300, 300, 7, 7, 316, 316, 5, 5, 5],
            ),
            ("all sealed", (0..sealed_rows).rev().collect()),
            ("all tail", (sealed_rows..total).collect()),
            (
                "seal boundary",
                vec![sealed_rows - 1, sealed_rows, sealed_rows + 1],
            ),
            ("single", vec![150]),
            ("long", (0..2100).map(|i| (i * 13) % total).collect()),
        ];
        for dtype in [MlxDtype::Bfloat16, MlxDtype::Float32] {
            let sealed = ramp(sealed_rows, dtype, 0.0);
            let tail = ramp(tail_rows, dtype, 6_000.0);
            // A seal bound above `tail_rows` keeps the second append in the
            // tail, while the first append of 300 rows still seals at once.
            let history = with_kv_segments(Some((8, 64)), || {
                KvHistory::empty().append(&sealed).append(&tail)
            });
            assert_eq!(
                history.segments().0.map(|array| array.shape()[1]),
                Some(sealed_rows)
            );
            assert_eq!(
                history.segments().1.map(|array| array.shape()[1]),
                Some(tail_rows)
            );
            let rows = history.batch_slice(0);
            let flat = history.flat().unwrap();
            for (label, indices) in &cases {
                let idx = int32_index(indices);
                let gathered = rows.gather(&idx).unwrap();
                let expected = take(&flat, &idx, 1, None);
                assert_rows_match(&gathered, &expected, &format!("{dtype:?} {label}"));
            }
        }
    }

    /// One-segment histories (no tail, or a tail that never sealed) gather
    /// exactly the rows of the single array.
    #[test]
    fn single_segment_gather_matches_take() {
        let rows = 17i32;
        let indices = vec![16, 0, 8, 8, 3];
        let idx = int32_index(&indices);
        for dtype in [MlxDtype::Bfloat16, MlxDtype::Float32] {
            let array = ramp(rows, dtype, 0.0);

            let sealed_only = KvHistory::from_flat(array.clone());
            assert!(sealed_only.segments().1.is_none());
            assert!(sealed_only.sealed_rows().is_none());

            // `min_tokens` reached but the tail stays under the seal bound, so
            // the history has one segment and it is the tail.
            let tail_only = with_kv_segments(Some((8, 64)), || KvHistory::empty().append(&array));
            assert!(tail_only.segments().0.is_none());
            assert!(tail_only.segments().1.is_some());

            for (label, history) in [("sealed only", &sealed_only), ("tail only", &tail_only)] {
                let gathered = history.batch_slice(0).gather(&idx).unwrap();
                assert_rows_match(
                    &gathered,
                    &take(&array, &idx, 1, None),
                    &format!("{dtype:?} {label}"),
                );
            }
        }
    }

    #[test]
    fn counters_see_the_segmented_and_flat_paths() {
        let dtype = MlxDtype::Float32;
        let sealed = ramp(9, dtype, 0.0);
        let tail = ramp(3, dtype, 6_000.0);
        let history = with_kv_segments(Some((8, 4)), || KvHistory::empty().append(&sealed));
        assert!(history.segments().1.is_none());
        let history = with_kv_segments(Some((8, 4)), || history.append(&tail));
        assert!(history.segments().1.is_some());

        reset_counters_for_test();
        let idx = int32_index(&[0, 8, 11]);
        assert!(history.batch_slice(0).gather(&idx).is_some());
        assert_eq!(segmented_gathers_for_test(), 1);
        assert_eq!(flat_concats_for_test(), 0);

        reset_counters_for_test();
        assert!(history.flat().is_some());
        assert_eq!(flat_concats_for_test(), 1);
        assert_eq!(segmented_gathers_for_test(), 0);
    }
}
