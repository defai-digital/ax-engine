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
//! step. [`KvHistory::flat`] still rebuilds the logical tensor for callers
//! that need one contiguous array.
//!
//! Segmentation is off below `min_tokens` rows, where a full copy is cheap,
//! and can be switched off with `AX_FLASH_NEXT_KV_SEGMENTS=0`.

use mlx_sys::{MlxArray, concatenate, contiguous};

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
                Some(contiguous(&concatenate(&[sealed, tail], 1, None), None))
            }
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
}
