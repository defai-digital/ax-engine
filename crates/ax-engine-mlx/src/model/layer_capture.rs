//! Scoped capture of per-layer outputs from the target forward.
//!
//! Consumed by the DFlash2 round loop once it lands (ADR-045 increment (c));
//! until then only tests construct a scope.
//!
//! The DFlash2 drafter conditions on the target's hidden states after a fixed
//! set of layers. A forward runs inside [`capture_layers`]; every `layer_forward`
//! reports its output through [`observe`], and the scope keeps the ones whose
//! layer index was requested. Outside a scope [`observe`] is a single
//! thread-local read.
//!
//! The captured arrays are lazy graph nodes of the forward that produced them,
//! so they cost no extra device work and no extra memory beyond keeping those
//! activations alive until they are consumed or dropped.

#![allow(dead_code)]

use std::cell::RefCell;

use mlx_sys::MlxArray;

struct Scope {
    layers: Vec<usize>,
    slots: Vec<Option<MlxArray>>,
    accumulate: Option<Accumulator>,
}

/// Keeps the trailing rows of every requested layer across several forwards.
struct Accumulator {
    max_rows: i32,
    /// Per requested layer: the chunks observed since the last collapse.
    chunks: Vec<Vec<MlxArray>>,
    /// Rows observed per requested layer since the scope began.
    rows_seen: Vec<i64>,
}

impl Accumulator {
    /// Joins a layer's chunks and keeps only the trailing `max_rows` rows, as a
    /// contiguous copy so the dropped rows' buffers are released.
    fn collapse(&mut self, slot: usize) {
        let joined = {
            let refs: Vec<&MlxArray> = self.chunks[slot].iter().collect();
            if refs.len() == 1 {
                refs[0].clone()
            } else {
                mlx_sys::concatenate(&refs, 1, None)
            }
        };
        let shape = joined.shape();
        let rows = shape[1];
        let tail = if rows > self.max_rows {
            mlx_sys::contiguous(
                &mlx_sys::slice(
                    &joined,
                    &[0, rows - self.max_rows, 0],
                    &[shape[0], rows, shape[2]],
                    &[1, 1, 1],
                    None,
                ),
                None,
            )
        } else {
            joined
        };
        self.chunks[slot] = vec![tail];
    }

    fn buffered_rows(&self, slot: usize) -> i32 {
        self.chunks[slot].iter().map(|chunk| chunk.shape()[1]).sum()
    }
}

thread_local! {
    static SCOPE: RefCell<Option<Scope>> = const { RefCell::new(None) };
}

/// Restores the previous capture scope when dropped.
#[must_use]
pub(crate) struct LayerCaptureGuard {
    previous: Option<Scope>,
}

impl Drop for LayerCaptureGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        SCOPE.with(|scope| *scope.borrow_mut() = previous);
    }
}

/// Start capturing the outputs of `layers` (strictly increasing layer indices).
pub(crate) fn capture_layers(layers: &[usize]) -> LayerCaptureGuard {
    let scope = Scope {
        layers: layers.to_vec(),
        slots: layers.iter().map(|_| None).collect(),
        accumulate: None,
    };
    let previous = SCOPE.with(|current| current.borrow_mut().replace(scope));
    LayerCaptureGuard { previous }
}

/// Like [`capture_layers`], but keeps the trailing `max_rows` rows of every
/// requested layer across any number of forwards (a chunked prefill). Read the
/// result with [`take_accumulated`].
pub(crate) fn capture_layers_accumulating(layers: &[usize], max_rows: i32) -> LayerCaptureGuard {
    let scope = Scope {
        layers: layers.to_vec(),
        slots: layers.iter().map(|_| None).collect(),
        accumulate: Some(Accumulator {
            max_rows,
            chunks: layers.iter().map(|_| Vec::new()).collect(),
            rows_seen: layers.iter().map(|_| 0).collect(),
        }),
    };
    let previous = SCOPE.with(|current| current.borrow_mut().replace(scope));
    LayerCaptureGuard { previous }
}

/// True while a capture scope is active on this thread.
pub(crate) fn active() -> bool {
    SCOPE.with(|scope| scope.borrow().is_some())
}

/// Called by `layer_forward` with the output of `layer`.
pub(crate) fn observe(layer: usize, hidden: &MlxArray) {
    SCOPE.with(|scope| {
        if let Some(scope) = scope.borrow_mut().as_mut()
            && let Some(slot) = scope.layers.iter().position(|id| *id == layer)
        {
            if let Some(accumulator) = scope.accumulate.as_mut() {
                accumulator.rows_seen[slot] += i64::from(hidden.shape()[1]);
                accumulator.chunks[slot].push(hidden.clone());
                if accumulator.buffered_rows(slot) > accumulator.max_rows * 2 {
                    accumulator.collapse(slot);
                }
            } else {
                scope.slots[slot] = Some(hidden.clone());
            }
        }
    });
}

/// The captured outputs in the requested layer order, once every requested
/// layer was seen since the scope began or the last take. Clears the slots so a
/// scope can serve several forwards in turn.
pub(crate) fn take() -> Option<Vec<MlxArray>> {
    SCOPE.with(|scope| {
        let mut borrowed = scope.borrow_mut();
        let scope = borrowed.as_mut()?;
        if scope.slots.iter().any(Option::is_none) {
            return None;
        }
        Some(scope.slots.iter_mut().filter_map(Option::take).collect())
    })
}

/// The accumulated tails of an accumulating scope: per requested layer an
/// `[1, <= max_rows, width]` array of its last rows, and the number of rows
/// every layer saw in total. `None` when no scope accumulates, nothing was
/// seen, or the layers disagree on how many rows they saw. Resets the scope so
/// it can keep accumulating.
pub(crate) fn take_accumulated() -> Option<(Vec<MlxArray>, i64)> {
    SCOPE.with(|scope| {
        let mut borrowed = scope.borrow_mut();
        let accumulator = borrowed.as_mut()?.accumulate.as_mut()?;
        let total = *accumulator.rows_seen.first()?;
        if total == 0 || accumulator.rows_seen.iter().any(|seen| *seen != total) {
            return None;
        }
        let mut tails = Vec::with_capacity(accumulator.chunks.len());
        for slot in 0..accumulator.chunks.len() {
            accumulator.collapse(slot);
            tails.push(accumulator.chunks[slot].first()?.clone());
        }
        for slot in 0..accumulator.chunks.len() {
            accumulator.chunks[slot].clear();
            accumulator.rows_seen[slot] = 0;
        }
        Some((tails, total))
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn value(x: f32) -> MlxArray {
        MlxArray::from_f32(x)
    }

    #[test]
    fn observe_outside_a_scope_is_a_no_op() {
        assert!(!active());
        observe(3, &value(1.0));
        assert!(take().is_none());
    }

    #[test]
    fn scope_keeps_only_requested_layers_in_requested_order() {
        let guard = capture_layers(&[2, 5]);
        assert!(active());
        for layer in 0..8 {
            observe(layer, &value(layer as f32));
        }
        let captured = take().expect("both layers were seen");
        assert_eq!(captured.len(), 2);
        mlx_sys::eval(&[&captured[0], &captured[1]]);
        assert_eq!(captured[0].data_f32()[0], 2.0);
        assert_eq!(captured[1].data_f32()[0], 5.0);
        // The slots are cleared: a second take needs a second forward.
        assert!(take().is_none());
        drop(guard);
        assert!(!active());
    }

    #[test]
    fn incomplete_capture_yields_nothing_and_nested_scopes_restore() {
        let outer = capture_layers(&[1]);
        observe(1, &value(7.0));
        {
            let inner = capture_layers(&[4]);
            observe(1, &value(9.0)); // not in the inner scope
            assert!(take().is_none());
            drop(inner);
        }
        // The outer scope's slot survived the inner scope.
        let captured = take().expect("outer layer seen");
        mlx_sys::eval(&[&captured[0]]);
        assert_eq!(captured[0].data_f32()[0], 7.0);
        drop(outer);
    }

    fn rows(first: f32, count: i32) -> MlxArray {
        let data: Vec<f32> = (0..count).map(|index| first + index as f32).collect();
        mlx_sys::reshape(&MlxArray::from_f32_slice(&data), &[1, count, 1], None)
    }

    #[test]
    fn accumulating_scope_keeps_the_trailing_rows_across_forwards() {
        let guard = capture_layers_accumulating(&[3, 7], 4);
        // Three forwards (chunks) of 3, 3 and 5 rows through two captured layers.
        for (first, count) in [(0.0, 3), (3.0, 3), (6.0, 5)] {
            observe(3, &rows(first, count));
            observe(5, &rows(-1.0, count)); // not requested
            observe(7, &rows(100.0 + first, count));
        }
        let (tails, total) = take_accumulated().expect("both layers saw 11 rows");
        assert_eq!(total, 11);
        let first = mlx_sys::contiguous(&tails[0], None);
        let second = mlx_sys::contiguous(&tails[1], None);
        mlx_sys::eval(&[&first, &second]);
        // The last 4 of 0..=10 and of 100..=110.
        assert_eq!(first.shape(), vec![1, 4, 1]);
        assert_eq!(first.data_f32(), &[7.0, 8.0, 9.0, 10.0]);
        assert_eq!(second.data_f32(), &[107.0, 108.0, 109.0, 110.0]);
        // The scope resets: nothing is pending until the next forward.
        assert!(take_accumulated().is_none());
        drop(guard);
        assert!(take_accumulated().is_none());
    }

    #[test]
    fn accumulated_layers_that_disagree_on_rows_yield_nothing() {
        let _guard = capture_layers_accumulating(&[1, 2], 8);
        observe(1, &rows(0.0, 3));
        observe(2, &rows(0.0, 2));
        assert!(take_accumulated().is_none());
    }
}
