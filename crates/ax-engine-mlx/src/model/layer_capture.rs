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
            scope.slots[slot] = Some(hidden.clone());
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
}
