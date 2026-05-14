//! Per-thread counter of MLX op-wrapper invocations.
//!
//! Each safe wrapper in `ops.rs` and `fast.rs` calls `bump()` exactly once.
//! `op_count_snapshot()` reads the running total for the calling thread
//! without resetting it; `op_count_take(prev)` returns the delta since `prev`.
//!
//! The counter is **thread-local** by design: ax-engine dispatches all MLX
//! ops for one decode step from a single thread (the runner holds a per-request
//! mutex during decode work), so a per-thread counter accurately reflects
//! per-step op cost while remaining isolated from any concurrent MLX work
//! happening on other threads (test parallelism, embedding workers, etc.).
//!
//! Cost is one cell read+write per op; on a single thread this is a few
//! nanoseconds, negligible against the µs-scale per-op MLX dispatch.

use std::cell::Cell;

thread_local! {
    static OP_COUNT: Cell<u64> = const { Cell::new(0) };
}

/// Increment this thread's op counter by one.
///
/// Called from every safe op wrapper that issues exactly one underlying MLX
/// FFI dispatch. Composition wrappers (e.g. `silu`, `gelu`) do **not** call
/// this — their dispatches are accounted for by the inner direct-FFI ops.
#[inline]
pub fn bump() {
    OP_COUNT.with(|c| c.set(c.get().saturating_add(1)));
}

/// Read the current cumulative op count for this thread without modifying it.
pub fn op_count_snapshot() -> u64 {
    OP_COUNT.with(|c| c.get())
}

/// Compute the delta between this thread's current op count and `prev`.
/// Useful for per-step or per-section bracketing:
///
/// ```ignore
/// let prev = mlx_sys::op_count_snapshot();
/// // … MLX work on this thread …
/// let n = mlx_sys::op_count_take(prev);
/// ```
#[inline]
pub fn op_count_take(prev: u64) -> u64 {
    op_count_snapshot().saturating_sub(prev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::MlxDtype;
    use crate::ops::{astype, multiply, reshape, silu, zeros};

    #[test]
    fn direct_ffi_op_bumps_counter() {
        let prev = op_count_snapshot();
        let _ = zeros(&[2, 2], MlxDtype::Float32, None);
        assert_eq!(op_count_take(prev), 1, "zeros should bump exactly once");
    }

    #[test]
    fn macro_generated_op_bumps_counter() {
        let a = zeros(&[2, 2], MlxDtype::Float32, None);
        let b = zeros(&[2, 2], MlxDtype::Float32, None);
        let prev = op_count_snapshot();
        let _ = multiply(&a, &b, None);
        assert_eq!(
            op_count_take(prev),
            1,
            "macro-generated op should bump exactly once"
        );
    }

    #[test]
    fn composition_op_does_not_double_count() {
        let x = zeros(&[2, 2], MlxDtype::Float32, None);
        let prev = op_count_snapshot();
        let _ = silu(&x, None);
        assert_eq!(
            op_count_take(prev),
            2,
            "silu must count as exactly its 2 callees (sigmoid + multiply)"
        );
    }

    #[test]
    fn snapshot_and_take_round_trip() {
        let start = op_count_snapshot();
        let a = zeros(&[1], MlxDtype::Float32, None);
        let _ = reshape(&a, &[1], None);
        let _ = astype(&a, MlxDtype::Bfloat16, None);
        assert_eq!(op_count_take(start), 3);
    }
}
