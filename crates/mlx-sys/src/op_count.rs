//! Per-thread counter of MLX op-wrapper invocations.
//!
//! Each safe wrapper in `ops.rs` and `fast.rs` calls `bump()` (equivalently
//! `bump_named(<op>)`) exactly once. `op_count_snapshot()` reads the running
//! total for the calling thread without resetting it; `op_count_take(prev)`
//! returns the delta since `prev`.
//!
//! The counter is **thread-local** by design: ax-engine dispatches all MLX
//! ops for one decode step from a single thread (the runner holds a per-request
//! mutex during decode work), so a per-thread counter accurately reflects
//! per-step op cost while remaining isolated from any concurrent MLX work
//! happening on other threads (test parallelism, embedding workers, etc.).
//!
//! Cost is one cell read+write per op; on a single thread this is a few
//! nanoseconds, negligible against the µs-scale per-op MLX dispatch.
//!
//! # Opt-in op trace
//!
//! [`op_trace_begin()`] opens a per-thread census window: every subsequent
//! `bump`/`bump_named` on that thread additionally records one `(caller, op)`
//! hit until [`op_trace_end()`] closes the window and drains the map. The hot
//! path pays one extra thread-local bool read per op; the expensive backtrace
//! walk lives in the cold tracer and never runs while the window is closed.

use std::backtrace::Backtrace;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

thread_local! {
    static OP_COUNT: Cell<u64> = const { Cell::new(0) };
    static TRACE_ON: Cell<bool> = const { Cell::new(false) };
    static TRACE: RefCell<HashMap<(String, &'static str), u64>> = RefCell::new(HashMap::new());
}

/// One `(caller, op)` hit recorded during an opt-in op trace window.
#[derive(Clone, Debug)]
pub struct OpTraceEntry {
    /// First frame outside this crate and the standard library at the bump
    /// site, optionally suffixed with ` @file.rs:line` when the location of
    /// that frame is known.
    pub caller: String,
    /// Op wrapper name (`stringify!($name)` for macro-generated wrappers, the
    /// function name for hand-written ones, `"?"` for plain `bump()` sites).
    pub op: &'static str,
    /// Number of hits recorded while the trace window was open.
    pub count: u64,
}

/// Increment this thread's op counter by one.
///
/// Called from every safe op wrapper that issues exactly one underlying MLX
/// FFI dispatch. Composition wrappers (e.g. `silu`, `gelu`) do **not** call
/// this — their dispatches are accounted for by the inner direct-FFI ops.
#[inline]
pub(crate) fn bump() {
    OP_COUNT.with(|c| c.set(c.get().saturating_add(1)));
    if TRACE_ON.with(Cell::get) {
        trace_slow("?");
    }
}

/// Increment this thread's op counter by one, attributing the hit to `name`.
///
/// Identical to [`bump`] except that, while an op trace window is open, the
/// recorded hit is keyed by `name` instead of `"?"`.
#[inline]
pub(crate) fn bump_named(name: &'static str) {
    OP_COUNT.with(|c| c.set(c.get().saturating_add(1)));
    if TRACE_ON.with(Cell::get) {
        trace_slow(name);
    }
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

/// Open this thread's op trace window: clear the map and start recording.
pub fn op_trace_begin() {
    TRACE.with(|trace| trace.borrow_mut().clear());
    TRACE_ON.with(|flag| flag.set(true));
}

/// Close this thread's op trace window and drain it.
///
/// Returns the recorded hits sorted by hit count (descending), then by caller
/// for stable output. Recording stops immediately; the counter itself is
/// untouched.
pub fn op_trace_end() -> Vec<OpTraceEntry> {
    TRACE_ON.with(|flag| flag.set(false));
    let mut entries: Vec<OpTraceEntry> = TRACE.with(|trace| {
        trace
            .borrow_mut()
            .drain()
            .map(|((caller, op), count)| OpTraceEntry { caller, op, count })
            .collect()
    });
    entries.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.caller.cmp(&b.caller)));
    entries
}

/// Record one traced hit under `(caller, op)`.
///
/// Cold path: only reachable while a trace window is open. Captures the
/// current backtrace and resolves the first frame outside this crate and the
/// standard library — the engine function that issued the op. Nothing is
/// cached; this path only runs while tracing.
#[cold]
#[inline(never)]
fn trace_slow(op: &'static str) {
    let backtrace = Backtrace::force_capture();
    let caller = resolve_caller(&backtrace.to_string());
    TRACE.with(|trace| {
        *trace.borrow_mut().entry((caller, op)).or_insert(0) += 1;
    });
}

/// Symbol prefixes that sit between the bump site and the engine code; the
/// caller is the first frame whose symbol starts with none of these.
const CALLER_EXCLUDED_PREFIXES: [&str; 6] = [
    "mlx_sys::",
    "std::",
    "core::",
    "alloc::",
    "backtrace::",
    "<",
];

/// Walk the rendered backtrace and resolve the engine caller frame.
///
/// Frame lines look like `  12: ax_engine_mlx::model::…::forward`, optionally
/// followed by a line `             at ./crates/…/file.rs:123:45`. The first
/// symbol that is not an internal frame wins; its trailing `::h<16 hex>`
/// hash is stripped and a known location is appended as ` @file.rs:123`
/// (basename and line only).
fn resolve_caller(backtrace: &str) -> String {
    let mut lines = backtrace.lines().peekable();
    while let Some(line) = lines.next() {
        let Some(symbol) = frame_symbol(line) else {
            continue;
        };
        if CALLER_EXCLUDED_PREFIXES
            .iter()
            .any(|prefix| symbol.starts_with(prefix))
        {
            continue;
        }
        let mut caller = strip_hash_suffix(symbol).to_owned();
        if let Some(location) = lines.peek().and_then(|next| at_location(next)) {
            caller.push_str(" @");
            caller.push_str(location.0);
            caller.push(':');
            caller.push_str(location.1);
        }
        return caller;
    }
    "<unknown>".to_owned()
}

/// Parse a frame line of the form `  12: symbol` into its symbol.
fn frame_symbol(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let (index, rest) = trimmed.split_once(": ")?;
    if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let symbol = rest.trim();
    if symbol.is_empty() {
        return None;
    }
    Some(symbol)
}

/// Drop the legacy `::h<16 hex chars>` symbol-hash suffix, if present.
fn strip_hash_suffix(symbol: &str) -> &str {
    const HASH_SUFFIX_LEN: usize = 3 + 16; // "::h" plus 16 hex digits
    let bytes = symbol.as_bytes();
    if bytes.len() > HASH_SUFFIX_LEN
        && &bytes[bytes.len() - HASH_SUFFIX_LEN..bytes.len() - 16] == b"::h"
        && bytes[bytes.len() - 16..]
            .iter()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return &symbol[..symbol.len() - HASH_SUFFIX_LEN];
    }
    symbol
}

/// Parse a location line `at ./crates/…/file.rs:123:45` into `(file, line)`.
/// Returns `None` for anything else (e.g. another frame line).
fn at_location(line: &str) -> Option<(&str, &str)> {
    let location = line.trim().strip_prefix("at ")?;
    let mut parts = location.trim().rsplit(':');
    let _column = parts.next()?;
    let line_no = parts.next()?;
    if line_no.is_empty() || !line_no.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let path = parts.next()?;
    let file = path.rsplit('/').next().unwrap_or(path);
    Some((file, line_no))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::MlxDtype;
    use crate::ops::{astype, multiply, reshape, silu, zeros};

    #[inline(never)]
    fn traced_helper() {
        let a = zeros(&[2, 2], MlxDtype::Float32, None);
        let _ = multiply(&a, &a, None);
        let _ = multiply(&a, &a, None);
    }

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

    #[test]
    fn op_trace_records_named_ops_with_caller() {
        // Traced run: 1 zeros + 2 multiply, and the counter must be unchanged
        // by tracing.
        op_trace_begin();
        let prev = op_count_snapshot();
        traced_helper();
        assert_eq!(op_count_take(prev), 3, "tracing must not change the count");
        let entries = op_trace_end();

        let multiply = entries
            .iter()
            .find(|entry| entry.op == "multiply")
            .expect("multiply must be traced");
        assert_eq!(multiply.count, 2, "multiply is issued twice");
        let zeros = entries
            .iter()
            .find(|entry| entry.op == "zeros")
            .expect("zeros must be traced");
        assert_eq!(zeros.count, 1, "zeros is issued once");
        for entry in &entries {
            assert!(!entry.caller.is_empty(), "every entry needs a caller");
        }
        assert!(
            entries
                .windows(2)
                .all(|pair| pair[0].count >= pair[1].count),
            "entries must be sorted by count descending"
        );

        // Untraced run: same op delta, nothing recorded.
        let prev = op_count_snapshot();
        traced_helper();
        assert_eq!(op_count_take(prev), 3, "untraced delta must be identical");
        assert!(
            op_trace_end().is_empty(),
            "nothing may be recorded while the window is closed"
        );
    }

    #[test]
    fn caller_resolution_parses_rendered_frames() {
        let rendered = concat!(
            "   0: mlx_sys::op_count::trace_slow\n",
            "             at ./crates/mlx-sys/src/op_count.rs:120:20\n",
            "   1: mlx_sys::op_count::bump_named\n",
            "   2: mlx_sys::ops::zeros\n",
            "             at ./crates/mlx-sys/src/ops.rs:2531:5\n",
            "   3: ax_engine_mlx::model::shared::qwen4_exp_gdn::Qwen4ExpGdn::forward::h1234567890abcdef\n",
            "             at ./crates/ax-engine-mlx/src/model/shared/qwen4_exp_gdn.rs:123:45\n",
            "   4: ax_engine_mlx::runner::decode\n",
        );
        assert_eq!(
            resolve_caller(rendered),
            "ax_engine_mlx::model::shared::qwen4_exp_gdn::Qwen4ExpGdn::forward @qwen4_exp_gdn.rs:123"
        );

        let unmatched = "   0: std::rt::lang_start_internal\n";
        assert_eq!(resolve_caller(unmatched), "<unknown>");
    }
}
