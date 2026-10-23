//! Compiled elementwise stretches of the Qwen 3.8 Flash Next (`qwen4_exp`)
//! layers, shared by the gated residual and the gated delta network.
//!
//! Why this exists: a decode step builds on the order of 10k MLX ops, and host
//! graph construction dominates the step. These layers spend most of those ops
//! on short chains of tiny elementwise work between projections, so each chain
//! is traced once per `(stretch, shape)` and compiled into a single call.
//! Projections stay outside the closures so every exactness policy keeps its
//! own kernels, and the closures take their weights as inputs because MLX
//! rejects captured arrays.
//!
//! Hardening, from the compile design review:
//!
//! - only decode-sized inputs are compiled ([`FUSE_DEFAULT_ROWS`] by default,
//!   at most [`FUSE_MAX_ROWS`]): the fusion only pays off while the host is
//!   the bottleneck, and a chunked prefill would otherwise trace one compiled
//!   graph per chunk length;
//! - a stretch whose closure fails to compile or to apply is tombstoned per
//!   thread, so later steps take the eager path permanently instead of
//!   re-flooding MLX errors every step;
//! - a compiled closure is dropped after [`refresh_threshold`] successful
//!   applies, so a long-lived process recompiles against fresh MLX state;
//! - the registry borrow is never held across an MLX call, because both
//!   building and applying a closure re-enter MLX on the same thread.
//!
//! On by default (single-token decode only). `AX_FLASH_NEXT_FUSE=0` is the kill
//! switch; a comma-separated stretch list enables just those stretches.

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(test)]
use std::collections::HashSet;
use std::sync::OnceLock;

use mlx_sys::{MlxArray, MlxClosure, MlxVectorArray};

use crate::per_layer_compile::{compile_cache_refresh_threshold, try_apply_with_abort_safety};

/// Mask for [`Stretch::ALL`]; parsed once, see [`parse_mask`].
const FUSE_ENV: &str = "AX_FLASH_NEXT_FUSE";

/// Absolute cap on the `seq` a fused stretch may run at.
///
/// A compiled closure is shape-specific (`compile(false)`), so every distinct
/// row count compiles each stretch once more: prefill chunks would grow the
/// registry by one graph per chunk length, and a stray 2..8 row remainder would
/// pay that compile inside a request.
pub(crate) const FUSE_MAX_ROWS: i32 = 8;

/// Rows used by default: single-token decode, the only place with a measured
/// win and the only shape the load-time warm-up compiles. Verifier windows are
/// opt-in through `AX_FLASH_NEXT_FUSE_ROWS` (clamped to [`FUSE_MAX_ROWS`]).
const FUSE_DEFAULT_ROWS: i32 = 1;

/// Env name of the opt-in row limit.
const FUSE_ROWS_ENV: &str = "AX_FLASH_NEXT_FUSE_ROWS";

/// Parse the row limit; unset, empty, zero or garbage keep the default.
fn parse_max_rows(raw: Option<&str>) -> i32 {
    raw.map(str::trim)
        .and_then(|raw| raw.parse::<i32>().ok())
        .filter(|rows| *rows >= 1)
        .map_or(FUSE_DEFAULT_ROWS, |rows| rows.min(FUSE_MAX_ROWS))
}

/// Largest row count a stretch may fuse at on this thread.
///
/// Tests that force a mask are exercising the machinery, not the shipping
/// default, so they get the full cap.
fn max_rows() -> i32 {
    if forced_only_bits().is_some() || forced_mask().is_some() {
        return FUSE_MAX_ROWS;
    }
    static ROWS: OnceLock<i32> = OnceLock::new();
    *ROWS.get_or_init(|| parse_max_rows(std::env::var(FUSE_ROWS_ENV).ok().as_deref()))
}

/// The pure elementwise stretches that can be compiled.
///
/// One variant per compile site rather than one per model: the registry key
/// already carries the geometry, so a stretch only has to name the graph shape
/// for logging and for the mask.
///
/// Only stretches proven bit-identical to the eager ops on a real pack belong
/// here; a bisect showed that compiling the GDN decay chain and the q/k L2
/// normalisations changes the recurrent state bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Stretch {
    HcNorm,
    HcLow,
    HcMix,
    HcGate,
    HcWrite,
    GdnBeta,
    GdnPost,
}

impl Stretch {
    /// Mask order; [`parse_mask`] indexes with [`Stretch::index`].
    pub(crate) const ALL: [Stretch; 7] = [
        Stretch::HcNorm,
        Stretch::HcLow,
        Stretch::HcMix,
        Stretch::HcGate,
        Stretch::HcWrite,
        Stretch::GdnBeta,
        Stretch::GdnPost,
    ];

    /// Name used in the `AX_FLASH_NEXT_FUSE` list and in the failure warning.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Stretch::HcNorm => "hc_norm",
            Stretch::HcLow => "hc_low",
            Stretch::HcMix => "hc_mix",
            Stretch::HcGate => "hc_gate",
            Stretch::HcWrite => "hc_write",
            Stretch::GdnBeta => "gdn_beta",
            Stretch::GdnPost => "gdn_post",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// Parse the fuse mask, indexed in [`Stretch::ALL`] order.
///
/// Kept pure and total so the accepted spellings are testable without touching
/// the environment. An unset variable follows `unset_means_all` (the shipping
/// default is on; unit tests default to off so each test opts in explicitly).
/// A set variable follows the repo's default-on switch convention:
/// empty/`0`/`false`/`off`/`no` disable everything, `1`/`true`/`on`/`all` enable
/// everything, and anything else is a comma-separated stretch list. Names are
/// trimmed, case-insensitive, and unknown names are ignored so a typo cannot
/// silently enable a stretch.
fn parse_mask(raw: Option<&str>, unset_means_all: bool) -> [bool; 7] {
    let mut mask = [false; 7];
    let Some(raw) = raw else {
        return [unset_means_all; 7];
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return mask;
    }
    let lowered = raw.to_ascii_lowercase();
    match lowered.as_str() {
        "0" | "false" | "off" | "no" => return mask,
        "1" | "true" | "on" | "all" => return [true; 7],
        _ => {}
    }
    for name in lowered.split(',') {
        let name = name.trim();
        if let Some(index) = Stretch::ALL
            .iter()
            .position(|stretch| stretch.name() == name)
        {
            mask[index] = true;
        }
    }
    mask
}

/// Process-wide mask; read once so the decode path never hits the environment.
fn env_mask() -> &'static [bool; 7] {
    static MASK: OnceLock<[bool; 7]> = OnceLock::new();
    MASK.get_or_init(|| parse_mask(std::env::var(FUSE_ENV).ok().as_deref(), !cfg!(test)))
}

#[cfg(test)]
fn forced_only_bits() -> Option<u16> {
    FORCE_ONLY.with(|only| only.get())
}

#[cfg(not(test))]
fn forced_only_bits() -> Option<u16> {
    None
}

#[cfg(test)]
fn forced_mask() -> Option<bool> {
    FORCE.with(|force| force.get())
}

#[cfg(not(test))]
fn forced_mask() -> Option<bool> {
    None
}

/// Whether `stretch` may use a compiled closure on this thread.
pub(crate) fn enabled(stretch: Stretch) -> bool {
    if let Some(bits) = forced_only_bits() {
        return bits & (1 << stretch.index()) != 0;
    }
    if let Some(forced) = forced_mask() {
        return forced;
    }
    env_mask()[stretch.index()]
}

/// Compile one closure body, or `None` when MLX rejects it.
///
/// `shapeless = false`: the registry keys on the input shape, so the tighter
/// fusion of a shape-specific graph is free.
pub(crate) fn compile(
    body: impl FnMut(&MlxVectorArray) -> Vec<MlxArray> + Send + 'static,
) -> Option<MlxClosure> {
    // MLX hands back the original function when this variable is set (to any
    // value), so "compiled" calls would run eagerly while still counting as
    // fused. Treat it as compilation being unavailable.
    if std::env::var_os("MLX_DISABLE_COMPILE").is_some() {
        return None;
    }
    MlxClosure::new_dyn(body).compile(false).ok()
}

/// One registry entry; `closure: None` is a tombstone.
struct Slot {
    closure: Option<MlxClosure>,
    applies: u64,
}

/// Registry key: the stretch plus its input shape, because closures are
/// compiled shape-specifically.
type SlotKey = (Stretch, [i64; 4]);

thread_local! {
    // Per-thread: closures are compiled, applied and dropped on one thread.
    // An absent entry is rebuildable, `Slot { closure: None, .. }` is not.
    static SLOTS: RefCell<HashMap<SlotKey, Slot>> = RefCell::new(HashMap::new());
}

#[cfg(test)]
fn threshold_override() -> Option<u64> {
    REFRESH_THRESHOLD.with(|cell| cell.get())
}

#[cfg(not(test))]
fn threshold_override() -> Option<u64> {
    None
}

/// Successful applies after which a closure is dropped and rebuilt.
fn refresh_threshold() -> u64 {
    threshold_override().unwrap_or_else(compile_cache_refresh_threshold)
}

/// Record the permanent eager fallback for `key`.
fn tombstone(key: SlotKey) {
    SLOTS.with(|slots| {
        slots.borrow_mut().insert(
            key,
            Slot {
                closure: None,
                applies: 0,
            },
        );
    });
}

fn apply_once(closure: &MlxClosure, inputs: &[&MlxArray]) -> Option<MlxArray> {
    let mut outputs = try_apply_with_abort_safety(closure, inputs)?;
    // Exactly one output is the contract; anything else means the closure does
    // not match the caller's graph and must not be trusted.
    if outputs.len() == 1 {
        outputs.pop()
    } else {
        None
    }
}

/// One elementwise stretch of `rows` positions: the compiled result, or `None`
/// when the caller must use its eager path.
///
/// `key` is the input shape. `build` runs at most once per (stretch, shape) and
/// only after [`enabled`] and the row gate said yes, so callers can pass a
/// closure that allocates.
pub(crate) fn apply(
    stretch: Stretch,
    rows: i32,
    key: [i64; 4],
    build: impl FnOnce() -> Option<MlxClosure>,
    inputs: &[&MlxArray],
) -> Option<MlxArray> {
    if !enabled(stretch) || !(1..=max_rows()).contains(&rows) {
        return None;
    }

    let slot_key = (stretch, key);
    // Take the slot out of the map instead of borrowing it: `build` and the
    // apply both re-enter MLX, and a live `RefCell` borrow across that would
    // panic on any re-entrant lookup (and hides the failure as a borrow error
    // rather than an eager fallback).
    let slot = SLOTS.with(|slots| slots.borrow_mut().remove(&slot_key));
    let (closure, applies) = match slot {
        Some(Slot {
            closure: None,
            applies,
        }) => {
            // Put the tombstone back before returning: it is the record that
            // keeps this (stretch, shape) on the eager path, and losing it
            // would rebuild the closure on the next step.
            SLOTS.with(|slots| {
                slots.borrow_mut().insert(
                    slot_key,
                    Slot {
                        closure: None,
                        applies,
                    },
                );
            });
            return None;
        }
        Some(Slot {
            closure: Some(closure),
            applies,
        }) => (closure, applies),
        None => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(build)) {
            Ok(Some(closure)) => (closure, 0),
            // `Ok(None)` and a panicking builder are both build failures:
            // either way the stretch must tombstone so later steps take the
            // eager path permanently instead of re-panicking every step.
            Ok(None) | Err(_) => {
                tracing::warn!(
                    stretch = stretch.name(),
                    "Flash Next elementwise stretch failed to compile; using eager ops"
                );
                tombstone(slot_key);
                return None;
            }
        },
    };

    let result = if failure_injected(stretch) {
        None
    } else {
        apply_once(&closure, inputs)
    };
    // A fresh closure's fused kernel is only compiled when it is first
    // evaluated, and a failure there would kill the request (or the process)
    // from inside a decode step. Evaluate the first output right away, where
    // an eager fallback is still possible.
    let result = result.filter(|output| applies > 0 || validate_first_use(stretch, output));

    match result {
        Some(output) => {
            note_applied(stretch);
            let applies = applies + 1;
            if applies < refresh_threshold() {
                SLOTS.with(|slots| {
                    slots.borrow_mut().insert(
                        slot_key,
                        Slot {
                            closure: Some(closure),
                            applies,
                        },
                    );
                });
            }
            // At the threshold the closure is dropped by leaving the entry
            // absent: the next call rebuilds instead of tombstoning.
            Some(output)
        }
        None => {
            // A failing closure is not retried; a tombstone keeps every later
            // step on the eager path on this thread.
            tombstone(slot_key);
            None
        }
    }
}

/// Evaluate `output` once; `false` means the compiled kernel cannot run here.
fn validate_first_use(stretch: Stretch, output: &MlxArray) -> bool {
    if eval_failure_injected(stretch) {
        return false;
    }
    match mlx_sys::try_eval(&[output]) {
        Ok(()) => true,
        Err(error) => {
            tracing::warn!(
                stretch = stretch.name(),
                %error,
                "Flash Next elementwise stretch failed its first evaluation; using eager ops"
            );
            false
        }
    }
}

#[cfg(not(test))]
fn eval_failure_injected(_stretch: Stretch) -> bool {
    false
}

#[cfg(test)]
fn eval_failure_injected(stretch: Stretch) -> bool {
    INJECTED_EVAL_FAILURES.with(|failures| failures.borrow_mut().remove(&stretch))
}

#[cfg(not(test))]
fn failure_injected(_stretch: Stretch) -> bool {
    false
}

#[cfg(test)]
fn note_applied(stretch: Stretch) {
    APPLIED.with(|counts| {
        let count = &counts[stretch.index()];
        count.set(count.get() + 1);
    });
}

#[cfg(not(test))]
fn note_applied(_stretch: Stretch) {}

#[cfg(test)]
fn failure_injected(stretch: Stretch) -> bool {
    INJECTED_FAILURES.with(|failures| failures.borrow_mut().remove(&stretch))
}

#[cfg(test)]
thread_local! {
    // Test override of the env mask; `None` falls back to it.
    static FORCE: Cell<Option<bool>> = const { Cell::new(None) };
    // Test override that enables exactly the stretches in a bit set (bit i is
    // `Stretch::ALL[i]`), for bisecting a divergence to one stretch.
    static FORCE_ONLY: Cell<Option<u16>> = const { Cell::new(None) };
    // Stretches whose next apply must behave as a failed apply.
    static INJECTED_FAILURES: RefCell<HashSet<Stretch>> = RefCell::new(HashSet::new());
    // Stretches whose next first-use evaluation must behave as a failed eval.
    static INJECTED_EVAL_FAILURES: RefCell<HashSet<Stretch>> = RefCell::new(HashSet::new());
    // Test override of `refresh_threshold`.
    static REFRESH_THRESHOLD: Cell<Option<u64>> = const { Cell::new(None) };
    // Successful compiled applies per stretch on this thread, so a
    // differential test can prove the fused path ran instead of silently
    // falling back to eager.
    static APPLIED: [Cell<u64>; 7] = const { [const { Cell::new(0) }; 7] };
}

/// Restores the previous test override on drop, so a test cannot leak a mask
/// into the next test that shares its thread.
#[cfg(test)]
#[must_use = "hold the guard for the whole test body"]
pub(crate) struct ForceGuard {
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for ForceGuard {
    fn drop(&mut self) {
        FORCE.with(|force| force.set(self.previous));
    }
}

/// Restores the previous exact-set override on drop.
#[cfg(test)]
#[must_use = "hold the guard for the whole test body"]
pub(crate) struct ForceOnlyGuard {
    previous: Option<u16>,
}

#[cfg(test)]
impl Drop for ForceOnlyGuard {
    fn drop(&mut self) {
        FORCE_ONLY.with(|only| only.set(self.previous));
    }
}

/// Enable exactly `stretches` (and no others), whatever the env mask says.
#[cfg(test)]
pub(crate) fn force_only_for_test(stretches: &[Stretch]) -> ForceOnlyGuard {
    let bits = stretches
        .iter()
        .fold(0u16, |bits, stretch| bits | (1 << stretch.index()));
    let previous = FORCE_ONLY.with(|only| only.replace(Some(bits)));
    ForceOnlyGuard { previous }
}

/// Override [`enabled`] for tests: `Some(true)` enables every stretch,
/// `Some(false)` disables every stretch, `None` uses the env mask.
#[cfg(test)]
pub(crate) fn force_for_test(mode: Option<bool>) -> ForceGuard {
    let previous = FORCE.with(|force| force.replace(mode));
    ForceGuard { previous }
}

/// Make the next apply of `stretch` fail, which must tombstone the slot.
#[cfg(test)]
pub(crate) fn fail_next_apply_for_test(stretch: Stretch) {
    INJECTED_FAILURES.with(|failures| {
        failures.borrow_mut().insert(stretch);
    });
}

/// Make the next first-use evaluation of `stretch` fail, which must tombstone
/// the slot just like a failed apply.
#[cfg(test)]
pub(crate) fn fail_next_eval_for_test(stretch: Stretch) {
    INJECTED_EVAL_FAILURES.with(|failures| {
        failures.borrow_mut().insert(stretch);
    });
}

/// Clear every slot, injected failure and threshold override.
#[cfg(test)]
pub(crate) fn reset_registry_for_test() {
    SLOTS.with(|slots| slots.borrow_mut().clear());
    INJECTED_FAILURES.with(|failures| failures.borrow_mut().clear());
    INJECTED_EVAL_FAILURES.with(|failures| failures.borrow_mut().clear());
    REFRESH_THRESHOLD.with(|cell| cell.set(None));
}

/// Whether any shape of `stretch` is tombstoned on this thread.
#[cfg(test)]
pub(crate) fn tombstoned_for_test(stretch: Stretch) -> bool {
    SLOTS.with(|slots| {
        slots
            .borrow()
            .iter()
            .any(|((slot_stretch, _), slot)| *slot_stretch == stretch && slot.closure.is_none())
    })
}

/// Compiled applies that succeeded for `stretch` on this thread since it
/// started.
#[cfg(test)]
pub(crate) fn applied_for_test(stretch: Stretch) -> u64 {
    APPLIED.with(|counts| counts[stretch.index()].get())
}

/// Override the apply count after which a closure is rebuilt.
#[cfg(test)]
pub(crate) fn set_refresh_threshold_for_test(threshold: Option<u64>) {
    REFRESH_THRESHOLD.with(|cell| cell.set(threshold));
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use mlx_sys::{MlxDtype, add, eval, multiply};

    const KEY: [i64; 4] = [1, 1, 4, 4];

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    fn eval_f32(array: &MlxArray) -> Vec<f32> {
        eval(&[array]);
        array.data_f32().to_vec()
    }

    /// Raw bits, because the compiled and eager graphs must be bit-identical.
    fn bits(array: &MlxArray) -> Vec<u32> {
        eval_f32(array)
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    fn wave(len: usize, freq: f32, amplitude: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + 1.0) * freq).sin() * amplitude + offset)
            .collect()
    }

    /// `[rows, 1, 4]` f32 inputs for a three-op stretch.
    fn tiny_inputs(rows: i32) -> [MlxArray; 3] {
        let len = rows as usize * 4;
        [
            array_f32(&wave(len, 0.37, 1.0, 0.5), &[rows, 1, 4]),
            array_f32(&wave(len, 0.61, 0.5, 0.25), &[rows, 1, 4]),
            array_f32(&wave(len, 0.23, 0.75, -0.5), &[rows, 1, 4]),
        ]
    }

    fn input_refs(inputs: &[MlxArray; 3]) -> [&MlxArray; 3] {
        [&inputs[0], &inputs[1], &inputs[2]]
    }

    /// `a * b + c`, the smallest stretch with the real call shape.
    fn tiny_body(inputs: &MlxVectorArray) -> Vec<MlxArray> {
        vec![add(
            &multiply(&inputs.get(0), &inputs.get(1), None),
            &inputs.get(2),
            None,
        )]
    }

    #[test]
    fn parse_mask_reads_stretch_names() {
        reset_registry_for_test();
        let _guard = force_for_test(None);

        const NONE: [bool; 7] = [false; 7];
        const ALL: [bool; 7] = [true; 7];
        let norm_only = [true, false, false, false, false, false, false];
        let norm_and_beta = [true, false, false, false, false, true, false];

        assert_eq!(
            Stretch::ALL.map(Stretch::name),
            [
                "hc_norm", "hc_low", "hc_mix", "hc_gate", "hc_write", "gdn_beta", "gdn_post",
            ],
            "mask order and names must not drift"
        );

        for (raw, expected) in [
            (None, NONE),
            (Some(""), NONE),
            (Some("0"), NONE),
            (Some("off"), NONE),
            (Some("1"), ALL),
            (Some("all"), ALL),
            (Some("hc_norm"), norm_only),
            (Some("hc_norm, GDN_BETA"), norm_and_beta),
            (Some("bogus"), NONE),
            (Some("hc_norm,bogus"), norm_only),
        ] {
            assert_eq!(parse_mask(raw, false), expected, "mask for {raw:?}");
            // A set variable never depends on the unset default.
            if raw.is_some() {
                assert_eq!(
                    parse_mask(raw, true),
                    expected,
                    "mask for {raw:?} (default on)"
                );
            }
        }
    }

    #[test]
    fn unset_variable_follows_the_default_policy() {
        assert_eq!(parse_mask(None, true), [true; 7], "shipping default is on");
        assert_eq!(
            parse_mask(None, false),
            [false; 7],
            "unit tests default off"
        );
        assert_eq!(parse_mask(Some("0"), true), [false; 7], "kill switch");
        assert_eq!(parse_mask(Some(""), true), [false; 7], "empty disables");
    }

    #[test]
    fn row_limit_defaults_to_single_token_decode() {
        assert_eq!(parse_max_rows(None), 1);
        assert_eq!(parse_max_rows(Some("")), 1);
        assert_eq!(parse_max_rows(Some("0")), 1);
        assert_eq!(parse_max_rows(Some("-3")), 1);
        assert_eq!(parse_max_rows(Some("abc")), 1);
        assert_eq!(parse_max_rows(Some("4")), 4);
        assert_eq!(parse_max_rows(Some(" 8 ")), 8);
        assert_eq!(parse_max_rows(Some("99")), FUSE_MAX_ROWS);
    }

    #[test]
    fn rows_gate_limits_fusion_to_decode_sized_stretches() {
        reset_registry_for_test();
        let _guard = force_for_test(Some(true));

        for rows in [0, FUSE_MAX_ROWS + 1] {
            let inputs = tiny_inputs(1);
            let key = [i64::from(rows), 1, 4, 4];
            let applied = apply(
                Stretch::HcNorm,
                rows,
                key,
                || compile(tiny_body),
                &input_refs(&inputs),
            );
            assert!(applied.is_none(), "rows={rows} must not compile");
        }
        // One key per row count: closures are shape-specific.
        for rows in [1, FUSE_MAX_ROWS] {
            let inputs = tiny_inputs(rows);
            let key = [i64::from(rows), 1, 4, 4];
            let applied = apply(
                Stretch::HcNorm,
                rows,
                key,
                || compile(tiny_body),
                &input_refs(&inputs),
            );
            assert!(applied.is_some(), "rows={rows} must compile");
        }
    }

    #[test]
    fn tiny_stretch_matches_eager_ops_bitwise() {
        reset_registry_for_test();
        let _guard = force_for_test(Some(true));
        let inputs = tiny_inputs(1);
        let refs = input_refs(&inputs);

        let eager = add(&multiply(&inputs[0], &inputs[1], None), &inputs[2], None);
        let expected = bits(&eager);

        let compiled = apply(Stretch::HcMix, 1, KEY, || compile(tiny_body), &refs);
        let compiled = compiled.expect("compiled stretch");
        assert_eq!(bits(&compiled), expected, "compiled graph diverged");

        {
            // Nested guard: masks only this block, the outer one stays active.
            let _disabled = force_for_test(Some(false));
            let skipped = apply(Stretch::HcGate, 1, KEY, || compile(tiny_body), &refs);
            assert!(skipped.is_none(), "disabled stretch must fall back");
        }

        let enabled_again = apply(Stretch::HcMix, 1, KEY, || compile(tiny_body), &refs);
        assert!(enabled_again.is_some(), "outer guard must be restored");
    }

    #[test]
    fn apply_failure_tombstones_the_stretch() {
        reset_registry_for_test();
        let _guard = force_for_test(Some(true));
        let inputs = tiny_inputs(1);
        let refs = input_refs(&inputs);
        let builds = Cell::new(0u32);
        let build = || {
            builds.set(builds.get() + 1);
            compile(tiny_body)
        };

        fail_next_apply_for_test(Stretch::HcLow);
        let first = apply(Stretch::HcLow, 1, KEY, build, &refs);
        assert!(first.is_none(), "injected failure must return None");
        assert!(tombstoned_for_test(Stretch::HcLow), "must tombstone");
        assert_eq!(builds.get(), 1, "the closure was built before it failed");

        let second = apply(Stretch::HcLow, 1, KEY, build, &refs);
        assert!(second.is_none(), "a tombstone stays on the eager path");
        assert_eq!(builds.get(), 1, "a tombstone must not rebuild");
        assert!(tombstoned_for_test(Stretch::HcLow));
    }

    /// Compiled closures live in a thread-local registry and are still live
    /// when their thread exits. Repeatedly spawn workers that apply a stretch
    /// and leave their slots in place; every worker must join cleanly and every
    /// closure payload must be released (no abort in a TLS destructor, no
    /// payload kept alive by MLX's compile cache).
    #[test]
    fn registry_closures_are_released_at_thread_exit() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static DROPPED: AtomicUsize = AtomicUsize::new(0);
        struct Probe;
        impl Drop for Probe {
            fn drop(&mut self) {
                DROPPED.fetch_add(1, Ordering::SeqCst);
            }
        }

        const WORKERS: usize = 6;
        for _ in 0..WORKERS {
            let worker = std::thread::spawn(|| {
                reset_registry_for_test();
                let _guard = force_for_test(Some(true));
                let inputs = tiny_inputs(1);
                let refs = input_refs(&inputs);
                let probe = Probe;
                let build = move || {
                    compile(move |inputs| {
                        let _keep = &probe;
                        tiny_body(inputs)
                    })
                };
                let expected = {
                    let _eager = force_for_test(Some(false));
                    let eager = add(&multiply(&inputs[0], &inputs[1], None), &inputs[2], None);
                    bits(&eager)
                };
                let first = apply(Stretch::HcMix, 1, KEY, build, &refs).expect("first apply");
                assert_eq!(bits(&first), expected);
                // The slot, with its live closure, stays in the registry until
                // this thread exits.
                assert!(!tombstoned_for_test(Stretch::HcMix));
            });
            worker.join().expect("worker must exit cleanly");
        }
        assert_eq!(
            DROPPED.load(Ordering::SeqCst),
            WORKERS,
            "every closure payload must be released when its thread exits"
        );
    }

    #[test]
    fn failed_first_evaluation_tombstones_the_stretch() {
        reset_registry_for_test();
        let _guard = force_for_test(Some(true));
        let inputs = tiny_inputs(1);
        let refs = input_refs(&inputs);
        let builds = Cell::new(0u32);
        let build = || {
            builds.set(builds.get() + 1);
            compile(tiny_body)
        };

        fail_next_eval_for_test(Stretch::HcGate);
        let first = apply(Stretch::HcGate, 1, KEY, build, &refs);
        assert!(first.is_none(), "a failed first evaluation must fall back");
        assert!(tombstoned_for_test(Stretch::HcGate));
        assert_eq!(
            applied_for_test(Stretch::HcGate),
            0,
            "a failed eval is not an apply"
        );
        let second = apply(Stretch::HcGate, 1, KEY, build, &refs);
        assert!(second.is_none(), "the tombstone must hold");
        assert_eq!(builds.get(), 1, "no rebuild after the tombstone");

        // A healthy stretch validates once and then keeps applying.
        let ok = apply(Stretch::HcLow, 1, KEY, build, &refs);
        assert!(ok.is_some(), "a healthy first evaluation must pass");
        assert!(!tombstoned_for_test(Stretch::HcLow));
    }

    #[test]
    fn panicking_builder_tombstones_the_stretch() {
        reset_registry_for_test();
        let _guard = force_for_test(Some(true));
        let inputs = tiny_inputs(1);
        let refs = input_refs(&inputs);
        let builds = Cell::new(0u32);
        let build = || {
            builds.set(builds.get() + 1);
            panic!("injected")
        };

        let first = apply(Stretch::GdnBeta, 1, KEY, build, &refs);
        assert!(first.is_none(), "a panicking builder must return None");
        assert!(tombstoned_for_test(Stretch::GdnBeta), "must tombstone");

        let second = apply(Stretch::GdnBeta, 1, KEY, build, &refs);
        assert!(second.is_none(), "a tombstone stays on the eager path");
        assert_eq!(builds.get(), 1, "a tombstone must not rebuild");
    }

    #[test]
    fn refresh_threshold_rebuilds_the_closure() {
        reset_registry_for_test();
        let _guard = force_for_test(Some(true));
        set_refresh_threshold_for_test(Some(3));
        let inputs = tiny_inputs(1);
        let refs = input_refs(&inputs);
        let builds = Cell::new(0u32);
        let build = || {
            builds.set(builds.get() + 1);
            compile(tiny_body)
        };

        let mut expected: Option<Vec<u32>> = None;
        for step in 0..7 {
            let output = apply(Stretch::HcWrite, 1, KEY, build, &refs);
            let actual = bits(&output.expect("apply"));
            let same = expected.as_ref().is_none_or(|expected| expected == &actual);
            assert!(same, "step {step} diverged from the first result");
            expected = Some(actual);
        }
        // Initial build, then one rebuild after 3 and after 6 applies.
        assert_eq!(builds.get(), 3);
    }
}
