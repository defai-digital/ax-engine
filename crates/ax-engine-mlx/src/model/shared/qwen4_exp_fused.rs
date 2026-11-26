//! Compiled elementwise stretches of the Qwen 3.8 Flash Next (`qwen4_exp`)
//! layers, shared by the gated residual, the gated delta network, the QSA
//! rotary application, the attention output gate and the MoE tails.
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
//! - decode-sized inputs are compiled up to [`FUSE_DEFAULT_ROWS`] by default
//!   (at most [`FUSE_MAX_ROWS`]), where the host is the bottleneck;
//! - prefill forwards of [`PREFILL_FUSE_MIN_ROWS`] rows or more run the
//!   compiled stretches too on Neural Accelerator hosts, where that was
//!   measured (`AX_FLASH_NEXT_FUSE_PREFILL=0` turns it off, `=1` turns it on
//!   on any host): there the device is the bottleneck, and one fused kernel
//!   replaces several passes over a chunk-sized tensor. Prefill has its
//!   own closures, apart from the decode ones, and a closure is rebuilt
//!   after it has traced [`TRACES_PER_CLOSURE`] input signatures. The
//!   native kernels stay decode-only: at prefill width they gain nothing
//!   and pay a first-use cost per row count;
//! - a stretch whose closure fails to compile or to apply is tombstoned per
//!   thread, so later steps take the eager path permanently instead of
//!   re-flooding MLX errors every step;
//! - a compiled closure is dropped after [`refresh_threshold`] successful
//!   applies, so a long-lived process recompiles against fresh MLX state;
//! - the registry borrow is never held across an MLX call, because both
//!   building and applying a closure re-enter MLX on the same thread.
//!
//! On by default. `AX_FLASH_NEXT_FUSE=0` is the kill switch; a comma-separated
//! stretch list enables just those stretches.

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

/// Cap on the `seq` of a decode-sized fused stretch.
///
/// A compiled closure is shape-specific (`compile(false)`), so every distinct
/// row count traces each stretch once more, and a stray 2..8 row remainder
/// would pay that inside a request.
pub(crate) const FUSE_MAX_ROWS: i32 = 8;

/// Rows used by default: single-token decode, the only place with a measured
/// win and the only shape the load-time warm-up compiles. Verifier windows are
/// opt-in through `AX_FLASH_NEXT_FUSE_ROWS` (clamped to [`FUSE_MAX_ROWS`]).
const FUSE_DEFAULT_ROWS: i32 = 1;

/// Env name of the opt-in row limit.
const FUSE_ROWS_ENV: &str = "AX_FLASH_NEXT_FUSE_ROWS";

/// Rows from which a prefill forward runs the compiled stretches.
///
/// MacBook Pro M5 Max 128 GB, Qwen 3.8 Flash Next MXFP4, one prefill forward,
/// eager against compiled: 160.5 against 156.6 ms at 64 rows, 449.3 against
/// 396.0 at 777 and 769.7 against 653.7 at 1,497, with bit-identical logits
/// and state. The first forward at a new row count evaluates every stretch
/// once: 741.8 against 639.9 ms at 1,411 rows, and 167.7 against 169.4 at
/// 103, where it costs what one later forward wins back. Shorter forwards
/// stay eager.
pub(crate) const PREFILL_FUSE_MIN_ROWS: i32 = 64;

/// Traces one closure holds before it is rebuilt.
///
/// MLX keeps a trace per input shape and dtype of a compiled function and
/// searches them one by one on every call, so a server that sees a new
/// prompt length on every request must not let them accumulate.
pub(crate) const TRACES_PER_CLOSURE: usize = 32;

/// Switch of the prefill-width stretches: off values turn them off, any
/// other value turns them on whatever the host.
const FUSE_PREFILL_ENV: &str = "AX_FLASH_NEXT_FUSE_PREFILL";

/// Parse the prefill switch: `None` when unset, otherwise whether it is on.
fn parse_prefill_switch(raw: Option<&str>) -> Option<bool> {
    raw.map(|raw| {
        !matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off" | "no"
        )
    })
}

/// Whether prefill-width rows may run the compiled stretches on this thread.
///
/// Unset, that is the case on Neural Accelerator hosts (M5 and later), the
/// only ones it was measured on. Like the stretch mask, unit tests start
/// with it off unless they force it.
fn prefill_rows_enabled() -> bool {
    if let Some(forced) = forced_prefill_rows() {
        return forced;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        parse_prefill_switch(std::env::var(FUSE_PREFILL_ENV).ok().as_deref())
            .unwrap_or_else(|| !cfg!(test) && crate::hardware::neural_accelerator_active())
    })
}

/// The row gate of one stretch: decode-sized rows for every stretch, and
/// prefill-width rows for the compiled ones while that is switched on.
fn row_gate(native: bool, rows: i32, decode_rows: i32, prefill_rows: bool) -> bool {
    (1..=decode_rows).contains(&rows) || (!native && prefill_rows && rows >= PREFILL_FUSE_MIN_ROWS)
}

/// Whether `stretch` may run at `rows` on this thread.
fn rows_allowed(stretch: Stretch, rows: i32) -> bool {
    row_gate(
        stretch.is_native(),
        rows,
        max_rows(),
        prefill_rows_enabled(),
    )
}

/// Parse the row limit; unset, empty, zero or garbage keep the default.
fn parse_max_rows(raw: Option<&str>) -> i32 {
    raw.map(str::trim)
        .and_then(|raw| raw.parse::<i32>().ok())
        .filter(|rows| *rows >= 1)
        .map_or(FUSE_DEFAULT_ROWS, |rows| rows.min(FUSE_MAX_ROWS))
}

/// Largest decode-sized row count a stretch may fuse at on this thread.
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
/// here. A probe over random bf16/f32/f16 inputs showed that every segment
/// between the decay's `log1p` and the L2 normalisation's `power(x, -0.5)` is
/// bit-identical when compiled, and that the three-call decay chain equals the
/// original 14-op eager chain bit for bit. Those two ops stay eager outside the
/// decay and L2 stretches: MLX leaves unqualified FP32 math functions on fast
/// variants inside JIT-fused kernels, so compiling `log1p` (after any preceding
/// op) or `power` changes the recurrent state bits. The two MoE tails were
/// probed bit-identical as well; the selected-expert index read-back and the
/// expert gathers stay outside every stretch. The two halves of the QSA rotary
/// product were probed bit-identical too; the half-table slices, the final
/// concatenate and the tail pass-through stay outside every stretch. The QSA
/// output gate and the MoE routing tail (whose per-row sum reduction is kept
/// exactly like the `gdn_l2_sum` reduction) were probed bit-identical too.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Stretch {
    HcNorm,
    HcLow,
    HcMix,
    HcGate,
    HcWrite,
    GdnBeta,
    GdnPost,
    GdnConvAct,
    GdnDecayPre,
    GdnDecayPost,
    GdnL2Sum,
    GdnQScale,
    GdnKScale,
    MoeSwiglu,
    MoeCombine,
    QsaRopeLo,
    QsaRopeHi,
    QsaGate,
    MoeRouteTail,
    /// Native (hand-written Metal) decay and beta of the GDN gates.
    GdnNativeDecay,
    /// Native q/k inverse-norm scale of the GDN L2 normalisation.
    GdnNativeQk,
    /// Native hyper-connection read: two GEMV kernels with the low-rank
    /// activation, write gate and stream mean fused into their epilogues.
    HcNativeLowGate,
}

impl Stretch {
    /// Mask order; [`parse_mask`] indexes with [`Stretch::index`].
    pub(crate) const ALL: [Stretch; 22] = [
        Stretch::HcNorm,
        Stretch::HcLow,
        Stretch::HcMix,
        Stretch::HcGate,
        Stretch::HcWrite,
        Stretch::GdnBeta,
        Stretch::GdnPost,
        Stretch::GdnConvAct,
        Stretch::GdnDecayPre,
        Stretch::GdnDecayPost,
        Stretch::GdnL2Sum,
        Stretch::GdnQScale,
        Stretch::GdnKScale,
        Stretch::MoeSwiglu,
        Stretch::MoeCombine,
        Stretch::QsaRopeLo,
        Stretch::QsaRopeHi,
        Stretch::QsaGate,
        Stretch::MoeRouteTail,
        Stretch::GdnNativeDecay,
        Stretch::GdnNativeQk,
        Stretch::HcNativeLowGate,
    ];

    /// Width of every mask, counter and forced-bit array.
    pub(crate) const COUNT: usize = Self::ALL.len();

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
            Stretch::GdnConvAct => "gdn_conv_act",
            Stretch::GdnDecayPre => "gdn_decay_pre",
            Stretch::GdnDecayPost => "gdn_decay_post",
            Stretch::GdnL2Sum => "gdn_l2_sum",
            Stretch::GdnQScale => "gdn_q_scale",
            Stretch::GdnKScale => "gdn_k_scale",
            Stretch::MoeSwiglu => "moe_swiglu",
            Stretch::MoeCombine => "moe_combine",
            Stretch::QsaRopeLo => "qsa_rope_lo",
            Stretch::QsaRopeHi => "qsa_rope_hi",
            Stretch::QsaGate => "qsa_gate",
            Stretch::MoeRouteTail => "moe_route_tail",
            Stretch::GdnNativeDecay => "gdn_native_decay",
            Stretch::GdnNativeQk => "gdn_native_qk",
            Stretch::HcNativeLowGate => "hc_native_low_gate",
        }
    }

    /// Whether this stretch is a hand-written Metal kernel rather than a
    /// compiled closure. Native stretches use [`native_allowed`].
    pub(crate) fn is_native(self) -> bool {
        matches!(
            self,
            Stretch::GdnNativeDecay | Stretch::GdnNativeQk | Stretch::HcNativeLowGate
        )
    }

    #[cfg(test)]
    /// The native stretch that takes over this compiled stretch's work when it
    /// applies, so tests can tell "never ran" from "replaced by its native
    /// twin".
    pub(crate) fn replaced_by(self) -> Option<Stretch> {
        match self {
            Stretch::GdnBeta | Stretch::GdnDecayPre | Stretch::GdnDecayPost => {
                Some(Stretch::GdnNativeDecay)
            }
            Stretch::GdnQScale | Stretch::GdnKScale => Some(Stretch::GdnNativeQk),
            Stretch::HcLow | Stretch::HcMix | Stretch::HcGate => Some(Stretch::HcNativeLowGate),
            _ => None,
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
fn parse_mask(raw: Option<&str>, unset_means_all: bool) -> [bool; Stretch::COUNT] {
    let mut mask = [false; Stretch::COUNT];
    let Some(raw) = raw else {
        return [unset_means_all; Stretch::COUNT];
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return mask;
    }
    let lowered = raw.to_ascii_lowercase();
    match lowered.as_str() {
        "0" | "false" | "off" | "no" => return mask,
        "1" | "true" | "on" | "all" => return [true; Stretch::COUNT],
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
fn env_mask() -> &'static [bool; Stretch::COUNT] {
    static MASK: OnceLock<[bool; Stretch::COUNT]> = OnceLock::new();
    MASK.get_or_init(|| parse_mask(std::env::var(FUSE_ENV).ok().as_deref(), !cfg!(test)))
}

#[cfg(test)]
fn forced_only_bits() -> Option<u32> {
    FORCE_ONLY.with(|only| only.get())
}

#[cfg(not(test))]
fn forced_only_bits() -> Option<u32> {
    None
}

#[cfg(test)]
fn forced_prefill_rows() -> Option<bool> {
    FORCE_PREFILL.with(|force| force.get())
}

#[cfg(not(test))]
fn forced_prefill_rows() -> Option<bool> {
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

/// Gate for a native stretch: the same enable mask as a compiled one, at
/// decode-sized rows only.
pub(crate) fn native_allowed(stretch: Stretch, rows: i32) -> bool {
    debug_assert!(stretch.is_native());
    enabled(stretch) && rows_allowed(stretch, rows)
}

/// Count one successful native application (test counters only).
pub(crate) fn note_native_applied(stretch: Stretch) {
    debug_assert!(stretch.is_native());
    note_applied(stretch);
}

/// Compile one closure body, or `None` when MLX rejects it.
///
/// `shapeless = false`: MLX traces the body once per input shape, which buys
/// the tighter fusion of a shape-specific graph.
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
    /// Signatures of the traces this closure holds, apart from the one-row
    /// decode trace, which the hot path does not hash.
    traces: Vec<u64>,
}

impl Slot {
    /// A live closure that would have to hold one trace too many.
    fn traces_exhausted(&self, signature: u64) -> bool {
        self.closure.is_some()
            && self.traces.len() >= TRACES_PER_CLOSURE
            && !self.traces.contains(&signature)
    }
}

/// Which closure of a stretch serves a row count.
///
/// Decode and prefill do not share one: a decode closure then holds the one
/// trace its hot path looks up, a prefill closure can be rebuilt without
/// making the next decode step trace again, and a failure at prefill width
/// leaves decode fused. The regime follows the width of the operation, not
/// the kind of forward: a decode step that recomputes 64 pooled blocks or
/// more after a restore runs the prefill closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum RowRegime {
    Decode,
    Prefill,
}

impl RowRegime {
    fn of(rows: i32) -> Self {
        if rows >= PREFILL_FUSE_MIN_ROWS {
            Self::Prefill
        } else {
            Self::Decode
        }
    }
}

/// What MLX keys a trace of one closure on: the shape and dtype of every
/// input.
fn trace_signature(inputs: &[&MlxArray]) -> u64 {
    // FNV-1a over the dtype, rank and extents of every input.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut mix = |value: i64| {
        hash = (hash ^ value.cast_unsigned()).wrapping_mul(0x0000_0100_0000_01b3);
    };
    for input in inputs {
        let shape = input.shape();
        mix(input.dtype() as i64);
        mix(shape.len() as i64);
        for extent in shape {
            mix(i64::from(extent));
        }
    }
    hash
}

/// Registry key: the stretch, the row regime and the caller's geometry key.
///
/// The geometry key names what the closure body depends on (head sizes,
/// dtype, layout) and never the row count: every row count of a regime goes
/// through the same closure, which MLX traces once per input shape.
type SlotKey = (Stretch, RowRegime, [i64; 4]);

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
                traces: Vec::new(),
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
/// `key` is the geometry the closure body depends on, without the row count
/// (see [`SlotKey`]). `build` runs only after [`enabled`] and the row gate
/// said yes, once per closure generation, so callers can pass a closure that
/// allocates. Its body must not capture the row count either.
pub(crate) fn apply(
    stretch: Stretch,
    rows: i32,
    key: [i64; 4],
    build: impl FnOnce() -> Option<MlxClosure>,
    inputs: &[&MlxArray],
) -> Option<MlxArray> {
    if !enabled(stretch) || !rows_allowed(stretch, rows) {
        return None;
    }

    let regime = RowRegime::of(rows);
    let slot_key = (stretch, regime, key);
    // The one-row decode step is the hot path and has one trace per closure;
    // everything else is identified, so that each new trace is validated and
    // the traces of a closure are counted.
    let signature = (rows > 1).then(|| trace_signature(inputs));
    // Take the slot out of the map instead of borrowing it: `build` and the
    // apply both re-enter MLX, and a live `RefCell` borrow across that would
    // panic on any re-entrant lookup (and hides the failure as a borrow error
    // rather than an eager fallback).
    // A closure out of traces is dropped here and rebuilt below.
    let slot = SLOTS
        .with(|slots| slots.borrow_mut().remove(&slot_key))
        .filter(|slot| !signature.is_some_and(|signature| slot.traces_exhausted(signature)));
    let (closure, applies, mut traces) = match slot {
        Some(Slot {
            closure: None,
            applies,
            traces,
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
                        traces,
                    },
                );
            });
            return None;
        }
        Some(Slot {
            closure: Some(closure),
            applies,
            traces,
        }) => (closure, applies, traces),
        None => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(build)) {
            Ok(Some(closure)) => (closure, 0, Vec::new()),
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
    // A new trace's fused kernel is only compiled when it is first
    // evaluated, and a failure there would kill the request (or the process)
    // from inside a forward. Evaluate its first output right away, where an
    // eager fallback is still possible: on a fresh closure, and for every
    // input signature a closure has not traced yet.
    let new_trace = applies == 0 || signature.is_some_and(|signature| !traces.contains(&signature));
    let result = result.filter(|output| !new_trace || validate_first_use(stretch, output));

    match result {
        Some(output) => {
            note_applied(stretch);
            let applies = applies + 1;
            if let Some(signature) = signature
                && !traces.contains(&signature)
            {
                traces.push(signature);
            }
            if applies < refresh_threshold() {
                SLOTS.with(|slots| {
                    slots.borrow_mut().insert(
                        slot_key,
                        Slot {
                            closure: Some(closure),
                            applies,
                            traces,
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
    static FORCE_ONLY: Cell<Option<u32>> = const { Cell::new(None) };
    // Test override of the prefill-width switch; `None` falls back to the env.
    static FORCE_PREFILL: Cell<Option<bool>> = const { Cell::new(None) };
    // Stretches whose next apply must behave as a failed apply.
    static INJECTED_FAILURES: RefCell<HashSet<Stretch>> = RefCell::new(HashSet::new());
    // Stretches whose next first-use evaluation must behave as a failed eval.
    static INJECTED_EVAL_FAILURES: RefCell<HashSet<Stretch>> = RefCell::new(HashSet::new());
    // Test override of `refresh_threshold`.
    static REFRESH_THRESHOLD: Cell<Option<u64>> = const { Cell::new(None) };
    // Successful compiled applies per stretch on this thread, so a
    // differential test can prove the fused path ran instead of silently
    // falling back to eager.
    static APPLIED: [Cell<u64>; Stretch::COUNT] =
        const { [const { Cell::new(0) }; Stretch::COUNT] };
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
    previous: Option<u32>,
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
        .fold(0u32, |bits, stretch| bits | (1 << stretch.index()));
    let previous = FORCE_ONLY.with(|only| only.replace(Some(bits)));
    ForceOnlyGuard { previous }
}

/// Override [`enabled`] for tests: `Some(true)` enables every stretch,
/// `Some(false)` disables every stretch, `None` uses the env mask.
/// Enable exactly the compiled stretches (every native kernel stays off), the
/// arm that proves the compiled path next to a native twin.
#[cfg(test)]
pub(crate) fn force_compiled_only_for_test() -> ForceOnlyGuard {
    let compiled: Vec<Stretch> = Stretch::ALL
        .iter()
        .copied()
        .filter(|stretch| !stretch.is_native())
        .collect();
    force_only_for_test(&compiled)
}

/// Restores the previous prefill-width override on drop.
#[cfg(test)]
#[must_use = "hold the guard for the whole test body"]
pub(crate) struct ForcePrefillGuard {
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for ForcePrefillGuard {
    fn drop(&mut self) {
        FORCE_PREFILL.with(|force| force.set(self.previous));
    }
}

/// Switch the prefill-width stretches on or off for the guard's lifetime.
#[cfg(test)]
pub(crate) fn force_prefill_rows_for_test(mode: bool) -> ForcePrefillGuard {
    let previous = FORCE_PREFILL.with(|force| force.replace(Some(mode)));
    ForcePrefillGuard { previous }
}

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
            .any(|((slot_stretch, _, _), slot)| *slot_stretch == stretch && slot.closure.is_none())
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
    use mlx_sys::{MlxDtype, add, astype, eval, multiply};

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
        tiny_inputs_shaped(rows, [1, 4])
    }

    /// The same four values per row, laid out as `[rows, tail[0], tail[1]]`.
    fn tiny_inputs_shaped(rows: i32, tail: [i32; 2]) -> [MlxArray; 3] {
        assert_eq!(tail[0] * tail[1], 4);
        let len = rows as usize * 4;
        let shape = [rows, tail[0], tail[1]];
        [
            array_f32(&wave(len, 0.37, 1.0, 0.5), &shape),
            array_f32(&wave(len, 0.61, 0.5, 0.25), &shape),
            array_f32(&wave(len, 0.23, 0.75, -0.5), &shape),
        ]
    }

    fn input_refs(inputs: &[MlxArray; 3]) -> [&MlxArray; 3] {
        [&inputs[0], &inputs[1], &inputs[2]]
    }

    /// The eager result of [`tiny_body`].
    fn tiny_body_eager(inputs: &[MlxArray; 3]) -> MlxArray {
        add(&multiply(&inputs[0], &inputs[1], None), &inputs[2], None)
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

        const NONE: [bool; Stretch::COUNT] = [false; Stretch::COUNT];
        const ALL: [bool; Stretch::COUNT] = [true; Stretch::COUNT];
        let norm_only = [
            true, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false,
        ];
        let norm_and_beta = [
            true, false, false, false, false, true, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false,
        ];

        assert_eq!(
            Stretch::ALL.map(Stretch::name),
            [
                "hc_norm",
                "hc_low",
                "hc_mix",
                "hc_gate",
                "hc_write",
                "gdn_beta",
                "gdn_post",
                "gdn_conv_act",
                "gdn_decay_pre",
                "gdn_decay_post",
                "gdn_l2_sum",
                "gdn_q_scale",
                "gdn_k_scale",
                "moe_swiglu",
                "moe_combine",
                "qsa_rope_lo",
                "qsa_rope_hi",
                "qsa_gate",
                "moe_route_tail",
                "gdn_native_decay",
                "gdn_native_qk",
                "hc_native_low_gate",
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
        assert_eq!(
            parse_mask(None, true),
            [true; Stretch::COUNT],
            "shipping default is on for every stretch, native ones included"
        );
        assert_eq!(
            parse_mask(None, false),
            [false; Stretch::COUNT],
            "unit tests default off"
        );
        assert_eq!(
            parse_mask(Some("0"), true),
            [false; Stretch::COUNT],
            "kill switch"
        );
        assert_eq!(
            parse_mask(Some(""), true),
            [false; Stretch::COUNT],
            "empty disables"
        );
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
    fn row_gate_admits_decode_rows_and_prefill_rows_of_compiled_stretches() {
        let wide = PREFILL_FUSE_MIN_ROWS;
        // (native, rows, decode cap, prefill switch, admitted)
        let cases = [
            (false, 0, 1, true, false),
            (false, 1, 1, true, true),
            (false, 2, 1, true, false),
            (false, 8, 8, false, true),
            (false, 9, 8, true, false),
            (false, wide - 1, 8, true, false),
            (false, wide, 1, true, true),
            (false, 2048, 1, true, true),
            (false, wide, 1, false, false),
            (true, 1, 1, true, true),
            (true, wide, 8, true, false),
            (true, 2048, 8, true, false),
        ];
        for (native, rows, decode_rows, prefill_rows, admitted) in cases {
            assert_eq!(
                row_gate(native, rows, decode_rows, prefill_rows),
                admitted,
                "native={native} rows={rows} decode_rows={decode_rows} prefill={prefill_rows}"
            );
        }
    }

    #[test]
    fn prefill_switch_leaves_the_host_default_alone_when_unset() {
        assert_eq!(parse_prefill_switch(None), None);
        for off in ["0", "false", "OFF", " no ", ""] {
            assert_eq!(parse_prefill_switch(Some(off)), Some(false), "{off:?}");
        }
        for on in ["1", "true", "on", "anything"] {
            assert_eq!(parse_prefill_switch(Some(on)), Some(true), "{on:?}");
        }
    }

    #[test]
    fn prefill_rows_run_compiled_stretches_and_skip_native_ones() {
        reset_registry_for_test();
        let _guard = force_for_test(Some(true));
        let apply_at = |rows: i32| {
            let inputs = tiny_inputs(rows);
            apply(
                Stretch::HcNorm,
                rows,
                [0, 1, 4, 4],
                || compile(tiny_body),
                &input_refs(&inputs),
            )
        };
        // Off (the unit-test default): prefill-width rows are skipped.
        assert!(apply_at(PREFILL_FUSE_MIN_ROWS).is_none());

        let _prefill = force_prefill_rows_for_test(true);
        for rows in [PREFILL_FUSE_MIN_ROWS, 301] {
            let inputs = tiny_inputs(rows);
            let fused = apply_at(rows).expect("prefill rows must fuse");
            let eager = tiny_body_eager(&inputs);
            eval(&[&fused, &eager]);
            assert_eq!(fused.data_f32(), eager.data_f32(), "rows={rows}");
            assert!(!native_allowed(Stretch::GdnNativeQk, rows), "rows={rows}");
        }
        // Rows between the decode cap and the prefill floor stay eager, and
        // the native kernels keep their decode rows.
        assert!(apply_at(PREFILL_FUSE_MIN_ROWS - 1).is_none());
        assert!(native_allowed(Stretch::GdnNativeQk, 1));
        assert!(!tombstoned_for_test(Stretch::HcNorm));
    }

    #[test]
    fn a_prefill_closure_is_rebuilt_after_its_traces_run_out() {
        reset_registry_for_test();
        let _guard = force_for_test(Some(true));
        let _prefill = force_prefill_rows_for_test(true);
        let builds = Cell::new(0u32);
        // Every apply must reproduce the eager result, whichever closure
        // generation serves it.
        let apply_shaped = |rows: i32, tail: [i32; 2]| {
            let inputs = tiny_inputs_shaped(rows, tail);
            let fused = apply(
                Stretch::HcNorm,
                rows,
                [0, 1, 4, 4],
                || {
                    builds.set(builds.get() + 1);
                    compile(tiny_body)
                },
                &input_refs(&inputs),
            );
            let Some(fused) = fused else {
                return false;
            };
            let eager = tiny_body_eager(&inputs);
            eval(&[&fused, &eager]);
            assert_eq!(fused.data_f32(), eager.data_f32(), "rows={rows}");
            true
        };
        let apply_at = |rows: i32| apply_shaped(rows, [1, 4]);
        let limit = i32::try_from(TRACES_PER_CLOSURE).expect("small limit");
        // Decode has a closure of its own.
        assert!(apply_at(1));
        assert_eq!(builds.get(), 1);
        for step in 0..limit {
            assert!(apply_at(PREFILL_FUSE_MIN_ROWS + step));
        }
        assert_eq!(
            builds.get(),
            2,
            "one prefill closure serves every shape it may keep"
        );
        // A shape it has already traced is served by the same closure.
        assert!(apply_at(PREFILL_FUSE_MIN_ROWS));
        assert_eq!(builds.get(), 2);
        // One more new shape drops it and builds a fresh one.
        assert!(apply_at(PREFILL_FUSE_MIN_ROWS + limit));
        assert_eq!(builds.get(), 3);
        assert!(apply_at(PREFILL_FUSE_MIN_ROWS + limit));
        assert_eq!(builds.get(), 3);
        // The fresh closure starts its own list, and the decode closure was
        // never touched.
        assert!(apply_at(PREFILL_FUSE_MIN_ROWS));
        assert!(apply_at(1));
        assert_eq!(builds.get(), 3);
        assert!(!tombstoned_for_test(Stretch::HcNorm));

        // Traces are counted by input shape, not by row count: the same row
        // count in another layout is a trace of its own.
        reset_registry_for_test();
        builds.set(0);
        for step in 0..limit / 2 {
            assert!(apply_shaped(PREFILL_FUSE_MIN_ROWS + step, [1, 4]));
            assert!(apply_shaped(PREFILL_FUSE_MIN_ROWS + step, [2, 2]));
        }
        assert_eq!(builds.get(), 1);
        assert!(apply_shaped(PREFILL_FUSE_MIN_ROWS, [4, 1]));
        assert_eq!(builds.get(), 2);
    }

    #[test]
    fn a_prefill_failure_leaves_decode_fused() {
        reset_registry_for_test();
        let _guard = force_for_test(Some(true));
        let _prefill = force_prefill_rows_for_test(true);
        let apply_at = |rows: i32| {
            let inputs = tiny_inputs(rows);
            apply(
                Stretch::HcNorm,
                rows,
                [0, 1, 4, 4],
                || compile(tiny_body),
                &input_refs(&inputs),
            )
            .is_some()
        };
        assert!(apply_at(1));
        fail_next_apply_for_test(Stretch::HcNorm);
        assert!(!apply_at(PREFILL_FUSE_MIN_ROWS));
        assert!(tombstoned_for_test(Stretch::HcNorm));
        // Prefill stays on the eager path; decode keeps its closure.
        assert!(!apply_at(PREFILL_FUSE_MIN_ROWS));
        assert!(!apply_at(PREFILL_FUSE_MIN_ROWS + 1));
        assert!(apply_at(1));

        // A first-use evaluation failure at prefill width does the same.
        reset_registry_for_test();
        assert!(apply_at(1));
        fail_next_eval_for_test(Stretch::HcNorm);
        assert!(!apply_at(PREFILL_FUSE_MIN_ROWS));
        assert!(apply_at(1));
    }

    /// Every input shape a prefill closure traces is evaluated once before
    /// its output is trusted, not only the first one.
    #[test]
    fn a_new_prefill_shape_is_validated_on_a_closure_already_in_use() {
        reset_registry_for_test();
        let _guard = force_for_test(Some(true));
        let _prefill = force_prefill_rows_for_test(true);
        let apply_at = |rows: i32| {
            let inputs = tiny_inputs(rows);
            apply(
                Stretch::HcNorm,
                rows,
                [0, 1, 4, 4],
                || compile(tiny_body),
                &input_refs(&inputs),
            )
            .is_some()
        };
        assert!(apply_at(PREFILL_FUSE_MIN_ROWS));
        // A shape already traced is not evaluated again: the injected
        // failure stays armed.
        fail_next_eval_for_test(Stretch::HcNorm);
        assert!(apply_at(PREFILL_FUSE_MIN_ROWS));
        assert!(!tombstoned_for_test(Stretch::HcNorm));
        // The next new shape meets it, falls back and tombstones the slot.
        assert!(!apply_at(PREFILL_FUSE_MIN_ROWS + 1));
        assert!(tombstoned_for_test(Stretch::HcNorm));
        assert!(!apply_at(PREFILL_FUSE_MIN_ROWS));

        // The closure built after the traces ran out is validated too.
        reset_registry_for_test();
        let limit = i32::try_from(TRACES_PER_CLOSURE).expect("small limit");
        for step in 0..limit {
            assert!(apply_at(PREFILL_FUSE_MIN_ROWS + step));
        }
        fail_next_eval_for_test(Stretch::HcNorm);
        assert!(!apply_at(PREFILL_FUSE_MIN_ROWS + limit));
        assert!(tombstoned_for_test(Stretch::HcNorm));
    }

    /// MLX traces a closure again for another input dtype at the same shape,
    /// so that is a new trace as well.
    #[test]
    fn a_new_input_dtype_is_validated_like_a_new_shape() {
        reset_registry_for_test();
        let _guard = force_for_test(Some(true));
        let _prefill = force_prefill_rows_for_test(true);
        let rows = PREFILL_FUSE_MIN_ROWS;
        let apply_as = |dtype: MlxDtype| {
            let inputs = tiny_inputs(rows).map(|input| astype(&input, dtype, None));
            apply(
                Stretch::HcNorm,
                rows,
                [0, 1, 4, 4],
                || compile(tiny_body),
                &input_refs(&inputs),
            )
            .is_some()
        };
        assert!(apply_as(MlxDtype::Float32));
        fail_next_eval_for_test(Stretch::HcNorm);
        assert!(apply_as(MlxDtype::Float32));
        assert!(!apply_as(MlxDtype::Float16));
        assert!(tombstoned_for_test(Stretch::HcNorm));
    }

    /// A verifier window (rows 2 to 8, opt-in) is a new trace of the decode
    /// closure and is validated; the one-row step is validated once, with
    /// the closure.
    #[test]
    fn a_multi_row_decode_window_is_validated_on_first_use() {
        reset_registry_for_test();
        let _guard = force_for_test(Some(true));
        let apply_at = |rows: i32| {
            let inputs = tiny_inputs(rows);
            apply(
                Stretch::HcNorm,
                rows,
                [0, 1, 4, 4],
                || compile(tiny_body),
                &input_refs(&inputs),
            )
            .is_some()
        };
        assert!(apply_at(1));
        fail_next_eval_for_test(Stretch::HcNorm);
        assert!(apply_at(1));
        assert!(!tombstoned_for_test(Stretch::HcNorm));
        assert!(!apply_at(4));
        assert!(tombstoned_for_test(Stretch::HcNorm));
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
