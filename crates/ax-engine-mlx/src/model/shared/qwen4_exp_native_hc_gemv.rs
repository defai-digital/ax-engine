//! Native hyper-connection read: two GEMV kernels with fused epilogues.
//!
//! One hyper-connection read used to run six dispatches: the `read_down` and
//! `write_inject` GEMVs, the `HcLow` and `HcGate` elementwise stretches, the
//! `read_up` GEMV and the `HcMix` stream mean. The two kernels here replace all
//! six. They reimplement MLX's `gemv` thread layout and accumulation order (the
//! "tall K" and "short K" configurations) and fuse the elementwise work into
//! the projection epilogue, so they read the stored `[out, in]` weights
//! directly and never materialise a merged weight or an `up` row.
//!
//! - [`down_gate`] (`ax_hc_down_gate_v1`, MLX "tall K"): the rows of
//!   `[read_down; write_inject]` from the two weight buffers, followed by
//!   `silu(down / C)` for the low-rank rows and `sigmoid(logits / C) * scale`
//!   for the gate rows.
//! - [`up_mix`] (`ax_hc_up_mix_v1`, MLX "short K"): the `read_up` GEMV with one
//!   simdgroup per hidden column, followed by `sigmoid(up) * normalized` per
//!   stream and the four-stream mean, entirely in lane 0.
//!
//! Both epilogues are the eager arithmetic
//! ([`silu_projection_dtype`], [`sigmoid_projection_dtype`],
//! [`stream_mean_with`]); the kernels only move it next to the accumulation.
//!
//! Fail closed at every step. The wrappers refuse any dtype, shape or MLX
//! `gemv` configuration they were not written for, the kernels are built with
//! [`MlxMetalKernel::try_new`], and a per-kernel monotone verdict turns a build
//! failure, a failed self-check or a rejected launch into "keep the eager path"
//! for the rest of the process. A caller must check the stretch kill switch
//! before touching any of it.

use mlx_sys::ops::cached_scalar;
use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, astype, divide,
    matmul, multiply, reshape, transpose, view,
};
use std::sync::OnceLock;

use super::qwen4_exp_residual::{
    sigmoid_projection_dtype, silu_projection_dtype, stream_mean_with,
};

const DOWN_GATE_NAME: &str = "ax_hc_down_gate_v1";
const UP_MIX_NAME: &str = "ax_hc_up_mix_v1";

/// Threads per threadgroup: eight simdgroups, like MLX's `gemv`.
const THREADS_PER_GROUP: i32 = 256;
/// Simdgroups per threadgroup.
const SIMDGROUPS: i32 = 8;
/// Lanes per simdgroup.
const LANES: i32 = 32;
/// Output rows one [`down_gate`] threadgroup owns.
const ROWS_PER_GROUP: i32 = 4;
/// Positions one lane owns per iteration: the four `tn` items.
const ITEMS_PER_LANE: i32 = 4;
/// K positions one tall-K iteration covers: `8 * 32 * 4`.
const TALL_K_STRIDE: i32 = SIMDGROUPS * LANES * ITEMS_PER_LANE;
/// K positions one short-K iteration covers: `32 * 4`.
const SHORT_K_STRIDE: i32 = LANES * ITEMS_PER_LANE;
/// Streams the mix epilogue reduces. The association is written for exactly
/// four, so callers must hold four streams.
pub(super) const STREAM_COUNT: i32 = 4;

/// Width of the self-check projection: a multiple of the tall-K stride.
const SELF_CHECK_WIDTH: i32 = 1024;
/// Rows the self-check puts in the low role.
const SELF_CHECK_ROWS: usize = 256;
/// Rows the self-check puts in the gate role (the tail of the sample).
const SELF_CHECK_GATES: usize = 4;
/// Streams the self-check divides by; any positive value works, because the
/// kernel takes it as an input.
const SELF_CHECK_STREAMS: f32 = 4.0;
/// Write-gate multiplier of the self-check.
const SELF_CHECK_SCALE: f32 = 2.0;
/// `1.0` as raw bf16 bits.
const BF16_ONE: u16 = 0x3F80;
/// Hidden width of the self-check mix case.
const SELF_CHECK_HIDDEN: usize = 32;
/// Low width of the self-check mix case: one short-K stride.
const SELF_CHECK_LOW: i32 = SHORT_K_STRIDE;

/// Self-check verdict of one kernel.
const UNKNOWN: u8 = 0;
const PASSED: u8 = 1;
const FAILED: u8 = 2;

/// The two kernels, each with its own build cell and verdict.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kernel {
    DownGate,
    UpMix,
}

impl Kernel {
    fn index(self) -> usize {
        self as usize
    }
}

/// Number of kernels, i.e. the width of the verdict arrays.
const KERNEL_COUNT: usize = 2;

static DOWN_GATE: OnceLock<Option<MlxMetalKernel>> = OnceLock::new();
static UP_MIX: OnceLock<Option<MlxMetalKernel>> = OnceLock::new();

#[cfg(not(test))]
static VERDICTS: [std::sync::atomic::AtomicU8; KERNEL_COUNT] = [
    std::sync::atomic::AtomicU8::new(UNKNOWN),
    std::sync::atomic::AtomicU8::new(UNKNOWN),
];

#[cfg(test)]
thread_local! {
    // Per thread, so one test's injected failure cannot disable a kernel for a
    // test running on another thread. Production keeps the process-wide
    // atomics above.
    static VERDICTS: [std::sync::atomic::AtomicU8; KERNEL_COUNT] = const {
        [
            std::sync::atomic::AtomicU8::new(UNKNOWN),
            std::sync::atomic::AtomicU8::new(UNKNOWN),
        ]
    };
}

#[cfg(not(test))]
fn verdict(index: usize) -> u8 {
    VERDICTS[index].load(std::sync::atomic::Ordering::Acquire)
}

#[cfg(test)]
fn verdict(index: usize) -> u8 {
    VERDICTS.with(|verdicts| verdicts[index].load(std::sync::atomic::Ordering::Acquire))
}

#[cfg(not(test))]
fn compare_exchange_verdict(index: usize, current: u8, new: u8) -> std::result::Result<u8, u8> {
    VERDICTS[index].compare_exchange(
        current,
        new,
        std::sync::atomic::Ordering::AcqRel,
        std::sync::atomic::Ordering::Acquire,
    )
}

#[cfg(test)]
fn compare_exchange_verdict(index: usize, current: u8, new: u8) -> std::result::Result<u8, u8> {
    VERDICTS.with(|verdicts| {
        verdicts[index].compare_exchange(
            current,
            new,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
    })
}

/// Record a verdict with a monotone, single-shot transition.
///
/// `UNKNOWN` yields to the first writer, `PASSED` can still be revoked by a
/// rejected launch, and `FAILED` is terminal. Two threads that both saw
/// `UNKNOWN` therefore cannot end with the loser overwriting a failure with a
/// pass: the first exchange wins and every later one is refused. Returns the
/// verdict in force afterwards.
fn record_verdict(index: usize, value: u8) -> u8 {
    let mut current = verdict(index);
    // Only two transitions exist (`UNKNOWN` to a verdict, then `PASSED` to
    // `FAILED`), so a losing exchange always leaves a state that is either
    // final or retried against the freshly observed value.
    for _ in 0..3 {
        if current == FAILED || current == value {
            return current;
        }
        match compare_exchange_verdict(index, current, value) {
            Ok(_) => return value,
            Err(observed) => current = observed,
        }
    }
    verdict(index)
}

/// Whether the kernels have a body for `dtype`.
///
/// Cheap and side-effect free, so a caller can reject a dtype before anything
/// is compiled, launched or built. Both kernels are bf16 only: the epilogues
/// are written for MLX's `bfloat16_t` rounding.
pub(super) fn supports_dtype(dtype: MlxDtype) -> bool {
    dtype == MlxDtype::Bfloat16
}

/// Whether `ax_hc_down_gate_v1` is known to reproduce the eager path.
///
/// The first call runs the self-check; the verdict is monotone, so a failure
/// disables the kernel permanently even when two threads race the first check.
pub(super) fn down_gate_supported() -> bool {
    supported(Kernel::DownGate)
}

/// Whether `ax_hc_up_mix_v1` is known to reproduce the eager path.
pub(super) fn up_mix_supported() -> bool {
    supported(Kernel::UpMix)
}

fn supported(kind: Kernel) -> bool {
    match verdict(kind.index()) {
        PASSED => true,
        FAILED => false,
        _ => {
            let passed = self_check_or_injected_failure(kind);
            record_verdict(kind.index(), if passed { PASSED } else { FAILED }) == PASSED
        }
    }
}

/// Whether MLX picks its tall-K `gemv` for an `N x K` projection.
///
/// `N` is the output row count of one projection, so [`down_gate`] asks this
/// for the low-rank rows and for the gate rows separately: MLX selects the
/// kernel per projection, and both must land on the configuration this module
/// reproduces.
pub(super) fn mlx_gemv_tall_k(n: i32, k: i32) -> bool {
    // `k <= i32::MAX / 4` keeps MLX's signed 32-bit offset arithmetic
    // (`mat_offset + bn`, four positions per lane) from overflowing.
    n > 0 && k > 64 && k <= i32::MAX / 4 && i64::from(k) >= 16 * i64::from(n)
}

/// Whether MLX picks its short-K `gemv` for an `N x K` projection.
pub(super) fn mlx_gemv_short_k(n: i32, k: i32) -> bool {
    // MLX evaluates `16 * n` in signed 32-bit arithmetic, so larger row counts
    // would not select the configuration this module reproduces.
    (4096..=i32::MAX / 16).contains(&n) && k > 64 && i64::from(k) < 16 * i64::from(n)
}

/// Launch `ax_hc_down_gate_v1`: the low-rank activation and the write gate of
/// one normalized hyper-connection row.
///
/// `w_down` is the dense `[R, K]` `read_down`, `w_inject` the dense `[S, K]`
/// `write_inject`, `x` the normalized row (`[K]` or `[1, 1, K]`), and `streams`
/// and `scale` are `[1]` bf16 scalars. Returns `low` and `gate` in the shape
/// convention of `x` (`[R]`/`[S]` or `[1, 1, R]`/`[1, 1, S]`). Quantized
/// weights are refused: the kernel reads the stored row-major matrix.
///
/// Layout contract: both weights are the stored, row-contiguous `[out, in]`
/// arrays (the layout MLX's non-transposed-RHS `gemv` path expects when the
/// reference multiplies by their transpose). A column-major or strided weight
/// would make MLX pick a different `gemv`, and this module cannot see strides,
/// so the caller proves the real weights with the per-module first-use check
/// before relying on the result.
pub(super) fn down_gate(
    w_down: &MlxArray,
    w_inject: &MlxArray,
    x: &MlxArray,
    streams: &MlxArray,
    scale: &MlxArray,
) -> Option<(MlxArray, MlxArray)> {
    let geometry = down_gate_geometry(w_down, w_inject, x, streams, scale)?;
    // MLX selects the tall-K kernel per projection, so both must qualify.
    if !mlx_gemv_tall_k(geometry.rows, geometry.width)
        || !mlx_gemv_tall_k(geometry.gates, geometry.width)
    {
        return None;
    }
    down_gate_launch(w_down, w_inject, x, streams, scale, geometry)
}

/// Launch `ax_hc_up_mix_v1`: the `read_up` projection, the sigmoid gate product
/// and the stream mean of one normalized hyper-connection row.
///
/// `w_up` is the dense `[C * H, L]` `read_up` with exactly [`STREAM_COUNT`]
/// streams, `low` the `[L]` (or `[1, 1, L]`) low-rank activation, `normalized`
/// the `[C * H]` (or `[1, 1, C * H]`) normalized row, and `streams_f32` the
/// `[1]` f32 stream count. Returns the `[H]` (or `[1, 1, H]`) mean. The layout
/// contract of [`down_gate`] applies to `w_up`.
pub(super) fn up_mix(
    w_up: &MlxArray,
    low: &MlxArray,
    normalized: &MlxArray,
    streams_f32: &MlxArray,
) -> Option<MlxArray> {
    let geometry = up_mix_geometry(w_up, low, normalized, streams_f32)?;
    if !mlx_gemv_short_k(geometry.rows, geometry.low_width) {
        return None;
    }
    up_mix_launch(w_up, low, normalized, streams_f32, geometry)
}

/// Validated [`down_gate`] geometry.
#[derive(Clone, Copy)]
struct DownGateGeometry {
    rows: i32,
    gates: i32,
    width: i32,
    /// Whether the caller spelled the row as `[1, 1, width]`.
    packed: bool,
}

/// Validated [`up_mix`] geometry.
#[derive(Clone, Copy)]
struct UpMixGeometry {
    rows: i32,
    hidden: i32,
    low_width: i32,
    packed: bool,
}

/// Whether `array` is a single flat row of `width` elements: `[width]` or
/// `[1, 1, width]`. The boolean result is the packed spelling, so the outputs
/// can use the caller's rank convention without a reshape.
fn row_is_flat(array: &MlxArray, width: i32) -> Option<bool> {
    match array.shape().as_slice() {
        [only] if *only == width => Some(false),
        [1, 1, only] if *only == width => Some(true),
        _ => None,
    }
}

fn down_gate_geometry(
    w_down: &MlxArray,
    w_inject: &MlxArray,
    x: &MlxArray,
    streams: &MlxArray,
    scale: &MlxArray,
) -> Option<DownGateGeometry> {
    for array in [w_down, w_inject, x, streams, scale] {
        if !supports_dtype(array.dtype()) {
            return None;
        }
    }
    if streams.shape() != [1] || scale.shape() != [1] {
        return None;
    }
    let down_shape = w_down.shape();
    let inject_shape = w_inject.shape();
    if down_shape.len() != 2 || inject_shape.len() != 2 || down_shape[1] != inject_shape[1] {
        return None;
    }
    let (rows, width) = (down_shape[0], down_shape[1]);
    let gates = inject_shape[0];
    if rows <= 0
        || gates <= 0
        || width <= 0
        || rows % ROWS_PER_GROUP != 0
        || gates % ROWS_PER_GROUP != 0
        || width % TALL_K_STRIDE != 0
    {
        return None;
    }
    let packed = row_is_flat(x, width)?;
    Some(DownGateGeometry {
        rows,
        gates,
        width,
        packed,
    })
}

fn up_mix_geometry(
    w_up: &MlxArray,
    low: &MlxArray,
    normalized: &MlxArray,
    streams_f32: &MlxArray,
) -> Option<UpMixGeometry> {
    if !supports_dtype(w_up.dtype())
        || !supports_dtype(low.dtype())
        || !supports_dtype(normalized.dtype())
    {
        return None;
    }
    if streams_f32.dtype() != MlxDtype::Float32 || streams_f32.shape() != [1] {
        return None;
    }
    let shape = w_up.shape();
    if shape.len() != 2 {
        return None;
    }
    let (rows, low_width) = (shape[0], shape[1]);
    // `rows` is `C * H`, and the epilogue association is written for exactly
    // four streams, so anything else must never reach it.
    if rows <= 0
        || low_width <= 0
        || rows % STREAM_COUNT != 0
        || rows % ROWS_PER_GROUP != 0
        || low_width % ITEMS_PER_LANE != 0
    {
        return None;
    }
    let packed = row_is_flat(normalized, rows)?;
    if row_is_flat(low, low_width)? != packed {
        return None;
    }
    Some(UpMixGeometry {
        rows,
        hidden: rows / STREAM_COUNT,
        low_width,
        packed,
    })
}

/// One raw `ax_hc_down_gate_v1` launch, without the configuration predicate.
///
/// The public wrapper adds the MLX `gemv` configuration check; the epilogue
/// tests call this directly because their shapes only put a single non-zero
/// product in each row.
fn down_gate_launch(
    w_down: &MlxArray,
    w_inject: &MlxArray,
    x: &MlxArray,
    streams: &MlxArray,
    scale: &MlxArray,
    geometry: DownGateGeometry,
) -> Option<(MlxArray, MlxArray)> {
    // A rejected launch or failed check is terminal for the process, also for
    // a caller that reaches the launch without asking `*_supported` first.
    if verdict(Kernel::DownGate.index()) == FAILED {
        return None;
    }
    #[cfg(test)]
    LAUNCHES.with(|launches| launches.set(launches.get() + 1));
    let kernel = kernel(Kernel::DownGate)?;
    #[cfg(test)]
    if take_injected_launch_failure() {
        return reject_launch(Kernel::DownGate);
    }
    let total = geometry.rows.checked_add(geometry.gates)?;
    let grid = total.checked_mul(THREADS_PER_GROUP / ROWS_PER_GROUP)?;
    let spec = |count: i32| KernelOutputSpec {
        shape: if geometry.packed {
            vec![1, 1, count]
        } else {
            vec![count]
        },
        dtype: MlxDtype::Bfloat16,
    };
    let Ok(mut outputs) = kernel.try_apply_with_template(
        &[w_down, w_inject, x, streams, scale],
        &[spec(geometry.rows), spec(geometry.gates)],
        &[
            KernelTemplateArg::Dtype {
                name: "T",
                dtype: MlxDtype::Bfloat16,
            },
            KernelTemplateArg::Int {
                name: "K",
                value: geometry.width,
            },
            KernelTemplateArg::Int {
                name: "R",
                value: geometry.rows,
            },
        ],
        (grid, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
        None,
    ) else {
        return reject_launch(Kernel::DownGate);
    };
    if outputs.len() != 2 {
        return reject_launch(Kernel::DownGate);
    }
    let gate = outputs.pop()?;
    let low = outputs.pop()?;
    Some((low, gate))
}

/// A launch the kernel rejected: revoke the verdict so no later read retries it
/// and let the caller fall back to the eager ops.
fn reject_launch<T>(kind: Kernel) -> Option<T> {
    record_verdict(kind.index(), FAILED);
    tracing::warn!(
        ?kind,
        "Flash Next hyper-connection GEMV launch was rejected; using eager ops"
    );
    None
}

/// One raw `ax_hc_up_mix_v1` launch, without the configuration predicate.
fn up_mix_launch(
    w_up: &MlxArray,
    low: &MlxArray,
    normalized: &MlxArray,
    streams_f32: &MlxArray,
    geometry: UpMixGeometry,
) -> Option<MlxArray> {
    if verdict(Kernel::UpMix.index()) == FAILED {
        return None;
    }
    #[cfg(test)]
    LAUNCHES.with(|launches| launches.set(launches.get() + 1));
    let kernel = kernel(Kernel::UpMix)?;
    #[cfg(test)]
    if take_injected_launch_failure() {
        return reject_launch(Kernel::UpMix);
    }
    // One simdgroup per hidden column, so the grid is in threads.
    let grid = geometry.hidden.checked_mul(LANES)?;
    let spec = KernelOutputSpec {
        shape: if geometry.packed {
            vec![1, 1, geometry.hidden]
        } else {
            vec![geometry.hidden]
        },
        dtype: MlxDtype::Bfloat16,
    };
    let Ok(outputs) = kernel.try_apply_with_template(
        &[w_up, low, normalized, streams_f32],
        &[spec],
        &[
            KernelTemplateArg::Dtype {
                name: "T",
                dtype: MlxDtype::Bfloat16,
            },
            KernelTemplateArg::Int {
                name: "H",
                value: geometry.hidden,
            },
            KernelTemplateArg::Int {
                name: "L",
                value: geometry.low_width,
            },
        ],
        (grid, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
        None,
    ) else {
        return reject_launch(Kernel::UpMix);
    };
    match outputs.into_iter().next() {
        Some(mean) => Some(mean),
        None => reject_launch(Kernel::UpMix),
    }
}

/// The built kernel of `kind`, or `None` after a failed build.
fn kernel(kind: Kernel) -> Option<&'static MlxMetalKernel> {
    let cell = match kind {
        Kernel::DownGate => &DOWN_GATE,
        Kernel::UpMix => &UP_MIX,
    };
    cell.get_or_init(|| build_kernel(kind)).as_ref()
}

fn build_kernel(kind: Kernel) -> Option<MlxMetalKernel> {
    #[cfg(test)]
    KERNEL_BUILDS.with(|builds| builds.set(builds.get() + 1));
    let (name, inputs, outputs, source) = match kind {
        Kernel::DownGate => (
            DOWN_GATE_NAME,
            &["w_down", "w_inject", "x", "streams", "scale"][..],
            &["low", "gate"][..],
            DOWN_GATE_SOURCE,
        ),
        Kernel::UpMix => (
            UP_MIX_NAME,
            &["w_up", "low", "normalized", "streams_f32"][..],
            &["mean"][..],
            UP_MIX_SOURCE,
        ),
    };
    // `ensure_row_contiguous` is what lets the kernels assume a plain
    // row-major `[out, in]` weight; MLX would copy anything else.
    match MlxMetalKernel::try_new(name, inputs, outputs, source, SIGMOID_HEADER, true) {
        Ok(kernel) => Some(kernel),
        Err(error) => {
            tracing::warn!(
                %error,
                ?kind,
                "Flash Next hyper-connection GEMV kernel failed to build; using eager ops"
            );
            None
        }
    }
}

/// Self-check with the test-only failure switch in front of it.
fn self_check_or_injected_failure(kind: Kernel) -> bool {
    #[cfg(test)]
    SELF_CHECK_RUNS.with(|runs| runs.set(runs.get() + 1));
    #[cfg(test)]
    let injected = take_injected_self_check_failure();
    #[cfg(not(test))]
    let injected = false;
    if injected {
        tracing::warn!(
            ?kind,
            "Flash Next hyper-connection GEMV self-check was injected to fail; using eager ops"
        );
        return false;
    }
    // An eager-op panic (a rejected graph) must disable the kernel rather than
    // escape from a decode step, exactly like a bit mismatch.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self_check(kind))).unwrap_or(false)
}

fn self_check(kind: Kernel) -> bool {
    match kind {
        Kernel::DownGate => self_check_down_gate(),
        Kernel::UpMix => self_check_up_mix(),
    }
}

/// Compare `ax_hc_down_gate_v1` with the MLX projection plus the eager
/// stretches on a sample of bf16 patterns, in both roles.
///
/// The sample row is `[1, 0, ...]`, so every row's GEMV is one exact product
/// and the comparison isolates the epilogue and the row plumbing rather than
/// the reduction order (the module-level proof covers that).
fn self_check_down_gate() -> bool {
    let patterns = self_check_patterns(SELF_CHECK_ROWS + SELF_CHECK_GATES);
    let (Some(head), Some(tail)) = (
        patterns.get(..SELF_CHECK_ROWS),
        patterns.get(SELF_CHECK_ROWS..),
    ) else {
        return false;
    };
    check_down_gate_case(head, tail) && check_down_gate_case(tail, head)
}

fn check_down_gate_case(down_patterns: &[u16], gate_patterns: &[u16]) -> bool {
    let width = SELF_CHECK_WIDTH as usize;
    let w_down = pattern_matrix(down_patterns, width);
    let w_inject = pattern_matrix(gate_patterns, width);
    let mut x_bits = vec![0u16; width];
    if let Some(slot) = x_bits.first_mut() {
        *slot = BF16_ONE;
    }
    let x = pattern_row(&x_bits);
    let streams = cached_scalar(SELF_CHECK_STREAMS, MlxDtype::Bfloat16);
    let scale = cached_scalar(SELF_CHECK_SCALE, MlxDtype::Bfloat16);
    let geometry = down_gate_geometry(&w_down, &w_inject, &x, &streams, &scale);
    let Some(geometry) = geometry else {
        return false;
    };
    let Some((low, gate)) = down_gate_launch(&w_down, &w_inject, &x, &streams, &scale, geometry)
    else {
        return false;
    };

    let row = reshape(&x, &[1, 1, width as i32], None);
    let down = matmul(&row, &transpose(&w_down, &[1, 0], None), None);
    let low_reference = silu_projection_dtype(&divide(&down, &streams, None));
    let logits = matmul(&row, &transpose(&w_inject, &[1, 0], None), None);
    let gate_reference = multiply(
        &sigmoid_projection_dtype(&divide(&logits, &streams, None)),
        &scale,
        None,
    );
    let (Some(low_actual), Some(low_expected), Some(gate_actual), Some(gate_expected)) = (
        raw_bits(&low),
        raw_bits(&low_reference),
        raw_bits(&gate),
        raw_bits(&gate_reference),
    ) else {
        return false;
    };
    low_actual == low_expected && gate_actual == gate_expected
}

/// Compare `ax_hc_up_mix_v1` with the MLX projection plus the eager mix on a
/// sample of bf16 patterns.
fn self_check_up_mix() -> bool {
    let hidden = SELF_CHECK_HIDDEN;
    let low_width = SELF_CHECK_LOW as usize;
    let rows = STREAM_COUNT as usize * hidden;
    let w_up = pattern_matrix(&self_check_patterns(rows), low_width);

    let mut low_bits = vec![0u16; low_width];
    if let Some(slot) = low_bits.first_mut() {
        *slot = BF16_ONE;
    }
    let low = pattern_row(&low_bits);
    let normalized = pattern_row(&self_check_normalized(rows));
    let streams_f32 = cached_scalar(STREAM_COUNT as f32, MlxDtype::Float32);

    let Some(geometry) = up_mix_geometry(&w_up, &low, &normalized, &streams_f32) else {
        return false;
    };
    let Some(mean) = up_mix_launch(&w_up, &low, &normalized, &streams_f32, geometry) else {
        return false;
    };

    let up = matmul(
        &reshape(&low, &[1, 1, low_width as i32], None),
        &transpose(&w_up, &[1, 0], None),
        None,
    );
    let stream_shape = [1, 1, STREAM_COUNT, hidden as i32];
    let gated = multiply(
        &reshape(&sigmoid_projection_dtype(&up), &stream_shape, None),
        &reshape(&normalized, &stream_shape, None),
        None,
    );
    let mean_reference = stream_mean_with(&gated, &streams_f32);
    let (Some(actual), Some(expected)) = (raw_bits(&mean), raw_bits(&mean_reference)) else {
        return false;
    };
    actual == expected
}

/// Deterministic bf16 patterns covering every value class.
fn self_check_patterns(count: usize) -> Vec<u16> {
    const SPECIAL: [u16; 16] = [
        0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x0001, 0x8001, 0x0080,
        0x7F7F, 0xFF7F, 0x3F00, 0xBF00, 0x7F81,
    ];
    let mut state = 0x51ED_2701_9E37_79B9u64;
    (0..count)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match SPECIAL.get(index) {
                Some(pattern) => *pattern,
                None => (state >> 24) as u16,
            }
        })
        .collect()
}

/// A deterministic sample of finite activations plus the special classes, for
/// the mix epilogue's gate product and mean.
fn self_check_normalized(count: usize) -> Vec<u16> {
    const SPECIAL: [u16; 8] = [
        0x0000, 0x8000, 0x0001, 0x3F80, 0xBF80, 0x7FC0, 0x4B00, 0xCB00,
    ];
    let mut state = 0x0BAD_C0DE_5EED_1234u64;
    (0..count)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match SPECIAL.get(index) {
                Some(pattern) => *pattern,
                None => {
                    // Finite values across a few binades: sign, exponent and a
                    // pseudo-random mantissa.
                    let sign = (((state >> 33) & 1) as u16) << 15;
                    let exponent = 0x7B + ((state >> 17) % 8) as u16;
                    let mantissa = ((state >> 3) & 0x7F) as u16;
                    sign | (exponent << 7) | mantissa
                }
            }
        })
        .collect()
}

/// A `[patterns.len(), width]` bf16 matrix whose first column holds `patterns`
/// verbatim and whose other entries are zero.
fn pattern_matrix(patterns: &[u16], width: usize) -> MlxArray {
    let Some(len) = patterns.len().checked_mul(width) else {
        return pattern_row(&[]);
    };
    let mut bits = vec![0u16; len];
    for (row, pattern) in patterns.iter().enumerate() {
        if let Some(slot) = bits.get_mut(row * width) {
            *slot = *pattern;
        }
    }
    pattern_matrix_from_bits(&bits, patterns.len(), width)
}

fn pattern_matrix_from_bits(bits: &[u16], rows: usize, width: usize) -> MlxArray {
    let raw = MlxArray::from_raw_data(
        bits.as_ptr().cast(),
        std::mem::size_of_val(bits),
        &[rows as i32, width as i32],
        MlxDtype::Uint16,
    );
    let _ = mlx_sys::try_eval(&[&raw]);
    view(&raw, MlxDtype::Bfloat16, None)
}

/// A bf16 row holding `bits` verbatim, including NaN payloads and subnormals.
fn pattern_row(bits: &[u16]) -> MlxArray {
    let raw = MlxArray::from_raw_data(
        bits.as_ptr().cast(),
        std::mem::size_of_val(bits),
        &[bits.len() as i32],
        MlxDtype::Uint16,
    );
    let _ = mlx_sys::try_eval(&[&raw]);
    view(&raw, MlxDtype::Bfloat16, None)
}

/// Raw bits of an evaluated float array, for bit-exact comparisons.
///
/// Reinterprets the buffer (`view`) instead of converting, so NaN payloads,
/// signed zeros and subnormals survive unchanged. Returns `None` when the array
/// cannot be evaluated or has no supported width.
pub(super) fn raw_bits(array: &MlxArray) -> Option<Vec<u32>> {
    mlx_sys::try_eval(&[array]).ok()?;
    let bits = match array.dtype() {
        MlxDtype::Float32 => view(array, MlxDtype::Uint32, None),
        MlxDtype::Float16 | MlxDtype::Bfloat16 => {
            astype(&view(array, MlxDtype::Uint16, None), MlxDtype::Uint32, None)
        }
        _ => return None,
    };
    mlx_sys::try_eval(&[&bits]).ok()?;
    Some(bits.data_u32().to_vec())
}

/// Raw fixed-width float array from host `f32` data at the given dtype.
pub(super) fn array_from_f32(data: &[f32], shape: &[i32], dtype: MlxDtype) -> MlxArray {
    astype(
        &MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        ),
        dtype,
        None,
    )
}

/// MLX's sigmoid, spelled so the JIT keeps the precise exponential.
///
/// The source bodies are kernel function bodies, so the helper has to live in
/// the header MLX prepends to the generated signature.
const SIGMOID_HEADER: &str = r#"
inline float ax_sigmoid(float x) {
  const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
  return (x < 0.0f) ? y : 1.0f - y;
}
"#;

/// MLX's "tall K" `gemv`, four rows per threadgroup, with the down/gate
/// epilogue in the threadgroup leader.
///
/// Lane `l` of simdgroup `g` owns positions `(32 * g + l) * 4 + 1024 * i +
/// {0..3}`, so one iteration covers a 1024-position stride and the eight
/// simdgroups split the row between them. Every thread accumulates all four
/// rows in order, reduces with MLX's `simd_shuffle_down` tree, and the leader
/// adds the eight simdgroup partials in order before the epilogue.
const DOWN_GATE_SOURCE: &str = r#"
#pragma clang fp contract(off)
const int first = static_cast<int>(threadgroup_position_in_grid.x) * 4;
const int tid = static_cast<int>(thread_position_in_threadgroup.x);
const int simdgroup = tid >> 5;
const int lane = tid & 31;
const int lane_base = (simdgroup * 32 + lane) * 4;
float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
for (int k = 0; k < K; k += 1024) {
  const int pos = k + lane_base;
  const T xs[4] = {x[pos], x[pos + 1], x[pos + 2], x[pos + 3]};
  for (int tm = 0; tm < 4; ++tm) {
    const int row = first + tm;
    // R is a multiple of four, so the four rows of a threadgroup always come
    // from the same weight buffer.
    const device T* w = (row < R) ? (w_down + static_cast<long>(row) * K)
                                  : (w_inject + static_cast<long>(row - R) * K);
    for (int tn = 0; tn < 4; ++tn) {
      acc[tm] += static_cast<float>(w[pos + tn]) * static_cast<float>(xs[tn]);
    }
  }
}
for (int offset = 16; offset > 0; offset >>= 1) {
  for (int tm = 0; tm < 4; ++tm) {
    acc[tm] += simd_shuffle_down(acc[tm], static_cast<ushort>(offset));
  }
}
threadgroup float partials[8][4];
if (lane == 0) {
  for (int tm = 0; tm < 4; ++tm) {
    partials[simdgroup][tm] = acc[tm];
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simdgroup != 0 || lane != 0) {
  return;
}
for (int group = 1; group < 8; ++group) {
  for (int tm = 0; tm < 4; ++tm) {
    acc[tm] += partials[group][tm];
  }
}
for (int tm = 0; tm < 4; ++tm) {
  const int row = first + tm;
  const T q = static_cast<T>(acc[tm]) / streams[0];
  if (row < R) {
    const float f = static_cast<float>(q);
    low[row] = static_cast<T>(f * ax_sigmoid(f));
  } else {
    const T value = static_cast<T>(ax_sigmoid(static_cast<float>(q)));
    gate[row - R] = value * scale[0];
  }
}
"#;

/// MLX's "short K" `gemv`, one simdgroup per hidden column, with the mix
/// epilogue in lane zero.
///
/// Lane `l` owns positions `4 * l + 128 * i + {0..3}` for the full iterations
/// and then one leftover block whose out-of-range positions load as zero and
/// still contribute their zero products. The four stream rows of the column are
/// the simdgroup's four accumulators, so the reduction tree leaves every
/// stream's dot product in lane zero and no cross-simdgroup step is needed.
const UP_MIX_SOURCE: &str = r#"
#pragma clang fp contract(off)
const int column = static_cast<int>(thread_position_in_grid.x) >> 5;
if (column >= H) {
  return;
}
const int lane = static_cast<int>(thread_position_in_grid.x) & 31;
const int lane_base = lane * 4;
float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
const int full = L / 128;
for (int i = 0; i < full; ++i) {
  const int pos = i * 128 + lane_base;
  const T xs[4] = {low[pos], low[pos + 1], low[pos + 2], low[pos + 3]};
  for (int s = 0; s < 4; ++s) {
    const device T* w = w_up + static_cast<long>(s * H + column) * L;
    for (int tn = 0; tn < 4; ++tn) {
      acc[s] += static_cast<float>(w[pos + tn]) * static_cast<float>(xs[tn]);
    }
  }
}
{
  const int start = full * 128 + lane_base;
  for (int s = 0; s < 4; ++s) {
    const device T* w = w_up + static_cast<long>(s * H + column) * L;
    for (int tn = 0; tn < 4; ++tn) {
      const int pos = start + tn;
      const float wv = (pos < L) ? static_cast<float>(w[pos]) : 0.0f;
      const float xv = (pos < L) ? static_cast<float>(low[pos]) : 0.0f;
      acc[s] += wv * xv;
    }
  }
}
for (int offset = 16; offset > 0; offset >>= 1) {
  for (int s = 0; s < 4; ++s) {
    acc[s] += simd_shuffle_down(acc[s], static_cast<ushort>(offset));
  }
}
if (lane != 0) {
  return;
}
float products[4];
for (int s = 0; s < 4; ++s) {
  const T u = static_cast<T>(acc[s]);
  const T sigmoid = static_cast<T>(ax_sigmoid(static_cast<float>(u)));
  const T gated = sigmoid * normalized[s * H + column];
  products[s] = static_cast<float>(gated) + 0.0f;
}
const float sum = products[3] + (products[2] + (products[1] + products[0]));
mean[column] = static_cast<T>(sum / streams_f32[0]);
"#;

#[cfg(test)]
thread_local! {
    /// Pending injected self-check failure on this thread.
    static INJECTED_SELF_CHECK_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Pending injected launch failure on this thread.
    static INJECTED_LAUNCH_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Kernel builds on this thread (test instrumentation).
    static KERNEL_BUILDS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Self-checks entered on this thread (test instrumentation).
    static SELF_CHECK_RUNS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Launches attempted on this thread (test instrumentation).
    static LAUNCHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Kernel builds on this thread, over both kernels (test instrumentation).
#[cfg(test)]
pub(crate) fn kernel_builds_for_test() -> u64 {
    KERNEL_BUILDS.with(std::cell::Cell::get)
}

/// Self-checks entered on this thread, over both kernels (test
/// instrumentation).
#[cfg(test)]
pub(crate) fn self_check_runs_for_test() -> u64 {
    SELF_CHECK_RUNS.with(std::cell::Cell::get)
}

/// Launches attempted on this thread, over both kernels (test instrumentation).
#[cfg(test)]
pub(crate) fn launches_for_test() -> u64 {
    LAUNCHES.with(std::cell::Cell::get)
}

/// Make the next self-check on this thread fail, and forget any earlier verdict
/// so the self-check actually runs again.
#[cfg(test)]
pub(crate) fn fail_next_self_check_for_test() {
    INJECTED_SELF_CHECK_FAILURE.with(|failure| failure.set(true));
    reset_verdicts_for_test();
}

/// Make the next kernel launch on this thread fail, which must disable that
/// kernel instead of retrying it on every later read.
#[cfg(test)]
pub(crate) fn fail_next_launch_for_test() {
    INJECTED_LAUNCH_FAILURE.with(|failure| failure.set(true));
}

/// Whether the injected launch failure is still pending, i.e. the launch under
/// test never reached the kernel.
#[cfg(test)]
pub(crate) fn launch_failure_pending_for_test() -> bool {
    INJECTED_LAUNCH_FAILURE.with(std::cell::Cell::get)
}

#[cfg(test)]
fn take_injected_self_check_failure() -> bool {
    INJECTED_SELF_CHECK_FAILURE.with(std::cell::Cell::take)
}

#[cfg(test)]
fn take_injected_launch_failure() -> bool {
    INJECTED_LAUNCH_FAILURE.with(std::cell::Cell::take)
}

/// Forget every recorded verdict on this thread, so the next read re-runs the
/// self-check. Test-only: production verdicts are monotone.
#[cfg(test)]
pub(crate) fn reset_verdicts_for_test() {
    VERDICTS.with(|verdicts| {
        for slot in verdicts {
            slot.store(UNKNOWN, std::sync::atomic::Ordering::Release);
        }
    });
}

/// Whether both kernels still have no recorded verdict (test instrumentation).
#[cfg(test)]
pub(crate) fn all_unchecked_for_test() -> bool {
    VERDICTS.with(|verdicts| {
        verdicts
            .iter()
            .all(|slot| slot.load(std::sync::atomic::Ordering::Acquire) == UNKNOWN)
    })
}

/// Whether any kernel is disabled by a failed check or a rejected launch (test
/// instrumentation).
#[cfg(test)]
pub(crate) fn any_verdict_failed_for_test() -> bool {
    VERDICTS.with(|verdicts| {
        verdicts
            .iter()
            .any(|slot| slot.load(std::sync::atomic::Ordering::Acquire) == FAILED)
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// Every bit pattern of a 16-bit float dtype.
    const PATTERNS: usize = 1 << 16;
    /// Real Flash Next geometry: `C = 4` streams of `H = 2560`, low rank 320.
    const PRODUCTION_HIDDEN: usize = 2_560;
    const PRODUCTION_RANK: usize = 320;
    const PRODUCTION_WIDTH: usize = STREAM_COUNT as usize * PRODUCTION_HIDDEN;
    /// Write-gate multiplier the module passes.
    const GATE_SCALE: f32 = 2.0;

    /// Deterministic xorshift generator, so every run replays the same values.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D))
        }

        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        /// Uniform f32 in `[0, 1)`.
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }

        /// Irwin-Hall approximation of `N(0, 1)`: deterministic and free of
        /// transcendentals.
        fn normal(&mut self) -> f32 {
            let sum: f32 = (0..12).map(|_| self.unit()).sum();
            sum - 6.0
        }

        fn slot(&mut self, len: usize) -> usize {
            (self.next() as usize) % len.max(1)
        }
    }

    /// A dense bf16 `[rows, width]` projection whose entries have the standard
    /// deviation of the checkpoint's ~`N(0, 0.02)` projections.
    ///
    /// `wide` adds the huge magnitudes a real checkpoint never has, so products
    /// can overflow the f32 accumulation. One PRNG draw per weight keeps the
    /// production-size fixtures affordable.
    fn weight_matrix(rows: usize, width: usize, rng: &mut Rng, wide: bool) -> MlxArray {
        const SCALE: f32 = 0.034_641_016;
        let mut data = Vec::with_capacity(rows * width);
        for index in 0..rows * width {
            let mut value = (rng.unit() * 2.0 - 1.0) * SCALE;
            if wide && index % 4093 == 0 {
                value = if index % 2 == 0 { 1e30 } else { -1e30 };
            }
            data.push(value);
        }
        array_from_f32(&data, &[rows as i32, width as i32], MlxDtype::Bfloat16)
    }

    /// A bf16 activation vector with the special-value class of `seed`.
    fn vector_with_specials(width: usize, seed: u64) -> MlxArray {
        let mut rng = Rng::new(seed ^ 0xC0FF_EE00_1234);
        let mut data: Vec<f32> = (0..width).map(|_| rng.normal()).collect();
        match seed % 4 {
            0 => {
                // Products that overflow the f32 accumulation to infinity.
                for _ in 0..4 {
                    let slot = rng.slot(width);
                    data[slot] = 1e30;
                }
            }
            1 => {
                // Cancelling pairs.
                for _ in 0..4 {
                    let slot = rng.slot(width.saturating_sub(1));
                    data[slot] = 3e38;
                    data[slot + 1] = -3e38;
                }
            }
            2 => {
                // Non-finite entries and NaN payloads.
                for value in [
                    f32::NAN,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::from_bits(0x7fc0_1234),
                    f32::from_bits(0xff80_0001),
                ] {
                    let slot = rng.slot(width);
                    data[slot] = value;
                }
            }
            _ => {
                // Subnormals and rounding boundaries.
                for index in 0..8u32 {
                    let slot = rng.slot(width);
                    data[slot] = f32::from_bits(0x0000_0001 + index);
                }
            }
        }
        // Every seed also carries a zero, a signed zero and a subnormal.
        if data.len() >= 3 {
            data[0] = 0.0;
            data[1] = -0.0;
            data[2] = 1e-40;
        }
        array_from_f32(&data, &[width as i32], MlxDtype::Bfloat16)
    }

    /// A bf16 activation row of mixed finite and special values.
    fn normalized_with_specials(count: usize, seed: u64) -> MlxArray {
        let mut rng = Rng::new(seed ^ 0x5EED_0011_7777);
        let mut data: Vec<f32> = (0..count).map(|_| rng.normal()).collect();
        for value in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            1e-40,
            1e30,
            -1.0,
            0.0,
        ] {
            let slot = rng.slot(count);
            data[slot] = value;
        }
        array_from_f32(&data, &[count as i32], MlxDtype::Bfloat16)
    }

    /// Production geometry: the native low/gate rows must match the MLX
    /// projection plus the eager stretches bit for bit on every special class.
    #[test]
    fn down_gate_matches_mlx_gemv_path() {
        let width = PRODUCTION_WIDTH;
        let rank = PRODUCTION_RANK;
        let gates = STREAM_COUNT as usize;
        let streams = cached_scalar(STREAM_COUNT as f32, MlxDtype::Bfloat16);
        let scale = cached_scalar(GATE_SCALE, MlxDtype::Bfloat16);
        let mut rng = Rng::new(0x51ED_2701);
        let mut w_down = weight_matrix(rank, width, &mut rng, false);
        let mut w_inject = weight_matrix(gates, width, &mut rng, false);

        for seed in 0..64u64 {
            // A GEMV row reduction depends on the shapes only, so the weights
            // are refreshed every eighth seed and every seed gets a fresh
            // vector with its own special-value class.
            if seed % 8 == 0 && seed > 0 {
                let wide = seed % 16 == 8;
                w_down = weight_matrix(rank, width, &mut rng, wide);
                w_inject = weight_matrix(gates, width, &mut rng, wide);
            }
            let x = vector_with_specials(width, seed);
            let (low, gate) = down_gate(&w_down, &w_inject, &x, &streams, &scale)
                .expect("native down/gate kernel");

            let row = reshape(&x, &[1, 1, width as i32], None);
            let down = matmul(&row, &transpose(&w_down, &[1, 0], None), None);
            let low_reference = silu_projection_dtype(&divide(&down, &streams, None));
            let logits = matmul(&row, &transpose(&w_inject, &[1, 0], None), None);
            let gate_reference = multiply(
                &sigmoid_projection_dtype(&divide(&logits, &streams, None)),
                &scale,
                None,
            );
            assert_eq!(
                raw_bits(&low).expect("low bits"),
                raw_bits(&low_reference).expect("low reference bits"),
                "seed {seed}: low rows differ"
            );
            assert_eq!(
                raw_bits(&gate).expect("gate bits"),
                raw_bits(&gate_reference).expect("gate reference bits"),
                "seed {seed}: gate rows differ"
            );
        }
    }

    /// Production geometry: the native mean must match the MLX projection plus
    /// the eager `sigmoid`, product and stream mean of `mix_from_low`.
    #[test]
    fn up_mix_matches_mlx_path() {
        let hidden = PRODUCTION_HIDDEN;
        let low_width = PRODUCTION_RANK;
        let rows = STREAM_COUNT as usize * hidden;
        let streams_f32 = cached_scalar(STREAM_COUNT as f32, MlxDtype::Float32);
        let mut rng = Rng::new(0x7A11_0C0D);
        let mut w_up = weight_matrix(rows, low_width, &mut rng, false);

        for seed in 0..64u64 {
            if seed % 8 == 0 && seed > 0 {
                let wide = seed % 16 == 8;
                w_up = weight_matrix(rows, low_width, &mut rng, wide);
            }
            let low = vector_with_specials(low_width, seed ^ 0xABCD);
            let normalized = normalized_with_specials(rows, seed);
            let mean =
                up_mix(&w_up, &low, &normalized, &streams_f32).expect("native up/mix kernel");

            let up = matmul(
                &reshape(&low, &[1, 1, low_width as i32], None),
                &transpose(&w_up, &[1, 0], None),
                None,
            );
            let stream_shape = [1, 1, STREAM_COUNT, hidden as i32];
            let gated = multiply(
                &reshape(&sigmoid_projection_dtype(&up), &stream_shape, None),
                &reshape(&normalized, &stream_shape, None),
                None,
            );
            let mean_reference = stream_mean_with(&gated, &streams_f32);
            assert_eq!(
                raw_bits(&mean).expect("mean bits"),
                raw_bits(&mean_reference).expect("mean reference bits"),
                "seed {seed}: mean differs"
            );
        }
    }

    /// The stream mean is a four-term f32 sum of bf16 values whose final bf16
    /// rounding hides almost every difference of association, so the ordinary
    /// data cannot see it. This case builds sums that sit exactly on a bf16
    /// rounding tie, nudged by two terms of half an f32 ulp each: the tie breaks
    /// upward only when those two terms are combined before they meet the large
    /// one, so every association rounds differently in some column.
    #[test]
    fn up_mix_mean_association_is_discriminated() {
        let hidden = PRODUCTION_HIDDEN;
        let low_width = PRODUCTION_RANK;
        let rows = STREAM_COUNT as usize * hidden;
        let streams_f32 = cached_scalar(STREAM_COUNT as f32, MlxDtype::Float32);

        // `up = 100` for every row (`w[r, 0] = 100`, `low = [1, 0, ...]`), so the
        // sigmoid is exactly one and the gated product is `normalized`.
        const BF16_HUNDRED: u16 = 0x42C8;
        let w_up = pattern_matrix(&vec![BF16_HUNDRED; rows], low_width);
        let mut low_bits = vec![0u16; low_width];
        low_bits[0] = BF16_ONE;
        let low = pattern_row(&low_bits);

        // All 24 placements of the multiset {1, 2^-8, 2^-24, 2^-24} over the four
        // streams, at several binades and both signs.
        let base = [1.0f32, 2.0f32.powi(-8), 2.0f32.powi(-24), 2.0f32.powi(-24)];
        let mut placements: Vec<[usize; 4]> = Vec::new();
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let mut seen = [false; 4];
                        for item in [a, b, c, d] {
                            seen[item] = true;
                        }
                        if seen.iter().all(|flag| *flag) {
                            placements.push([a, b, c, d]);
                        }
                    }
                }
            }
        }
        assert_eq!(placements.len(), 24);
        let mut data = vec![0.0f32; rows];
        for column in 0..hidden {
            let order = placements[column % placements.len()];
            let binade = ((column / placements.len()) % 8) as i32 - 4;
            let sign = if (column / (placements.len() * 8)).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            for stream in 0..STREAM_COUNT as usize {
                data[stream * hidden + column] = sign * base[order[stream]] * 2.0f32.powi(binade);
            }
        }
        let normalized = array_from_f32(&data, &[rows as i32], MlxDtype::Bfloat16);

        let mean = up_mix(&w_up, &low, &normalized, &streams_f32).expect("native up/mix kernel");
        let actual = raw_bits(&mean).expect("mean bits");

        let up = matmul(
            &reshape(&low, &[1, 1, low_width as i32], None),
            &transpose(&w_up, &[1, 0], None),
            None,
        );
        let stream_shape = [1, 1, STREAM_COUNT, hidden as i32];
        let gated = multiply(
            &reshape(&sigmoid_projection_dtype(&up), &stream_shape, None),
            &reshape(&normalized, &stream_shape, None),
            None,
        );
        let mean_reference = stream_mean_with(&gated, &streams_f32);
        assert_eq!(
            actual,
            raw_bits(&mean_reference).expect("mean reference bits"),
            "the tie-engineered mean must match MLX's association"
        );

        // The data really discriminates: other associations of the same gated
        // products round differently in some column (host f32 arithmetic, the
        // same operations the kernel performs). MLX's `p3 + (p2 + (p1 + p0))`
        // is the left fold `((p0 + p1) + p2) + p3`, since f32 addition commutes.
        let gated_bits = raw_bits(&gated).expect("gated bits");
        let products: Vec<f32> = gated_bits
            .iter()
            .map(|bits| f32::from_bits(bits << 16))
            .collect();
        let to_bf16 = |value: f32| -> u32 {
            let bits = value.to_bits();
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            rounded >> 16
        };
        let streams = STREAM_COUNT as f32;
        let mut opposite = 0usize;
        let mut interleaved = 0usize;
        let mut pairwise = 0usize;
        for column in 0..hidden {
            let p: Vec<f32> = (0..STREAM_COUNT as usize)
                .map(|stream| products[stream * hidden + column] + 0.0)
                .collect();
            let reference = to_bf16((p[3] + (p[2] + (p[1] + p[0]))) / streams);
            assert_eq!(
                actual[column], reference,
                "column {column}: host replay of MLX's association"
            );
            opposite +=
                usize::from(to_bf16((p[0] + (p[1] + (p[2] + p[3]))) / streams) != reference);
            interleaved +=
                usize::from(to_bf16(((p[0] + p[2]) + (p[1] + p[3])) / streams) != reference);
            pairwise +=
                usize::from(to_bf16(((p[0] + p[1]) + (p[2] + p[3])) / streams) != reference);
        }
        assert!(
            opposite > 0 && interleaved > 0 && pairwise > 0,
            "the tie data must separate the associations \
             (opposite {opposite}, interleaved {interleaved}, pairwise {pairwise})"
        );
    }

    /// Every bf16 pattern through both epilogues: the GEMV is one exact product
    /// (`pattern * 1.0`), so the comparison is the epilogue against the eager
    /// ops for every value the dtype can hold.
    #[test]
    fn epilogues_match_eager_for_every_bf16_value() {
        let patterns: Vec<u16> = (0..PATTERNS).map(|bits| bits as u16).collect();
        let split = PATTERNS - SELF_CHECK_GATES;
        let (head, tail) = patterns.split_at(split);
        // Low role: every pattern, the last four in the gate role. Then the
        // roles swap, so every pattern reaches the gate epilogue too.
        check_down_gate_patterns(head, tail);
        check_down_gate_patterns(tail, head);
        check_up_mix_patterns(&patterns);
    }

    fn check_down_gate_patterns(down: &[u16], gates: &[u16]) {
        let width = SELF_CHECK_WIDTH as usize;
        let w_down = pattern_matrix(down, width);
        let w_inject = pattern_matrix(gates, width);
        let mut x_bits = vec![0u16; width];
        x_bits[0] = BF16_ONE;
        let x = pattern_row(&x_bits);
        let streams = cached_scalar(SELF_CHECK_STREAMS, MlxDtype::Bfloat16);
        let scale = cached_scalar(SELF_CHECK_SCALE, MlxDtype::Bfloat16);
        let geometry = down_gate_geometry(&w_down, &w_inject, &x, &streams, &scale)
            .expect("exhaustive down/gate geometry");
        // The internal launch is used on purpose: at this geometry MLX would
        // not pick the tall-K kernel, but every row has a single product.
        let (low, gate) = down_gate_launch(&w_down, &w_inject, &x, &streams, &scale, geometry)
            .expect("native down/gate epilogue");

        let row = reshape(&x, &[1, 1, width as i32], None);
        let down_reference = matmul(&row, &transpose(&w_down, &[1, 0], None), None);
        let low_reference = silu_projection_dtype(&divide(&down_reference, &streams, None));
        let logits = matmul(&row, &transpose(&w_inject, &[1, 0], None), None);
        let gate_reference = multiply(
            &sigmoid_projection_dtype(&divide(&logits, &streams, None)),
            &scale,
            None,
        );
        assert_eq!(
            raw_bits(&low).expect("low bits"),
            raw_bits(&low_reference).expect("low reference bits"),
            "low epilogue rows {}",
            down.len()
        );
        assert_eq!(
            raw_bits(&gate).expect("gate bits"),
            raw_bits(&gate_reference).expect("gate reference bits"),
            "gate epilogue rows {}",
            gates.len()
        );
    }

    /// Every pattern through the mix epilogue, one stream at a time.
    ///
    /// The other three streams carry zero weights and a zero activation
    /// (`sigmoid(0) * 0 = 0`), so a wrong finite epilogue cannot hide behind a
    /// NaN or infinity that another stream contributes to the same mean. The
    /// activation of the varying stream rotates with the stream index so that
    /// every pattern also meets different gate products.
    fn check_up_mix_patterns(patterns: &[u16]) {
        let hidden = patterns.len();
        let low_width = SELF_CHECK_LOW as usize;
        let rows = STREAM_COUNT as usize * hidden;
        let mut low_bits = vec![0u16; low_width];
        low_bits[0] = BF16_ONE;
        let low = pattern_row(&low_bits);
        let streams_f32 = cached_scalar(STREAM_COUNT as f32, MlxDtype::Float32);
        let activations = self_check_normalized(hidden);
        for varying in 0..STREAM_COUNT as usize {
            let mut weight_patterns = vec![0u16; rows];
            let mut normalized_bits = vec![0u16; rows];
            for (column, pattern) in patterns.iter().enumerate() {
                let row = varying * hidden + column;
                weight_patterns[row] = *pattern;
                normalized_bits[row] = activations[(column + 17 * varying) % hidden];
            }
            let w_up = pattern_matrix(&weight_patterns, low_width);
            let normalized = pattern_row(&normalized_bits);
            let geometry = up_mix_geometry(&w_up, &low, &normalized, &streams_f32)
                .expect("exhaustive up/mix geometry");
            let mean = up_mix_launch(&w_up, &low, &normalized, &streams_f32, geometry)
                .expect("native up/mix epilogue");

            let up = matmul(
                &reshape(&low, &[1, 1, low_width as i32], None),
                &transpose(&w_up, &[1, 0], None),
                None,
            );
            let stream_shape = [1, 1, STREAM_COUNT, hidden as i32];
            let gated = multiply(
                &reshape(&sigmoid_projection_dtype(&up), &stream_shape, None),
                &reshape(&normalized, &stream_shape, None),
                None,
            );
            let mean_reference = stream_mean_with(&gated, &streams_f32);
            assert_eq!(
                raw_bits(&mean).expect("mean bits"),
                raw_bits(&mean_reference).expect("mean reference bits"),
                "mix epilogue, varying stream {varying}"
            );
        }
    }

    /// The MLX `gemv` predicates only allow the shapes the kernels reproduce,
    /// and the public wrappers are the ones that enforce them.
    #[test]
    fn wrappers_require_the_mlx_gemv_config() {
        assert!(mlx_gemv_tall_k(
            PRODUCTION_RANK as i32,
            PRODUCTION_WIDTH as i32
        ));
        assert!(mlx_gemv_tall_k(STREAM_COUNT, PRODUCTION_WIDTH as i32));
        assert!(!mlx_gemv_short_k(
            PRODUCTION_RANK as i32,
            PRODUCTION_WIDTH as i32
        ));
        assert!(mlx_gemv_short_k(
            PRODUCTION_WIDTH as i32,
            PRODUCTION_RANK as i32
        ));
        assert!(!mlx_gemv_tall_k(
            PRODUCTION_WIDTH as i32,
            PRODUCTION_RANK as i32
        ));
        assert!(!mlx_gemv_tall_k(128, SELF_CHECK_WIDTH));
        assert!(!mlx_gemv_short_k(128, SELF_CHECK_WIDTH));

        // A short-K projection is refused by the wrapper and admitted by the
        // internal launch, which is what the epilogue tests rely on.
        let width = SELF_CHECK_WIDTH as usize;
        let rows = 128;
        let w_down = pattern_matrix(&self_check_patterns(rows), width);
        let w_inject = pattern_matrix(&self_check_patterns(SELF_CHECK_GATES), width);
        let mut x_bits = vec![0u16; width];
        x_bits[0] = BF16_ONE;
        let x = pattern_row(&x_bits);
        let streams = cached_scalar(SELF_CHECK_STREAMS, MlxDtype::Bfloat16);
        let scale = cached_scalar(SELF_CHECK_SCALE, MlxDtype::Bfloat16);
        assert!(
            down_gate(&w_down, &w_inject, &x, &streams, &scale).is_none(),
            "the wrapper must refuse a projection outside the tall-K config"
        );
        assert!(
            down_gate_geometry(&w_down, &w_inject, &x, &streams, &scale).is_some(),
            "the geometry itself is valid"
        );
    }

    /// Structural refusals: dtypes, scalar layouts, shapes and stream counts.
    #[test]
    fn unsupported_dtypes_and_shapes_do_not_dispatch() {
        let width = SELF_CHECK_WIDTH as usize;
        let w_down = pattern_matrix(&self_check_patterns(4), width);
        let w_inject = pattern_matrix(&self_check_patterns(STREAM_COUNT as usize), width);
        let mut x_bits = vec![0u16; width];
        x_bits[0] = BF16_ONE;
        let x = pattern_row(&x_bits);
        let streams = cached_scalar(SELF_CHECK_STREAMS, MlxDtype::Bfloat16);
        let scale = cached_scalar(SELF_CHECK_SCALE, MlxDtype::Bfloat16);
        assert!(supports_dtype(MlxDtype::Bfloat16));
        assert!(!supports_dtype(MlxDtype::Float16));
        assert!(!supports_dtype(MlxDtype::Float32));

        let f32_values = vec![0.0f32; 4 * width];
        let f32_down = array_from_f32(&f32_values, &[4, width as i32], MlxDtype::Float32);
        assert!(
            down_gate(&f32_down, &w_inject, &x, &streams, &scale).is_none(),
            "float32 weights must stay on the eager path"
        );
        let half_streams = cached_scalar(SELF_CHECK_STREAMS, MlxDtype::Float16);
        assert!(
            down_gate(&w_down, &w_inject, &x, &half_streams, &scale).is_none(),
            "the scalars must be bf16"
        );
        let wide_x = reshape(&x, &[1, 2, (width / 2) as i32], None);
        assert!(
            down_gate(&w_down, &w_inject, &wide_x, &streams, &scale).is_none(),
            "the row must be a single flat row"
        );
        let narrow = pattern_matrix(&self_check_patterns(4), width / 2);
        assert!(
            down_gate(&w_down, &narrow, &x, &streams, &scale).is_none(),
            "both projections must share the input width"
        );

        // The mix wrapper cannot infer the stream count from `[C * H, L]`: the
        // caller holds that invariant (`STREAM_COUNT`). What the wrapper does
        // enforce is the MLX short-K configuration.
        let rows = 8 * 16;
        let w_up = pattern_matrix(&self_check_patterns(rows), SELF_CHECK_LOW as usize);
        let zero_low = vec![0u16; SELF_CHECK_LOW as usize];
        let zero_normalized = vec![0u16; rows];
        let low = pattern_row(&zero_low);
        let normalized = pattern_row(&zero_normalized);
        let streams_f32 = cached_scalar(STREAM_COUNT as f32, MlxDtype::Float32);
        assert!(
            up_mix(&w_up, &low, &normalized, &streams_f32).is_none(),
            "a projection outside the short-K config must be refused"
        );
    }

    /// The dtype probe is what lets a caller reject an unsupported row before
    /// anything is compiled or built.
    #[test]
    fn dtype_support_is_cheap_and_side_effect_free() {
        let checks = self_check_runs_for_test();
        let builds = kernel_builds_for_test();
        assert!(supports_dtype(MlxDtype::Bfloat16));
        assert!(!supports_dtype(MlxDtype::Float32));
        assert_eq!(
            self_check_runs_for_test(),
            checks,
            "the dtype probe must not run a self-check"
        );
        assert_eq!(
            kernel_builds_for_test(),
            builds,
            "the dtype probe must not create a kernel"
        );
    }

    /// A recorded failure survives a later success, so a racing thread that
    /// passed its own check can never resurrect a kernel another one failed.
    #[test]
    fn failed_verdict_is_terminal() {
        reset_verdicts_for_test();
        let index = Kernel::DownGate.index();
        assert_eq!(record_verdict(index, FAILED), FAILED);
        assert!(any_verdict_failed_for_test());
        assert_eq!(
            record_verdict(index, PASSED),
            FAILED,
            "a later success must not resurrect a failed kernel"
        );

        // The same rule still admits the two legal transitions.
        reset_verdicts_for_test();
        assert_eq!(record_verdict(index, PASSED), PASSED);
        assert_eq!(
            record_verdict(index, PASSED),
            PASSED,
            "PASSED is idempotent"
        );
        assert_eq!(
            record_verdict(index, FAILED),
            FAILED,
            "a rejected launch revokes PASSED"
        );
        assert_eq!(
            record_verdict(index, PASSED),
            FAILED,
            "FAILED stays terminal"
        );
        reset_verdicts_for_test();
        assert!(all_unchecked_for_test());
    }
}
