//! Native hyper-connection low/gate kernel (`ax_hc_low_gate_v1`).
//!
//! One hyper-connection read used to run two independent elementwise
//! stretches on the same `[batch, seq, C * H]` row: `HcLow`
//! (`silu_projection_dtype(down / C)`) and the write gate
//! (`sigmoid_projection_dtype(logits / C) * scale`). The upstream projections
//! are merged into one GEMM, so the concatenated row `[down; logits]` arrives as
//! a single tensor and this kernel produces both outputs in one dispatch:
//! `rank` low-rank activations followed by `stream_count` write gates.
//!
//! Fail closed at every step. Only `bfloat16` and `float16` rows dispatch; the
//! kernel is built with [`MlxMetalKernel::try_new`]; and a one-time per-dtype
//! self-check compares the kernel with the eager two-stretch reference on every
//! special value class of both dtypes, bit for bit. A build failure, a rejected
//! launch or any mismatch leaves the caller on its eager path for the rest of
//! the process.

use mlx_sys::ops::cached_scalar;
use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, astype, view,
};
use std::sync::OnceLock;

use super::qwen4_exp_residual::{sigmoid_projection_dtype, silu_projection_dtype};

const KERNEL_NAME: &str = "ax_hc_low_gate_v1";

/// Threads per threadgroup; the kernel guards the tail threads.
const THREADS_PER_GROUP: i32 = 256;

/// Scale the write gate multiplies the sigmoid by, matching the eager stretch.
const GATE_SCALE: f32 = 2.0;

/// Deterministic values the self-check runs through both roles.
const SELF_CHECK_VALUES: usize = 4096;

/// Stream count the self-check divides by; any positive value works because the
/// kernel takes it as an input.
const SELF_CHECK_STREAMS: f32 = 4.0;

/// Small role of the rotated self-check launch.
const SELF_CHECK_TAIL: usize = 4;

/// Self-check verdict of one dtype.
const UNKNOWN: u8 = 0;
const PASSED: u8 = 1;
const FAILED: u8 = 2;

static KERNEL: OnceLock<Option<MlxMetalKernel>> = OnceLock::new();

#[cfg(not(test))]
static VERDICTS: [std::sync::atomic::AtomicU8; 2] = [
    std::sync::atomic::AtomicU8::new(UNKNOWN),
    std::sync::atomic::AtomicU8::new(UNKNOWN),
];

#[cfg(test)]
thread_local! {
    // Per thread, so one test's injected self-check failure cannot disable the
    // kernel for a test running on another thread. Production keeps the
    // process-wide atomics above.
    static VERDICTS: [std::sync::atomic::AtomicU8; 2] = const {
        [
            std::sync::atomic::AtomicU8::new(UNKNOWN),
            std::sync::atomic::AtomicU8::new(UNKNOWN),
        ]
    };
}

/// Index of a dtype the kernel supports.
fn dtype_index(dtype: MlxDtype) -> Option<usize> {
    match dtype {
        MlxDtype::Bfloat16 => Some(0),
        MlxDtype::Float16 => Some(1),
        _ => None,
    }
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

/// Whether the kernel is known to reproduce the eager stretches for `dtype`.
///
/// The first call per dtype runs the self-check; the recorded verdict is
/// monotone, so a failure disables the kernel for that dtype permanently even
/// when two threads race the first check.
pub(super) fn supported(dtype: MlxDtype) -> bool {
    let Some(index) = dtype_index(dtype) else {
        return false;
    };
    match verdict(index) {
        PASSED => true,
        FAILED => false,
        _ => {
            let passed = self_check_or_injected_failure(dtype);
            record_verdict(index, if passed { PASSED } else { FAILED }) == PASSED
        }
    }
}

/// Whether the kernel has a body for `dtype`.
///
/// Cheap and side-effect free, so a caller can reject an unsupported dtype
/// before anything is compiled, launched or built.
pub(super) fn supports_dtype(dtype: MlxDtype) -> bool {
    dtype_index(dtype).is_some()
}

/// Launch `ax_hc_low_gate_v1` on one `[rank + stream_count]` row.
///
/// `streams` is the stream count as a `[1]` scalar in the row dtype and `scale`
/// the `[1]` write-gate multiplier. Returns the `[rank]` low-rank activation and
/// the `[stream_count]` write gate, both in the row dtype; callers reshape them
/// to their batch/seq shape.
///
/// A launch that the kernel rejects after the dtype passed its self-check
/// disables that dtype for the rest of the process, so a failing dispatch is
/// paid once instead of on every read.
pub(super) fn low_gate(
    cat: &MlxArray,
    streams: &MlxArray,
    scale: &MlxArray,
    rank: i32,
    stream_count: i32,
) -> Option<(MlxArray, MlxArray)> {
    let dtype = cat.dtype();
    if rank <= 0 || stream_count <= 0 {
        return None;
    }
    let total = rank.checked_add(stream_count)?;
    if cat.shape() != [total] {
        return None;
    }
    if streams.shape() != [1]
        || scale.shape() != [1]
        || streams.dtype() != dtype
        || scale.dtype() != dtype
    {
        return None;
    }
    let index = dtype_index(dtype)?;
    if !supported(dtype) {
        return None;
    }
    match launch(cat, streams, scale, rank, stream_count, dtype) {
        Some(outputs) => Some(outputs),
        None => {
            // This dtype already passed the self-check, so the rejection is a
            // runtime failure rather than an unsupported configuration.
            // Disable it instead of retrying a failing launch on every read.
            record_verdict(index, FAILED);
            tracing::warn!(
                ?dtype,
                rank,
                stream_count,
                "Flash Next hyper-connection low/gate kernel rejected its launch; using eager ops"
            );
            None
        }
    }
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

fn kernel() -> Option<&'static MlxMetalKernel> {
    KERNEL
        .get_or_init(|| {
            #[cfg(test)]
            KERNEL_BUILDS.with(|builds| builds.set(builds.get() + 1));
            match MlxMetalKernel::try_new(
                KERNEL_NAME,
                &["cat", "streams", "scale"],
                &["low", "gate"],
                KERNEL_SOURCE,
                SIGMOID_HEADER,
                true,
            ) {
                Ok(kernel) => Some(kernel),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "Flash Next hyper-connection low/gate kernel failed to build; using eager ops"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// One raw launch, without the self-check. Returns `(low, gate)`.
fn launch(
    cat: &MlxArray,
    streams: &MlxArray,
    scale: &MlxArray,
    rank: i32,
    stream_count: i32,
    dtype: MlxDtype,
) -> Option<(MlxArray, MlxArray)> {
    #[cfg(test)]
    LAUNCHES.with(|launches| launches.set(launches.get() + 1));
    let kernel = kernel()?;
    #[cfg(test)]
    if take_injected_launch_failure() {
        return None;
    }
    let total = rank.checked_add(stream_count)?;
    let mut outputs = kernel
        .try_apply_with_template(
            &[cat, streams, scale],
            &[
                KernelOutputSpec {
                    shape: vec![rank],
                    dtype,
                },
                KernelOutputSpec {
                    shape: vec![stream_count],
                    dtype,
                },
            ],
            &[
                KernelTemplateArg::Dtype { name: "T", dtype },
                KernelTemplateArg::Int {
                    name: "Rank",
                    value: rank,
                },
                KernelTemplateArg::Int {
                    name: "Streams",
                    value: stream_count,
                },
            ],
            (total, 1, 1),
            (THREADS_PER_GROUP, 1, 1),
            None,
        )
        .ok()?;
    if outputs.len() != 2 {
        return None;
    }
    let gate = outputs.pop()?;
    let low = outputs.pop()?;
    Some((low, gate))
}

/// Self-check with the test-only failure switch in front of it.
fn self_check_or_injected_failure(dtype: MlxDtype) -> bool {
    #[cfg(test)]
    SELF_CHECK_RUNS.with(|runs| runs.set(runs.get() + 1));
    #[cfg(test)]
    let injected = take_injected_failure();
    #[cfg(not(test))]
    let injected = false;
    if injected {
        tracing::warn!(
            ?dtype,
            "Flash Next hyper-connection low/gate kernel self-check was injected to fail; using eager ops"
        );
        return false;
    }
    // An eager-op panic (a rejected graph) must disable the kernel rather than
    // escape from a decode step, exactly like a bit mismatch.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self_check(dtype))).unwrap_or(false)
}

/// Compare the kernel with the eager two-stretch reference on the special-value
/// sample of `dtype`, running the whole sample through both roles.
fn self_check(dtype: MlxDtype) -> bool {
    let count = SELF_CHECK_VALUES as i32;
    let tail = SELF_CHECK_TAIL as i32;
    let sample = array_from_f32(&self_check_values(), &[count], dtype);
    if mlx_sys::try_eval(&[&sample]).is_err() {
        warn_self_check(dtype);
        return false;
    }

    let streams = cached_scalar(SELF_CHECK_STREAMS, dtype);
    let scale = cached_scalar(GATE_SCALE, dtype);
    let scaled = mlx_sys::divide(&sample, &streams, None);
    let low_reference = silu_projection_dtype(&scaled);
    let gate_reference = mlx_sys::multiply(&sigmoid_projection_dtype(&scaled), &scale, None);
    let head = mlx_sys::slice(&sample, &[0], &[tail], &[1], None);

    // Low role: the whole sample, with the short role taken from its head.
    let low_row = mlx_sys::concatenate(&[&sample, &head], 0, None);
    let Some((low, short_gate)) = launch(&low_row, &streams, &scale, count, tail, dtype) else {
        warn_self_check(dtype);
        return false;
    };
    // Gate role: the whole sample again, so every value also reaches the gate.
    let gate_row = mlx_sys::concatenate(&[&head, &sample], 0, None);
    let Some((short_low, gate)) = launch(&gate_row, &streams, &scale, tail, count, dtype) else {
        warn_self_check(dtype);
        return false;
    };

    let (Some(low_actual), Some(gate_actual), Some(short_low_actual), Some(short_gate_actual)) = (
        raw_bits(&low),
        raw_bits(&gate),
        raw_bits(&short_low),
        raw_bits(&short_gate),
    ) else {
        warn_self_check(dtype);
        return false;
    };
    let (Some(low_expected), Some(gate_expected)) =
        (raw_bits(&low_reference), raw_bits(&gate_reference))
    else {
        warn_self_check(dtype);
        return false;
    };

    // The short roles read the sample head, so they compare against the head of
    // the matching reference.
    let matches = low_expected.len() == SELF_CHECK_VALUES
        && gate_expected.len() == SELF_CHECK_VALUES
        && low_actual[..] == low_expected[..]
        && gate_actual[..] == gate_expected[..]
        && short_low_actual[..] == low_expected[..SELF_CHECK_TAIL]
        && short_gate_actual[..] == gate_expected[..SELF_CHECK_TAIL];
    if !matches {
        warn_self_check(dtype);
    }
    matches
}

fn warn_self_check(dtype: MlxDtype) {
    tracing::warn!(
        ?dtype,
        "Flash Next hyper-connection low/gate kernel does not match the eager stretches; using eager ops"
    );
}

/// Deterministic sample: every bf16/f16 special class plus ordinary magnitudes,
/// so rounding boundaries and non-finite payloads are compared.
fn self_check_values() -> Vec<f32> {
    let special = [
        0.0f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        -1.0,
        88.0,
        -88.0,
        65504.0,
        -65504.0,
        f32::MAX,
        f32::MIN,
        1e-40,
        -1e-40,
        f32::from_bits(0x7fc0_0000),
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xffc1_2345),
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x007f_ffff),
        f32::from_bits(0x8000_0001),
        2.0,
        -2.0,
        0.5,
        -0.5,
        1e10,
        -1e10,
        1e-10,
        -1e-10,
        3.0,
        -3.0,
    ];
    let mut values = Vec::with_capacity(SELF_CHECK_VALUES);
    values.extend_from_slice(&special);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    while values.len() < SELF_CHECK_VALUES {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
        let exponent = (values.len() % 24) as i32 - 12;
        values.push((unit * 2.0 - 1.0) * 2f32.powi(exponent));
    }
    values
}

#[cfg(test)]
thread_local! {
    static INJECTED_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Make the next self-check on this thread fail, and forget any earlier verdict
/// so the self-check actually runs again.
#[cfg(test)]
pub(crate) fn fail_next_self_check_for_test() {
    INJECTED_FAILURE.with(|failure| failure.set(true));
    reset_verdicts_for_test();
}

#[cfg(test)]
fn take_injected_failure() -> bool {
    INJECTED_FAILURE.with(std::cell::Cell::take)
}

#[cfg(test)]
thread_local! {
    /// Kernel builds on this thread (test instrumentation).
    static KERNEL_BUILDS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Self-checks entered on this thread (test instrumentation).
    static SELF_CHECK_RUNS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Launches attempted on this thread (test instrumentation).
    static LAUNCHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Pending injected launch failure on this thread.
    static INJECTED_LAUNCH_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Kernel builds on this thread (test instrumentation).
#[cfg(test)]
pub(crate) fn kernel_builds_for_test() -> u64 {
    KERNEL_BUILDS.with(std::cell::Cell::get)
}

/// Self-checks entered on this thread (test instrumentation).
#[cfg(test)]
pub(crate) fn self_check_runs_for_test() -> u64 {
    SELF_CHECK_RUNS.with(std::cell::Cell::get)
}

/// Launches attempted on this thread (test instrumentation).
#[cfg(test)]
pub(crate) fn launches_for_test() -> u64 {
    LAUNCHES.with(std::cell::Cell::get)
}

/// Make the next kernel launch on this thread fail, which must disable the
/// dtype instead of retrying it on every later read.
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

/// Whether `dtype` still has no recorded verdict (test instrumentation).
#[cfg(test)]
pub(crate) fn unchecked_for_test(dtype: MlxDtype) -> bool {
    dtype_index(dtype).is_some_and(|index| verdict(index) == UNKNOWN)
}

/// Whether `dtype` is disabled by a failed check or a rejected launch (test
/// instrumentation).
#[cfg(test)]
pub(crate) fn verdict_failed_for_test(dtype: MlxDtype) -> bool {
    dtype_index(dtype).is_some_and(|index| verdict(index) == FAILED)
}

const KERNEL_SOURCE: &str = r#"
#pragma clang fp contract(off)
const int j = static_cast<int>(thread_position_in_grid.x);
if (j >= Rank + Streams) {
  return;
}
// dtype-T division, exactly like `divide(row, streams)` on the eager path.
const T x = cat[j] / streams[0];
if (j < Rank) {
  // silu_projection_dtype: f32 silu, rounded once to T.
  const float f = static_cast<float>(x);
  low[j] = static_cast<T>(f * ax_sigmoid(f));
} else {
  // sigmoid_projection_dtype, then the T-typed scale product.
  const T gate_value = static_cast<T>(ax_sigmoid(static_cast<float>(x)));
  gate[j - Rank] = gate_value * scale[0];
}
"#;

/// MLX's sigmoid, spelled so the JIT keeps the precise exponential.
///
/// The source body is a kernel function body, so the helper has to live in the
/// header MLX prepends to the generated signature.
const SIGMOID_HEADER: &str = r#"
inline float ax_sigmoid(float x) {
  const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"#;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use mlx_sys::{divide, eval, multiply, slice};

    /// Every bit pattern of a 16-bit float dtype.
    const PATTERNS: usize = 1 << 16;
    /// Streams the eager reference divides by.
    const STREAMS: usize = 4;

    fn all_patterns() -> Vec<u16> {
        (0..PATTERNS).map(|bits| bits as u16).collect()
    }

    /// A fixed-width float array holding `bits` verbatim, including NaN
    /// payloads and subnormals.
    fn pattern_array(bits: &[u16], dtype: MlxDtype) -> MlxArray {
        let raw = MlxArray::from_raw_data(
            bits.as_ptr().cast(),
            std::mem::size_of_val(bits),
            &[bits.len() as i32],
            MlxDtype::Uint16,
        );
        eval(&[&raw]);
        view(&raw, dtype, None)
    }

    /// One launch with `bits[..rank]` in the low role and the rest in the gate.
    fn check_role(bits: &[u16], dtype: MlxDtype, rank: usize, stream_count: usize) {
        let cat = pattern_array(bits, dtype);
        let streams = cached_scalar(STREAMS as f32, dtype);
        let scale = cached_scalar(GATE_SCALE, dtype);
        let (low, gate) = low_gate(&cat, &streams, &scale, rank as i32, stream_count as i32)
            .expect("native hyper-connection low/gate kernel");
        assert_eq!(low.shape(), [rank as i32], "{dtype:?} low shape");
        assert_eq!(gate.shape(), [stream_count as i32], "{dtype:?} gate shape");

        let low_input = slice(&cat, &[0], &[rank as i32], &[1], None);
        let gate_input = slice(
            &cat,
            &[rank as i32],
            &[(rank + stream_count) as i32],
            &[1],
            None,
        );
        let low_reference = silu_projection_dtype(&divide(&low_input, &streams, None));
        let gate_reference = multiply(
            &sigmoid_projection_dtype(&divide(&gate_input, &streams, None)),
            &scale,
            None,
        );
        assert_eq!(
            raw_bits(&low).expect("low bits"),
            raw_bits(&low_reference).expect("low reference bits"),
            "low role {dtype:?} rank {rank}"
        );
        assert_eq!(
            raw_bits(&gate).expect("gate bits"),
            raw_bits(&gate_reference).expect("gate reference bits"),
            "gate role {dtype:?} streams {stream_count}"
        );
    }

    #[test]
    fn native_hc_low_gate_matches_eager_for_every_bf16_and_f16_value() {
        let patterns = all_patterns();
        // Rotated so the four patterns the first launch puts in the gate role
        // move to the front; the second launch then runs every pattern through
        // the gate role as well.
        let rotation = PATTERNS - STREAMS;
        let mut rotated = Vec::with_capacity(PATTERNS);
        rotated.extend_from_slice(&patterns[rotation..]);
        rotated.extend_from_slice(&patterns[..rotation]);

        for dtype in [MlxDtype::Bfloat16, MlxDtype::Float16] {
            // Every pattern in the low role, the last four in the gate role.
            check_role(&patterns, dtype, rotation, STREAMS);
            // Every pattern in the gate role, the first four in the low role.
            check_role(&rotated, dtype, STREAMS, rotation);
        }
    }

    #[test]
    fn unsupported_dtypes_and_shapes_do_not_dispatch() {
        let values = vec![0.25f32; 8];
        let row = array_from_f32(&values, &[8], MlxDtype::Float32);
        let streams = cached_scalar(4.0, MlxDtype::Float32);
        let scale = cached_scalar(GATE_SCALE, MlxDtype::Float32);
        assert!(
            low_gate(&row, &streams, &scale, 4, 4).is_none(),
            "float32 rows must stay on the eager path"
        );

        let half = array_from_f32(&values, &[8], MlxDtype::Float16);
        assert!(
            low_gate(&half, &streams, &scale, 4, 4).is_none(),
            "the scalars must share the row dtype"
        );
        let half_streams = cached_scalar(4.0, MlxDtype::Float16);
        let half_scale = cached_scalar(GATE_SCALE, MlxDtype::Float16);
        assert!(
            low_gate(&half, &half_streams, &half_scale, 4, 8).is_none(),
            "the row width must be rank + streams"
        );
    }

    /// The dtype probe is what lets a caller reject an unsupported row before
    /// anything is compiled or built.
    #[test]
    fn dtype_support_is_cheap_and_side_effect_free() {
        let checks = self_check_runs_for_test();
        let builds = kernel_builds_for_test();
        assert!(supports_dtype(MlxDtype::Bfloat16));
        assert!(supports_dtype(MlxDtype::Float16));
        assert!(!supports_dtype(MlxDtype::Float32));
        assert_eq!(
            self_check_runs_for_test(),
            checks,
            "the dtype probe must not run a self-check"
        );
        assert_eq!(
            kernel_builds_for_test(),
            builds,
            "the dtype probe must not create the kernel"
        );
    }

    /// A recorded failure survives a later success, so a racing thread that
    /// passed its own check can never resurrect a dtype another one failed.
    #[test]
    fn failed_verdict_is_terminal() {
        reset_verdicts_for_test();
        let index = dtype_index(MlxDtype::Bfloat16).expect("bfloat16 is a kernel dtype");
        assert_eq!(record_verdict(index, FAILED), FAILED);
        assert!(verdict_failed_for_test(MlxDtype::Bfloat16));
        assert_eq!(
            record_verdict(index, PASSED),
            FAILED,
            "a later success must not resurrect a failed dtype"
        );
        assert!(
            !supported(MlxDtype::Bfloat16),
            "a failed dtype must stay disabled"
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
    }

    /// Host cost of one native launch against the compiled two-stretch pair it
    /// replaces. Prints microseconds per call; informational, no assertion.
    #[test]
    #[ignore = "host timing probe; run with --nocapture on the build host"]
    fn native_hc_host_cost_probe() {
        use crate::model::shared::qwen4_exp_fused::{self, Stretch};
        use mlx_sys::ops::cached_scalar;
        use std::time::Instant;
        let dtype = MlxDtype::Bfloat16;
        let (rank, streams) = (320i32, 4i32);
        let total = rank + streams;
        let data: Vec<f32> = (0..total).map(|i| (i as f32 - 160.0) * 0.01).collect();
        let cat = array_from_f32(&data, &[total], dtype);
        let streams_scalar = cached_scalar(4.0, dtype);
        let scale = cached_scalar(2.0, dtype);
        assert!(supported(dtype));
        let n = 20_000;
        let warm = low_gate(&cat, &streams_scalar, &scale, rank, streams).is_some();
        assert!(warm);
        let started = Instant::now();
        let mut keep = Vec::with_capacity(8);
        for i in 0..n {
            let out = low_gate(&cat, &streams_scalar, &scale, rank, streams).unwrap();
            if i % 4096 == 0 {
                keep.push(out);
            }
        }
        let native = started.elapsed().as_secs_f64() / n as f64 * 1e6;
        drop(keep);
        let low_in = array_from_f32(&data[..rank as usize], &[1, 1, rank], dtype);
        let gate_in = array_from_f32(&data[rank as usize..], &[1, 1, streams], dtype);
        let _guard = qwen4_exp_fused::force_for_test(Some(true));
        let started = Instant::now();
        let mut keep = Vec::with_capacity(8);
        for i in 0..n {
            let low = divide(&low_in, &streams_scalar, None);
            let gate = divide(&gate_in, &streams_scalar, None);
            if i % 4096 == 0 {
                keep.push((low, gate));
            }
        }
        let two_eager_ops = started.elapsed().as_secs_f64() / n as f64 * 1e6;
        drop(keep);
        let _ = Stretch::HcLow;
        eprintln!(
            "HOST_COST native low_gate {native:.2} us/call; two plain elementwise ops {two_eager_ops:.2} us"
        );
    }
}
