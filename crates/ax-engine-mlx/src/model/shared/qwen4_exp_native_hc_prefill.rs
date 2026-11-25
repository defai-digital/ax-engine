//! Prefill-sized hyper-connection kernels for Flash Next.
//!
//! A gated residual touches the packed `[rows, streams * hidden]` residual
//! about a dozen times through MLX ops: the grouped norm alone widens it to
//! f32, normalizes it, scales it and narrows it again. At prefill row counts
//! every one of those passes is a memory pass over tens of megabytes, 96 times
//! per forward. The three kernels here do the same arithmetic in one pass each:
//!
//! - `ax_hc_prefill_norm_v1`: grouped RMS norm, gain and the bf16 store;
//! - `ax_hc_prefill_mix_v1`: `mean_streams(sigmoid(up) * normalized)`;
//! - `ax_hc_prefill_write_v1`: `residual + gate * branch`.
//!
//! Each kernel reproduces the MLX op chain it replaces bit for bit, rounding
//! through bf16 at every point the chain stores bf16, and the norm sums its
//! squares in the order MLX's `rms_norm` does (four per thread, one
//! `simd_sum` per simdgroup, one over the simdgroup partials). A process-wide
//! self-check at the caller's geometry compares all three with the MLX chain
//! before the first use; a kernel that fails to build, launch or match leaves
//! every residual on the MLX ops.
//!
//! Decode rows and verifier windows never reach these kernels ([`MIN_ROWS`]).

use mlx_sys::ops::cached_scalar;
use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, astype, multiply,
    reshape,
};
use std::sync::{Mutex, OnceLock};

use super::qwen4_exp_native_hc_gemv::{array_from_f32, raw_bits};
use super::qwen4_exp_residual::{sigmoid_projection_dtype, stream_mean_with};

/// Fewest rows the kernels run at. Below it the MLX ops cost the same.
pub(super) const MIN_ROWS: i32 = 64;

/// Values one norm thread reads, as in MLX's `rms_norm`.
const NORM_READS: i32 = 4;
/// Threads of one simdgroup.
const SIMD_WIDTH: i32 = 32;
/// Largest threadgroup the norm asks for.
const MAX_THREADS_PER_GROUP: i32 = 1024;
/// Threadgroup of the two elementwise kernels.
const ELEMENT_THREADS: i32 = 256;
/// Most streams the mix sums in MLX's order: up to here MLX's small strided
/// reduction keeps one accumulator per stream and folds them in order.
const MAX_STREAMS: i32 = 8;
/// Rows of the self-check tensors.
const SELF_CHECK_ROWS: i32 = 3;

const NORM_NAME: &str = "ax_hc_prefill_norm_v1";
const MIX_NAME: &str = "ax_hc_prefill_mix_v1";
const WRITE_NAME: &str = "ax_hc_prefill_write_v1";

static NORM: OnceLock<Option<MlxMetalKernel>> = OnceLock::new();
static MIX: OnceLock<Option<MlxMetalKernel>> = OnceLock::new();
static WRITE: OnceLock<Option<MlxMetalKernel>> = OnceLock::new();

/// `(streams, hidden, eps bits)` of a checked residual and its verdict.
type Verdict = ((i32, i32, u32), bool);

/// Self-check verdicts, one per geometry seen.
static VERDICTS: Mutex<Vec<Verdict>> = Mutex::new(Vec::new());

/// Stream geometry of a gated residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Geometry {
    pub(super) streams: i32,
    pub(super) hidden: i32,
}

impl Geometry {
    fn packed(self) -> Option<i32> {
        self.streams.checked_mul(self.hidden)
    }

    /// Threads of one norm threadgroup, when the hidden size fits the
    /// single-threadgroup reduction the kernel shares with MLX.
    fn norm_threads(self) -> Option<i32> {
        let threads = self.hidden / NORM_READS;
        // One stream is a reduction MLX skips, more than eight is one it
        // splits across accumulators.
        ((2..=MAX_STREAMS).contains(&self.streams)
            && self.hidden > 0
            && self.hidden % (NORM_READS * SIMD_WIDTH) == 0
            && threads <= MAX_THREADS_PER_GROUP)
            .then_some(threads)
    }
}

/// Whether a residual of `rows` rows may use the kernels at all: the switch,
/// the row floor and a geometry the norm supports. The self-check runs later,
/// on the first call that also has bf16 tensors.
fn admitted(geometry: Geometry, rows: i32) -> bool {
    rows >= MIN_ROWS
        && geometry.norm_threads().is_some()
        && crate::fastpath::flash_next_hc_prefill_kernels_enabled()
}

fn is_bf16(arrays: &[&MlxArray]) -> bool {
    arrays
        .iter()
        .all(|array| array.dtype() == MlxDtype::Bfloat16)
}

/// Grouped RMS norm of `packed` (`[batch, rows, streams * hidden]`, bf16)
/// times `gain` (`[streams * hidden]`), stored as bf16.
pub(super) fn norm(
    geometry: Geometry,
    packed: &MlxArray,
    gain: &MlxArray,
    eps: f32,
    rows: i32,
) -> Option<MlxArray> {
    if !admitted(geometry, rows) || !is_bf16(&[packed]) || !verified(geometry, eps) {
        return None;
    }
    norm_launch(geometry, packed, gain, eps)
}

/// `mean_streams(sigmoid(up) * normalized)` as bf16 `[batch, rows, hidden]`.
pub(super) fn mix(
    geometry: Geometry,
    up: &MlxArray,
    normalized: &MlxArray,
    eps: f32,
    rows: i32,
) -> Option<MlxArray> {
    if !admitted(geometry, rows) || !is_bf16(&[up, normalized]) || !verified(geometry, eps) {
        return None;
    }
    mix_launch(geometry, up, normalized)
}

/// `residual + gate * branch` over the packed residual, all bf16.
pub(super) fn write(
    geometry: Geometry,
    residual: &MlxArray,
    gate: &MlxArray,
    branch: &MlxArray,
    eps: f32,
    rows: i32,
) -> Option<MlxArray> {
    if !admitted(geometry, rows) || !is_bf16(&[residual, gate, branch]) || !verified(geometry, eps)
    {
        return None;
    }
    write_launch(geometry, residual, gate, branch)
}

fn template(geometry: Geometry) -> [KernelTemplateArg<'static>; 2] {
    [
        KernelTemplateArg::Int {
            name: "S",
            value: geometry.streams,
        },
        KernelTemplateArg::Int {
            name: "H",
            value: geometry.hidden,
        },
    ]
}

/// `(batch, rows)` of a rank-three tensor whose last axis is `width`.
fn leading_dims(array: &MlxArray, width: i32) -> Option<(i32, i32)> {
    match array.shape()[..] {
        [batch, rows, last] if last == width && batch > 0 && rows > 0 => Some((batch, rows)),
        _ => None,
    }
}

fn norm_launch(
    geometry: Geometry,
    packed: &MlxArray,
    gain: &MlxArray,
    eps: f32,
) -> Option<MlxArray> {
    let width = geometry.packed()?;
    let threads = geometry.norm_threads()?;
    let (batch, rows) = leading_dims(packed, width)?;
    if gain.shape() != [width] {
        return None;
    }
    let groups = batch.checked_mul(rows)?.checked_mul(geometry.streams)?;
    let grid = groups.checked_mul(threads)?;
    let kernel = kernel(&NORM, NORM_NAME, &["x", "gain", "eps", "axis"], NORM_SOURCE)?;
    let gain = astype(gain, MlxDtype::Float32, None);
    let eps = cached_scalar(eps, MlxDtype::Float32);
    // The axis size is a runtime operand, as in MLX's kernel, so the compiler
    // cannot turn the mean into a multiplication by a rounded reciprocal.
    let axis = astype(
        &cached_scalar(geometry.hidden as f32, MlxDtype::Float32),
        MlxDtype::Uint32,
        None,
    );
    single_output(kernel.try_apply_with_template(
        &[packed, &gain, &eps, &axis],
        &[KernelOutputSpec {
            shape: vec![batch, rows, width],
            dtype: MlxDtype::Bfloat16,
        }],
        &template(geometry),
        (grid, 1, 1),
        (threads, 1, 1),
        None,
    ))
}

fn mix_launch(geometry: Geometry, up: &MlxArray, normalized: &MlxArray) -> Option<MlxArray> {
    let width = geometry.packed()?;
    let (batch, rows) = leading_dims(up, width)?;
    if leading_dims(normalized, width)? != (batch, rows) {
        return None;
    }
    let grid = batch.checked_mul(rows)?.checked_mul(geometry.hidden)?;
    let kernel = kernel(&MIX, MIX_NAME, &["up", "normalized", "streams"], MIX_SOURCE)?;
    let streams = cached_scalar(geometry.streams as f32, MlxDtype::Float32);
    single_output(kernel.try_apply_with_template(
        &[up, normalized, &streams],
        &[KernelOutputSpec {
            shape: vec![batch, rows, geometry.hidden],
            dtype: MlxDtype::Bfloat16,
        }],
        &template(geometry),
        (grid, 1, 1),
        (ELEMENT_THREADS, 1, 1),
        None,
    ))
}

fn write_launch(
    geometry: Geometry,
    residual: &MlxArray,
    gate: &MlxArray,
    branch: &MlxArray,
) -> Option<MlxArray> {
    let width = geometry.packed()?;
    let (batch, rows) = leading_dims(residual, width)?;
    if leading_dims(gate, geometry.streams)? != (batch, rows)
        || leading_dims(branch, geometry.hidden)? != (batch, rows)
    {
        return None;
    }
    let grid = batch.checked_mul(rows)?.checked_mul(width)?;
    let kernel = kernel(
        &WRITE,
        WRITE_NAME,
        &["residual", "gate", "branch"],
        WRITE_SOURCE,
    )?;
    single_output(kernel.try_apply_with_template(
        &[residual, gate, branch],
        &[KernelOutputSpec {
            shape: vec![batch, rows, width],
            dtype: MlxDtype::Bfloat16,
        }],
        &template(geometry),
        (grid, 1, 1),
        (ELEMENT_THREADS, 1, 1),
        None,
    ))
}

pub(super) fn single_output(outputs: Result<Vec<MlxArray>, String>) -> Option<MlxArray> {
    match outputs {
        Ok(mut outputs) if outputs.len() == 1 => outputs.pop(),
        Ok(_) => None,
        Err(error) => {
            tracing::warn!(
                %error,
                "Flash Next prefill kernel rejected a launch; using MLX ops"
            );
            None
        }
    }
}

fn kernel(
    cell: &'static OnceLock<Option<MlxMetalKernel>>,
    name: &str,
    inputs: &[&str],
    source: &str,
) -> Option<&'static MlxMetalKernel> {
    cell.get_or_init(|| {
        // `ensure_row_contiguous` lets the kernels index plain row-major
        // tensors; MLX copies anything else.
        match MlxMetalKernel::try_new(name, inputs, &["out"], source, SIGMOID_HEADER, true) {
            Ok(kernel) => Some(kernel),
            Err(error) => {
                tracing::warn!(
                    %error,
                    name,
                    "Flash Next hyper-connection prefill kernel failed to build; using MLX ops"
                );
                None
            }
        }
    })
    .as_ref()
}

/// Whether the three kernels reproduced the MLX chain at this geometry. The
/// first call per geometry runs the check; the verdict never changes after.
fn verified(geometry: Geometry, eps: f32) -> bool {
    let key = (geometry.streams, geometry.hidden, eps.to_bits());
    let Ok(mut verdicts) = VERDICTS.lock() else {
        return false;
    };
    if let Some((_, verdict)) = verdicts.iter().find(|(seen, _)| *seen == key) {
        return *verdict;
    }
    let verdict = self_check(geometry, eps).unwrap_or(false);
    if !verdict {
        tracing::warn!(
            streams = geometry.streams,
            hidden = geometry.hidden,
            "Flash Next hyper-connection prefill kernels differ from the MLX ops; using MLX ops"
        );
    }
    verdicts.push((key, verdict));
    verdict
}

/// The MLX chain the norm kernel replaces.
fn norm_reference(geometry: Geometry, packed: &MlxArray, gain: &MlxArray, eps: f32) -> MlxArray {
    let shape = packed.shape();
    let grouped = reshape(
        &astype(packed, MlxDtype::Float32, None),
        &[shape[0], shape[1], geometry.streams, geometry.hidden],
        None,
    );
    let unit = reshape(&mlx_sys::rms_norm(&grouped, None, eps, None), &shape, None);
    astype(
        &multiply(&unit, &astype(gain, MlxDtype::Float32, None), None),
        MlxDtype::Bfloat16,
        None,
    )
}

/// The MLX chain the mix kernel replaces.
fn mix_reference(geometry: Geometry, up: &MlxArray, normalized: &MlxArray) -> MlxArray {
    let shape = up.shape();
    let streams = [shape[0], shape[1], geometry.streams, geometry.hidden];
    let gated = multiply(
        &reshape(&sigmoid_projection_dtype(up), &streams, None),
        &reshape(normalized, &streams, None),
        None,
    );
    stream_mean_with(
        &gated,
        &cached_scalar(geometry.streams as f32, MlxDtype::Float32),
    )
}

/// The MLX chain the write kernel replaces.
fn write_reference(
    geometry: Geometry,
    residual: &MlxArray,
    gate: &MlxArray,
    branch: &MlxArray,
) -> MlxArray {
    let shape = residual.shape();
    let delta = reshape(
        &multiply(
            &reshape(gate, &[shape[0], shape[1], geometry.streams, 1], None),
            &reshape(branch, &[shape[0], shape[1], 1, geometry.hidden], None),
            None,
        ),
        &shape,
        None,
    );
    mlx_sys::add(residual, &delta, None)
}

/// Deterministic bf16 test tensor: a spread of signs and magnitudes with the
/// occasional large value, so the norm, the sigmoid tails and the bf16
/// rounding points all see more than one regime.
pub(super) fn pattern(shape: &[i32], phase: u32, spread: f32) -> MlxArray {
    let count: i32 = shape.iter().product();
    let data: Vec<f32> = (0..count.max(0) as u32)
        .map(|index| {
            let mixed = index
                .wrapping_add(phase)
                .wrapping_mul(2_654_435_761)
                .rotate_left(13);
            let unit = (mixed % 20_011) as f32 / 10_005.0 - 1.0;
            let boost = if mixed % 97 == 0 { 9.0 } else { 1.0 };
            unit * spread * boost
        })
        .collect();
    array_from_f32(&data, shape, MlxDtype::Bfloat16)
}

pub(super) fn same_bits(left: &MlxArray, right: &MlxArray) -> Option<bool> {
    Some(left.shape() == right.shape() && raw_bits(left)? == raw_bits(right)?)
}

fn self_check(geometry: Geometry, eps: f32) -> Option<bool> {
    let width = geometry.packed()?;
    let rows = SELF_CHECK_ROWS;
    let packed = pattern(&[1, rows, width], 1, 0.75);
    let gain = astype(&pattern(&[width], 7, 0.05), MlxDtype::Float32, None);
    let gain = mlx_sys::add(&gain, &cached_scalar(1.0, MlxDtype::Float32), None);
    let up = pattern(&[1, rows, width], 11, 3.0);
    let gate = pattern(&[1, rows, geometry.streams], 17, 1.0);
    let branch = pattern(&[1, rows, geometry.hidden], 23, 1.5);

    let normalized = norm_reference(geometry, &packed, &gain, eps);
    let checks = [
        (
            norm_launch(geometry, &packed, &gain, eps)?,
            normalized.clone(),
        ),
        (
            mix_launch(geometry, &up, &normalized)?,
            mix_reference(geometry, &up, &normalized),
        ),
        (
            write_launch(geometry, &packed, &gate, &branch)?,
            write_reference(geometry, &packed, &gate, &branch),
        ),
    ];
    for (actual, expected) in &checks {
        if !same_bits(actual, expected)? {
            return Some(false);
        }
    }
    Some(true)
}

/// MLX's sigmoid, spelled so the JIT keeps the precise exponential.
const SIGMOID_HEADER: &str = r#"
inline float ax_hc_prefill_sigmoid(float x) {
  const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
  return (x < 0.0f) ? y : 1.0f - y;
}
"#;

/// One threadgroup per `(row, stream)`: `H / 4` threads each read four values,
/// the squares are summed per thread, per simdgroup and then over the
/// simdgroup partials, and every thread stores its four scaled values.
const NORM_SOURCE: &str = r#"
const uint gid = threadgroup_position_in_grid.x;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint simdgroup = simdgroup_index_in_threadgroup;

threadgroup float partials[32];
threadgroup float inverse[1];

const size_t offset = size_t(gid) * H + tid * 4;
const uint column = (gid % S) * H + tid * 4;

float values[4];
float acc = 0;
for (int i = 0; i < 4; i++) {
    values[i] = x[offset + i];
    acc += values[i] * values[i];
}
acc = simd_sum(acc);
if (simdgroup == 0) {
    partials[lane] = 0;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) {
    partials[simdgroup] = acc;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simdgroup == 0) {
    acc = simd_sum(partials[lane]);
    if (lane == 0) {
        inverse[0] = metal::precise::rsqrt(acc / axis[0] + eps[0]);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int i = 0; i < 4; i++) {
    const float unit = values[i] * inverse[0];
    out[offset + i] = static_cast<bfloat16_t>(unit * gain[column + i]);
}
"#;

/// One thread per `(row, hidden column)`. The gate and the gated value round
/// through bf16 exactly where the MLX chain stores them.
const MIX_SOURCE: &str = r#"
const uint index = thread_position_in_grid.x;
const uint row = index / H;
const uint column = index % H;
const size_t base = size_t(row) * S * H + column;
float acc = 0;
for (int s = 0; s < S; s++) {
    const size_t at = base + size_t(s) * H;
    const bfloat16_t gate = static_cast<bfloat16_t>(ax_hc_prefill_sigmoid(float(up[at])));
    const bfloat16_t gated = static_cast<bfloat16_t>(float(gate) * float(normalized[at]));
    acc += float(gated);
}
out[index] = static_cast<bfloat16_t>(acc / streams[0]);
"#;

/// One thread per packed element.
const WRITE_SOURCE: &str = r#"
const uint index = thread_position_in_grid.x;
const uint width = S * H;
const uint row = index / width;
const uint column = index % width;
const bfloat16_t delta = static_cast<bfloat16_t>(
    float(gate[row * S + column / H]) * float(branch[size_t(row) * H + column % H]));
out[index] = static_cast<bfloat16_t>(float(residual[index]) + float(delta));
"#;

#[cfg(test)]
mod tests {
    use super::*;

    const GEOMETRY: Geometry = Geometry {
        streams: 4,
        hidden: 2560,
    };
    const EPS: f32 = 1.0e-6;

    #[test]
    fn geometry_admits_only_single_threadgroup_norms() {
        assert_eq!(GEOMETRY.norm_threads(), Some(640));
        let narrow = Geometry {
            streams: 4,
            hidden: 128,
        };
        assert_eq!(narrow.norm_threads(), Some(32));
        for hidden in [0, 100, 2561, 4224, 8192] {
            let geometry = Geometry { streams: 4, hidden };
            assert_eq!(geometry.norm_threads(), None, "hidden {hidden}");
        }
        for streams in [0, 1, 9] {
            let geometry = Geometry {
                streams,
                hidden: 2560,
            };
            assert_eq!(geometry.norm_threads(), None, "streams {streams}");
        }
    }

    #[test]
    fn self_check_passes_at_the_flash_next_geometry() {
        assert_eq!(self_check(GEOMETRY, EPS), Some(true));
        let narrow = Geometry {
            streams: 2,
            hidden: 256,
        };
        assert_eq!(self_check(narrow, EPS), Some(true));
    }

    /// Prefill-sized tensors, including a row count that is no multiple of
    /// anything in the kernels, against the MLX chain bit for bit.
    #[test]
    fn kernels_match_the_mlx_ops_bit_for_bit_at_prefill_rows() {
        for (rows, phase) in [(64, 3u32), (97, 101), (209, 977)] {
            let width = GEOMETRY.streams * GEOMETRY.hidden;
            let packed = pattern(&[1, rows, width], phase, 0.6);
            let gain = mlx_sys::add(
                &astype(&pattern(&[width], phase + 5, 0.08), MlxDtype::Float32, None),
                &cached_scalar(1.0, MlxDtype::Float32),
                None,
            );
            let up = pattern(&[1, rows, width], phase + 9, 4.0);
            let gate = pattern(&[1, rows, GEOMETRY.streams], phase + 13, 1.0);
            let branch = pattern(&[1, rows, GEOMETRY.hidden], phase + 19, 2.0);

            let normalized = norm_reference(GEOMETRY, &packed, &gain, EPS);
            let norm = norm_launch(GEOMETRY, &packed, &gain, EPS).unwrap();
            assert_eq!(
                same_bits(&norm, &normalized),
                Some(true),
                "norm rows {rows}"
            );

            let mix = mix_launch(GEOMETRY, &up, &normalized).unwrap();
            let expected = mix_reference(GEOMETRY, &up, &normalized);
            assert_eq!(same_bits(&mix, &expected), Some(true), "mix rows {rows}");

            let write = write_launch(GEOMETRY, &packed, &gate, &branch).unwrap();
            let expected = write_reference(GEOMETRY, &packed, &gate, &branch);
            assert_eq!(
                same_bits(&write, &expected),
                Some(true),
                "write rows {rows}"
            );
        }
    }

    #[test]
    fn launches_reject_mismatched_shapes() {
        let width = GEOMETRY.streams * GEOMETRY.hidden;
        let packed = pattern(&[1, 2, width], 1, 0.5);
        let gain = astype(&pattern(&[width - 1], 2, 0.5), MlxDtype::Float32, None);
        assert!(norm_launch(GEOMETRY, &packed, &gain, EPS).is_none());
        let short = pattern(&[1, 1, width], 3, 0.5);
        assert!(mix_launch(GEOMETRY, &packed, &short).is_none());
        let gate = pattern(&[1, 2, GEOMETRY.streams], 4, 0.5);
        let branch = pattern(&[1, 2, GEOMETRY.hidden + 1], 5, 0.5);
        assert!(write_launch(GEOMETRY, &packed, &gate, &branch).is_none());
    }
}
