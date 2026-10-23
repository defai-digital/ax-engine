//! Native GDN input stretches: the decay/beta pair and the q/k/v input scale.
//!
//! Two elementwise stretches of the Flash Next GDN input path used to run as
//! separate graphs:
//!
//! * `ax_gdn_decay_beta_v1` folds the decay projection, the `dt_bias` add, the
//!   softplus (`log1p(exp(min(a, -a)))`), the `exp_a_log` product and the beta
//!   sigmoid into one dispatch, producing both `[.., heads]` rows at once.
//! * `ax_gdn_qkv_scale_v1` reads the merged convolution row `[q | k | v]` and
//!   writes the three normalized views, folding the per-head `sum^-0.5`
//!   inverse, the `sqrt(dk)` divisor and the value copy into one dispatch.
//!
//! Fail closed at every step, exactly like the hyper-connection module this one
//! mirrors. Only the documented dtypes dispatch; each kernel is built through
//! [`MlxMetalKernel::try_new`]; and a one-time per-dtype-combination self-check
//! compares the kernel with the eager chain, bit for bit. A build failure, a
//! rejected launch or any mismatch leaves the caller on its eager path for the
//! rest of the process.
//!
//! Kernel A takes the projection dtype (`TP`) as a template argument and reads
//! `dt_bias` in its own dtype, so every `(projection, bias)` dtype pair is
//! self-checked separately. Kernel B only depends on the convolution dtype.

use mlx_sys::ops::cached_scalar;
use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, astype, view,
};
use std::sync::OnceLock;

const DECAY_KERNEL_NAME: &str = "ax_gdn_decay_beta_v1";
const QKV_KERNEL_NAME: &str = "ax_gdn_qkv_scale_v1";

/// Threads per threadgroup; both kernels guard the tail threads.
const THREADS_PER_GROUP: i32 = 256;

/// Number of dtypes each kernel can read: bf16, f16, f32.
const DTYPE_SLOTS: usize = 3;

/// Verdict slots of the decay/beta kernel: one per `(projection, bias)` pair.
const DECAY_SLOTS: usize = DTYPE_SLOTS * DTYPE_SLOTS;

/// First verdict slot of the q/k/v kernel (one slot per convolution dtype).
const QKV_OFFSET: usize = DECAY_SLOTS;

/// Total verdict slots of both kernels.
const VERDICT_SLOTS: usize = QKV_OFFSET + DTYPE_SLOTS;

/// Self-check verdict of one dtype combination.
const UNKNOWN: u8 = 0;
const PASSED: u8 = 1;
const FAILED: u8 = 2;

/// Decay/beta self-check shape: `hv = 48` value heads over 85 rows.
const DECAY_CHECK_ROWS: usize = 85;
const DECAY_CHECK_HEADS: usize = 48;

/// q/k/v self-check geometry: `(hk, dk, hv, dv) = (2, 32, 6, 32)` over 5 rows.
const QKV_CHECK_KEY_HEADS: i32 = 2;
const QKV_CHECK_VALUE_HEADS: i32 = 6;
const QKV_CHECK_HEAD_DIM: i32 = 32;
const QKV_CHECK_ROWS: i32 = 5;

/// The eager divisor the q path applies after the inverse head sum
/// (`sqrt(32)`, the self-check head dim).
const QKV_CHECK_SQRT_DK: f32 = 5.656_854_5;

static DECAY_KERNEL: OnceLock<Option<MlxMetalKernel>> = OnceLock::new();
static QKV_KERNEL: OnceLock<Option<MlxMetalKernel>> = OnceLock::new();

#[cfg(not(test))]
static VERDICTS: [std::sync::atomic::AtomicU8; VERDICT_SLOTS] =
    [const { std::sync::atomic::AtomicU8::new(UNKNOWN) }; VERDICT_SLOTS];

#[cfg(test)]
thread_local! {
    // Per thread, so one test's injected self-check failure cannot disable a
    // kernel for a test running on another thread. Production keeps the
    // process-wide atomics above.
    static VERDICTS: [std::sync::atomic::AtomicU8; VERDICT_SLOTS] = const {
        [const { std::sync::atomic::AtomicU8::new(UNKNOWN) }; VERDICT_SLOTS]
    };
}

/// Slot of a single dtype, in bf16/f16/f32 order.
fn dtype_slot(dtype: MlxDtype) -> Option<usize> {
    match dtype {
        MlxDtype::Bfloat16 => Some(0),
        MlxDtype::Float16 => Some(1),
        MlxDtype::Float32 => Some(2),
        _ => None,
    }
}

/// Verdict slot of one `(projection, bias)` dtype pair of the decay/beta kernel.
fn decay_slot(projection: MlxDtype, bias: MlxDtype) -> Option<usize> {
    Some(dtype_slot(projection)? * DTYPE_SLOTS + dtype_slot(bias)?)
}

/// Verdict slot of one convolution dtype of the q/k/v kernel.
fn qkv_slot(dtype: MlxDtype) -> Option<usize> {
    Some(QKV_OFFSET + dtype_slot(dtype)?)
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

/// Geometry of the merged q/k/v convolution row.
///
/// `key_dim = hk * dk` and `value_dim = hv * dv`, with the value head dim equal
/// to the key head dim (`dv = dk`), which every Flash Next GDN configuration
/// uses.
#[cfg_attr(not(test), allow(dead_code))] // wired up by the caller in a follow-up
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QkvGeometry {
    pub(crate) key_dim: i32,
    pub(crate) head_dim: i32,
    pub(crate) value_dim: i32,
}

/// Native decay and beta for a `[.., heads]` projection pair.
///
/// Returns `(decay, beta)`, both `f32` and shaped like `decay_proj`. `dt_bias`
/// and `exp_a_log` carry one value per head; `exp_a_log` must be `f32` while
/// `dt_bias` keeps its own dtype. Any unsupported dtype mix, shape mismatch or
/// failed verdict returns `None`, leaving the caller on its eager path.
#[cfg_attr(not(test), allow(dead_code))] // wired up by the caller in a follow-up
pub(crate) fn decay_beta(
    decay_proj: &MlxArray,
    dt_bias: &MlxArray,
    exp_a_log: &MlxArray,
    beta_proj: &MlxArray,
) -> Option<(MlxArray, MlxArray)> {
    let dtype = decay_proj.dtype();
    if beta_proj.dtype() != dtype {
        return None;
    }
    if exp_a_log.dtype() != MlxDtype::Float32 {
        return None;
    }
    let shape = decay_proj.shape();
    let &heads = shape.last()?;
    if heads <= 0 || beta_proj.shape() != shape {
        return None;
    }
    if !one_value_per_head(dt_bias, heads) || !one_value_per_head(exp_a_log, heads) {
        return None;
    }
    let elements = element_count(&shape)?;
    let index = decay_slot(dtype, dt_bias.dtype())?;
    if !decay_supported(dtype, dt_bias.dtype(), index) {
        return None;
    }
    match launch_decay(
        decay_proj, dt_bias, exp_a_log, beta_proj, dtype, heads, elements,
    ) {
        Some(outputs) => Some(outputs),
        None => {
            // This pair already passed the self-check, so the rejection is a
            // runtime failure rather than an unsupported configuration.
            // Disable it instead of retrying a failing launch on every read.
            record_verdict(index, FAILED);
            tracing::warn!(
                ?dtype,
                bias = ?dt_bias.dtype(),
                heads,
                "Flash Next GDN decay/beta kernel rejected its launch; using eager ops"
            );
            None
        }
    }
}

/// Native q/k/v input scale over a merged `[batch, seq, 2 * key_dim + value_dim]`
/// convolution row.
///
/// Returns `(q, k, v)`: `q`/`k` as `f32 [batch, seq, key_dim / head_dim,
/// head_dim]` and `v` as `f32 [batch, seq, value_dim / head_dim, head_dim]`.
/// `sum_q`/`sum_k` hold one `f32` inverse-sum input per key head and row, and
/// `sqrt_dk` is the `f32 [1]` divisor. Any unsupported dtype, shape mismatch or
/// failed verdict returns `None`, leaving the caller on its eager path.
#[cfg_attr(not(test), allow(dead_code))] // wired up by the caller in a follow-up
pub(crate) fn qkv_scale(
    conv: &MlxArray,
    sum_q: &MlxArray,
    sum_k: &MlxArray,
    sqrt_dk: &MlxArray,
    geometry: QkvGeometry,
) -> Option<(MlxArray, MlxArray, MlxArray)> {
    let QkvGeometry {
        key_dim,
        head_dim,
        value_dim,
    } = geometry;
    if key_dim <= 0 || head_dim <= 0 || value_dim <= 0 {
        return None;
    }
    if key_dim % head_dim != 0 || value_dim % head_dim != 0 {
        return None;
    }
    let conv_dim = key_dim.checked_mul(2)?.checked_add(value_dim)?;
    let conv_shape = conv.shape();
    let (batch, seq) = match conv_shape.as_slice() {
        [batch, seq, dim] if *dim == conv_dim => (*batch, *seq),
        [rows, dim] if *dim == conv_dim => (1, *rows),
        _ => return None,
    };
    if batch <= 0 || seq <= 0 {
        return None;
    }
    let rows = batch.checked_mul(seq)?;
    let key_heads = key_dim / head_dim;
    let value_heads = value_dim / head_dim;
    let sum_count = rows.checked_mul(key_heads)?;
    if sum_q.dtype() != MlxDtype::Float32
        || sum_k.dtype() != MlxDtype::Float32
        || sqrt_dk.dtype() != MlxDtype::Float32
        || element_count(&sum_q.shape())? != sum_count
        || element_count(&sum_k.shape())? != sum_count
        || element_count(&sqrt_dk.shape())? != 1
    {
        return None;
    }
    let dtype = conv.dtype();
    let total = rows.checked_mul(conv_dim)?;
    let index = qkv_slot(dtype)?;
    if !qkv_supported(dtype, index) {
        return None;
    }
    match launch_qkv(
        conv,
        sum_q,
        sum_k,
        sqrt_dk,
        geometry,
        batch,
        seq,
        conv_dim,
        key_heads,
        value_heads,
        total,
    ) {
        Some(outputs) => Some(outputs),
        None => {
            record_verdict(index, FAILED);
            tracing::warn!(
                ?dtype,
                key_dim,
                value_dim,
                "Flash Next GDN q/k/v scale kernel rejected its launch; using eager ops"
            );
            None
        }
    }
}

/// Element count of a shape, or `None` on overflow.
fn element_count(shape: &[i32]) -> Option<i32> {
    shape
        .iter()
        .try_fold(1i32, |total, &dim| total.checked_mul(dim))
}

/// Whether `array` holds exactly one value per head, in any leading shape.
fn one_value_per_head(array: &MlxArray, heads: i32) -> bool {
    array.shape().last() == Some(&heads) && element_count(&array.shape()) == Some(heads)
}

/// Whether the decay/beta kernel is known to reproduce the eager chain for the
/// `(projection, bias)` dtype pair with verdict slot `index`.
///
/// The first call per pair runs the self-check; the recorded verdict is
/// monotone, so a failure disables that pair permanently even when two threads
/// race the first check.
fn decay_supported(projection: MlxDtype, bias: MlxDtype, index: usize) -> bool {
    match verdict(index) {
        PASSED => true,
        FAILED => false,
        _ => {
            let passed = self_check_or_injected_failure(|| decay_self_check(projection, bias));
            record_verdict(index, if passed { PASSED } else { FAILED }) == PASSED
        }
    }
}

/// Whether the q/k/v kernel is known to reproduce the eager expressions for
/// `dtype` with verdict slot `index`, running the self-check on first use.
fn qkv_supported(dtype: MlxDtype, index: usize) -> bool {
    match verdict(index) {
        PASSED => true,
        FAILED => false,
        _ => {
            let passed = self_check_or_injected_failure(|| qkv_self_check(dtype));
            record_verdict(index, if passed { PASSED } else { FAILED }) == PASSED
        }
    }
}

/// Self-check with the test-only failure switch in front of it.
fn self_check_or_injected_failure(check: impl FnOnce() -> bool) -> bool {
    #[cfg(test)]
    SELF_CHECK_RUNS.with(|runs| runs.set(runs.get() + 1));
    #[cfg(test)]
    let injected = take_injected_failure();
    #[cfg(not(test))]
    let injected = false;
    if injected {
        tracing::warn!(
            "Flash Next GDN native kernel self-check was injected to fail; using eager ops"
        );
        return false;
    }
    // An eager-op panic (a rejected graph) must disable the kernel rather than
    // escape from a decode step, exactly like a bit mismatch.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(check)).unwrap_or(false)
}

/// Raw bits of an evaluated float array, for bit-exact comparisons.
///
/// Reinterprets the buffer (`view`) instead of converting, so NaN payloads,
/// signed zeros and subnormals survive unchanged. Returns `None` when the array
/// cannot be evaluated or has no supported width.
fn raw_bits(array: &MlxArray) -> Option<Vec<u32>> {
    // A strided view (a reshaped slice) must be materialised before its raw
    // buffer is read; the copy preserves every bit.
    let array = &mlx_sys::contiguous(array, None);
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
fn array_from_f32(data: &[f32], shape: &[i32], dtype: MlxDtype) -> MlxArray {
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

fn decay_kernel() -> Option<&'static MlxMetalKernel> {
    DECAY_KERNEL
        .get_or_init(|| {
            #[cfg(test)]
            KERNEL_BUILDS.with(|builds| builds.set(builds.get() + 1));
            match MlxMetalKernel::try_new(
                DECAY_KERNEL_NAME,
                &["decay_proj", "dt_bias", "exp_a_log", "beta_proj"],
                &["decay", "beta"],
                DECAY_SOURCE,
                SIGMOID_LOG1P_HEADER,
                true,
            ) {
                Ok(kernel) => Some(kernel),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "Flash Next GDN decay/beta kernel failed to build; using eager ops"
                    );
                    None
                }
            }
        })
        .as_ref()
}

fn qkv_kernel() -> Option<&'static MlxMetalKernel> {
    QKV_KERNEL
        .get_or_init(|| {
            #[cfg(test)]
            KERNEL_BUILDS.with(|builds| builds.set(builds.get() + 1));
            match MlxMetalKernel::try_new(
                QKV_KERNEL_NAME,
                &["conv", "sum_q", "sum_k", "sqrt_dk"],
                &["q", "k", "v"],
                QKV_SOURCE,
                "",
                true,
            ) {
                Ok(kernel) => Some(kernel),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "Flash Next GDN q/k/v scale kernel failed to build; using eager ops"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// One raw decay/beta launch, without the self-check.
fn launch_decay(
    decay_proj: &MlxArray,
    dt_bias: &MlxArray,
    exp_a_log: &MlxArray,
    beta_proj: &MlxArray,
    dtype: MlxDtype,
    heads: i32,
    elements: i32,
) -> Option<(MlxArray, MlxArray)> {
    #[cfg(test)]
    LAUNCHES.with(|launches| launches.set(launches.get() + 1));
    let kernel = decay_kernel()?;
    #[cfg(test)]
    if take_injected_launch_failure() {
        return None;
    }
    let shape = decay_proj.shape();
    let mut outputs = kernel
        .try_apply_with_template(
            &[decay_proj, dt_bias, exp_a_log, beta_proj],
            &[
                KernelOutputSpec {
                    shape: shape.clone(),
                    dtype: MlxDtype::Float32,
                },
                KernelOutputSpec {
                    shape,
                    dtype: MlxDtype::Float32,
                },
            ],
            &[
                KernelTemplateArg::Dtype { name: "TP", dtype },
                KernelTemplateArg::Int {
                    name: "Heads",
                    value: heads,
                },
                KernelTemplateArg::Int {
                    name: "Elements",
                    value: elements,
                },
            ],
            (elements, 1, 1),
            (THREADS_PER_GROUP, 1, 1),
            None,
        )
        .ok()?;
    if outputs.len() != 2 {
        return None;
    }
    let beta = outputs.pop()?;
    let decay = outputs.pop()?;
    Some((decay, beta))
}

/// One raw q/k/v scale launch, without the self-check.
#[allow(clippy::too_many_arguments)]
fn launch_qkv(
    conv: &MlxArray,
    sum_q: &MlxArray,
    sum_k: &MlxArray,
    sqrt_dk: &MlxArray,
    geometry: QkvGeometry,
    batch: i32,
    seq: i32,
    conv_dim: i32,
    key_heads: i32,
    value_heads: i32,
    total: i32,
) -> Option<(MlxArray, MlxArray, MlxArray)> {
    #[cfg(test)]
    LAUNCHES.with(|launches| launches.set(launches.get() + 1));
    let kernel = qkv_kernel()?;
    #[cfg(test)]
    if take_injected_launch_failure() {
        return None;
    }
    let q_shape = vec![batch, seq, key_heads, geometry.head_dim];
    let v_shape = vec![batch, seq, value_heads, geometry.head_dim];
    let mut outputs = kernel
        .try_apply_with_template(
            &[conv, sum_q, sum_k, sqrt_dk],
            &[
                KernelOutputSpec {
                    shape: q_shape.clone(),
                    dtype: MlxDtype::Float32,
                },
                KernelOutputSpec {
                    shape: q_shape,
                    dtype: MlxDtype::Float32,
                },
                KernelOutputSpec {
                    shape: v_shape,
                    dtype: MlxDtype::Float32,
                },
            ],
            &[
                KernelTemplateArg::Int {
                    name: "KeyDim",
                    value: geometry.key_dim,
                },
                KernelTemplateArg::Int {
                    name: "Dk",
                    value: geometry.head_dim,
                },
                KernelTemplateArg::Int {
                    name: "ValueDim",
                    value: geometry.value_dim,
                },
                KernelTemplateArg::Int {
                    name: "Conv",
                    value: conv_dim,
                },
                KernelTemplateArg::Int {
                    name: "Total",
                    value: total,
                },
            ],
            (total, 1, 1),
            (THREADS_PER_GROUP, 1, 1),
            None,
        )
        .ok()?;
    if outputs.len() != 3 {
        return None;
    }
    let v = outputs.pop()?;
    let k = outputs.pop()?;
    let q = outputs.pop()?;
    Some((q, k, v))
}

/// The eager decay/beta chain the kernel must reproduce bit for bit.
fn decay_reference(
    decay_proj: &MlxArray,
    dt_bias: &MlxArray,
    exp_a_log: &MlxArray,
) -> (MlxArray, MlxArray) {
    let projection = astype(decay_proj, MlxDtype::Float32, None);
    let summed = mlx_sys::add(&projection, &astype(dt_bias, MlxDtype::Float32, None), None);
    let negative = mlx_sys::negative(&summed, None);
    let exponent = mlx_sys::exp(&mlx_sys::minimum(&summed, &negative, None), None);
    let softplus = mlx_sys::add(
        &mlx_sys::maximum(&summed, &cached_scalar(0.0, MlxDtype::Float32), None),
        &mlx_sys::log1p(&exponent, None),
        None,
    );
    let scaled = mlx_sys::multiply(exp_a_log, &softplus, None);
    let decay = mlx_sys::exp(&mlx_sys::negative(&scaled, None), None);
    let beta = astype(
        &astype(
            &mlx_sys::sigmoid(&projection, None),
            decay_proj.dtype(),
            None,
        ),
        MlxDtype::Float32,
        None,
    );
    (decay, beta)
}

/// The eager q/k/v expressions the kernel must reproduce bit for bit.
fn qkv_reference(
    conv: &MlxArray,
    sum_q: &MlxArray,
    sum_k: &MlxArray,
    sqrt_dk: &MlxArray,
    geometry: QkvGeometry,
    batch: i32,
    seq: i32,
) -> (MlxArray, MlxArray, MlxArray) {
    let QkvGeometry {
        key_dim,
        head_dim,
        value_dim,
    } = geometry;
    let conv_dim = 2 * key_dim + value_dim;
    let key_heads = key_dim / head_dim;
    let value_heads = value_dim / head_dim;
    let key_shape = [batch, seq, key_heads, head_dim];
    let value_shape = [batch, seq, value_heads, head_dim];
    let xq = mlx_sys::reshape(
        &mlx_sys::slice_last_dim(conv, 0, key_dim, None),
        &key_shape,
        None,
    );
    let xk = mlx_sys::reshape(
        &mlx_sys::slice_last_dim(conv, key_dim, 2 * key_dim, None),
        &key_shape,
        None,
    );
    let xv = mlx_sys::reshape(
        &mlx_sys::slice_last_dim(conv, 2 * key_dim, conv_dim, None),
        &value_shape,
        None,
    );
    let inverse_q = mlx_sys::power(sum_q, &cached_scalar(-0.5, MlxDtype::Float32), None);
    let inverse_k = mlx_sys::power(sum_k, &cached_scalar(-0.5, MlxDtype::Float32), None);
    let q = mlx_sys::divide(
        &mlx_sys::multiply(&astype(&xq, MlxDtype::Float32, None), &inverse_q, None),
        sqrt_dk,
        None,
    );
    let k = mlx_sys::multiply(&astype(&xk, MlxDtype::Float32, None), &inverse_k, None);
    let v = astype(&xv, MlxDtype::Float32, None);
    (q, k, v)
}

/// Compare the decay/beta kernel with the eager chain on the special-value
/// sample, for one `(projection, bias)` dtype pair.
fn decay_self_check(projection: MlxDtype, bias: MlxDtype) -> bool {
    let heads = DECAY_CHECK_HEADS as i32;
    let elements = DECAY_CHECK_ROWS * DECAY_CHECK_HEADS;
    let shape = [1, DECAY_CHECK_ROWS as i32, heads];
    let projection_values = self_check_values(elements, 0);
    let decay_proj = array_from_f32(&projection_values, &shape, projection);
    let beta_proj = array_from_f32(&projection_values, &shape, projection);
    let bias_values = self_check_values(DECAY_CHECK_HEADS, 1);
    let dt_bias = array_from_f32(&bias_values, &[heads], bias);
    let exp_a_log = array_from_f32(
        &decay_parameter_values(DECAY_CHECK_HEADS),
        &[heads],
        MlxDtype::Float32,
    );
    if mlx_sys::try_eval(&[&decay_proj, &beta_proj, &dt_bias, &exp_a_log]).is_err() {
        warn_decay_self_check(projection, bias);
        return false;
    }

    let (decay_expected, beta_expected) = decay_reference(&decay_proj, &dt_bias, &exp_a_log);
    let Some((decay, beta)) = launch_decay(
        &decay_proj,
        &dt_bias,
        &exp_a_log,
        &beta_proj,
        projection,
        heads,
        elements as i32,
    ) else {
        warn_decay_self_check(projection, bias);
        return false;
    };

    let (Some(decay_actual), Some(beta_actual), Some(decay_target), Some(beta_target)) = (
        raw_bits(&decay),
        raw_bits(&beta),
        raw_bits(&decay_expected),
        raw_bits(&beta_expected),
    ) else {
        warn_decay_self_check(projection, bias);
        return false;
    };
    let matches = decay_actual.len() == elements
        && beta_actual.len() == elements
        && decay_actual == decay_target
        && beta_actual == beta_target;
    if !matches {
        warn_decay_self_check(projection, bias);
    }
    matches
}

fn warn_decay_self_check(projection: MlxDtype, bias: MlxDtype) {
    tracing::warn!(
        ?projection,
        ?bias,
        "Flash Next GDN decay/beta kernel does not match the eager chain; using eager ops"
    );
}

/// Compare the q/k/v kernel with the eager expressions on the special-value
/// sample, for one convolution dtype.
fn qkv_self_check(dtype: MlxDtype) -> bool {
    let key_dim = QKV_CHECK_KEY_HEADS * QKV_CHECK_HEAD_DIM;
    let value_dim = QKV_CHECK_VALUE_HEADS * QKV_CHECK_HEAD_DIM;
    let conv_dim = 2 * key_dim + value_dim;
    let geometry = QkvGeometry {
        key_dim,
        head_dim: QKV_CHECK_HEAD_DIM,
        value_dim,
    };
    let conv = array_from_f32(
        &self_check_values((QKV_CHECK_ROWS * conv_dim) as usize, 3),
        &[1, QKV_CHECK_ROWS, conv_dim],
        dtype,
    );
    let sum_count = (QKV_CHECK_ROWS * QKV_CHECK_KEY_HEADS) as usize;
    let sum_shape = [1, QKV_CHECK_ROWS, QKV_CHECK_KEY_HEADS, 1];
    let sum_q = array_from_f32(
        &inverse_sum_values(sum_count, 5),
        &sum_shape,
        MlxDtype::Float32,
    );
    let sum_k = array_from_f32(
        &inverse_sum_values(sum_count, 7),
        &sum_shape,
        MlxDtype::Float32,
    );
    let sqrt_dk = array_from_f32(&[QKV_CHECK_SQRT_DK], &[1], MlxDtype::Float32);
    if mlx_sys::try_eval(&[&conv, &sum_q, &sum_k, &sqrt_dk]).is_err() {
        warn_qkv_self_check(dtype);
        return false;
    }

    let (q_expected, k_expected, v_expected) =
        qkv_reference(&conv, &sum_q, &sum_k, &sqrt_dk, geometry, 1, QKV_CHECK_ROWS);
    let Some((q, k, v)) = launch_qkv(
        &conv,
        &sum_q,
        &sum_k,
        &sqrt_dk,
        geometry,
        1,
        QKV_CHECK_ROWS,
        conv_dim,
        QKV_CHECK_KEY_HEADS,
        QKV_CHECK_VALUE_HEADS,
        QKV_CHECK_ROWS * conv_dim,
    ) else {
        warn_qkv_self_check(dtype);
        return false;
    };

    let (
        Some(q_actual),
        Some(k_actual),
        Some(v_actual),
        Some(q_target),
        Some(k_target),
        Some(v_target),
    ) = (
        raw_bits(&q),
        raw_bits(&k),
        raw_bits(&v),
        raw_bits(&q_expected),
        raw_bits(&k_expected),
        raw_bits(&v_expected),
    )
    else {
        warn_qkv_self_check(dtype);
        return false;
    };
    let matches = q_actual == q_target && k_actual == k_target && v_actual == v_target;
    if !matches {
        warn_qkv_self_check(dtype);
    }
    matches
}

fn warn_qkv_self_check(dtype: MlxDtype) {
    tracing::warn!(
        ?dtype,
        "Flash Next GDN q/k/v scale kernel does not match the eager expressions; using eager ops"
    );
}

/// Deterministic sample: every bf16/f16 special class plus ordinary magnitudes,
/// so rounding boundaries and non-finite payloads are compared.
///
/// `salt` rotates the special-value table and seeds the pseudo-random tail, so
/// samples of different lengths still cover every class.
fn self_check_values(count: usize, salt: u64) -> Vec<f32> {
    const SPECIAL: [f32; 32] = [
        0.0,
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
        3.4e30,
        -3.4e30,
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
    let offset = (salt as usize) % SPECIAL.len();
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ salt.wrapping_mul(0x2545_F491_4F6C_DD1D);
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        if index < SPECIAL.len() {
            values.push(SPECIAL[(offset + index) % SPECIAL.len()]);
            continue;
        }
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        if index % 2 == 1 {
            values.push(SPECIAL[(offset + index) % SPECIAL.len()]);
            continue;
        }
        let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
        let exponent = (index % 24) as i32 - 12;
        values.push((unit * 2.0 - 1.0) * 2f32.powi(exponent));
    }
    values
}

/// Per-head `exp_a_log` sample: zero, tiny, one, the configured value, huge and
/// negative magnitudes.
fn decay_parameter_values(count: usize) -> Vec<f32> {
    const PARAMETERS: [f32; 16] = [
        0.0, -0.0, 1e-30, 1.0, 13.7, 1e30, -1.0, -13.7, 0.0625, -4.0, 4.0, 1e-45, 88.0, -88.0, 2.0,
        0.5,
    ];
    (0..count)
        .map(|index| PARAMETERS[index % PARAMETERS.len()])
        .collect()
}

/// Per-head inverse-sum sample: zero, subnormal, large, infinite and NaN values
/// the `pow(x, -0.5)` path has to reproduce.
fn inverse_sum_values(count: usize, salt: u64) -> Vec<f32> {
    const SPECIAL: [f32; 17] = [
        0.0,
        -0.0,
        1.0,
        f32::from_bits(0x0000_0001),
        1e-45,
        1e-30,
        1e30,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0000),
        f32::from_bits(0xffc1_2345),
        4.0,
        0.25,
        12345.0,
        -1.0,
        -3.0,
    ];
    let offset = (salt as usize) % SPECIAL.len();
    (0..count)
        .map(|index| SPECIAL[(offset + index) % SPECIAL.len()])
        .collect()
}

#[cfg(test)]
thread_local! {
    static INJECTED_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Make the next self-check on this thread fail, and forget any earlier verdict
/// so the self-check actually runs again.
#[cfg(test)]
fn fail_next_self_check_for_test() {
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
fn kernel_builds_for_test() -> u64 {
    KERNEL_BUILDS.with(std::cell::Cell::get)
}

/// Self-checks entered on this thread (test instrumentation).
#[cfg(test)]
fn self_check_runs_for_test() -> u64 {
    SELF_CHECK_RUNS.with(std::cell::Cell::get)
}

/// Launches attempted on this thread (test instrumentation).
#[cfg(test)]
fn launches_for_test() -> u64 {
    LAUNCHES.with(std::cell::Cell::get)
}

/// Make the next kernel launch on this thread fail, which must disable the
/// dtype combination instead of retrying it on every later read.
#[cfg(test)]
fn fail_next_launch_for_test() {
    INJECTED_LAUNCH_FAILURE.with(|failure| failure.set(true));
}

/// Whether the injected launch failure is still pending, i.e. the launch under
/// test never reached the kernel.
#[cfg(test)]
fn launch_failure_pending_for_test() -> bool {
    INJECTED_LAUNCH_FAILURE.with(std::cell::Cell::get)
}

#[cfg(test)]
fn take_injected_launch_failure() -> bool {
    INJECTED_LAUNCH_FAILURE.with(std::cell::Cell::take)
}

/// Forget every recorded verdict on this thread, so the next read re-runs the
/// self-check. Test-only: production verdicts are monotone.
#[cfg(test)]
fn reset_verdicts_for_test() {
    VERDICTS.with(|verdicts| {
        for slot in verdicts {
            slot.store(UNKNOWN, std::sync::atomic::Ordering::Release);
        }
    });
}

/// Whether the decay/beta pair is disabled by a failed check or a rejected
/// launch (test instrumentation).
#[cfg(test)]
fn decay_verdict_failed_for_test(projection: MlxDtype, bias: MlxDtype) -> bool {
    decay_slot(projection, bias).is_some_and(|index| verdict(index) == FAILED)
}

/// Whether the q/k/v kernel is disabled for `dtype` (test instrumentation).
#[cfg(test)]
fn qkv_verdict_failed_for_test(dtype: MlxDtype) -> bool {
    qkv_slot(dtype).is_some_and(|index| verdict(index) == FAILED)
}

/// The sigmoid and log1p helpers, spelled so the JIT keeps the precise
/// exponential. Both are the eager MLX definitions the kernel has to match:
/// `sigmoid(x) = x < 0 ? 1 / (1 + exp(|x|)) : 1 - 1 / (1 + exp(|x|))` and
/// `log1p(x) = x * (log(1 + x) / ((1 + x) - 1))`, with the two MLX edge
/// branches.
///
/// The body is a kernel function body, so these helpers have to live in the
/// header MLX prepends to the generated signature.
const SIGMOID_LOG1P_HEADER: &str = r#"
inline float ax_sigmoid(float x) {
  float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
  return (x < 0.0f) ? y : 1.0f - y;
}
inline float ax_log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == Limits<float>::max) { return Limits<float>::max; }
  if (xp1 == 1.0f) { return x; }
  return x * (metal::precise::log(xp1) / (xp1 - 1.0f));
}
"#;

const DECAY_SOURCE: &str = r#"
#pragma clang fp contract(off)
const size_t i = thread_position_in_grid.x;
if (i >= Elements) return;
const size_t h = i % Heads;
const float a = float(decay_proj[i]) + float(dt_bias[h]);
const float neg = -a;
const float m = metal::isnan(a) ? a : (a < neg ? a : neg);
const float e = metal::precise::exp(m);
const float l1 = ax_log1p(e);
const float mx = metal::isnan(a) ? a : (a > 0.0f ? a : 0.0f);
const float sp = mx + l1;
decay[i] = metal::precise::exp(-(exp_a_log[h] * sp));
const float s = ax_sigmoid(float(beta_proj[i]));
beta[i] = float(TP(s));
"#;

const QKV_SOURCE: &str = r#"
#pragma clang fp contract(off)
const size_t t = thread_position_in_grid.x;
if (t >= Total) return;
const size_t row = t / Conv;
const size_t col = t % Conv;
const float x = float(conv[t]);
if (col < KeyDim) {
  const size_t head = col / Dk;
  const float r = metal::precise::pow(sum_q[row * (KeyDim / Dk) + head], -0.5f);
  q[row * KeyDim + col] = (x * r) / sqrt_dk[0];
} else if (col < 2 * KeyDim) {
  const size_t kc = col - KeyDim;
  const size_t head = kc / Dk;
  const float r = metal::precise::pow(sum_k[row * (KeyDim / Dk) + head], -0.5f);
  k[row * KeyDim + kc] = x * r;
} else {
  v[row * ValueDim + (col - 2 * KeyDim)] = x;
}
"#;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use mlx_sys::{eval, reshape};

    /// Every `(projection, bias)` dtype pair the decay/beta kernel can read.
    const DECAY_DTYPE_PAIRS: [(MlxDtype, MlxDtype); 9] = [
        (MlxDtype::Bfloat16, MlxDtype::Bfloat16),
        (MlxDtype::Bfloat16, MlxDtype::Float16),
        (MlxDtype::Bfloat16, MlxDtype::Float32),
        (MlxDtype::Float16, MlxDtype::Bfloat16),
        (MlxDtype::Float16, MlxDtype::Float16),
        (MlxDtype::Float16, MlxDtype::Float32),
        (MlxDtype::Float32, MlxDtype::Bfloat16),
        (MlxDtype::Float32, MlxDtype::Float16),
        (MlxDtype::Float32, MlxDtype::Float32),
    ];

    /// Convolution dtypes the q/k/v kernel can read.
    const QKV_DTYPES: [MlxDtype; 3] = [MlxDtype::Bfloat16, MlxDtype::Float16, MlxDtype::Float32];

    /// Every bit pattern of a 16-bit float dtype.
    const PATTERNS: usize = 1 << 16;

    /// Elements per chunk of the exhaustive sweeps.
    const EXHAUSTIVE_CHUNK: u32 = 1 << 24;

    /// The two geometries the q/k/v comparison covers: the Flash Next decode and
    /// prefill shapes, and a small geometry that exercises uneven head counts.
    fn qkv_geometries() -> [QkvGeometry; 2] {
        [
            QkvGeometry {
                key_dim: 16 * 128,
                head_dim: 128,
                value_dim: 48 * 128,
            },
            QkvGeometry {
                key_dim: 2 * 32,
                head_dim: 32,
                value_dim: 6 * 32,
            },
        ]
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

    /// One decay/beta comparison at the given shape.
    fn check_decay(projection: MlxDtype, bias: MlxDtype, heads: usize, rows: usize, salt: u64) {
        let elements = rows * heads;
        let shape = [1, rows as i32, heads as i32];
        let projection_values = self_check_values(elements, salt);
        let decay_proj = array_from_f32(&projection_values, &shape, projection);
        let beta_proj = array_from_f32(&projection_values, &shape, projection);
        let dt_bias = array_from_f32(&self_check_values(heads, salt + 1), &[heads as i32], bias);
        let exp_a_log = array_from_f32(
            &decay_parameter_values(heads),
            &[heads as i32],
            MlxDtype::Float32,
        );

        let (decay, beta) = decay_beta(&decay_proj, &dt_bias, &exp_a_log, &beta_proj)
            .unwrap_or_else(|| {
                panic!("native decay/beta {projection:?} {bias:?} heads {heads} rows {rows}")
            });
        assert_eq!(decay.shape(), shape, "{projection:?} {bias:?} decay shape");
        assert_eq!(beta.shape(), shape, "{projection:?} {bias:?} beta shape");
        assert_eq!(decay.dtype(), MlxDtype::Float32);
        assert_eq!(beta.dtype(), MlxDtype::Float32);

        let (decay_reference, beta_reference) = decay_reference(&decay_proj, &dt_bias, &exp_a_log);
        assert_eq!(
            raw_bits(&decay).expect("decay bits"),
            raw_bits(&decay_reference).expect("decay reference bits"),
            "decay {projection:?} {bias:?} heads {heads} rows {rows}"
        );
        assert_eq!(
            raw_bits(&beta).expect("beta bits"),
            raw_bits(&beta_reference).expect("beta reference bits"),
            "beta {projection:?} {bias:?} heads {heads} rows {rows}"
        );
    }

    #[test]
    fn native_decay_beta_matches_eager_on_adversarial_inputs() {
        for (projection, bias) in DECAY_DTYPE_PAIRS {
            for heads in [48usize, 8, 3] {
                for rows in 1..=8usize {
                    check_decay(projection, bias, heads, rows, (rows * 11 + heads) as u64);
                }
            }
        }
    }

    /// One q/k/v comparison at the given geometry.
    fn check_qkv(dtype: MlxDtype, geometry: QkvGeometry, rows: i32, salt: u64) {
        let conv_dim = 2 * geometry.key_dim + geometry.value_dim;
        let key_heads = geometry.key_dim / geometry.head_dim;
        let value_heads = geometry.value_dim / geometry.head_dim;
        let conv = array_from_f32(
            &self_check_values((rows * conv_dim) as usize, salt),
            &[1, rows, conv_dim],
            dtype,
        );
        let sum_shape = [1, rows, key_heads, 1];
        let sum_count = (rows * key_heads) as usize;
        let sum_q = array_from_f32(
            &inverse_sum_values(sum_count, salt + 1),
            &sum_shape,
            MlxDtype::Float32,
        );
        let sum_k = array_from_f32(
            &inverse_sum_values(sum_count, salt + 2),
            &sum_shape,
            MlxDtype::Float32,
        );
        let sqrt_dk = array_from_f32(&[QKV_CHECK_SQRT_DK], &[1], MlxDtype::Float32);

        let (q, k, v) = qkv_scale(&conv, &sum_q, &sum_k, &sqrt_dk, geometry)
            .unwrap_or_else(|| panic!("native q/k/v scale {dtype:?} {geometry:?} rows {rows}"));
        assert_eq!(q.shape(), [1, rows, key_heads, geometry.head_dim]);
        assert_eq!(k.shape(), [1, rows, key_heads, geometry.head_dim]);
        assert_eq!(v.shape(), [1, rows, value_heads, geometry.head_dim]);
        assert_eq!(q.dtype(), MlxDtype::Float32);
        assert_eq!(v.shape()[2], value_heads);

        let (q_reference, k_reference, v_reference) =
            qkv_reference(&conv, &sum_q, &sum_k, &sqrt_dk, geometry, 1, rows);
        assert_eq!(
            raw_bits(&q).expect("q bits"),
            raw_bits(&q_reference).expect("q reference bits"),
            "q {dtype:?} {geometry:?} rows {rows}"
        );
        assert_eq!(
            raw_bits(&k).expect("k bits"),
            raw_bits(&k_reference).expect("k reference bits"),
            "k {dtype:?} {geometry:?} rows {rows}"
        );
        assert_eq!(
            raw_bits(&v).expect("v bits"),
            raw_bits(&v_reference).expect("v reference bits"),
            "v {dtype:?} {geometry:?} rows {rows}"
        );
    }

    #[test]
    fn native_qkv_scale_matches_eager_on_adversarial_inputs() {
        for dtype in QKV_DTYPES {
            for geometry in qkv_geometries() {
                for rows in 1..=8i32 {
                    check_qkv(dtype, geometry, rows, (rows * 13) as u64);
                }
            }
        }
    }

    /// Self-check shaped decay/beta inputs at one dtype pair.
    fn decay_check_inputs(projection: MlxDtype, bias: MlxDtype) -> (MlxArray, MlxArray, MlxArray) {
        let heads = DECAY_CHECK_HEADS as i32;
        let shape = [1, DECAY_CHECK_ROWS as i32, heads];
        let decay_proj = array_from_f32(
            &self_check_values(DECAY_CHECK_ROWS * DECAY_CHECK_HEADS, 0),
            &shape,
            projection,
        );
        let dt_bias = array_from_f32(&self_check_values(DECAY_CHECK_HEADS, 1), &[heads], bias);
        let exp_a_log = array_from_f32(
            &decay_parameter_values(DECAY_CHECK_HEADS),
            &[heads],
            MlxDtype::Float32,
        );
        (decay_proj, dt_bias, exp_a_log)
    }

    /// The small q/k/v inputs the rejection and injection tests reuse.
    fn qkv_check_inputs() -> (MlxArray, MlxArray, MlxArray, QkvGeometry) {
        let geometry = QkvGeometry {
            key_dim: 4,
            head_dim: 2,
            value_dim: 2,
        };
        let conv = array_from_f32(&[1.0f32; 10], &[1, 1, 10], MlxDtype::Float16);
        let sum = array_from_f32(&[1.0, 4.0], &[1, 1, 2, 1], MlxDtype::Float32);
        let sqrt_dk = array_from_f32(&[2.0], &[1], MlxDtype::Float32);
        (conv, sum, sqrt_dk, geometry)
    }

    #[test]
    fn unsupported_inputs_return_none() {
        // The final asserts need an unrecorded verdict, whatever ran before.
        reset_verdicts_for_test();
        let launches = launches_for_test();
        let values = vec![0.25f32; 8];
        let bf16 = array_from_f32(&values, &[1, 2, 4], MlxDtype::Bfloat16);
        let f16 = array_from_f32(&values, &[1, 2, 4], MlxDtype::Float16);
        let bias = array_from_f32(&[0.5, 0.5, 0.5, 0.5], &[4], MlxDtype::Float32);
        let exp_a_log = array_from_f32(&[1.0, 1.0, 1.0, 1.0], &[4], MlxDtype::Float32);

        assert!(
            decay_beta(&bf16, &bias, &exp_a_log, &f16).is_none(),
            "the two projections must share a dtype"
        );
        assert!(
            decay_beta(
                &bf16,
                &bias,
                &exp_a_log,
                &array_from_f32(&values, &[1, 4, 2], MlxDtype::Bfloat16)
            )
            .is_none(),
            "the beta projection must match the decay projection shape"
        );
        assert!(
            decay_beta(
                &bf16,
                &array_from_f32(&[0.5, 0.5], &[2], MlxDtype::Float32),
                &exp_a_log,
                &bf16
            )
            .is_none(),
            "dt_bias must hold one value per head"
        );
        assert!(
            decay_beta(
                &bf16,
                &bias,
                &array_from_f32(&[1.0, 1.0, 1.0, 1.0], &[4], MlxDtype::Float16),
                &bf16
            )
            .is_none(),
            "exp_a_log must be float32"
        );

        let (conv, sum, sqrt_dk, geometry) = qkv_check_inputs();
        assert!(
            qkv_scale(
                &array_from_f32(&[1.0f32; 10], &[1, 1, 10], MlxDtype::Int32),
                &sum,
                &sum,
                &sqrt_dk,
                geometry
            )
            .is_none(),
            "an unsupported convolution dtype must stay on the eager path"
        );
        let narrow = array_from_f32(&[1.0f32; 9], &[1, 1, 9], MlxDtype::Float16);
        assert!(
            qkv_scale(&narrow, &sum, &sum, &sqrt_dk, geometry).is_none(),
            "the convolution width must be 2 * key_dim + value_dim"
        );
        assert!(
            qkv_scale(
                &conv,
                &array_from_f32(&[1.0], &[1], MlxDtype::Float32),
                &sum,
                &sqrt_dk,
                geometry
            )
            .is_none(),
            "the sums must hold one value per key head and row"
        );
        assert!(
            qkv_scale(
                &conv,
                &sum,
                &sum,
                &array_from_f32(&[2.0], &[1], MlxDtype::Float16),
                geometry
            )
            .is_none(),
            "sqrt_dk must be float32"
        );
        assert_eq!(
            launches_for_test(),
            launches,
            "unsupported inputs must not dispatch"
        );
        assert!(
            decay_beta(&bf16, &bias, &exp_a_log, &bf16).is_some(),
            "the rejected decay inputs must be the only unsupported ones"
        );
        assert!(
            qkv_scale(&conv, &sum, &sum, &sqrt_dk, geometry).is_some(),
            "the rejected q/k/v inputs must be the only unsupported ones"
        );
    }

    #[test]
    fn injected_self_check_failure_disables_both_kernels() {
        fail_next_self_check_for_test();
        let (projection, bias) = (MlxDtype::Bfloat16, MlxDtype::Bfloat16);
        let (decay_proj, dt_bias, exp_a_log) = decay_check_inputs(projection, bias);
        assert!(
            decay_beta(&decay_proj, &dt_bias, &exp_a_log, &decay_proj).is_none(),
            "an injected self-check failure must return None"
        );
        assert!(decay_verdict_failed_for_test(projection, bias));
        assert!(
            decay_beta(&decay_proj, &dt_bias, &exp_a_log, &decay_proj).is_none(),
            "a failed pair must stay disabled"
        );

        fail_next_self_check_for_test();
        let (conv, sum, sqrt_dk, geometry) = qkv_check_inputs();
        assert!(
            qkv_scale(&conv, &sum, &sum, &sqrt_dk, geometry).is_none(),
            "an injected self-check failure must return None"
        );
        assert!(qkv_verdict_failed_for_test(MlxDtype::Float16));
        assert!(
            qkv_scale(&conv, &sum, &sum, &sqrt_dk, geometry).is_none(),
            "a failed convolution dtype must stay disabled"
        );
        let launches = launches_for_test();
        assert!(
            qkv_scale(&conv, &sum, &sum, &sqrt_dk, geometry).is_none(),
            "a failed convolution dtype must never launch again"
        );
        assert_eq!(launches_for_test(), launches);
        reset_verdicts_for_test();
    }

    /// Every dtype combination carries its own verdict, so failing one does not
    /// disable a supported neighbour.
    #[test]
    fn verdicts_are_indexed_per_dtype_combination() {
        reset_verdicts_for_test();
        fail_next_self_check_for_test();
        let (mismatched, dt_bias, exp_a_log) =
            decay_check_inputs(MlxDtype::Bfloat16, MlxDtype::Float32);
        assert!(decay_beta(&mismatched, &dt_bias, &exp_a_log, &mismatched).is_none());
        assert!(decay_verdict_failed_for_test(
            MlxDtype::Bfloat16,
            MlxDtype::Float32
        ));
        assert!(
            !decay_verdict_failed_for_test(MlxDtype::Bfloat16, MlxDtype::Bfloat16),
            "a neighbouring pair must keep its own verdict"
        );

        let (decay_proj, bf16_bias, _) = decay_check_inputs(MlxDtype::Bfloat16, MlxDtype::Bfloat16);
        assert!(
            decay_beta(&decay_proj, &bf16_bias, &exp_a_log, &decay_proj).is_some(),
            "the untouched pair must still self-check and dispatch"
        );
        reset_verdicts_for_test();
    }

    /// A launch the kernel rejects after the self-check disables the dtype, so a
    /// failing dispatch is paid once instead of on every read.
    #[test]
    fn rejected_launch_records_failed() {
        reset_verdicts_for_test();
        let (projection, bias) = (MlxDtype::Float16, MlxDtype::Float16);
        let (decay_proj, dt_bias, exp_a_log) = decay_check_inputs(projection, bias);
        assert!(decay_beta(&decay_proj, &dt_bias, &exp_a_log, &decay_proj).is_some());
        assert!(!decay_verdict_failed_for_test(projection, bias));

        fail_next_launch_for_test();
        assert!(
            decay_beta(&decay_proj, &dt_bias, &exp_a_log, &decay_proj).is_none(),
            "a rejected launch must return None"
        );
        assert!(
            !launch_failure_pending_for_test(),
            "the injected failure must have reached the launch"
        );
        assert!(decay_verdict_failed_for_test(projection, bias));
        let launches = launches_for_test();
        assert!(decay_beta(&decay_proj, &dt_bias, &exp_a_log, &decay_proj).is_none());
        assert_eq!(
            launches_for_test(),
            launches,
            "a disabled dtype must not launch again"
        );
        reset_verdicts_for_test();
    }

    /// A supported input with an unrecorded verdict pays exactly one self-check
    /// and one build, whether or not the check passes.
    #[test]
    fn self_check_runs_once_per_dtype_combination() {
        reset_verdicts_for_test();
        let (decay_proj, dt_bias, exp_a_log) =
            decay_check_inputs(MlxDtype::Bfloat16, MlxDtype::Bfloat16);
        assert!(decay_beta(&decay_proj, &dt_bias, &exp_a_log, &decay_proj).is_some());
        let checks = self_check_runs_for_test();
        let builds = kernel_builds_for_test();
        assert!(decay_beta(&decay_proj, &dt_bias, &exp_a_log, &decay_proj).is_some());
        assert_eq!(
            self_check_runs_for_test(),
            checks,
            "a recorded verdict must skip the self-check"
        );
        assert_eq!(
            kernel_builds_for_test(),
            builds,
            "the kernel is built once per process"
        );
        reset_verdicts_for_test();
    }

    #[test]
    #[ignore = "exhaustive sweep; run on the build host"]
    fn native_pow_exhaustive() {
        let dtype = MlxDtype::Float32;
        let sqrt_dk = array_from_f32(&[1.0], &[1], MlxDtype::Float32);
        let total = 0x7f80_0000u64;
        let mut cursor = 1u64;
        while cursor < total {
            let len = (total - cursor).min(EXHAUSTIVE_CHUNK as u64) as usize;
            let sums: Vec<f32> = (0..len)
                .map(|offset| f32::from_bits((cursor + offset as u64) as u32))
                .collect();
            let key_dim = len as i32;
            let geometry = QkvGeometry {
                key_dim,
                head_dim: 1,
                value_dim: 1,
            };
            let conv_dim = 2 * key_dim + 1;
            let conv = array_from_f32(&vec![1.0f32; conv_dim as usize], &[1, 1, conv_dim], dtype);
            let sum_shape = [1, 1, key_dim, 1];
            let sum_q = array_from_f32(&sums, &sum_shape, MlxDtype::Float32);
            let sum_k = array_from_f32(&sums, &sum_shape, MlxDtype::Float32);
            let (q, k, v) = qkv_scale(&conv, &sum_q, &sum_k, &sqrt_dk, geometry)
                .unwrap_or_else(|| panic!("native q/k/v pow sweep at {cursor:#010x}"));
            let expected = mlx_sys::power(&sum_q, &cached_scalar(-0.5, MlxDtype::Float32), None);
            let expected_k = mlx_sys::power(&sum_k, &cached_scalar(-0.5, MlxDtype::Float32), None);
            assert_eq!(
                raw_bits(&q).expect("q bits"),
                raw_bits(&expected).expect("pow bits"),
                "pow sweep at {cursor:#010x}"
            );
            assert_eq!(
                raw_bits(&k).expect("k bits"),
                raw_bits(&expected_k).expect("key pow bits"),
                "key pow sweep at {cursor:#010x}"
            );
            assert_eq!(
                raw_bits(&v).expect("v bits")[0],
                1.0f32.to_bits(),
                "the value path must copy the convolution at {cursor:#010x}"
            );
            cursor += len as u64;
        }
    }

    #[test]
    #[ignore = "exhaustive sweep; run on the build host"]
    fn native_decay_chain_exhaustive() {
        let total = 1u64 << 32;
        for parameter in [1.0f32, 0.0625, 13.7] {
            let dt_bias = array_from_f32(&[0.0], &[1], MlxDtype::Float32);
            let exp_a_log = array_from_f32(&[parameter], &[1], MlxDtype::Float32);
            let mut cursor = 0u64;
            while cursor < total {
                let len = (total - cursor).min(EXHAUSTIVE_CHUNK as u64) as usize;
                let values: Vec<f32> = (0..len)
                    .map(|offset| f32::from_bits((cursor + offset as u64) as u32))
                    .collect();
                let shape = [1, len as i32, 1];
                let decay_proj = array_from_f32(&values, &shape, MlxDtype::Float32);
                let (decay, beta) = decay_beta(&decay_proj, &dt_bias, &exp_a_log, &decay_proj)
                    .unwrap_or_else(|| panic!("native decay chain at {cursor:#010x}"));
                let (decay_expected, beta_expected) =
                    decay_reference(&decay_proj, &dt_bias, &exp_a_log);
                assert_eq!(
                    raw_bits(&decay).expect("decay bits"),
                    raw_bits(&decay_expected).expect("decay reference bits"),
                    "decay chain at {cursor:#010x} with {parameter}"
                );
                assert_eq!(
                    raw_bits(&beta).expect("beta bits"),
                    raw_bits(&beta_expected).expect("beta reference bits"),
                    "beta chain at {cursor:#010x} with {parameter}"
                );
                cursor += len as u64;
            }
        }
    }

    #[test]
    #[ignore = "exhaustive sweep; run on the build host"]
    fn native_sigmoid_exhaustive_bf16_f16() {
        let patterns: Vec<u16> = (0..PATTERNS).map(|bits| bits as u16).collect();
        let dt_bias = array_from_f32(&[0.0], &[1], MlxDtype::Float32);
        let exp_a_log = array_from_f32(&[1.0], &[1], MlxDtype::Float32);
        for dtype in [MlxDtype::Bfloat16, MlxDtype::Float16] {
            let proj = reshape(
                &pattern_array(&patterns, dtype),
                &[1, PATTERNS as i32, 1],
                None,
            );
            let (decay, beta) = decay_beta(&proj, &dt_bias, &exp_a_log, &proj)
                .unwrap_or_else(|| panic!("native sigmoid sweep for {dtype:?}"));
            let (decay_expected, beta_expected) = decay_reference(&proj, &dt_bias, &exp_a_log);
            assert_eq!(
                raw_bits(&beta).expect("beta bits"),
                raw_bits(&beta_expected).expect("beta reference bits"),
                "sigmoid sweep for {dtype:?}"
            );
            assert_eq!(
                raw_bits(&decay).expect("decay bits"),
                raw_bits(&decay_expected).expect("decay reference bits"),
                "decay sweep for {dtype:?}"
            );
        }
    }
}
