use crate::array::{MlxArray, MlxDtype, null_ffi_array};
use crate::ffi;
use crate::stream::{MlxStream, default_gpu_raw};

unsafe extern "C" {
    fn ax_mlx_gelu_approx_mul(
        res: *mut ffi::mlx_array,
        gate: ffi::mlx_array,
        x: ffi::mlx_array,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_gelu_approx_mul_matmul(
        res: *mut ffi::mlx_array,
        gate: ffi::mlx_array,
        x: ffi::mlx_array,
        weight: ffi::mlx_array,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_gelu_approx_quantized_ffn(
        res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        gate_up_weight: ffi::mlx_array,
        gate_up_scales: ffi::mlx_array,
        gate_up_biases: ffi::mlx_array,
        down_weight: ffi::mlx_array,
        down_scales: ffi::mlx_array,
        down_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_qk_norm_rope_bhsd_from_proj(
        res: *mut ffi::mlx_array,
        proj: ffi::mlx_array,
        norm: ffi::mlx_array,
        n_heads: libc::c_int,
        head_dim: libc::c_int,
        eps: libc::c_float,
        rope_dims: libc::c_int,
        traditional: libc::c_int,
        has_base: libc::c_int,
        base: libc::c_float,
        offset: libc::c_int,
        freqs: ffi::mlx_array,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;
}

fn optional_int(value: Option<i32>, default: i32) -> ffi::mlx_optional_int_ {
    ffi::mlx_optional_int_ {
        has_value: value.is_some(),
        value: value.unwrap_or(default),
    }
}

fn optional_dtype(value: Option<MlxDtype>, default: MlxDtype) -> ffi::mlx_optional_dtype_ {
    ffi::mlx_optional_dtype_ {
        has_value: value.is_some(),
        value: value.unwrap_or(default).to_ffi(),
    }
}

macro_rules! unary_op {
    ($name:ident, $ffi_fn:ident) => {
        pub fn $name(a: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
            $crate::op_count::bump();
            unsafe {
                let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
                let mut res = MlxArray::empty();
                ffi::$ffi_fn(&mut res.inner, a.inner, stream);
                res
            }
        }
    };
}

macro_rules! binary_op {
    ($name:ident, $ffi_fn:ident) => {
        pub fn $name(a: &MlxArray, b: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
            $crate::op_count::bump();
            unsafe {
                let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
                let mut res = MlxArray::empty();
                ffi::$ffi_fn(&mut res.inner, a.inner, b.inner, stream);
                res
            }
        }
    };
}

binary_op!(add, mlx_add);
binary_op!(subtract, mlx_subtract);
binary_op!(divide, mlx_divide);
binary_op!(multiply, mlx_multiply);
binary_op!(matmul, mlx_matmul);
binary_op!(greater_equal, mlx_greater_equal);
binary_op!(less, mlx_less);
binary_op!(logical_and, mlx_logical_and);
binary_op!(maximum, mlx_maximum);
binary_op!(minimum, mlx_minimum);
binary_op!(power, mlx_power);

unary_op!(sigmoid, mlx_sigmoid);
unary_op!(tanh, mlx_tanh);
unary_op!(erf, mlx_erf);
unary_op!(exp, mlx_exp);
unary_op!(log, mlx_log);
unary_op!(log1p, mlx_log1p);
unary_op!(negative, mlx_negative);

/// silu(x) = x * sigmoid(x)
pub fn silu(x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    let sig = sigmoid(x, s);
    multiply(x, &sig, s)
}

/// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))  — exact GELU used for GEGLU activations.
pub fn gelu(x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    let dtype = x.dtype();
    let mk_scalar = |v: f32| cached_scalar(v, dtype);
    let inv_sqrt2 = mk_scalar(std::f32::consts::FRAC_1_SQRT_2);
    let scaled = multiply(x, &inv_sqrt2, s);
    let erf_val = erf(&scaled, s);
    let one_plus_erf = add(&erf_val, &mk_scalar(1.0), s);
    let half_x = multiply(x, &mk_scalar(0.5), s);
    multiply(&half_x, &one_plus_erf, s)
}

/// gelu_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// Matches mlx-lm's `nn.gelu_approx`. Used by Gemma4 per-layer input gate.
pub fn gelu_approx(x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    let dtype = x.dtype();
    let mk_scalar = |v: f32| cached_scalar(v, dtype);
    // sqrt(2/π)
    let sqrt_2_over_pi: f32 = 0.797_884_6;
    let coeff: f32 = 0.044_715;
    let x2 = multiply(x, x, s);
    let x3 = multiply(&x2, x, s);
    let cx3 = multiply(&mk_scalar(coeff), &x3, s);
    let inner = add(x, &cx3, s);
    let t = tanh(&multiply(&mk_scalar(sqrt_2_over_pi), &inner, s), s);
    let one_plus_t = add(&mk_scalar(1.0), &t, s);
    multiply(&multiply(&mk_scalar(0.5), x, s), &one_plus_t, s)
}

/// Compute `gelu_approx(gate) * x` through AX's direct MLX C++ shim.
///
/// This keeps the exact mlx-lm Gemma-family math but collapses the Rust ->
/// `mlx-c` call boundary for the scalar-heavy activation chain to one FFI call.
/// If the direct shim reports an error, fall back to the portable wrapper
/// composition rather than surfacing a hard runtime failure.
pub fn gelu_approx_mul(gate: &MlxArray, x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_gelu_approx_mul(&mut res.inner, gate.inner, x.inner, stream);
        if rc == 0 {
            return res;
        }
    }
    multiply(&gelu_approx(gate, s), x, s)
}

/// Compute `matmul(gelu_approx(gate) * x, weight)` through AX's direct MLX C++ shim.
///
/// This is a microbenchmark/probe surface for the direct-MLX PRD. It collapses
/// the activation and the following dense projection behind one Rust FFI call.
/// Runtime model code should keep using the portable path until a real-shape
/// artifact proves this candidate clears the promotion gate.
pub fn gelu_approx_mul_matmul(
    gate: &MlxArray,
    x: &MlxArray,
    weight: &MlxArray,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_gelu_approx_mul_matmul(
            &mut res.inner,
            gate.inner,
            x.inner,
            weight.inner,
            stream,
        );
        if rc == 0 {
            return res;
        }
    }
    matmul(&multiply(&gelu_approx(gate, s), x, s), weight, s)
}

/// Compute a packed dense GEGLU FFN block through AX's direct MLX C++ shim.
///
/// The expression is:
///
/// ```text
/// gate_up = quantized_matmul(x, gate_up_weight, ...)
/// gate, up = split(gate_up, 2, axis=-1)
/// hidden = gelu_approx(gate) * up
/// out = quantized_matmul(hidden, down_weight, ...)
/// ```
///
/// This remains a probe surface until a real-shape artifact clears the PRD's
/// promotion gate. Production model code should continue using the portable
/// route unless a later commit adds explicit routing and kill switches.
#[allow(clippy::too_many_arguments)]
pub fn gelu_approx_quantized_ffn(
    x: &MlxArray,
    gate_up_weight: &MlxArray,
    gate_up_scales: &MlxArray,
    gate_up_biases: Option<&MlxArray>,
    down_weight: &MlxArray,
    down_scales: &MlxArray,
    down_biases: Option<&MlxArray>,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let gate_up_biases = gate_up_biases
            .map(|biases| biases.inner)
            .unwrap_or_else(null_ffi_array);
        let down_biases = down_biases
            .map(|biases| biases.inner)
            .unwrap_or_else(null_ffi_array);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_gelu_approx_quantized_ffn(
            &mut res.inner,
            x.inner,
            gate_up_weight.inner,
            gate_up_scales.inner,
            gate_up_biases,
            down_weight.inner,
            down_scales.inner,
            down_biases,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            return res;
        }
    }
    let gate_up = quantized_matmul(
        x,
        gate_up_weight,
        gate_up_scales,
        gate_up_biases,
        true,
        Some(group_size),
        Some(bits),
        s,
    );
    let packed_dim = gate_up
        .shape()
        .last()
        .copied()
        .expect("gate/up projection must have a last dimension");
    let half = packed_dim / 2;
    let gate = slice_last_dim(&gate_up, 0, half, s);
    let up = slice_last_dim(&gate_up, half, packed_dim, s);
    let hidden = gelu_approx_mul(&gate, &up, s);
    quantized_matmul(
        &hidden,
        down_weight,
        down_scales,
        down_biases,
        true,
        Some(group_size),
        Some(bits),
        s,
    )
}

/// Probe-only direct C++ shim for:
///
/// ```text
/// as_strided([B, S, H * D] -> [B, H, S, D])
/// rms_norm(..., norm)
/// rope(...)
/// ```
///
/// This is intentionally not used by production model code yet. It gives the
/// direct-MLX PRD a narrow measurement surface for the QK-norm + RoPE region
/// before any routing or kill-switch work is considered.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_bhsd_from_proj(
    proj: &MlxArray,
    norm: Option<&MlxArray>,
    n_heads: i32,
    head_dim: i32,
    eps: f32,
    rope_dims: i32,
    traditional: bool,
    base: Option<f32>,
    offset: i32,
    freqs: Option<&MlxArray>,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_qk_norm_rope_bhsd_from_proj(
            &mut res.inner,
            proj.inner,
            norm.map(|n| n.inner).unwrap_or_else(null_ffi_array),
            n_heads,
            head_dim,
            eps,
            rope_dims,
            i32::from(traditional),
            i32::from(base.is_some()),
            base.unwrap_or(1.0),
            offset,
            freqs.map(|f| f.inner).unwrap_or_else(null_ffi_array),
            stream,
        );
        if rc == 0 {
            return res;
        }
    }

    let shape = proj.shape();
    let batch = shape.first().copied().unwrap_or(1);
    let seq = shape.get(1).copied().unwrap_or(1);
    let width = i64::from(n_heads) * i64::from(head_dim);
    let bhsd = as_strided(
        proj,
        &[batch, n_heads, seq, head_dim],
        &[i64::from(seq) * width, i64::from(head_dim), width, 1],
        0,
        s,
    );
    let normed = crate::fast::rms_norm(&bhsd, norm, eps, s);
    crate::fast::rope(&normed, rope_dims, traditional, base, 1.0, offset, freqs, s)
}

pub fn astype(a: &MlxArray, dtype: MlxDtype, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_astype(&mut res.inner, a.inner, dtype.to_ffi(), stream);
        res
    }
}

pub fn arange(
    start: f64,
    stop: f64,
    step: f64,
    dtype: MlxDtype,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_arange(&mut res.inner, start, stop, step, dtype.to_ffi(), stream);
        res
    }
}

pub fn reshape(a: &MlxArray, shape: &[i32], s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_reshape(&mut res.inner, a.inner, shape.as_ptr(), shape.len(), stream);
        res
    }
}

pub fn broadcast_to(a: &MlxArray, shape: &[i32], s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_broadcast_to(&mut res.inner, a.inner, shape.as_ptr(), shape.len(), stream);
        res
    }
}

pub fn transpose(a: &MlxArray, axes: &[i32], s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_transpose_axes(&mut res.inner, a.inner, axes.as_ptr(), axes.len(), stream);
        res
    }
}

pub fn expand_dims(a: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_expand_dims(&mut res.inner, a.inner, axis, stream);
        res
    }
}

pub fn expand_dims_axes(a: &MlxArray, axes: &[i32], s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_expand_dims_axes(&mut res.inner, a.inner, axes.as_ptr(), axes.len(), stream);
        res
    }
}

pub fn softmax(a: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_softmax_axis(&mut res.inner, a.inner, axis, false, stream);
        res
    }
}

pub fn softmax_precise(a: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_softmax_axis(&mut res.inner, a.inner, axis, true, stream);
        res
    }
}

pub fn concatenate(arrays: &[&MlxArray], axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let vec = ffi::mlx_vector_array_new();
        for arr in arrays {
            ffi::mlx_vector_array_append_value(vec, arr.inner);
        }
        let mut res = MlxArray::empty();
        ffi::mlx_concatenate_axis(&mut res.inner, vec, axis, stream);
        ffi::mlx_vector_array_free(vec);
        res
    }
}

pub fn stack(arrays: &[&MlxArray], axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let vec = ffi::mlx_vector_array_new();
        for arr in arrays {
            ffi::mlx_vector_array_append_value(vec, arr.inner);
        }
        let mut res = MlxArray::empty();
        ffi::mlx_stack_axis(&mut res.inner, vec, axis, stream);
        ffi::mlx_vector_array_free(vec);
        res
    }
}

/// Gather rows by integer indices along axis 0.
pub fn take(a: &MlxArray, indices: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_take_axis(&mut res.inner, a.inner, indices.inner, axis, stream);
        res
    }
}

/// Argmax over the last axis.
/// Repeat elements of an array along an axis (repeat-interleave semantics).
///
/// `axis=1` on `[1, n_kv_heads, seq, head_dim]` with `repeats=4` produces
/// `[1, n_heads, seq, head_dim]` suitable for GQA head expansion.
pub fn repeat_axis(a: &MlxArray, repeats: i32, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_repeat_axis(&mut res.inner, a.inner, repeats, axis, stream);
        res
    }
}

/// Slice array along every axis.
///
/// `start`, `stop`, and `strides` must each have length equal to `a.ndim()`.
pub fn slice(
    a: &MlxArray,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_slice(
            &mut res.inner,
            a.inner,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream,
        );
        res
    }
}

/// Reinterpret a contiguous array with a new shape and explicit strides.
///
/// Creates a non-contiguous view of the underlying buffer — no data is copied.
/// The caller must ensure the strides describe a valid layout of the source data.
///
/// Typical use: replace reshape + transpose (2 graph nodes) with a single
/// `as_strided` view (1 graph node).
pub fn as_strided(
    a: &MlxArray,
    shape: &[i32],
    strides: &[i64],
    offset: usize,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_as_strided(
            &mut res.inner,
            a.inner,
            shape.as_ptr(),
            shape.len(),
            strides.as_ptr(),
            strides.len(),
            offset,
            stream,
        );
        res
    }
}

/// Split `a` into `num_splits` equal parts along `axis`.
///
/// Returns `num_splits` views of the original data (no copy when the split is
/// uniform).  `a.shape()[axis]` must be divisible by `num_splits`.
pub fn split(a: &MlxArray, num_splits: i32, axis: i32, s: Option<&MlxStream>) -> Vec<MlxArray> {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut out_vec = ffi::mlx_vector_array_new();
        ffi::mlx_split(&mut out_vec, a.inner, num_splits, axis, stream);
        let n = ffi::mlx_vector_array_size(out_vec);
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut arr = MlxArray::empty();
            ffi::mlx_vector_array_get(&mut arr.inner, out_vec, i);
            result.push(arr);
        }
        ffi::mlx_vector_array_free(out_vec);
        result
    }
}

/// Slice the last dimension of `a` from index `start` to `end` (exclusive).
pub fn slice_last_dim(a: &MlxArray, start: i32, end: i32, s: Option<&MlxStream>) -> MlxArray {
    let ndim = a.ndim();
    assert!(
        ndim > 0,
        "slice_last_dim requires at least 1-dimensional array"
    );
    let shape = a.shape();
    let mut start_vec = vec![0i32; ndim];
    let mut stop_vec = shape.clone();
    let strides_vec = vec![1i32; ndim];
    start_vec[ndim - 1] = start;
    stop_vec[ndim - 1] = end;
    slice(a, &start_vec, &stop_vec, &strides_vec, s)
}

/// Allocate a zero-filled array with the given shape and dtype.
pub fn zeros(shape: &[i32], dtype: MlxDtype, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_zeros(
            &mut res.inner,
            shape.as_ptr(),
            shape.len(),
            dtype.to_ffi(),
            stream,
        );
        res
    }
}

/// Return a copy of `src` with `update` written at `src[start:stop:strides]`.
///
/// Mirrors Python `mx.array[start:stop:strides] = update`.  Unlike concatenate,
/// this avoids re-copying existing data — only the `update` region is written.
pub fn slice_update(
    src: &MlxArray,
    update: &MlxArray,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_slice_update(
            &mut res.inner,
            src.inner,
            update.inner,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream,
        );
        res
    }
}

pub fn contiguous(a: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_contiguous(&mut res.inner, a.inner, false, stream);
        res
    }
}

pub fn clip(a: &MlxArray, min: &MlxArray, max: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_clip(&mut res.inner, a.inner, min.inner, max.inner, stream);
        res
    }
}

pub fn where_cond(
    condition: &MlxArray,
    x: &MlxArray,
    y: &MlxArray,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_where(&mut res.inner, condition.inner, x.inner, y.inner, stream);
        res
    }
}

pub fn conv1d(
    input: &MlxArray,
    weight: &MlxArray,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_conv1d(
            &mut res.inner,
            input.inner,
            weight.inner,
            stride,
            padding,
            dilation,
            groups,
            stream,
        );
        res
    }
}

pub fn argmax(a: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let ndim = a.ndim();
        assert!(ndim > 0, "argmax requires at least 1-dimensional array");
        let axis = ndim as i32 - 1;
        let mut res = MlxArray::empty();
        ffi::mlx_argmax_axis(&mut res.inner, a.inner, axis, false, stream);
        res
    }
}

pub fn argsort_axis(a: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_argsort_axis(&mut res.inner, a.inner, axis, stream);
        res
    }
}

/// Argpartition along `axis`. Returns indices such that the element at position
/// `kth` would be in its sorted position. Negative kth counts from the end.
pub fn argpartition_axis(a: &MlxArray, kth: i32, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_argpartition_axis(&mut res.inner, a.inner, kth, axis, stream);
        res
    }
}

pub fn take_along_axis(
    a: &MlxArray,
    indices: &MlxArray,
    axis: i32,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_take_along_axis(&mut res.inner, a.inner, indices.inner, axis, stream);
        res
    }
}

pub fn put_along_axis(
    a: &MlxArray,
    indices: &MlxArray,
    values: &MlxArray,
    axis: i32,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_put_along_axis(
            &mut res.inner,
            a.inner,
            indices.inner,
            values.inner,
            axis,
            stream,
        );
        res
    }
}

pub fn sum_axis(a: &MlxArray, axis: i32, keepdims: bool, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_sum_axis(&mut res.inner, a.inner, axis, keepdims, stream);
        res
    }
}

/// Batched matmul selecting rows of `b` via `rhs_indices`.
///
/// Computes `a[lhs_i] @ b[rhs_i]` for each index pair. `lhs_indices` is
/// always null here (use all rows of `a`). `b` shape: `[N, K, L]`.
pub fn gather_mm(
    a: &MlxArray,
    b: &MlxArray,
    rhs_indices: &MlxArray,
    sorted_indices: bool,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let null_arr = null_ffi_array();
        let mut res = MlxArray::empty();
        ffi::mlx_gather_mm(
            &mut res.inner,
            a.inner,
            b.inner,
            null_arr,
            rhs_indices.inner,
            sorted_indices,
            stream,
        );
        res
    }
}

/// Quantized batched matmul selecting experts via `rhs_indices`.
///
/// Equivalent to `gather_mm` on the dequantized weight. With `transpose=true`,
/// computes `x @ w[rhs_i].T` for each selected expert.
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
    x: &MlxArray,
    w: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    rhs_indices: &MlxArray,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    sorted_indices: bool,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases_raw = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);
        let null_arr = null_ffi_array();
        let gs = ffi::mlx_optional_int_ {
            has_value: group_size.is_some(),
            value: group_size.unwrap_or(64),
        };
        let bs = ffi::mlx_optional_int_ {
            has_value: bits.is_some(),
            value: bits.unwrap_or(4),
        };
        let mut res = MlxArray::empty();
        ffi::mlx_gather_qmm(
            &mut res.inner,
            x.inner,
            w.inner,
            scales.inner,
            biases_raw,
            null_arr,
            rhs_indices.inner,
            transpose,
            gs,
            bs,
            c"affine".as_ptr(),
            sorted_indices,
            stream,
        );
        res
    }
}

/// Dequantize a packed-int4 weight tensor to floating point.
///
/// `w` must be the packed uint32 tensor from MLX quantized format.
/// `group_size` and `bits` default to 64 and 4 when `None`.
pub fn dequantize(
    w: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    group_size: Option<i32>,
    bits: Option<i32>,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases_raw = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);
        let gs = ffi::mlx_optional_int_ {
            has_value: group_size.is_some(),
            value: group_size.unwrap_or(64),
        };
        let bs = ffi::mlx_optional_int_ {
            has_value: bits.is_some(),
            value: bits.unwrap_or(4),
        };
        let no_dtype = ffi::mlx_optional_dtype_ {
            has_value: false,
            value: ffi::mlx_dtype_::MLX_FLOAT32,
        };
        let mut res = MlxArray::empty();
        ffi::mlx_dequantize(
            &mut res.inner,
            w.inner,
            scales.inner,
            biases_raw,
            gs,
            bs,
            c"affine".as_ptr(),
            null_ffi_array(),
            no_dtype,
            stream,
        );
        res
    }
}

/// MLX quantization modes supported by `quantize` / `dequantize_with_mode`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxQuantizationMode {
    Affine,
    Mxfp4,
    Mxfp8,
    Nvfp4,
}

impl MlxQuantizationMode {
    fn as_ptr(self) -> *const std::ffi::c_char {
        match self {
            Self::Affine => c"affine".as_ptr(),
            Self::Mxfp4 => c"mxfp4".as_ptr(),
            Self::Mxfp8 => c"mxfp8".as_ptr(),
            Self::Nvfp4 => c"nvfp4".as_ptr(),
        }
    }

    fn default_group_size(self) -> i32 {
        match self {
            Self::Affine => 64,
            Self::Nvfp4 => 16,
            Self::Mxfp4 | Self::Mxfp8 => 32,
        }
    }

    fn default_bits(self) -> i32 {
        match self {
            Self::Mxfp8 => 8,
            Self::Affine | Self::Mxfp4 | Self::Nvfp4 => 4,
        }
    }
}

/// Quantize a floating-point weight matrix along its last axis.
///
/// Affine mode returns `[packed_weight, scales, biases]`; FP modes return
/// `[packed_weight, scales]`, matching MLX's `mx.quantize` contract.
pub fn quantize(
    w: &MlxArray,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: MlxQuantizationMode,
    global_scale: Option<&MlxArray>,
    s: Option<&MlxStream>,
) -> Vec<MlxArray> {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let global_scale = global_scale
            .map(|scale| scale.inner)
            .unwrap_or_else(null_ffi_array);
        let gs = optional_int(group_size, mode.default_group_size());
        let bs = optional_int(bits, mode.default_bits());
        let mut raw = ffi::mlx_vector_array {
            ctx: std::ptr::null_mut(),
        };
        ffi::mlx_quantize(
            &mut raw,
            w.inner,
            gs,
            bs,
            mode.as_ptr(),
            global_scale,
            stream,
        );
        let len = ffi::mlx_vector_array_size(raw);
        let mut result = Vec::with_capacity(len);
        for idx in 0..len {
            let mut arr = null_ffi_array();
            ffi::mlx_vector_array_get(&mut arr, raw, idx);
            result.push(MlxArray::from_raw(arr));
        }
        ffi::mlx_vector_array_free(raw);
        result
    }
}

/// Dequantize a matrix produced by `quantize`.
///
/// This is the mode-aware counterpart to `dequantize`, which preserves the
/// legacy affine-only wrapper used by existing model-loading code.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_with_mode(
    w: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: MlxQuantizationMode,
    global_scale: Option<&MlxArray>,
    dtype: Option<MlxDtype>,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases_raw = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);
        let global_scale = global_scale
            .map(|scale| scale.inner)
            .unwrap_or_else(null_ffi_array);
        let gs = optional_int(group_size, mode.default_group_size());
        let bs = optional_int(bits, mode.default_bits());
        let dtype = optional_dtype(dtype, MlxDtype::Float32);
        let mut res = MlxArray::empty();
        ffi::mlx_dequantize(
            &mut res.inner,
            w.inner,
            scales.inner,
            biases_raw,
            gs,
            bs,
            mode.as_ptr(),
            global_scale,
            dtype,
            stream,
        );
        res
    }
}

/// Convert a floating-point array to MLX's E4M3 float8 representation.
pub fn to_fp8(x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_to_fp8(&mut res.inner, x.inner, stream);
        res
    }
}

/// Convert an MLX E4M3 float8 array back to a floating-point dtype.
pub fn from_fp8(x: &MlxArray, dtype: MlxDtype, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_from_fp8(&mut res.inner, x.inner, dtype.to_ffi(), stream);
        res
    }
}

/// Quantized matmul: x @ dequantize(w, scales, biases).
///
/// `group_size` and `bits` use the MLX defaults (64 and 4) when `None`.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul(
    x: &MlxArray,
    w: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases_raw = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);

        let gs = ffi::mlx_optional_int_ {
            has_value: group_size.is_some(),
            value: group_size.unwrap_or(64),
        };
        let bs = ffi::mlx_optional_int_ {
            has_value: bits.is_some(),
            value: bits.unwrap_or(4),
        };

        let mut res = MlxArray::empty();
        ffi::mlx_quantized_matmul(
            &mut res.inner,
            x.inner,
            w.inner,
            scales.inner,
            biases_raw,
            transpose,
            gs,
            bs,
            c"affine".as_ptr(),
            stream,
        );
        res
    }
}

/// Return a cached scalar `MlxArray` of `value` materialised in `dtype`.
///
/// Functions like `gelu` and `gelu_approx` previously built four short-lived
/// scalar arrays per call (each call to `from_raw_data` + `astype`), which
/// allocated 8 `MlxArray` wrappers per activation invocation. On Gemma 4
/// E2B decode that contributed ~500 transient MlxArray instances per
/// forward pass (~30% of the AX-vs-mlx_lm gap surfaced in
/// `gemma4_e2b_eval_barrier_audit.v1`). Caching scalars by
/// `(value_bits, dtype)` collapses those allocations to one per unique
/// (constant, dtype) pair across the entire process lifetime.
pub fn cached_scalar(value: f32, dtype: MlxDtype) -> MlxArray {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<(u32, MlxDtype), MlxArray>>> = OnceLock::new();
    let key = (value.to_bits(), dtype);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().expect("cached_scalar cache poisoned");
    if let Some(arr) = map.get(&key) {
        return arr.clone();
    }
    let f32_arr = MlxArray::from_raw_data(
        &value as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    );
    let out = if dtype == MlxDtype::Float32 {
        f32_arr
    } else {
        astype(&f32_arr, dtype, None)
    };
    map.insert(key, out.clone());
    out
}

/// Top-k values along the last axis.
pub fn topk(a: &MlxArray, k: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_topk(&mut res.inner, a.inner, k, stream);
        res
    }
}

/// Top-k values along `axis`.
pub fn topk_axis(a: &MlxArray, k: i32, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_topk_axis(&mut res.inner, a.inner, k, axis, stream);
        res
    }
}

/// Flatten axes `start_axis..=end_axis` into a single axis.
pub fn flatten(a: &MlxArray, start_axis: i32, end_axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_flatten(&mut res.inner, a.inner, start_axis, end_axis, stream);
        res
    }
}

/// Repeat the array `repeats` times along a new leading axis.
///
/// Unlike `repeat_axis` (interleave semantics), this tiles the entire array.
pub fn repeat(a: &MlxArray, repeats: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_repeat(&mut res.inner, a.inner, repeats, stream);
        res
    }
}

unary_op!(cos, mlx_cos);
unary_op!(sin, mlx_sin);
unary_op!(floor, mlx_floor);
unary_op!(stop_gradient, mlx_stop_gradient);
binary_op!(outer, mlx_outer);

/// Pad array along specified axes with constant value.
pub fn pad(
    a: &MlxArray,
    axes: &[i32],
    low_pad: &[i32],
    high_pad: &[i32],
    pad_value: &MlxArray,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_pad(
            &mut res.inner,
            a.inner,
            axes.as_ptr(),
            axes.len(),
            low_pad.as_ptr(),
            low_pad.len(),
            high_pad.as_ptr(),
            high_pad.len(),
            pad_value.inner,
            c"constant".as_ptr(),
            stream,
        );
        res
    }
}

/// Unflatten axis into a new shape (inverse of `flatten`).
pub fn unflatten(a: &MlxArray, axis: i32, shape: &[i32], s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_unflatten(
            &mut res.inner,
            a.inner,
            axis,
            shape.as_ptr(),
            shape.len(),
            stream,
        );
        res
    }
}

/// GPU-side categorical sampling from logits.
///
/// Returns a `[1]` shaped `u32` array containing the sampled token index.
/// Equivalent to `mx.random.categorical(logits * (1/temperature), axis=-1)`.
/// The caller must scale logits by `1/temperature` before calling this
/// (or pass unscaled logits for temperature=1.0).
///
/// Uses MLX's internal RNG state (`key=null`). Not reproducible across runs.
pub fn random_categorical(logits: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        ffi::mlx_random_categorical(&mut res.inner, logits.inner, -1, null_ffi_array(), stream);
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::eval;

    #[test]
    fn conv1d_depthwise_reports_reference_shape() {
        let input = zeros(&[1, 6, 2], MlxDtype::Float32, None);
        let weight = zeros(&[2, 3, 1], MlxDtype::Float32, None);
        let out = conv1d(&input, &weight, 1, 0, 1, 2, None);

        assert_eq!(out.shape(), vec![1, 4, 2]);
    }

    #[test]
    fn stack_and_where_preserve_expected_shapes() {
        let a = zeros(&[2, 3], MlxDtype::Float32, None);
        let b = zeros(&[2, 3], MlxDtype::Float32, None);
        let condition = zeros(&[2, 3], MlxDtype::Bool, None);

        assert_eq!(stack(&[&a, &b], 1, None).shape(), vec![2, 2, 3]);
        assert_eq!(where_cond(&condition, &a, &b, None).shape(), vec![2, 3]);
    }

    #[test]
    fn scalar_math_wrappers_keep_input_shape() {
        let a = zeros(&[2, 3], MlxDtype::Float32, None);
        let min_value = 0.0_f32;
        let max_value = 1.0_f32;
        let min = MlxArray::from_raw_data(
            &min_value as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[1],
            MlxDtype::Float32,
        );
        let max = MlxArray::from_raw_data(
            &max_value as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[1],
            MlxDtype::Float32,
        );

        assert_eq!(
            log1p(&exp(&negative(&a, None), None), None).shape(),
            vec![2, 3]
        );
        assert_eq!(clip(&a, &min, &max, None).shape(), vec![2, 3]);
    }

    #[test]
    fn gelu_approx_mul_direct_matches_portable_composition() {
        let gate_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 12.0) * 0.125).collect();
        let x_data: Vec<f32> = (0..24).map(|i| ((i as f32) + 1.0) * 0.03125).collect();
        let gate = MlxArray::from_raw_data(
            gate_data.as_ptr() as *const u8,
            std::mem::size_of_val(gate_data.as_slice()),
            &[2, 3, 4],
            MlxDtype::Float32,
        );
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[2, 3, 4],
            MlxDtype::Float32,
        );

        let prev = crate::op_count::op_count_snapshot();
        let direct = gelu_approx_mul(&gate, &x, None);
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ activation shim should count as one Rust FFI dispatch"
        );
        let portable = multiply(&gelu_approx(&gate, None), &x, None);
        let direct_f32 = astype(&direct, MlxDtype::Float32, None);
        let portable_f32 = astype(&portable, MlxDtype::Float32, None);
        eval(&[&direct_f32, &portable_f32]);

        assert_eq!(direct_f32.shape(), vec![2, 3, 4]);
        assert_eq!(
            direct_f32.data_f32().to_vec(),
            portable_f32.data_f32().to_vec(),
            "direct C++ activation shim must preserve mlx-lm GEGLU math"
        );
    }

    #[test]
    fn gelu_approx_mul_matmul_direct_matches_portable_composition() {
        let gate_data: Vec<f32> = (0..6).map(|i| ((i as f32) - 3.0) * 0.25).collect();
        let x_data: Vec<f32> = (0..6).map(|i| ((i as f32) + 1.0) * 0.125).collect();
        let weight_data: Vec<f32> = (0..12).map(|i| ((i as f32) - 6.0) * 0.0625).collect();
        let gate = MlxArray::from_raw_data(
            gate_data.as_ptr() as *const u8,
            std::mem::size_of_val(gate_data.as_slice()),
            &[2, 3],
            MlxDtype::Float32,
        );
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[2, 3],
            MlxDtype::Float32,
        );
        let weight = MlxArray::from_raw_data(
            weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(weight_data.as_slice()),
            &[3, 4],
            MlxDtype::Float32,
        );

        let prev = crate::op_count::op_count_snapshot();
        let direct = gelu_approx_mul_matmul(&gate, &x, &weight, None);
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ activation+matmul shim should count as one Rust FFI dispatch"
        );
        let portable = matmul(
            &multiply(&gelu_approx(&gate, None), &x, None),
            &weight,
            None,
        );
        eval(&[&direct, &portable]);

        assert_eq!(direct.shape(), vec![2, 4]);
        assert_eq!(
            direct.data_f32().to_vec(),
            portable.data_f32().to_vec(),
            "direct C++ activation+matmul shim must preserve portable math"
        );
    }

    #[test]
    fn qk_norm_rope_direct_matches_portable_composition() {
        let seq = 3_i32;
        let n_heads = 2_i32;
        let head_dim = 4_i32;
        let width = n_heads * head_dim;
        let proj_data: Vec<f32> = (0..(seq * width))
            .map(|i| ((i as f32) - 11.0) * 0.03125)
            .collect();
        let norm_data: Vec<f32> = (0..head_dim).map(|i| 0.75 + (i as f32) * 0.125).collect();
        let proj = MlxArray::from_raw_data(
            proj_data.as_ptr() as *const u8,
            std::mem::size_of_val(proj_data.as_slice()),
            &[1, seq, width],
            MlxDtype::Float32,
        );
        let norm = MlxArray::from_raw_data(
            norm_data.as_ptr() as *const u8,
            std::mem::size_of_val(norm_data.as_slice()),
            &[head_dim],
            MlxDtype::Float32,
        );

        let prev = crate::op_count::op_count_snapshot();
        let direct = qk_norm_rope_bhsd_from_proj(
            &proj,
            Some(&norm),
            n_heads,
            head_dim,
            1.0e-6,
            head_dim,
            false,
            Some(10_000.0),
            2,
            None,
            None,
        );
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ QK-norm+RoPE probe should count as one Rust FFI dispatch"
        );

        let bhsd = as_strided(
            &proj,
            &[1, n_heads, seq, head_dim],
            &[
                i64::from(seq * width),
                i64::from(head_dim),
                i64::from(width),
                1,
            ],
            0,
            None,
        );
        let normed = crate::fast::rms_norm(&bhsd, Some(&norm), 1.0e-6, None);
        let reference =
            crate::fast::rope(&normed, head_dim, false, Some(10_000.0), 1.0, 2, None, None);
        let direct = contiguous(&direct, None);
        let reference = contiguous(&reference, None);
        eval(&[&direct, &reference]);

        assert_eq!(direct.shape(), vec![1, n_heads, seq, head_dim]);
        assert_eq!(reference.shape(), direct.shape());
        assert_close_f32(direct.data_f32(), reference.data_f32(), 1.0e-6);
    }

    #[test]
    fn gelu_approx_quantized_ffn_direct_matches_portable_composition() {
        let x_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.03125).collect();
        let gate_up_weight_data: Vec<f32> =
            (0..2048).map(|i| ((i as f32) - 1024.0) * 0.0005).collect();
        let down_weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 512.0) * 0.0005).collect();
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[2, 32],
            MlxDtype::Float32,
        );
        let gate_up_weight = MlxArray::from_raw_data(
            gate_up_weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(gate_up_weight_data.as_slice()),
            &[64, 32],
            MlxDtype::Float32,
        );
        let down_weight = MlxArray::from_raw_data(
            down_weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(down_weight_data.as_slice()),
            &[32, 32],
            MlxDtype::Float32,
        );
        let gate_up_q = quantize(
            &gate_up_weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let down_q = quantize(
            &down_weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert_eq!(gate_up_q.len(), 3);
        assert_eq!(down_q.len(), 3);

        let prev = crate::op_count::op_count_snapshot();
        let direct = gelu_approx_quantized_ffn(
            &x,
            &gate_up_q[0],
            &gate_up_q[1],
            Some(&gate_up_q[2]),
            &down_q[0],
            &down_q[1],
            Some(&down_q[2]),
            32,
            4,
            None,
        );
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ quantized FFN shim should count as one Rust FFI dispatch"
        );

        let gate_up = quantized_matmul(
            &x,
            &gate_up_q[0],
            &gate_up_q[1],
            Some(&gate_up_q[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let gate = slice_last_dim(&gate_up, 0, 32, None);
        let up = slice_last_dim(&gate_up, 32, 64, None);
        let hidden = gelu_approx_mul(&gate, &up, None);
        let portable = quantized_matmul(
            &hidden,
            &down_q[0],
            &down_q[1],
            Some(&down_q[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        eval(&[&direct, &portable]);

        assert_eq!(direct.shape(), vec![2, 32]);
        assert_eq!(
            direct.data_f32().to_vec(),
            portable.data_f32().to_vec(),
            "direct C++ quantized FFN shim must preserve portable math"
        );
    }

    #[test]
    fn quantize_affine_round_trip_reports_expected_shapes() {
        let values = (0..128)
            .map(|i| (i as f32 - 64.0) / 64.0)
            .collect::<Vec<_>>();
        let w = MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values.as_slice()),
            &[2, 64],
            MlxDtype::Float32,
        );

        let quantized = quantize(
            &w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );

        assert_eq!(quantized.len(), 3);
        assert_eq!(quantized[0].shape(), vec![2, 8]);
        assert_eq!(quantized[0].dtype(), MlxDtype::Uint32);
        assert_eq!(quantized[1].shape(), vec![2, 2]);
        assert_eq!(quantized[1].dtype(), MlxDtype::Float32);
        assert_eq!(quantized[2].shape(), vec![2, 2]);
        assert_eq!(quantized[2].dtype(), MlxDtype::Float32);

        let restored = dequantize_with_mode(
            &quantized[0],
            &quantized[1],
            Some(&quantized[2]),
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            Some(MlxDtype::Float32),
            None,
        );
        eval(&[&restored]);

        assert_eq!(restored.shape(), vec![2, 64]);
        assert_eq!(restored.dtype(), MlxDtype::Float32);
    }

    #[test]
    fn fp8_conversion_wrappers_preserve_shape_contract() {
        let values = [0.0_f32, 1.0, -2.0, 4.0];
        let x = MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(&values),
            &[2, 2],
            MlxDtype::Float32,
        );

        let fp8 = to_fp8(&x, None);
        eval(&[&fp8]);
        assert_eq!(fp8.shape(), vec![2, 2]);
        assert_eq!(fp8.dtype(), MlxDtype::Uint8);

        let restored = from_fp8(&fp8, MlxDtype::Float32, None);
        eval(&[&restored]);
        assert_eq!(restored.shape(), vec![2, 2]);
        assert_eq!(restored.dtype(), MlxDtype::Float32);
    }

    fn assert_close_f32(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= tolerance,
                "mismatch at {idx}: actual={a}, expected={e}, delta={delta}, tolerance={tolerance}"
            );
        }
    }
}
