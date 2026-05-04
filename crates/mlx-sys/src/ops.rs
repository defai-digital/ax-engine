use crate::array::{MlxArray, MlxDtype};
use crate::ffi;
use crate::stream::MlxStream;

/// Get current thread's default GPU stream WITHOUT taking ownership.
/// Never call mlx_stream_free on this — it is the thread-local default.
fn gpu() -> ffi::mlx_stream {
    unsafe { ffi::mlx_default_gpu_stream_new() }
}

macro_rules! unary_op {
    ($name:ident, $ffi_fn:ident) => {
        pub fn $name(a: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
            unsafe {
                let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
            unsafe {
                let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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

/// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
///
/// Matches mlx-lm's `nn.gelu_approx` closely enough for inference correctness.
pub fn gelu(x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    let dtype = x.dtype();
    let mk_scalar = |v: f32| {
        let a = MlxArray::from_raw_data(
            &v as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[1_i32],
            MlxDtype::Float32,
        );
        astype(&a, dtype, s)
    };
    let inv_sqrt2 = mk_scalar(std::f32::consts::FRAC_1_SQRT_2);
    let scaled = multiply(x, &inv_sqrt2, s);
    let erf_val = erf(&scaled, s);
    let one_plus_erf = add(&erf_val, &mk_scalar(1.0), s);
    let half_x = multiply(x, &mk_scalar(0.5), s);
    multiply(&half_x, &one_plus_erf, s)
}

pub fn astype(a: &MlxArray, dtype: MlxDtype, s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mut res = MlxArray::empty();
        ffi::mlx_arange(&mut res.inner, start, stop, step, dtype.to_ffi(), stream);
        res
    }
}

pub fn reshape(a: &MlxArray, shape: &[i32], s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mut res = MlxArray::empty();
        ffi::mlx_reshape(&mut res.inner, a.inner, shape.as_ptr(), shape.len(), stream);
        res
    }
}

pub fn transpose(a: &MlxArray, axes: &[i32], s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mut res = MlxArray::empty();
        ffi::mlx_transpose_axes(&mut res.inner, a.inner, axes.as_ptr(), axes.len(), stream);
        res
    }
}

pub fn expand_dims(a: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mut res = MlxArray::empty();
        ffi::mlx_expand_dims(&mut res.inner, a.inner, axis, stream);
        res
    }
}

pub fn expand_dims_axes(a: &MlxArray, axes: &[i32], s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mut res = MlxArray::empty();
        ffi::mlx_expand_dims_axes(&mut res.inner, a.inner, axes.as_ptr(), axes.len(), stream);
        res
    }
}

pub fn softmax(a: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mut res = MlxArray::empty();
        ffi::mlx_softmax_axis(&mut res.inner, a.inner, axis, false, stream);
        res
    }
}

pub fn concatenate(arrays: &[&MlxArray], axis: i32, s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mut res = MlxArray::empty();
        ffi::mlx_contiguous(&mut res.inner, a.inner, false, stream);
        res
    }
}

pub fn clip(a: &MlxArray, min: &MlxArray, max: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let ndim = a.ndim();
        assert!(ndim > 0, "argmax requires at least 1-dimensional array");
        let axis = if ndim > 0 { ndim as i32 - 1 } else { 0 };
        let mut res = MlxArray::empty();
        ffi::mlx_argmax_axis(&mut res.inner, a.inner, axis, false, stream);
        res
    }
}


pub fn argsort_axis(a: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mut res = MlxArray::empty();
        ffi::mlx_argsort_axis(&mut res.inner, a.inner, axis, stream);
        res
    }
}

/// Argpartition along `axis`. Returns indices such that the element at position
/// `kth` would be in its sorted position. Negative kth counts from the end.
pub fn argpartition_axis(a: &MlxArray, kth: i32, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mut res = MlxArray::empty();
        ffi::mlx_take_along_axis(&mut res.inner, a.inner, indices.inner, axis, stream);
        res
    }
}

pub fn sum_axis(a: &MlxArray, axis: i32, keepdims: bool, s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let null_arr = ffi::mlx_array {
            ctx: std::ptr::null_mut(),
        };
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let biases_raw = biases.map(|b| b.inner).unwrap_or(ffi::mlx_array {
            ctx: std::ptr::null_mut(),
        });
        let null_arr = ffi::mlx_array {
            ctx: std::ptr::null_mut(),
        };
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let biases_raw = biases.map(|b| b.inner).unwrap_or(ffi::mlx_array {
            ctx: std::ptr::null_mut(),
        });
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
            ffi::mlx_array {
                ctx: std::ptr::null_mut(),
            },
            no_dtype,
            stream,
        );
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
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let biases_raw = biases.map(|b| b.inner).unwrap_or(ffi::mlx_array {
            ctx: std::ptr::null_mut(),
        });

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
