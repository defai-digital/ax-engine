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
binary_op!(multiply, mlx_multiply);
binary_op!(matmul, mlx_matmul);

unary_op!(sigmoid, mlx_sigmoid);

/// silu(x) = x * sigmoid(x)
pub fn silu(x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    let sig = sigmoid(x, s);
    multiply(x, &sig, s)
}

pub fn astype(a: &MlxArray, dtype: MlxDtype, s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mut res = MlxArray::empty();
        ffi::mlx_astype(&mut res.inner, a.inner, dtype.to_ffi(), stream);
        res
    }
}

pub fn reshape(a: &MlxArray, shape: &[i32], s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mut res = MlxArray::empty();
        ffi::mlx_reshape(
            &mut res.inner,
            a.inner,
            shape.as_ptr(),
            shape.len(),
            stream,
        );
        res
    }
}

pub fn transpose(a: &MlxArray, axes: &[i32], s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mut res = MlxArray::empty();
        ffi::mlx_transpose_axes(
            &mut res.inner,
            a.inner,
            axes.as_ptr(),
            axes.len(),
            stream,
        );
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
pub fn slice(a: &MlxArray, start: &[i32], stop: &[i32], strides: &[i32], s: Option<&MlxStream>) -> MlxArray {
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
    assert!(ndim > 0, "slice_last_dim requires at least 1-dimensional array");
    let shape = a.shape();
    let mut start_vec = vec![0i32; ndim];
    let mut stop_vec = shape.clone();
    let strides_vec = vec![1i32; ndim];
    start_vec[ndim - 1] = start;
    stop_vec[ndim - 1] = end;
    slice(a, &start_vec, &stop_vec, &strides_vec, s)
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
        let biases_raw = biases
            .map(|b| b.inner)
            .unwrap_or(ffi::mlx_array { ctx: std::ptr::null_mut() });
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
            ffi::mlx_array { ctx: std::ptr::null_mut() },
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
        let biases_raw = biases
            .map(|b| b.inner)
            .unwrap_or(ffi::mlx_array { ctx: std::ptr::null_mut() });

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
