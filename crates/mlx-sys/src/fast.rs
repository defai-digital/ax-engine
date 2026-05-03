use std::ffi::CString;
use std::ptr;

use crate::array::MlxArray;
use crate::ffi;
use crate::stream::MlxStream;

fn gpu() -> ffi::mlx_stream {
    unsafe { ffi::mlx_default_gpu_stream_new() }
}

/// RMS layer normalization.
pub fn rms_norm(x: &MlxArray, weight: Option<&MlxArray>, eps: f32, s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let weight_raw = weight
            .map(|w| w.inner)
            .unwrap_or(ffi::mlx_array { ctx: ptr::null_mut() });
        let mut res = MlxArray::empty();
        ffi::mlx_fast_rms_norm(&mut res.inner, x.inner, weight_raw, eps, stream);
        res
    }
}

/// Rotary position embedding.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    x: &MlxArray,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: i32,
    freqs: Option<&MlxArray>,
    s: Option<&MlxStream>,
) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let base_opt = ffi::mlx_optional_float_ {
            has_value: base.is_some(),
            value: base.unwrap_or(10000.0),
        };
        let freqs_raw = freqs
            .map(|f| f.inner)
            .unwrap_or(ffi::mlx_array { ctx: ptr::null_mut() });
        let mut res = MlxArray::empty();
        ffi::mlx_fast_rope(
            &mut res.inner,
            x.inner,
            dims,
            traditional,
            base_opt,
            scale,
            offset,
            freqs_raw,
            stream,
        );
        res
    }
}

/// Scaled dot-product attention (flash attention).
///
/// `causal`: when true applies a causal (lower-triangular) mask; required for
/// prefill (seq > 1). During single-token decode no mask is needed.
pub fn scaled_dot_product_attention(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    causal: bool,
    s: Option<&MlxStream>,
) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(gpu);
        let mask_mode = if causal {
            CString::new("causal").unwrap()
        } else {
            CString::new("").unwrap()
        };
        let null_arr = ffi::mlx_array { ctx: ptr::null_mut() };
        let mut res = MlxArray::empty();
        ffi::mlx_fast_scaled_dot_product_attention(
            &mut res.inner,
            queries.inner,
            keys.inner,
            values.inner,
            scale,
            mask_mode.as_ptr(),
            null_arr, // mask_arr
            null_arr, // sinks
            stream,
        );
        res
    }
}
