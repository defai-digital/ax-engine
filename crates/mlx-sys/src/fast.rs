use std::ffi::CString;

use crate::array::{MlxArray, null_ffi_array};
use crate::ffi;
use crate::stream::{MlxStream, default_gpu_raw};

/// Attention mask accepted by MLX fast SDPA.
pub enum ScaledDotProductAttentionMask<'a> {
    None,
    Causal,
    Array(&'a MlxArray),
}

/// RMS layer normalization.
pub fn rms_norm(
    x: &MlxArray,
    weight: Option<&MlxArray>,
    eps: f32,
    s: Option<&MlxStream>,
) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let weight_raw = weight.map(|w| w.inner).unwrap_or_else(null_ffi_array);
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
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let base_opt = ffi::mlx_optional_float_ {
            has_value: base.is_some(),
            value: base.unwrap_or(10000.0),
        };
        let freqs_raw = freqs.map(|f| f.inner).unwrap_or_else(null_ffi_array);
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
    let mask = if causal {
        ScaledDotProductAttentionMask::Causal
    } else {
        ScaledDotProductAttentionMask::None
    };
    scaled_dot_product_attention_with_mask(queries, keys, values, scale, mask, s)
}

/// Scaled dot-product attention with an explicit MLX mask.
pub fn scaled_dot_product_attention_with_mask(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    mask: ScaledDotProductAttentionMask<'_>,
    s: Option<&MlxStream>,
) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mask_mode = match mask {
            ScaledDotProductAttentionMask::Causal => CString::new("causal").unwrap(),
            ScaledDotProductAttentionMask::None | ScaledDotProductAttentionMask::Array(_) => {
                CString::new("").unwrap()
            }
        };
        let null_arr = null_ffi_array();
        let mask_arr = match mask {
            ScaledDotProductAttentionMask::Array(mask) => mask.inner,
            ScaledDotProductAttentionMask::None | ScaledDotProductAttentionMask::Causal => null_arr,
        };
        let mut res = MlxArray::empty();
        ffi::mlx_fast_scaled_dot_product_attention(
            &mut res.inner,
            queries.inner,
            keys.inner,
            values.inner,
            scale,
            mask_mode.as_ptr(),
            mask_arr,
            null_arr, // sinks
            stream,
        );
        res
    }
}
