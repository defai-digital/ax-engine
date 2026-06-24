//! Internal MLX runtime boundary for AX Engine.
//!
//! This crate intentionally contains both the raw bindgen output from
//! `ax_shim.h` and the safe Rust wrappers used by the rest of the workspace.
//! That is a pragmatic internal layout: the `-sys` suffix normally means raw
//! FFI only, but AX Engine currently keeps the safe layer here to avoid an
//! extra crate boundary around code that is not published as an independent
//! API.
//!
//! Callers should use the safe modules and re-exports such as [`MlxArray`],
//! [`MlxStream`], ops, transforms, safetensors loading, and Metal kernel
//! wrappers. The raw [`ffi`] module is public only for wrapper implementation
//! and emergency escape hatches. New runtime code should add a safe wrapper
//! instead of calling `ffi` directly.
//!
//! If this crate becomes a published or shared dependency, split the raw
//! bindings into `mlx-sys` and move the safe layer into a wrapper crate such as
//! `mlx-rs` or `mlx-runtime`.

#![allow(unsafe_code)]

pub mod array;
pub mod closure;
pub mod error;
pub mod fast;
pub mod io;
pub mod mempressure;
pub mod metal;
pub mod op_count;
pub mod ops;
pub mod stream;
pub mod transforms;

/// Raw auto-generated FFI bindings.
///
/// Prefer the safe modules above. This module remains public so wrappers can
/// cover new MLX APIs incrementally without changing crate boundaries.
#[doc(hidden)]
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    clippy::all
)]
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use array::{MlxArray, MlxDtype};
pub use closure::{MlxClosure, MlxVectorArray};
pub use error::{install_recoverable_error_handler, take_last_error};
pub use fast::{
    ScaledDotProductAttentionMask, rms_norm, rope, scaled_dot_product_attention,
    scaled_dot_product_attention_with_mask,
};
pub use io::{load_safetensors, load_safetensors_mmap};
pub use mempressure::{
    device_active_bytes, device_cache_bytes, device_peak_bytes,
    device_recommended_working_set_bytes, host_resident_bytes,
};
pub use metal::{KernelOutputSpec, KernelTemplateArg, MlxMetalKernel};
pub use op_count::{op_count_snapshot, op_count_take};
pub use ops::{
    MlxQuantizationMode, add, add_rms_norm_pair, arange, argmax, argpartition_axis, argsort_axis,
    as_strided, astype, broadcast_to, clip, concatenate, contiguous, conv1d, cos, cumsum,
    dequantize, dequantize_with_mode, divide, equal, erf, exp, expand_dims, expand_dims_axes,
    flatten, floor, from_fp8, gather_mm, gather_qmm, gelu, gelu_approx, gelu_approx_mul,
    gelu_approx_mul_quantized_matmul, gemma4_post_attn_ffn_block, greater_equal, layer_norm, less,
    less_equal, log, log1p, logical_and, matmul, maximum, minimum, multiply, negative, not_equal,
    outer, pad, power, put_along_axis, qk_norm_rope_bhsd_from_proj, quantize, quantized_matmul,
    quantized_matmul_rms_norm, qwen_linear_attention_inputs_packed,
    qwen_linear_attention_post_input, random_categorical, repeat, repeat_axis, reshape, sigmoid,
    silu_mul, sin, slice, slice_last_dim, slice_update, softmax, softmax_precise, split, stack,
    stop_gradient, subtract, sum_axis, take, take_along_axis, tanh, to_fp8, topk, topk_axis,
    transpose, unflatten, view, where_cond, zeros,
};
pub use stream::MlxStream;
pub use transforms::{
    async_eval, clear_cache, enable_compile, eval, eval_first_u32, get_cache_memory,
    get_peak_memory, max_recommended_working_set_size, reset_peak_memory, set_cache_limit,
    set_memory_limit, set_wired_limit, try_eval,
};
