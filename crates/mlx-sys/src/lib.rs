//! Internal MLX runtime boundary for AX Engine.
//!
//! This crate intentionally contains both the raw bindgen output for `mlx-c`
//! and the safe Rust wrappers used by the rest of the workspace. That is a
//! pragmatic internal layout: the `-sys` suffix normally means raw FFI only,
//! but AX Engine currently keeps the safe layer here to avoid an extra crate
//! boundary around code that is not published as an independent API.
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
/// cover new `mlx-c` APIs incrementally without changing crate boundaries.
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
pub use fast::{
    ScaledDotProductAttentionMask, rms_norm, rope, scaled_dot_product_attention,
    scaled_dot_product_attention_with_mask,
};
pub use io::{load_safetensors, load_safetensors_mmap};
pub use mempressure::{
    device_active_bytes, device_recommended_working_set_bytes, host_resident_bytes,
};
pub use metal::{KernelOutputSpec, KernelTemplateArg, MlxMetalKernel};
pub use op_count::{op_count_snapshot, op_count_take};
pub use ops::{
    MlxQuantizationMode, add, arange, argmax, argpartition_axis, argsort_axis, as_strided, astype,
    broadcast_to, clip, concatenate, contiguous, conv1d, cos, dequantize, dequantize_with_mode,
    divide, erf, exp, expand_dims, expand_dims_axes, flatten, floor, from_fp8, gather_mm,
    gather_qmm, gelu, gelu_approx, gelu_approx_mul, gemma4_post_attn_ffn_block, greater_equal,
    less, log, log1p, logical_and, matmul, maximum, minimum, multiply, negative, outer, pad, power,
    put_along_axis, qk_norm_rope_bhsd_from_proj, quantize, quantized_matmul,
    qwen_linear_attention_inputs_packed, random_categorical, repeat, repeat_axis, reshape, sin,
    slice, slice_last_dim, slice_update, softmax, softmax_precise, split, stack, stop_gradient,
    subtract, sum_axis, take, take_along_axis, tanh, to_fp8, topk, topk_axis, transpose, unflatten,
    where_cond, zeros,
};
pub use stream::MlxStream;
pub use transforms::{
    async_eval, clear_cache, enable_compile, eval, eval_first_u32,
    max_recommended_working_set_size, set_wired_limit,
};
