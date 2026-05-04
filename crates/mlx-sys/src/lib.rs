//! Safe Rust wrappers around the mlx-c C API.
//!
//! The raw FFI bindings are in the `ffi` module. All other modules expose
//! safe RAII types and functions. Callers should use the safe layer only.

#![allow(unsafe_code)]

pub mod array;
pub mod fast;
pub mod io;
pub mod metal;
pub mod ops;
pub mod stream;
pub mod transforms;

/// Raw auto-generated FFI bindings. Prefer the safe modules above.
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
pub use fast::{
    ScaledDotProductAttentionMask, rms_norm, rope, scaled_dot_product_attention,
    scaled_dot_product_attention_with_mask,
};
pub use io::load_safetensors;
pub use metal::{KernelOutputSpec, KernelTemplateArg, MlxMetalKernel};
pub use ops::{
    add, arange, argmax, argpartition_axis, argsort_axis, as_strided, astype, clip, concatenate,
    contiguous, conv1d, dequantize, divide, erf, exp, expand_dims, expand_dims_axes, gather_mm,
    gather_qmm, gelu, gelu_approx, greater_equal, less, log, log1p, logical_and, matmul, maximum, minimum,
    multiply, negative, power, quantized_matmul, repeat_axis, reshape, slice, slice_last_dim,
    slice_update, softmax, stack, subtract, sum_axis, take, take_along_axis, tanh, transpose,
    where_cond, zeros,
};
pub use stream::MlxStream;
pub use transforms::{
    async_eval, clear_cache, enable_compile, eval, max_recommended_working_set_size,
    set_wired_limit,
};
