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
pub use fast::{rms_norm, rope, scaled_dot_product_attention};
pub use io::load_safetensors;
pub use metal::MlxMetalKernel;
pub use ops::{
    add, argmax, as_strided, astype, concatenate, dequantize, expand_dims, matmul, multiply,
    quantized_matmul, repeat_axis, reshape, slice, slice_last_dim, softmax, take, transpose,
};
pub use stream::MlxStream;
pub use transforms::{async_eval, clear_cache, enable_compile, eval};
