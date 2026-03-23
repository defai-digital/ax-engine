// AX Engine — llama.cpp C API shim layer
//
// Exports #[no_mangle] extern "C" functions matching llama.h signatures.
// Produces libllama.dylib for drop-in binary compatibility.
//
// All public functions are extern "C" entry points that take raw pointers
// from C callers. They perform null checks internally.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-shim only supports aarch64-apple-darwin (Apple Silicon M3+)");

pub mod backend;
pub mod context;
pub mod eval;
pub mod model;
pub mod sampling;
pub mod tokenize;
pub mod types;
