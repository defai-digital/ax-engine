//! AX Engine high-level Rust SDK facade.
//!
//! This crate provides the stable, high-level integration surface for Rust
//! applications that want to consume AX Engine without depending directly on
//! `ax-engine-core` internals.

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-engine only supports aarch64-apple-darwin (Apple Silicon M3+)");

mod llama_cpp_process;
mod model;
mod routing;
mod session;

pub use model::{BackendKind, LoadOptions, Model, ModelInfo};
pub use routing::{InferenceBackendKind, RoutingPreview, preview_routing};
pub use session::{
    ChatMessage, ChatRole, FinishReason, GenerationOptions, GenerationOutput, Session,
    SessionOptions, TextStream, Usage,
};
