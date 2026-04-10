//! AX Engine high-level Rust SDK facade.
//!
//! This crate provides the stable, high-level integration surface for Rust
//! applications that want to consume AX Engine without depending directly on
//! `ax-engine-core` internals.

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-engine only supports aarch64-apple-darwin (Apple Silicon M3+)");

pub use ax_engine_core::tokenizer::TokenPiece;

mod model;
mod session;

pub use model::{BackendKind, LoadOptions, Model, ModelInfo};
pub use session::{
    ChatMessage, ChatRole, FinishReason, GenerationOptions, GenerationOutput, PromptCacheStats,
    Session, SessionOptions, SessionSnapshot, TextStream, Usage,
};
