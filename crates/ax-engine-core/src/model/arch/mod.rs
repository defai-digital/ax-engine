//! Architecture-specific forward-pass implementations and helpers.

pub mod gemma3;
pub mod llama;
pub mod qwen3_5;
pub use qwen3_5 as qwen35;
