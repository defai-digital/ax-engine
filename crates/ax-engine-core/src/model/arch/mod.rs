//! Architecture-specific forward-pass implementations and helpers.

pub mod gemma3;
pub mod gemma4;
pub mod qwen3_5;
pub mod qwen3_moe;
pub use qwen3_5 as qwen35;
