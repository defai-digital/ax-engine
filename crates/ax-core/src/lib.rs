// AX Engine v2 — Core inference engine
//
// Mac M3+ only. Compile-time platform gate.

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-engine only supports aarch64-apple-darwin (Apple Silicon M3+)");

pub mod backend;
pub mod compute;
pub mod gguf;
pub mod kv;
pub mod memory;
pub mod metrics;
pub mod model;
pub mod quant;
pub mod sampling;
pub mod scheduler;
pub mod speculative;
pub mod thermal;
pub mod tokenizer;
