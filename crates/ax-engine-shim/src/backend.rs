//! Backend lifecycle: init/free and default params.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::types::*;

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Initialize the AX Engine backend. Must be called before any other function.
#[unsafe(no_mangle)]
pub extern "C" fn llama_backend_init() {
    if INITIALIZED.swap(true, Ordering::SeqCst) {
        return; // Already initialized
    }
    // Use try_init to avoid panic if tracing was already set up
    let _ = tracing_subscriber::fmt().try_init();
    tracing::info!("AX Engine backend initialized");
}

/// Free the AX Engine backend resources.
#[unsafe(no_mangle)]
pub extern "C" fn llama_backend_free() {
    INITIALIZED.store(false, Ordering::SeqCst);
    tracing::info!("AX Engine backend freed");
}

/// Return default model params.
#[unsafe(no_mangle)]
pub extern "C" fn llama_model_default_params() -> LlamaModelParams {
    LlamaModelParams::default()
}

/// Return default context params.
#[unsafe(no_mangle)]
pub extern "C" fn llama_context_default_params() -> LlamaContextParams {
    LlamaContextParams::default()
}
