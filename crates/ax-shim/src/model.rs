//! Model load/free and model-level queries.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::Path;

use ax_core::backend::{BackendConfig, create_backend};
use ax_core::gguf::MappedModel;
use ax_core::model::{LlamaModel as CoreLlamaModel, ModelConfig};
use ax_core::tokenizer::Tokenizer;

use crate::types::*;

/// Load a GGUF model from file.
///
/// Returns null on failure.
#[unsafe(no_mangle)]
pub extern "C" fn llama_model_load_from_file(
    path: *const c_char,
    _params: LlamaModelParams,
) -> *mut LlamaModel {
    if path.is_null() {
        tracing::error!("llama_model_load_from_file: null path");
        return std::ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("llama_model_load_from_file: invalid UTF-8 path: {e}");
            return std::ptr::null_mut();
        }
    };

    let mapped = match MappedModel::open(Path::new(path_str)) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("llama_model_load_from_file: failed to open: {e}");
            return std::ptr::null_mut();
        }
    };

    let config = match ModelConfig::from_gguf(&mapped.header) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("llama_model_load_from_file: failed to parse config: {e}");
            return std::ptr::null_mut();
        }
    };

    let tokenizer = match Tokenizer::from_gguf(&mapped.header) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("llama_model_load_from_file: failed to parse tokenizer: {e}");
            return std::ptr::null_mut();
        }
    };

    // Validate architecture before with_backend (which panics on unsupported arch)
    if let Err(e) = ax_core::model::arch_registry::forward_for_arch(&config.architecture) {
        tracing::error!("llama_model_load_from_file: {e}");
        return std::ptr::null_mut();
    }

    // Use Hybrid backend (Metal GPU + CPU) when GPU layers requested, else CPU only
    let backend_config = if _params.n_gpu_layers > 0 {
        BackendConfig::default() // Hybrid
    } else {
        BackendConfig::Cpu
    };
    let backend = match create_backend(backend_config) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("Failed to create GPU backend, falling back to CPU: {e}");
            match create_backend(BackendConfig::Cpu) {
                Ok(b) => b,
                Err(e) => {
                    tracing::error!("Failed to create CPU backend: {e}");
                    return std::ptr::null_mut();
                }
            }
        }
    };
    let model = CoreLlamaModel::with_backend(config.clone(), backend);

    tracing::info!(
        "Model loaded: {} layers, {} vocab, {:.0}MB",
        config.n_layers,
        config.vocab_size,
        mapped.total_tensor_bytes() as f64 / 1024.0 / 1024.0,
    );

    Box::into_raw(Box::new(LlamaModel {
        mapped,
        config,
        tokenizer,
        model,
    }))
}

/// Free a loaded model.
#[unsafe(no_mangle)]
pub extern "C" fn llama_free_model(model: *mut LlamaModel) {
    if !model.is_null() {
        unsafe { drop(Box::from_raw(model)) };
        tracing::debug!("Model freed");
    }
}

/// Get vocabulary size.
#[unsafe(no_mangle)]
pub extern "C" fn llama_n_vocab(model: *const LlamaModel) -> i32 {
    if model.is_null() {
        return 0;
    }
    unsafe { (*model).config.vocab_size as i32 }
}

/// Get number of layers.
#[unsafe(no_mangle)]
pub extern "C" fn llama_n_layer(model: *const LlamaModel) -> i32 {
    if model.is_null() {
        return 0;
    }
    unsafe { (*model).config.n_layers as i32 }
}

/// Get embedding dimension.
#[unsafe(no_mangle)]
pub extern "C" fn llama_n_embd(model: *const LlamaModel) -> i32 {
    if model.is_null() {
        return 0;
    }
    unsafe { (*model).config.embedding_dim as i32 }
}

/// Get BOS token ID.
#[unsafe(no_mangle)]
pub extern "C" fn llama_token_bos(model: *const LlamaModel) -> LlamaToken {
    if model.is_null() {
        return -1;
    }
    unsafe { (*model).tokenizer.bos_id() as LlamaToken }
}

/// Get EOS token ID.
#[unsafe(no_mangle)]
pub extern "C" fn llama_token_eos(model: *const LlamaModel) -> LlamaToken {
    if model.is_null() {
        return -1;
    }
    unsafe { (*model).tokenizer.eos_id() as LlamaToken }
}
