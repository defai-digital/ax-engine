//! Context creation, free, KV cache management.

// v2: KV created via LlamaModel::create_model_kv_for_weights()

use ax_engine_core::backend::KvPlannerRequirements;
use ax_engine_core::model::WeightStore;

use crate::types::*;

fn resolve_context_len(requested: u32, model_max: u32) -> u32 {
    if requested == 0 {
        model_max
    } else {
        requested.min(model_max)
    }
}

/// Create a new inference context from a loaded model.
///
/// Returns null on failure.
#[unsafe(no_mangle)]
pub extern "C" fn llama_new_context_with_model(
    model: *mut LlamaModel,
    params: LlamaContextParams,
) -> *mut LlamaContext {
    if model.is_null() {
        tracing::error!("llama_new_context_with_model: null model");
        return std::ptr::null_mut();
    }

    let model_ref = unsafe { &*model };

    // Use requested context size, clamped to model's maximum
    let n_ctx = resolve_context_len(params.n_ctx, model_ref.config.context_length);

    let weights = WeightStore::new(&model_ref.mapped);
    let kv = match model_ref
        .model
        .create_model_kv_for_weights_with_requirements(
            &weights,
            KvPlannerRequirements {
                max_seq_len_override: Some(n_ctx as usize),
                ..Default::default()
            },
        ) {
        Ok(kv) => kv,
        Err(e) => {
            tracing::error!("llama_new_context_with_model: failed to create KV: {e}");
            return std::ptr::null_mut();
        }
    };

    let logits = vec![0.0f32; model_ref.config.vocab_size as usize];

    // Seed the RNG
    let rng_state = if params.seed == u32::MAX {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    } else {
        params.seed as u64
    };

    tracing::info!("Context created: n_ctx={n_ctx}");

    Box::into_raw(Box::new(LlamaContext {
        model: model as *const LlamaModel,
        kv,
        logits,
        position: 0,
        n_ctx,
        rng_state,
    }))
}

/// Free an inference context.
#[unsafe(no_mangle)]
pub extern "C" fn llama_free(ctx: *mut LlamaContext) {
    if !ctx.is_null() {
        unsafe { drop(Box::from_raw(ctx)) };
        tracing::debug!("Context freed");
    }
}

/// Get context size.
#[unsafe(no_mangle)]
pub extern "C" fn llama_n_ctx(ctx: *const LlamaContext) -> i32 {
    if ctx.is_null() {
        return 0;
    }
    unsafe { (*ctx).n_ctx as i32 }
}

/// Clear the KV cache (reset for new conversation).
#[unsafe(no_mangle)]
pub extern "C" fn llama_kv_cache_clear(ctx: *mut LlamaContext) {
    if ctx.is_null() {
        return;
    }
    unsafe {
        (*ctx).kv.clear();
        (*ctx).position = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_context_len_uses_model_max_when_requested_is_zero() {
        assert_eq!(resolve_context_len(0, 4096), 4096);
    }

    #[test]
    fn test_resolve_context_len_clamps_to_model_max() {
        assert_eq!(resolve_context_len(8192, 4096), 4096);
    }

    #[test]
    fn test_resolve_context_len_preserves_smaller_request() {
        assert_eq!(resolve_context_len(2048, 4096), 2048);
    }
}
