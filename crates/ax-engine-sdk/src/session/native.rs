use ax_engine_core::EngineCore;

use super::config::EngineSessionConfig;
use super::errors::EngineSessionError;

#[cfg(feature = "mlx-native")]
use ax_engine_mlx::MlxPrefixCacheStore;

#[cfg(feature = "mlx-native")]
pub(super) fn build_native_core(
    config: &EngineSessionConfig,
) -> Result<EngineCore, EngineSessionError> {
    build_native_core_with_mlx_prefix_cache(config, None)
}

#[cfg(not(feature = "mlx-native"))]
pub(super) fn build_native_core(
    config: &EngineSessionConfig,
) -> Result<EngineCore, EngineSessionError> {
    if !config.resolved_backend.selected_backend.is_mlx() {
        return Ok(EngineCore::with_kv_config(config.kv_config));
    }

    Err(EngineSessionError::MlxRuntimeUnavailable)
}

#[cfg(feature = "mlx-native")]
pub(super) fn build_native_core_with_mlx_prefix_cache(
    config: &EngineSessionConfig,
    prefix_cache_store: Option<MlxPrefixCacheStore>,
) -> Result<EngineCore, EngineSessionError> {
    if config.resolved_backend.selected_backend == crate::backend::SelectedBackend::Mlx {
        return build_mlx_core(config, prefix_cache_store);
    }

    if !config.resolved_backend.selected_backend.is_mlx() {
        return Ok(EngineCore::with_kv_config(config.kv_config));
    }

    Err(EngineSessionError::MlxRuntimeUnavailable)
}

#[cfg(feature = "mlx-native")]
fn build_mlx_core(
    config: &EngineSessionConfig,
    prefix_cache_store: Option<MlxPrefixCacheStore>,
) -> Result<EngineCore, EngineSessionError> {
    use ax_engine_core::{DeterministicSampler, NativeModelArtifacts};
    use ax_engine_mlx::{MlxRunner, generate::DEFAULT_PREFILL_CHUNK};

    let model_dir = config
        .mlx_model_artifacts_dir()
        .ok_or(EngineSessionError::MlxRuntimeArtifactsRequired)?;

    let artifacts = NativeModelArtifacts::from_dir(model_dir)
        .map_err(|e| EngineSessionError::MetalRuntime(e.into()))?;

    let prefill_chunk = config
        .mlx_prefill_chunk
        .map(|n| n.max(1))
        .unwrap_or(DEFAULT_PREFILL_CHUNK);

    let runner = match prefix_cache_store {
        Some(prefix_cache_store) => MlxRunner::from_artifacts_with_prefix_cache_and_mtp_options(
            &artifacts,
            prefill_chunk,
            config.mlx_disable_ngram_acceleration,
            config.mlx_mtp_disable_ngram_stacking,
            config.mlx_kv_compression,
            prefix_cache_store,
        ),
        None => MlxRunner::from_artifacts_with_mtp_options(
            &artifacts,
            prefill_chunk,
            config.mlx_disable_ngram_acceleration,
            config.mlx_mtp_disable_ngram_stacking,
            config.mlx_kv_compression,
        ),
    }
    .map_err(|e| {
        EngineSessionError::MetalRuntime(ax_engine_core::MetalRuntimeError::Generic(e.to_string()))
    })?;

    Ok(EngineCore::with_runtime_components(
        config.kv_config,
        runner,
        DeterministicSampler,
    ))
}
