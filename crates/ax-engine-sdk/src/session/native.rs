use ax_engine_core::EngineCore;

use super::config::EngineSessionConfig;
use super::errors::EngineSessionError;

pub(super) fn build_native_core(
    config: &EngineSessionConfig,
) -> Result<EngineCore, EngineSessionError> {
    #[cfg(feature = "mlx-native")]
    if config.resolved_backend.selected_backend == crate::backend::SelectedBackend::Mlx {
        return build_mlx_core(config);
    }

    if !config.resolved_backend.selected_backend.is_mlx() {
        return Ok(EngineCore::with_kv_config(config.kv_config));
    }

    Err(EngineSessionError::MlxRuntimeUnavailable)
}

#[cfg(feature = "mlx-native")]
fn build_mlx_core(config: &EngineSessionConfig) -> Result<EngineCore, EngineSessionError> {
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

    let runner = MlxRunner::from_artifacts(
        &artifacts,
        prefill_chunk,
        config.mlx_disable_ngram_acceleration,
        config.mlx_kv_compression,
    )
    .map_err(|e| {
        EngineSessionError::MetalRuntime(ax_engine_core::MetalRuntimeError::Generic(e.to_string()))
    })?;

    Ok(EngineCore::with_runtime_components(
        config.kv_config,
        runner,
        DeterministicSampler,
    ))
}
