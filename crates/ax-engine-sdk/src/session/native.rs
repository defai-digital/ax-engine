use ax_engine_core::EngineCore;

use super::config::EngineSessionConfig;
use super::errors::EngineSessionError;

#[cfg(feature = "mlx-native")]
use ax_engine_mlx::{MlxPrefixCacheStore, MlxSharedWeightsCell};

#[cfg(feature = "mlx-native")]
pub(super) fn build_native_core(
    config: &EngineSessionConfig,
) -> Result<EngineCore, EngineSessionError> {
    build_native_core_with_mlx_shares(config, None, None)
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
pub(super) fn build_native_core_with_mlx_shares(
    config: &EngineSessionConfig,
    prefix_cache_store: Option<MlxPrefixCacheStore>,
    shared_weights: Option<&MlxSharedWeightsCell>,
) -> Result<EngineCore, EngineSessionError> {
    if config.resolved_backend.selected_backend == crate::backend::SelectedBackend::Mlx {
        return build_mlx_core(config, prefix_cache_store, shared_weights);
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
    shared_weights: Option<&MlxSharedWeightsCell>,
) -> Result<EngineCore, EngineSessionError> {
    use ax_engine_core::{DeterministicSampler, NativeModelArtifacts};
    use ax_engine_mlx::{MlxRunner, generate::DEFAULT_PREFILL_CHUNK};

    // Install the speculation-profile override (ADR-022) from the resolved CLI
    // flag before the runner reads it. This is a safe alternative to mutating the
    // process environment, which the server/SDK crates cannot do under
    // `unsafe_code = "forbid"`. An unparsable name is ignored (falls back to env
    // / `auto`).
    if let Some(name) = config.mlx_speculation_profile.as_deref() {
        if let Some(profile) = ax_engine_mlx::speculation_profile::SpeculationProfile::parse(name) {
            ax_engine_mlx::speculation_profile::set_speculation_profile_override(profile);
        }
    }

    let model_dir = config
        .mlx_model_artifacts_dir()
        .ok_or(EngineSessionError::MlxRuntimeArtifactsRequired)?;

    let artifacts = NativeModelArtifacts::from_dir(model_dir)
        .map_err(|e| EngineSessionError::MetalRuntime(e.into()))?;

    let prefill_chunk = config
        .mlx_prefill_chunk
        .map(|n| n.max(1))
        .unwrap_or(DEFAULT_PREFILL_CHUNK);

    let runner = MlxRunner::from_artifacts_with_runtime_shares(
        &artifacts,
        prefill_chunk,
        config.mlx_disable_ngram_acceleration,
        config.mlx_mtp_disable_ngram_stacking,
        config.mlx_kv_compression,
        prefix_cache_store,
        shared_weights,
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
