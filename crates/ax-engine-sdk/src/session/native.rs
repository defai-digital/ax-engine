#[cfg(feature = "mlx-native")]
use std::env;

use ax_engine_core::EngineCore;

use super::config::EngineSessionConfig;
#[cfg(any(feature = "mlx-native", test))]
use super::config::MlxMtpPolicy;
use super::errors::EngineSessionError;

#[cfg(feature = "mlx-native")]
use ax_engine_mlx::{MlxPrefixCacheStore, MlxSharedWeightsCell, WhisperModel};

#[cfg(any(feature = "mlx-native", test))]
const PREFIX_REUSE_DISABLED_ENV: &str = "AX_ENGINE_PREFIX_REUSE_DISABLED";

#[cfg(any(feature = "mlx-native", test))]
fn prefix_reuse_disabled_value(value: &str) -> bool {
    matches!(value.trim(), "1")
        || value.trim().eq_ignore_ascii_case("true")
        || value.trim().eq_ignore_ascii_case("yes")
}

#[cfg(feature = "mlx-native")]
fn native_prefix_reuse_enabled() -> bool {
    !env::var(PREFIX_REUSE_DISABLED_ENV)
        .ok()
        .as_deref()
        .is_some_and(prefix_reuse_disabled_value)
}

/// Apply session-level scheduler policy knobs onto a freshly built core.
fn apply_scheduler_policy(core: &mut EngineCore, config: &EngineSessionConfig) {
    core.set_multi_prefill_fair(
        config.multi_prefill_fair,
        config.max_prefill_tokens_per_request_per_step,
        config.max_inflight_prefill_requests,
    );
}

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
        let mut core = EngineCore::with_kv_config(config.kv_config);
        apply_scheduler_policy(&mut core, config);
        return Ok(core);
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
        let mut core = EngineCore::with_kv_config(config.kv_config);
        apply_scheduler_policy(&mut core, config);
        return Ok(core);
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

    // SSD expert streaming admission (--stream-experts / AX_STREAM_EXPERTS):
    // latch before `load_weights` reads it. Default Auto streams required
    // packs and packs that cannot fit; Off fails closed on required packs.
    ax_engine_mlx::expert_stream::set_stream_experts_mode(match config.mlx_stream_experts {
        crate::MlxStreamExpertsMode::Off => ax_engine_mlx::expert_stream::StreamExpertsMode::Off,
        crate::MlxStreamExpertsMode::Auto => ax_engine_mlx::expert_stream::StreamExpertsMode::Auto,
        crate::MlxStreamExpertsMode::On => ax_engine_mlx::expert_stream::StreamExpertsMode::On,
    });

    let model_dir = config
        .mlx_model_artifacts_dir()
        .ok_or(EngineSessionError::MlxRuntimeArtifactsRequired)?;

    let artifacts = NativeModelArtifacts::from_dir_or_convert(model_dir)
        .map_err(|e| EngineSessionError::MetalRuntime(e.into()))?;

    // Whisper is an encoder/decoder speech model with its own decoding loop,
    // not an autoregressive `ExecutionRunner`. Keep the scheduler/session
    // shell alive for server lifecycle management while `EngineSession`
    // owns the dedicated speech runtime.
    if artifacts.manifest().model_family == "whisper" {
        let mut core = EngineCore::with_kv_config(config.kv_config);
        apply_scheduler_policy(&mut core, config);
        return Ok(core);
    }

    let prefill_chunk = config
        .mlx_prefill_chunk
        .map(|n| n.max(1))
        .unwrap_or(DEFAULT_PREFILL_CHUNK);

    let mut runner = MlxRunner::from_artifacts_with_runtime_shares(
        &artifacts,
        prefill_chunk,
        config.mlx_disable_ngram_acceleration,
        config.mlx_mtp_disable_ngram_stacking,
        prefix_cache_store,
        shared_weights,
    )
    .map_err(|e| {
        EngineSessionError::MetalRuntime(ax_engine_core::MetalRuntimeError::Generic(e.to_string()))
    })?;
    if config.mlx_mtp_policy == MlxMtpPolicy::Required && !runner.has_mtp() {
        return Err(EngineSessionError::MlxMtpRequiredButUnavailable);
    }
    runner.set_mtp_requested(mtp_requested_for_policy(
        config.mlx_mtp_policy,
        runner.mtp_certified_default_on(),
    ));
    // Couple PR4 FA private pool capacity to the session logical block table
    // when the opt-in flag is engaged (default remains OFF / contiguous).
    runner.align_fa_block_pool_to_kv(
        config.kv_config.block_size_tokens,
        config.kv_config.total_blocks,
    );

    let mut core =
        EngineCore::with_runtime_components(config.kv_config, runner, DeterministicSampler);
    core.set_prefix_reuse_enabled(native_prefix_reuse_enabled());
    apply_scheduler_policy(&mut core, config);
    // ADR-038: bind generation strategy so the scheduler plans DenoiseStep for
    // diffusion models (and PrefillChunk / TokenDecode for AR) per request.
    core.set_generation_kind(artifacts.manifest().generation_kind());
    Ok(core)
}

/// Map the session MTP policy to the runner's `set_mtp_requested` argument.
///
/// `Disabled` and `Required` are explicit and unconditional (`Required`
/// availability is checked separately before this call). `Auto` — the SDK
/// and server default — follows the pack's publisher certification so an
/// uncertified pack decodes direct by default instead of paying an
/// unmeasured MTP verify tax. `set_mtp_requested` itself still enforces
/// route safety and the `AX_NO_SPEC` kill switch.
#[cfg(any(feature = "mlx-native", test))]
fn mtp_requested_for_policy(policy: MlxMtpPolicy, certified_default_on: bool) -> bool {
    match policy {
        MlxMtpPolicy::Disabled => false,
        MlxMtpPolicy::Required => true,
        MlxMtpPolicy::Auto => certified_default_on,
    }
}

#[cfg(feature = "mlx-native")]
pub(super) fn load_native_whisper_model(
    config: &EngineSessionConfig,
) -> Result<Option<WhisperModel>, EngineSessionError> {
    if config.resolved_backend.selected_backend != crate::backend::SelectedBackend::Mlx {
        return Ok(None);
    }
    let model_dir = config
        .mlx_model_artifacts_dir()
        .ok_or(EngineSessionError::MlxRuntimeArtifactsRequired)?;
    let artifacts = ax_engine_core::NativeModelArtifacts::from_dir(model_dir)
        .map_err(|error| EngineSessionError::MetalRuntime(error.into()))?;
    if artifacts.manifest().model_family != "whisper" {
        return Ok(None);
    }
    WhisperModel::load(&artifacts)
        .map(Some)
        .map_err(|error| EngineSessionError::WhisperFailed {
            message: error.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_reuse_disable_values_are_explicit() {
        for value in ["1", "true", "TRUE", "yes", " YES "] {
            assert!(prefix_reuse_disabled_value(value));
        }
        for value in ["", "0", "false", "no", "enabled"] {
            assert!(!prefix_reuse_disabled_value(value));
        }
    }

    #[test]
    fn mtp_policy_tri_state_respects_pack_certification() {
        for certified in [false, true] {
            assert!(!mtp_requested_for_policy(MlxMtpPolicy::Disabled, certified));
            assert!(mtp_requested_for_policy(MlxMtpPolicy::Required, certified));
            assert_eq!(
                mtp_requested_for_policy(MlxMtpPolicy::Auto, certified),
                certified,
                "Auto must follow the pack certification verdict"
            );
        }
    }
}
