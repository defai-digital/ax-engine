use std::path::PathBuf;

use ax_engine_core::model::{NativeRuntimeStatus, NativeSourceQuantization};
use ax_engine_core::{NativeModelArtifactsSummary, NativeModelBindingSummary, NativeTensorFormat};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::delegated_http::DelegatedHttpTimeouts;
use crate::llama_cpp::{LlamaCppConfig, LlamaCppServerCompletionConfig};
use crate::mlx_lm::{MlxLmConfig, MlxLmServerCompletionConfig};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionPolicy {
    #[default]
    MlxOnly,
    PreferMlx,
    AllowMlxLmDelegated,
    AllowLlamaCpp,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectedBackend {
    Mlx,
    MlxLmDelegated,
    LlamaCpp,
}

impl SelectedBackend {
    pub const fn is_mlx(self) -> bool {
        matches!(self, Self::Mlx)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportTier {
    MlxCertified,
    MlxPreview,
    MlxLmDelegated,
    LlamaCpp,
    Unsupported,
}

impl SupportTier {
    pub const fn is_mlx(self) -> bool {
        matches!(self, Self::MlxCertified | Self::MlxPreview)
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct BackendPolicy {
    #[serde(default)]
    pub resolution_policy: ResolutionPolicy,
}

impl BackendPolicy {
    pub const fn new(resolution_policy: ResolutionPolicy) -> Self {
        Self { resolution_policy }
    }

    pub const fn mlx_only() -> Self {
        Self::new(ResolutionPolicy::MlxOnly)
    }

    pub const fn prefer_mlx() -> Self {
        Self::new(ResolutionPolicy::PreferMlx)
    }

    pub const fn allow_llama_cpp() -> Self {
        Self::new(ResolutionPolicy::AllowLlamaCpp)
    }

    pub const fn allow_mlx_lm_delegated() -> Self {
        Self::new(ResolutionPolicy::AllowMlxLmDelegated)
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityLevel {
    #[default]
    Unsupported,
    Preview,
    Supported,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PreviewBackendMode {
    #[default]
    Explicit,
    ShippingDefaultLlamaCpp,
    ShippingMlx,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct CapabilityReport {
    #[serde(default)]
    pub text_generation: bool,
    #[serde(default)]
    pub token_streaming: bool,
    #[serde(default)]
    pub deterministic_mode: bool,
    #[serde(default)]
    pub prefix_reuse: bool,
    #[serde(default)]
    pub long_context_validation: CapabilityLevel,
    #[serde(default)]
    pub benchmark_metrics: CapabilityLevel,
}

impl CapabilityReport {
    pub const fn mlx_certified() -> Self {
        Self {
            text_generation: true,
            token_streaming: true,
            deterministic_mode: true,
            prefix_reuse: true,
            long_context_validation: CapabilityLevel::Supported,
            benchmark_metrics: CapabilityLevel::Supported,
        }
    }

    pub const fn mlx_preview() -> Self {
        Self {
            text_generation: true,
            token_streaming: true,
            deterministic_mode: true,
            prefix_reuse: true,
            long_context_validation: CapabilityLevel::Preview,
            benchmark_metrics: CapabilityLevel::Preview,
        }
    }

    pub const fn llama_cpp_baseline() -> Self {
        Self {
            text_generation: true,
            token_streaming: true,
            deterministic_mode: false,
            prefix_reuse: false,
            long_context_validation: CapabilityLevel::Unsupported,
            benchmark_metrics: CapabilityLevel::Unsupported,
        }
    }

    pub const fn mlx_lm_delegated_text() -> Self {
        Self {
            text_generation: true,
            token_streaming: true,
            deterministic_mode: false,
            prefix_reuse: false,
            long_context_validation: CapabilityLevel::Unsupported,
            benchmark_metrics: CapabilityLevel::Unsupported,
        }
    }

    pub const fn llama_cpp_cli_baseline() -> Self {
        Self {
            text_generation: true,
            token_streaming: false,
            deterministic_mode: false,
            prefix_reuse: false,
            long_context_validation: CapabilityLevel::Unsupported,
            benchmark_metrics: CapabilityLevel::Unsupported,
        }
    }

    pub const fn unsupported() -> Self {
        Self {
            text_generation: false,
            token_streaming: false,
            deterministic_mode: false,
            prefix_reuse: false,
            long_context_validation: CapabilityLevel::Unsupported,
            benchmark_metrics: CapabilityLevel::Unsupported,
        }
    }

    pub const fn for_resolution(
        selected_backend: SelectedBackend,
        support_tier: SupportTier,
    ) -> Self {
        match (selected_backend, support_tier) {
            (SelectedBackend::Mlx, SupportTier::MlxPreview) => Self::mlx_preview(),
            (SelectedBackend::Mlx, SupportTier::MlxCertified) => Self::mlx_certified(),
            (SelectedBackend::MlxLmDelegated, SupportTier::MlxLmDelegated) => {
                Self::mlx_lm_delegated_text()
            }
            (_, SupportTier::LlamaCpp) => Self::llama_cpp_baseline(),
            (_, SupportTier::Unsupported) => Self::unsupported(),
            _ => Self::unsupported(),
        }
    }

    pub fn for_llama_cpp_backend(config: &LlamaCppConfig) -> Self {
        match config {
            LlamaCppConfig::Cli(_) => Self::llama_cpp_cli_baseline(),
            LlamaCppConfig::ServerCompletion(_) => Self::llama_cpp_baseline(),
        }
    }

    pub fn for_mlx_lm_backend(_config: &MlxLmConfig) -> Self {
        Self::mlx_lm_delegated_text()
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct HostReport {
    pub os: String,
    pub arch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detected_soc: Option<String>,
    #[serde(default)]
    pub supported_mlx_runtime: bool,
    #[serde(default)]
    pub unsupported_host_override_active: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolStatusReport {
    #[serde(default)]
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalToolchainReport {
    #[serde(default)]
    pub fully_available: bool,
    #[serde(default)]
    pub metal: ToolStatusReport,
    #[serde(default)]
    pub metallib: ToolStatusReport,
    #[serde(default)]
    pub metal_ar: ToolStatusReport,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRunnerKind {
    MetalBringup,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRuntimeArtifactsSource {
    ExplicitConfig,
    ExplicitEnv,
    RepoAutoDetect,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeModelArtifactsSource {
    ExplicitConfig,
    ExplicitEnv,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeRuntimeReport {
    pub runner: NativeRunnerKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts_source: Option<NativeRuntimeArtifactsSource>,
}

impl NativeRuntimeReport {
    pub const fn metal_bringup(source: NativeRuntimeArtifactsSource) -> Self {
        Self {
            runner: NativeRunnerKind::MetalBringup,
            artifacts_source: Some(source),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeModelReport {
    pub artifacts_source: NativeModelArtifactsSource,
    pub model_family: String,
    pub tensor_format: NativeTensorFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_quantization: Option<NativeSourceQuantization>,
    #[serde(
        default,
        skip_serializing_if = "NativeRuntimeStatus::ready_without_details"
    )]
    pub runtime_status: NativeRuntimeStatus,
    pub layer_count: u32,
    pub tensor_count: u32,
    pub tie_word_embeddings: bool,
    /// True when the model uses a mixture-of-experts FFN (e.g. Gemma 4, Qwen3-MoE).
    #[serde(default)]
    pub is_moe: bool,
    /// True when the model interleaves linear-attention and standard-attention layers
    /// (e.g. Qwen3.5, Qwen3-Next).
    #[serde(default)]
    pub is_hybrid_attention: bool,
    /// For hybrid-attention models: stride between full-attention layers.
    /// None for pure-attention or pure-linear-attention models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hybrid_full_attention_interval: Option<u32>,
    /// For MLA models: latent KV dimension (`kv_lora_rank` in the manifest).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mla_kv_latent_dim: Option<u32>,
    /// For MoE models: active experts selected per token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_active_experts: Option<u32>,
    #[serde(default)]
    pub bindings_prepared: bool,
    #[serde(default)]
    pub buffers_bound: bool,
    #[serde(default)]
    pub buffer_count: u32,
    #[serde(default)]
    pub buffer_bytes: u64,
    #[serde(default)]
    pub source_quantized_binding_count: u32,
    #[serde(default)]
    pub source_q4_k_binding_count: u32,
    #[serde(default)]
    pub source_q5_k_binding_count: u32,
    #[serde(default)]
    pub source_q6_k_binding_count: u32,
    #[serde(default)]
    pub source_q8_0_binding_count: u32,
}

impl NativeModelReport {
    pub fn from_summary(
        artifacts_source: NativeModelArtifactsSource,
        summary: NativeModelArtifactsSummary,
        binding: Option<NativeModelBindingSummary>,
    ) -> Self {
        let binding = binding.unwrap_or_default();
        Self {
            artifacts_source,
            model_family: summary.model_family,
            tensor_format: summary.tensor_format,
            source_quantization: summary.source_quantization,
            runtime_status: summary.runtime_status,
            layer_count: summary.layer_count,
            tensor_count: summary.tensor_count,
            tie_word_embeddings: summary.tie_word_embeddings,
            is_moe: summary.is_moe,
            is_hybrid_attention: summary.is_hybrid_attention,
            hybrid_full_attention_interval: summary.hybrid_full_attention_interval,
            mla_kv_latent_dim: summary.mla_kv_latent_dim,
            moe_active_experts: summary.moe_active_experts,
            bindings_prepared: binding.bindings_prepared,
            buffers_bound: binding.buffers_bound,
            buffer_count: binding.buffer_count,
            buffer_bytes: binding.buffer_bytes,
            source_quantized_binding_count: binding.source_quantized_binding_count,
            source_q4_k_binding_count: binding.source_q4_k_binding_count,
            source_q5_k_binding_count: binding.source_q5_k_binding_count,
            source_q6_k_binding_count: binding.source_q6_k_binding_count,
            source_q8_0_binding_count: binding.source_q8_0_binding_count,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResolvedBackend {
    pub selected_backend: SelectedBackend,
    pub support_tier: SupportTier,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    #[serde(default)]
    pub capabilities: CapabilityReport,
}

impl ResolvedBackend {
    pub fn new(
        selected_backend: SelectedBackend,
        support_tier: SupportTier,
        fallback_reason: Option<String>,
    ) -> Self {
        Self {
            selected_backend,
            support_tier,
            fallback_reason,
            capabilities: CapabilityReport::for_resolution(selected_backend, support_tier),
        }
    }

    pub fn mlx_preview() -> Self {
        Self::new(SelectedBackend::Mlx, SupportTier::MlxPreview, None)
    }

    pub fn mlx_certified() -> Self {
        Self::new(SelectedBackend::Mlx, SupportTier::MlxCertified, None)
    }

    pub fn llama_cpp(
        selected_backend: SelectedBackend,
        fallback_reason: impl Into<String>,
    ) -> Self {
        Self::new(
            selected_backend,
            SupportTier::LlamaCpp,
            Some(fallback_reason.into()),
        )
    }

    pub fn mlx_lm_delegated(reason: impl Into<String>) -> Self {
        Self::new(
            SelectedBackend::MlxLmDelegated,
            SupportTier::MlxLmDelegated,
            Some(reason.into()),
        )
    }

    pub fn validate_against(
        &self,
        backend_policy: &BackendPolicy,
    ) -> Result<(), BackendContractError> {
        if self.support_tier == SupportTier::Unsupported {
            return Err(BackendContractError::UnsupportedCannotResolve);
        }

        if self.selected_backend.is_mlx() {
            if !self.support_tier.is_mlx() {
                return Err(BackendContractError::MlxBackendRequiresMlxTier {
                    support_tier: self.support_tier,
                });
            }
            if self.fallback_reason.is_some() {
                return Err(BackendContractError::MlxBackendCannotHaveFallbackReason);
            }
            return Ok(());
        }

        if self.selected_backend == SelectedBackend::MlxLmDelegated {
            if self.support_tier != SupportTier::MlxLmDelegated {
                return Err(BackendContractError::MlxLmBackendRequiresMlxLmTier {
                    support_tier: self.support_tier,
                });
            }
            if matches!(backend_policy.resolution_policy, ResolutionPolicy::MlxOnly) {
                return Err(BackendContractError::MlxOnlyPolicyCannotResolveMlxLmDelegated);
            }
            let has_reason = self
                .fallback_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty());
            if !has_reason {
                return Err(BackendContractError::MlxLmBackendRequiresDelegationReason);
            }
            return Ok(());
        }

        if self.support_tier != SupportTier::LlamaCpp {
            return Err(BackendContractError::LlamaCppBackendRequiresLlamaCppTier {
                selected_backend: self.selected_backend,
                support_tier: self.support_tier,
            });
        }

        if matches!(backend_policy.resolution_policy, ResolutionPolicy::MlxOnly) {
            return Err(BackendContractError::MlxOnlyPolicyCannotResolveLlamaCpp {
                selected_backend: self.selected_backend,
            });
        }

        let has_reason = self
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| !reason.trim().is_empty());
        if !has_reason {
            return Err(
                BackendContractError::LlamaCppBackendRequiresFallbackReason {
                    selected_backend: self.selected_backend,
                },
            );
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreviewBackendRequest {
    pub mode: PreviewBackendMode,
    pub support_tier: SupportTier,
    pub llama_cli_path: PathBuf,
    pub llama_model_path: Option<PathBuf>,
    pub llama_server_url: Option<String>,
    pub mlx_lm_server_url: Option<String>,
    pub delegated_http_timeouts: DelegatedHttpTimeouts,
}

impl Default for PreviewBackendRequest {
    fn default() -> Self {
        Self {
            mode: PreviewBackendMode::Explicit,
            support_tier: SupportTier::MlxPreview,
            llama_cli_path: PathBuf::from("llama-cli"),
            llama_model_path: None,
            llama_server_url: None,
            mlx_lm_server_url: None,
            delegated_http_timeouts: DelegatedHttpTimeouts::default(),
        }
    }
}

impl PreviewBackendRequest {
    pub fn new(support_tier: SupportTier) -> Self {
        Self {
            support_tier,
            ..Self::default()
        }
    }

    pub fn shipping_default_llama_cpp(
        llama_cli_path: impl Into<PathBuf>,
        llama_model_path: Option<PathBuf>,
        llama_server_url: Option<String>,
    ) -> Self {
        Self {
            mode: PreviewBackendMode::ShippingDefaultLlamaCpp,
            support_tier: SupportTier::LlamaCpp,
            llama_cli_path: llama_cli_path.into(),
            llama_model_path,
            llama_server_url,
            ..Self::default()
        }
    }

    pub fn shipping_mlx() -> Self {
        Self {
            mode: PreviewBackendMode::ShippingMlx,
            support_tier: SupportTier::MlxPreview,
            ..Self::default()
        }
    }

    pub fn with_delegated_http_timeouts(mut self, timeouts: DelegatedHttpTimeouts) -> Self {
        self.delegated_http_timeouts = timeouts;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreviewBackendResolution {
    pub backend_policy: BackendPolicy,
    pub resolved_backend: ResolvedBackend,
    pub llama_backend: Option<LlamaCppConfig>,
    pub mlx_lm_backend: Option<MlxLmConfig>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PreviewBackendResolutionError {
    #[error(
        "unsupported support_tier {value}; expected mlx_preview, mlx_certified, mlx_lm_delegated, or llama_cpp"
    )]
    UnsupportedSupportTierLabel { value: String },
    #[error("support_tier llama_cpp accepts either llama_server_url or llama_model_path, not both")]
    LlamaCppTargetConflict,
    #[error("support_tier llama_cpp requires llama_server_url or llama_model_path")]
    MissingLlamaCppTarget,
    #[error("support_tier mlx_lm_delegated requires mlx_lm_server_url")]
    MissingMlxLmServerUrl,
}

pub fn preview_support_tier_from_label(
    value: &str,
) -> Result<SupportTier, PreviewBackendResolutionError> {
    match value {
        "mlx_preview" => Ok(SupportTier::MlxPreview),
        "mlx_certified" => Ok(SupportTier::MlxCertified),
        "mlx_lm_delegated" => Ok(SupportTier::MlxLmDelegated),
        "llama_cpp" => Ok(SupportTier::LlamaCpp),
        other => Err(PreviewBackendResolutionError::UnsupportedSupportTierLabel {
            value: other.to_string(),
        }),
    }
}

pub fn resolve_preview_backend(
    request: PreviewBackendRequest,
) -> Result<PreviewBackendResolution, PreviewBackendResolutionError> {
    match request.mode {
        PreviewBackendMode::Explicit => resolve_explicit_preview_backend(request),
        PreviewBackendMode::ShippingDefaultLlamaCpp => {
            let llama_backend = resolve_llama_cpp_target(
                request.llama_cli_path,
                request.llama_model_path,
                request.llama_server_url,
                request.delegated_http_timeouts,
            )?;

            Ok(PreviewBackendResolution {
                backend_policy: BackendPolicy::allow_llama_cpp(),
                resolved_backend: ResolvedBackend::llama_cpp(
                    SelectedBackend::LlamaCpp,
                    "shipping default route selected llama.cpp bypass",
                ),
                llama_backend: Some(llama_backend),
                mlx_lm_backend: None,
            })
        }
        PreviewBackendMode::ShippingMlx => Ok(PreviewBackendResolution {
            backend_policy: BackendPolicy::mlx_only(),
            resolved_backend: ResolvedBackend::mlx_preview(),
            llama_backend: None,
            mlx_lm_backend: None,
        }),
    }
}

fn resolve_explicit_preview_backend(
    request: PreviewBackendRequest,
) -> Result<PreviewBackendResolution, PreviewBackendResolutionError> {
    match request.support_tier {
        SupportTier::MlxPreview => Ok(PreviewBackendResolution {
            backend_policy: BackendPolicy::mlx_only(),
            resolved_backend: ResolvedBackend::mlx_preview(),
            llama_backend: None,
            mlx_lm_backend: None,
        }),
        SupportTier::MlxCertified => Ok(PreviewBackendResolution {
            backend_policy: BackendPolicy::mlx_only(),
            resolved_backend: ResolvedBackend::new(
                SelectedBackend::Mlx,
                SupportTier::MlxCertified,
                None,
            ),
            llama_backend: None,
            mlx_lm_backend: None,
        }),
        SupportTier::MlxLmDelegated => Ok(PreviewBackendResolution {
            backend_policy: BackendPolicy::allow_mlx_lm_delegated(),
            resolved_backend: ResolvedBackend::mlx_lm_delegated(
                "mlx-lm delegated backend explicitly requested by preview session config",
            ),
            llama_backend: None,
            mlx_lm_backend: Some(resolve_mlx_lm_target(
                request.mlx_lm_server_url,
                request.delegated_http_timeouts,
            )?),
        }),
        SupportTier::LlamaCpp => {
            let llama_backend = resolve_llama_cpp_target(
                request.llama_cli_path,
                request.llama_model_path,
                request.llama_server_url,
                request.delegated_http_timeouts,
            )?;

            Ok(PreviewBackendResolution {
                backend_policy: BackendPolicy::allow_llama_cpp(),
                resolved_backend: ResolvedBackend::llama_cpp(
                    SelectedBackend::LlamaCpp,
                    "llama.cpp backend explicitly requested by preview session config",
                ),
                llama_backend: Some(llama_backend),
                mlx_lm_backend: None,
            })
        }
        SupportTier::Unsupported => {
            Err(PreviewBackendResolutionError::UnsupportedSupportTierLabel {
                value: support_tier_label(SupportTier::Unsupported).to_string(),
            })
        }
    }
}

fn resolve_mlx_lm_target(
    mlx_lm_server_url: Option<String>,
    timeouts: DelegatedHttpTimeouts,
) -> Result<MlxLmConfig, PreviewBackendResolutionError> {
    let server_url =
        mlx_lm_server_url.ok_or(PreviewBackendResolutionError::MissingMlxLmServerUrl)?;
    Ok(MlxLmConfig::ServerCompletion(
        MlxLmServerCompletionConfig::new(server_url).with_timeouts(timeouts),
    ))
}

fn resolve_llama_cpp_target(
    llama_cli_path: PathBuf,
    llama_model_path: Option<PathBuf>,
    llama_server_url: Option<String>,
    timeouts: DelegatedHttpTimeouts,
) -> Result<LlamaCppConfig, PreviewBackendResolutionError> {
    if llama_server_url.is_some() && llama_model_path.is_some() {
        return Err(PreviewBackendResolutionError::LlamaCppTargetConflict);
    }

    if let Some(server_url) = llama_server_url {
        return Ok(LlamaCppConfig::ServerCompletion(
            LlamaCppServerCompletionConfig::new(server_url).with_timeouts(timeouts),
        ));
    }

    let model_path =
        llama_model_path.ok_or(PreviewBackendResolutionError::MissingLlamaCppTarget)?;
    Ok(LlamaCppConfig::new(llama_cli_path, model_path))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RuntimeReport {
    pub selected_backend: SelectedBackend,
    pub support_tier: SupportTier,
    pub resolution_policy: ResolutionPolicy,
    #[serde(default)]
    pub capabilities: CapabilityReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    #[serde(default)]
    pub host: HostReport,
    #[serde(default)]
    pub metal_toolchain: MetalToolchainReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlx_runtime: Option<NativeRuntimeReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlx_model: Option<NativeModelReport>,
}

impl RuntimeReport {
    pub fn from_resolution(
        backend_policy: &BackendPolicy,
        resolved_backend: &ResolvedBackend,
    ) -> Self {
        Self {
            selected_backend: resolved_backend.selected_backend,
            support_tier: resolved_backend.support_tier,
            resolution_policy: backend_policy.resolution_policy,
            capabilities: resolved_backend.capabilities.clone(),
            fallback_reason: resolved_backend.fallback_reason.clone(),
            host: crate::host::runtime_host_report(),
            metal_toolchain: crate::host::runtime_metal_toolchain_report(),
            mlx_runtime: None,
            mlx_model: None,
        }
    }

    pub fn with_mlx_runtime(mut self, native_runtime: Option<NativeRuntimeReport>) -> Self {
        self.mlx_runtime = native_runtime;
        self
    }

    pub fn with_mlx_model(mut self, native_model: Option<NativeModelReport>) -> Self {
        self.mlx_model = native_model;
        self
    }
}

pub fn current_host_report() -> HostReport {
    crate::host::runtime_host_report()
}

pub fn current_metal_toolchain_report() -> MetalToolchainReport {
    crate::host::runtime_metal_toolchain_report()
}

fn support_tier_label(support_tier: SupportTier) -> &'static str {
    match support_tier {
        SupportTier::MlxCertified => "mlx_certified",
        SupportTier::MlxPreview => "mlx_preview",
        SupportTier::MlxLmDelegated => "mlx_lm_delegated",
        SupportTier::LlamaCpp => "llama_cpp",
        SupportTier::Unsupported => "unsupported",
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum BackendContractError {
    #[error("unsupported support tier cannot resolve to a runnable backend")]
    UnsupportedCannotResolve,
    #[error("mlx backend requires a MLX support tier, got {support_tier:?}")]
    MlxBackendRequiresMlxTier { support_tier: SupportTier },
    #[error("mlx backend cannot carry a fallback reason")]
    MlxBackendCannotHaveFallbackReason,
    #[error(
        "llama.cpp backend {selected_backend:?} requires llama.cpp support tier, got {support_tier:?}"
    )]
    LlamaCppBackendRequiresLlamaCppTier {
        selected_backend: SelectedBackend,
        support_tier: SupportTier,
    },
    #[error("mlx_only policy cannot resolve to llama.cpp backend {selected_backend:?}")]
    MlxOnlyPolicyCannotResolveLlamaCpp { selected_backend: SelectedBackend },
    #[error("mlx_only policy cannot resolve to mlx-lm delegated backend")]
    MlxOnlyPolicyCannotResolveMlxLmDelegated,
    #[error(
        "mlx-lm delegated backend requires mlx_lm_delegated support tier, got {support_tier:?}"
    )]
    MlxLmBackendRequiresMlxLmTier { support_tier: SupportTier },
    #[error("mlx-lm delegated backend requires fallback_reason/delegation reason")]
    MlxLmBackendRequiresDelegationReason,
    #[error("llama.cpp backend {selected_backend:?} requires fallback_reason")]
    LlamaCppBackendRequiresFallbackReason { selected_backend: SelectedBackend },
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn rejects_llama_backend_under_mlx_only_policy() {
        let resolved = ResolvedBackend::llama_cpp(
            SelectedBackend::LlamaCpp,
            "MLX runtime not available for requested model",
        );

        let error = resolved
            .validate_against(&BackendPolicy::mlx_only())
            .expect_err("MLX-only policy should reject llama.cpp resolution");

        assert_eq!(
            error,
            BackendContractError::MlxOnlyPolicyCannotResolveLlamaCpp {
                selected_backend: SelectedBackend::LlamaCpp,
            }
        );
    }

    #[test]
    fn rejects_mlx_backend_with_fallback_reason() {
        let resolved = ResolvedBackend::new(
            SelectedBackend::Mlx,
            SupportTier::MlxPreview,
            Some("should not be present".to_string()),
        );

        let error = resolved
            .validate_against(&BackendPolicy::prefer_mlx())
            .expect_err("mlx backend should not carry fallback_reason");

        assert_eq!(
            error,
            BackendContractError::MlxBackendCannotHaveFallbackReason
        );
    }

    #[test]
    fn preview_support_tier_parser_accepts_preview_labels() {
        assert_eq!(
            preview_support_tier_from_label("mlx_preview"),
            Ok(SupportTier::MlxPreview)
        );
        assert_eq!(
            preview_support_tier_from_label("mlx_certified"),
            Ok(SupportTier::MlxCertified)
        );
        assert_eq!(
            preview_support_tier_from_label("mlx_lm_delegated"),
            Ok(SupportTier::MlxLmDelegated)
        );
        assert_eq!(
            preview_support_tier_from_label("llama_cpp"),
            Ok(SupportTier::LlamaCpp)
        );
    }

    #[test]
    fn preview_resolution_mlx_lm_delegated_server_url_uses_sdk_contract() {
        let resolution = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::MlxLmDelegated,
            mlx_lm_server_url: Some("http://127.0.0.1:8090".to_string()),
            ..PreviewBackendRequest::default()
        })
        .expect("mlx-lm delegated resolution should succeed");

        assert_eq!(
            resolution.backend_policy,
            BackendPolicy::allow_mlx_lm_delegated()
        );
        assert_eq!(
            resolution.resolved_backend,
            ResolvedBackend::mlx_lm_delegated(
                "mlx-lm delegated backend explicitly requested by preview session config",
            )
        );
        assert_eq!(
            resolution.mlx_lm_backend,
            Some(MlxLmConfig::server_completion("http://127.0.0.1:8090"))
        );
        assert!(resolution.llama_backend.is_none());
    }

    #[test]
    fn preview_resolution_mlx_lm_delegated_requires_server_url() {
        let error = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::MlxLmDelegated,
            ..PreviewBackendRequest::default()
        })
        .expect_err("mlx-lm delegated resolution should require a server URL");

        assert_eq!(error, PreviewBackendResolutionError::MissingMlxLmServerUrl);
    }

    #[test]
    fn rejects_mlx_lm_delegated_under_mlx_only_policy() {
        let resolved = ResolvedBackend::mlx_lm_delegated("explicit compatibility route");

        let error = resolved
            .validate_against(&BackendPolicy::mlx_only())
            .expect_err("MLX-only policy should reject delegated mlx-lm");

        assert_eq!(
            error,
            BackendContractError::MlxOnlyPolicyCannotResolveMlxLmDelegated
        );
    }

    #[test]
    fn preview_support_tier_parser_rejects_unsupported_label() {
        assert_eq!(
            preview_support_tier_from_label("unsupported"),
            Err(PreviewBackendResolutionError::UnsupportedSupportTierLabel {
                value: "unsupported".to_string(),
            })
        );
    }

    #[test]
    fn preview_resolution_llama_cpp_server_url_uses_sdk_contract() {
        let resolution = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::LlamaCpp,
            llama_server_url: Some("http://127.0.0.1:8080".to_string()),
            ..PreviewBackendRequest::default()
        })
        .expect("llama.cpp resolution should succeed");

        assert_eq!(resolution.backend_policy, BackendPolicy::allow_llama_cpp());
        assert_eq!(
            resolution.resolved_backend,
            ResolvedBackend::llama_cpp(
                SelectedBackend::LlamaCpp,
                "llama.cpp backend explicitly requested by preview session config",
            )
        );
        assert_eq!(
            resolution.llama_backend,
            Some(LlamaCppConfig::server_completion("http://127.0.0.1:8080"))
        );
    }

    #[test]
    fn preview_resolution_applies_delegated_http_timeouts_to_server_backends() {
        let timeouts = DelegatedHttpTimeouts::from_secs(2, 11, 13);
        let resolution = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::LlamaCpp,
            llama_server_url: Some("http://127.0.0.1:8080".to_string()),
            delegated_http_timeouts: timeouts,
            ..PreviewBackendRequest::default()
        })
        .expect("llama.cpp resolution should succeed");
        assert_eq!(
            resolution.llama_backend,
            Some(LlamaCppConfig::ServerCompletion(
                LlamaCppServerCompletionConfig::new("http://127.0.0.1:8080")
                    .with_timeouts(timeouts)
            ))
        );

        let resolution = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::MlxLmDelegated,
            mlx_lm_server_url: Some("http://127.0.0.1:8090".to_string()),
            delegated_http_timeouts: timeouts,
            ..PreviewBackendRequest::default()
        })
        .expect("mlx-lm resolution should succeed");
        assert_eq!(
            resolution.mlx_lm_backend,
            Some(MlxLmConfig::ServerCompletion(
                MlxLmServerCompletionConfig::new("http://127.0.0.1:8090").with_timeouts(timeouts)
            ))
        );
    }

    #[test]
    fn preview_resolution_rejects_multiple_llama_cpp_targets() {
        let error = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::LlamaCpp,
            llama_model_path: Some(PathBuf::from("/tmp/model.gguf")),
            llama_server_url: Some("http://127.0.0.1:8080".to_string()),
            ..PreviewBackendRequest::default()
        })
        .expect_err("llama.cpp resolution should reject multiple targets");

        assert_eq!(error, PreviewBackendResolutionError::LlamaCppTargetConflict);
    }
}
