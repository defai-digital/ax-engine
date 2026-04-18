use std::path::{Path, PathBuf};

use ax_engine_core::{NativeModelArtifactsSummary, NativeModelBindingSummary, NativeTensorFormat};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::compat::{
    CompatibilityBackendConfig, LlamaCppConfig, MlxConfig, OpenAiCompatibleServerConfig,
};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionPolicy {
    #[default]
    StrictNative,
    PreferNative,
    AllowCompat,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectedBackend {
    AxNative,
    LlamaCpp,
    Vllm,
    MistralRs,
    Mlx,
}

impl SelectedBackend {
    pub const fn is_native(self) -> bool {
        matches!(self, Self::AxNative)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportTier {
    NativeCertified,
    NativePreview,
    Compatibility,
    Unsupported,
}

impl SupportTier {
    pub const fn is_native(self) -> bool {
        matches!(self, Self::NativeCertified | Self::NativePreview)
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

    pub const fn strict_native() -> Self {
        Self::new(ResolutionPolicy::StrictNative)
    }

    pub const fn prefer_native() -> Self {
        Self::new(ResolutionPolicy::PreferNative)
    }

    pub const fn allow_compat() -> Self {
        Self::new(ResolutionPolicy::AllowCompat)
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
pub enum CompatibilityBackendKind {
    #[default]
    LlamaCpp,
    Vllm,
    MistralRs,
    Mlx,
}

impl CompatibilityBackendKind {
    pub const fn selected_backend(self) -> SelectedBackend {
        match self {
            Self::LlamaCpp => SelectedBackend::LlamaCpp,
            Self::Vllm => SelectedBackend::Vllm,
            Self::MistralRs => SelectedBackend::MistralRs,
            Self::Mlx => SelectedBackend::Mlx,
        }
    }
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
    pub const fn native_certified() -> Self {
        Self {
            text_generation: true,
            token_streaming: true,
            deterministic_mode: true,
            prefix_reuse: true,
            long_context_validation: CapabilityLevel::Supported,
            benchmark_metrics: CapabilityLevel::Supported,
        }
    }

    pub const fn native_preview() -> Self {
        Self {
            text_generation: true,
            token_streaming: true,
            deterministic_mode: true,
            prefix_reuse: true,
            long_context_validation: CapabilityLevel::Preview,
            benchmark_metrics: CapabilityLevel::Preview,
        }
    }

    pub const fn compatibility_baseline() -> Self {
        Self {
            text_generation: true,
            token_streaming: true,
            deterministic_mode: false,
            prefix_reuse: false,
            long_context_validation: CapabilityLevel::Unsupported,
            benchmark_metrics: CapabilityLevel::Unsupported,
        }
    }

    pub const fn compatibility_cli_baseline() -> Self {
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
            (SelectedBackend::AxNative, SupportTier::NativeCertified) => Self::native_certified(),
            (SelectedBackend::AxNative, SupportTier::NativePreview) => Self::native_preview(),
            (_, SupportTier::Compatibility) => Self::compatibility_baseline(),
            (_, SupportTier::Unsupported) => Self::unsupported(),
            _ => Self::unsupported(),
        }
    }

    pub fn for_compatibility_backend(config: &CompatibilityBackendConfig) -> Self {
        match config {
            CompatibilityBackendConfig::LlamaCpp(LlamaCppConfig::Cli(_))
            | CompatibilityBackendConfig::Mlx(MlxConfig::Cli(_)) => {
                Self::compatibility_cli_baseline()
            }
            CompatibilityBackendConfig::LlamaCpp(LlamaCppConfig::ServerCompletion(_))
            | CompatibilityBackendConfig::Vllm(_)
            | CompatibilityBackendConfig::MistralRs(_)
            | CompatibilityBackendConfig::Mlx(MlxConfig::ServerCompletions(_)) => {
                Self::compatibility_baseline()
            }
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct HostReport {
    pub os: String,
    pub arch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detected_soc: Option<String>,
    #[serde(default)]
    pub supported_native_runtime: bool,
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
    Deterministic,
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
    pub const fn deterministic() -> Self {
        Self {
            runner: NativeRunnerKind::Deterministic,
            artifacts_source: None,
        }
    }

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
    pub layer_count: u32,
    pub tensor_count: u32,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bindings_prepared: bool,
    #[serde(default)]
    pub buffers_bound: bool,
    #[serde(default)]
    pub buffer_count: u32,
    #[serde(default)]
    pub buffer_bytes: u64,
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
            layer_count: summary.layer_count,
            tensor_count: summary.tensor_count,
            tie_word_embeddings: summary.tie_word_embeddings,
            bindings_prepared: binding.bindings_prepared,
            buffers_bound: binding.buffers_bound,
            buffer_count: binding.buffer_count,
            buffer_bytes: binding.buffer_bytes,
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

    pub fn native_preview() -> Self {
        Self::new(SelectedBackend::AxNative, SupportTier::NativePreview, None)
    }

    pub fn native_certified() -> Self {
        Self::new(
            SelectedBackend::AxNative,
            SupportTier::NativeCertified,
            None,
        )
    }

    pub fn compatibility(
        selected_backend: SelectedBackend,
        fallback_reason: impl Into<String>,
    ) -> Self {
        Self::new(
            selected_backend,
            SupportTier::Compatibility,
            Some(fallback_reason.into()),
        )
    }

    pub fn validate_against(
        &self,
        backend_policy: &BackendPolicy,
    ) -> Result<(), BackendContractError> {
        if self.support_tier == SupportTier::Unsupported {
            return Err(BackendContractError::UnsupportedCannotResolve);
        }

        if self.selected_backend.is_native() {
            if !self.support_tier.is_native() {
                return Err(BackendContractError::NativeBackendRequiresNativeTier {
                    support_tier: self.support_tier,
                });
            }
            if self.fallback_reason.is_some() {
                return Err(BackendContractError::NativeBackendCannotHaveFallbackReason);
            }
            return Ok(());
        }

        if self.support_tier != SupportTier::Compatibility {
            return Err(
                BackendContractError::CompatibilityBackendRequiresCompatibilityTier {
                    selected_backend: self.selected_backend,
                    support_tier: self.support_tier,
                },
            );
        }

        if matches!(
            backend_policy.resolution_policy,
            ResolutionPolicy::StrictNative
        ) {
            return Err(
                BackendContractError::StrictNativePolicyCannotResolveCompatibility {
                    selected_backend: self.selected_backend,
                },
            );
        }

        let has_reason = self
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| !reason.trim().is_empty());
        if !has_reason {
            return Err(
                BackendContractError::CompatibilityBackendRequiresFallbackReason {
                    selected_backend: self.selected_backend,
                },
            );
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreviewBackendRequest {
    pub support_tier: SupportTier,
    pub compat_backend: CompatibilityBackendKind,
    pub compat_cli_path: PathBuf,
    pub compat_model_path: Option<PathBuf>,
    pub compat_server_url: Option<String>,
}

impl Default for PreviewBackendRequest {
    fn default() -> Self {
        Self {
            support_tier: SupportTier::NativePreview,
            compat_backend: CompatibilityBackendKind::LlamaCpp,
            compat_cli_path: PathBuf::from("llama-cli"),
            compat_model_path: None,
            compat_server_url: None,
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreviewBackendResolution {
    pub backend_policy: BackendPolicy,
    pub resolved_backend: ResolvedBackend,
    pub compatibility_backend: Option<CompatibilityBackendConfig>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PreviewBackendResolutionError {
    #[error(
        "unsupported support_tier {value}; expected native_preview, native_certified, or compatibility"
    )]
    UnsupportedSupportTierLabel { value: String },
    #[error(
        "support_tier compatibility accepts either compat_server_url or compat_model_path, not both"
    )]
    CompatibilityTargetConflict,
    #[error("support_tier compatibility requires compat_server_url or compat_model_path")]
    MissingCompatibilityTarget,
    #[error(
        "compatibility backend {backend:?} requires compat_server_url; compat_model_path is only supported for llama.cpp or mlx local CLI fallback"
    )]
    CompatibilityCliUnsupported { backend: CompatibilityBackendKind },
}

pub fn preview_support_tier_from_label(
    value: &str,
) -> Result<SupportTier, PreviewBackendResolutionError> {
    match value {
        "native_preview" => Ok(SupportTier::NativePreview),
        "native_certified" => Ok(SupportTier::NativeCertified),
        "compatibility" => Ok(SupportTier::Compatibility),
        other => Err(PreviewBackendResolutionError::UnsupportedSupportTierLabel {
            value: other.to_string(),
        }),
    }
}

pub fn resolve_preview_backend(
    request: PreviewBackendRequest,
) -> Result<PreviewBackendResolution, PreviewBackendResolutionError> {
    match request.support_tier {
        SupportTier::NativePreview => Ok(PreviewBackendResolution {
            backend_policy: BackendPolicy::strict_native(),
            resolved_backend: ResolvedBackend::native_preview(),
            compatibility_backend: None,
        }),
        SupportTier::NativeCertified => Ok(PreviewBackendResolution {
            backend_policy: BackendPolicy::strict_native(),
            resolved_backend: ResolvedBackend::native_certified(),
            compatibility_backend: None,
        }),
        SupportTier::Compatibility => {
            if request.compat_server_url.is_some() && request.compat_model_path.is_some() {
                return Err(PreviewBackendResolutionError::CompatibilityTargetConflict);
            }

            let compatibility_backend = if let Some(server_url) = request.compat_server_url {
                match request.compat_backend {
                    CompatibilityBackendKind::LlamaCpp => CompatibilityBackendConfig::LlamaCpp(
                        LlamaCppConfig::server_completion(server_url),
                    ),
                    CompatibilityBackendKind::Vllm => CompatibilityBackendConfig::Vllm(
                        OpenAiCompatibleServerConfig::new(server_url),
                    ),
                    CompatibilityBackendKind::MistralRs => CompatibilityBackendConfig::MistralRs(
                        OpenAiCompatibleServerConfig::new(server_url),
                    ),
                    CompatibilityBackendKind::Mlx => {
                        CompatibilityBackendConfig::Mlx(MlxConfig::server_completions(server_url))
                    }
                }
            } else {
                if !matches!(
                    request.compat_backend,
                    CompatibilityBackendKind::LlamaCpp | CompatibilityBackendKind::Mlx
                ) {
                    return Err(PreviewBackendResolutionError::CompatibilityCliUnsupported {
                        backend: request.compat_backend,
                    });
                }
                let model_path = request
                    .compat_model_path
                    .ok_or(PreviewBackendResolutionError::MissingCompatibilityTarget)?;
                match request.compat_backend {
                    CompatibilityBackendKind::LlamaCpp => CompatibilityBackendConfig::LlamaCpp(
                        LlamaCppConfig::new(request.compat_cli_path, model_path),
                    ),
                    CompatibilityBackendKind::Mlx => {
                        let cli_path = if request.compat_cli_path == Path::new("llama-cli") {
                            PathBuf::from("python3")
                        } else {
                            request.compat_cli_path
                        };
                        CompatibilityBackendConfig::Mlx(MlxConfig::cli(cli_path, model_path))
                    }
                    CompatibilityBackendKind::Vllm | CompatibilityBackendKind::MistralRs => {
                        unreachable!("validated compatibility cli target backend")
                    }
                }
            };

            Ok(PreviewBackendResolution {
                backend_policy: BackendPolicy::allow_compat(),
                resolved_backend: ResolvedBackend::compatibility(
                    request.compat_backend.selected_backend(),
                    "compatibility backend explicitly requested by preview session config",
                ),
                compatibility_backend: Some(compatibility_backend),
            })
        }
        SupportTier::Unsupported => {
            Err(PreviewBackendResolutionError::UnsupportedSupportTierLabel {
                value: support_tier_label(SupportTier::Unsupported).to_string(),
            })
        }
    }
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
    pub native_runtime: Option<NativeRuntimeReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_model: Option<NativeModelReport>,
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
            native_runtime: None,
            native_model: None,
        }
    }

    pub fn with_native_runtime(mut self, native_runtime: Option<NativeRuntimeReport>) -> Self {
        self.native_runtime = native_runtime;
        self
    }

    pub fn with_native_model(mut self, native_model: Option<NativeModelReport>) -> Self {
        self.native_model = native_model;
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
        SupportTier::NativeCertified => "native_certified",
        SupportTier::NativePreview => "native_preview",
        SupportTier::Compatibility => "compatibility",
        SupportTier::Unsupported => "unsupported",
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum BackendContractError {
    #[error("unsupported support tier cannot resolve to a runnable backend")]
    UnsupportedCannotResolve,
    #[error("native backend requires a native support tier, got {support_tier:?}")]
    NativeBackendRequiresNativeTier { support_tier: SupportTier },
    #[error("native backend cannot carry a fallback reason")]
    NativeBackendCannotHaveFallbackReason,
    #[error(
        "compatibility backend {selected_backend:?} requires compatibility support tier, got {support_tier:?}"
    )]
    CompatibilityBackendRequiresCompatibilityTier {
        selected_backend: SelectedBackend,
        support_tier: SupportTier,
    },
    #[error("strict_native policy cannot resolve to compatibility backend {selected_backend:?}")]
    StrictNativePolicyCannotResolveCompatibility { selected_backend: SelectedBackend },
    #[error("compatibility backend {selected_backend:?} requires fallback_reason")]
    CompatibilityBackendRequiresFallbackReason { selected_backend: SelectedBackend },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_compatibility_backend_under_strict_native_policy() {
        let resolved = ResolvedBackend::compatibility(
            SelectedBackend::LlamaCpp,
            "native runtime not available for requested model",
        );

        let error = resolved
            .validate_against(&BackendPolicy::strict_native())
            .expect_err("strict native policy should reject compatibility resolution");

        assert_eq!(
            error,
            BackendContractError::StrictNativePolicyCannotResolveCompatibility {
                selected_backend: SelectedBackend::LlamaCpp,
            }
        );
    }

    #[test]
    fn rejects_native_backend_with_fallback_reason() {
        let resolved = ResolvedBackend::new(
            SelectedBackend::AxNative,
            SupportTier::NativePreview,
            Some("should not be present".to_string()),
        );

        let error = resolved
            .validate_against(&BackendPolicy::prefer_native())
            .expect_err("native backend should not carry fallback_reason");

        assert_eq!(
            error,
            BackendContractError::NativeBackendCannotHaveFallbackReason
        );
    }

    #[test]
    fn preview_support_tier_parser_accepts_preview_labels() {
        assert_eq!(
            preview_support_tier_from_label("native_preview"),
            Ok(SupportTier::NativePreview)
        );
        assert_eq!(
            preview_support_tier_from_label("native_certified"),
            Ok(SupportTier::NativeCertified)
        );
        assert_eq!(
            preview_support_tier_from_label("compatibility"),
            Ok(SupportTier::Compatibility)
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
    fn preview_resolution_compatibility_server_url_uses_sdk_contract() {
        let resolution = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::Compatibility,
            compat_server_url: Some("http://127.0.0.1:8080".to_string()),
            ..PreviewBackendRequest::default()
        })
        .expect("compatibility resolution should succeed");

        assert_eq!(resolution.backend_policy, BackendPolicy::allow_compat());
        assert_eq!(
            resolution.resolved_backend,
            ResolvedBackend::compatibility(
                SelectedBackend::LlamaCpp,
                "compatibility backend explicitly requested by preview session config",
            )
        );
        assert_eq!(
            resolution.compatibility_backend,
            Some(CompatibilityBackendConfig::LlamaCpp(
                LlamaCppConfig::server_completion("http://127.0.0.1:8080"),
            ))
        );
    }

    #[test]
    fn preview_resolution_vllm_server_url_uses_openai_compatible_contract() {
        let resolution = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::Compatibility,
            compat_backend: CompatibilityBackendKind::Vllm,
            compat_server_url: Some("http://127.0.0.1:8000".to_string()),
            ..PreviewBackendRequest::default()
        })
        .expect("vllm compatibility resolution should succeed");

        assert_eq!(
            resolution.resolved_backend,
            ResolvedBackend::compatibility(
                SelectedBackend::Vllm,
                "compatibility backend explicitly requested by preview session config",
            )
        );
        assert_eq!(
            resolution.compatibility_backend,
            Some(CompatibilityBackendConfig::Vllm(
                OpenAiCompatibleServerConfig::new("http://127.0.0.1:8000"),
            ))
        );
    }

    #[test]
    fn preview_resolution_mistral_rs_server_url_uses_openai_compatible_contract() {
        let resolution = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::Compatibility,
            compat_backend: CompatibilityBackendKind::MistralRs,
            compat_server_url: Some("http://127.0.0.1:1234".to_string()),
            ..PreviewBackendRequest::default()
        })
        .expect("mistral.rs compatibility resolution should succeed");

        assert_eq!(
            resolution.resolved_backend,
            ResolvedBackend::compatibility(
                SelectedBackend::MistralRs,
                "compatibility backend explicitly requested by preview session config",
            )
        );
        assert_eq!(
            resolution.compatibility_backend,
            Some(CompatibilityBackendConfig::MistralRs(
                OpenAiCompatibleServerConfig::new("http://127.0.0.1:1234"),
            ))
        );
    }

    #[test]
    fn preview_resolution_mlx_server_url_uses_openai_compatible_contract() {
        let resolution = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::Compatibility,
            compat_backend: CompatibilityBackendKind::Mlx,
            compat_server_url: Some("http://127.0.0.1:8082".to_string()),
            ..PreviewBackendRequest::default()
        })
        .expect("mlx compatibility resolution should succeed");

        assert_eq!(
            resolution.resolved_backend,
            ResolvedBackend::compatibility(
                SelectedBackend::Mlx,
                "compatibility backend explicitly requested by preview session config",
            )
        );
        assert_eq!(
            resolution.compatibility_backend,
            Some(CompatibilityBackendConfig::Mlx(
                MlxConfig::server_completions("http://127.0.0.1:8082",)
            ))
        );
    }

    #[test]
    fn preview_resolution_rejects_multiple_compatibility_targets() {
        let error = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::Compatibility,
            compat_model_path: Some(PathBuf::from("/tmp/model.gguf")),
            compat_server_url: Some("http://127.0.0.1:8080".to_string()),
            ..PreviewBackendRequest::default()
        })
        .expect_err("compatibility resolution should reject multiple targets");

        assert_eq!(
            error,
            PreviewBackendResolutionError::CompatibilityTargetConflict
        );
    }

    #[test]
    fn preview_resolution_rejects_cli_target_for_non_llama_compat_backend() {
        let error = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::Compatibility,
            compat_backend: CompatibilityBackendKind::Vllm,
            compat_model_path: Some(PathBuf::from("/tmp/model.gguf")),
            ..PreviewBackendRequest::default()
        })
        .expect_err("non-llama compatibility cli target should be rejected");

        assert_eq!(
            error,
            PreviewBackendResolutionError::CompatibilityCliUnsupported {
                backend: CompatibilityBackendKind::Vllm,
            }
        );
    }

    #[test]
    fn preview_resolution_mlx_cli_target_uses_python3_default_fallback() {
        let resolution = resolve_preview_backend(PreviewBackendRequest {
            support_tier: SupportTier::Compatibility,
            compat_backend: CompatibilityBackendKind::Mlx,
            compat_model_path: Some(PathBuf::from("/tmp/mlx-model")),
            ..PreviewBackendRequest::default()
        })
        .expect("mlx cli compatibility resolution should succeed");

        assert_eq!(
            resolution.compatibility_backend,
            Some(CompatibilityBackendConfig::Mlx(MlxConfig::cli(
                "python3",
                "/tmp/mlx-model",
            )))
        );
    }
}
