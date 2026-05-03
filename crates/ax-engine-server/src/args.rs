use ax_engine_sdk::{
    is_gguf_path, is_initial_native_mode_model_id, native_mode_model_requirement_message,
    CompatibilityBackendKind, EngineSessionConfig, PreviewBackendRequest,
    PreviewSessionConfigRequest, SupportTier, INITIAL_NATIVE_MODE_MODEL_ID,
};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum PreviewSupportTier {
    NativeCertified,
    NativePreview,
    Compatibility,
}

impl PreviewSupportTier {
    pub fn as_sdk_support_tier(self) -> SupportTier {
        match self {
            Self::NativeCertified => SupportTier::NativeCertified,
            Self::NativePreview => SupportTier::NativePreview,
            Self::Compatibility => SupportTier::Compatibility,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum PreviewCompatibilityBackend {
    #[default]
    LlamaCpp,
    Vllm,
    MistralRs,
    Mlx,
}

impl PreviewCompatibilityBackend {
    pub fn as_sdk_backend_kind(self) -> CompatibilityBackendKind {
        match self {
            Self::LlamaCpp => CompatibilityBackendKind::LlamaCpp,
            Self::Vllm => CompatibilityBackendKind::Vllm,
            Self::MistralRs => CompatibilityBackendKind::MistralRs,
            Self::Mlx => CompatibilityBackendKind::Mlx,
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(name = "ax-engine-server", version, about)]
pub struct ServerArgs {
    #[arg(long = "host", default_value = "127.0.0.1")]
    pub host: String,

    #[arg(long = "port", default_value_t = 8080)]
    pub port: u16,

    #[arg(long = "model-id", default_value = "qwen3_dense")]
    pub model_id: String,

    #[arg(long = "deterministic", default_value_t = true)]
    pub deterministic: bool,

    #[arg(long = "max-batch-tokens", default_value_t = 2048)]
    pub max_batch_tokens: u32,

    #[arg(long = "cache-group-id", default_value_t = 0)]
    pub cache_group_id: u16,

    #[arg(long = "block-size-tokens", default_value_t = 16)]
    pub block_size_tokens: u32,

    #[arg(long = "total-blocks", default_value_t = 1024)]
    pub total_blocks: u32,

    #[arg(long = "native-mode", default_value_t = false, conflicts_with_all = ["mlx", "mlx_native"])]
    pub native_mode: bool,

    #[arg(long = "mlx", default_value_t = false, conflicts_with_all = ["native_mode", "mlx_native"])]
    pub mlx: bool,

    #[arg(long = "mlx-native", default_value_t = false, conflicts_with_all = ["native_mode", "mlx"])]
    pub mlx_native: bool,

    #[arg(long = "support-tier", value_enum, default_value_t = PreviewSupportTier::Compatibility)]
    pub support_tier: PreviewSupportTier,

    #[arg(long = "compat-backend", value_enum, default_value_t = PreviewCompatibilityBackend::LlamaCpp)]
    pub compat_backend: PreviewCompatibilityBackend,

    #[arg(long = "compat-cli-path", default_value = "llama-cli")]
    pub compat_cli_path: String,

    #[arg(long = "compat-model-path")]
    pub compat_model_path: Option<PathBuf>,

    #[arg(long = "compat-server-url")]
    pub compat_server_url: Option<String>,

    #[arg(long = "llama-fallback-cli-path", default_value = "llama-cli")]
    pub llama_fallback_cli_path: String,

    #[arg(long = "llama-fallback-model-path")]
    pub llama_fallback_model_path: Option<PathBuf>,

    #[arg(long = "llama-fallback-server-url")]
    pub llama_fallback_server_url: Option<String>,

    #[arg(long = "native-runtime-artifacts-dir")]
    pub native_runtime_artifacts_dir: Option<PathBuf>,

    #[arg(long = "native-model-artifacts-dir")]
    pub native_model_artifacts_dir: Option<PathBuf>,
}

impl ServerArgs {
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn session_config(&self) -> Result<EngineSessionConfig, String> {
        let backend_request = if self.mlx_native {
            PreviewBackendRequest::shipping_mlx_native()
        } else if self.native_mode {
            if !is_initial_native_mode_model_id(&self.model_id) {
                return Err(native_mode_model_requirement_message(&self.model_id));
            }
            let (llama_fallback_cli_path, llama_fallback_model_path, llama_fallback_server_url) =
                self.native_mode_llama_fallback_target();
            PreviewBackendRequest::shipping_native_with_llama_fallback(
                llama_fallback_cli_path,
                llama_fallback_model_path,
                llama_fallback_server_url,
            )
        } else if self.mlx {
            PreviewBackendRequest::shipping_mlx_with_llama_fallback(
                PathBuf::from(&self.compat_cli_path),
                self.compat_model_path.clone(),
                self.compat_server_url.clone(),
                PathBuf::from(&self.llama_fallback_cli_path),
                self.llama_fallback_model_path.clone(),
                self.llama_fallback_server_url.clone(),
            )
        } else if self.support_tier == PreviewSupportTier::Compatibility {
            let effective_backend = self.effective_compat_backend();
            if effective_backend == PreviewCompatibilityBackend::LlamaCpp {
                PreviewBackendRequest::shipping_default_llama_cpp(
                    PathBuf::from(&self.compat_cli_path),
                    self.compat_model_path.clone(),
                    self.compat_server_url.clone(),
                )
            } else {
                PreviewBackendRequest {
                    support_tier: SupportTier::Compatibility,
                    compat_backend: effective_backend.as_sdk_backend_kind(),
                    compat_cli_path: PathBuf::from(&self.compat_cli_path),
                    compat_model_path: self.compat_model_path.clone(),
                    compat_server_url: self.compat_server_url.clone(),
                    ..PreviewBackendRequest::default()
                }
            }
        } else {
            PreviewBackendRequest {
                support_tier: self.support_tier.as_sdk_support_tier(),
                compat_backend: self.compat_backend.as_sdk_backend_kind(),
                compat_cli_path: PathBuf::from(&self.compat_cli_path),
                compat_model_path: self.compat_model_path.clone(),
                compat_server_url: self.compat_server_url.clone(),
                ..PreviewBackendRequest::default()
            }
        };

        EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(self.cache_group_id),
            block_size_tokens: self.block_size_tokens,
            total_blocks: self.total_blocks,
            deterministic: self.deterministic,
            allow_deterministic_native_fallback: false,
            max_batch_tokens: self.max_batch_tokens,
            backend_request,
            native_runtime_artifacts_dir: self.native_runtime_artifacts_dir.clone(),
            native_model_artifacts_dir: self.native_model_artifacts_dir.clone(),
        })
        .map_err(|error| error.to_string())
    }

    fn native_mode_llama_fallback_target(&self) -> (PathBuf, Option<PathBuf>, Option<String>) {
        if self.llama_fallback_model_path.is_some() || self.llama_fallback_server_url.is_some() {
            (
                PathBuf::from(&self.llama_fallback_cli_path),
                self.llama_fallback_model_path.clone(),
                self.llama_fallback_server_url.clone(),
            )
        } else {
            (
                PathBuf::from(&self.compat_cli_path),
                self.compat_model_path.as_deref().filter(|p| is_gguf_path(p)).map(PathBuf::from),
                self.compat_server_url.clone(),
            )
        }
    }

    fn effective_compat_backend(&self) -> PreviewCompatibilityBackend {
        if let Some(path) = &self.compat_model_path {
            if self.compat_server_url.is_none() {
                return if is_gguf_path(path) {
                    PreviewCompatibilityBackend::LlamaCpp
                } else {
                    PreviewCompatibilityBackend::Mlx
                };
            }
        }
        self.compat_backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_sdk::{
        BackendPolicy, CompatibilityBackendConfig, LlamaCppConfig, MlxConfig,
        OpenAiCompatibleServerConfig, ResolvedBackend, SelectedBackend,
    };

    fn base_args() -> ServerArgs {
        ServerArgs {
            host: "127.0.0.1".to_string(),
            port: 8080,
            model_id: "qwen3_dense".to_string(),
            deterministic: true,
            max_batch_tokens: 2048,
            cache_group_id: 0,
            block_size_tokens: 16,
            total_blocks: 1024,
            native_mode: false,
            mlx: false,
            mlx_native: false,
            support_tier: PreviewSupportTier::Compatibility,
            compat_backend: PreviewCompatibilityBackend::LlamaCpp,
            compat_cli_path: "llama-cli".to_string(),
            compat_model_path: None,
            compat_server_url: None,
            llama_fallback_cli_path: "llama-cli".to_string(),
            llama_fallback_model_path: None,
            llama_fallback_server_url: None,
            native_runtime_artifacts_dir: None,
            native_model_artifacts_dir: None,
        }
    }

    fn assert_configs_match(actual: &EngineSessionConfig, expected: &EngineSessionConfig) {
        assert_eq!(actual.kv_config, expected.kv_config);
        assert_eq!(actual.deterministic, expected.deterministic);
        assert_eq!(
            actual.allow_deterministic_native_fallback,
            expected.allow_deterministic_native_fallback
        );
        assert_eq!(actual.max_batch_tokens, expected.max_batch_tokens);
        assert_eq!(actual.backend_policy, expected.backend_policy);
        assert_eq!(actual.resolved_backend, expected.resolved_backend);
        assert_eq!(actual.compatibility_backend, expected.compatibility_backend);
        assert_eq!(
            actual.fallback_compatibility_backend,
            expected.fallback_compatibility_backend
        );
        assert_eq!(
            actual.native_runtime_artifacts_dir,
            expected.native_runtime_artifacts_dir
        );
        assert_eq!(
            actual.native_runtime_artifacts_source,
            expected.native_runtime_artifacts_source
        );
        assert_eq!(
            actual.native_model_artifacts_dir,
            expected.native_model_artifacts_dir
        );
        assert_eq!(
            actual.native_model_artifacts_source,
            expected.native_model_artifacts_source
        );
    }

    #[test]
    fn preview_support_tier_maps_to_sdk_support_tier() {
        assert_eq!(
            PreviewSupportTier::NativeCertified.as_sdk_support_tier(),
            SupportTier::NativeCertified
        );
        assert_eq!(
            PreviewSupportTier::NativePreview.as_sdk_support_tier(),
            SupportTier::NativePreview
        );
        assert_eq!(
            PreviewSupportTier::Compatibility.as_sdk_support_tier(),
            SupportTier::Compatibility
        );
    }

    #[test]
    fn session_config_matches_sdk_preview_factory_for_native_preview() {
        let args = ServerArgs {
            model_id: INITIAL_NATIVE_MODE_MODEL_ID.to_string(),
            cache_group_id: 3,
            total_blocks: 2048,
            support_tier: PreviewSupportTier::NativePreview,
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");
        let expected = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(3),
            block_size_tokens: 16,
            total_blocks: 2048,
            deterministic: true,
            allow_deterministic_native_fallback: false,
            max_batch_tokens: 2048,
            native_runtime_artifacts_dir: None,
            native_model_artifacts_dir: None,
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::NativePreview,
                compat_backend: CompatibilityBackendKind::LlamaCpp,
                compat_cli_path: PathBuf::from("llama-cli"),
                compat_model_path: None,
                compat_server_url: None,
                ..PreviewBackendRequest::default()
            },
        })
        .expect("sdk preview config should build");

        assert_configs_match(&actual, &expected);
        assert_eq!(actual.backend_policy, BackendPolicy::strict_native());
        assert_eq!(actual.resolved_backend, ResolvedBackend::native_preview());
        assert!(actual.compatibility_backend.is_none());
    }

    #[test]
    fn session_config_native_mode_without_fallback_target_stays_native() {
        let args = ServerArgs {
            model_id: INITIAL_NATIVE_MODE_MODEL_ID.to_string(),
            native_mode: true,
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(actual.backend_policy, BackendPolicy::strict_native());
        assert_eq!(actual.resolved_backend, ResolvedBackend::native_preview());
        assert!(actual.compatibility_backend.is_none());
        assert!(actual.fallback_compatibility_backend.is_none());
    }

    #[test]
    fn session_config_native_mode_does_not_reuse_mlx_model_dir_as_llama_fallback() {
        let args = ServerArgs {
            model_id: INITIAL_NATIVE_MODE_MODEL_ID.to_string(),
            native_mode: true,
            compat_model_path: Some(PathBuf::from("/tmp/qwen3.5-mlx")),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(actual.backend_policy, BackendPolicy::strict_native());
        assert!(actual.fallback_compatibility_backend.is_none());
    }

    #[test]
    fn session_config_matches_sdk_preview_factory_for_compatibility_server() {
        let args = ServerArgs {
            port: 8081,
            deterministic: false,
            max_batch_tokens: 1024,
            cache_group_id: 9,
            total_blocks: 512,
            compat_server_url: Some("http://127.0.0.1:8088".to_string()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");
        let expected = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(9),
            block_size_tokens: 16,
            total_blocks: 512,
            deterministic: false,
            allow_deterministic_native_fallback: false,
            max_batch_tokens: 1024,
            native_runtime_artifacts_dir: None,
            native_model_artifacts_dir: None,
            backend_request: PreviewBackendRequest::shipping_default_llama_cpp(
                PathBuf::from("llama-cli"),
                None,
                Some("http://127.0.0.1:8088".to_string()),
            ),
        })
        .expect("sdk preview config should build");

        assert_configs_match(&actual, &expected);
        assert_eq!(actual.backend_policy, BackendPolicy::allow_compat());
        assert_eq!(
            actual.resolved_backend.selected_backend,
            SelectedBackend::LlamaCpp
        );
        assert_eq!(
            actual.compatibility_backend,
            Some(CompatibilityBackendConfig::LlamaCpp(
                LlamaCppConfig::server_completion("http://127.0.0.1:8088")
            ))
        );
    }

    #[test]
    fn session_config_rejects_native_mode_for_non_initial_model() {
        let args = ServerArgs {
            native_mode: true,
            model_id: "qwen3_dense".to_string(),
            ..base_args()
        };

        let error = args
            .session_config()
            .expect_err("native mode should be allowlisted");

        assert!(error.contains(INITIAL_NATIVE_MODE_MODEL_ID));
    }

    #[test]
    fn session_config_routes_default_local_mlx_model_to_mlx_cli() {
        let mlx_model_path = PathBuf::from("/tmp/qwen3.5-mlx");
        let args = ServerArgs {
            compat_model_path: Some(mlx_model_path.clone()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(
            actual.resolved_backend.selected_backend,
            SelectedBackend::Mlx
        );
        assert_eq!(
            actual.compatibility_backend,
            Some(CompatibilityBackendConfig::Mlx(MlxConfig::cli(
                "python3",
                mlx_model_path,
            )))
        );
    }

    #[test]
    fn session_config_routes_default_gguf_model_to_llama_cpp() {
        let gguf_model_path = PathBuf::from("/tmp/qwen3.5-9b-q4.gguf");
        let args = ServerArgs {
            compat_model_path: Some(gguf_model_path.clone()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(
            actual.resolved_backend.selected_backend,
            SelectedBackend::LlamaCpp
        );
        assert_eq!(
            actual.compatibility_backend,
            Some(CompatibilityBackendConfig::LlamaCpp(LlamaCppConfig::new(
                "llama-cli",
                gguf_model_path,
            )))
        );
    }

    #[test]
    fn session_config_matches_sdk_preview_factory_for_vllm_compatibility_server() {
        let args = ServerArgs {
            port: 8081,
            deterministic: false,
            max_batch_tokens: 1024,
            cache_group_id: 9,
            total_blocks: 512,
            support_tier: PreviewSupportTier::Compatibility,
            compat_backend: PreviewCompatibilityBackend::Vllm,
            compat_server_url: Some("http://127.0.0.1:8000".to_string()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");
        let expected = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(9),
            block_size_tokens: 16,
            total_blocks: 512,
            deterministic: false,
            allow_deterministic_native_fallback: false,
            max_batch_tokens: 1024,
            native_runtime_artifacts_dir: None,
            native_model_artifacts_dir: None,
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::Compatibility,
                compat_backend: CompatibilityBackendKind::Vllm,
                compat_cli_path: PathBuf::from("llama-cli"),
                compat_model_path: None,
                compat_server_url: Some("http://127.0.0.1:8000".to_string()),
                ..PreviewBackendRequest::default()
            },
        })
        .expect("sdk preview config should build");

        assert_configs_match(&actual, &expected);
        assert_eq!(actual.backend_policy, BackendPolicy::allow_compat());
        assert_eq!(
            actual.resolved_backend.selected_backend,
            SelectedBackend::Vllm
        );
        assert_eq!(
            actual.compatibility_backend,
            Some(CompatibilityBackendConfig::Vllm(
                OpenAiCompatibleServerConfig::new("http://127.0.0.1:8000")
            ))
        );
    }

    #[test]
    fn session_config_matches_sdk_preview_factory_for_mistral_rs_compatibility_server() {
        let args = ServerArgs {
            port: 8081,
            deterministic: false,
            max_batch_tokens: 1024,
            cache_group_id: 9,
            total_blocks: 512,
            support_tier: PreviewSupportTier::Compatibility,
            compat_backend: PreviewCompatibilityBackend::MistralRs,
            compat_server_url: Some("http://127.0.0.1:8001".to_string()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");
        let expected = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(9),
            block_size_tokens: 16,
            total_blocks: 512,
            deterministic: false,
            allow_deterministic_native_fallback: false,
            max_batch_tokens: 1024,
            native_runtime_artifacts_dir: None,
            native_model_artifacts_dir: None,
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::Compatibility,
                compat_backend: CompatibilityBackendKind::MistralRs,
                compat_cli_path: PathBuf::from("llama-cli"),
                compat_model_path: None,
                compat_server_url: Some("http://127.0.0.1:8001".to_string()),
                ..PreviewBackendRequest::default()
            },
        })
        .expect("sdk preview config should build");

        assert_configs_match(&actual, &expected);
        assert_eq!(actual.backend_policy, BackendPolicy::allow_compat());
        assert_eq!(
            actual.resolved_backend.selected_backend,
            SelectedBackend::MistralRs
        );
        assert_eq!(
            actual.compatibility_backend,
            Some(CompatibilityBackendConfig::MistralRs(
                OpenAiCompatibleServerConfig::new("http://127.0.0.1:8001")
            ))
        );
    }

    #[test]
    fn session_config_matches_sdk_preview_factory_for_mlx_compatibility_server() {
        let args = ServerArgs {
            port: 8081,
            deterministic: false,
            max_batch_tokens: 1024,
            cache_group_id: 9,
            total_blocks: 512,
            support_tier: PreviewSupportTier::Compatibility,
            compat_backend: PreviewCompatibilityBackend::Mlx,
            compat_server_url: Some("http://127.0.0.1:8082".to_string()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");
        let expected = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(9),
            block_size_tokens: 16,
            total_blocks: 512,
            deterministic: false,
            allow_deterministic_native_fallback: false,
            max_batch_tokens: 1024,
            native_runtime_artifacts_dir: None,
            native_model_artifacts_dir: None,
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::Compatibility,
                compat_backend: CompatibilityBackendKind::Mlx,
                compat_cli_path: PathBuf::from("llama-cli"),
                compat_model_path: None,
                compat_server_url: Some("http://127.0.0.1:8082".to_string()),
                ..PreviewBackendRequest::default()
            },
        })
        .expect("sdk preview config should build");

        assert_configs_match(&actual, &expected);
        assert_eq!(actual.backend_policy, BackendPolicy::allow_compat());
        assert_eq!(
            actual.resolved_backend.selected_backend,
            SelectedBackend::Mlx
        );
        assert_eq!(
            actual.compatibility_backend,
            Some(CompatibilityBackendConfig::Mlx(
                MlxConfig::server_completions("http://127.0.0.1:8082")
            ))
        );
    }

    #[test]
    fn session_config_preserves_explicit_native_artifact_dirs() {
        let native_runtime_artifacts_dir = PathBuf::from("/tmp/ax-metal");
        let native_model_artifacts_dir = PathBuf::from("/tmp/ax-model");
        let args = ServerArgs {
            port: 8081,
            max_batch_tokens: 1024,
            cache_group_id: 9,
            total_blocks: 512,
            support_tier: PreviewSupportTier::NativePreview,
            native_runtime_artifacts_dir: Some(native_runtime_artifacts_dir.clone()),
            native_model_artifacts_dir: Some(native_model_artifacts_dir.clone()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(
            actual.native_runtime_artifacts_dir.as_deref(),
            Some(native_runtime_artifacts_dir.as_path())
        );
        assert_eq!(
            actual.native_runtime_artifacts_source,
            Some(ax_engine_sdk::NativeRuntimeArtifactsSource::ExplicitConfig)
        );
        assert_eq!(
            actual.native_model_artifacts_dir.as_deref(),
            Some(native_model_artifacts_dir.as_path())
        );
        assert_eq!(
            actual.native_model_artifacts_source,
            Some(ax_engine_sdk::NativeModelArtifactsSource::ExplicitConfig)
        );
    }

    #[test]
    fn session_config_marks_explicit_gguf_model_path_as_generated_bridge() {
        let native_model_artifacts_dir = PathBuf::from("/tmp/google_gemma-4-26b-it-q4_k_m.gguf");
        let args = ServerArgs {
            port: 8081,
            max_batch_tokens: 1024,
            cache_group_id: 9,
            total_blocks: 512,
            support_tier: PreviewSupportTier::NativePreview,
            native_runtime_artifacts_dir: None,
            native_model_artifacts_dir: Some(native_model_artifacts_dir.clone()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(
            actual.native_model_artifacts_dir.as_deref(),
            Some(native_model_artifacts_dir.as_path())
        );
        assert_eq!(
            actual.native_model_artifacts_source,
            Some(ax_engine_sdk::NativeModelArtifactsSource::GeneratedFromGguf)
        );
    }
}
