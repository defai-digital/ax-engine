use ax_engine_sdk::{
    EngineSessionConfig, PreviewBackendRequest, PreviewSessionConfigRequest, SupportTier,
};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum PreviewSupportTier {
    MlxCertified,
    MlxPreview,
    LlamaCpp,
}

impl PreviewSupportTier {
    pub fn as_sdk_support_tier(self) -> SupportTier {
        match self {
            Self::MlxCertified => SupportTier::MlxCertified,
            Self::MlxPreview => SupportTier::MlxPreview,
            Self::LlamaCpp => SupportTier::LlamaCpp,
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

    #[arg(long = "mlx", default_value_t = false)]
    pub mlx: bool,

    #[arg(long = "support-tier", value_enum, default_value_t = PreviewSupportTier::LlamaCpp)]
    pub support_tier: PreviewSupportTier,

    #[arg(long = "llama-cli-path", default_value = "llama-cli")]
    pub llama_cli_path: String,

    #[arg(long = "llama-model-path")]
    pub llama_model_path: Option<PathBuf>,

    #[arg(long = "llama-server-url")]
    pub llama_server_url: Option<String>,
    #[arg(long = "mlx-model-artifacts-dir")]
    pub mlx_model_artifacts_dir: Option<PathBuf>,

    /// Disable n-gram speculative decode and use single-token greedy decode instead.
    /// Useful for establishing a clean greedy baseline in benchmarks.
    #[arg(long = "no-speculative-decode", default_value_t = false)]
    pub no_speculative_decode: bool,
}

impl ServerArgs {
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn session_config(&self) -> Result<EngineSessionConfig, String> {
        let backend_request = if self.mlx {
            PreviewBackendRequest::shipping_mlx()
        } else if self.support_tier == PreviewSupportTier::LlamaCpp {
            PreviewBackendRequest::shipping_default_llama_cpp(
                PathBuf::from(&self.llama_cli_path),
                self.llama_model_path.clone(),
                self.llama_server_url.clone(),
            )
        } else {
            PreviewBackendRequest {
                support_tier: self.support_tier.as_sdk_support_tier(),
                llama_cli_path: PathBuf::from(&self.llama_cli_path),
                llama_model_path: self.llama_model_path.clone(),
                llama_server_url: self.llama_server_url.clone(),
                ..PreviewBackendRequest::default()
            }
        };

        let mlx_model_artifacts_dir = if self.mlx {
            self.mlx_model_artifacts_dir
                .clone()
                .or_else(|| self.llama_model_path.clone())
        } else {
            self.mlx_model_artifacts_dir.clone()
        };

        EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(self.cache_group_id),
            block_size_tokens: self.block_size_tokens,
            total_blocks: self.total_blocks,
            deterministic: self.deterministic,
            max_batch_tokens: self.max_batch_tokens,
            backend_request,
            mlx_runtime_artifacts_dir: None,
            mlx_model_artifacts_dir,
            mlx_no_speculative_decode: self.no_speculative_decode,
        })
        .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_sdk::{BackendPolicy, LlamaCppConfig, ResolvedBackend, SelectedBackend};

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
            mlx: false,
            support_tier: PreviewSupportTier::LlamaCpp,
            llama_cli_path: "llama-cli".to_string(),
            llama_model_path: None,
            llama_server_url: None,
            mlx_model_artifacts_dir: None,
            no_speculative_decode: false,
        }
    }

    fn assert_configs_match(actual: &EngineSessionConfig, expected: &EngineSessionConfig) {
        assert_eq!(actual.kv_config, expected.kv_config);
        assert_eq!(actual.deterministic, expected.deterministic);
        assert_eq!(actual.max_batch_tokens, expected.max_batch_tokens);
        assert_eq!(actual.backend_policy, expected.backend_policy);
        assert_eq!(actual.resolved_backend, expected.resolved_backend);
        assert_eq!(actual.llama_backend, expected.llama_backend);
        assert_eq!(
            actual.mlx_model_artifacts_dir,
            expected.mlx_model_artifacts_dir
        );
        assert_eq!(
            actual.mlx_model_artifacts_source,
            expected.mlx_model_artifacts_source
        );
    }

    #[test]
    fn preview_support_tier_maps_to_sdk_support_tier() {
        assert_eq!(
            PreviewSupportTier::MlxCertified.as_sdk_support_tier(),
            SupportTier::MlxCertified
        );
        assert_eq!(
            PreviewSupportTier::MlxPreview.as_sdk_support_tier(),
            SupportTier::MlxPreview
        );
        assert_eq!(
            PreviewSupportTier::LlamaCpp.as_sdk_support_tier(),
            SupportTier::LlamaCpp
        );
    }

    #[test]
    fn session_config_matches_sdk_preview_factory_for_mlx_preview() {
        let args = ServerArgs {
            model_id: "qwen3_dense".to_string(),
            cache_group_id: 3,
            total_blocks: 2048,
            support_tier: PreviewSupportTier::MlxPreview,
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");
        let expected = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(3),
            block_size_tokens: 16,
            total_blocks: 2048,
            deterministic: true,
            max_batch_tokens: 2048,
            mlx_runtime_artifacts_dir: None,
            mlx_model_artifacts_dir: None,
            mlx_no_speculative_decode: false,
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::MlxPreview,
                llama_cli_path: PathBuf::from("llama-cli"),
                llama_model_path: None,
                llama_server_url: None,
                ..PreviewBackendRequest::default()
            },
        })
        .expect("sdk preview config should build");

        assert_configs_match(&actual, &expected);
        assert_eq!(actual.backend_policy, BackendPolicy::mlx_only());
        assert_eq!(actual.resolved_backend, ResolvedBackend::mlx_preview());
        assert!(actual.llama_backend.is_none());
    }

    #[test]
    fn session_config_matches_sdk_preview_factory_for_llama_cpp_server() {
        let args = ServerArgs {
            port: 8081,
            deterministic: false,
            max_batch_tokens: 1024,
            cache_group_id: 9,
            total_blocks: 512,
            llama_server_url: Some("http://127.0.0.1:8088".to_string()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");
        let expected = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(9),
            block_size_tokens: 16,
            total_blocks: 512,
            deterministic: false,
            max_batch_tokens: 1024,
            mlx_runtime_artifacts_dir: None,
            mlx_model_artifacts_dir: None,
            mlx_no_speculative_decode: false,
            backend_request: PreviewBackendRequest::shipping_default_llama_cpp(
                PathBuf::from("llama-cli"),
                None,
                Some("http://127.0.0.1:8088".to_string()),
            ),
        })
        .expect("sdk preview config should build");

        assert_configs_match(&actual, &expected);
        assert_eq!(actual.backend_policy, BackendPolicy::allow_llama_cpp());
        assert_eq!(
            actual.resolved_backend.selected_backend,
            SelectedBackend::LlamaCpp
        );
        assert_eq!(
            actual.llama_backend,
            Some(LlamaCppConfig::server_completion("http://127.0.0.1:8088"))
        );
    }

    #[test]
    fn session_config_routes_default_local_model_to_llama_cpp() {
        let model_path = PathBuf::from("/tmp/qwen3.5-mlx");
        let args = ServerArgs {
            llama_model_path: Some(model_path.clone()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(
            actual.resolved_backend.selected_backend,
            SelectedBackend::LlamaCpp
        );
        assert_eq!(
            actual.llama_backend,
            Some(LlamaCppConfig::new("llama-cli", model_path))
        );
    }

    #[test]
    fn session_config_routes_default_gguf_model_to_llama_cpp() {
        let gguf_model_path = PathBuf::from("/tmp/qwen3.5-9b-q4.gguf");
        let args = ServerArgs {
            llama_model_path: Some(gguf_model_path.clone()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(
            actual.resolved_backend.selected_backend,
            SelectedBackend::LlamaCpp
        );
        assert_eq!(
            actual.llama_backend,
            Some(LlamaCppConfig::new("llama-cli", gguf_model_path))
        );
    }

    #[test]
    fn session_config_preserves_explicit_mlx_model_artifacts_dir() {
        let mlx_model_artifacts_dir = PathBuf::from("/tmp/ax-model");
        let args = ServerArgs {
            port: 8081,
            max_batch_tokens: 1024,
            cache_group_id: 9,
            total_blocks: 512,
            support_tier: PreviewSupportTier::MlxPreview,
            mlx_model_artifacts_dir: Some(mlx_model_artifacts_dir.clone()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        let expected_mlx_runtime_artifacts_dir =
            EngineSessionConfig::default_mlx_runtime_artifacts_dir();
        assert_eq!(
            actual.mlx_runtime_artifacts_dir,
            expected_mlx_runtime_artifacts_dir
        );
        assert_eq!(
            actual.mlx_runtime_artifacts_source,
            actual
                .mlx_runtime_artifacts_dir
                .as_ref()
                .map(|_| ax_engine_sdk::NativeRuntimeArtifactsSource::RepoAutoDetect)
        );
        assert_eq!(
            actual.mlx_model_artifacts_dir.as_deref(),
            Some(mlx_model_artifacts_dir.as_path())
        );
        assert_eq!(
            actual.mlx_model_artifacts_source,
            Some(ax_engine_sdk::NativeModelArtifactsSource::ExplicitConfig)
        );
    }

    #[test]
    fn no_speculative_decode_flag_sets_mlx_no_speculative_decode() {
        let args = ServerArgs {
            mlx: true,
            no_speculative_decode: true,
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert!(
            actual.mlx_no_speculative_decode,
            "--no-speculative-decode must propagate to mlx_no_speculative_decode; \
             check args.rs session_config() and EngineSessionConfig::from_preview_request"
        );
    }

    #[test]
    fn default_args_do_not_disable_speculative_decode() {
        let args = ServerArgs {
            mlx: true,
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert!(
            !actual.mlx_no_speculative_decode,
            "speculative decode should be enabled by default"
        );
    }

    #[test]
    fn session_config_preserves_explicit_gguf_model_path_source() {
        let mlx_model_artifacts_dir = PathBuf::from("/tmp/google_gemma-4-26b-it-q4_k_m.gguf");
        let args = ServerArgs {
            port: 8081,
            max_batch_tokens: 1024,
            cache_group_id: 9,
            total_blocks: 512,
            support_tier: PreviewSupportTier::MlxPreview,
            mlx_model_artifacts_dir: Some(mlx_model_artifacts_dir.clone()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(
            actual.mlx_model_artifacts_dir.as_deref(),
            Some(mlx_model_artifacts_dir.as_path())
        );
        assert_eq!(
            actual.mlx_model_artifacts_source,
            Some(ax_engine_sdk::NativeModelArtifactsSource::ExplicitConfig)
        );
    }
}
