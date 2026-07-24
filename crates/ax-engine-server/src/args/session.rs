use ax_engine_sdk::{
    DelegatedHttpTimeouts, EngineSessionConfig, MlxMtpPolicy, PreviewBackendRequest,
    PreviewSessionConfigRequest, SupportTier,
};
use std::env;
use std::path::PathBuf;

use super::artifacts::{
    hf_cache_roots, infer_model_id_from_artifacts, resolve_hf_cache_model_artifacts,
};
use super::presets::PresetDefinition;
use super::{
    DEFAULT_MODEL_ID, MODEL_ARTIFACTS_ENV, ModelArtifactResolution, PreviewSupportTier, ServerArgs,
    ServerPreset,
};

impl PreviewSupportTier {
    pub fn as_sdk_support_tier(self) -> SupportTier {
        match self {
            Self::MlxCertified => SupportTier::MlxCertified,
            Self::MlxPreview => SupportTier::MlxPreview,
            Self::MlxLmDelegated => SupportTier::MlxLmDelegated,
            Self::LlamaCpp => SupportTier::LlamaCpp,
            Self::TensorRtEdgeLlm => SupportTier::TensorRtEdgeLlm,
            Self::TensorRtLlm => SupportTier::TensorRtLlm,
        }
    }
}

impl ServerArgs {
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn effective_model_id(&self) -> Result<String, String> {
        if let Some(preset) = self.preset {
            return Ok(preset.definition().model_id.to_string());
        }
        if !self.model_id.trim().is_empty() {
            return Ok(self.model_id.trim().to_string());
        }
        if let Some(artifacts_dir) = self.model_id_inference_artifacts_dir()? {
            if let Some(model_id) = infer_model_id_from_artifacts(&artifacts_dir)? {
                return Ok(model_id);
            }
            return Err(format!(
                "could not infer --model-id from {}; pass --model-id explicitly",
                artifacts_dir.display()
            ));
        }
        Ok(DEFAULT_MODEL_ID.to_string())
    }

    pub fn effective_support_tier(&self) -> PreviewSupportTier {
        self.preset
            .map(|preset| preset.definition().support_tier)
            .unwrap_or(self.support_tier)
    }

    pub fn session_config(&self) -> Result<EngineSessionConfig, String> {
        let preset = self.preset.map(ServerPreset::definition);
        let effective_support_tier = self.effective_support_tier();
        // A preset fully specifies its backend tier (it conflicts with
        // --model-id and --support-tier). Without a preset, MLX support tiers
        // are direct by default; --mlx remains a compatibility alias for the
        // same native path.
        let effective_mlx = match preset {
            Some(definition) => matches!(
                definition.support_tier,
                PreviewSupportTier::MlxPreview | PreviewSupportTier::MlxCertified
            ),
            None => {
                self.mlx
                    || matches!(
                        effective_support_tier,
                        PreviewSupportTier::MlxPreview | PreviewSupportTier::MlxCertified
                    )
            }
        };
        let effective_max_batch_tokens = preset
            .map(|definition| definition.max_batch_tokens)
            .unwrap_or(self.max_batch_tokens);
        let delegated_http_timeouts = self.delegated_http_timeouts()?;
        let mlx_model_artifacts_dir =
            self.resolve_mlx_model_artifacts_dir(preset.as_ref(), effective_mlx)?;

        let backend_request = if effective_mlx {
            PreviewBackendRequest::shipping_mlx()
        } else if effective_support_tier == PreviewSupportTier::LlamaCpp {
            PreviewBackendRequest::shipping_default_llama_cpp(
                PathBuf::from(&self.llama_cli_path),
                self.llama_model_path.clone(),
                self.llama_server_url.clone(),
            )
            .with_delegated_http_timeouts(delegated_http_timeouts)
        } else {
            PreviewBackendRequest {
                support_tier: effective_support_tier.as_sdk_support_tier(),
                llama_cli_path: PathBuf::from(&self.llama_cli_path),
                llama_model_path: self.llama_model_path.clone(),
                llama_server_url: self.llama_server_url.clone(),
                mlx_lm_server_url: self.mlx_lm_server_url.clone(),
                edge_llm_server_url: self.edge_llm_server_url.clone(),
                tensor_rt_llm_server_url: self.tensor_rt_llm_server_url.clone(),
                delegated_http_timeouts,
                ..PreviewBackendRequest::default()
            }
        };

        let mlx_mtp_disable_ngram_stacking =
            self.mlx_mtp_disable_ngram_stacking || !self.mlx_mtp_enable_ngram_stacking;

        EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(self.cache_group_id),
            block_size_tokens: self.block_size_tokens,
            total_blocks: self.total_blocks,
            deterministic: self.deterministic,
            max_batch_tokens: effective_max_batch_tokens,
            backend_request,
            mlx_runtime_artifacts_dir: None,
            mlx_model_artifacts_dir,
            mlx_mtp_policy: MlxMtpPolicy::Auto,
            mlx_disable_ngram_acceleration: self.disable_ngram_acceleration,
            mlx_mtp_disable_ngram_stacking,
            mlx_speculation_profile: self
                .speculation_profile
                .map(|profile| profile.as_name().to_string()),
            mlx_prefill_chunk: self.prefill_chunk,
            multi_prefill_fair: self.multi_prefill_fair,
            max_prefill_tokens_per_request_per_step: self.max_prefill_tokens_per_request_per_step,
            max_inflight_prefill_requests: self.max_inflight_prefill_requests,
        })
        .map_err(|error| error.to_string())
    }

    fn delegated_http_timeouts(&self) -> Result<DelegatedHttpTimeouts, String> {
        if self.delegated_http_connect_timeout_secs == 0
            || self.delegated_http_read_timeout_secs == 0
            || self.delegated_http_write_timeout_secs == 0
        {
            return Err("delegated HTTP timeout values must be greater than zero".to_string());
        }
        Ok(DelegatedHttpTimeouts::from_secs(
            self.delegated_http_connect_timeout_secs,
            self.delegated_http_read_timeout_secs,
            self.delegated_http_write_timeout_secs,
        ))
    }

    fn resolve_mlx_model_artifacts_dir(
        &self,
        preset: Option<&PresetDefinition>,
        effective_mlx: bool,
    ) -> Result<Option<PathBuf>, String> {
        if effective_mlx {
            if let Some(path) = self
                .mlx_model_artifacts_dir
                .clone()
                .or_else(|| self.llama_model_path.clone())
            {
                return Ok(Some(path));
            }
            if env::var_os(MODEL_ARTIFACTS_ENV).is_some_and(|v| !v.is_empty()) {
                return Ok(None);
            }
            if self.resolve_model_artifacts == ModelArtifactResolution::HfCache {
                let Some(preset) = preset else {
                    return Err(
                        "--resolve-model-artifacts hf-cache requires --preset so AX can validate the expected model family"
                            .to_string(),
                    );
                };
                return resolve_hf_cache_model_artifacts(
                    preset,
                    hf_cache_roots(self.hf_cache_root.clone()),
                )
                .map(Some);
            }
            return Ok(None);
        }

        Ok(self.mlx_model_artifacts_dir.clone())
    }

    fn model_id_inference_artifacts_dir(&self) -> Result<Option<PathBuf>, String> {
        let preset = self.preset.map(ServerPreset::definition);
        let effective_support_tier = self.effective_support_tier();
        let effective_mlx = match preset {
            Some(definition) => matches!(
                definition.support_tier,
                PreviewSupportTier::MlxPreview | PreviewSupportTier::MlxCertified
            ),
            None => {
                self.mlx
                    || matches!(
                        effective_support_tier,
                        PreviewSupportTier::MlxPreview | PreviewSupportTier::MlxCertified
                    )
            }
        };
        if !effective_mlx {
            return Ok(None);
        }
        if let Some(path) = self
            .mlx_model_artifacts_dir
            .clone()
            .or_else(|| self.llama_model_path.clone())
        {
            return Ok(Some(path));
        }
        if let Some(path) = env::var_os(MODEL_ARTIFACTS_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
        {
            return Ok(Some(path));
        }
        if self.resolve_model_artifacts == ModelArtifactResolution::HfCache {
            let Some(preset) = preset else {
                return Ok(None);
            };
            return resolve_hf_cache_model_artifacts(
                &preset,
                hf_cache_roots(self.hf_cache_root.clone()),
            )
            .map(Some);
        }
        Ok(None)
    }
}
