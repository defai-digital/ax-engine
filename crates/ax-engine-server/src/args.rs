use ax_engine_sdk::{
    EngineSessionConfig, MlxKvCompressionConfig, MlxKvCompressionMode, MlxTurboQuantPreset,
    PreviewBackendRequest, PreviewSessionConfigRequest, SupportTier,
};
use clap::{Parser, ValueEnum};
use serde_json::Value;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const MODEL_ARTIFACTS_ENV: &str = "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR";
const MODEL_MANIFEST_FILE: &str = "model-manifest.json";

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum PreviewSupportTier {
    MlxCertified,
    MlxPreview,
    MlxLmDelegated,
    LlamaCpp,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum PreviewMlxKvCompression {
    #[default]
    Disabled,
    #[value(name = "turboquant-shadow")]
    TurboQuantShadow,
    #[value(name = "turboquant-fused-experimental")]
    TurboQuantFusedExperimental,
}

impl PreviewMlxKvCompression {
    fn as_config(
        self,
        hot_window_tokens: usize,
        min_context_tokens: usize,
    ) -> MlxKvCompressionConfig {
        match self {
            Self::Disabled => MlxKvCompressionConfig {
                hot_window_tokens,
                min_context_tokens,
                ..MlxKvCompressionConfig::disabled()
            },
            Self::TurboQuantShadow => MlxKvCompressionConfig {
                mode: MlxKvCompressionMode::TurboQuantShadow,
                preset: MlxTurboQuantPreset::K8V4,
                hot_window_tokens,
                min_context_tokens,
            },
            Self::TurboQuantFusedExperimental => MlxKvCompressionConfig {
                mode: MlxKvCompressionMode::TurboQuantFusedExperimental,
                preset: MlxTurboQuantPreset::K8V4,
                hot_window_tokens,
                min_context_tokens,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ServerPreset {
    #[value(name = "gemma4-e2b")]
    Gemma4E2b,
    #[value(name = "gemma4-31b")]
    Gemma4_31b,
    #[value(name = "qwen3.6-35b", alias = "qwen36-35b")]
    Qwen36_35b,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ModelArtifactResolution {
    ExplicitOnly,
    HfCache,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PresetDefinition {
    preset: ServerPreset,
    label: &'static str,
    model_id: &'static str,
    aliases: &'static [&'static str],
    model_types: &'static [&'static str],
    support_tier: PreviewSupportTier,
    max_batch_tokens: u32,
}

impl PreviewSupportTier {
    pub fn as_sdk_support_tier(self) -> SupportTier {
        match self {
            Self::MlxCertified => SupportTier::MlxCertified,
            Self::MlxPreview => SupportTier::MlxPreview,
            Self::MlxLmDelegated => SupportTier::MlxLmDelegated,
            Self::LlamaCpp => SupportTier::LlamaCpp,
        }
    }
}

impl ServerPreset {
    fn definition(self) -> PresetDefinition {
        match self {
            Self::Gemma4E2b => PresetDefinition {
                preset: self,
                label: "gemma4-e2b",
                model_id: "gemma4-e2b",
                aliases: &["gemma4-e2b", "gemma-4-e2b", "gemma-4-e2b-it"],
                model_types: &["gemma4"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Gemma4_31b => PresetDefinition {
                preset: self,
                label: "gemma4-31b",
                model_id: "gemma4-31b",
                aliases: &["gemma4-31b", "gemma-4-31b", "gemma-4-31b-it"],
                model_types: &["gemma4"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Qwen36_35b => PresetDefinition {
                preset: self,
                label: "qwen3.6-35b",
                model_id: "qwen3.6-35b",
                aliases: &[
                    "qwen3.6-35b",
                    "qwen36-35b",
                    "qwen3-6-35b",
                    "qwen3.6-35b-a3b",
                    "qwen36-35b-a3b",
                ],
                model_types: &["qwen3_next", "qwen3_6", "qwen3.6"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
        }
    }
}

pub fn render_presets() -> String {
    [
        ServerPreset::Gemma4E2b,
        ServerPreset::Gemma4_31b,
        ServerPreset::Qwen36_35b,
    ]
    .into_iter()
    .map(|preset| {
        let definition = preset.definition();
        format!(
            "{}\tmodel_id={}\tsupport_tier={:?}\trequires --mlx-model-artifacts-dir or explicit resolver",
            definition.label, definition.model_id, definition.support_tier
        )
    })
    .collect::<Vec<_>>()
    .join("\n")
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

    #[arg(long = "preset", value_enum, conflicts_with_all = ["model_id", "support_tier"])]
    pub preset: Option<ServerPreset>,

    #[arg(long = "list-presets", default_value_t = false)]
    pub list_presets: bool,

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
    #[arg(long = "mlx-lm-server-url")]
    pub mlx_lm_server_url: Option<String>,
    #[arg(long = "mlx-model-artifacts-dir")]
    pub mlx_model_artifacts_dir: Option<PathBuf>,

    #[arg(long = "resolve-model-artifacts", value_enum, default_value_t = ModelArtifactResolution::ExplicitOnly)]
    pub resolve_model_artifacts: ModelArtifactResolution,

    #[arg(long = "hf-cache-root")]
    pub hf_cache_root: Option<PathBuf>,

    /// Disable n-gram acceleration and run the direct same-policy decode path.
    /// Useful for establishing clean benchmark comparisons against mlx_lm.
    #[arg(long = "disable-ngram-acceleration", default_value_t = false)]
    pub disable_ngram_acceleration: bool,

    /// Experimental MLX KV compression policy. Disabled keeps the existing KV path unchanged.
    #[arg(long = "experimental-mlx-kv-compression", value_enum, default_value_t = PreviewMlxKvCompression::Disabled)]
    pub experimental_mlx_kv_compression: PreviewMlxKvCompression,

    /// Full-precision tail retained when experimental MLX KV compression is enabled.
    #[arg(long = "experimental-mlx-kv-compression-hot-window-tokens", default_value_t = MlxKvCompressionConfig::DEFAULT_HOT_WINDOW_TOKENS)]
    pub experimental_mlx_kv_compression_hot_window_tokens: usize,

    /// Minimum context before experimental MLX KV compression becomes eligible.
    #[arg(long = "experimental-mlx-kv-compression-min-context-tokens", default_value_t = MlxKvCompressionConfig::DEFAULT_MIN_CONTEXT_TOKENS)]
    pub experimental_mlx_kv_compression_min_context_tokens: usize,
}

impl ServerArgs {
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn effective_model_id(&self) -> &str {
        self.preset
            .map(|preset| preset.definition().model_id)
            .unwrap_or(self.model_id.as_str())
    }

    pub fn effective_support_tier(&self) -> PreviewSupportTier {
        self.preset
            .map(|preset| preset.definition().support_tier)
            .unwrap_or(self.support_tier)
    }

    pub fn session_config(&self) -> Result<EngineSessionConfig, String> {
        let preset = self.preset.map(ServerPreset::definition);
        let effective_mlx = self.mlx || preset.is_some();
        let effective_support_tier = self.effective_support_tier();
        let effective_max_batch_tokens = preset
            .map(|definition| definition.max_batch_tokens)
            .unwrap_or(self.max_batch_tokens);
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
        } else {
            PreviewBackendRequest {
                support_tier: effective_support_tier.as_sdk_support_tier(),
                llama_cli_path: PathBuf::from(&self.llama_cli_path),
                llama_model_path: self.llama_model_path.clone(),
                llama_server_url: self.llama_server_url.clone(),
                mlx_lm_server_url: self.mlx_lm_server_url.clone(),
                ..PreviewBackendRequest::default()
            }
        };

        EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(self.cache_group_id),
            block_size_tokens: self.block_size_tokens,
            total_blocks: self.total_blocks,
            deterministic: self.deterministic,
            max_batch_tokens: effective_max_batch_tokens,
            backend_request,
            mlx_runtime_artifacts_dir: None,
            mlx_model_artifacts_dir,
            mlx_disable_ngram_acceleration: self.disable_ngram_acceleration,
            mlx_kv_compression: self.experimental_mlx_kv_compression.as_config(
                self.experimental_mlx_kv_compression_hot_window_tokens,
                self.experimental_mlx_kv_compression_min_context_tokens,
            ),
        })
        .map_err(|error| error.to_string())
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
            if env::var_os(MODEL_ARTIFACTS_ENV).is_some() {
                return Ok(None);
            }
            if self.resolve_model_artifacts == ModelArtifactResolution::HfCache {
                let Some(preset) = preset else {
                    return Err(
                        "--resolve-model-artifacts hf-cache requires --preset so AX can validate the expected model family"
                            .to_string(),
                    );
                };
                return resolve_hf_cache_model_artifacts(preset, self.hf_cache_roots()).map(Some);
            }
            return Ok(None);
        }

        Ok(self.mlx_model_artifacts_dir.clone())
    }

    fn hf_cache_roots(&self) -> Vec<PathBuf> {
        if let Some(root) = self.hf_cache_root.clone() {
            return vec![root];
        }

        let mut roots = Vec::new();
        if let Some(root) = env::var_os("HF_HUB_CACHE").map(PathBuf::from) {
            roots.push(root);
        }
        if let Some(home) = env::var_os("HF_HOME").map(PathBuf::from) {
            roots.push(home.join("hub"));
        }
        if let Some(home) = env::var_os("HOME").map(PathBuf::from) {
            roots.push(home.join(".cache").join("huggingface").join("hub"));
        }
        dedupe_paths(roots)
    }
}

fn resolve_hf_cache_model_artifacts(
    preset: &PresetDefinition,
    roots: Vec<PathBuf>,
) -> Result<PathBuf, String> {
    if roots.is_empty() {
        return Err(
            "no Hugging Face cache root found; set HF_HUB_CACHE, HF_HOME, HOME, or pass --hf-cache-root"
                .to_string(),
        );
    }

    let mut candidates = Vec::new();
    let mut rejected = Vec::new();
    for root in roots {
        if !root.is_dir() {
            continue;
        }
        for path in hf_cache_candidate_dirs(&root, preset) {
            match validate_preset_model_artifacts(&path, preset) {
                Ok(()) => candidates.push(path),
                Err(message) => rejected.push(format!("{} ({message})", path.display())),
            }
        }
    }

    candidates.sort();
    candidates.dedup();

    match candidates.len() {
        1 => Ok(candidates.remove(0)),
        0 => {
            let mut message = format!(
                "no valid AX model artifacts found in Hugging Face cache for preset {}; expected config.json, {MODEL_MANIFEST_FILE}, safetensors, and model_type in {:?}",
                preset.label, preset.model_types
            );
            if !rejected.is_empty() {
                message.push_str("; rejected candidates: ");
                message.push_str(&rejected.join("; "));
            }
            Err(message)
        }
        _ => Err(format!(
            "multiple Hugging Face cache candidates matched preset {}; pass --mlx-model-artifacts-dir explicitly: {}",
            preset.label,
            candidates
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn hf_cache_candidate_dirs(root: &Path, preset: &PresetDefinition) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if looks_like_artifact_dir(root) && path_matches_preset(root, preset) {
        candidates.push(root.to_path_buf());
    }

    let Ok(entries) = fs::read_dir(root) else {
        return candidates;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() || !path_matches_preset(&path, preset) {
            continue;
        }
        let snapshots = path.join("snapshots");
        if snapshots.is_dir() {
            if let Ok(snapshot_entries) = fs::read_dir(&snapshots) {
                candidates.extend(
                    snapshot_entries
                        .flatten()
                        .map(|entry| entry.path())
                        .filter(|path| path.is_dir()),
                );
            }
        } else {
            candidates.push(path);
        }
    }

    candidates
}

fn validate_preset_model_artifacts(path: &Path, preset: &PresetDefinition) -> Result<(), String> {
    if !path.join("config.json").is_file() {
        return Err("missing config.json".to_string());
    }
    if !path.join(MODEL_MANIFEST_FILE).is_file() {
        return Err(format!(
            "missing {MODEL_MANIFEST_FILE}; run `cargo run -p ax-engine-core --bin generate-manifest -- <model-dir>` before using this snapshot as AX MLX artifacts"
        ));
    }
    if !dir_contains_safetensors(path) {
        return Err("missing safetensors file".to_string());
    }

    let config = load_json(path.join("config.json").as_path())?;
    let model_type = config_model_type(&config).ok_or("missing model_type in config.json")?;
    if !preset.model_types.contains(&model_type) {
        return Err(format!(
            "model_type {model_type} does not match preset {} expected {:?}",
            preset.label, preset.model_types
        ));
    }

    Ok(())
}

fn looks_like_artifact_dir(path: &Path) -> bool {
    path.join("config.json").is_file()
}

fn path_matches_preset(path: &Path, preset: &PresetDefinition) -> bool {
    let normalized = normalize_model_label(&path.display().to_string());
    preset
        .aliases
        .iter()
        .any(|alias| normalized.contains(&normalize_model_label(alias)))
}

fn normalize_model_label(value: &str) -> String {
    value
        .to_ascii_lowercase()
        .replace("--", "-")
        .replace(['_', '/', '.'], "-")
}

fn dir_contains_safetensors(path: &Path) -> bool {
    fs::read_dir(path).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            entry
                .path()
                .extension()
                .and_then(|extension| extension.to_str())
                == Some("safetensors")
        })
    })
}

fn load_json(path: &Path) -> Result<Value, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

fn config_model_type(config: &Value) -> Option<&str> {
    config
        .get("model_type")
        .and_then(Value::as_str)
        .or_else(|| config.get("text_config")?.get("model_type")?.as_str())
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for path in paths {
        if !deduped.contains(&path) {
            deduped.push(path);
        }
    }
    deduped
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_sdk::{BackendPolicy, LlamaCppConfig, ResolvedBackend, SelectedBackend};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_test_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ax-engine-server-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn write_hf_snapshot(
        root: &Path,
        repo_dir: &str,
        snapshot_id: &str,
        model_type: &str,
    ) -> PathBuf {
        let snapshot = root.join(repo_dir).join("snapshots").join(snapshot_id);
        fs::create_dir_all(&snapshot).expect("snapshot dir should create");
        fs::write(
            snapshot.join("config.json"),
            format!(r#"{{"model_type":"{model_type}"}}"#),
        )
        .expect("config should write");
        fs::write(snapshot.join(MODEL_MANIFEST_FILE), "{}").expect("manifest marker should write");
        fs::write(snapshot.join("model.safetensors"), b"placeholder")
            .expect("safetensors marker should write");
        snapshot
    }

    fn base_args() -> ServerArgs {
        ServerArgs {
            host: "127.0.0.1".to_string(),
            port: 8080,
            model_id: "qwen3_dense".to_string(),
            preset: None,
            list_presets: false,
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
            mlx_lm_server_url: None,
            mlx_model_artifacts_dir: None,
            resolve_model_artifacts: ModelArtifactResolution::ExplicitOnly,
            hf_cache_root: None,
            disable_ngram_acceleration: false,
            experimental_mlx_kv_compression: PreviewMlxKvCompression::Disabled,
            experimental_mlx_kv_compression_hot_window_tokens:
                MlxKvCompressionConfig::DEFAULT_HOT_WINDOW_TOKENS,
            experimental_mlx_kv_compression_min_context_tokens:
                MlxKvCompressionConfig::DEFAULT_MIN_CONTEXT_TOKENS,
        }
    }

    fn assert_configs_match(actual: &EngineSessionConfig, expected: &EngineSessionConfig) {
        assert_eq!(actual.kv_config, expected.kv_config);
        assert_eq!(actual.deterministic, expected.deterministic);
        assert_eq!(actual.max_batch_tokens, expected.max_batch_tokens);
        assert_eq!(actual.backend_policy, expected.backend_policy);
        assert_eq!(actual.resolved_backend, expected.resolved_backend);
        assert_eq!(actual.llama_backend, expected.llama_backend);
        assert_eq!(actual.mlx_lm_backend, expected.mlx_lm_backend);
        assert_eq!(
            actual.mlx_model_artifacts_dir,
            expected.mlx_model_artifacts_dir
        );
        assert_eq!(
            actual.mlx_model_artifacts_source,
            expected.mlx_model_artifacts_source
        );
        assert_eq!(actual.mlx_kv_compression, expected.mlx_kv_compression);
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
            PreviewSupportTier::MlxLmDelegated.as_sdk_support_tier(),
            SupportTier::MlxLmDelegated
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
            mlx_disable_ngram_acceleration: false,
            mlx_kv_compression: MlxKvCompressionConfig::disabled(),
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
            mlx_disable_ngram_acceleration: false,
            mlx_kv_compression: MlxKvCompressionConfig::disabled(),
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
    fn session_config_matches_sdk_preview_factory_for_mlx_lm_delegated_server() {
        let args = ServerArgs {
            support_tier: PreviewSupportTier::MlxLmDelegated,
            mlx_lm_server_url: Some("http://127.0.0.1:8090".to_string()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");
        let expected = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(0),
            block_size_tokens: 16,
            total_blocks: 1024,
            deterministic: true,
            max_batch_tokens: 2048,
            mlx_runtime_artifacts_dir: None,
            mlx_model_artifacts_dir: None,
            mlx_disable_ngram_acceleration: false,
            mlx_kv_compression: MlxKvCompressionConfig::disabled(),
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::MlxLmDelegated,
                mlx_lm_server_url: Some("http://127.0.0.1:8090".to_string()),
                ..PreviewBackendRequest::default()
            },
        })
        .expect("sdk preview config should build");

        assert_configs_match(&actual, &expected);
        assert_eq!(
            actual.resolved_backend.selected_backend,
            SelectedBackend::MlxLmDelegated
        );
        assert!(actual.llama_backend.is_none());
        assert!(actual.mlx_lm_backend.is_some());
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
    fn preset_selects_mlx_preview_defaults() {
        let mlx_model_artifacts_dir = PathBuf::from("/tmp/gemma-4-e2b-it-4bit");
        let args = ServerArgs {
            preset: Some(ServerPreset::Gemma4E2b),
            mlx_model_artifacts_dir: Some(mlx_model_artifacts_dir.clone()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(args.effective_model_id(), "gemma4-e2b");
        assert_eq!(
            args.effective_support_tier(),
            PreviewSupportTier::MlxPreview
        );
        assert_eq!(
            actual.resolved_backend.selected_backend,
            SelectedBackend::Mlx
        );
        assert_eq!(
            actual.mlx_model_artifacts_dir.as_deref(),
            Some(mlx_model_artifacts_dir.as_path())
        );
        assert!(!actual.mlx_disable_ngram_acceleration);
    }

    #[test]
    fn preset_hf_cache_resolution_finds_single_valid_snapshot() {
        let root = unique_test_dir("hf-cache-single");
        let expected = write_hf_snapshot(
            &root,
            "models--mlx-community--gemma-4-e2b-it-4bit",
            "abc123",
            "gemma4",
        );
        let args = ServerArgs {
            preset: Some(ServerPreset::Gemma4E2b),
            resolve_model_artifacts: ModelArtifactResolution::HfCache,
            hf_cache_root: Some(root.clone()),
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(
            actual.mlx_model_artifacts_dir.as_deref(),
            Some(expected.as_path())
        );
        fs::remove_dir_all(root).expect("test dir should clean up");
    }

    #[test]
    fn preset_hf_cache_resolution_rejects_ambiguous_snapshots() {
        let root = unique_test_dir("hf-cache-ambiguous");
        write_hf_snapshot(
            &root,
            "models--mlx-community--gemma-4-e2b-it-4bit",
            "abc123",
            "gemma4",
        );
        write_hf_snapshot(
            &root,
            "models--mlx-community--gemma-4-e2b-it-4bit",
            "def456",
            "gemma4",
        );
        let args = ServerArgs {
            preset: Some(ServerPreset::Gemma4E2b),
            resolve_model_artifacts: ModelArtifactResolution::HfCache,
            hf_cache_root: Some(root.clone()),
            ..base_args()
        };

        let error = args
            .session_config()
            .expect_err("ambiguous cache should fail closed");

        assert!(error.contains("multiple Hugging Face cache candidates"));
        fs::remove_dir_all(root).expect("test dir should clean up");
    }

    #[test]
    fn preset_hf_cache_resolution_requires_ax_manifest() {
        let root = unique_test_dir("hf-cache-no-manifest");
        let snapshot = root
            .join("models--mlx-community--gemma-4-e2b-it-4bit")
            .join("snapshots")
            .join("abc123");
        fs::create_dir_all(&snapshot).expect("snapshot dir should create");
        fs::write(snapshot.join("config.json"), r#"{"model_type":"gemma4"}"#)
            .expect("config should write");
        fs::write(snapshot.join("model.safetensors"), b"placeholder")
            .expect("safetensors marker should write");
        let args = ServerArgs {
            preset: Some(ServerPreset::Gemma4E2b),
            resolve_model_artifacts: ModelArtifactResolution::HfCache,
            hf_cache_root: Some(root.clone()),
            ..base_args()
        };

        let error = args
            .session_config()
            .expect_err("plain HF snapshot should fail closed without AX manifest");

        assert!(error.contains("missing model-manifest.json"));
        fs::remove_dir_all(root).expect("test dir should clean up");
    }

    #[test]
    fn disable_ngram_acceleration_flag_sets_mlx_disable_ngram_acceleration() {
        let args = ServerArgs {
            mlx: true,
            disable_ngram_acceleration: true,
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert!(
            actual.mlx_disable_ngram_acceleration,
            "--disable-ngram-acceleration must propagate to mlx_disable_ngram_acceleration; \
             check args.rs session_config() and EngineSessionConfig::from_preview_request"
        );
    }

    #[test]
    fn default_args_do_not_disable_ngram_acceleration() {
        let args = ServerArgs {
            mlx: true,
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert!(
            !actual.mlx_disable_ngram_acceleration,
            "n-gram acceleration should be enabled by default"
        );
    }

    #[test]
    fn default_args_leave_mlx_kv_compression_disabled() {
        let args = ServerArgs {
            mlx: true,
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(
            actual.mlx_kv_compression,
            MlxKvCompressionConfig::disabled()
        );
    }

    #[test]
    fn experimental_mlx_kv_compression_flag_is_opt_in_shadow_config() {
        let args = ServerArgs {
            mlx: true,
            experimental_mlx_kv_compression: PreviewMlxKvCompression::TurboQuantShadow,
            experimental_mlx_kv_compression_hot_window_tokens: 384,
            experimental_mlx_kv_compression_min_context_tokens: 768,
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(
            actual.mlx_kv_compression,
            MlxKvCompressionConfig {
                mode: MlxKvCompressionMode::TurboQuantShadow,
                preset: MlxTurboQuantPreset::K8V4,
                hot_window_tokens: 384,
                min_context_tokens: 768,
            }
        );
    }

    #[test]
    fn experimental_mlx_kv_compression_flag_can_request_fused_route_selection() {
        let args = ServerArgs {
            mlx: true,
            experimental_mlx_kv_compression: PreviewMlxKvCompression::TurboQuantFusedExperimental,
            experimental_mlx_kv_compression_hot_window_tokens: 384,
            experimental_mlx_kv_compression_min_context_tokens: 768,
            ..base_args()
        };

        let actual = args.session_config().expect("session config should build");

        assert_eq!(
            actual.mlx_kv_compression,
            MlxKvCompressionConfig {
                mode: MlxKvCompressionMode::TurboQuantFusedExperimental,
                preset: MlxTurboQuantPreset::K8V4,
                hot_window_tokens: 384,
                min_context_tokens: 768,
            }
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
