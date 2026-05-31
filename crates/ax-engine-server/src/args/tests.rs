use super::*;
use ax_engine_sdk::{
    BackendPolicy, EngineSessionConfig, KvCompressionMode, LlamaCppConfig, PreviewBackendRequest,
    PreviewSessionConfigRequest, ResolvedBackend, SelectedBackend, SupportTier, TurboQuantPreset,
};
use std::fs;
use std::path::Path;
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

fn write_hf_snapshot(root: &Path, repo_dir: &str, snapshot_id: &str, model_type: &str) -> PathBuf {
    let snapshot = root.join(repo_dir).join("snapshots").join(snapshot_id);
    fs::create_dir_all(&snapshot).expect("snapshot dir should create");
    fs::write(
        snapshot.join("config.json"),
        format!(r#"{{"model_type":"{model_type}"}}"#),
    )
    .expect("config should write");
    fs::write(snapshot.join(artifacts::MODEL_MANIFEST_FILE), "{}")
        .expect("manifest marker should write");
    fs::write(snapshot.join("model.safetensors"), b"placeholder")
        .expect("safetensors marker should write");
    snapshot
}

fn base_args() -> ServerArgs {
    ServerArgs {
        host: "127.0.0.1".to_string(),
        port: 8080,
        model_id: "qwen3".to_string(),
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
        delegated_http_connect_timeout_secs: DelegatedHttpTimeouts::default_connect_secs(),
        delegated_http_read_timeout_secs: DelegatedHttpTimeouts::default_io_secs(),
        delegated_http_write_timeout_secs: DelegatedHttpTimeouts::default_io_secs(),
        mlx_model_artifacts_dir: None,
        resolve_model_artifacts: ModelArtifactResolution::ExplicitOnly,
        hf_cache_root: None,
        disable_ngram_acceleration: false,
        mlx_mtp_disable_ngram_stacking: false,
        prefill_chunk: None,
        experimental_mlx_kv_compression: PreviewMlxKvCompression::Disabled,
        experimental_mlx_kv_compression_hot_window_tokens:
            KvCompressionConfig::DEFAULT_HOT_WINDOW_TOKENS,
        experimental_mlx_kv_compression_min_context_tokens:
            KvCompressionConfig::DEFAULT_MIN_CONTEXT_TOKENS,
        grpc_bind_address: None,
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
    assert_eq!(
        actual.mlx_mtp_disable_ngram_stacking,
        expected.mlx_mtp_disable_ngram_stacking
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
        model_id: "qwen3".to_string(),
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
        mlx_mtp_disable_ngram_stacking: false,
        mlx_kv_compression: KvCompressionConfig::disabled(),
        mlx_prefill_chunk: None,
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
        mlx_mtp_disable_ngram_stacking: false,
        mlx_kv_compression: KvCompressionConfig::disabled(),
        mlx_prefill_chunk: None,
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
fn session_config_applies_delegated_http_timeouts_to_server_backends() {
    let timeouts = DelegatedHttpTimeouts::from_secs(2, 11, 13);
    let args = ServerArgs {
        llama_server_url: Some("http://127.0.0.1:8088".to_string()),
        delegated_http_connect_timeout_secs: 2,
        delegated_http_read_timeout_secs: 11,
        delegated_http_write_timeout_secs: 13,
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");
    assert_eq!(
        actual.llama_backend,
        Some(LlamaCppConfig::ServerCompletion(
            ax_engine_sdk::LlamaCppServerCompletionConfig::new("http://127.0.0.1:8088")
                .with_timeouts(timeouts)
        ))
    );

    let args = ServerArgs {
        support_tier: PreviewSupportTier::MlxLmDelegated,
        mlx_lm_server_url: Some("http://127.0.0.1:8090".to_string()),
        delegated_http_connect_timeout_secs: 2,
        delegated_http_read_timeout_secs: 11,
        delegated_http_write_timeout_secs: 13,
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");
    assert_eq!(
        actual.mlx_lm_backend,
        Some(ax_engine_sdk::MlxLmConfig::ServerCompletion(
            ax_engine_sdk::MlxLmServerCompletionConfig::new("http://127.0.0.1:8090")
                .with_timeouts(timeouts)
        ))
    );
}

#[test]
fn session_config_rejects_zero_delegated_http_timeout() {
    let args = ServerArgs {
        llama_server_url: Some("http://127.0.0.1:8088".to_string()),
        delegated_http_read_timeout_secs: 0,
        ..base_args()
    };

    let error = args
        .session_config()
        .expect_err("zero delegated timeout should fail closed");
    assert!(error.contains("greater than zero"));
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
        mlx_mtp_disable_ngram_stacking: false,
        mlx_kv_compression: KvCompressionConfig::disabled(),
        mlx_prefill_chunk: None,
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
fn glm_preset_selects_mlx_preview_defaults() {
    let mlx_model_artifacts_dir = PathBuf::from("/tmp/GLM-4.7-Flash-4bit");
    let args = ServerArgs {
        preset: Some(ServerPreset::Glm47Flash4bit),
        mlx_model_artifacts_dir: Some(mlx_model_artifacts_dir.clone()),
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");

    assert_eq!(args.effective_model_id(), "glm4_moe_lite");
    assert_eq!(
        args.effective_support_tier(),
        PreviewSupportTier::MlxPreview
    );
    assert_eq!(
        actual.resolved_backend.selected_backend,
        SelectedBackend::Mlx
    );
    assert_eq!(actual.max_batch_tokens, 2048);
    assert_eq!(
        actual.mlx_model_artifacts_dir.as_deref(),
        Some(mlx_model_artifacts_dir.as_path())
    );
}

#[test]
fn render_presets_lists_glm_preset() {
    let presets = render_presets();

    assert!(presets.contains("glm4.7-flash-4bit\tmodel_id=glm4_moe_lite"));
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
fn glm_preset_hf_cache_resolution_finds_single_valid_snapshot() {
    let root = unique_test_dir("hf-cache-glm");
    let expected = write_hf_snapshot(
        &root,
        "models--mlx-community--GLM-4.7-Flash-4bit",
        "abc123",
        "glm4_moe_lite",
    );
    let args = ServerArgs {
        preset: Some(ServerPreset::Glm47Flash4bit),
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
fn mlx_mtp_disable_ngram_stacking_flag_sets_session_config() {
    let args = ServerArgs {
        mlx: true,
        mlx_mtp_disable_ngram_stacking: true,
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");

    assert!(
        actual.mlx_mtp_disable_ngram_stacking,
        "--mlx-mtp-disable-ngram-stacking must propagate to the MLX session config"
    );
    assert!(
        !actual.mlx_disable_ngram_acceleration,
        "pure-MTP stacking disable must not disable MTP/n-gram acceleration globally"
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
    assert!(
        !actual.mlx_mtp_disable_ngram_stacking,
        "MTP n-gram stacking should be enabled by default"
    );
}

#[test]
fn default_args_leave_mlx_kv_compression_disabled() {
    let args = ServerArgs {
        mlx: true,
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");

    assert_eq!(actual.mlx_kv_compression, KvCompressionConfig::disabled());
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
        KvCompressionConfig {
            mode: KvCompressionMode::TurboQuantShadow,
            preset: TurboQuantPreset::K8V4,
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
        KvCompressionConfig {
            mode: KvCompressionMode::TurboQuantFusedExperimental,
            preset: TurboQuantPreset::K8V4,
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
