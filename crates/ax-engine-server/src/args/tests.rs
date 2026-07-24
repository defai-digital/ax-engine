use super::*;
use ax_engine_sdk::{
    BackendPolicy, EngineSessionConfig, LlamaCppConfig, MlxMtpPolicy, PreviewBackendRequest,
    PreviewSessionConfigRequest, ResolvedBackend, SelectedBackend, SupportTier,
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

fn write_artifact_dir(root: &Path, model_type: &str) -> PathBuf {
    fs::create_dir_all(root).expect("artifact dir should create");
    fs::write(
        root.join("config.json"),
        format!(r#"{{"model_type":"{model_type}"}}"#),
    )
    .expect("config should write");
    root.to_path_buf()
}

fn base_args() -> ServerArgs {
    ServerArgs {
        host: "127.0.0.1".to_string(),
        port: DEFAULT_INFERENCE_PORT,
        model_id: "qwen3".to_string(),
        api_key: None,
        preset: None,
        list_presets: false,
        deterministic: true,
        max_batch_tokens: 2048,
        cache_group_id: 0,
        block_size_tokens: 16,
        total_blocks: 1024,
        mlx: false,
        support_tier: PreviewSupportTier::MlxPreview,
        llama_cli_path: "llama-cli".to_string(),
        llama_model_path: None,
        llama_server_url: None,
        mlx_lm_server_url: None,
        edge_llm_server_url: None,
        tensor_rt_llm_server_url: None,
        vllm_server_url: None,
        vllm_upstream_model_id: None,
        vllm_api_key_env: VLLM_API_KEY_ENV.to_string(),
        vllm_api_key_file: None,
        vllm_allow_remote: false,
        vllm_ca_cert: None,
        vllm_max_in_flight: None,
        vllm_model_profile: VllmModelProfileArg::OpenAiCompatible,
        vllm_runtime_profile: None,
        delegated_http_connect_timeout_secs: DelegatedHttpTimeouts::default_connect_secs(),
        delegated_http_read_timeout_secs: DelegatedHttpTimeouts::default_io_secs(),
        delegated_http_write_timeout_secs: DelegatedHttpTimeouts::default_io_secs(),
        mlx_model_artifacts_dir: None,
        resolve_model_artifacts: ModelArtifactResolution::ExplicitOnly,
        hf_cache_root: None,
        disable_ngram_acceleration: false,
        mlx_mtp_enable_ngram_stacking: false,
        mlx_mtp_disable_ngram_stacking: false,
        speculation_profile: None,
        prefill_chunk: None,
        multi_prefill_fair: false,
        max_prefill_tokens_per_request_per_step: 0,
        max_inflight_prefill_requests: 0,
        grpc_bind_address: None,
        max_concurrent_requests: None,
        max_concurrent_requests_per_model: None,
        max_request_body_bytes: None,
        request_timeout_secs: None,
        grpc_request_timeout_secs: None,
        rate_limit_rps: None,
        rate_limit_burst: None,
        stream_idle_timeout_secs: None,
        model_idle_timeout_secs: None,
        stream_max_duration_secs: None,
        advertise_lan: false,
        lan_cluster: None,
        lan_instance_name: None,
        lan_advertise_host: None,
        allow_open_lan: false,
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
    assert_eq!(actual.multi_prefill_fair, expected.multi_prefill_fair);
    assert_eq!(
        actual.max_prefill_tokens_per_request_per_step,
        expected.max_prefill_tokens_per_request_per_step
    );
    assert_eq!(
        actual.max_inflight_prefill_requests,
        expected.max_inflight_prefill_requests
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
        PreviewSupportTier::MlxLmDelegated.as_sdk_support_tier(),
        SupportTier::MlxLmDelegated
    );
    assert_eq!(
        PreviewSupportTier::LlamaCpp.as_sdk_support_tier(),
        SupportTier::LlamaCpp
    );
    assert_eq!(
        PreviewSupportTier::TensorRtEdgeLlm.as_sdk_support_tier(),
        SupportTier::TensorRtEdgeLlm
    );
    assert_eq!(
        PreviewSupportTier::TensorRtLlm.as_sdk_support_tier(),
        SupportTier::TensorRtLlm
    );
    assert_eq!(
        PreviewSupportTier::Vllm.as_sdk_support_tier(),
        SupportTier::Vllm
    );
}

#[test]
fn vllm_session_config_wires_one_provider_for_both_cuda_architectures() {
    let args = ServerArgs {
        support_tier: PreviewSupportTier::Vllm,
        model_id: "public-ocr".to_string(),
        vllm_server_url: Some("http://127.0.0.1:18000".to_string()),
        vllm_upstream_model_id: Some("baidu/Unlimited-OCR".to_string()),
        vllm_api_key_env: "AX_ENGINE_TEST_VLLM_KEY_NOT_SET".to_string(),
        vllm_max_in_flight: Some(3),
        vllm_model_profile: VllmModelProfileArg::UnlimitedOcr,
        vllm_runtime_profile: Some("cuda-linux-aarch64-thor-sm110".to_string()),
        max_concurrent_requests: Some(5),
        ..base_args()
    };

    let config = args
        .session_config()
        .expect("vLLM session config should resolve");
    assert_eq!(
        config.resolved_backend.selected_backend,
        SelectedBackend::Vllm
    );
    assert_eq!(config.resolved_backend.support_tier, SupportTier::Vllm);
    let backend = config
        .vllm_backend
        .as_ref()
        .expect("vLLM backend should be configured")
        .server();
    assert_eq!(backend.base_url.as_str(), "http://127.0.0.1:18000/v1");
    assert_eq!(backend.upstream_model_id, "baidu/Unlimited-OCR");
    assert_eq!(
        backend.model_profile,
        ax_engine_sdk::VllmModelProfile::UnlimitedOcr
    );
    assert_eq!(
        backend.runtime_profile.as_deref(),
        Some("cuda-linux-aarch64-thor-sm110")
    );
    assert_eq!(backend.max_in_flight, Some(3));
    assert_eq!(args.resolved_max_concurrent_requests(), Some(3));

    let runtime = config.runtime_report();
    let delegated = runtime
        .delegated_runtime
        .as_ref()
        .expect("delegated runtime report should be present");
    assert_eq!(delegated.provider, "vllm");
    assert_eq!(delegated.endpoint.authority, "loopback");
    assert_eq!(
        delegated.readiness,
        ax_engine_sdk::DelegatedReadiness::Configured
    );
}

#[test]
fn vllm_options_fail_closed_outside_vllm_and_reject_zero_admission_limit() {
    let unrelated = ServerArgs {
        vllm_server_url: Some("http://127.0.0.1:18000".to_string()),
        ..base_args()
    };
    let error = unrelated
        .session_config()
        .expect_err("vLLM options must not silently select the provider");
    assert!(error.contains("--support-tier vllm"));

    let zero_limit = ServerArgs {
        support_tier: PreviewSupportTier::Vllm,
        vllm_server_url: Some("http://127.0.0.1:18000".to_string()),
        vllm_api_key_env: "AX_ENGINE_TEST_VLLM_KEY_NOT_SET".to_string(),
        vllm_max_in_flight: Some(0),
        ..base_args()
    };
    let error = zero_limit
        .session_config()
        .expect_err("zero max-in-flight must fail closed");
    assert!(error.contains("greater than zero"));
}

#[cfg(unix)]
#[test]
fn vllm_api_key_file_rejects_symbolic_links() {
    use std::os::unix::fs::symlink;

    let root = unique_test_dir("vllm-api-key-symlink");
    fs::create_dir_all(&root).expect("test dir should create");
    let secret = root.join("api-key");
    let link = root.join("api-key-link");
    fs::write(&secret, "top-secret\n").expect("secret should write");
    symlink(&secret, &link).expect("symlink should create");

    let args = ServerArgs {
        support_tier: PreviewSupportTier::Vllm,
        vllm_server_url: Some("http://127.0.0.1:18000".to_string()),
        vllm_api_key_env: "AX_ENGINE_TEST_VLLM_KEY_NOT_SET".to_string(),
        vllm_api_key_file: Some(link),
        ..base_args()
    };
    let error = args
        .session_config()
        .expect_err("symbolic-link credentials must fail closed");
    assert!(error.contains("must not be a symbolic link"));

    fs::remove_dir_all(root).expect("test dir should remove");
}

#[test]
fn tensor_rt_llm_session_config_wires_server_url() {
    let args = ServerArgs {
        support_tier: PreviewSupportTier::TensorRtLlm,
        tensor_rt_llm_server_url: Some("http://127.0.0.1:8000".to_string()),
        // Sibling Edge URL must not be selected for TensorRT-LLM.
        edge_llm_server_url: Some("http://127.0.0.1:8090".to_string()),
        model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        ..base_args()
    };
    let config = args
        .session_config()
        .expect("tensorrt-llm session config should resolve");
    assert_eq!(
        config.resolved_backend.selected_backend,
        ax_engine_sdk::SelectedBackend::TensorRtLlm
    );
    assert_eq!(
        config.resolved_backend.support_tier,
        SupportTier::TensorRtLlm
    );
    assert!(config.tensor_rt_llm_backend.is_some());
    assert!(config.edge_llm_backend.is_none());
    let runtime = config.runtime_report();
    assert_eq!(
        runtime.selected_backend,
        ax_engine_sdk::SelectedBackend::TensorRtLlm
    );
}

#[test]
fn tensor_rt_edge_llm_session_config_wires_server_url() {
    let args = ServerArgs {
        support_tier: PreviewSupportTier::TensorRtEdgeLlm,
        edge_llm_server_url: Some("http://127.0.0.1:8090".to_string()),
        // Sibling TRT-LLM URL must not be selected for Edge-LLM.
        tensor_rt_llm_server_url: Some("http://127.0.0.1:8000".to_string()),
        model_id: "qwen3".to_string(),
        ..base_args()
    };
    let config = args
        .session_config()
        .expect("tensorrt edge-llm session config should resolve");
    assert_eq!(
        config.resolved_backend.selected_backend,
        ax_engine_sdk::SelectedBackend::TensorRtEdgeLlm
    );
    assert_eq!(
        config.resolved_backend.support_tier,
        SupportTier::TensorRtEdgeLlm
    );
    assert!(config.edge_llm_backend.is_some());
    assert!(config.tensor_rt_llm_backend.is_none());
    let runtime = config.runtime_report();
    assert_eq!(
        runtime.selected_backend,
        ax_engine_sdk::SelectedBackend::TensorRtEdgeLlm
    );
}

#[test]
fn api_key_flag_trims_empty_values() {
    let args = ServerArgs {
        api_key: Some("  secret  ".to_string()),
        ..base_args()
    };
    assert_eq!(args.resolved_api_key(), Some("secret".to_string()));

    let disabled = ServerArgs {
        api_key: Some("   ".to_string()),
        ..base_args()
    };
    assert_eq!(disabled.resolved_api_key(), None);
}

#[test]
fn cli_defaults_to_mlx_preview_support_tier() {
    use clap::Parser;

    let args = ServerArgs::try_parse_from(["ax-engine-server"]).expect("default args should parse");

    assert_eq!(args.port, DEFAULT_INFERENCE_PORT);
    assert_eq!(args.support_tier, PreviewSupportTier::MlxPreview);
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
        mlx_mtp_policy: MlxMtpPolicy::Auto,
        mlx_disable_ngram_acceleration: false,
        mlx_mtp_disable_ngram_stacking: true,
        mlx_speculation_profile: None,
        mlx_prefill_chunk: None,
        multi_prefill_fair: false,
        max_prefill_tokens_per_request_per_step: 0,
        max_inflight_prefill_requests: 0,
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
        support_tier: PreviewSupportTier::LlamaCpp,
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
        mlx_mtp_policy: MlxMtpPolicy::Auto,
        mlx_disable_ngram_acceleration: false,
        mlx_mtp_disable_ngram_stacking: true,
        mlx_speculation_profile: None,
        mlx_prefill_chunk: None,
        multi_prefill_fair: false,
        max_prefill_tokens_per_request_per_step: 0,
        max_inflight_prefill_requests: 0,
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
fn session_config_propagates_non_default_multi_prefill_fair_settings() {
    // Every other multi_prefill_fair fixture in this file uses the
    // false/0/0 defaults, which cannot catch a constructor/mapping
    // function that silently drops one of these fields back to its
    // default. Exercise non-default values end-to-end from ServerArgs
    // through session_config() into EngineSessionConfig.
    let args = ServerArgs {
        multi_prefill_fair: true,
        max_prefill_tokens_per_request_per_step: 64,
        max_inflight_prefill_requests: 3,
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");

    assert!(actual.multi_prefill_fair);
    assert_eq!(actual.max_prefill_tokens_per_request_per_step, 64);
    assert_eq!(actual.max_inflight_prefill_requests, 3);
}

#[test]
fn session_config_applies_delegated_http_timeouts_to_server_backends() {
    let timeouts = DelegatedHttpTimeouts::from_secs(2, 11, 13);
    let args = ServerArgs {
        support_tier: PreviewSupportTier::LlamaCpp,
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
        support_tier: PreviewSupportTier::LlamaCpp,
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
        mlx_mtp_policy: MlxMtpPolicy::Auto,
        mlx_disable_ngram_acceleration: false,
        mlx_mtp_disable_ngram_stacking: true,
        mlx_speculation_profile: None,
        mlx_prefill_chunk: None,
        multi_prefill_fair: false,
        max_prefill_tokens_per_request_per_step: 0,
        max_inflight_prefill_requests: 0,
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
fn session_config_routes_default_local_model_to_mlx_direct() {
    let model_path = PathBuf::from("/tmp/qwen3.5-mlx");
    let args = ServerArgs {
        llama_model_path: Some(model_path.clone()),
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");

    assert_eq!(
        actual.resolved_backend.selected_backend,
        SelectedBackend::Mlx
    );
    assert_eq!(
        actual.mlx_model_artifacts_dir.as_deref(),
        Some(model_path.as_path())
    );
    assert!(actual.llama_backend.is_none());
}

#[test]
fn session_config_routes_explicit_gguf_model_to_llama_cpp() {
    let gguf_model_path = PathBuf::from("/tmp/qwen3.5-9b-q4.gguf");
    let args = ServerArgs {
        support_tier: PreviewSupportTier::LlamaCpp,
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
fn explicit_gemma4_assistant_mtp_artifacts_infer_model_id_when_omitted() {
    let root = unique_test_dir("infer-gemma4-assistant-mtp");
    let artifact_dir = write_artifact_dir(&root, "gemma4_unified");
    fs::write(
        artifact_dir.join("ax_gemma4_assistant_mtp.json"),
        r#"{
            "schema_version": "ax.gemma4_assistant_mtp.v1",
            "target_model_id": "gemma-4-12b-it",
            "assistant_model_id": "gemma-4-12b-it-assistant"
        }"#,
    )
    .expect("assistant MTP contract should write");
    let args = ServerArgs {
        model_id: String::new(),
        mlx: true,
        mlx_model_artifacts_dir: Some(artifact_dir.clone()),
        ..base_args()
    };

    assert_eq!(
        args.effective_model_id()
            .expect("model id should infer from Gemma assistant-MTP contract"),
        "gemma-4-12b-it-assistant-mtp"
    );
    fs::remove_dir_all(root).expect("test dir should clean up");
}

#[test]
fn explicit_glm_mtp_artifacts_infer_model_id_when_omitted() {
    let root = unique_test_dir("infer-glm-mtp");
    let artifact_dir = write_artifact_dir(&root, "glm4_moe_lite");
    fs::write(artifact_dir.join("ax_glm_mtp_manifest.json"), "{}")
        .expect("GLM MTP manifest should write");
    let args = ServerArgs {
        model_id: String::new(),
        mlx: true,
        mlx_model_artifacts_dir: Some(artifact_dir.clone()),
        ..base_args()
    };

    assert_eq!(
        args.effective_model_id()
            .expect("model id should infer from GLM MTP manifest"),
        "glm4_moe_lite"
    );
    fs::remove_dir_all(root).expect("test dir should clean up");
}

#[test]
fn explicit_qwen36_mtp_artifacts_infer_model_id_when_omitted() {
    let root = unique_test_dir("infer-qwen36-mtp");
    let artifact_dir = root
        .join("models--ax-local--Qwen3.6-27B-MTP")
        .join("snapshots")
        .join("v1");
    write_artifact_dir(&artifact_dir, "qwen3_5");
    let args = ServerArgs {
        model_id: String::new(),
        mlx: true,
        mlx_model_artifacts_dir: Some(artifact_dir.clone()),
        ..base_args()
    };

    assert_eq!(
        args.effective_model_id()
            .expect("model id should infer from Qwen MTP path"),
        "qwen3.6-27b-mtp"
    );
    fs::remove_dir_all(root).expect("test dir should clean up");
}

#[test]
fn explicit_automatosx_pack_artifacts_infer_product_model_ids() {
    // AutomatosX org packs (https://huggingface.co/AutomatosX) resolve to
    // the same product model ids as the unbranded artifact layouts.
    let cases = [
        (
            "models--AutomatosX--AX-Qwen3.6-27B-MLX-OptiQ-4bit-MTP",
            "qwen3_5",
            "qwen3.6-27b-mtp",
        ),
        (
            "models--AutomatosX--AX-Qwen3-Coder-Next-MLX-4bit",
            "qwen3_next",
            "qwen3-coder-next",
        ),
        (
            "models--AutomatosX--AX-EmbeddingGemma-300M-MLX-8bit",
            "gemma3_text",
            "embeddinggemma-300m",
        ),
        (
            "models--AutomatosX--AX-Qwen3-Embedding-0.6B-MLX-8bit",
            "qwen3",
            "qwen3-embedding-0.6b",
        ),
        (
            "models--AutomatosX--AX-Qwen3-Embedding-8B-MLX-4bit-DWQ",
            "qwen3",
            "qwen3-embedding-8b",
        ),
    ];
    for (cache_dir, model_type, expected) in cases {
        let root = unique_test_dir("infer-automatosx-pack");
        let artifact_dir = root.join(cache_dir).join("snapshots").join("v1");
        write_artifact_dir(&artifact_dir, model_type);
        let args = ServerArgs {
            model_id: String::new(),
            mlx: true,
            mlx_model_artifacts_dir: Some(artifact_dir.clone()),
            ..base_args()
        };

        assert_eq!(
            args.effective_model_id()
                .expect("model id should infer from AutomatosX pack path"),
            expected,
            "{cache_dir}"
        );
        fs::remove_dir_all(root).expect("test dir should clean up");
    }
}

#[test]
fn omitted_model_id_without_explicit_artifacts_keeps_qwen_default() {
    let args = ServerArgs {
        model_id: String::new(),
        ..base_args()
    };

    assert_eq!(
        args.effective_model_id()
            .expect("default model id should be available without artifacts"),
        "qwen3"
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

    assert_eq!(args.effective_model_id().unwrap(), "gemma4-e2b");
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
fn gemma4_12b_preset_selects_mlx_preview_defaults() {
    let mlx_model_artifacts_dir = PathBuf::from("/tmp/gemma-4-12B-it-4bit");
    let args = ServerArgs {
        preset: Some(ServerPreset::Gemma4_12b),
        mlx_model_artifacts_dir: Some(mlx_model_artifacts_dir.clone()),
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");

    assert_eq!(args.effective_model_id().unwrap(), "gemma4-12b");
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
fn qwen35_9b_preset_selects_mlx_preview_defaults() {
    let mlx_model_artifacts_dir = PathBuf::from("/tmp/Qwen3.5-9B-MLX-4bit");
    let args = ServerArgs {
        preset: Some(ServerPreset::Qwen35_9b),
        mlx_model_artifacts_dir: Some(mlx_model_artifacts_dir.clone()),
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");

    assert_eq!(args.effective_model_id().unwrap(), "qwen3.5-9b");
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
fn qwen36_27b_preset_selects_mlx_preview_defaults() {
    let mlx_model_artifacts_dir = PathBuf::from("/tmp/Qwen3.6-27B-4bit");
    let args = ServerArgs {
        preset: Some(ServerPreset::Qwen36_27b),
        mlx_model_artifacts_dir: Some(mlx_model_artifacts_dir.clone()),
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");

    assert_eq!(args.effective_model_id().unwrap(), "qwen36-27b");
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
fn qwen36_35b_preset_selects_mlx_preview_defaults() {
    let mlx_model_artifacts_dir = PathBuf::from("/tmp/Qwen3.6-35B-A3B-4bit");
    let args = ServerArgs {
        preset: Some(ServerPreset::Qwen36_35b),
        mlx_model_artifacts_dir: Some(mlx_model_artifacts_dir.clone()),
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");

    assert_eq!(args.effective_model_id().unwrap(), "qwen3.6-35b");
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
fn qwen3_coder_next_preset_selects_mlx_preview_defaults() {
    let mlx_model_artifacts_dir = PathBuf::from("/tmp/AX-Qwen3-Coder-Next-MLX-4bit");
    let args = ServerArgs {
        preset: Some(ServerPreset::Qwen3CoderNext),
        mlx_model_artifacts_dir: Some(mlx_model_artifacts_dir.clone()),
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");

    assert_eq!(args.effective_model_id().unwrap(), "qwen3-coder-next");
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
}

#[test]
fn glm_preset_selects_native_mlx_by_default() {
    // GLM 4.7 Flash is a direct-support model: the preset selects the native
    // MLX tier by default. Delegation requires an explicit delegated tier.
    let args = ServerArgs {
        preset: Some(ServerPreset::Glm47Flash4bit),
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");

    assert_eq!(args.effective_model_id().unwrap(), "glm4_moe_lite");
    assert_eq!(
        args.effective_support_tier(),
        PreviewSupportTier::MlxPreview
    );
    assert_eq!(actual.max_batch_tokens, 2048);
}

#[test]
fn render_presets_lists_glm_preset() {
    let presets = render_presets();

    assert!(presets.contains("gemma4-12b\tmodel_id=gemma4-12b"));
    assert!(presets.contains("qwen3.5-9b\tmodel_id=qwen3.5-9b"));
    assert!(presets.contains("qwen3.6-27b\tmodel_id=qwen36-27b"));
    assert!(presets.contains("qwen3-coder-next\tmodel_id=qwen3-coder-next"));
    assert!(presets.contains("glm4.7-flash-4bit\tmodel_id=glm4_moe_lite"));
    assert!(presets.contains("llama3.3-70b\tmodel_id=llama3.3-70b"));
    assert!(presets.contains("mistral-small\tmodel_id=mistral-small"));
    assert!(presets.contains("gpt-oss-20b\tmodel_id=gpt-oss-20b"));
}

#[test]
fn secondary_family_presets_select_mlx_preview() {
    for (preset, model_id) in [
        (ServerPreset::Llama31_8b, "llama3.1-8b"),
        (ServerPreset::Llama33_70b, "llama3.3-70b"),
        (ServerPreset::Llama4Scout, "llama4-scout"),
        (ServerPreset::MistralSmall, "mistral-small"),
        (ServerPreset::Ministral8b, "ministral-8b"),
        (ServerPreset::DevstralSmall, "devstral-small"),
        (ServerPreset::GptOss20b, "gpt-oss-20b"),
        (ServerPreset::GptOss120b, "gpt-oss-120b"),
    ] {
        let args = ServerArgs {
            preset: Some(preset),
            ..base_args()
        };
        assert_eq!(args.effective_model_id().unwrap(), model_id);
        assert_eq!(
            args.effective_support_tier(),
            PreviewSupportTier::MlxPreview
        );
    }
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
fn qwen36_27b_preset_hf_cache_resolution_accepts_cached_snapshot() {
    let root = unique_test_dir("hf-cache-qwen36-27b");
    let expected = write_hf_snapshot(
        &root,
        "models--mlx-community--Qwen3.6-27B-4bit",
        "abc123",
        "qwen3_5",
    );
    let args = ServerArgs {
        preset: Some(ServerPreset::Qwen36_27b),
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
fn qwen36_35b_preset_hf_cache_resolution_accepts_cached_snapshot() {
    let root = unique_test_dir("hf-cache-qwen36-35b");
    let expected = write_hf_snapshot(
        &root,
        "models--mlx-community--Qwen3.6-35B-A3B-4bit",
        "abc123",
        "qwen3_5_moe",
    );
    let args = ServerArgs {
        preset: Some(ServerPreset::Qwen36_35b),
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
fn preset_hf_cache_resolution_accepts_gemma4_12b_unified_snapshot() {
    let root = unique_test_dir("hf-cache-gemma4-12b");
    let expected = write_hf_snapshot(
        &root,
        "models--mlx-community--gemma-4-12B-it-4bit",
        "abc123",
        "gemma4_unified",
    );
    let args = ServerArgs {
        preset: Some(ServerPreset::Gemma4_12b),
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
fn mlx_mtp_enable_ngram_stacking_flag_sets_session_config() {
    let args = ServerArgs {
        mlx: true,
        mlx_mtp_enable_ngram_stacking: true,
        ..base_args()
    };

    let actual = args.session_config().expect("session config should build");

    assert!(
        !actual.mlx_mtp_disable_ngram_stacking,
        "--mlx-mtp-enable-ngram-stacking must opt in to MTP n-gram stacking"
    );
    assert!(
        !actual.mlx_disable_ngram_acceleration,
        "MTP n-gram stacking opt-in must not disable n-gram acceleration globally"
    );
}

#[test]
fn default_args_do_not_disable_global_ngram_acceleration() {
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
        actual.mlx_mtp_disable_ngram_stacking,
        "MTP n-gram stacking should be opt-in by default"
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

#[test]
fn speculation_profile_flag_parses_all_forms() {
    use clap::Parser;

    let coding = ServerArgs::try_parse_from(["ax-engine-server", "-s", "coding"])
        .expect("`-s coding` should parse");
    assert_eq!(
        coding.speculation_profile,
        Some(SpeculationProfileArg::Coding)
    );

    // Hidden `--spec` alias.
    let agentic = ServerArgs::try_parse_from(["ax-engine-server", "--spec", "agentic"])
        .expect("`--spec agentic` should parse");
    assert_eq!(
        agentic.speculation_profile,
        Some(SpeculationProfileArg::Agentic)
    );

    let chatbot =
        ServerArgs::try_parse_from(["ax-engine-server", "--speculation-profile", "chatbot"])
            .expect("`--speculation-profile chatbot` should parse");
    assert_eq!(
        chatbot.speculation_profile,
        Some(SpeculationProfileArg::Chatbot)
    );

    // Unset stays None so the runtime falls back to env / built-in `auto`.
    let unset = ServerArgs::try_parse_from(["ax-engine-server"]).expect("no flag should parse");
    assert_eq!(unset.speculation_profile, None);

    // value_enum rejects unknown postures.
    assert!(ServerArgs::try_parse_from(["ax-engine-server", "-s", "bogus"]).is_err());
}

#[test]
fn speculation_profile_maps_into_session_config() {
    let args = ServerArgs {
        support_tier: PreviewSupportTier::MlxPreview,
        speculation_profile: Some(SpeculationProfileArg::Coding),
        ..base_args()
    };
    let config = args.session_config().expect("session config should build");
    assert_eq!(config.mlx_speculation_profile.as_deref(), Some("coding"));

    let unset = ServerArgs {
        support_tier: PreviewSupportTier::MlxPreview,
        speculation_profile: None,
        ..base_args()
    };
    let config = unset.session_config().expect("session config should build");
    assert_eq!(config.mlx_speculation_profile, None);
}

#[test]
fn resolved_limits_default_to_unset_when_no_flags_given() {
    let args = base_args();
    assert_eq!(args.resolved_max_concurrent_requests(), None);
    assert_eq!(args.resolved_max_concurrent_requests_per_model(), None);
    assert_eq!(
        args.resolved_max_request_body_bytes(),
        crate::DEFAULT_MAX_REQUEST_BODY_BYTES
    );
    assert_eq!(args.resolved_request_timeout(), None);
    assert_eq!(args.resolved_grpc_request_timeout(), None);
    assert_eq!(args.resolved_rate_limit(), None);
    let deadlines = args.resolved_stream_deadlines();
    assert_eq!(deadlines.idle_timeout, None);
    assert_eq!(deadlines.max_duration, None);
}

#[test]
fn resolved_limits_honor_explicit_flags() {
    let args = ServerArgs {
        max_concurrent_requests: Some(4),
        max_concurrent_requests_per_model: Some(2),
        max_request_body_bytes: Some(1024),
        request_timeout_secs: Some(30),
        ..base_args()
    };
    assert_eq!(args.resolved_max_concurrent_requests(), Some(4));
    assert_eq!(args.resolved_max_concurrent_requests_per_model(), Some(2));
    assert_eq!(args.resolved_max_request_body_bytes(), 1024);
    assert_eq!(
        args.resolved_request_timeout(),
        Some(std::time::Duration::from_secs(30))
    );
    // Unset gRPC-specific timeout falls back to the shared HTTP timeout.
    assert_eq!(
        args.resolved_grpc_request_timeout(),
        Some(std::time::Duration::from_secs(30))
    );
}

#[test]
fn grpc_request_timeout_overrides_shared_timeout_when_set() {
    let args = ServerArgs {
        request_timeout_secs: Some(30),
        grpc_request_timeout_secs: Some(120),
        ..base_args()
    };
    assert_eq!(
        args.resolved_grpc_request_timeout(),
        Some(std::time::Duration::from_secs(120))
    );
}

#[test]
fn rate_limit_burst_defaults_to_rps_when_unset() {
    let args = ServerArgs {
        rate_limit_rps: Some(5.0),
        ..base_args()
    };
    let cfg = args
        .resolved_rate_limit()
        .expect("rate limit should be enabled");
    assert_eq!(cfg.rps, 5.0);
    assert_eq!(cfg.burst, 5.0);

    let with_burst = ServerArgs {
        rate_limit_rps: Some(5.0),
        rate_limit_burst: Some(20.0),
        ..base_args()
    };
    let cfg = with_burst
        .resolved_rate_limit()
        .expect("rate limit should be enabled");
    assert_eq!(cfg.burst, 20.0);
}

#[test]
fn stream_deadlines_reflect_explicit_flags() {
    let args = ServerArgs {
        stream_idle_timeout_secs: Some(15),
        stream_max_duration_secs: Some(600),
        ..base_args()
    };
    let deadlines = args.resolved_stream_deadlines();
    assert_eq!(
        deadlines.idle_timeout,
        Some(std::time::Duration::from_secs(15))
    );
    assert_eq!(
        deadlines.max_duration,
        Some(std::time::Duration::from_secs(600))
    );
}
