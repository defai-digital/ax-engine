use std::path::{Path, PathBuf};

use ax_engine_core::{EngineCore, MetalKernelAssets};

use crate::backend::{NativeModelArtifactsSource, NativeModelReport, NativeRuntimeArtifactsSource};

use super::config::EngineSessionConfig;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MlxRuntimeArtifactsSelection {
    pub(super) dir: PathBuf,
    pub(super) source: NativeRuntimeArtifactsSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MlxModelArtifactsSelection {
    pub(super) dir: PathBuf,
    pub(super) source: NativeModelArtifactsSource,
}

pub(super) fn resolve_default_mlx_runtime_artifacts_selection(
    explicit_dir: Option<PathBuf>,
    current_dir: Option<&Path>,
) -> Option<MlxRuntimeArtifactsSelection> {
    explicit_dir
        .map(|dir| MlxRuntimeArtifactsSelection {
            dir,
            source: NativeRuntimeArtifactsSource::ExplicitEnv,
        })
        .or_else(|| current_dir.and_then(detect_repo_owned_mlx_runtime_artifacts_dir_from))
}

fn detect_repo_owned_mlx_runtime_artifacts_dir_from(
    start_dir: &Path,
) -> Option<MlxRuntimeArtifactsSelection> {
    for candidate_root in start_dir.ancestors().take(20) {
        let manifest_path = candidate_root.join("metal/phase1-kernels.json");
        let build_dir = candidate_root.join("build/metal");
        let build_report_path = build_dir.join("build_report.json");

        if !manifest_path.is_file() || !build_report_path.is_file() {
            continue;
        }

        // Repo auto-detect should stay conservative: only opt into the Metal
        // bring-up path when the checked-in asset contract validates end to end.
        if MetalKernelAssets::from_build_dir(&build_dir).is_ok() {
            return Some(MlxRuntimeArtifactsSelection {
                dir: build_dir,
                source: NativeRuntimeArtifactsSource::RepoAutoDetect,
            });
        }
    }

    None
}

pub(super) fn resolve_native_model_report(
    config: &EngineSessionConfig,
    core: &EngineCore,
) -> Option<NativeModelReport> {
    let source = config.mlx_model_artifacts_source?;
    let summary = core.native_model_artifacts_summary()?;
    let binding = core.native_model_binding_summary();
    Some(NativeModelReport::from_summary(source, summary, binding))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use ax_engine_core::metal::{
        MetalBuildOutputs, PHASE1_DEFAULT_BLOCK_SIZE_TOKENS, PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS,
    };
    use ax_engine_core::{
        AX_NATIVE_MODEL_MANIFEST_FILE, AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, CacheGroupId,
        DeterministicSampler, ExecutionRunner, ExecutionStatus, KvManagerConfig, KvWriteSummary,
        MetalBuildDoctorReport, MetalBuildHostReport, MetalBuildReport, MetalBuildStatus,
        MetalBuildToolStatus, MetalBuildToolchainReport, MetalKernelManifest, MetalKernelSpec,
        MetalKernelTier, NativeDiffusionConfig, NativeLinearAttentionConfig, NativeModelArtifacts,
        NativeModelArtifactsSummary, NativeModelManifest, NativeTensorDataType, NativeTensorFormat,
        NativeTensorRole, NativeTensorSpec, PHASE1_METAL_BUILD_GATE,
        PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION, PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION,
        PHASE1_METAL_LANGUAGE_STANDARD, PHASE1_METAL_LIBRARY_NAME, PHASE1_MLX_METAL_TARGET,
        RequestExecutionUpdate, RunnerInput, RunnerOutput,
    };
    use sha2::{Digest, Sha256};

    use crate::backend::{NativeRuntimeArtifactsSource, NativeRuntimeReport};

    #[derive(Clone, Debug)]
    struct NativeModelReportingRunner {
        summary: NativeModelArtifactsSummary,
        binding: Option<ax_engine_core::NativeModelBindingSummary>,
    }

    impl ExecutionRunner for NativeModelReportingRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            let blocks_touched = input
                .block_tables
                .iter()
                .map(|resolved| resolved.block_table.block_ids.len() as u32)
                .sum();
            let logits_handles = input
                .execution_batch
                .items
                .iter()
                .filter(|item| matches!(item.mode, ax_engine_core::ExecutionMode::Decode))
                .map(|item| item.request_id)
                .collect();
            let request_updates = input
                .execution_batch
                .items
                .iter()
                .map(|item| RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: item.scheduled_token_count,
                    output_token: None,
                    output_tokens: Vec::new(),
                    stop_reason: None,
                    error: None,
                    diffusion_schedule: None,
                })
                .collect();

            RunnerOutput {
                step_id: input.execution_batch.step_id,
                request_updates,
                logits_handles,
                logits_outputs: Vec::new(),
                kv_write_summary: KvWriteSummary {
                    tokens_written: input.execution_batch.total_scheduled_tokens,
                    blocks_touched,
                },
                route_metadata: input.execution_batch.route_metadata.clone(),
                execution_status: ExecutionStatus::Success,
            }
        }

        fn native_model_artifacts_summary(&self) -> Option<NativeModelArtifactsSummary> {
            Some(self.summary.clone())
        }

        fn native_model_binding_summary(
            &self,
        ) -> Option<ax_engine_core::NativeModelBindingSummary> {
            self.binding
        }
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        static NEXT_SUFFIX: AtomicU64 = AtomicU64::new(0);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let suffix = NEXT_SUFFIX.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ax-engine-session-{label}-{unique}-{suffix}"))
    }

    fn write_json_file<T: serde::Serialize>(path: &Path, value: &T) {
        let parent = path.parent().expect("test JSON path should have a parent");
        fs::create_dir_all(parent).expect("test JSON parent should create");
        let body = serde_json::to_string_pretty(value).expect("test JSON should serialize");
        fs::write(path, body).expect("test JSON should write");
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    fn phase1_source_text() -> &'static str {
        "#include <metal_stdlib>\nusing namespace metal;\n"
    }

    fn phase1_kernel_specs() -> Vec<MetalKernelSpec> {
        vec![MetalKernelSpec {
            name: "ax_phase1_dense_path".to_string(),
            tier: MetalKernelTier::Required,
            purpose: "test fixture".to_string(),
        }]
    }

    fn sample_build_doctor(
        bringup_allowed: bool,
        metal_toolchain_fully_available: bool,
    ) -> MetalBuildDoctorReport {
        MetalBuildDoctorReport {
            status: "pass".to_string(),
            bringup_allowed,
            mlx_runtime_ready: bringup_allowed && metal_toolchain_fully_available,
            metal_toolchain_fully_available,
            host: MetalBuildHostReport {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
                detected_soc: Some("Apple M-series".to_string()),
                supported_mlx_runtime: true,
                unsupported_host_override_active: false,
            },
            metal_toolchain: MetalBuildToolchainReport {
                fully_available: metal_toolchain_fully_available,
                metal: MetalBuildToolStatus {
                    available: metal_toolchain_fully_available,
                    version: Some("test-metal".to_string()),
                },
                metallib: MetalBuildToolStatus {
                    available: metal_toolchain_fully_available,
                    version: Some("test-metallib".to_string()),
                },
                metal_ar: MetalBuildToolStatus {
                    available: metal_toolchain_fully_available,
                    version: Some("test-metal-ar".to_string()),
                },
            },
        }
    }

    fn write_repo_owned_mlx_runtime_status_fixture(status: &str) -> (PathBuf, PathBuf) {
        let repo_root = unique_test_dir("metal-runtime-fixture");
        let manifest_path = repo_root.join("metal/phase1-kernels.json");
        let build_dir = repo_root.join("build/metal");
        let build_report_path = build_dir.join("build_report.json");

        fs::create_dir_all(
            manifest_path
                .parent()
                .expect("manifest parent should exist for fixture"),
        )
        .expect("fixture manifest directory should create");
        fs::create_dir_all(&build_dir).expect("fixture build directory should create");
        fs::write(&manifest_path, "{}").expect("fixture manifest should write");
        fs::write(
            &build_report_path,
            serde_json::json!({ "status": status }).to_string(),
        )
        .expect("fixture build report should write");

        (repo_root, build_dir)
    }

    fn write_valid_repo_owned_mlx_runtime_fixture() -> (PathBuf, PathBuf) {
        let repo_root = unique_test_dir("metal-runtime-valid-fixture");
        let manifest_path = repo_root.join("metal/phase1-kernels.json");
        let source_path = repo_root.join("metal/kernels/phase1_dense_path.metal");
        let build_dir = repo_root.join("build/metal");
        let air_path = build_dir.join("ax_phase1_dense_path.air");
        let metalar_path = build_dir.join("ax_phase1_dense_path.metalar");
        let metallib_path = build_dir.join("ax_phase1_dense_path.metallib");

        fs::create_dir_all(
            manifest_path
                .parent()
                .expect("manifest parent should exist for fixture"),
        )
        .expect("fixture manifest directory should create");
        fs::create_dir_all(
            source_path
                .parent()
                .expect("source parent should exist for fixture"),
        )
        .expect("fixture source directory should create");
        fs::create_dir_all(&build_dir).expect("fixture build directory should create");

        let source_text = phase1_source_text();
        fs::write(&source_path, source_text.as_bytes()).expect("fixture source should write");
        fs::write(&air_path, b"fake-air").expect("fixture air should write");
        fs::write(&metalar_path, b"fake-metalar").expect("fixture metalar should write");
        fs::write(&metallib_path, b"fake-metallib").expect("fixture metallib should write");

        let manifest = MetalKernelManifest {
            schema_version: PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION.to_string(),
            mlx_target: PHASE1_MLX_METAL_TARGET.to_string(),
            metal_language_standard: PHASE1_METAL_LANGUAGE_STANDARD.to_string(),
            library_name: PHASE1_METAL_LIBRARY_NAME.to_string(),
            default_block_size_tokens: PHASE1_DEFAULT_BLOCK_SIZE_TOKENS,
            supported_block_size_tokens: PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS.to_vec(),
            source_file: PathBuf::from("metal/kernels/phase1_dense_path.metal"),
            toolchain_requirements: ["xcrun metal", "xcrun metallib"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            build_gate: PHASE1_METAL_BUILD_GATE.to_string(),
            kernels: phase1_kernel_specs(),
        };
        write_json_file(&manifest_path, &manifest);

        let build_report = MetalBuildReport {
            schema_version: PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION.to_string(),
            manifest_path: manifest_path.clone(),
            source_file: source_path.clone(),
            mlx_target: manifest.mlx_target.clone(),
            metal_language_standard: manifest.metal_language_standard.clone(),
            library_name: manifest.library_name.clone(),
            default_block_size_tokens: manifest.default_block_size_tokens,
            supported_block_size_tokens: manifest.supported_block_size_tokens.clone(),
            toolchain_requirements: manifest.toolchain_requirements.clone(),
            doctor: sample_build_doctor(true, true),
            kernels: manifest.kernels.clone(),
            source_sha256: sha256_hex(source_text.as_bytes()),
            outputs: MetalBuildOutputs {
                air: Some(air_path.clone()),
                metalar: Some(metalar_path.clone()),
                metallib: Some(metallib_path.clone()),
                air_sha256: Some(sha256_hex(b"fake-air")),
                metalar_sha256: Some(sha256_hex(b"fake-metalar")),
                metallib_sha256: Some(sha256_hex(b"fake-metallib")),
            },
            compile_commands: vec![
                vec!["xcrun".to_string(), "metal".to_string()],
                vec!["xcrun".to_string(), "metal-ar".to_string()],
                vec!["xcrun".to_string(), "metallib".to_string()],
            ],
            status: MetalBuildStatus::Compiled,
            reason: None,
        };
        write_json_file(&build_dir.join("build_report.json"), &build_report);

        (repo_root, build_dir)
    }

    fn write_valid_native_model_fixture_into(root_dir: &Path) {
        fs::create_dir_all(root_dir).expect("native model fixture directory should create");
        fs::write(root_dir.join("model.safetensors"), vec![0_u8; 4096])
            .expect("native model weights should write");
        let manifest = NativeModelManifest {
            schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: ax_engine_core::model::NativeRuntimeStatus::default(),
            layer_count: 1,
            hidden_size: 2048,
            intermediate_size: 11008,
            attention_head_count: 16,
            attention_head_dim: 128,
            kv_head_count: 8,
            vocab_size: 151936,
            tie_word_embeddings: false,
            rope_theta: None,
            rope_theta_swa: None,
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_low_freq_factor: None,
            rope_high_freq_factor: None,
            rope_original_context_len: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: None,
            attn_temperature_scale: None,
            intermediate_size_mlp: 0,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            rms_norm_eps: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: std::collections::BTreeMap::new(),
            final_logit_softcapping: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig::default(),
            mla_attention: Default::default(),
            moe: ax_engine_core::NativeMoeConfig::default(),
            glm_router: Default::default(),
            weight_sanitize: ax_engine_core::WeightSanitize::None,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: NativeDiffusionConfig::default(),
            dropped_tensors: Default::default(),
            tensors: vec![
                native_model_tensor(
                    "model.embed_tokens.weight",
                    NativeTensorRole::TokenEmbedding,
                    None,
                    vec![151_936, 2_048],
                ),
                native_model_tensor(
                    "model.norm.weight",
                    NativeTensorRole::FinalNorm,
                    None,
                    vec![2_048],
                ),
                native_model_tensor(
                    "lm_head.weight",
                    NativeTensorRole::LmHead,
                    None,
                    vec![151_936, 2_048],
                ),
                native_model_tensor(
                    "model.layers.0.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(0),
                    vec![2_048],
                ),
                native_model_tensor(
                    "model.layers.0.self_attn.qkv_proj.weight",
                    NativeTensorRole::AttentionQkvPacked,
                    Some(0),
                    vec![4_096, 2_048],
                ),
                native_model_tensor(
                    "model.layers.0.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(0),
                    vec![2_048, 2_048],
                ),
                native_model_tensor(
                    "model.layers.0.post_attention_layernorm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(0),
                    vec![2_048],
                ),
                native_model_tensor(
                    "model.layers.0.mlp.gate_up_proj.weight",
                    NativeTensorRole::FfnGateUpPacked,
                    Some(0),
                    vec![8_192, 2_048],
                ),
                native_model_tensor(
                    "model.layers.0.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(0),
                    vec![2_048, 4_096],
                ),
            ],
        };
        write_json_file(&root_dir.join(AX_NATIVE_MODEL_MANIFEST_FILE), &manifest);
    }

    fn write_valid_native_model_fixture() -> PathBuf {
        let root_dir = unique_test_dir("native-model-valid-fixture");
        write_valid_native_model_fixture_into(&root_dir);
        root_dir
    }

    fn native_model_tensor(
        name: &str,
        role: NativeTensorRole,
        layer_index: Option<u32>,
        shape: Vec<u64>,
    ) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index,
            dtype: NativeTensorDataType::F16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape,
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 32,
        }
    }

    #[test]
    fn default_mlx_runtime_artifacts_dir_prefers_explicit_env_path() {
        let (repo_root, _build_dir) = write_valid_repo_owned_mlx_runtime_fixture();
        let nested_dir = repo_root.join("crates/ax-engine-sdk/src");
        fs::create_dir_all(&nested_dir).expect("fixture nested directory should create");
        let explicit_dir = unique_test_dir("explicit-metal-build-dir");

        let detected = resolve_default_mlx_runtime_artifacts_selection(
            Some(explicit_dir.clone()),
            Some(&nested_dir),
        );

        assert_eq!(
            detected.as_ref().map(|selection| selection.dir.as_path()),
            Some(explicit_dir.as_path())
        );
        assert_eq!(
            detected.as_ref().map(|selection| selection.source),
            Some(NativeRuntimeArtifactsSource::ExplicitEnv)
        );
        let _ = fs::remove_dir_all(repo_root);
    }

    #[test]
    fn default_mlx_runtime_artifacts_dir_ignores_non_compiled_repo_fixture() {
        let (repo_root, _) =
            write_repo_owned_mlx_runtime_status_fixture("skipped_toolchain_unavailable");
        let nested_dir = repo_root.join("crates/ax-engine-sdk/src");
        fs::create_dir_all(&nested_dir).expect("fixture nested directory should create");

        let detected = resolve_default_mlx_runtime_artifacts_selection(None, Some(&nested_dir));

        assert_eq!(detected, None);
        let _ = fs::remove_dir_all(repo_root);
    }

    #[test]
    fn default_mlx_runtime_artifacts_dir_ignores_invalid_compiled_repo_fixture() {
        let (repo_root, _) = write_repo_owned_mlx_runtime_status_fixture("compiled");
        let nested_dir = repo_root.join("crates/ax-engine-sdk/src");
        fs::create_dir_all(&nested_dir).expect("fixture nested directory should create");

        let detected = resolve_default_mlx_runtime_artifacts_selection(None, Some(&nested_dir));

        assert_eq!(detected, None);
        let _ = fs::remove_dir_all(repo_root);
    }

    #[test]
    fn mlx_runtime_report_marks_explicit_config_metal_runner() {
        let report = EngineSessionConfig {
            mlx_runtime_artifacts_dir: Some(Path::new("/tmp/ax-metal").to_path_buf()),
            mlx_runtime_artifacts_source: None,
            ..EngineSessionConfig::default()
        }
        .runtime_report();

        assert_eq!(
            report.mlx_runtime,
            Some(NativeRuntimeReport::metal_bringup(
                NativeRuntimeArtifactsSource::ExplicitConfig,
            ))
        );
    }

    #[test]
    fn resolve_native_model_report_uses_runner_owned_validated_summary() {
        let config = EngineSessionConfig {
            mlx_model_artifacts_dir: Some(PathBuf::from("/tmp/ax-model")),
            mlx_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
            ..EngineSessionConfig::default()
        };
        let model_dir = write_valid_native_model_fixture();
        let summary = NativeModelArtifacts::from_dir(&model_dir)
            .expect("native model fixture should validate")
            .summary();
        let core = EngineCore::with_runtime_components(
            KvManagerConfig::validated(CacheGroupId(0), 16, 64),
            NativeModelReportingRunner {
                summary,
                binding: Some(ax_engine_core::NativeModelBindingSummary {
                    bindings_prepared: true,
                    buffers_bound: true,
                    buffer_count: 9,
                    buffer_bytes: 288,
                    source_quantized_binding_count: 3,
                    source_q4_k_binding_count: 1,
                    source_q5_k_binding_count: 1,
                    source_q6_k_binding_count: 1,
                    source_q8_0_binding_count: 0,
                }),
            },
            DeterministicSampler,
        );

        let report =
            resolve_native_model_report(&config, &core).expect("native model report should build");

        assert_eq!(
            report,
            NativeModelReport {
                artifacts_source: NativeModelArtifactsSource::ExplicitConfig,
                model_family: "qwen3".to_string(),
                tensor_format: NativeTensorFormat::Safetensors,
                source_quantization: None,
                runtime_status: ax_engine_core::model::NativeRuntimeStatus::default(),
                layer_count: 1,
                tensor_count: 9,
                tie_word_embeddings: false,
                is_moe: false,
                is_hybrid_attention: false,
                hybrid_full_attention_interval: None,
                mla_kv_latent_dim: None,
                moe_active_experts: None,
                bindings_prepared: true,
                buffers_bound: true,
                buffer_count: 9,
                buffer_bytes: 288,
                source_quantized_binding_count: 3,
                source_q4_k_binding_count: 1,
                source_q5_k_binding_count: 1,
                source_q6_k_binding_count: 1,
                source_q8_0_binding_count: 0,
            }
        );

        let _ = fs::remove_dir_all(model_dir);
    }
}
