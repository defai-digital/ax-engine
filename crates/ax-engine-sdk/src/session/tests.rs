use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_core::metal::{
    MetalBuildOutputs, MetalDispatchExecutionInfo, MetalDispatchKvMetadata,
    MetalDispatchNumericTrace, MetalDispatchRuntimeInfo, MetalNumericValidationSummary,
    PHASE1_DEFAULT_BLOCK_SIZE_TOKENS, PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS,
};
use ax_engine_core::{
    AX_NATIVE_MODEL_MANIFEST_FILE, AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, CacheGroupId,
    DeterministicSampler, ExecutionRunner, ExecutionStatus, KvManagerConfig, KvWriteSummary,
    MetalBinaryArchiveInfo, MetalBinaryArchiveState, MetalBuildDoctorReport, MetalBuildHostReport,
    MetalBuildReport, MetalBuildStatus, MetalBuildToolStatus, MetalBuildToolchainReport,
    MetalCommandBufferStatus, MetalDispatchArenaInfo, MetalDispatchKernelTrace, MetalDispatchTrace,
    MetalDispatchWorkload, MetalKernelManifest, MetalKernelSpec, MetalKernelTier,
    MetalThreadgroupSize, MlxKvCompressionConfig, ModelId, NativeLinearAttentionConfig,
    NativeModelArtifacts, NativeModelArtifactsSummary, NativeModelManifest, NativeTensorDataType,
    NativeTensorFormat, NativeTensorRole, NativeTensorSpec, PHASE1_METAL_BUILD_GATE,
    PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION, PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION,
    PHASE1_METAL_LANGUAGE_STANDARD, PHASE1_METAL_LIBRARY_NAME, PHASE1_MLX_METAL_TARGET,
    RequestExecutionUpdate, RunnerInput, RunnerOutput, SamplingParams, SequenceNo,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::*;
use crate::backend::{
    BackendPolicy, NativeModelArtifactsSource, NativeModelReport, NativeRuntimeArtifactsSource,
    NativeRuntimeReport, PreviewBackendRequest, ResolvedBackend, SupportTier,
};
use crate::generate::GenerateFinishReason;
use crate::session::artifacts::resolve_default_mlx_runtime_artifacts_selection;
use crate::session::stream::{
    finish_reason_from_stop_type, terminal_stop_reason_from_finish_reason,
};
use crate::{LlamaCppBackendError, MlxLmConfig};

fn sample_submission() -> RequestSubmission {
    RequestSubmission {
        request_id: RequestId(1),
        model_id: ModelId("qwen3".into()),
        input_tokens: vec![1, 2, 3, 4],
        sampling_params: SamplingParams::default(),
        max_output_tokens: 2,
        arrival_sequence: SequenceNo(1),
        metadata: None,
    }
}

fn mlx_test_session_config() -> EngineSessionConfig {
    EngineSessionConfig {
        // Explicitly clear auto-detected repo artifacts because this test
        // injects a custom in-memory runner instead of building MLX artifacts.
        mlx_runtime_artifacts_dir: None,
        mlx_runtime_artifacts_source: None,
        ..EngineSessionConfig::default()
    }
}

fn sample_terminal_llama_report(request_id: u64) -> SessionRequestReport {
    SessionRequestReport {
        request_id,
        model_id: "qwen3".to_string(),
        state: SessionRequestState::Finished,
        prompt_tokens: vec![1, 2, 3],
        processed_prompt_tokens: 3,
        output_tokens: vec![4, 5],
        output_token_logprobs: vec![Some(-0.25), Some(-0.5)],
        prompt_len: 3,
        output_len: 2,
        max_output_tokens: 2,
        cancel_requested: false,
        execution_plan_ref: Some(LLAMA_CPP_STREAM_EXECUTION_PLAN.to_string()),
        route: llama_cpp_stream_route(),
        finish_reason: Some(GenerateFinishReason::Stop),
        terminal_stop_reason: None,
        last_error: None,
    }
}

#[derive(Clone, Debug)]
struct TraceReportingRunner {
    trace: MetalDispatchTrace,
}

#[derive(Clone, Debug)]
struct TerminalRouteReportingRunner {
    trace: MetalDispatchTrace,
}

#[derive(Clone, Debug)]
struct NativeModelReportingRunner {
    summary: NativeModelArtifactsSummary,
    binding: Option<ax_engine_core::NativeModelBindingSummary>,
}

impl ExecutionRunner for TraceReportingRunner {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        let request_updates = input
            .execution_batch
            .items
            .iter()
            .map(|item| RequestExecutionUpdate {
                request_id: item.request_id,
                tokens_executed: item.scheduled_token_count,
                output_token: None,
                stop_reason: None,
                error: None,
            })
            .collect::<Vec<_>>();
        let mut route_metadata = input.execution_batch.route_metadata.clone();
        route_metadata.attention_route = Some("mock_native_attention".to_string());
        route_metadata
            .crossover_decisions
            .push(("metal_dispatch_completed".to_string(), 1));

        RunnerOutput {
            step_id: input.execution_batch.step_id,
            request_updates,
            logits_handles: Vec::new(),
            logits_outputs: Vec::new(),
            kv_write_summary: KvWriteSummary {
                tokens_written: input.execution_batch.total_scheduled_tokens,
                blocks_touched: input
                    .block_tables
                    .iter()
                    .map(|resolved| resolved.block_table.block_ids.len() as u32)
                    .sum(),
            },
            route_metadata,
            execution_status: ExecutionStatus::Success,
        }
    }

    fn metal_dispatch_trace(&self) -> Option<MetalDispatchTrace> {
        Some(self.trace.clone())
    }
}

impl ExecutionRunner for TerminalRouteReportingRunner {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        let request_updates = input
            .execution_batch
            .items
            .iter()
            .map(|item| RequestExecutionUpdate {
                request_id: item.request_id,
                tokens_executed: item.scheduled_token_count,
                output_token: Some(42),
                stop_reason: Some(ax_engine_core::StopReason::MaxOutputTokens),
                error: None,
            })
            .collect::<Vec<_>>();
        let mut route_metadata = input.execution_batch.route_metadata.clone();
        route_metadata.attention_route = Some("mock_native_attention".to_string());
        route_metadata
            .crossover_decisions
            .push(("metal_dispatch_completed".to_string(), 1));

        RunnerOutput {
            step_id: input.execution_batch.step_id,
            request_updates,
            logits_handles: Vec::new(),
            logits_outputs: Vec::new(),
            kv_write_summary: KvWriteSummary {
                tokens_written: input.execution_batch.total_scheduled_tokens,
                blocks_touched: input
                    .block_tables
                    .iter()
                    .map(|resolved| resolved.block_table.block_ids.len() as u32)
                    .sum(),
            },
            route_metadata,
            execution_status: ExecutionStatus::Success,
        }
    }

    fn metal_dispatch_trace(&self) -> Option<MetalDispatchTrace> {
        Some(self.trace.clone())
    }
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
                stop_reason: None,
                error: None,
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

    fn native_model_binding_summary(&self) -> Option<ax_engine_core::NativeModelBindingSummary> {
        self.binding
    }
}

fn sample_metal_dispatch_trace() -> MetalDispatchTrace {
    MetalDispatchTrace {
        command_queue_label: "ax.queue".to_string(),
        command_buffer_label: "ax.buffer".to_string(),
        command_buffer_status: MetalCommandBufferStatus::Completed,
        runtime: MetalDispatchRuntimeInfo {
            device_name: "Apple M4 Max".to_string(),
            required_pipeline_count: 4,
            max_thread_execution_width: 64,
            binary_archive: MetalBinaryArchiveInfo {
                path: std::path::PathBuf::from("/tmp/ax_phase1_dense_path.binary_archive.metallib"),
                state: MetalBinaryArchiveState::Loaded,
                attached_pipeline_count: 4,
                serialized: true,
                note: None,
            },
            command_queue_ready: true,
            model_conditioned_inputs: true,
            real_model_tensor_inputs: true,
            complete_model_forward_supported: true,
            model_bindings_prepared: true,
            model_buffers_bound: true,
            model_buffer_count: 12,
            model_buffer_bytes: 4096,
            native_dense_kernel_coverage: Default::default(),
            model: Some(NativeModelArtifactsSummary {
                model_family: "qwen3".to_string(),
                tensor_format: NativeTensorFormat::Safetensors,
                source_quantization: None,
                runtime_status: ax_engine_core::model::NativeRuntimeStatus::default(),
                layer_count: 36,
                tensor_count: 512,
                tie_word_embeddings: false,
                is_moe: false,
                is_hybrid_attention: false,
                hybrid_full_attention_interval: None,
                mla_kv_latent_dim: None,
                moe_active_experts: None,
            }),
        },
        workload: MetalDispatchWorkload {
            scheduled_requests: 1,
            prefill_requests: 1,
            decode_requests: 0,
            scheduled_tokens: 2,
            scheduled_token_ids: vec![10, 11],
            scheduled_positions: vec![0, 1],
            resolved_blocks: 1,
            token_elements: 2,
            block_elements: 16,
            scratch_elements: 32,
            kv_slot_capacity: 16,
            kv_block_capacity: 1,
            numeric_layout: ax_engine_core::MetalDispatchNumericLayout::default(),
            kv_metadata: MetalDispatchKvMetadata {
                block_size_tokens: 16,
                slot_mapping: vec![0, 1],
                attention_block_table: vec![0, 1],
                gather_block_table: vec![0],
                gather_block_table_stride: 1,
                copy_block_mapping: vec![[0, 0]],
                seq_lens: vec![2],
                cu_seq_lens: vec![0, 2],
                scheduled_cu_seq_lens: vec![0, 2],
            },
        },
        arena: MetalDispatchArenaInfo {
            token_capacity: 8,
            slot_capacity: 64,
            attention_ref_capacity: 8,
            gather_ref_capacity: 8,
            gather_output_capacity: 8,
            copy_pair_capacity: 4,
            sequence_capacity: 4,
            reused_existing: true,
            grew_existing: false,
        },
        execution: MetalDispatchExecutionInfo {
            direct_decode_token_count: 1,
            direct_decode_checksum_lo: 0x1234,
            logits_output_count: 1,
            remaining_logits_handle_count: 0,
            model_bound_ffn_decode: true,
            real_model_forward_completed: true,
            prefix_native_dispatch_count: 35,
            prefix_cpu_reference_dispatch_count: 1,
            qkv_projection_token_count: 72,
            layer_continuation_token_count: 37,
            logits_projection_token_count: 1,
            logits_vocab_scan_row_count: 151936,
            direct_decode_native_projection_row_count: 0,
            direct_decode_cpu_projection_row_count: 0,
            direct_decode_native_rms_norm_element_count: 0,
            direct_decode_cpu_rms_norm_element_count: 0,
            prefix_native_projection_row_count: 0,
            prefix_cpu_projection_row_count: 0,
            prefix_native_rms_norm_element_count: 0,
            prefix_cpu_rms_norm_element_count: 0,
            direct_decode_native_ffn_activation_element_count: 0,
            direct_decode_cpu_ffn_activation_element_count: 0,
            prefix_native_ffn_activation_element_count: 0,
            prefix_cpu_ffn_activation_element_count: 0,
            direct_decode_batched_logits_group_count: 0,
            direct_decode_batched_logits_token_count: 0,
            direct_decode_batched_group_fallback_count: 0,
            direct_decode_batched_group_fallback_token_count: 0,
            direct_decode_native_residual_add_element_count: 0,
            direct_decode_cpu_residual_add_element_count: 0,
            prefix_native_residual_add_element_count: 0,
            prefix_cpu_residual_add_element_count: 0,
            prefix_native_scale_element_count: 0,
            prefix_cpu_scale_element_count: 0,
            direct_decode_native_scale_element_count: 0,
            direct_decode_cpu_scale_element_count: 0,
        },
        kernels: vec![
            MetalDispatchKernelTrace {
                function_name: "reshape_and_cache".to_string(),
                element_count: 32,
                threads_per_grid: MetalThreadgroupSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
                threads_per_threadgroup: MetalThreadgroupSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            },
            MetalDispatchKernelTrace {
                function_name: "paged_decode_attention".to_string(),
                element_count: 16,
                threads_per_grid: MetalThreadgroupSize {
                    width: 16,
                    height: 1,
                    depth: 1,
                },
                threads_per_threadgroup: MetalThreadgroupSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            },
        ],
        numeric: MetalDispatchNumericTrace {
            attention_output_bits: vec![0],
            key_cache_checksum: 0x11,
            attention_output_checksum: 0x22,
            gather_output_checksum: 0x33,
            copy_output_checksum: 0x44,
            validation: Some(MetalNumericValidationSummary {
                expected_key_cache_checksum: 0x11,
                expected_attention_output_checksum: 0x22,
                expected_gather_output_checksum: 0x33,
                expected_copy_output_checksum: 0x44,
                attention_max_abs_diff_microunits: 0,
            }),
        },
    }
}

fn unique_test_dir(label: &str) -> std::path::PathBuf {
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

fn read_http_request(stream: &mut impl Read) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let read = stream.read(&mut buffer).expect("request should read");
        if read == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..read]);
        if let Some(header_end) = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
        {
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.strip_prefix("Content-Length:")
                        .or_else(|| line.strip_prefix("content-length:"))
                        .and_then(|value| value.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            if request.len() >= header_end + content_length {
                break;
            }
        }
    }
    request
}

fn write_repo_owned_mlx_runtime_status_fixture(
    status: &str,
) -> (std::path::PathBuf, std::path::PathBuf) {
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

fn write_valid_repo_owned_mlx_runtime_fixture() -> (std::path::PathBuf, std::path::PathBuf) {
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
        source_file: std::path::PathBuf::from("metal/kernels/phase1_dense_path.metal"),
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

fn write_valid_native_model_fixture() -> std::path::PathBuf {
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

fn fake_llama_cpp_script() -> std::path::PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("ax-engine-session-llama-cpp-{unique}.py"));
    let script = r#"#!/usr/bin/env python3
from __future__ import annotations

import sys

args = sys.argv[1:]
prompt = args[args.index("--prompt") + 1]
sys.stdout.write(f"session::{prompt}")
"#;

    fs::write(&path, script).expect("fake script should be written");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(&path)
            .expect("script metadata should exist")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).expect("script should be executable");
    }
    path
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
        KvManagerConfig::new(CacheGroupId(0), 16, 64),
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

#[test]
fn preview_session_config_factory_builds_mlx_preview_defaults() {
    let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest::default())
        .expect("preview config factory should build MLX defaults");

    assert_eq!(
        config.kv_config,
        KvManagerConfig::new(CacheGroupId(0), 16, 1024)
    );
    assert!(config.deterministic);
    assert_eq!(config.max_batch_tokens, 2048);
    assert_eq!(config.backend_policy, BackendPolicy::mlx_only());
    assert_eq!(config.resolved_backend, ResolvedBackend::mlx_preview());
    assert!(config.llama_backend.is_none());
}

#[test]
fn preview_session_config_factory_builds_llama_cpp_server_backend() {
    let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
        backend_request: PreviewBackendRequest {
            support_tier: SupportTier::LlamaCpp,
            llama_server_url: Some("http://127.0.0.1:8080".to_string()),
            ..PreviewBackendRequest::default()
        },
        ..PreviewSessionConfigRequest::default()
    })
    .expect("preview config factory should build llama.cpp config");

    assert_eq!(config.backend_policy, BackendPolicy::allow_llama_cpp());
    assert_eq!(config.resolved_backend.support_tier, SupportTier::LlamaCpp);
    assert_eq!(
        config.resolved_backend.selected_backend,
        SelectedBackend::LlamaCpp
    );
    assert!(config.llama_backend.is_some());
}

#[test]
fn preview_session_config_factory_builds_mlx_lm_delegated_backend() {
    let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
        backend_request: PreviewBackendRequest {
            support_tier: SupportTier::MlxLmDelegated,
            mlx_lm_server_url: Some("http://127.0.0.1:8090".to_string()),
            ..PreviewBackendRequest::default()
        },
        ..PreviewSessionConfigRequest::default()
    })
    .expect("preview config factory should build mlx-lm delegated config");

    assert_eq!(
        config.backend_policy,
        BackendPolicy::allow_mlx_lm_delegated()
    );
    assert_eq!(
        config.resolved_backend.selected_backend,
        SelectedBackend::MlxLmDelegated
    );
    assert_eq!(
        config.resolved_backend.support_tier,
        SupportTier::MlxLmDelegated
    );
    assert!(config.llama_backend.is_none());
    assert_eq!(
        config.mlx_lm_backend,
        Some(MlxLmConfig::server_completion("http://127.0.0.1:8090"))
    );

    let runtime = config.runtime_report();
    assert_eq!(runtime.selected_backend, SelectedBackend::MlxLmDelegated);
    assert_eq!(runtime.support_tier, SupportTier::MlxLmDelegated);
    assert!(runtime.capabilities.text_generation);
    assert!(runtime.capabilities.token_streaming);
    assert!(runtime.mlx_runtime.is_none());
}

#[test]
fn preview_session_config_factory_preserves_explicit_native_artifact_dirs() {
    let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
        mlx_runtime_artifacts_dir: Some(Path::new("/tmp/ax-metal").to_path_buf()),
        mlx_model_artifacts_dir: Some(Path::new("/tmp/ax-model").to_path_buf()),
        ..PreviewSessionConfigRequest::default()
    })
    .expect("preview config factory should preserve explicit native artifact config");

    assert_eq!(
        config.mlx_runtime_artifacts_dir.as_deref(),
        Some(Path::new("/tmp/ax-metal"))
    );
    assert_eq!(
        config.mlx_runtime_artifacts_source,
        Some(NativeRuntimeArtifactsSource::ExplicitConfig)
    );
    assert_eq!(
        config.mlx_model_artifacts_dir.as_deref(),
        Some(Path::new("/tmp/ax-model"))
    );
    assert_eq!(
        config.mlx_model_artifacts_source,
        Some(NativeModelArtifactsSource::ExplicitConfig)
    );
}

#[test]
fn preview_session_config_factory_preserves_explicit_gguf_model_path_source() {
    let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
        mlx_model_artifacts_dir: Some(
            Path::new("/tmp/google_gemma-4-26b-it-q4_k_m.gguf").to_path_buf(),
        ),
        ..PreviewSessionConfigRequest::default()
    })
    .expect("preview config factory should preserve explicit gguf model config");

    assert_eq!(
        config.mlx_model_artifacts_dir.as_deref(),
        Some(Path::new("/tmp/google_gemma-4-26b-it-q4_k_m.gguf"))
    );
    assert_eq!(
        config.mlx_model_artifacts_source,
        Some(NativeModelArtifactsSource::ExplicitConfig)
    );
}

#[test]
fn resolved_session_config_factory_preserves_supplied_runtime_fields() {
    let config = EngineSessionConfig::from_resolved_request(ResolvedSessionConfigRequest {
        cache_group_id: CacheGroupId(7),
        block_size_tokens: 32,
        total_blocks: 2048,
        deterministic: false,
        max_batch_tokens: 4096,
        backend_policy: BackendPolicy::allow_llama_cpp(),
        resolved_backend: ResolvedBackend::llama_cpp(
            SelectedBackend::LlamaCpp,
            "MLX preview not ready for this model",
        ),
        llama_backend: Some(crate::llama_cpp::LlamaCppConfig::server_completion(
            "http://127.0.0.1:8080".to_string(),
        )),
        mlx_lm_backend: None,
        mlx_runtime_artifacts_dir: Some(Path::new("/tmp/ax-metal").to_path_buf()),
        mlx_runtime_artifacts_source: Some(NativeRuntimeArtifactsSource::ExplicitConfig),
        mlx_model_artifacts_dir: Some(Path::new("/tmp/ax-model").to_path_buf()),
        mlx_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
        mlx_disable_ngram_acceleration: true,
        mlx_kv_compression: MlxKvCompressionConfig::turboquant_shadow(),
        mlx_prefill_chunk: None,
    });

    assert_eq!(
        config.kv_config,
        KvManagerConfig::new(CacheGroupId(7), 32, 2048)
    );
    assert!(!config.deterministic);
    assert_eq!(config.max_batch_tokens, 4096);
    assert_eq!(config.backend_policy, BackendPolicy::allow_llama_cpp());
    assert_eq!(
        config.resolved_backend,
        ResolvedBackend::llama_cpp(
            SelectedBackend::LlamaCpp,
            "MLX preview not ready for this model",
        )
    );
    assert!(config.llama_backend.is_some());
    assert_eq!(
        config.mlx_runtime_artifacts_dir.as_deref(),
        Some(Path::new("/tmp/ax-metal"))
    );
    assert_eq!(
        config.mlx_runtime_artifacts_source,
        Some(NativeRuntimeArtifactsSource::ExplicitConfig)
    );
    assert_eq!(
        config.mlx_model_artifacts_dir.as_deref(),
        Some(Path::new("/tmp/ax-model"))
    );
    assert_eq!(
        config.mlx_model_artifacts_source,
        Some(NativeModelArtifactsSource::ExplicitConfig)
    );
    assert!(config.mlx_disable_ngram_acceleration);
    assert_eq!(
        config.mlx_kv_compression,
        MlxKvCompressionConfig::turboquant_shadow()
    );
}

fn llama_cpp_server_session(server_url: String) -> EngineSession {
    let config = EngineSessionConfig {
        backend_policy: BackendPolicy::allow_llama_cpp(),
        resolved_backend: ResolvedBackend::llama_cpp(
            SelectedBackend::LlamaCpp,
            "MLX preview not ready for this model",
        ),
        llama_backend: Some(crate::llama_cpp::LlamaCppConfig::server_completion(
            server_url,
        )),
        ..EngineSessionConfig::default()
    };
    EngineSession::new(config).expect("llama.cpp session should build")
}

fn spawn_scripted_llama_cpp_completion_stream_server(
    expected_requests: usize,
    response_for_request: impl Fn(usize, &Value) -> Vec<Value> + Send + 'static,
) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
    let address = listener.local_addr().expect("listener should have address");
    let handle = thread::spawn(move || {
        for request_index in 0..expected_requests {
            let (mut stream, _) = listener.accept().expect("request should arrive");
            let request = read_http_request(&mut stream);
            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4)
                .expect("request should include header terminator");
            let body =
                String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
            let payload: Value = serde_json::from_str(&body).expect("request body should be json");

            let chunks = response_for_request(request_index, &payload);
            let mut body = String::new();
            for chunk in &chunks {
                body.push_str("data: ");
                body.push_str(
                    &serde_json::to_string(chunk).expect("chunk payload should serialize"),
                );
                body.push_str("\n\n");
            }
            body.push_str("data: [DONE]\n\n");

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("response should write");
        }
    });

    (format!("http://{address}"), handle)
}

fn spawn_llama_cpp_completion_stream_server(
    expected_requests: usize,
    chunks: Vec<Value>,
    assert_request: impl Fn(Value) + Send + Sync + 'static,
) -> (String, thread::JoinHandle<()>) {
    spawn_scripted_llama_cpp_completion_stream_server(expected_requests, move |_, payload| {
        assert_request(payload.clone());
        chunks.clone()
    })
}

// Same SSE wire format as spawn_llama_cpp_completion_stream_server; the difference is that the
// caller supplies OpenAI-format JSON payloads ({"choices": [...]}) rather than llama.cpp-native
// ones ({"content": ..., "stop": ...}).
#[test]
fn llama_cpp_stream_generate_supports_server_completion_adapter() {
    let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![
            serde_json::json!({
                "content": "hello",
                "tokens": [4],
                "stop": false
            }),
            serde_json::json!({
                "content": " world",
                "tokens": [5],
                "stop": true,
                "stop_type": "limit"
            }),
        ],
        |payload| {
            assert_eq!(payload.get("prompt"), Some(&serde_json::json!([1, 2, 3])));
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
        },
    );
    let mut session = llama_cpp_server_session(server_url);

    let events = session
        .stream_generate(GenerateRequest {
            model_id: "qwen3".to_string(),
            input_tokens: vec![1, 2, 3],
            input_text: None,
            max_output_tokens: 2,
            sampling: Default::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        })
        .expect("llama.cpp stream should start")
        .collect::<Result<Vec<_>, _>>()
        .expect("llama.cpp stream should complete");

    server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert!(matches!(
        events.first(),
        Some(GenerateStreamEvent::Request(_))
    ));
    assert_eq!(events.len(), 4);

    let GenerateStreamEvent::Step(first_step) = &events[1] else {
        panic!("second event should be first step");
    };
    assert_eq!(first_step.request.state, SessionRequestState::Running);
    assert_eq!(first_step.delta_tokens, vec![4]);
    assert_eq!(first_step.delta_token_logprobs, vec![None]);
    assert_eq!(first_step.step.ttft_events, 1);

    let GenerateStreamEvent::Step(final_step) = &events[2] else {
        panic!("third event should be terminal step");
    };
    assert_eq!(final_step.request.state, SessionRequestState::Finished);
    assert_eq!(final_step.delta_tokens, vec![5]);
    assert_eq!(final_step.delta_token_logprobs, vec![None]);

    let GenerateStreamEvent::Response(response_event) = &events[3] else {
        panic!("final event should be response");
    };
    assert_eq!(response_event.response.request_id, 1);
    assert_eq!(response_event.response.prompt_tokens, vec![1, 2, 3]);
    assert_eq!(response_event.response.output_tokens, vec![4, 5]);
    assert_eq!(
        response_event.response.output_token_logprobs,
        vec![None, None]
    );
    assert_eq!(
        response_event.response.output_text.as_deref(),
        Some("hello world")
    );
    assert_eq!(
        response_event.response.route.execution_plan.as_deref(),
        Some(LLAMA_CPP_STREAM_EXECUTION_PLAN)
    );
    assert_eq!(response_event.response.step_count, 2);
    assert_eq!(response_event.response.ttft_step, Some(1));
    assert_eq!(
        response_event.response.runtime.selected_backend,
        SelectedBackend::LlamaCpp
    );
    assert_eq!(
        response_event.response.finish_reason,
        Some(crate::generate::GenerateFinishReason::MaxOutputTokens)
    );
}

#[test]
fn llama_cpp_stream_generate_rejects_cli_fallback_adapter() {
    let script_path = fake_llama_cpp_script();
    let model_path = std::env::temp_dir().join("ax-engine-session-stream-cli-fallback-model.gguf");
    fs::write(&model_path, "fake gguf").expect("fake model should be written");

    let mut session = EngineSession::new(EngineSessionConfig {
        backend_policy: BackendPolicy::allow_llama_cpp(),
        resolved_backend: ResolvedBackend::llama_cpp(
            SelectedBackend::LlamaCpp,
            "MLX preview not ready for this model",
        ),
        llama_backend: Some(crate::llama_cpp::LlamaCppConfig::new(
            script_path,
            &model_path,
        )),
        ..EngineSessionConfig::default()
    })
    .expect("llama.cpp session should build");

    let error = session
        .stream_generate(GenerateRequest {
            model_id: "qwen3".to_string(),
            input_tokens: Vec::new(),
            input_text: Some("cli fallback stream".to_string()),
            max_output_tokens: 2,
            sampling: Default::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        })
        .expect_err("cli fallback streaming should fail closed");

    assert!(matches!(
        error,
        EngineSessionError::LlamaCpp(LlamaCppBackendError::StreamingNotSupported {
            selected_backend: SelectedBackend::LlamaCpp,
        })
    ));
}

#[test]
fn llama_cpp_cli_submit_generate_fails_closed() {
    let script_path = fake_llama_cpp_script();
    let model_path = std::env::temp_dir().join("ax-engine-session-fake-model-lifecycle.gguf");
    fs::write(&model_path, "fake gguf").expect("fake model should be written");

    let mut session = EngineSession::new(EngineSessionConfig {
        backend_policy: BackendPolicy::allow_llama_cpp(),
        resolved_backend: ResolvedBackend::llama_cpp(
            SelectedBackend::LlamaCpp,
            "MLX preview not ready for this model",
        ),
        llama_backend: Some(crate::llama_cpp::LlamaCppConfig::new(
            script_path,
            &model_path,
        )),
        ..EngineSessionConfig::default()
    })
    .expect("llama.cpp session should build");

    let error = session
        .submit_generate(GenerateRequest {
            model_id: "qwen3".to_string(),
            input_tokens: Vec::new(),
            input_text: Some("unsupported lifecycle".to_string()),
            max_output_tokens: 2,
            sampling: Default::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        })
        .expect_err("llama.cpp CLI submit should fail closed");

    match error {
        EngineSessionError::LlamaCpp(LlamaCppBackendError::StreamingNotSupported {
            selected_backend,
        }) => {
            assert_eq!(selected_backend, SelectedBackend::LlamaCpp);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn llama_cpp_stepwise_lifecycle_supports_server_completion_adapter() {
    let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![
            serde_json::json!({
                "content": "hello",
                "tokens": [4],
                "stop": false
            }),
            serde_json::json!({
                "content": " world",
                "tokens": [5],
                "stop": true,
                "stop_type": "limit"
            }),
        ],
        |payload| {
            assert_eq!(payload.get("prompt"), Some(&serde_json::json!([1, 2, 3])));
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
        },
    );
    let mut session = llama_cpp_server_session(server_url);

    let request_id = session
        .submit_generate(GenerateRequest {
            model_id: "qwen3".to_string(),
            input_tokens: vec![1, 2, 3],
            input_text: None,
            max_output_tokens: 2,
            sampling: Default::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        })
        .expect("llama.cpp submit should succeed");

    let initial = session
        .request_report(request_id)
        .expect("llama.cpp request should exist");
    assert_eq!(initial.state, SessionRequestState::Waiting);

    let first_step = session.step_report().expect("first step should succeed");
    assert_eq!(first_step.scheduled_requests, 1);
    assert_eq!(first_step.scheduled_tokens, 1);
    assert_eq!(first_step.ttft_events, 1);
    assert_eq!(
        first_step
            .route
            .as_ref()
            .and_then(|route| route.execution_plan.as_deref()),
        Some(LLAMA_CPP_STREAM_EXECUTION_PLAN)
    );

    let running = session
        .request_report(request_id)
        .expect("running llama.cpp request should exist");
    assert_eq!(running.state, SessionRequestState::Running);
    assert_eq!(running.output_tokens, vec![4]);

    let second_step = session.step_report().expect("second step should succeed");
    assert_eq!(second_step.scheduled_requests, 1);
    assert_eq!(second_step.scheduled_tokens, 1);
    assert_eq!(
        second_step
            .route
            .as_ref()
            .and_then(|route| route.execution_plan.as_deref()),
        Some(LLAMA_CPP_STREAM_EXECUTION_PLAN)
    );

    let terminal = session
        .request_report(request_id)
        .expect("terminal llama.cpp request should exist");
    assert_eq!(terminal.state, SessionRequestState::Finished);
    assert_eq!(terminal.output_tokens, vec![4, 5]);
    assert_eq!(
        terminal.execution_plan_ref.as_deref(),
        Some(LLAMA_CPP_STREAM_EXECUTION_PLAN)
    );

    server_handle
        .join()
        .expect("llama.cpp server thread should finish");
}

#[test]
fn llama_cpp_stepwise_lifecycle_reports_delegated_prompt_cache_hits() {
    let (server_url, server_handle) =
        spawn_scripted_llama_cpp_completion_stream_server(1, |_, payload| {
            assert_eq!(payload.get("prompt"), Some(&serde_json::json!([1, 2, 3])));
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));

            vec![
                serde_json::json!({
                    "content": "",
                    "tokens": [],
                    "stop": false,
                    "prompt_progress": {
                        "total": 3,
                        "cache": 2,
                        "processed": 2,
                        "time_ms": 1.0
                    }
                }),
                serde_json::json!({
                    "content": " cache",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " hit",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ]
        });
    let mut session = llama_cpp_server_session(server_url);

    let request_id = session
        .submit_generate(GenerateRequest {
            model_id: "qwen3".to_string(),
            input_tokens: vec![1, 2, 3],
            input_text: None,
            max_output_tokens: 2,
            sampling: Default::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        })
        .expect("llama.cpp submit should succeed");

    let progress_step = session.step_report().expect("progress step should succeed");
    assert_eq!(progress_step.scheduled_requests, 1);
    assert_eq!(progress_step.scheduled_tokens, 0);
    assert_eq!(progress_step.prefix_hits, 1);

    let progress_report = session
        .request_report(request_id)
        .expect("llama.cpp request should exist after progress");
    assert_eq!(progress_report.processed_prompt_tokens, 2);
    assert_eq!(progress_report.prompt_len, 3);
    assert_eq!(
        progress_report.route.prefix_cache_path.as_deref(),
        Some("delegated_prompt_cache")
    );
    assert_eq!(
        progress_report
            .route
            .crossover_decisions
            .get("delegated_cached_tokens"),
        Some(&2)
    );

    let first_decode_step = session
        .step_report()
        .expect("first decode step should succeed");
    assert_eq!(first_decode_step.scheduled_requests, 1);
    assert_eq!(first_decode_step.scheduled_tokens, 1);
    assert_eq!(first_decode_step.prefix_hits, 0);

    let second_decode_step = session
        .step_report()
        .expect("second decode step should succeed");
    assert_eq!(second_decode_step.scheduled_requests, 1);
    assert_eq!(second_decode_step.scheduled_tokens, 1);

    let terminal = session
        .request_report(request_id)
        .expect("terminal llama.cpp request should exist");
    assert_eq!(terminal.state, SessionRequestState::Finished);
    assert_eq!(terminal.output_tokens, vec![4, 5]);
    assert_eq!(
        terminal.route.prefix_cache_path.as_deref(),
        Some("delegated_prompt_cache")
    );

    server_handle
        .join()
        .expect("llama.cpp server thread should finish");
}

#[test]
fn llama_cpp_stepwise_lifecycle_advances_multiple_active_requests() {
    let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
        2,
        vec![
            serde_json::json!({
                "content": "hello",
                "tokens": [4],
                "stop": false
            }),
            serde_json::json!({
                "content": " world",
                "tokens": [5],
                "stop": true,
                "stop_type": "limit"
            }),
        ],
        |payload| {
            let prompt = payload.get("prompt").expect("prompt should be present");
            assert!(
                prompt == &serde_json::json!([1, 2, 3]) || prompt == &serde_json::json!([7, 8, 9])
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
        },
    );
    let mut session = llama_cpp_server_session(server_url);

    let first_request_id = session
        .submit_generate(GenerateRequest {
            model_id: "qwen3".to_string(),
            input_tokens: vec![1, 2, 3],
            input_text: None,
            max_output_tokens: 2,
            sampling: Default::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        })
        .expect("first llama.cpp submit should succeed");

    let second_request_id = session
        .submit_generate(GenerateRequest {
            model_id: "qwen3".to_string(),
            input_tokens: vec![7, 8, 9],
            input_text: None,
            max_output_tokens: 2,
            sampling: Default::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        })
        .expect("second llama.cpp submit should also succeed");

    let first_step = session
        .step_report()
        .expect("first aggregated step should succeed");
    assert_eq!(first_step.scheduled_requests, 2);
    assert_eq!(first_step.scheduled_tokens, 2);
    assert_eq!(first_step.ttft_events, 2);

    let first_running = session
        .request_report(first_request_id)
        .expect("first llama.cpp request should exist");
    assert_eq!(first_running.state, SessionRequestState::Running);
    assert_eq!(first_running.output_tokens, vec![4]);
    let second_running = session
        .request_report(second_request_id)
        .expect("second llama.cpp request should exist");
    assert_eq!(second_running.state, SessionRequestState::Running);
    assert_eq!(second_running.output_tokens, vec![4]);

    let second_step = session
        .step_report()
        .expect("second aggregated step should succeed");
    assert_eq!(second_step.scheduled_requests, 2);
    assert_eq!(second_step.scheduled_tokens, 2);

    let first_terminal = session
        .request_report(first_request_id)
        .expect("first terminal request should exist");
    assert_eq!(first_terminal.state, SessionRequestState::Finished);
    assert_eq!(first_terminal.output_tokens, vec![4, 5]);
    let second_terminal = session
        .request_report(second_request_id)
        .expect("second terminal request should exist");
    assert_eq!(second_terminal.state, SessionRequestState::Finished);
    assert_eq!(second_terminal.output_tokens, vec![4, 5]);

    server_handle
        .join()
        .expect("llama.cpp server thread should finish");
}

#[test]
fn llama_cpp_cancelled_request_does_not_block_other_active_requests() {
    let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
        2,
        vec![
            serde_json::json!({
                "content": "hello",
                "tokens": [4],
                "stop": false
            }),
            serde_json::json!({
                "content": " world",
                "tokens": [5],
                "stop": true,
                "stop_type": "limit"
            }),
        ],
        |payload| {
            let prompt = payload.get("prompt").expect("prompt should be present");
            assert!(
                prompt == &serde_json::json!([1, 2, 3]) || prompt == &serde_json::json!([7, 8, 9])
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
        },
    );
    let mut session = llama_cpp_server_session(server_url);

    let first_request_id = session
        .submit_generate(GenerateRequest {
            model_id: "qwen3".to_string(),
            input_tokens: vec![1, 2, 3],
            input_text: None,
            max_output_tokens: 2,
            sampling: Default::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        })
        .expect("first llama.cpp submit should succeed");
    let second_request_id = session
        .submit_generate(GenerateRequest {
            model_id: "qwen3".to_string(),
            input_tokens: vec![7, 8, 9],
            input_text: None,
            max_output_tokens: 2,
            sampling: Default::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        })
        .expect("second llama.cpp submit should succeed");

    session
        .cancel_request(first_request_id)
        .expect("llama.cpp cancel should succeed");
    let cancelled = session
        .request_report(first_request_id)
        .expect("cancelled llama.cpp request should exist");
    assert_eq!(cancelled.state, SessionRequestState::Cancelled);
    assert!(cancelled.cancel_requested);

    let first_step = session
        .step_report()
        .expect("aggregated step after cancel should succeed");
    assert_eq!(first_step.scheduled_requests, 1);
    assert_eq!(first_step.scheduled_tokens, 1);
    assert_eq!(first_step.ttft_events, 1);

    let running = session
        .request_report(second_request_id)
        .expect("remaining llama.cpp request should exist");
    assert_eq!(running.state, SessionRequestState::Running);
    assert_eq!(running.output_tokens, vec![4]);

    let second_step = session
        .step_report()
        .expect("terminal step after cancel should succeed");
    assert_eq!(second_step.scheduled_requests, 1);
    assert_eq!(second_step.scheduled_tokens, 1);

    let terminal = session
        .request_report(second_request_id)
        .expect("terminal llama.cpp request should exist");
    assert_eq!(terminal.state, SessionRequestState::Finished);
    assert_eq!(terminal.output_tokens, vec![4, 5]);

    server_handle
        .join()
        .expect("llama.cpp server thread should finish");
}

#[test]
fn native_step_report_surfaces_route_and_metal_dispatch_summary() {
    let core = EngineCore::with_runtime_components(
        KvManagerConfig::new(CacheGroupId(0), 16, 64),
        TraceReportingRunner {
            trace: sample_metal_dispatch_trace(),
        },
        DeterministicSampler,
    );
    let config = mlx_test_session_config();
    let mut session = EngineSession {
        core,
        runtime: config.runtime_report(),
        config,
        next_request_id: 2,
        native_request_routes: BTreeMap::new(),
        native_route_report_order: VecDeque::new(),
        llama_requests: BTreeMap::new(),
        llama_terminal_request_order: VecDeque::new(),
    };

    session
        .submit(sample_submission())
        .expect("submission should succeed");

    let step = session.step_report().expect("step should succeed");

    assert_eq!(step.scheduled_requests, 1);
    assert_eq!(
        step.route
            .as_ref()
            .and_then(|route| route.attention_route.as_deref()),
        Some("mock_native_attention")
    );
    assert_eq!(
        step.route
            .as_ref()
            .and_then(|route| route.crossover_decisions.get("metal_dispatch_completed")),
        Some(&1)
    );
    let request_report = session
        .request_report(1)
        .expect("native request report should exist after step");
    assert_eq!(
        request_report.route.attention_route.as_deref(),
        Some("mock_native_attention")
    );
    assert_eq!(
        request_report
            .route
            .crossover_decisions
            .get("metal_dispatch_completed"),
        Some(&1)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.runtime_required_pipeline_count),
        Some(4)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.binary_archive_state),
        Some(MetalBinaryArchiveState::Loaded)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .and_then(|dispatch| dispatch.numeric.validation.as_ref())
            .map(|validation| validation.attention_max_abs_diff_microunits),
        Some(0)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.runtime_model_conditioned_inputs),
        Some(true)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.runtime_real_model_tensor_inputs),
        Some(true)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.runtime_complete_model_forward_supported),
        Some(true)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.runtime_model_buffers_bound),
        Some(true)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.runtime_model_buffer_count),
        Some(12)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .and_then(|dispatch| dispatch.runtime_model_family.as_deref()),
        Some("qwen3")
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.execution_direct_decode_token_count),
        Some(1)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.execution_logits_output_count),
        Some(1)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.execution_remaining_logits_handle_count),
        Some(0)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.execution_model_bound_ffn_decode),
        Some(true)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.execution_real_model_forward_completed),
        Some(true)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.execution_prefix_native_dispatch_count),
        Some(35)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.execution_prefix_cpu_reference_dispatch_count),
        Some(1)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.execution_qkv_projection_token_count),
        Some(72)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.execution_layer_continuation_token_count),
        Some(37)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.execution_logits_projection_token_count),
        Some(1)
    );
    assert_eq!(
        step.metal_dispatch
            .as_ref()
            .map(|dispatch| dispatch.execution_logits_vocab_scan_row_count),
        Some(151936)
    );
}

#[test]
fn native_generate_response_surfaces_runner_route_metadata() {
    let core = EngineCore::with_runtime_components(
        KvManagerConfig::new(CacheGroupId(0), 16, 64),
        TerminalRouteReportingRunner {
            trace: sample_metal_dispatch_trace(),
        },
        DeterministicSampler,
    );
    let config = mlx_test_session_config();
    let mut session = EngineSession {
        core,
        runtime: config.runtime_report(),
        config,
        next_request_id: 2,
        native_request_routes: BTreeMap::new(),
        native_route_report_order: VecDeque::new(),
        llama_requests: BTreeMap::new(),
        llama_terminal_request_order: VecDeque::new(),
    };

    let response = session
        .generate_with_request_id(
            41,
            GenerateRequest {
                model_id: "qwen3".to_string(),
                input_tokens: vec![1, 2, 3, 4],
                input_text: None,
                max_output_tokens: 1,
                sampling: Default::default(),
                stop_sequences: Vec::new(),
                metadata: None,
            },
        )
        .expect("native blocking generate should succeed");

    assert_eq!(response.status, crate::generate::GenerateStatus::Finished);
    assert_eq!(response.step_count, 1);
    assert_eq!(
        response.route.attention_route.as_deref(),
        Some("mock_native_attention")
    );
    assert_eq!(
        response
            .route
            .crossover_decisions
            .get("metal_dispatch_completed"),
        Some(&1)
    );
}

#[test]
fn slice_output_token_logprobs_fails_closed_on_length_mismatch() {
    let mut report = sample_terminal_llama_report(41);
    report.output_token_logprobs.pop();

    let error = slice_output_token_logprobs(&report, 0, 2)
        .expect_err("mismatched logprob lengths should fail closed");

    assert!(matches!(
        error,
        EngineSessionError::RequestReportInvariantViolation { request_id: 41, .. }
    ));
}

#[test]
fn llama_cpp_stream_finish_reason_preserves_content_filter() {
    assert_eq!(
        finish_reason_from_stop_type(true, Some("content_filter")),
        Some(crate::generate::GenerateFinishReason::ContentFilter)
    );
    assert_eq!(
        finish_reason_from_stop_type(true, Some("backend_error")),
        Some(crate::generate::GenerateFinishReason::Error)
    );
    assert_eq!(
        terminal_stop_reason_from_finish_reason(Some(
            crate::generate::GenerateFinishReason::ContentFilter
        )),
        Some(ax_engine_core::StopReason::Error)
    );
}

#[test]
fn llama_cpp_terminal_requests_are_pruned_after_retention_limit() {
    let mut session = llama_cpp_server_session("http://127.0.0.1:1".to_string());

    for request_id in 1..=(MAX_LLAMA_CPP_TERMINAL_REQUESTS as u64 + 1) {
        session.store_terminal_llama_report(request_id, sample_terminal_llama_report(request_id));
    }

    assert!(session.request_report(1).is_none());
    assert!(
        session
            .request_report(MAX_LLAMA_CPP_TERMINAL_REQUESTS as u64 + 1)
            .is_some()
    );
    assert_eq!(
        session.llama_terminal_request_order.len(),
        MAX_LLAMA_CPP_TERMINAL_REQUESTS
    );
}
