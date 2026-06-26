use super::*;
use crate::commands::*;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

fn make_route_metadata(
    execution_plan: &str,
    attention_route: &str,
    kv_mode: &str,
    barrier_mode: &str,
) -> RouteMetadata {
    RouteMetadata {
        execution_plan: Some(execution_plan.into()),
        attention_route: Some(attention_route.into()),
        kv_mode: Some(kv_mode.into()),
        prefix_cache_path: Some("metadata_lookup".into()),
        barrier_mode: Some(barrier_mode.into()),
        crossover_decisions: Vec::new(),
    }
}

fn sample_metal_dispatch_trace() -> MetalDispatchTrace {
    MetalDispatchTrace {
        command_queue_label: "ax.queue".to_string(),
        command_buffer_label: "ax.buffer".to_string(),
        command_buffer_status: ax_engine_core::MetalCommandBufferStatus::Completed,
        runtime: ax_engine_core::metal::MetalDispatchRuntimeInfo {
            device_name: "Apple M4 Max".to_string(),
            required_pipeline_count: 4,
            max_thread_execution_width: 64,
            binary_archive: ax_engine_core::MetalBinaryArchiveInfo {
                path: PathBuf::from("/tmp/ax_phase1_dense_path.binary_archive.metallib"),
                state: ax_engine_core::MetalBinaryArchiveState::Loaded,
                attached_pipeline_count: 4,
                serialized: true,
                note: None,
            },
            command_queue_ready: true,
            model_conditioned_inputs: false,
            real_model_tensor_inputs: false,
            complete_model_forward_supported: false,
            model_bindings_prepared: false,
            model_buffers_bound: false,
            model_buffer_count: 0,
            model_buffer_bytes: 0,
            native_dense_kernel_coverage:
                ax_engine_core::metal::MetalNativeDenseKernelCoverage::default(),
            model: None,
        },
        workload: ax_engine_core::MetalDispatchWorkload {
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
            kv_metadata: ax_engine_core::metal::MetalDispatchKvMetadata {
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
        arena: ax_engine_core::MetalDispatchArenaInfo {
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
        execution: ax_engine_core::metal::MetalDispatchExecutionInfo {
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
            prefix_native_projection_row_count: 4096,
            prefix_cpu_projection_row_count: 512,
            prefix_native_rms_norm_element_count: 2048,
            prefix_cpu_rms_norm_element_count: 128,
            prefix_native_ffn_activation_element_count: 1024,
            prefix_cpu_ffn_activation_element_count: 64,
            prefix_native_residual_add_element_count: 4096,
            prefix_cpu_residual_add_element_count: 0,
            prefix_native_scale_element_count: 2048,
            prefix_cpu_scale_element_count: 0,
            direct_decode_native_projection_row_count: 151936,
            direct_decode_cpu_projection_row_count: 3072,
            direct_decode_native_rms_norm_element_count: 6144,
            direct_decode_cpu_rms_norm_element_count: 0,
            direct_decode_native_ffn_activation_element_count: 1536,
            direct_decode_cpu_ffn_activation_element_count: 0,
            direct_decode_native_residual_add_element_count: 3072,
            direct_decode_cpu_residual_add_element_count: 0,
            direct_decode_native_scale_element_count: 1536,
            direct_decode_cpu_scale_element_count: 0,
            direct_decode_batched_logits_group_count: 1,
            direct_decode_batched_logits_token_count: 2,
            direct_decode_batched_group_fallback_count: 0,
            direct_decode_batched_group_fallback_token_count: 0,
        },
        kernels: vec![
            ax_engine_core::MetalDispatchKernelTrace {
                function_name: "reshape_and_cache".to_string(),
                element_count: 32,
                threads_per_grid: ax_engine_core::MetalThreadgroupSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
                threads_per_threadgroup: ax_engine_core::MetalThreadgroupSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            },
            ax_engine_core::MetalDispatchKernelTrace {
                function_name: "paged_decode_attention".to_string(),
                element_count: 16,
                threads_per_grid: ax_engine_core::MetalThreadgroupSize {
                    width: 16,
                    height: 1,
                    depth: 1,
                },
                threads_per_threadgroup: ax_engine_core::MetalThreadgroupSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            },
        ],
        numeric: ax_engine_core::metal::MetalDispatchNumericTrace {
            attention_output_bits: vec![0],
            key_cache_checksum: 0x11,
            attention_output_checksum: 0x22,
            gather_output_checksum: 0x33,
            copy_output_checksum: 0x44,
            validation: Some(ax_engine_core::metal::MetalNumericValidationSummary {
                expected_key_cache_checksum: 0x11,
                expected_attention_output_checksum: 0x22,
                expected_gather_output_checksum: 0x33,
                expected_copy_output_checksum: 0x44,
                attention_max_abs_diff_microunits: 0,
            }),
        },
    }
}

#[test]
fn step_trace_json_includes_metal_dispatch_summary() {
    let entry = StepTraceEntry {
        t_ms: 12,
        step_id: Some(StepId(3)),
        admitted_request_ids: vec![RequestId(1)],
        selected_request_ids: vec![RequestId(1)],
        deferred_request_ids: Vec::new(),
        memory_blocked_request_ids: Vec::new(),
        cleanup_request_ids: Vec::new(),
        scheduled_tokens: 2,
        prefix_hits: 0,
        cpu_time_us: 123,
        runner_time_us: 45,
        kv_usage_blocks: 1,
        evictions: 2,
        preempted_requests: 0,
        preempted_tokens: 0,
        runner_executed: true,
        route_metadata: RouteMetadata::empty(),
        metal_dispatch: Some(MetalDispatchStepTrace::from_trace(
            &sample_metal_dispatch_trace(),
        )),
        items: Vec::new(),
    };

    let json = entry.json();

    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("command_buffer_status"))
            .and_then(Value::as_str),
        Some("completed")
    );
    assert_eq!(json.get("evictions").and_then(Value::as_u64), Some(2));
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("runtime_required_pipeline_count"))
            .and_then(Value::as_u64),
        Some(4)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("runtime_complete_model_forward_supported"))
            .and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("runtime_model_buffers_bound"))
            .and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_direct_decode_token_count"))
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_real_model_forward_completed"))
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_prefix_native_dispatch_count"))
            .and_then(Value::as_u64),
        Some(35)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_prefix_cpu_reference_dispatch_count"))
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_qkv_projection_token_count"))
            .and_then(Value::as_u64),
        Some(72)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_layer_continuation_token_count"))
            .and_then(Value::as_u64),
        Some(37)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_logits_projection_token_count"))
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_logits_vocab_scan_row_count"))
            .and_then(Value::as_u64),
        Some(151936)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_prefix_native_projection_row_count"))
            .and_then(Value::as_u64),
        Some(4096)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_prefix_cpu_projection_row_count"))
            .and_then(Value::as_u64),
        Some(512)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_prefix_native_rms_norm_element_count"))
            .and_then(Value::as_u64),
        Some(2048)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_prefix_cpu_rms_norm_element_count"))
            .and_then(Value::as_u64),
        Some(128)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_prefix_native_residual_add_element_count"))
            .and_then(Value::as_u64),
        Some(4096)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_prefix_cpu_residual_add_element_count"))
            .and_then(Value::as_u64),
        Some(0)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_prefix_native_scale_element_count"))
            .and_then(Value::as_u64),
        Some(2048)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_prefix_cpu_scale_element_count"))
            .and_then(Value::as_u64),
        Some(0)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_direct_decode_native_projection_row_count"))
            .and_then(Value::as_u64),
        Some(151936)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_direct_decode_cpu_projection_row_count"))
            .and_then(Value::as_u64),
        Some(3072)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| {
                trace.get("execution_direct_decode_native_rms_norm_element_count")
            })
            .and_then(Value::as_u64),
        Some(6144)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_direct_decode_cpu_rms_norm_element_count"))
            .and_then(Value::as_u64),
        Some(0)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| {
                trace.get("execution_direct_decode_native_residual_add_element_count")
            })
            .and_then(Value::as_u64),
        Some(3072)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| {
                trace.get("execution_direct_decode_cpu_residual_add_element_count")
            })
            .and_then(Value::as_u64),
        Some(0)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| { trace.get("execution_direct_decode_native_scale_element_count") })
            .and_then(Value::as_u64),
        Some(1536)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| { trace.get("execution_direct_decode_cpu_scale_element_count") })
            .and_then(Value::as_u64),
        Some(0)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_direct_decode_batched_logits_group_count"))
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("execution_direct_decode_batched_logits_token_count"))
            .and_then(Value::as_u64),
        Some(2)
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("binary_archive_state"))
            .and_then(Value::as_str),
        Some("loaded")
    );
    assert_eq!(
        json.get("metal_dispatch")
            .and_then(|trace| trace.get("numeric"))
            .and_then(|numeric| numeric.get("validation"))
            .and_then(|validation| validation.get("attention_max_abs_diff_microunits"))
            .and_then(Value::as_u64),
        Some(0)
    );
}

#[test]
fn metrics_json_surfaces_observed_evictions() {
    let execution = RuntimeResult {
        tool_mode: "mlx_runtime",
        runtime: RuntimeConfig {
            deterministic: true,
            max_batch_tokens: 8,
            block_size_tokens: 4,
            kv_total_blocks: Some(2),
            flags: RuntimeFlags::default(),
            llama_cpp_preset: None,
            backend_policy: BackendPolicy::new(ResolutionPolicy::MlxOnly),
            resolved_backend: ResolvedBackend::new(
                SelectedBackend::Mlx,
                SupportTier::MlxPreview,
                None,
            ),
            backend_adapter: None,
            mlx_model_artifacts_dir: None,
            mlx_model_artifacts_source: None,
        },
        observation: RuntimeObservation {
            evictions: 7,
            ..RuntimeObservation::default()
        },
        correctness: GateStatus::pass(),
        determinism: GateStatus::pass(),
    };

    let metrics = build_metrics_json("run-evictions", &execution);

    assert_eq!(
        nested_value(&metrics, &["metrics", "evictions"]).and_then(Value::as_f64),
        Some(7.0)
    );
    assert!(
        nested_value(&metrics, &["metrics", "delegated_llama_cpp"]).is_none(),
        "MLX metrics must not publish delegated llama.cpp counters"
    );
}

#[test]
fn metrics_json_nests_delegated_llama_cpp_metrics_under_metrics() {
    let mut route = RouteMetadata::empty();
    route.prefix_cache_path = Some("delegated_prompt_cache".to_string());
    route
        .crossover_decisions
        .push(("delegated_cached_tokens".to_string(), 64));
    let execution = RuntimeResult {
        tool_mode: "llama_cpp_stepwise_runtime",
        runtime: runtime_config_from_manifest(&llama_cpp_scenario_manifest(
            "http://127.0.0.1:8081",
        ))
        .expect("llama.cpp runtime should load"),
        observation: RuntimeObservation {
            kv_peak_blocks: 3,
            llama_cpp_processing_request_events: 2,
            llama_cpp_deferred_request_events: 1,
            route_metadata: route,
            ..RuntimeObservation::default()
        },
        correctness: GateStatus::pass(),
        determinism: GateStatus::pass(),
    };

    let metrics = build_metrics_json("run-llama", &execution);
    let delegated = nested_value(&metrics, &["metrics", "delegated_llama_cpp"])
        .expect("delegated llama.cpp metrics should live under metrics");

    assert!(
        metrics.get("delegated_llama_cpp").is_none(),
        "delegated metrics must not drift to the root artifact object"
    );
    assert_eq!(
        delegated
            .get("backend_reported_cached_prompt_tokens")
            .and_then(Value::as_u64),
        Some(64)
    );
    assert_eq!(
        delegated
            .get("requests_processing_events")
            .and_then(Value::as_u64),
        Some(2)
    );
    assert_eq!(
        delegated
            .get("requests_deferred_events")
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        delegated
            .get("cache_reuse_observed")
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        nested_value(&metrics, &["memory_peak_estimate", "kind"]).and_then(Value::as_str),
        Some("kv_cache_legacy_estimate")
    );
    assert_eq!(
        nested_value(
            &metrics,
            &["memory_peak_estimate", "estimated_bytes_per_token"]
        )
        .and_then(Value::as_f64),
        Some(LEGACY_KV_CACHE_ESTIMATED_BYTES_PER_TOKEN)
    );
    assert_eq!(
        nested_value(&metrics, &["token_accounting", "contains_estimates"])
            .and_then(Value::as_bool),
        Some(false)
    );
}

#[test]
fn token_accounting_json_marks_synthetic_estimates() {
    let mut observation = RuntimeObservation::default();
    observation.record_token_accounting_source("prompt", "backend_reported_usage");
    observation.record_token_accounting_source("output", "synthetic_text_estimate");

    let accounting = token_accounting_json(&observation);

    assert_eq!(
        nested_value(&accounting, &["sources", "prompt.backend_reported_usage"])
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        nested_value(&accounting, &["sources", "output.synthetic_text_estimate"])
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        accounting
            .get("contains_estimates")
            .and_then(Value::as_bool),
        Some(true)
    );
}

fn find_repo_root_from(start: &Path) -> Option<PathBuf> {
    start.ancestors().find_map(|ancestor| {
        (ancestor.join("Cargo.toml").is_file() && ancestor.join("benchmarks/manifests").is_dir())
            .then(|| ancestor.to_path_buf())
    })
}

fn repo_root_path() -> PathBuf {
    if let Ok(cwd) = std::env::current_dir()
        && let Some(root) = find_repo_root_from(&cwd)
    {
        return root;
    }

    find_repo_root_from(Path::new(env!("CARGO_MANIFEST_DIR")))
        .expect("workspace root should contain Cargo.toml and benchmarks/manifests")
}

fn repo_manifest_path(relative: &str) -> String {
    repo_root_path()
        .join(relative)
        .to_string_lossy()
        .into_owned()
}

fn load_test_manifest(path: &str) -> BenchmarkManifest {
    let mut manifest = load_manifest(Path::new(path)).expect("manifest should load");
    if manifest.runtime.selected_backend == SelectedBackend::Mlx {
        manifest.runtime.deterministic = false;
        manifest.runtime.mlx_model_artifacts_dir = Some(write_valid_native_model_fixture());
    }
    manifest
}

fn load_test_manifest_json(path: &str) -> Value {
    load_json_value(Path::new(path)).expect("manifest JSON should load")
}

fn unique_test_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ax-engine-bench-{label}-{}-{nanos}",
        std::process::id()
    ))
}

fn test_request_spec(request_id: u64, max_output_tokens: u32) -> SyntheticRequestSpec {
    SyntheticRequestSpec {
        external_id: format!("req-{request_id}"),
        request_id: RequestId(request_id),
        arrival_sequence: SequenceNo(request_id),
        model_family: "qwen3_5".to_string(),
        prompt_token_target: 8,
        input_tokens: vec![1, 2, 3, 4],
        input_text: None,
        max_output_tokens,
        sampling_params: SamplingParams::default(),
        metadata: None,
    }
}

#[test]
fn workload_step_guard_uses_longest_single_request_budget() {
    let specs = (1..=8)
        .map(|request_id| test_request_spec(request_id, 512))
        .collect::<Vec<_>>();

    assert_eq!(
        workload_step_guard_from_specs(&specs),
        512 + WORKLOAD_STEP_GUARD_SLACK
    );
}

#[test]
fn replay_step_guard_preserves_timeline_offset_without_summing_batch_budget() {
    let events = vec![
        ReplayEvent::Submit {
            t_ms: 10,
            spec: test_request_spec(1, 128),
        },
        ReplayEvent::Submit {
            t_ms: 10,
            spec: test_request_spec(2, 512),
        },
        ReplayEvent::Cancel {
            t_ms: 100,
            request_id: RequestId(1),
        },
    ];

    assert_eq!(
        replay_step_guard(&events),
        100 + 512 + WORKLOAD_STEP_GUARD_SLACK
    );
}

#[test]
fn create_unique_result_dir_does_not_reuse_same_component_directory() {
    let root = unique_test_dir("unique-result-dir");
    fs::create_dir_all(&root).expect("result root should create");

    let (first_id, first_dir) =
        create_unique_result_dir(&root, None, "same-manifest").expect("first dir");
    let (second_id, second_dir) =
        create_unique_result_dir(&root, None, "same-manifest").expect("second dir");

    assert_ne!(first_id, second_id);
    assert_ne!(first_dir, second_dir);
    assert!(first_dir.is_dir());
    assert!(second_dir.is_dir());

    fs::remove_dir_all(root).expect("test directory should clean up");
}

#[test]
fn unique_run_suffix_uses_twelve_hex_digits() {
    let suffix = unique_run_suffix(0).expect("suffix should generate");

    assert_eq!(suffix.len(), 12);
    assert!(
        suffix
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    );
}

#[test]
fn write_execution_artifacts_can_skip_trace_file() {
    let root = unique_test_dir("no-trace-artifacts");
    fs::create_dir_all(&root).expect("result root should create");
    let manifest_path = PathBuf::from(repo_manifest_path(
        "benchmarks/manifests/scenario/chat_qwen_short.json",
    ));
    let manifest = load_manifest(&manifest_path).expect("test manifest should load");
    let runtime = runtime_config_from_manifest(&manifest).expect("runtime config should build");
    let execution = RuntimeResult {
        tool_mode: runtime.tool_mode(),
        runtime,
        observation: RuntimeObservation::default(),
        correctness: GateStatus::pass(),
        determinism: GateStatus::pass(),
    };

    let result_dir = write_execution_artifacts(
        "scenario",
        &manifest_path,
        &manifest,
        &root,
        123,
        &execution,
        false,
    )
    .expect("execution artifacts should write");

    assert!(result_dir.join("metrics.json").is_file());
    assert!(result_dir.join("routes.json").is_file());
    assert!(
        !result_dir.join("trace.json").exists(),
        "--no-trace should skip trace artifact materialization"
    );

    fs::remove_dir_all(root).expect("test directory should clean up");
}

#[test]
fn generate_manifest_args_accept_json_flag_after_model_dir() {
    let args = parse_generate_manifest_args(&["/tmp/model".to_string(), "--json".to_string()])
        .expect("generate-manifest args should parse");

    assert_eq!(args.model_dir, PathBuf::from("/tmp/model"));
    assert!(args.json);
    assert!(!args.validate);
}

#[test]
fn generate_manifest_args_accept_json_flag_before_model_dir() {
    let args = parse_generate_manifest_args(&["--json".to_string(), "/tmp/model".to_string()])
        .expect("generate-manifest args should parse");

    assert_eq!(args.model_dir, PathBuf::from("/tmp/model"));
    assert!(args.json);
    assert!(!args.validate);
}

#[test]
fn generate_manifest_args_accept_validate_flag() {
    let args = parse_generate_manifest_args(&[
        "--validate".to_string(),
        "/tmp/model".to_string(),
        "--json".to_string(),
    ])
    .expect("generate-manifest args should parse");

    assert_eq!(args.model_dir, PathBuf::from("/tmp/model"));
    assert!(args.json);
    assert!(args.validate);
}

#[test]
fn generate_manifest_args_reject_missing_model_dir() {
    let error = parse_generate_manifest_args(&["--json".to_string()])
        .expect_err("missing model directory should fail");

    assert!(
        matches!(error, CliError::Usage(message) if message.contains("generate-manifest <model-dir> [--json] [--validate]"))
    );
}

#[test]
fn generate_manifest_summary_uses_stable_json_contract() {
    let summary = GenerateManifestSummary {
        schema_version: GENERATE_MANIFEST_SCHEMA_VERSION,
        model_dir: "/tmp/model".to_string(),
        manifest_path: "/tmp/model/model-manifest.json".to_string(),
        status: GenerateManifestStatus::AlreadyExists,
        manifest_present: true,
        validation: Some(GenerateManifestValidationSummary { passed: true }),
    };

    let value = serde_json::to_value(summary).expect("generate-manifest summary should serialize");

    assert_eq!(
        value.get("schema_version").and_then(Value::as_str),
        Some("ax.generate_manifest.v1")
    );
    assert_eq!(
        value.get("status").and_then(Value::as_str),
        Some("already_exists")
    );
    assert_eq!(
        value.get("manifest_present").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        nested_value(&value, &["validation", "passed"]).and_then(Value::as_bool),
        Some(true)
    );
}

#[test]
fn benchmark_artifact_summary_uses_stable_json_contract() {
    let summary = BenchmarkArtifactSummary::manifest_run(
        "scenario",
        Path::new("/tmp/manifest.json"),
        Path::new("/tmp/results"),
        Path::new("/tmp/results/scenario-1"),
        "pass",
    );

    let value = serde_json::to_value(summary).expect("benchmark artifact summary should serialize");

    assert_eq!(
        value.get("schema_version").and_then(Value::as_str),
        Some("ax.benchmark_artifact.v1")
    );
    assert_eq!(
        value.get("command").and_then(Value::as_str),
        Some("scenario")
    );
    assert_eq!(
        value.get("result_dir").and_then(Value::as_str),
        Some("/tmp/results/scenario-1")
    );
    assert_eq!(value.get("status").and_then(Value::as_str), Some("pass"));
}

#[test]
fn benchmark_artifact_text_keeps_legacy_result_dir_shape() {
    let summary = BenchmarkArtifactSummary::comparison(
        "compare",
        Path::new("/tmp/baseline"),
        Path::new("/tmp/candidate"),
        Path::new("/tmp/results"),
        Path::new("/tmp/results/compare-1"),
    );

    let text = render_benchmark_artifact_summary(&summary);

    assert!(text.contains("ax-engine-bench compare"));
    assert!(text.contains("baseline=/tmp/baseline"));
    assert!(text.contains("candidate=/tmp/candidate"));
    assert!(text.contains("result_dir=/tmp/results/compare-1"));
    assert!(
        !text.contains("status=written"),
        "legacy successful compare text did not include status"
    );
}

fn write_doctor_model_manifest(model_dir: &Path) {
    fs::write(model_dir.join("model-manifest.json"), "{}").expect("model manifest should write");
}

fn write_doctor_safetensors(model_dir: &Path) {
    fs::write(model_dir.join("model.safetensors"), b"placeholder")
        .expect("safetensors marker should write");
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn compiled_repo_metal_build_dir() -> Option<PathBuf> {
    let build_dir = repo_root_path().join("build/metal");
    let build_report = load_json_value(&build_dir.join("build_report.json")).ok()?;
    (build_report.get("status").and_then(Value::as_str) == Some("compiled")).then_some(build_dir)
}

fn native_model_tensor(
    name: &str,
    role: ax_engine_core::NativeTensorRole,
    layer_index: Option<u32>,
    shape: Vec<u64>,
) -> ax_engine_core::NativeTensorSpec {
    ax_engine_core::NativeTensorSpec {
        name: name.to_string(),
        role,
        layer_index,
        dtype: ax_engine_core::NativeTensorDataType::F16,
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

fn write_valid_native_model_fixture() -> PathBuf {
    let root_dir = unique_test_dir("native-model-runtime-metadata");
    fs::create_dir_all(&root_dir).expect("native model fixture directory should create");
    fs::write(root_dir.join("model.safetensors"), vec![0_u8; 4096])
        .expect("native model weights should write");

    let manifest = ax_engine_core::NativeModelManifest {
        schema_version: ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: "qwen3".to_string(),
        tensor_format: ax_engine_core::NativeTensorFormat::Safetensors,
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
        linear_attention: ax_engine_core::NativeLinearAttentionConfig::default(),
        mla_attention: Default::default(),
        moe: ax_engine_core::NativeMoeConfig::default(),
        glm_router: Default::default(),
        weight_sanitize: ax_engine_core::WeightSanitize::None,
        think_start_token_id: None,
        think_end_token_id: None,
        diffusion: ax_engine_core::NativeDiffusionConfig::default(),
        tensors: vec![
            native_model_tensor(
                "model.embed_tokens.weight",
                ax_engine_core::NativeTensorRole::TokenEmbedding,
                None,
                vec![151_936, 2_048],
            ),
            native_model_tensor(
                "model.norm.weight",
                ax_engine_core::NativeTensorRole::FinalNorm,
                None,
                vec![2_048],
            ),
            native_model_tensor(
                "lm_head.weight",
                ax_engine_core::NativeTensorRole::LmHead,
                None,
                vec![151_936, 2_048],
            ),
            native_model_tensor(
                "model.layers.0.input_layernorm.weight",
                ax_engine_core::NativeTensorRole::AttentionNorm,
                Some(0),
                vec![2_048],
            ),
            native_model_tensor(
                "model.layers.0.self_attn.qkv_proj.weight",
                ax_engine_core::NativeTensorRole::AttentionQkvPacked,
                Some(0),
                vec![4_096, 2_048],
            ),
            native_model_tensor(
                "model.layers.0.self_attn.o_proj.weight",
                ax_engine_core::NativeTensorRole::AttentionO,
                Some(0),
                vec![2_048, 2_048],
            ),
            native_model_tensor(
                "model.layers.0.post_attention_layernorm.weight",
                ax_engine_core::NativeTensorRole::FfnNorm,
                Some(0),
                vec![2_048],
            ),
            native_model_tensor(
                "model.layers.0.mlp.gate_up_proj.weight",
                ax_engine_core::NativeTensorRole::FfnGateUpPacked,
                Some(0),
                vec![8_192, 2_048],
            ),
            native_model_tensor(
                "model.layers.0.mlp.down_proj.weight",
                ax_engine_core::NativeTensorRole::FfnDown,
                Some(0),
                vec![2_048, 4_096],
            ),
        ],
    };

    fs::write(
        root_dir.join(ax_engine_core::AX_NATIVE_MODEL_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).expect("native model manifest should serialize"),
    )
    .expect("native model manifest should write");

    root_dir
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn native_model_tensor_with_file(
    name: &str,
    role: ax_engine_core::NativeTensorRole,
    layer_index: Option<u32>,
    shape: &[u64],
    file: &str,
    length_bytes: u64,
) -> ax_engine_core::NativeTensorSpec {
    ax_engine_core::NativeTensorSpec {
        name: name.to_string(),
        role,
        layer_index,
        dtype: ax_engine_core::NativeTensorDataType::F32,
        source_tensor_type: None,
        source_quantized: false,
        quantization: None,
        quantized_source: None,
        shape: shape.to_vec(),
        file: PathBuf::from(file),
        offset_bytes: 0,
        length_bytes,
    }
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn write_f32_tensor_file(root_dir: &Path, file_name: &str, values: &[f32]) {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(root_dir.join(file_name), bytes).expect("tensor bytes should write");
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn write_projection_native_model_fixture() -> PathBuf {
    let root_dir = unique_test_dir("native-model-projection");
    fs::create_dir_all(&root_dir).expect("projection fixture directory should create");

    let embedding = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
        3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, //
        4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
    ];
    let ones = vec![1.0_f32; 8];
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let double_identity = identity.iter().map(|value| value * 2.0).collect::<Vec<_>>();
    let triple_identity = identity.iter().map(|value| value * 3.0).collect::<Vec<_>>();
    let zero_matrix = vec![0.0_f32; 64];

    write_f32_tensor_file(&root_dir, "embed.bin", &embedding);
    write_f32_tensor_file(&root_dir, "final_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "lm_head.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "attn_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "attn_q.bin", &identity);
    write_f32_tensor_file(&root_dir, "attn_k.bin", &double_identity);
    write_f32_tensor_file(&root_dir, "attn_v.bin", &triple_identity);
    write_f32_tensor_file(&root_dir, "attn_o.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "ffn_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "ffn_gate.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "ffn_up.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "ffn_down.bin", &zero_matrix);

    let matrix_bytes = (64 * std::mem::size_of::<f32>()) as u64;
    let vector_bytes = (8 * std::mem::size_of::<f32>()) as u64;
    let embedding_bytes = (embedding.len() * std::mem::size_of::<f32>()) as u64;
    let manifest = ax_engine_core::NativeModelManifest {
        schema_version: ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: "qwen3".to_string(),
        tensor_format: ax_engine_core::NativeTensorFormat::Safetensors,
        source_quantization: None,
        runtime_status: ax_engine_core::model::NativeRuntimeStatus::default(),
        layer_count: 1,
        hidden_size: 8,
        intermediate_size: 24,
        attention_head_count: 2,
        attention_head_dim: 4,
        kv_head_count: 2,
        vocab_size: 5,
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
        linear_attention: ax_engine_core::NativeLinearAttentionConfig::default(),
        mla_attention: Default::default(),
        moe: ax_engine_core::NativeMoeConfig::default(),
        glm_router: Default::default(),
        weight_sanitize: ax_engine_core::WeightSanitize::None,
        think_start_token_id: None,
        think_end_token_id: None,
        diffusion: ax_engine_core::NativeDiffusionConfig::default(),
        tensors: vec![
            native_model_tensor_with_file(
                "model.embed_tokens.weight",
                ax_engine_core::NativeTensorRole::TokenEmbedding,
                None,
                &[5, 8],
                "embed.bin",
                embedding_bytes,
            ),
            native_model_tensor_with_file(
                "model.norm.weight",
                ax_engine_core::NativeTensorRole::FinalNorm,
                None,
                &[8],
                "final_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "lm_head.weight",
                ax_engine_core::NativeTensorRole::LmHead,
                None,
                &[5, 8],
                "lm_head.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.input_layernorm.weight",
                ax_engine_core::NativeTensorRole::AttentionNorm,
                Some(0),
                &[8],
                "attn_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.q_proj.weight",
                ax_engine_core::NativeTensorRole::AttentionQ,
                Some(0),
                &[8, 8],
                "attn_q.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.k_proj.weight",
                ax_engine_core::NativeTensorRole::AttentionK,
                Some(0),
                &[8, 8],
                "attn_k.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.v_proj.weight",
                ax_engine_core::NativeTensorRole::AttentionV,
                Some(0),
                &[8, 8],
                "attn_v.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.o_proj.weight",
                ax_engine_core::NativeTensorRole::AttentionO,
                Some(0),
                &[8, 8],
                "attn_o.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.post_attention_layernorm.weight",
                ax_engine_core::NativeTensorRole::FfnNorm,
                Some(0),
                &[8],
                "ffn_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.gate_proj.weight",
                ax_engine_core::NativeTensorRole::FfnGate,
                Some(0),
                &[8, 8],
                "ffn_gate.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.up_proj.weight",
                ax_engine_core::NativeTensorRole::FfnUp,
                Some(0),
                &[8, 8],
                "ffn_up.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.down_proj.weight",
                ax_engine_core::NativeTensorRole::FfnDown,
                Some(0),
                &[8, 8],
                "ffn_down.bin",
                matrix_bytes,
            ),
        ],
    };

    fs::write(
        root_dir.join(ax_engine_core::AX_NATIVE_MODEL_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest)
            .expect("projection native model manifest should serialize"),
    )
    .expect("projection native model manifest should write");

    root_dir
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn write_multilayer_direct_decode_native_model_fixture() -> PathBuf {
    let root_dir = write_projection_native_model_fixture();
    let zero_matrix = vec![0.0_f32; 64];
    let mut lm_head = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
        3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, //
        4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
    ];
    lm_head.resize(64, 0.0);

    write_f32_tensor_file(&root_dir, "attn_q_l1.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "attn_k_l1.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "attn_v_l1.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "lm_head.bin", &lm_head);

    let manifest_path = root_dir.join(ax_engine_core::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("manifest should read");
    let mut manifest =
        serde_json::from_slice::<ax_engine_core::NativeModelManifest>(&manifest_bytes)
            .expect("manifest should parse");
    let matrix_bytes = (64 * std::mem::size_of::<f32>()) as u64;
    let vector_bytes = (8 * std::mem::size_of::<f32>()) as u64;
    manifest.layer_count = 2;
    manifest.tensors.extend([
        native_model_tensor_with_file(
            "model.layers.1.input_layernorm.weight",
            ax_engine_core::NativeTensorRole::AttentionNorm,
            Some(1),
            &[8],
            "attn_norm.bin",
            vector_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.self_attn.q_proj.weight",
            ax_engine_core::NativeTensorRole::AttentionQ,
            Some(1),
            &[8, 8],
            "attn_q_l1.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.self_attn.k_proj.weight",
            ax_engine_core::NativeTensorRole::AttentionK,
            Some(1),
            &[8, 8],
            "attn_k_l1.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.self_attn.v_proj.weight",
            ax_engine_core::NativeTensorRole::AttentionV,
            Some(1),
            &[8, 8],
            "attn_v_l1.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.self_attn.o_proj.weight",
            ax_engine_core::NativeTensorRole::AttentionO,
            Some(1),
            &[8, 8],
            "attn_o.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.post_attention_layernorm.weight",
            ax_engine_core::NativeTensorRole::FfnNorm,
            Some(1),
            &[8],
            "ffn_norm.bin",
            vector_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.mlp.gate_proj.weight",
            ax_engine_core::NativeTensorRole::FfnGate,
            Some(1),
            &[8, 8],
            "ffn_gate.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.mlp.up_proj.weight",
            ax_engine_core::NativeTensorRole::FfnUp,
            Some(1),
            &[8, 8],
            "ffn_up.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.mlp.down_proj.weight",
            ax_engine_core::NativeTensorRole::FfnDown,
            Some(1),
            &[8, 8],
            "ffn_down.bin",
            matrix_bytes,
        ),
    ]);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
    )
    .expect("manifest should write");

    root_dir
}

fn native_environment_fixture() -> Value {
    json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "device": "Mac17,6",
            "hostname": "fixture-host",
            "system_model": "Mac17,6",
            "soc": "Apple M4 Max",
            "memory_gb": 128,
            "memory_bytes": 137438953472_u64,
            "logical_cpu_count": 16
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "mlx_runtime",
            "ax_engine_bench_version": "0.1.0"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    })
}

fn mlx_runtime_fixture() -> Value {
    json!({
        "selected_backend": "mlx",
        "support_tier": "mlx_preview",
        "resolution_policy": "mlx_only",
        "mlx_runtime": {
            "runner": "metal_bringup",
            "artifacts_source": "explicit_config"
        },
        "host": runtime_host_fixture(),
        "metal_toolchain": runtime_metal_toolchain_fixture()
    })
}

#[test]
fn mlx_runtime_tool_mode_inference_recognizes_metal_bringup() {
    let runtime = mlx_runtime_fixture();
    assert_eq!(
        inferred_tool_mode_from_runtime_json(&runtime),
        Some("engine_bringup_runtime")
    );
}

#[test]
fn compare_result_label_distinguishes_metal_execution_semantics() {
    assert_eq!(
        compare_result_label("engine_bringup_runtime", "metal_numeric_scaffold_only"),
        "metal_numeric_scaffold_compare"
    );
    assert_eq!(
        compare_result_label(
            "engine_bringup_runtime",
            "metal_model_conditioned_numeric_scaffold"
        ),
        "metal_model_conditioned_scaffold_compare"
    );
    assert_eq!(
        compare_result_label("engine_bringup_runtime", "metal_real_model_forward"),
        "metal_real_model_forward_compare"
    );
    assert_eq!(
        compare_result_label(
            "engine_bringup_runtime",
            "metal_multilayer_mixed_prefix_attention"
        ),
        "metal_multilayer_mixed_prefix_attention_compare"
    );
    assert_eq!(
        compare_result_label(
            "engine_bringup_runtime",
            "metal_multilayer_native_prefix_attention"
        ),
        "metal_multilayer_native_prefix_attention_compare"
    );
    assert_eq!(
        compare_result_label(
            "engine_bringup_runtime",
            "metal_multilayer_model_incomplete"
        ),
        "metal_multilayer_model_incomplete_compare"
    );
    assert_eq!(
        compare_result_label("engine_bringup_runtime", "metal_model_bound_ffn_decode"),
        "metal_model_bound_ffn_decode_compare"
    );
    assert_eq!(
        compare_result_label("mlx_runtime", "metal_real_model_forward"),
        "metal_real_model_forward_compare"
    );
}

#[test]
fn route_execution_semantics_marks_model_conditioned_numeric_scaffold_without_runtime_tensors() {
    let mut route = RouteMetadata::empty();
    route
        .crossover_decisions
        .push(("metal_dispatch_numeric_scaffold_only".to_string(), 1));
    route
        .crossover_decisions
        .push(("metal_dispatch_model_conditioned_inputs".to_string(), 1));

    assert_eq!(
        route_execution_semantics_label(&route),
        "metal_model_conditioned_numeric_scaffold"
    );
}

#[test]
fn route_execution_semantics_prefers_runtime_real_model_tensor_inputs() {
    let mut route = RouteMetadata::empty();
    route
        .crossover_decisions
        .push(("metal_dispatch_model_conditioned_inputs".to_string(), 1));
    route.crossover_decisions.push((
        "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
        1,
    ));

    assert_eq!(
        route_execution_semantics_label(&route),
        "metal_real_model_tensor_inputs"
    );
}

#[test]
fn route_execution_semantics_marks_multilayer_model_incomplete() {
    let mut route = RouteMetadata::empty();
    route.crossover_decisions.push((
        "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
        1,
    ));
    route
        .crossover_decisions
        .push(("metal_dispatch_model_layer_count".to_string(), 36));

    assert_eq!(
        route_execution_semantics_label(&route),
        "metal_multilayer_model_incomplete"
    );
}

#[test]
fn route_execution_semantics_keeps_multilayer_incomplete_without_real_forward_marker() {
    let mut route = RouteMetadata::empty();
    route.crossover_decisions.push((
        "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
        1,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_runtime_complete_model_forward_supported".to_string(),
        1,
    ));
    route
        .crossover_decisions
        .push(("metal_dispatch_model_layer_count".to_string(), 36));

    assert_eq!(
        route_execution_semantics_label(&route),
        "metal_multilayer_model_incomplete"
    );
}

#[test]
fn route_execution_semantics_marks_multilayer_native_prefix_attention() {
    let mut route = RouteMetadata::empty();
    route.crossover_decisions.push((
        "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
        1,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_layers_native_attention".to_string(),
        1,
    ));
    route
        .crossover_decisions
        .push(("metal_dispatch_model_layer_count".to_string(), 36));

    assert_eq!(
        route_execution_semantics_label(&route),
        "metal_multilayer_native_prefix_attention"
    );
}

#[test]
fn route_execution_semantics_marks_multilayer_mixed_prefix_attention() {
    let mut route = RouteMetadata::empty();
    route.crossover_decisions.push((
        "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
        1,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_layers_native_attention".to_string(),
        1,
    ));
    route
        .crossover_decisions
        .push(("metal_dispatch_prefix_layers_cpu_reference".to_string(), 1));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_native_dispatch_count".to_string(),
        35,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_cpu_reference_dispatch_count".to_string(),
        1,
    ));
    route
        .crossover_decisions
        .push(("metal_dispatch_model_layer_count".to_string(), 36));

    assert_eq!(
        route_execution_semantics_label(&route),
        "metal_multilayer_mixed_prefix_attention"
    );
}

#[test]
fn route_execution_semantics_prefers_explicit_real_model_forward_marker() {
    let mut route = RouteMetadata::empty();
    route.crossover_decisions.push((
        "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
        1,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_model_bound_ffn".to_string(),
        1,
    ));
    route
        .crossover_decisions
        .push(("metal_dispatch_real_model_forward".to_string(), 1));

    assert_eq!(
        route_execution_semantics_label(&route),
        "metal_real_model_forward"
    );
}

#[test]
fn route_execution_semantics_marks_model_bound_ffn_decode() {
    let mut route = RouteMetadata::empty();
    route.crossover_decisions.push((
        "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
        1,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_model_bound_ffn".to_string(),
        1,
    ));

    assert_eq!(
        route_execution_semantics_label(&route),
        "metal_model_bound_ffn_decode"
    );
}

#[test]
fn route_execution_semantics_from_json_marks_multilayer_model_incomplete() {
    let route_json = json!({
        "metal_real_model_tensor_inputs": true,
        "metal_model_layer_count": 28
    });

    assert_eq!(
        route_execution_semantics_from_json(&route_json),
        "metal_multilayer_model_incomplete"
    );
}

#[test]
fn route_execution_semantics_from_json_marks_multilayer_native_prefix_attention() {
    let route_json = json!({
        "metal_real_model_tensor_inputs": true,
        "mlx_metal_prefix_layers_attention": true,
        "metal_model_layer_count": 28
    });

    assert_eq!(
        route_execution_semantics_from_json(&route_json),
        "metal_multilayer_native_prefix_attention"
    );
}

#[test]
fn route_execution_semantics_from_json_prefers_explicit_real_model_forward_marker() {
    let route_json = json!({
        "metal_real_model_tensor_inputs": true,
        "mlx_metal_prefix_layers_attention": true,
        "metal_model_layer_count": 28,
        "metal_real_model_forward": true
    });

    assert_eq!(
        route_execution_semantics_from_json(&route_json),
        "metal_real_model_forward"
    );
}

#[test]
fn route_execution_semantics_from_json_marks_multilayer_mixed_prefix_attention() {
    let route_json = json!({
        "metal_real_model_tensor_inputs": true,
        "mlx_metal_prefix_layers_attention": true,
        "metal_prefix_layers_cpu_reference": true,
        "mlx_metal_prefix_dispatch_count": 35,
        "metal_prefix_cpu_reference_dispatch_count": 1,
        "metal_model_layer_count": 28
    });

    assert_eq!(
        route_execution_semantics_from_json(&route_json),
        "metal_multilayer_mixed_prefix_attention"
    );
}

#[test]
fn serialize_route_metadata_surfaces_runtime_complete_model_forward_support() {
    let mut route = RouteMetadata::empty();
    route.crossover_decisions.push((
        "metal_dispatch_runtime_complete_model_forward_supported".to_string(),
        1,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_native_dispatch_count".to_string(),
        35,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_cpu_reference_dispatch_count".to_string(),
        1,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_native_projection_row_count".to_string(),
        4096,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_cpu_projection_row_count".to_string(),
        512,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_native_rms_norm_element_count".to_string(),
        2048,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_cpu_rms_norm_element_count".to_string(),
        128,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_native_ffn_activation_element_count".to_string(),
        1024,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_cpu_ffn_activation_element_count".to_string(),
        64,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_native_residual_add_element_count".to_string(),
        4096,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_prefix_cpu_residual_add_element_count".to_string(),
        0,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_native_projection_f16_binding_count".to_string(),
        28,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_native_rms_norm_bf16_binding_count".to_string(),
        28,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_native_projection_row_count".to_string(),
        151936,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_cpu_projection_row_count".to_string(),
        3072,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_native_rms_norm_element_count".to_string(),
        6144,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_cpu_rms_norm_element_count".to_string(),
        0,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_native_ffn_activation_element_count".to_string(),
        1536,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_cpu_ffn_activation_element_count".to_string(),
        0,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_native_residual_add_element_count".to_string(),
        3072,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_cpu_residual_add_element_count".to_string(),
        0,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_batched_logits_group_count".to_string(),
        1,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_batched_logits_token_count".to_string(),
        2,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_batched_group_fallback_count".to_string(),
        1,
    ));
    route.crossover_decisions.push((
        "metal_dispatch_direct_decode_batched_group_fallback_token_count".to_string(),
        2,
    ));

    let route_json = serialize_route_metadata(&route);

    assert_eq!(
        route_json.get("metal_complete_model_forward_supported"),
        Some(&Value::Bool(true))
    );
    assert_eq!(
        route_json.get("mlx_metal_prefix_dispatch_count"),
        Some(&Value::from(35))
    );
    assert_eq!(
        route_json.get("metal_prefix_cpu_reference_dispatch_count"),
        Some(&Value::from(1))
    );
    assert_eq!(
        route_json.get("mlx_metal_prefix_projection_row_count"),
        Some(&Value::from(4096))
    );
    assert_eq!(
        route_json.get("metal_prefix_cpu_projection_row_count"),
        Some(&Value::from(512))
    );
    assert_eq!(
        route_json.get("mlx_metal_prefix_rms_norm_element_count"),
        Some(&Value::from(2048))
    );
    assert_eq!(
        route_json.get("metal_prefix_cpu_rms_norm_element_count"),
        Some(&Value::from(128))
    );
    assert_eq!(
        route_json.get("mlx_metal_prefix_ffn_activation_element_count"),
        Some(&Value::from(1024))
    );
    assert_eq!(
        route_json.get("metal_prefix_cpu_ffn_activation_element_count"),
        Some(&Value::from(64))
    );
    assert_eq!(
        route_json.get("mlx_metal_prefix_residual_add_element_count"),
        Some(&Value::from(4096))
    );
    assert_eq!(
        route_json.get("metal_prefix_cpu_residual_add_element_count"),
        Some(&Value::from(0))
    );
    assert_eq!(
        route_json.get("metal_prefix_projection_mlx_metal_dispatch_share"),
        Some(&Value::from(4096.0_f64 / 4608.0_f64))
    );
    assert_eq!(
        route_json.get("metal_prefix_rms_norm_mlx_metal_dispatch_share"),
        Some(&Value::from(2048.0_f64 / 2176.0_f64))
    );
    assert_eq!(
        route_json.get("metal_prefix_ffn_activation_mlx_metal_dispatch_share"),
        Some(&Value::from(1024.0_f64 / 1088.0_f64))
    );
    assert_eq!(
        route_json.get("metal_prefix_residual_add_mlx_metal_dispatch_share"),
        Some(&Value::from(1.0_f64))
    );
    assert_eq!(
        route_json.get("mlx_metal_projection_f16_binding_count"),
        Some(&Value::from(28))
    );
    assert_eq!(
        route_json.get("mlx_metal_rms_norm_bf16_binding_count"),
        Some(&Value::from(28))
    );
    assert_eq!(
        route_json.get("mlx_metal_direct_decode_projection_row_count"),
        Some(&Value::from(151936))
    );
    assert_eq!(
        route_json.get("metal_direct_decode_cpu_projection_row_count"),
        Some(&Value::from(3072))
    );
    assert_eq!(
        route_json.get("mlx_metal_direct_decode_rms_norm_element_count"),
        Some(&Value::from(6144))
    );
    assert_eq!(
        route_json.get("metal_direct_decode_cpu_rms_norm_element_count"),
        Some(&Value::from(0))
    );
    assert_eq!(
        route_json.get("mlx_metal_direct_decode_ffn_activation_element_count"),
        Some(&Value::from(1536))
    );
    assert_eq!(
        route_json.get("metal_direct_decode_cpu_ffn_activation_element_count"),
        Some(&Value::from(0))
    );
    assert_eq!(
        route_json.get("mlx_metal_direct_decode_residual_add_element_count"),
        Some(&Value::from(3072))
    );
    assert_eq!(
        route_json.get("metal_direct_decode_cpu_residual_add_element_count"),
        Some(&Value::from(0))
    );
    assert_eq!(
        route_json.get("metal_direct_decode_batched_logits_group_count"),
        Some(&Value::from(1))
    );
    assert_eq!(
        route_json.get("metal_direct_decode_batched_logits_token_count"),
        Some(&Value::from(2))
    );
    assert_eq!(
        route_json.get("metal_direct_decode_batched_group_fallback_count"),
        Some(&Value::from(1))
    );
    assert_eq!(
        route_json.get("metal_direct_decode_batched_group_fallback_token_count"),
        Some(&Value::from(2))
    );
    assert_eq!(
        route_json.get("metal_direct_decode_projection_mlx_metal_dispatch_share"),
        Some(&Value::from(151936.0_f64 / 155008.0_f64))
    );
    assert_eq!(
        route_json.get("metal_direct_decode_rms_norm_mlx_metal_dispatch_share"),
        Some(&Value::from(1.0_f64))
    );
    assert_eq!(
        route_json.get("metal_direct_decode_ffn_activation_mlx_metal_dispatch_share"),
        Some(&Value::from(1.0_f64))
    );
    assert_eq!(
        route_json.get("metal_direct_decode_residual_add_mlx_metal_dispatch_share"),
        Some(&Value::from(1.0_f64))
    );
}

#[test]
fn serialize_route_metadata_preserves_mlx_kv_cache_counters() {
    let mut route = RouteMetadata::empty();
    route.crossover_decisions.push((
        ax_engine_core::ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB.to_string(),
        24,
    ));
    route.crossover_decisions.push((
        ax_engine_core::ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT.to_string(),
        3,
    ));
    route.crossover_decisions.push((
        ax_engine_core::ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB.to_string(),
        8,
    ));

    let route_json = serialize_route_metadata(&route);
    let decisions = route_json
        .get("crossover_decisions")
        .and_then(Value::as_object)
        .expect("route decisions should serialize as an object");

    assert_eq!(
        decisions.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB),
        Some(&Value::from(24))
    );
    assert_eq!(
        decisions.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT),
        Some(&Value::from(3))
    );
    assert_eq!(
        decisions.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB),
        Some(&Value::from(8))
    );
    assert_eq!(
        route_json.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB),
        Some(&Value::from(24))
    );
    assert_eq!(
        route_json.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT),
        Some(&Value::from(3))
    );
    assert_eq!(
        route_json.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB),
        Some(&Value::from(8))
    );
}

#[test]
fn trusted_baseline_json_infers_tool_mode_from_runtime_when_software_tool_mode_is_missing() {
    let manifest = json!({
        "id": "chat_qwen_short",
        "class": "scenario",
        "scenario": "chat",
        "model": {
            "family": "qwen3"
        }
    });
    let mut environment = native_environment_fixture();
    environment
        .get_mut("software")
        .and_then(Value::as_object_mut)
        .expect("fixture software should serialize as object")
        .remove("tool_mode");
    let metrics = json!({
        "run_id": "run-a",
        "metrics": {
            "ttft_ms": 10.0,
            "decode_tok_s": 120.0,
            "memory_peak_mb": 512.0,
            "prefix_hit_rate": 0.0
        }
    });

    let baseline = build_trusted_baseline_json(
        "Dense Qwen M4 Baseline",
        "Dense-Qwen-M4-Baseline",
        Path::new("/tmp/ax-engine-bench-baseline"),
        &manifest,
        &environment,
        &metrics,
    )
    .expect("trusted baseline JSON should build");

    assert_eq!(
        nested_value(&baseline, &["runtime", "tool_mode"]).and_then(Value::as_str),
        Some("engine_bringup_runtime")
    );
    assert_eq!(
        nested_value(&baseline, &["route", "execution_semantics"]).and_then(Value::as_str),
        Some("none_observed")
    );
}

#[test]
fn trusted_baseline_json_preserves_native_model_provenance() {
    let manifest = json!({
        "id": "chat_qwen_short",
        "class": "scenario",
        "scenario": "chat",
        "model": {
            "family": "qwen3"
        }
    });
    let mut environment = native_environment_fixture();
    environment["runtime"]["mlx_model"] = json!({
        "artifacts_source": "explicit_config",
        "model_family": "qwen3",
        "tensor_format": "safetensors",
        "layer_count": 36,
        "tensor_count": 402,
        "tie_word_embeddings": false,
        "bindings_prepared": false,
        "buffers_bound": false,
        "buffer_count": 0,
        "buffer_bytes": 0
    });
    let metrics = json!({
        "run_id": "run-native-model",
        "metrics": {
            "ttft_ms": 10.0,
            "decode_tok_s": 120.0,
            "memory_peak_mb": 512.0,
            "prefix_hit_rate": 0.0
        }
    });

    let baseline = build_trusted_baseline_json(
        "Dense Qwen Native Model Baseline",
        "Dense-Qwen-Native-Model-Baseline",
        Path::new("/tmp/ax-engine-bench-baseline"),
        &manifest,
        &environment,
        &metrics,
    )
    .expect("trusted baseline JSON should build");

    assert_eq!(
        nested_value(&baseline, &["runtime", "mlx_model", "artifacts_source"])
            .and_then(Value::as_str),
        Some("explicit_config")
    );
    assert_eq!(
        nested_value(&baseline, &["runtime", "mlx_model", "tensor_format"]).and_then(Value::as_str),
        Some("safetensors")
    );
    assert_eq!(
        nested_value(&baseline, &["runtime", "mlx_model", "tensor_count"]).and_then(Value::as_u64),
        Some(402)
    );
}

#[test]
fn matrix_compare_member_result_infers_deterministic_native_tool_mode_from_runtime() {
    let baseline_member = json!({
        "label": "Chat Qwen Short"
    });
    let regression = json!({
        "runtime": mlx_runtime_fixture(),
        "summary": {
            "selected_backend": "mlx",
            "support_tier": "mlx_preview",
            "resolution_policy": "mlx_only"
        },
        "comparison": {
            "ttft_ms_pct": -5.0,
            "decode_tok_s_pct": 8.0,
            "memory_peak_mb_pct": 0.0,
            "prefix_hit_rate_pct": 0.0
        }
    });

    let member = build_matrix_compare_member_result(
        &baseline_member,
        "chat_qwen_short",
        Path::new("/tmp/ax-engine-bench-compare-member"),
        &regression,
    )
    .expect("member result should build");

    assert_eq!(member.tool_mode, "engine_bringup_runtime");
    assert_eq!(member.compare_mode, "engine_bringup_compare");
    assert_eq!(member.execution_semantics, "unknown");
}

#[test]
fn serialize_runtime_metadata_includes_native_model_provenance() {
    let model_dir = write_valid_native_model_fixture();
    let runtime = RuntimeConfig {
        deterministic: false,
        max_batch_tokens: 128,
        block_size_tokens: 16,
        kv_total_blocks: Some(32),
        flags: RuntimeFlags::default(),
        llama_cpp_preset: None,
        backend_policy: BackendPolicy::new(ResolutionPolicy::MlxOnly),
        resolved_backend: ResolvedBackend::new(SelectedBackend::Mlx, SupportTier::MlxPreview, None),
        backend_adapter: None,
        mlx_model_artifacts_dir: Some(model_dir.clone()),
        mlx_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
    };

    let runtime_json = serialize_runtime_metadata(&runtime, None);

    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("artifacts_source"))
            .and_then(Value::as_str),
        Some("explicit_config")
    );
    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("model_family"))
            .and_then(Value::as_str),
        Some("qwen3")
    );
    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("layer_count"))
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("tensor_count"))
            .and_then(Value::as_u64),
        Some(9)
    );
    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("bindings_prepared"))
            .and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("buffers_bound"))
            .and_then(Value::as_bool),
        Some(false)
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[test]
fn serialize_runtime_metadata_keeps_native_model_summary_when_runtime_binding_fails() {
    let model_dir = write_valid_native_model_fixture();
    let _missing_build_dir = unique_test_dir("missing-native-runtime-build");
    let runtime = RuntimeConfig {
        deterministic: false,
        max_batch_tokens: 128,
        block_size_tokens: 16,
        kv_total_blocks: Some(32),
        flags: RuntimeFlags::default(),
        llama_cpp_preset: None,
        backend_policy: BackendPolicy::new(ResolutionPolicy::MlxOnly),
        resolved_backend: ResolvedBackend::new(SelectedBackend::Mlx, SupportTier::MlxPreview, None),
        backend_adapter: None,
        mlx_model_artifacts_dir: Some(model_dir.clone()),
        mlx_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
    };

    let runtime_json = serialize_runtime_metadata(&runtime, None);

    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("model_family"))
            .and_then(Value::as_str),
        Some("qwen3")
    );
    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("bindings_prepared"))
            .and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("buffers_bound"))
            .and_then(Value::as_bool),
        Some(false)
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[test]
fn serialize_runtime_metadata_omits_config_side_model_summary_for_file_path() {
    let runtime = RuntimeConfig {
        deterministic: false,
        max_batch_tokens: 128,
        block_size_tokens: 16,
        kv_total_blocks: Some(32),
        flags: RuntimeFlags::default(),
        llama_cpp_preset: None,
        backend_policy: BackendPolicy::new(ResolutionPolicy::MlxOnly),
        resolved_backend: ResolvedBackend::new(SelectedBackend::Mlx, SupportTier::MlxPreview, None),
        backend_adapter: None,
        mlx_model_artifacts_dir: Some(PathBuf::from("/tmp/google_gemma-4-26b-it-q4_k_m.gguf")),
        mlx_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
    };

    let runtime_json = serialize_runtime_metadata(&runtime, None);

    assert_eq!(
        runtime_json
            .get("mlx_runtime")
            .and_then(|value| value.get("runner"))
            .and_then(Value::as_str),
        None
    );
    assert_eq!(runtime_json.get("mlx_model"), None);
}

#[test]
fn serialize_runtime_metadata_prefers_actual_runtime_report_over_config_side_metadata() {
    let model_dir = write_valid_native_model_fixture();
    let runtime = RuntimeConfig {
        deterministic: false,
        max_batch_tokens: 128,
        block_size_tokens: 16,
        kv_total_blocks: Some(32),
        flags: RuntimeFlags::default(),
        llama_cpp_preset: None,
        backend_policy: BackendPolicy::new(ResolutionPolicy::MlxOnly),
        resolved_backend: ResolvedBackend::new(SelectedBackend::Mlx, SupportTier::MlxPreview, None),
        backend_adapter: None,
        mlx_model_artifacts_dir: Some(model_dir.clone()),
        mlx_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
    };
    let actual_runtime = RuntimeReport {
        selected_backend: SelectedBackend::Mlx,
        support_tier: SupportTier::MlxCertified,
        resolution_policy: ResolutionPolicy::MlxOnly,
        capabilities: ResolvedBackend::mlx_certified().capabilities,
        fallback_reason: None,
        host: HostReport {
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
            detected_soc: Some("Apple M4 Max".to_string()),
            supported_mlx_runtime: true,
            unsupported_host_override_active: false,
            detection_error: None,
        },
        metal_toolchain: MetalToolchainReport {
            fully_available: true,
            metal: ToolStatusReport {
                available: true,
                version: Some("metal actual".to_string()),
            },
            metallib: ToolStatusReport {
                available: true,
                version: Some("metallib actual".to_string()),
            },
            metal_ar: ToolStatusReport {
                available: true,
                version: Some("metal-ar actual".to_string()),
            },
        },
        mlx_runtime: Some(NativeRuntimeReport::metal_bringup(
            NativeRuntimeArtifactsSource::ExplicitConfig,
        )),
        mlx_model: Some(NativeModelReport {
            artifacts_source: NativeModelArtifactsSource::ExplicitEnv,
            model_family: "actual_bound_model".to_string(),
            tensor_format: ax_engine_core::NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: ax_engine_core::model::NativeRuntimeStatus::default(),
            layer_count: 42,
            tensor_count: 420,
            tie_word_embeddings: true,
            is_moe: false,
            is_hybrid_attention: false,
            hybrid_full_attention_interval: None,
            mla_kv_latent_dim: None,
            moe_active_experts: None,
            bindings_prepared: true,
            buffers_bound: true,
            buffer_count: 12,
            buffer_bytes: 4096,
            source_quantized_binding_count: 0,
            source_q4_k_binding_count: 0,
            source_q5_k_binding_count: 0,
            source_q6_k_binding_count: 0,
            source_q8_0_binding_count: 0,
        }),
    };

    let runtime_json = serialize_runtime_metadata(&runtime, Some(&actual_runtime));

    assert_eq!(
        runtime_json.get("support_tier").and_then(Value::as_str),
        Some("mlx_certified")
    );
    assert_eq!(
        runtime_json
            .get("capabilities")
            .and_then(|value| value.get("long_context_validation"))
            .and_then(Value::as_str),
        Some("supported")
    );
    assert_eq!(
        runtime_json
            .get("metal_toolchain")
            .and_then(|value| value.get("metal"))
            .and_then(|value| value.get("version"))
            .and_then(Value::as_str),
        Some("metal actual")
    );
    assert_eq!(
        runtime_json
            .get("mlx_runtime")
            .and_then(|value| value.get("runner"))
            .and_then(Value::as_str),
        Some("metal_bringup")
    );
    assert_eq!(
        runtime_json
            .get("mlx_runtime")
            .and_then(|value| value.get("artifacts_source"))
            .and_then(Value::as_str),
        Some("explicit_config")
    );
    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("artifacts_source"))
            .and_then(Value::as_str),
        Some("explicit_env")
    );
    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("model_family"))
            .and_then(Value::as_str),
        Some("actual_bound_model")
    );
    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("bindings_prepared"))
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        runtime_json
            .get("mlx_model")
            .and_then(|value| value.get("buffers_bound"))
            .and_then(Value::as_bool),
        Some(true)
    );

    let _ = fs::remove_dir_all(model_dir);
}

fn llama_runtime_fixture(server_url: &str) -> Value {
    json!({
        "selected_backend": "llama_cpp",
        "support_tier": "llama_cpp",
        "resolution_policy": "allow_llama_cpp",
        "backend_adapter": {
            "kind": "llama_cpp_server_completion",
            "server_url": server_url
        },
        "host": runtime_host_fixture(),
        "metal_toolchain": runtime_metal_toolchain_fixture()
    })
}

fn runtime_host_fixture() -> Value {
    json!({
        "os": "macos",
        "arch": "aarch64",
        "detected_soc": "Apple M4 Max",
        "supported_mlx_runtime": true,
        "unsupported_host_override_active": false
    })
}

fn runtime_metal_toolchain_fixture() -> Value {
    json!({
        "fully_available": true,
        "metal": {
            "available": true,
            "version": "Apple metal version 36000.4"
        },
        "metallib": {
            "available": true,
            "version": "Apple metallib version 36000.4"
        },
        "metal_ar": {
            "available": true,
            "version": "Apple metal-ar version 36000.4"
        }
    })
}

fn doctor_host_fixture(
    supported_mlx_runtime: bool,
    unsupported_host_override_active: bool,
    detected_soc: Option<&str>,
) -> HostReport {
    HostReport {
        os: "macos".to_string(),
        arch: "aarch64".to_string(),
        detected_soc: detected_soc.map(str::to_string),
        supported_mlx_runtime,
        unsupported_host_override_active,
        detection_error: None,
    }
}

fn doctor_tool_status_fixture(available: bool, version: Option<&str>) -> ToolStatusReport {
    ToolStatusReport {
        available,
        version: version.map(str::to_string),
    }
}

fn doctor_metal_toolchain_fixture(
    metal_available: bool,
    metallib_available: bool,
    metal_ar_available: bool,
) -> MetalToolchainReport {
    MetalToolchainReport {
        fully_available: metal_available && metallib_available,
        metal: doctor_tool_status_fixture(metal_available, Some("Apple metal version 36000.4")),
        metallib: doctor_tool_status_fixture(
            metallib_available,
            Some("Apple metallib version 36000.4"),
        ),
        metal_ar: doctor_tool_status_fixture(
            metal_ar_available,
            Some("Apple metal-ar version 36000.4"),
        ),
    }
}

#[test]
fn doctor_report_marks_supported_m4_host_ready() {
    let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
    let toolchain = doctor_metal_toolchain_fixture(true, true, true);

    let report = build_doctor_report(host, toolchain);

    assert_eq!(report.status, DoctorStatus::Ready);
    assert!(report.mlx_runtime_ready);
    assert!(report.bringup_allowed);
    assert!(report.issues.is_empty());
    assert_eq!(
        report.notes,
        vec!["llama.cpp backends do not widen supported host scope".to_string()]
    );
    assert!(
        report
            .performance_advice
            .iter()
            .any(|advice| advice.id == "ngram_acceleration_default_on")
    );
    assert!(
        report
            .performance_advice
            .iter()
            .any(|advice| advice.id == "swiftlm_is_baseline_only")
    );
}

#[test]
fn doctor_report_marks_override_host_as_bringup_only() {
    let host = doctor_host_fixture(false, true, Some("Apple M3 Max"));
    let toolchain = doctor_metal_toolchain_fixture(true, true, true);

    let report = build_doctor_report(host, toolchain);

    assert_eq!(report.status, DoctorStatus::BringupOnly);
    assert!(!report.mlx_runtime_ready);
    assert!(report.bringup_allowed);
    assert!(
        report
            .issues
            .iter()
            .any(|issue| issue.contains("Apple M3 Max"))
    );
    assert!(
        report
            .issues
            .iter()
            .any(|issue| issue.contains("AX_ALLOW_UNSUPPORTED_HOST"))
    );
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("development or CI bring-up"))
    );
}

#[test]
fn doctor_report_marks_missing_metal_toolchain_not_ready() {
    let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
    let toolchain = doctor_metal_toolchain_fixture(true, false, false);

    let report = build_doctor_report(host, toolchain);

    assert_eq!(report.status, DoctorStatus::NotReady);
    assert!(!report.mlx_runtime_ready);
    assert!(!report.bringup_allowed);
    assert!(
        report
            .issues
            .iter()
            .any(|issue| issue.contains("xcrun metallib"))
    );
    assert!(
        !report
            .issues
            .iter()
            .any(|issue| issue.contains("xcrun metal-ar"))
    );
}

#[test]
fn doctor_report_accepts_command_line_tools_without_metal_ar() {
    let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
    let toolchain = doctor_metal_toolchain_fixture(true, true, false);

    let report = build_doctor_report(host, toolchain);

    assert_eq!(report.status, DoctorStatus::Ready);
    assert!(report.mlx_runtime_ready);
    assert!(report.bringup_allowed);
    assert!(report.issues.is_empty());
}

#[test]
fn doctor_report_marks_model_artifacts_not_selected_by_default() {
    let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
    let toolchain = doctor_metal_toolchain_fixture(true, true, true);

    let report = build_doctor_report(host, toolchain);

    assert_eq!(
        report.model_artifacts.status,
        DoctorModelArtifactsStatus::NotSelected
    );
    assert!(!report.model_artifacts.selected);
    assert!(report.model_artifacts.path.is_none());
}

#[test]
fn doctor_report_surfaces_ready_model_artifacts() {
    let root = unique_test_dir("doctor-ready-model-artifacts");
    let model_dir = root.join("ready-mlx-snapshot");
    fs::create_dir_all(&model_dir).expect("model dir should create");
    fs::write(
        model_dir.join("config.json"),
        r#"{
                "model_type": "qwen3",
                "quantization": {
                    "mode": "affine",
                    "group_size": 64,
                    "bits": 4
                }
            }"#,
    )
    .expect("config should write");
    write_doctor_model_manifest(&model_dir);
    write_doctor_safetensors(&model_dir);
    let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
    let toolchain = doctor_metal_toolchain_fixture(true, true, true);

    let report = build_doctor_report_for_model(host, toolchain, Some(&model_dir));

    assert_eq!(
        report.model_artifacts.status,
        DoctorModelArtifactsStatus::Ready
    );
    assert!(report.model_artifacts.config_present);
    assert!(report.model_artifacts.manifest_present);
    assert!(report.model_artifacts.safetensors_present);
    assert_eq!(report.model_artifacts.model_type.as_deref(), Some("qwen3"));
    assert_eq!(
        report
            .model_artifacts
            .quantization
            .as_ref()
            .and_then(|quantization| quantization.bits),
        Some(4)
    );
    assert!(report.model_artifacts.issues.is_empty());

    fs::remove_dir_all(root).expect("test dir should clean up");
}

#[test]
fn doctor_report_surfaces_model_artifact_blockers() {
    let root = unique_test_dir("doctor-model-artifact-blockers");
    let model_dir = root.join("plain-mlx-snapshot");
    fs::create_dir_all(&model_dir).expect("model dir should create");
    fs::write(model_dir.join("config.json"), r#"{"model_type":"qwen3"}"#)
        .expect("config should write");
    let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
    let toolchain = doctor_metal_toolchain_fixture(true, true, true);

    let report = build_doctor_report_for_model(host, toolchain, Some(&model_dir));

    assert_eq!(
        report.model_artifacts.status,
        DoctorModelArtifactsStatus::NotReady
    );
    assert!(report.model_artifacts.config_present);
    assert!(!report.model_artifacts.manifest_present);
    assert!(!report.model_artifacts.safetensors_present);
    assert!(
        report
            .model_artifacts
            .issues
            .iter()
            .any(|issue| issue.contains("model-manifest.json"))
    );
    assert!(
        report
            .model_artifacts
            .issues
            .iter()
            .any(|issue| issue.contains("safetensors"))
    );

    fs::remove_dir_all(root).expect("test dir should clean up");
}

#[test]
fn doctor_workflow_detects_source_checkout_commands() {
    let root = unique_test_dir("doctor-source-workflow");
    let nested = root.join("crates/ax-engine-bench");
    fs::create_dir_all(root.join("scripts")).expect("scripts dir should create");
    fs::create_dir_all(&nested).expect("nested crate dir should create");
    fs::create_dir_all(root.join("crates/ax-engine-server"))
        .expect("server crate dir should create");
    fs::write(root.join("Cargo.toml"), "[workspace]\n").expect("workspace toml should write");
    fs::write(root.join("scripts/download_model.py"), "").expect("download script should write");
    fs::write(
        nested.join("Cargo.toml"),
        "[package]\nname = \"ax-engine-bench\"\n",
    )
    .expect("bench toml should write");
    fs::write(
        root.join("crates/ax-engine-server/Cargo.toml"),
        "[package]\nname = \"ax-engine-server\"\n",
    )
    .expect("server toml should write");

    let report = doctor_workflow_report_for_cwd(&nested);

    assert_eq!(report.mode, DoctorWorkflowMode::SourceCheckout);
    assert_eq!(report.source_root, Some(path_string(&root)));
    assert_eq!(
        report.doctor.argv,
        vec![
            "cargo",
            "run",
            "-p",
            "ax-engine-bench",
            "--bin",
            "ax-engine-bench",
            "--",
            "doctor",
            "--json"
        ]
    );
    assert!(command_text(&report.doctor).contains("[in:"));
    assert_eq!(
        report.download_model.as_ref().map(|command| &command.argv),
        Some(&vec![
            "python3".to_string(),
            "scripts/download_model.py".to_string(),
            "<repo-id>".to_string(),
            "--json".to_string()
        ])
    );

    fs::remove_dir_all(root).expect("test dir should clean up");
}

#[test]
fn doctor_quantization_without_bits_does_not_infer_four_bit_policy() {
    let root = unique_test_dir("doctor-quantization-missing-bits");
    let model_dir = root.join("gemma4-no-bits");
    fs::create_dir_all(&model_dir).expect("model dir should create");
    fs::write(
        model_dir.join("config.json"),
        r#"{
                "model_type": "gemma4",
                "quantization": {
                    "mode": "affine",
                    "group_size": 64
                }
            }"#,
    )
    .expect("config should write");
    write_doctor_model_manifest(&model_dir);
    let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
    let toolchain = doctor_metal_toolchain_fixture(true, true, true);

    let report = build_doctor_report_for_model(host, toolchain, Some(&model_dir));

    assert_eq!(
        report
            .model_artifacts
            .quantization
            .as_ref()
            .and_then(|quantization| quantization.bits),
        None
    );
    assert!(
        report
            .performance_advice
            .iter()
            .any(|advice| advice.id == "quantization_bits_missing")
    );
    assert!(
        !report
            .performance_advice
            .iter()
            .any(|advice| advice.id == "gemma4_4bit_first")
    );

    fs::remove_dir_all(root).expect("test dir should clean up");
}

#[test]
fn doctor_workflow_uses_installed_tools_outside_source_checkout() {
    let root = unique_test_dir("doctor-installed-workflow");
    fs::create_dir_all(&root).expect("test dir should create");

    let report = doctor_workflow_report_for_cwd(&root);

    assert_eq!(report.mode, DoctorWorkflowMode::InstalledTools);
    assert!(report.source_root.is_none());
    assert_eq!(
        report.doctor.argv,
        vec!["ax-engine-bench", "doctor", "--json"]
    );
    assert_eq!(report.server.argv, vec!["ax-engine-server"]);
    assert!(report.download_model.is_none());

    fs::remove_dir_all(root).expect("test dir should clean up");
}

#[test]
fn render_doctor_report_includes_status_and_issue_sections() {
    let host = doctor_host_fixture(false, true, Some("Apple M3 Max"));
    let toolchain = doctor_metal_toolchain_fixture(true, false, true);
    let report = build_doctor_report(host, toolchain);

    let text = render_doctor_report(&report);

    assert!(text.contains("AX Engine v6 doctor"));
    assert!(text.contains("Status: not ready"));
    assert!(text.contains("Summary:"));
    assert!(text.contains("  - Host: Apple M3 Max (macos/aarch64)"));
    assert!(text.contains("Workflow:"));
    assert!(text.contains("  - Mode: unknown"));
    assert!(text.contains("Model artifacts:"));
    assert!(text.contains("Issues:"));
    assert!(text.contains("Notes:"));
    assert!(text.contains("Performance advice:"));
    assert!(text.contains("Warnings:"));
    assert!(text.contains("Info:"));
    assert!(text.contains("ngram_acceleration_default_on"));
    assert!(text.contains("llama.cpp backends do not widen supported host scope"));
}

#[test]
fn doctor_report_adds_qwen36_quantization_advice_from_model_config() {
    let root = unique_test_dir("doctor-qwen36");
    let model_dir = root.join("Qwen3.6-35B-A3B-4bit");
    fs::create_dir_all(&model_dir).expect("model dir should create");
    fs::write(
        model_dir.join("config.json"),
        r#"{
                "model_type": "qwen3_next",
                "quantization": {
                    "mode": "affine",
                    "group_size": 64,
                    "bits": 4
                }
            }"#,
    )
    .expect("config should write");
    write_doctor_model_manifest(&model_dir);
    let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
    let toolchain = doctor_metal_toolchain_fixture(true, true, true);

    let report = build_doctor_report_for_model(host, toolchain, Some(&model_dir));

    assert!(
        report
            .performance_advice
            .iter()
            .any(|advice| advice.id == "qwen36_quantization_compare"
                && advice.severity == DoctorAdviceSeverity::Warning)
    );
    assert!(
        report
            .performance_advice
            .iter()
            .any(|advice| advice.id == "qwen_gated_delta_prefill_scope")
    );

    fs::remove_dir_all(root).expect("test dir should clean up");
}

#[test]
fn doctor_report_adds_gemma4_quantization_advice_from_model_config() {
    let root = unique_test_dir("doctor-gemma4");
    let model_dir = root.join("gemma-4-e2b-it-4bit");
    fs::create_dir_all(&model_dir).expect("model dir should create");
    fs::write(
        model_dir.join("config.json"),
        r#"{
                "model_type": "gemma4",
                "quantization_config": {
                    "mode": "affine",
                    "group_size": 64,
                    "bits": 4
                }
            }"#,
    )
    .expect("config should write");
    write_doctor_model_manifest(&model_dir);
    let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
    let toolchain = doctor_metal_toolchain_fixture(true, true, true);

    let report = build_doctor_report_for_model(host, toolchain, Some(&model_dir));

    assert!(
        report
            .performance_advice
            .iter()
            .any(|advice| advice.id == "gemma4_4bit_first")
    );

    fs::remove_dir_all(root).expect("test dir should clean up");
}

#[test]
fn doctor_report_points_plain_mlx_snapshot_to_manifest_generation() {
    let root = unique_test_dir("doctor-missing-manifest");
    let model_dir = root.join("plain-mlx-snapshot");
    fs::create_dir_all(&model_dir).expect("model dir should create");
    fs::write(
        model_dir.join("config.json"),
        r#"{"model_type":"qwen3","quantization":{"mode":"affine","group_size":64,"bits":4}}"#,
    )
    .expect("config should write");
    let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
    let toolchain = doctor_metal_toolchain_fixture(true, true, true);

    let report = build_doctor_report_for_model(host, toolchain, Some(&model_dir));

    assert!(report.performance_advice.iter().any(|advice| {
        advice.id == "model_artifacts_unreadable"
            && advice.detail.contains("model-manifest.json")
            && advice.detail.contains("generate-manifest")
    }));

    fs::remove_dir_all(root).expect("test dir should clean up");
}

#[test]
fn parse_doctor_args_accepts_json_and_rejects_unknown_flags() {
    let args = vec![
        "--json".to_string(),
        "--mlx-model-artifacts-dir".to_string(),
        "/tmp/google_gemma-4-26b-it-q4_k_m.gguf".to_string(),
    ];
    let parsed = parse_doctor_args(&args).expect("doctor args should parse");
    assert!(parsed.json);
    assert_eq!(
        parsed.mlx_model_artifacts_dir,
        Some(PathBuf::from("/tmp/google_gemma-4-26b-it-q4_k_m.gguf"))
    );

    let error = parse_doctor_args(&["--bogus".to_string()])
        .expect_err("doctor args should reject unknown flags");
    let CliError::Usage(message) = error else {
        panic!("doctor args should surface a usage error");
    };
    assert!(message.contains("unknown flag for doctor"));
    assert!(message.contains("ax-engine-bench doctor [--json] [--mlx-model-artifacts-dir <path>]"));
}

#[test]
fn parse_autotune_args_accepts_defaults_and_explicit_overrides() {
    let parsed = parse_autotune_args(&[
        "--manifest".to_string(),
        "/tmp/manifest.json".to_string(),
        "--output-root".to_string(),
        "/tmp/out".to_string(),
        "--iterations".to_string(),
        "5".to_string(),
        "--exploration-weight".to_string(),
        "1.25".to_string(),
        "--max-batch-token-options".to_string(),
        "64,128,256".to_string(),
        "--kv-total-block-options".to_string(),
        "none,32,64".to_string(),
        "--prefix-cache-options".to_string(),
        "true,false".to_string(),
    ])
    .expect("autotune args should parse");

    assert_eq!(
        parsed,
        AutotuneArgs {
            manifest_path: PathBuf::from("/tmp/manifest.json"),
            output_root: PathBuf::from("/tmp/out"),
            iterations: 5,
            exploration_weight: 1.25,
            max_batch_token_options: Some(vec![64, 128, 256]),
            kv_total_block_options: Some(vec![None, Some(32), Some(64)]),
            prefix_cache_options: Some(vec![false, true]),
            disable_history: false,
        }
    );
}

#[test]
fn parse_autotune_args_accepts_disable_history_flag() {
    let parsed = parse_autotune_args(&[
        "--manifest".to_string(),
        "/tmp/manifest.json".to_string(),
        "--output-root".to_string(),
        "/tmp/out".to_string(),
        "--disable-history".to_string(),
    ])
    .expect("autotune args should parse");

    assert!(parsed.disable_history);
}

#[test]
fn autotune_candidate_configs_keep_manifest_runtime_as_first_trial() {
    let manifest = load_test_manifest(
        repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
    );

    let search_space = resolve_autotune_search_space(
        &manifest,
        &AutotuneArgs {
            manifest_path: PathBuf::from("/tmp/manifest.json"),
            output_root: PathBuf::from("/tmp/out"),
            iterations: 8,
            exploration_weight: 0.5,
            max_batch_token_options: None,
            kv_total_block_options: None,
            prefix_cache_options: None,
            disable_history: false,
        },
    );
    let candidates = autotune_candidate_configs(&manifest, &search_space);

    assert!(!candidates.is_empty());
    assert_eq!(
        candidates[0],
        AutotuneCandidateConfig {
            max_batch_tokens: manifest.runtime.max_batch_tokens,
            kv_total_blocks: manifest.runtime.kv_total_blocks,
            prefix_cache: manifest.runtime.flags.prefix_cache,
        }
    );
}

#[test]
fn autotune_candidate_configs_respect_explicit_search_space_overrides() {
    let manifest = load_test_manifest(
        repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
    );
    let search_space = resolve_autotune_search_space(
        &manifest,
        &AutotuneArgs {
            manifest_path: PathBuf::from("/tmp/manifest.json"),
            output_root: PathBuf::from("/tmp/out"),
            iterations: 4,
            exploration_weight: 0.5,
            max_batch_token_options: Some(vec![64, 256]),
            kv_total_block_options: Some(vec![None, Some(48)]),
            prefix_cache_options: Some(vec![true]),
            disable_history: false,
        },
    );

    let candidates = autotune_candidate_configs(&manifest, &search_space);

    assert_eq!(search_space.max_batch_token_options, vec![64, 256]);
    assert_eq!(search_space.kv_total_block_options, vec![None, Some(48)]);
    assert_eq!(search_space.prefix_cache_options, vec![true]);
    assert_eq!(candidates.len(), 4);
    assert!(candidates.iter().all(|candidate| candidate.prefix_cache));
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.max_batch_tokens == 64 || candidate.max_batch_tokens == 256)
    );
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.kv_total_blocks.is_none()
                || candidate.kv_total_blocks == Some(48))
    );
}

#[test]
fn autotune_selector_picks_untried_candidate_after_base_trial() {
    let candidates = vec![
        AutotuneCandidateConfig {
            max_batch_tokens: 128,
            kv_total_blocks: Some(32),
            prefix_cache: false,
        },
        AutotuneCandidateConfig {
            max_batch_tokens: 256,
            kv_total_blocks: Some(32),
            prefix_cache: false,
        },
        AutotuneCandidateConfig {
            max_batch_tokens: 128,
            kv_total_blocks: Some(64),
            prefix_cache: true,
        },
    ];
    let search_space = AutotuneSearchSpace {
        max_batch_token_options: vec![128, 256],
        kv_total_block_options: vec![Some(32), Some(64)],
        prefix_cache_options: vec![false, true],
    };
    let trials = vec![AutotuneTrialRecord {
        trial_index: 0,
        candidate: candidates[0].clone(),
        selection: AutotuneSelectionDiagnostics {
            strategy: "base_config_seed".to_string(),
            predicted_mean: None,
            uncertainty: None,
            acquisition: None,
            good_density: None,
            bad_density: None,
            density_ratio: None,
            novelty_bonus: None,
        },
        probe: None,
        score: 42.0,
        status: "ok".to_string(),
        result_dir: None,
        error: None,
        metrics: None,
    }];

    let selected = select_next_autotune_candidate(&candidates, &search_space, &trials, 0.5);

    assert!(selected.candidate_index == 1 || selected.candidate_index == 2);
    assert_ne!(selected.candidate_index, 0);
    assert_eq!(selected.diagnostics.strategy, "coverage_explore");
    assert!(selected.diagnostics.predicted_mean.is_none());
    assert!(selected.diagnostics.uncertainty.is_some());
    assert!(selected.diagnostics.acquisition.is_some());
    assert!(selected.diagnostics.novelty_bonus.is_some());
}

#[test]
fn autotune_selector_uses_tpe_ratio_after_multiple_trials() {
    let candidates = vec![
        AutotuneCandidateConfig {
            max_batch_tokens: 128,
            kv_total_blocks: Some(32),
            prefix_cache: false,
        },
        AutotuneCandidateConfig {
            max_batch_tokens: 256,
            kv_total_blocks: Some(32),
            prefix_cache: false,
        },
        AutotuneCandidateConfig {
            max_batch_tokens: 256,
            kv_total_blocks: Some(64),
            prefix_cache: true,
        },
    ];
    let search_space = AutotuneSearchSpace {
        max_batch_token_options: vec![128, 256],
        kv_total_block_options: vec![Some(32), Some(64)],
        prefix_cache_options: vec![false, true],
    };
    let trials = vec![
        AutotuneTrialRecord {
            trial_index: 0,
            candidate: candidates[0].clone(),
            selection: AutotuneSelectionDiagnostics {
                strategy: "base_config_seed".to_string(),
                predicted_mean: None,
                uncertainty: None,
                acquisition: None,
                good_density: None,
                bad_density: None,
                density_ratio: None,
                novelty_bonus: None,
            },
            probe: None,
            score: 10.0,
            status: "ok".to_string(),
            result_dir: None,
            error: None,
            metrics: None,
        },
        AutotuneTrialRecord {
            trial_index: 1,
            candidate: candidates[1].clone(),
            selection: AutotuneSelectionDiagnostics {
                strategy: "coverage_explore".to_string(),
                predicted_mean: None,
                uncertainty: Some(0.5),
                acquisition: Some(0.5),
                good_density: None,
                bad_density: None,
                density_ratio: None,
                novelty_bonus: Some(0.5),
            },
            probe: None,
            score: 100.0,
            status: "ok".to_string(),
            result_dir: None,
            error: None,
            metrics: None,
        },
    ];

    let selected = select_next_autotune_candidate(&candidates, &search_space, &trials, 0.5);

    assert_eq!(selected.candidate_index, 2);
    assert_eq!(selected.diagnostics.strategy, "tpe_ratio");
    assert!(selected.diagnostics.predicted_mean.is_some());
    assert!(selected.diagnostics.good_density.is_some());
    assert!(selected.diagnostics.bad_density.is_some());
    assert!(selected.diagnostics.density_ratio.is_some());
    assert!(selected.diagnostics.novelty_bonus.is_some());
}

#[test]
fn autotune_probe_manifest_reduces_shape_and_disables_determinism_check() {
    let manifest = load_test_manifest(
        repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
    );

    let probe_manifest = autotune_probe_manifest(
        &manifest,
        0,
        &AutotuneCandidateConfig {
            max_batch_tokens: manifest.runtime.max_batch_tokens,
            kv_total_blocks: manifest.runtime.kv_total_blocks,
            prefix_cache: manifest.runtime.flags.prefix_cache,
        },
    )
    .expect("probe manifest should build");

    let original_shape = manifest.shape.expect("scenario shape should exist");
    let probe_shape = probe_manifest.shape.expect("probe shape should exist");
    assert!(probe_manifest.id.ends_with("-probe"));
    assert!(!probe_manifest.checks.expect_deterministic);
    assert!(probe_shape.input_tokens_target <= original_shape.input_tokens_target);
    assert!(probe_shape.output_tokens_target <= original_shape.output_tokens_target);
    assert!(probe_shape.concurrency <= original_shape.concurrency);
    assert!(probe_shape.output_tokens_target >= 1);
}

#[test]
fn autotune_probe_skip_reason_flags_far_worse_cpu_fallback_probe() {
    let incumbent_metrics = AutotuneTrialMetrics {
        decode_tok_s: 100.0,
        mlx_metal_hot_path_cpu_fallback_free: true,
        real_model_forward: true,
        ..AutotuneTrialMetrics::default()
    };
    let observed_trials = vec![AutotuneTrialRecord {
        trial_index: 0,
        candidate: AutotuneCandidateConfig {
            max_batch_tokens: 128,
            kv_total_blocks: Some(32),
            prefix_cache: false,
        },
        selection: AutotuneSelectionDiagnostics {
            strategy: "base_config_seed".to_string(),
            predicted_mean: None,
            uncertainty: None,
            acquisition: None,
            good_density: None,
            bad_density: None,
            density_ratio: None,
            novelty_bonus: None,
        },
        probe: None,
        score: 1000.0,
        status: "ok".to_string(),
        result_dir: None,
        error: None,
        metrics: Some(incumbent_metrics),
    }];
    let execution = RuntimeResult {
        tool_mode: "engine_bringup_runtime",
        runtime: runtime_config_from_manifest(&load_test_manifest(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        ))
        .expect("runtime config should build"),
        observation: RuntimeObservation::default(),
        correctness: GateStatus::pass(),
        determinism: GateStatus::pass(),
    };
    let probe_metrics = AutotuneTrialMetrics {
        decode_tok_s: 40.0,
        prefix_cpu_reference_dispatch_count: 1,
        direct_decode_cpu_projection_row_count: 8,
        mlx_metal_hot_path_cpu_fallback_free: false,
        real_model_forward: true,
        ..AutotuneTrialMetrics::default()
    };
    let baseline = autotune_probe_baseline(&observed_trials);

    let reason = autotune_probe_skip_reason(&execution, &probe_metrics, baseline.as_ref(), 400.0);

    assert!(reason.is_some());
    let reason = reason.expect("skip reason should exist");
    assert!(
        reason.contains("CPU fallback")
            || reason.contains("fell well below incumbent")
            || reason.contains("adaptive floor"),
        "{reason}"
    );
}

#[test]
fn autotune_probe_baseline_uses_observed_trial_distribution() {
    let observed_trials = vec![
        AutotuneTrialRecord {
            trial_index: 0,
            candidate: AutotuneCandidateConfig {
                max_batch_tokens: 128,
                kv_total_blocks: Some(32),
                prefix_cache: false,
            },
            selection: AutotuneSelectionDiagnostics {
                strategy: "base_config_seed".to_string(),
                predicted_mean: None,
                uncertainty: None,
                acquisition: None,
                good_density: None,
                bad_density: None,
                density_ratio: None,
                novelty_bonus: None,
            },
            probe: None,
            score: 80.0,
            status: "ok".to_string(),
            result_dir: None,
            error: None,
            metrics: Some(AutotuneTrialMetrics {
                decode_tok_s: 90.0,
                mlx_metal_hot_path_cpu_fallback_free: true,
                ttft_ms: Some(12),
                ..AutotuneTrialMetrics::default()
            }),
        },
        AutotuneTrialRecord {
            trial_index: 1,
            candidate: AutotuneCandidateConfig {
                max_batch_tokens: 256,
                kv_total_blocks: Some(64),
                prefix_cache: true,
            },
            selection: AutotuneSelectionDiagnostics {
                strategy: "tpe_ratio".to_string(),
                predicted_mean: Some(0.9),
                uncertainty: Some(0.1),
                acquisition: Some(1.0),
                good_density: Some(0.6),
                bad_density: Some(0.2),
                density_ratio: Some(3.0),
                novelty_bonus: Some(0.2),
            },
            probe: None,
            score: 120.0,
            status: "ok".to_string(),
            result_dir: None,
            error: None,
            metrics: Some(AutotuneTrialMetrics {
                decode_tok_s: 110.0,
                mlx_metal_hot_path_cpu_fallback_free: true,
                ttft_ms: Some(10),
                ..AutotuneTrialMetrics::default()
            }),
        },
    ];

    let baseline = autotune_probe_baseline(&observed_trials).expect("baseline should exist");

    assert_eq!(baseline.observed_trial_count, 2);
    assert_eq!(baseline.incumbent_score, Some(120.0));
    assert_eq!(baseline.incumbent_decode_tok_s, Some(110.0));
    assert!(baseline.score_floor.is_some());
    assert!(baseline.decode_tok_s_floor.is_some());
    assert!(baseline.fallback_free_decode_tok_s_floor.is_some());
    assert!(baseline.ttft_ceiling_ms.is_some());
}

#[test]
fn usage_mentions_generate_and_stream_commands() {
    let text = usage();
    assert!(text.contains("ax-engine-bench generate"));
    assert!(text.contains("ax-engine-bench stream"));
    assert!(
        !text.contains("ax-engine-bench autotune"),
        "public usage should not advertise unfinished autotune commands"
    );
}

#[test]
fn handle_autotune_returns_contract_error_while_hidden_from_usage() {
    let error = handle_autotune(&[
        "--manifest".to_string(),
        "/tmp/manifest.json".to_string(),
        "--output-root".to_string(),
        "/tmp/out".to_string(),
    ])
    .expect_err("autotune should remain unavailable");

    let CliError::Contract(message) = error else {
        panic!("autotune should return a contract error");
    };
    assert!(message.contains("not yet available"));
}

#[test]
fn parse_inference_args_accepts_llama_cpp_alias() {
    let args = vec![
        "--prompt".to_string(),
        "hello cli".to_string(),
        "--support-tier".to_string(),
        "llama_cpp".to_string(),
        "--llama-server-url".to_string(),
        "http://127.0.0.1:8081".to_string(),
        "--json".to_string(),
    ];

    let parsed = parse_inference_args(&args, "generate").expect("inference args should parse");

    assert_eq!(parsed.input_text.as_deref(), Some("hello cli"));
    assert_eq!(parsed.support_tier, SupportTier::LlamaCpp);
    assert_eq!(
        parsed.llama_server_url.as_deref(),
        Some("http://127.0.0.1:8081")
    );
    assert!(parsed.json);
}

#[test]
fn parse_inference_args_accepts_mlx_lm_delegated_server_url() {
    let args = vec![
        "--prompt".to_string(),
        "hello mlx-lm".to_string(),
        "--support-tier".to_string(),
        "mlx_lm_delegated".to_string(),
        "--mlx-lm-server-url".to_string(),
        "http://127.0.0.1:8090".to_string(),
    ];

    let parsed = parse_inference_args(&args, "generate").expect("inference args should parse");

    assert_eq!(parsed.input_text.as_deref(), Some("hello mlx-lm"));
    assert_eq!(parsed.support_tier, SupportTier::MlxLmDelegated);
    assert_eq!(
        parsed.mlx_lm_server_url.as_deref(),
        Some("http://127.0.0.1:8090")
    );
}

#[test]
fn parse_inference_args_accepts_explicit_mlx_model_artifacts_dir() {
    let args = vec![
        "--tokens".to_string(),
        "1,2,3".to_string(),
        "--mlx-model-artifacts-dir".to_string(),
        "/tmp/ax-model".to_string(),
    ];

    let parsed = parse_inference_args(&args, "generate").expect("inference args should parse");

    assert_eq!(
        parsed.mlx_model_artifacts_dir,
        Some(PathBuf::from("/tmp/ax-model"))
    );
}

#[test]
fn parse_inference_args_accepts_ignore_eos_for_fixed_token_benchmarks() {
    let args = vec![
        "--tokens".to_string(),
        "1,2,3".to_string(),
        "--ignore-eos".to_string(),
    ];

    let parsed = parse_inference_args(&args, "generate").expect("inference args should parse");
    let request = parsed.generate_request();

    assert!(parsed.sampling.ignore_eos);
    assert!(request.sampling.ignore_eos);
}

#[test]
fn parse_inference_args_preserves_gemma4_multimodal_inputs_json() {
    let args = vec![
        "--tokens".to_string(),
        "10,258880,11".to_string(),
        "--multimodal-inputs-json".to_string(),
        json!({
            "gemma4_unified": {
                "images": [{
                    "span": {
                        "modality": "image",
                        "placeholder_index": 1,
                        "replacement_start": 1,
                        "soft_token_count": 1,
                        "replacement_token_count": 3
                    },
                    "pixel_values": [0.0, 1.0, 2.0],
                    "pixel_position_ids": [[0, 0]]
                }],
                "audios": [],
                "videos": []
            }
        })
        .to_string(),
    ];

    let parsed = parse_inference_args(&args, "generate").expect("inference args should parse");
    let request = parsed.generate_request();
    let inputs = request
        .multimodal_inputs
        .gemma4_unified
        .expect("Gemma4 multimodal inputs should be preserved");

    assert_eq!(request.input_tokens, vec![10, 258880, 11]);
    assert_eq!(inputs.images.len(), 1);
    assert_eq!(inputs.images[0].span.replacement_start, 1);
    assert_eq!(inputs.images[0].pixel_position_ids, vec![[0, 0]]);
}

#[test]
fn parse_inference_args_rejects_multimodal_inputs_without_tokens() {
    let args = vec![
        "--prompt".to_string(),
        "describe this".to_string(),
        "--multimodal-inputs-json".to_string(),
        json!({
            "gemma4_unified": {
                "images": [{
                    "span": {
                        "modality": "image",
                        "placeholder_index": 1,
                        "replacement_start": 1,
                        "soft_token_count": 1,
                        "replacement_token_count": 3
                    },
                    "pixel_values": [0.0, 1.0, 2.0],
                    "pixel_position_ids": [[0, 0]]
                }],
                "audios": [],
                "videos": []
            }
        })
        .to_string(),
    ];

    let error =
        parse_inference_args(&args, "generate").expect_err("text multimodal prompt should fail");
    assert!(
        error
            .to_string()
            .contains("multimodal inputs require --tokens"),
        "unexpected error: {error}"
    );
}

fn native_metrics_fixture(run_id: &str, ttft_ms: f64, decode_tok_s: f64) -> Value {
    json!({
        "schema_version": "ax.engine_bench.metrics.v1",
        "run_id": run_id,
        "status": "pass",
        "runtime": mlx_runtime_fixture(),
        "metrics": {
            "ttft_ms": ttft_ms,
            "prefill_tok_s": 2048.0,
            "decode_tok_s": decode_tok_s,
            "e2e_latency_ms": 12.0,
            "memory_peak_mb": 512.0,
            "cpu_time_per_token_us": 10.0,
            "runner_time_per_token_us": 8.0,
            "runner_time_share_pct": 70.0,
            "prefix_hit_rate": 0.0,
            "kv_block_usage": 24.0,
            "evictions": 0.0
        },
        "correctness": {
            "passed": true,
            "reason": null
        },
        "determinism": {
            "passed": true,
            "reason": null
        },
        "step_count": 4,
        "scheduled_tokens_per_step": 64.0,
        "memory_blocked_steps": 0,
        "memory_blocked_request_events": 0,
        "replay_status": "not_applicable",
        "churn_status": "pass"
    })
}

fn write_execution_artifact_fixture(
    root: &Path,
    run_dir_name: &str,
    run_id: &str,
    ttft_ms: f64,
    decode_tok_s: f64,
) -> PathBuf {
    let run_dir = root.join(run_dir_name);
    fs::create_dir_all(&run_dir).expect("fixture run dir should create");
    write_json_file(
        &run_dir.join("manifest.json"),
        &load_test_manifest_json(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        ),
    )
    .expect("fixture manifest should write");
    write_json_file(
        &run_dir.join("environment.json"),
        &native_environment_fixture(),
    )
    .expect("fixture environment should write");
    write_json_file(
        &run_dir.join("metrics.json"),
        &native_metrics_fixture(run_id, ttft_ms, decode_tok_s),
    )
    .expect("fixture metrics should write");
    write_json_file(
        &run_dir.join("routes.json"),
        &json!({
            "schema_version": "ax.engine_bench.routes.v1",
            "run_id": run_id,
            "runtime": {
                "selected_backend": "mlx",
                "support_tier": "mlx_preview",
                "resolution_policy": "mlx_only"
            },
            "route": {
                "execution_plan": "mixed_step_plans",
                "attention_route": "mixed_attention_routes",
                "kv_mode": "paged_metadata",
                "barrier_mode": "serial",
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "mode": "engine_bringup_runtime"
        }),
    )
    .expect("fixture routes should write");
    write_json_file(
        &run_dir.join("trace.json"),
        &json!({
            "schema_version": "ax.engine_bench.trace.v1",
            "run_id": run_id,
            "mode": "engine_bringup_runtime",
            "runtime": {
                "selected_backend": "mlx",
                "support_tier": "mlx_preview",
                "resolution_policy": "mlx_only"
            },
            "steps": []
        }),
    )
    .expect("fixture trace should write");
    fs::write(
        run_dir.join("summary.md"),
        format!("# Benchmark Run\n\n- run_id: `{run_id}`\n- selected_backend: `mlx`\n"),
    )
    .expect("fixture summary should write");
    run_dir
}

fn write_matrix_result_fixture(
    root: &Path,
    dir_name: &str,
    run_id: &str,
    members: &[Value],
) -> PathBuf {
    let result_dir = root.join(dir_name);
    fs::create_dir_all(&result_dir).expect("matrix result dir should create");
    write_json_file(
        &result_dir.join("matrix.json"),
        &json!({
            "schema_version": "ax.engine_bench.matrix_result.v1",
            "run_id": run_id,
            "id": "test_mlx_mlx_matrix",
            "class": "scenario_matrix",
            "status": "ok",
            "matrix_manifest_path": "/tmp/test-matrix.json",
            "matrix_manifest_fingerprint_fnv1a64": "abc123def4567890",
            "result_dir": result_dir.display().to_string(),
            "summary": {
                "member_count": members.len(),
                "ok_count": members.len(),
                "completed_with_failures_count": 0,
                "contract_failure_count": 0
            },
            "members": members
        }),
    )
    .expect("matrix result JSON should write");
    fs::write(
        result_dir.join("summary.md"),
        format!("# Benchmark Matrix\n\n- run_id: `{run_id}`\n"),
    )
    .expect("matrix summary should write");
    result_dir
}

fn route_decision(observation: &RuntimeObservation, key: &str) -> u32 {
    observation
        .route_metadata
        .crossover_decisions
        .iter()
        .find(|(decision, _)| decision == key)
        .map(|(_, value)| *value)
        .unwrap_or(0)
}

fn llama_cpp_scenario_manifest(server_url: &str) -> BenchmarkManifest {
    serde_json::from_value(json!({
        "schema_version": "ax.engine_bench.manifest.v1",
        "id": "llama_cpp_chat_qwen_short",
        "class": "scenario",
        "scenario": "chat",
        "model": {
            "family": "qwen3",
            "revision": "phase1-llama-cpp",
            "quant": "q4_k_m",
            "tokenizer_revision": "qwen-tokenizer-v1",
            "chat_template_revision": "chatml-v1"
        },
        "runtime": {
            "selected_backend": "llama_cpp",
            "support_tier": "llama_cpp",
            "resolution_policy": "allow_llama_cpp",
            "fallback_reason": "benchmark requested delegated runtime",
            "backend_adapter": {
                "kind": "llama_cpp_server_completion",
                "server_url": server_url
            },
            "deterministic": true,
            "max_batch_tokens": 2048,
            "flags": {
                "prefix_cache": false
            }
        },
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": 1234
        },
        "shape": {
            "input_tokens_target": 32,
            "output_tokens_target": 4,
            "concurrency": 1
        },
        "checks": {
            "expect_deterministic": true,
            "require_prefix_reuse": false
        }
    }))
    .expect("llama.cpp manifest should parse")
}

#[test]
fn scenario_sampling_manifest_preserves_ignore_eos() {
    let mut manifest = llama_cpp_scenario_manifest("http://127.0.0.1:8081");
    manifest.sampling.ignore_eos = true;

    let sampling = sampling_from_manifest(&manifest).expect("sampling should load");
    let specs = scenario_specs_from_manifest(&manifest).expect("scenario specs should build");
    let request = generate_request_from_spec(
        specs.first().expect("scenario should contain one spec"),
        &manifest,
    );

    assert!(sampling.ignore_eos);
    assert!(
        specs
            .first()
            .expect("scenario should contain one spec")
            .sampling_params
            .ignore_eos
    );
    assert!(request.sampling.ignore_eos);
}

fn llama_cpp_shared_prefix_scenario_manifest(server_url: &str) -> BenchmarkManifest {
    let mut manifest = llama_cpp_scenario_manifest(server_url);
    manifest.id = "llama_cpp_shared_prefix_qwen_short".to_string();
    manifest.scenario = "shared_prefix".to_string();
    manifest
        .shape
        .as_mut()
        .expect("shape should exist")
        .input_tokens_target = 128;
    manifest
        .shape
        .as_mut()
        .expect("shape should exist")
        .concurrency = 2;
    manifest.checks.require_prefix_reuse = true;
    manifest
}

fn llama_cpp_server_scenario_manifest(server_url: &str) -> BenchmarkManifest {
    let mut manifest = llama_cpp_scenario_manifest(server_url);
    manifest.id = "llama_cpp_chat_qwen_short".to_string();
    manifest.runtime.selected_backend = SelectedBackend::LlamaCpp;
    manifest.runtime.backend_adapter = Some(BackendAdapterManifest::LlamaCppServerCompletion {
        server_url: server_url.to_string(),
    });
    manifest.checks.expect_deterministic = false;
    manifest
}

fn write_named_autotune_history_fixture(
    output_root: &Path,
    dir_name: &str,
    manifest: &BenchmarkManifest,
    candidate: &AutotuneCandidateConfig,
    score: f64,
) -> PathBuf {
    let result_dir = output_root.join(dir_name);
    fs::create_dir_all(&result_dir).expect("historical autotune dir should create");
    write_json_file(
        &result_dir.join("autotune.json"),
        &json!({
            "schema_version": AUTOTUNE_SCHEMA_VERSION,
            "manifest_path": "/tmp/historical-manifest.json",
            "manifest_id": manifest.id,
            "selected_backend": selected_backend_label(manifest.runtime.selected_backend),
            "iterations_requested": 1,
            "iterations_completed": 1,
            "exploration_weight": 0.5,
            "warm_start_history": {
                "enabled": true,
                "loaded_trial_count": 0,
                "loaded_result_count": 0,
                "source_dirs": []
            },
            "search_space": {
                "max_batch_tokens": [candidate.max_batch_tokens],
                "kv_total_blocks": [candidate.kv_total_blocks],
                "prefix_cache": [candidate.prefix_cache],
                "candidate_count": 1
            },
            "trials": [{
                "label": "trial-001",
                "score": score,
                "status": "ok",
                "config": {
                    "max_batch_tokens": candidate.max_batch_tokens,
                    "kv_total_blocks": candidate.kv_total_blocks,
                    "prefix_cache": candidate.prefix_cache
                },
                "selection": {
                    "strategy": "base_config_seed",
                    "predicted_mean": null,
                    "uncertainty": null,
                    "acquisition": null,
                    "good_density": null,
                    "bad_density": null,
                    "density_ratio": null,
                    "novelty_bonus": null
                },
                "result_dir": null,
                "error": null,
                "metrics": {
                    "ttft_ms": 5,
                    "prefill_tok_s": 1000.0,
                    "decode_tok_s": 500.0,
                    "prefix_hit_rate": 0.0,
                    "prefix_native_dispatch_count": 0,
                    "prefix_cpu_reference_dispatch_count": 0,
                    "direct_decode_batched_group_fallback_count": 0,
                    "mlx_metal_hot_path_cpu_fallback_free": true,
                    "real_model_forward": true,
                    "model_bound_ffn_decode": true
                }
            }]
        }),
    )
    .expect("historical autotune JSON should write");
    result_dir
}

fn write_autotune_history_fixture(
    output_root: &Path,
    manifest: &BenchmarkManifest,
    candidate: &AutotuneCandidateConfig,
    score: f64,
) -> PathBuf {
    write_named_autotune_history_fixture(
        output_root,
        "historical-autotune",
        manifest,
        candidate,
        score,
    )
}

#[test]
fn load_autotune_warm_start_history_prefers_index_summary_when_present() {
    let root = unique_test_dir("autotune-history-index");
    let output_root = root.join("out");
    fs::create_dir_all(&output_root).expect("autotune output root should create");
    let manifest = load_test_manifest(
        repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
    );
    let search_space = resolve_autotune_search_space(
        &manifest,
        &AutotuneArgs {
            manifest_path: PathBuf::from("/tmp/manifest.json"),
            output_root: output_root.clone(),
            iterations: 1,
            exploration_weight: 0.5,
            max_batch_token_options: None,
            kv_total_block_options: None,
            prefix_cache_options: None,
            disable_history: false,
        },
    );
    let indexed_result_dir = output_root.join("indexed-result");
    write_json_file(
        &autotune_history_index_path(&output_root),
        &json!({
            "schema_version": AUTOTUNE_HISTORY_INDEX_SCHEMA_VERSION,
            "entry_count": 1,
            "summary_count": 1,
            "entries": [{
                "result_dir": output_root.join("stale-entry").display().to_string(),
                "manifest_id": "wrong-manifest",
                "selected_backend": "mlx",
                "trials": []
            }],
            "summaries": [{
                "manifest_id": manifest.id,
                "selected_backend": selected_backend_label(manifest.runtime.selected_backend),
                "source_dirs": [indexed_result_dir.display().to_string()],
                "best_trials": [{
                    "label": "trial-001",
                    "score": 77.0,
                    "status": "ok",
                    "config": {
                        "max_batch_tokens": manifest.runtime.max_batch_tokens,
                        "kv_total_blocks": manifest.runtime.kv_total_blocks,
                        "prefix_cache": manifest.runtime.flags.prefix_cache
                    },
                    "selection": {
                        "strategy": "base_config_seed",
                        "predicted_mean": null,
                        "uncertainty": null,
                        "acquisition": null,
                        "good_density": null,
                        "bad_density": null,
                        "density_ratio": null,
                        "novelty_bonus": null
                    },
                    "result_dir": indexed_result_dir.display().to_string(),
                    "error": null,
                    "metrics": {
                        "ttft_ms": 5,
                        "prefill_tok_s": 1000.0,
                        "decode_tok_s": 500.0,
                        "prefix_hit_rate": 0.0,
                        "prefix_native_dispatch_count": 0,
                        "prefix_cpu_reference_dispatch_count": 0,
                        "direct_decode_batched_group_fallback_count": 0,
                        "mlx_metal_hot_path_cpu_fallback_free": true,
                        "real_model_forward": true,
                        "model_bound_ffn_decode": true
                    }
                }]
            }]
        }),
    )
    .expect("history index should write");

    let history = load_autotune_warm_start_history(&output_root, &manifest, &search_space)
        .expect("warm-start history should load from index");

    assert_eq!(history.trials.len(), 1);
    assert_eq!(history.trials[0].score, 77.0);
    assert_eq!(history.source_dirs, vec![indexed_result_dir]);
}

#[test]
fn load_autotune_warm_start_history_falls_back_to_index_entries_when_summary_missing() {
    let root = unique_test_dir("autotune-history-index-entries");
    let output_root = root.join("out");
    fs::create_dir_all(&output_root).expect("autotune output root should create");
    let manifest = load_test_manifest(
        repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
    );
    let search_space = resolve_autotune_search_space(
        &manifest,
        &AutotuneArgs {
            manifest_path: PathBuf::from("/tmp/manifest.json"),
            output_root: output_root.clone(),
            iterations: 1,
            exploration_weight: 0.5,
            max_batch_token_options: None,
            kv_total_block_options: None,
            prefix_cache_options: None,
            disable_history: false,
        },
    );
    let indexed_result_dir = output_root.join("indexed-entry-result");
    write_json_file(
        &autotune_history_index_path(&output_root),
        &json!({
            "schema_version": AUTOTUNE_HISTORY_INDEX_SCHEMA_VERSION,
            "entry_count": 1,
            "entries": [{
                "result_dir": indexed_result_dir.display().to_string(),
                "manifest_id": manifest.id,
                "selected_backend": selected_backend_label(manifest.runtime.selected_backend),
                "trials": [{
                    "label": "trial-001",
                    "score": 66.0,
                    "status": "ok",
                    "config": {
                        "max_batch_tokens": manifest.runtime.max_batch_tokens,
                        "kv_total_blocks": manifest.runtime.kv_total_blocks,
                        "prefix_cache": manifest.runtime.flags.prefix_cache
                    },
                    "selection": {
                        "strategy": "base_config_seed",
                        "predicted_mean": null,
                        "uncertainty": null,
                        "acquisition": null,
                        "good_density": null,
                        "bad_density": null,
                        "density_ratio": null,
                        "novelty_bonus": null
                    },
                    "result_dir": indexed_result_dir.display().to_string(),
                    "error": null,
                    "metrics": {
                        "ttft_ms": 5,
                        "prefill_tok_s": 1000.0,
                        "decode_tok_s": 500.0,
                        "prefix_hit_rate": 0.0,
                        "prefix_native_dispatch_count": 0,
                        "prefix_cpu_reference_dispatch_count": 0,
                        "direct_decode_batched_group_fallback_count": 0,
                        "mlx_metal_hot_path_cpu_fallback_free": true,
                        "real_model_forward": true,
                        "model_bound_ffn_decode": true
                    }
                }]
            }]
        }),
    )
    .expect("history index should write");

    let history = load_autotune_warm_start_history(&output_root, &manifest, &search_space)
        .expect("warm-start history should load from entry fallback");

    assert_eq!(history.trials.len(), 1);
    assert_eq!(history.trials[0].score, 66.0);
    assert_eq!(history.source_dirs, vec![indexed_result_dir]);
}

#[test]
fn load_autotune_warm_start_history_falls_back_to_directory_scan_when_index_missing() {
    let root = unique_test_dir("autotune-history-scan");
    let output_root = root.join("out");
    fs::create_dir_all(&output_root).expect("autotune output root should create");
    let manifest = load_test_manifest(
        repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
    );
    let search_space = resolve_autotune_search_space(
        &manifest,
        &AutotuneArgs {
            manifest_path: PathBuf::from("/tmp/manifest.json"),
            output_root: output_root.clone(),
            iterations: 1,
            exploration_weight: 0.5,
            max_batch_token_options: None,
            kv_total_block_options: None,
            prefix_cache_options: None,
            disable_history: false,
        },
    );
    let result_dir = write_autotune_history_fixture(
        &output_root,
        &manifest,
        &AutotuneCandidateConfig {
            max_batch_tokens: manifest.runtime.max_batch_tokens,
            kv_total_blocks: manifest.runtime.kv_total_blocks,
            prefix_cache: manifest.runtime.flags.prefix_cache,
        },
        55.0,
    );

    let history = load_autotune_warm_start_history(&output_root, &manifest, &search_space)
        .expect("warm-start history should load from directory scan");

    assert_eq!(history.trials.len(), 1);
    assert_eq!(history.trials[0].score, 55.0);
    assert_eq!(history.source_dirs, vec![result_dir]);
}

#[test]
fn write_autotune_history_index_incremental_merges_new_result_into_existing_index() {
    let root = unique_test_dir("autotune-history-incremental");
    let output_root = root.join("out");
    fs::create_dir_all(&output_root).expect("autotune output root should create");
    let manifest = load_test_manifest(
        repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
    );
    let historical_dir = write_named_autotune_history_fixture(
        &output_root,
        "historical-autotune-a",
        &manifest,
        &AutotuneCandidateConfig {
            max_batch_tokens: manifest.runtime.max_batch_tokens,
            kv_total_blocks: manifest.runtime.kv_total_blocks,
            prefix_cache: manifest.runtime.flags.prefix_cache,
        },
        55.0,
    );
    write_autotune_history_index(&output_root).expect("history index should build");
    let new_dir = write_named_autotune_history_fixture(
        &output_root,
        "historical-autotune-b",
        &manifest,
        &AutotuneCandidateConfig {
            max_batch_tokens: manifest.runtime.max_batch_tokens.saturating_mul(2),
            kv_total_blocks: manifest.runtime.kv_total_blocks,
            prefix_cache: !manifest.runtime.flags.prefix_cache,
        },
        88.0,
    );
    let new_result_json =
        load_json_value(&new_dir.join("autotune.json")).expect("new autotune JSON should load");

    write_autotune_history_index_incremental(&output_root, &new_dir, &new_result_json)
        .expect("incremental history index should update");

    let history_index = load_json_value(&autotune_history_index_path(&output_root))
        .expect("history index should load");
    assert_eq!(
        history_index.get("entry_count").and_then(Value::as_u64),
        Some(2)
    );
    assert_eq!(
        history_index.get("summary_count").and_then(Value::as_u64),
        Some(1)
    );
    let source_dirs = history_index
        .get("summaries")
        .and_then(Value::as_array)
        .and_then(|summaries| summaries.first())
        .and_then(|summary| summary.get("source_dirs"))
        .and_then(Value::as_array)
        .expect("summary source dirs should exist")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    let historical_dir_string = historical_dir.display().to_string();
    let new_dir_string = new_dir.display().to_string();
    assert!(source_dirs.contains(&historical_dir_string.as_str()));
    assert!(source_dirs.contains(&new_dir_string.as_str()));
}

fn llama_cpp_replay_manifest(server_url: &str) -> BenchmarkManifest {
    serde_json::from_value(json!({
        "schema_version": "ax.engine_bench.manifest.v1",
        "id": "llama_cpp_replay_dual_cancel",
        "class": "replay",
        "scenario": "replay",
        "model": {
            "family": "qwen3",
            "revision": "phase1-llama-cpp",
            "quant": "q4_k_m",
            "tokenizer_revision": "qwen-tokenizer-v1",
            "chat_template_revision": "chatml-v1"
        },
        "runtime": {
            "selected_backend": "llama_cpp",
            "support_tier": "llama_cpp",
            "resolution_policy": "allow_llama_cpp",
            "fallback_reason": "benchmark requested delegated runtime",
            "backend_adapter": {
                "kind": "llama_cpp_server_completion",
                "server_url": server_url
            },
            "deterministic": true,
            "max_batch_tokens": 2048,
            "flags": {
                "prefix_cache": false
            }
        },
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": 1234
        },
        "events": [
            {
                "t_ms": 0,
                "type": "submit",
                "request_id": "req-1",
                "prompt_ref": "prompts/replay_req_a.txt",
                "output_tokens_target": 2
            },
            {
                "t_ms": 0,
                "type": "submit",
                "request_id": "req-2",
                "prompt_ref": "prompts/replay_req_b.txt",
                "output_tokens_target": 2
            },
            {
                "t_ms": 1,
                "type": "cancel",
                "request_id": "req-2"
            }
        ],
        "checks": {
            "expect_deterministic": true,
            "require_prefix_reuse": false
        }
    }))
    .expect("llama.cpp replay manifest should parse")
}

fn llama_cpp_server_replay_manifest(server_url: &str) -> BenchmarkManifest {
    let mut manifest = llama_cpp_replay_manifest(server_url);
    manifest.id = "llama_cpp_replay_dual_llama_cpp".to_string();
    manifest.runtime.selected_backend = SelectedBackend::LlamaCpp;
    manifest.runtime.backend_adapter = Some(BackendAdapterManifest::LlamaCppServerCompletion {
        server_url: server_url.to_string(),
    });
    manifest
}

fn llama_cpp_cli_manifest() -> BenchmarkManifest {
    serde_json::from_value(json!({
        "schema_version": "ax.engine_bench.manifest.v1",
        "id": "llama_cpp_cli_forbidden",
        "class": "scenario",
        "scenario": "chat",
        "model": {
            "family": "qwen3",
            "revision": "phase1-llama-cpp",
            "quant": "q4_k_m",
            "tokenizer_revision": "qwen-tokenizer-v1",
            "chat_template_revision": "chatml-v1"
        },
        "runtime": {
            "selected_backend": "llama_cpp",
            "support_tier": "llama_cpp",
            "resolution_policy": "allow_llama_cpp",
            "backend_adapter": {
                "kind": "llama_cpp_cli",
                "cli_path": "llama-cli",
                "model_path": "/tmp/model.gguf"
            },
            "deterministic": true,
            "max_batch_tokens": 2048,
            "flags": {
                "prefix_cache": false
            }
        },
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": 1234
        },
        "shape": {
            "input_tokens_target": 32,
            "output_tokens_target": 4,
            "concurrency": 1
        },
        "checks": {
            "expect_deterministic": true,
            "require_prefix_reuse": false
        }
    }))
    .expect("llama.cpp CLI manifest should parse")
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

fn spawn_single_json_completion_server(
    expected_path: &'static str,
    response_body: String,
    assert_request: impl Fn(&Value) + Send + 'static,
) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
    let address = listener.local_addr().expect("listener should have address");
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("request should arrive");
        let request = read_http_request(&mut stream);
        let request_text = String::from_utf8(request.clone()).expect("request should be utf8");
        assert!(
            request_text.starts_with(&format!("POST {expected_path} HTTP/1.1")),
            "unexpected request path: {request_text}"
        );

        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
            .expect("request should include header terminator");
        let body = String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
        let payload: Value = serde_json::from_str(&body).expect("request body should be json");
        assert_request(&payload);

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream
            .write_all(response.as_bytes())
            .expect("response should write");
    });

    (format!("http://{address}"), handle)
}

fn spawn_llama_cpp_completion_stream_server(
    expected_requests: usize,
    chunks: Vec<Value>,
    assert_request: impl Fn(&Value) + Send + 'static,
) -> (String, thread::JoinHandle<()>) {
    spawn_scripted_llama_cpp_completion_stream_server(expected_requests, move |_, payload| {
        assert_request(payload);
        chunks.clone()
    })
}

fn read_http_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1024];
    let mut header_end = None;
    let mut content_length = None;

    loop {
        let bytes_read = stream.read(&mut buffer).expect("request should read");
        assert!(
            bytes_read > 0,
            "client closed connection before request completed"
        );
        request.extend_from_slice(&buffer[..bytes_read]);

        if header_end.is_none() {
            header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4);
            if let Some(end) = header_end {
                let headers =
                    String::from_utf8(request[..end].to_vec()).expect("headers should be utf8");
                content_length = Some(parse_content_length(&headers));
            }
        }

        if let (Some(end), Some(length)) = (header_end, content_length) {
            if request.len() >= end + length {
                request.truncate(end + length);
                return request;
            }
        }
    }
}

fn parse_content_length(headers: &str) -> usize {
    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.eq_ignore_ascii_case("content-length") {
                Some(
                    value
                        .trim()
                        .parse::<usize>()
                        .expect("content-length should parse"),
                )
            } else {
                None
            }
        })
        .expect("content-length header should exist")
}

#[test]
fn route_aggregation_marks_mixed_step_labels_when_runner_routes_differ() {
    let mut observation = RuntimeObservation::default();

    observation.merge_route_metadata(&make_route_metadata(
        "phase1.qwen3.dense_prefill",
        "qwen3_prefill",
        "paged_metadata",
        "serial",
    ));
    observation.merge_route_metadata(&make_route_metadata(
        "phase1.qwen3.paged_decode",
        "qwen3_paged_decode",
        "paged_metadata",
        "serial",
    ));

    assert_eq!(
        observation.route_metadata.execution_plan.as_deref(),
        Some("mixed_step_plans")
    );
    assert_eq!(
        observation.route_metadata.attention_route.as_deref(),
        Some("mixed_attention_routes")
    );
    assert_eq!(
        observation.route_metadata.kv_mode.as_deref(),
        Some("paged_metadata")
    );
    assert_eq!(
        observation.route_metadata.barrier_mode.as_deref(),
        Some("serial")
    );
    assert_eq!(route_decision(&observation, "execution_plan_variants"), 2);
    assert_eq!(route_decision(&observation, "attention_route_variants"), 2);
    assert_eq!(route_decision(&observation, "kv_mode_variants"), 1);
    assert_eq!(route_decision(&observation, "barrier_mode_variants"), 1);
}

#[test]
fn route_aggregation_saturates_crossover_decisions_instead_of_overflowing() {
    let mut observation = RuntimeObservation::default();
    let mut first = make_route_metadata(
        "phase1.qwen3.dense_prefill",
        "qwen3_prefill",
        "paged_metadata",
        "serial",
    );
    first
        .crossover_decisions
        .push(("live_share_hits".to_string(), u32::MAX));
    observation.merge_route_metadata(&first);

    let mut second = make_route_metadata(
        "phase1.qwen3.paged_decode",
        "qwen3_paged_decode",
        "paged_metadata",
        "serial",
    );
    second
        .crossover_decisions
        .push(("live_share_hits".to_string(), 1));
    observation.merge_route_metadata(&second);

    assert_eq!(route_decision(&observation, "live_share_hits"), u32::MAX);
}

#[test]
fn route_decision_upsert_replaces_existing_value_and_removes_duplicates() {
    let mut decisions = vec![
        ("delegated_cached_tokens".to_string(), 32),
        ("other_counter".to_string(), 7),
        ("delegated_cached_tokens".to_string(), 64),
    ];

    upsert_route_decision(&mut decisions, "delegated_cached_tokens", 96);

    assert_eq!(
        decisions,
        vec![
            ("delegated_cached_tokens".to_string(), 96),
            ("other_counter".to_string(), 7),
        ]
    );
}

#[test]
fn deterministic_digest_ignores_route_telemetry_counters() {
    let first = json!({
        "step_count": 2,
        "route": {
            "crossover_decisions": {
                "ax_mlx_decode_steps": 4,
                "ax_mlx_decode_wall_us": 100,
                "ax_mlx_prefill_wall_us": 50
            }
        },
        "requests": [{"request_id": 1, "generated_tokens": [1, 2, 3]}]
    });
    let second = json!({
        "step_count": 2,
        "route": {
            "crossover_decisions": {
                "ax_mlx_decode_steps": 4,
                "ax_mlx_decode_wall_us": 999,
                "ax_mlx_prefill_wall_us": 777
            }
        },
        "requests": [{"request_id": 1, "generated_tokens": [1, 2, 3]}]
    });

    assert_eq!(
        deterministic_runtime_digest(&first),
        deterministic_runtime_digest(&second)
    );
}

#[test]
fn deterministic_digest_keeps_generated_token_differences() {
    let first = json!({
        "route": {"crossover_decisions": {"ax_mlx_decode_wall_us": 100}},
        "requests": [{"request_id": 1, "generated_tokens": [1, 2, 3]}]
    });
    let second = json!({
        "route": {"crossover_decisions": {"ax_mlx_decode_wall_us": 999}},
        "requests": [{"request_id": 1, "generated_tokens": [1, 2, 4]}]
    });

    assert_ne!(
        deterministic_runtime_digest(&first),
        deterministic_runtime_digest(&second)
    );
}

#[test]
fn decode_tok_s_returns_zero_without_runner_timing() {
    let observation = RuntimeObservation {
        decode_tokens: 4,
        decode_steps: 2,
        total_decode_runner_time_us: 0,
        ..RuntimeObservation::default()
    };

    assert_eq!(observation.decode_tok_s(), 0.0);
}

#[test]
fn decode_tok_s_uses_runner_timing_when_available() {
    let observation = RuntimeObservation {
        decode_tokens: 4,
        decode_steps: 2,
        total_decode_runner_time_us: 2_000,
        ..RuntimeObservation::default()
    };

    assert_eq!(observation.decode_tok_s(), 2_000.0);
}

#[test]
fn correctness_fails_finished_requests_with_zero_output_tokens() {
    let manifest = llama_cpp_scenario_manifest("http://127.0.0.1:1");
    let observation = RuntimeObservation {
        final_requests: vec![FinalRequestState {
            external_id: "req-1".to_string(),
            request_id: RequestId(1),
            state: "Finished".to_string(),
            processed_prompt_tokens: 32,
            generated_tokens: Vec::new(),
            cancel_requested: false,
            last_error: None,
        }],
        decode_tokens: 0,
        ..RuntimeObservation::default()
    };

    let status = evaluate_correctness(&manifest, &observation)
        .expect("correctness evaluation should not fail structurally");

    assert!(!status.passed);
    assert_eq!(
        status.reason.as_deref(),
        Some("one or more finished requests produced zero output tokens")
    );
}

#[test]
fn llama_cpp_scenario_executes_through_server_completion_adapter() {
    let placeholder_manifest = llama_cpp_scenario_manifest("http://127.0.0.1:1");
    let expected_prompt = scenario_specs_from_manifest(&placeholder_manifest)
        .expect("scenario specs should build")[0]
        .input_tokens
        .clone();
    let expected_prompt_for_request = expected_prompt.clone();
    let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
        2,
        vec![
            serde_json::json!({
                "content": "llama benchmark",
                "tokens": [91, 92],
                "stop": false
            }),
            serde_json::json!({
                "content": " output",
                "tokens": [93, 94],
                "stop": true,
                "stop_type": "limit"
            }),
        ],
        move |payload| {
            assert_eq!(
                payload.get("prompt"),
                Some(&json!(expected_prompt_for_request))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("n_predict"), Some(&Value::Number(4.into())));
        },
    );
    let manifest = llama_cpp_scenario_manifest(&server_url);

    let result = execute_manifest_runtime(&manifest).expect("llama.cpp scenario should execute");
    server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(result.tool_mode, "llama_cpp_stepwise_runtime");
    assert!(
        result.correctness.passed,
        "correctness reason: {:?}",
        result.correctness.reason
    );
    assert!(result.determinism.passed);
    assert_eq!(
        result.observation.final_requests[0].generated_tokens,
        vec![91, 92, 93, 94]
    );
    assert_eq!(
        result.observation.route_metadata.execution_plan.as_deref(),
        Some("llama_cpp.server_completion_stream")
    );
    assert_eq!(result.observation.replay_status(), "not_applicable");
    assert!(result.observation.e2e_latency_ms > 0);
    assert_eq!(result.observation.step_count, 2);
    assert_eq!(result.observation.decode_tokens, 4);
    assert_eq!(
        result.observation.prefill_tokens,
        expected_prompt.len() as u64
    );
    assert_eq!(result.observation.digest.get("ttft_ms"), Some(&Value::Null));
    assert_eq!(result.observation.step_trace.len(), 2);
    assert_eq!(result.observation.step_trace[0].scheduled_tokens, 2);

    let environment = build_environment_json(
        "llama-run",
        "scenario",
        &{
            let manifest_path =
                std::env::temp_dir().join("ax-engine-bench-llama-cpp-scenario-manifest.json");
            write_json_file(&manifest_path, &manifest).expect("llama.cpp manifest should write");
            manifest_path
        },
        Path::new("/tmp/ax-engine-bench-llama-cpp"),
        123,
        &result,
    )
    .expect("environment artifact should build");
    assert_eq!(
        environment
            .get("software")
            .and_then(|software| software.get("tool_mode"))
            .and_then(Value::as_str),
        Some("llama_cpp_stepwise_runtime")
    );
    assert_eq!(
        environment
            .get("runtime")
            .and_then(|runtime| runtime.get("backend_adapter"))
            .and_then(|adapter| adapter.get("kind"))
            .and_then(Value::as_str),
        Some("llama_cpp_server_completion")
    );
}

#[test]
fn llama_cpp_server_scenario_executes_through_blocking_runtime() {
    let (server_url, server_handle) =
        spawn_scripted_llama_cpp_completion_stream_server(1, |_request_index, payload| {
            assert_eq!(
                payload
                    .get("prompt")
                    .and_then(Value::as_array)
                    .map(Vec::len),
                Some(32)
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            vec![json!({
                "content": "llama benchmark output",
                "tokens": [11, 12, 13, 14],
                "stop": true,
                "stop_type": "limit"
            })]
        });
    let manifest = llama_cpp_server_scenario_manifest(&server_url);

    let result =
        execute_manifest_runtime(&manifest).expect("streaming llama.cpp scenario should execute");
    server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(result.tool_mode, "llama_cpp_stepwise_runtime");
    assert!(
        result.correctness.passed,
        "correctness reason: {:?}, route decisions: {:?}",
        result.correctness.reason, result.observation.route_metadata.crossover_decisions
    );
    assert_eq!(result.observation.step_count, 1);
    assert_eq!(result.observation.prefill_tokens, 32);
    assert_eq!(result.observation.decode_tokens, 4);
    assert_eq!(result.observation.step_trace.len(), 1);
    assert_eq!(
        result.observation.route_metadata.execution_plan.as_deref(),
        Some("llama_cpp.server_completion_stream")
    );
}

#[test]
fn generate_command_executes_server_backed_openai_compatible_prompt() {
    let (server_url, server_handle) = spawn_single_json_completion_server(
        "/completion",
        json!({
            "content": "llama::hello cli",
            "tokens": [4, 5, 6],
            "tokens_evaluated": 2,
            "tokens_cached": 0,
            "stop": true,
            "stop_type": "limit"
        })
        .to_string(),
        |payload| {
            assert_eq!(payload["prompt"], Value::String("hello cli".to_string()));
            assert_eq!(payload["stream"], Value::Bool(false));
        },
    );
    let args = parse_inference_args(
        &[
            "--prompt".to_string(),
            "hello cli".to_string(),
            "--support-tier".to_string(),
            "llama_cpp".to_string(),
            "--llama-server-url".to_string(),
            server_url.clone(),
        ],
        "generate",
    )
    .expect("generate args should parse");

    let response =
        run_inference_generate(&args).expect("llama.cpp generate command should succeed");
    server_handle.join().expect("server thread should finish");

    assert_eq!(response.output_text.as_deref(), Some("llama::hello cli"));
    assert_eq!(response.runtime.selected_backend, SelectedBackend::LlamaCpp);
    let rendered = render_generate_response(&response, false).expect("response should render");
    assert!(rendered.starts_with("llama::hello cli\n"));
    assert!(rendered.contains("status=finished"));
    assert!(rendered.contains("finish_reason=max_output_tokens"));
    assert!(rendered.contains("execution_plan=llama_cpp.server_completion"));
}

#[test]
fn stream_command_collects_server_backed_llama_cpp_events() {
    let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![
            json!({
                "content": "hello",
                "tokens": [4],
                "stop": false
            }),
            json!({
                "content": " world",
                "tokens": [5],
                "stop": true,
                "stop_type": "limit"
            }),
        ],
        |payload| {
            assert_eq!(payload.get("prompt"), Some(&json!([1, 2, 3])));
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
        },
    );
    let args = parse_inference_args(
        &[
            "--tokens".to_string(),
            "1,2,3".to_string(),
            "--support-tier".to_string(),
            "llama_cpp".to_string(),
            "--llama-server-url".to_string(),
            server_url.clone(),
        ],
        "stream",
    )
    .expect("stream args should parse");

    let events =
        collect_inference_stream_events(&args).expect("llama.cpp stream command should succeed");
    server_handle.join().expect("server thread should finish");

    assert_eq!(events.len(), 4);
    assert_eq!(events[0].event_name(), "request");
    assert_eq!(events[1].event_name(), "step");
    assert_eq!(events[2].event_name(), "step");
    assert_eq!(events[3].event_name(), "response");

    let step_rendered = render_stream_event(&events[2], false).expect("event should render");
    assert!(step_rendered.contains("delta_token_logprobs=[None]"));
    assert!(step_rendered.contains("delta_text=\" world\""));
    assert!(step_rendered.contains("finish_reason=max_output_tokens"));
    let rendered = render_stream_event(&events[3], false).expect("event should render");
    assert!(rendered.contains("finish_reason=max_output_tokens"));
    assert!(rendered.contains("output_token_logprobs=[None, None]"));
    assert!(rendered.contains("output_text=\"hello world\""));
}

#[test]
fn llama_cpp_server_replay_is_rejected_for_blocking_adapter() {
    let manifest = llama_cpp_server_replay_manifest("http://127.0.0.1:8081");

    let error = execute_manifest_runtime(&manifest)
        .expect_err("llama.cpp replay without a server should fail closed");
    assert!(format!("{error}").contains("llama.cpp"));
}

#[test]
fn llama_cpp_server_shared_prefix_is_rejected_for_blocking_adapter() {
    let mut manifest = llama_cpp_server_scenario_manifest("http://127.0.0.1:8081");
    manifest.scenario = "shared_prefix".to_string();
    manifest.checks.require_prefix_reuse = true;

    let error = execute_manifest_runtime(&manifest)
        .expect_err("llama.cpp shared-prefix without a server should fail closed");
    assert!(format!("{error}").contains("llama.cpp"));
}

#[test]
fn llama_cpp_replay_executes_through_shared_sdk_session() {
    let placeholder_manifest = llama_cpp_replay_manifest("http://127.0.0.1:1");
    let replay_events =
        replay_events_from_manifest(&placeholder_manifest).expect("replay events should build");
    let expected_specs = replay_events
        .iter()
        .filter_map(|event| match event {
            ReplayEvent::Submit { spec, .. } => Some(spec.clone()),
            ReplayEvent::Cancel { .. } => None,
        })
        .collect::<Vec<_>>();
    let expected_prompt_total = expected_specs
        .iter()
        .map(|spec| spec.input_tokens.len() as u64)
        .sum::<u64>();
    let expected_prompts = expected_specs
        .iter()
        .map(|spec| spec.input_tokens.clone())
        .collect::<Vec<_>>();
    let expected_prompts_for_request = expected_prompts.clone();
    let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
        expected_prompts.len() * 2,
        vec![
            serde_json::json!({
                "content": "llama replay",
                "tokens": [111],
                "stop": false
            }),
            serde_json::json!({
                "content": " done",
                "tokens": [112],
                "stop": true,
                "stop_type": "limit"
            }),
        ],
        move |payload| {
            let prompt = payload.get("prompt").expect("prompt should be present");
            assert!(
                expected_prompts_for_request
                    .iter()
                    .any(|candidate| prompt == &json!(candidate))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("n_predict"), Some(&Value::Number(2.into())));
        },
    );
    let manifest = llama_cpp_replay_manifest(&server_url);

    let result = execute_manifest_runtime(&manifest).expect("llama.cpp replay should execute");
    server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(result.tool_mode, "llama_cpp_stepwise_runtime");
    assert!(
        result.correctness.passed,
        "correctness reason: {:?}",
        result.correctness.reason
    );
    assert!(result.determinism.passed);
    assert_eq!(result.observation.final_requests.len(), 2);
    assert_eq!(
        result
            .observation
            .final_requests
            .iter()
            .map(|request| request.state.as_str())
            .collect::<Vec<_>>(),
        vec!["Finished", "Cancelled"]
    );
    assert_eq!(
        result
            .observation
            .final_requests
            .iter()
            .map(|request| request.generated_tokens.clone())
            .collect::<Vec<_>>(),
        vec![vec![111, 112], vec![111]]
    );
    assert_eq!(
        result
            .observation
            .final_requests
            .iter()
            .map(|request| request.cancel_requested)
            .collect::<Vec<_>>(),
        vec![false, true]
    );
    assert_eq!(
        result.observation.route_metadata.execution_plan.as_deref(),
        Some("llama_cpp.server_completion_stream")
    );
    assert_eq!(result.observation.replay_status(), "pass");
    assert_eq!(result.observation.step_count, 2);
    assert_eq!(result.observation.step_trace.len(), 2);
    assert_eq!(result.observation.decode_tokens, 3);
    assert_eq!(result.observation.prefill_tokens, expected_prompt_total);
    assert_eq!(result.observation.digest.get("ttft_ms"), Some(&Value::Null));
}

#[test]
fn llama_cpp_concurrent_scenario_executes_through_shared_sdk_session() {
    let mut manifest = llama_cpp_scenario_manifest("http://127.0.0.1:8081");
    manifest
        .shape
        .as_mut()
        .expect("shape should exist")
        .concurrency = 2;
    manifest.scenario = "concurrent".to_string();
    let expected_specs =
        scenario_specs_from_manifest(&manifest).expect("scenario specs should build");
    let expected_prompt_total = expected_specs
        .iter()
        .map(|spec| spec.input_tokens.len() as u64)
        .sum::<u64>();
    let expected_prompts = expected_specs
        .iter()
        .map(|spec| spec.input_tokens.clone())
        .collect::<Vec<_>>();
    let expected_prompts_for_request = expected_prompts.clone();
    let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
        expected_prompts.len() * 2,
        vec![
            serde_json::json!({
                "content": "llama concurrent",
                "tokens": [101, 102],
                "stop": false
            }),
            serde_json::json!({
                "content": " output",
                "tokens": [103, 104],
                "stop": true,
                "stop_type": "limit"
            }),
        ],
        move |payload| {
            let prompt = payload.get("prompt").expect("prompt should be present");
            assert!(
                expected_prompts_for_request
                    .iter()
                    .any(|candidate| prompt == &json!(candidate))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
        },
    );
    manifest.runtime.backend_adapter =
        Some(BackendAdapterManifest::LlamaCppServerCompletion { server_url });

    let result =
        execute_manifest_runtime(&manifest).expect("llama.cpp concurrent benchmark should execute");
    server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(result.tool_mode, "llama_cpp_stepwise_runtime");
    assert_eq!(result.observation.final_requests.len(), 2);
    assert_eq!(result.observation.step_count, 2);
    assert_eq!(result.observation.step_trace.len(), 2);
    assert_eq!(result.observation.decode_tokens, 8);
    assert_eq!(result.observation.prefill_tokens, expected_prompt_total);
    assert_eq!(
        result
            .observation
            .final_requests
            .iter()
            .map(|request| request.generated_tokens.clone())
            .collect::<Vec<_>>(),
        vec![vec![101, 102, 103, 104], vec![101, 102, 103, 104]]
    );
    assert_eq!(result.observation.digest.get("ttft_ms"), Some(&Value::Null));
}

#[test]
fn llama_cpp_shared_prefix_scenario_executes_with_delegated_prompt_cache_reuse() {
    let mut manifest = llama_cpp_shared_prefix_scenario_manifest("http://127.0.0.1:8081");
    let expected_specs =
        scenario_specs_from_manifest(&manifest).expect("scenario specs should build");
    assert_eq!(expected_specs.len(), 2);
    assert_eq!(
        expected_specs[0].input_tokens[..64],
        expected_specs[1].input_tokens[..64]
    );
    assert_ne!(
        expected_specs[0].input_tokens[64..],
        expected_specs[1].input_tokens[64..]
    );

    let expected_prompt_total = expected_specs
        .iter()
        .map(|spec| spec.input_tokens.len() as u64)
        .sum::<u64>();
    let expected_prompts = expected_specs
        .iter()
        .map(|spec| spec.input_tokens.clone())
        .collect::<Vec<_>>();
    let expected_prompts_for_request = expected_prompts.clone();
    let (server_url, server_handle) = spawn_scripted_llama_cpp_completion_stream_server(
        expected_prompts.len() * 2,
        move |request_index, payload| {
            let prompt = payload.get("prompt").expect("prompt should be present");
            assert!(
                expected_prompts_for_request
                    .iter()
                    .any(|candidate| prompt == &json!(candidate))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));

            let cache_tokens = if request_index % 2 == 0 { 0 } else { 64 };
            vec![
                serde_json::json!({
                    "content": "",
                    "tokens": [],
                    "stop": false,
                    "prompt_progress": {
                        "total": 128,
                        "cache": cache_tokens,
                        "processed": 128_u32.saturating_sub(cache_tokens),
                        "time_ms": 1.0
                    }
                }),
                serde_json::json!({
                    "content": "shared-prefix",
                    "tokens": [141, 142, 143, 144],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ]
        },
    );
    manifest.runtime.backend_adapter =
        Some(BackendAdapterManifest::LlamaCppServerCompletion { server_url });

    let result = execute_manifest_runtime(&manifest)
        .expect("llama.cpp shared-prefix scenario should execute");
    server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(result.tool_mode, "llama_cpp_stepwise_runtime");
    assert!(result.correctness.passed);
    assert!(result.determinism.passed);
    assert_eq!(result.observation.final_requests.len(), 2);
    assert_eq!(result.observation.step_count, 2);
    assert_eq!(result.observation.step_trace.len(), 2);
    assert_eq!(result.observation.prefill_tokens, expected_prompt_total);
    assert_eq!(result.observation.decode_tokens, 8);
    assert!(result.observation.prefix_hit_rate() > 0.0);
    assert_eq!(
        result
            .observation
            .route_metadata
            .prefix_cache_path
            .as_deref(),
        Some("delegated_prompt_cache")
    );
    assert_eq!(
        route_decision(&result.observation, "delegated_cached_tokens"),
        64
    );
    assert_eq!(result.observation.digest.get("ttft_ms"), Some(&Value::Null));

    let expected_prompts_for_artifacts = expected_prompts.clone();
    let (artifact_server_url, artifact_server_handle) =
        spawn_scripted_llama_cpp_completion_stream_server(
            expected_prompts.len() * 2,
            move |request_index, payload| {
                let prompt = payload.get("prompt").expect("prompt should be present");
                assert!(
                    expected_prompts_for_artifacts
                        .iter()
                        .any(|candidate| prompt == &json!(candidate))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));

                let cache_tokens = if request_index % 2 == 0 { 0 } else { 64 };
                vec![
                    serde_json::json!({
                        "content": "",
                        "tokens": [],
                        "stop": false,
                        "prompt_progress": {
                            "total": 128,
                            "cache": cache_tokens,
                            "processed": 128_u32.saturating_sub(cache_tokens),
                            "time_ms": 1.0
                        }
                    }),
                    serde_json::json!({
                        "content": "shared-prefix",
                        "tokens": [141, 142, 143, 144],
                        "stop": true,
                        "stop_type": "limit"
                    }),
                ]
            },
        );
    let root = unique_test_dir("llama-cpp-shared-prefix-success");
    let output_root = root.join("results");
    fs::create_dir_all(&output_root).expect("output root should create");
    manifest.runtime.backend_adapter = Some(BackendAdapterManifest::LlamaCppServerCompletion {
        server_url: artifact_server_url,
    });
    let manifest_path = root.join("llama-cpp-shared-prefix.json");
    write_json_file(&manifest_path, &manifest).expect("manifest should write");
    let args = vec![
        "--manifest".to_string(),
        manifest_path.display().to_string(),
        "--output-root".to_string(),
        output_root.display().to_string(),
    ];
    handle_scenario(&args).expect("shared-prefix scenario should write success artifacts");
    artifact_server_handle
        .join()
        .expect("artifact server thread should finish");
    let runs = fs::read_dir(&output_root)
        .expect("output root should be readable")
        .map(|entry| entry.expect("directory entry should read").path())
        .collect::<Vec<_>>();
    assert_eq!(runs.len(), 1);
    let run_dir = &runs[0];
    assert!(run_dir.join("environment.json").is_file());
    assert!(run_dir.join("metrics.json").is_file());
    assert!(run_dir.join("routes.json").is_file());
    assert!(!run_dir.join("contract_failure.json").exists());

    let environment = load_json_value(&run_dir.join("environment.json"))
        .expect("environment artifact should load");
    assert_eq!(
        environment
            .get("route")
            .and_then(|route| route.get("prefix_cache_path"))
            .and_then(Value::as_str),
        Some("delegated_prompt_cache")
    );
    assert_eq!(
        environment
            .get("route")
            .and_then(|route| route.get("prefix_reuse_provenance"))
            .and_then(Value::as_str),
        Some("delegated_backend_prompt_cache")
    );
    assert_eq!(
        environment
            .get("route")
            .and_then(|route| route.get("backend_reported_cached_prompt_tokens"))
            .and_then(Value::as_u64),
        Some(64)
    );

    fs::remove_dir_all(&root).expect("test directory should clean up");
}

#[test]
fn contract_failure_json_classifies_missing_backend_adapter() {
    let mut manifest = llama_cpp_scenario_manifest("http://127.0.0.1:8081");
    manifest.runtime.backend_adapter = None;
    let artifact = build_contract_failure_json(
        "run-1",
        "scenario",
        Path::new("/tmp/manifest.json"),
        &manifest,
        Path::new("/tmp/results"),
        123,
        "runtime.backend_adapter is required when selected_backend is llama_cpp",
    )
    .expect("contract failure artifact should build");

    assert_eq!(
        artifact.get("tool_mode").and_then(Value::as_str),
        Some("llama_cpp_blocking_runtime")
    );
    assert_eq!(
        artifact
            .get("failure")
            .and_then(|failure| failure.get("code"))
            .and_then(Value::as_str),
        Some("llama_backend_adapter_required")
    );
}

#[test]
fn build_session_preserves_llama_runtime_fields_via_sdk_factory() {
    let server_url = "http://127.0.0.1:8081";
    let mut manifest = llama_cpp_scenario_manifest(server_url);
    manifest.runtime.deterministic = false;
    manifest.runtime.max_batch_tokens = 4096;
    manifest.runtime.kv_total_blocks = Some(42);
    let expected_runtime = manifest.runtime.clone();

    let runtime = runtime_config_from_manifest(&manifest).expect("runtime config should load");
    let specs = scenario_specs_from_manifest(&manifest).expect("scenario specs should build");

    let config = session_config_from_runtime(&runtime, &specs);

    assert_eq!(config.kv_config.cache_group_id, CacheGroupId(0));
    assert_eq!(config.kv_config.block_size_tokens, 16);
    assert_eq!(config.kv_config.total_blocks, 42);
    assert!(!config.deterministic);
    assert_eq!(config.max_batch_tokens, 4096);
    assert_eq!(
        config.backend_policy,
        BackendPolicy::new(expected_runtime.resolution_policy)
    );
    assert_eq!(
        config.resolved_backend,
        ResolvedBackend::new(
            expected_runtime.selected_backend,
            expected_runtime.support_tier,
            expected_runtime.fallback_reason.clone(),
        )
    );
    assert_eq!(
        config.llama_backend,
        expected_runtime
            .backend_adapter
            .as_ref()
            .map(BackendAdapterManifest::to_sdk_config)
    );
}

#[test]
fn build_session_preserves_mlx_runtime_artifact_defaults_via_sdk_factory() {
    let manifest = load_test_manifest(
        repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
    );
    let runtime = runtime_config_from_manifest(&manifest).expect("runtime config should load");
    let specs = scenario_specs_from_manifest(&manifest).expect("scenario specs should build");
    let expected_default = EngineSessionConfig::default();

    let config = session_config_from_runtime(&runtime, &specs);

    assert_eq!(
        config.mlx_runtime_artifacts_dir,
        expected_default.mlx_runtime_artifacts_dir
    );
    assert_eq!(
        config.mlx_runtime_artifacts_source,
        expected_default.mlx_runtime_artifacts_source()
    );
    assert_eq!(
        config.mlx_model_artifacts_dir,
        manifest.runtime.mlx_model_artifacts_dir
    );
    assert_eq!(
        config.mlx_model_artifacts_source,
        Some(NativeModelArtifactsSource::ExplicitConfig)
    );
    assert_eq!(config.backend_policy, runtime.backend_policy);
    assert_eq!(config.resolved_backend, runtime.resolved_backend);
    assert_eq!(config.llama_backend, None);
}

#[test]
fn build_session_preserves_explicit_mlx_model_artifact_manifest_override() {
    let mut manifest = load_test_manifest(
        repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
    );
    manifest.runtime.mlx_model_artifacts_dir = Some(PathBuf::from("/tmp/ax-mlx-model"));

    let runtime = runtime_config_from_manifest(&manifest).expect("runtime config should load");
    let specs = scenario_specs_from_manifest(&manifest).expect("scenario specs should build");

    let config = session_config_from_runtime(&runtime, &specs);

    let expected_mlx_runtime_artifacts_dir =
        EngineSessionConfig::default_mlx_runtime_artifacts_dir().or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|dir| dir.join("build/metal"))
                .filter(|dir| dir.join("build_report.json").is_file())
        });
    assert_eq!(
        config.mlx_runtime_artifacts_dir,
        expected_mlx_runtime_artifacts_dir
    );
    assert_eq!(
        config.mlx_runtime_artifacts_source,
        config
            .mlx_runtime_artifacts_dir
            .as_ref()
            .map(|_| NativeRuntimeArtifactsSource::RepoAutoDetect)
    );
    assert_eq!(
        config.mlx_model_artifacts_dir,
        Some(PathBuf::from("/tmp/ax-mlx-model"))
    );
    assert_eq!(
        config.mlx_model_artifacts_source,
        Some(NativeModelArtifactsSource::ExplicitConfig)
    );
}

#[test]
fn build_session_preserves_gguf_model_override_as_explicit_source() {
    let mut manifest = load_test_manifest(
        repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
    );
    manifest.runtime.mlx_model_artifacts_dir =
        Some(PathBuf::from("/tmp/google_gemma-4-26b-it-q4_k_m.gguf"));

    let runtime = runtime_config_from_manifest(&manifest).expect("runtime config should load");
    let specs = scenario_specs_from_manifest(&manifest).expect("scenario specs should build");

    let config = session_config_from_runtime(&runtime, &specs);

    assert_eq!(
        config.mlx_model_artifacts_dir,
        Some(PathBuf::from("/tmp/google_gemma-4-26b-it-q4_k_m.gguf"))
    );
    assert_eq!(
        config.mlx_model_artifacts_source,
        Some(NativeModelArtifactsSource::ExplicitConfig)
    );
}

#[test]
fn load_manifest_resolves_native_artifact_paths_relative_to_manifest_dir() {
    let root = unique_test_dir("manifest-relative-mlx-paths");
    let manifest_dir = root.join("benchmarks/manifests/scenario");
    fs::create_dir_all(&manifest_dir).expect("manifest dir should create");
    let manifest_path = manifest_dir.join("relative-native.json");
    fs::write(
        &manifest_path,
        r#"{
  "schema_version": "ax.engine_bench.manifest.v1",
  "id": "relative_native",
  "class": "scenario",
  "scenario": "chat",
  "model": {
    "family": "qwen3",
    "revision": "phase1-canonical",
    "quant": "q4_k_m",
    "tokenizer_revision": "qwen-tokenizer-v1",
    "chat_template_revision": "chatml-v1"
  },
  "runtime": {
    "selected_backend": "mlx",
    "support_tier": "mlx_preview",
    "resolution_policy": "mlx_only",
    "deterministic": true,
    "max_batch_tokens": 2048,
    "mlx_model_artifacts_dir": "../../../models/qwen-mlx",
    "flags": {
      "prefix_cache": false
    }
  },
  "sampling": {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 1234
  },
  "shape": {
    "input_tokens_target": 64,
    "output_tokens_target": 32,
    "concurrency": 1
  }
}"#,
    )
    .expect("manifest should write");

    let manifest = load_manifest(&manifest_path).expect("manifest should load");

    assert_eq!(
        manifest.runtime.mlx_model_artifacts_dir,
        Some(root.join("models/qwen-mlx"))
    );

    fs::remove_dir_all(&root).expect("test dir should clean up");
}

#[test]
fn load_manifest_expands_native_artifact_env_paths() {
    let Some(home_dir) = env::var_os("HOME").map(PathBuf::from) else {
        return;
    };
    let root = unique_test_dir("manifest-env-mlx-paths");
    fs::create_dir_all(&root).expect("test dir should create");
    let manifest_path = root.join("env-native.json");
    fs::write(
        &manifest_path,
        r#"{
  "schema_version": "ax.engine_bench.manifest.v1",
  "id": "env_mlx",
  "class": "scenario",
  "scenario": "chat",
  "model": {
    "family": "qwen3",
    "revision": "phase1-canonical",
    "quant": "q4_k_m",
    "tokenizer_revision": "qwen-tokenizer-v1",
    "chat_template_revision": "chatml-v1"
  },
  "runtime": {
    "selected_backend": "mlx",
    "support_tier": "mlx_preview",
    "resolution_policy": "mlx_only",
    "deterministic": true,
    "max_batch_tokens": 2048,
    "mlx_model_artifacts_dir": "$HOME/ax-mlx-model",
    "flags": {
      "prefix_cache": false
    }
  },
  "sampling": {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 1234
  },
  "shape": {
    "input_tokens_target": 64,
    "output_tokens_target": 32,
    "concurrency": 1
  }
}"#,
    )
    .expect("manifest should write");

    let manifest = load_manifest(&manifest_path).expect("manifest should load");

    assert_eq!(
        manifest.runtime.mlx_model_artifacts_dir,
        Some(home_dir.join("ax-mlx-model"))
    );

    fs::remove_dir_all(&root).expect("test dir should clean up");
}

#[test]
fn llama_cpp_replay_executes_with_delegated_prompt_cache_reuse() {
    let mut manifest = load_test_manifest(
        repo_manifest_path("benchmarks/manifests/replay/llama_cpp_prompt_cache_reuse_dual.json")
            .as_str(),
    );

    let replay_events = replay_events_from_manifest(&manifest).expect("replay events should build");
    let expected_prompts = replay_events
        .iter()
        .filter_map(|event| match event {
            ReplayEvent::Submit { spec, .. } => Some(spec.input_tokens.clone()),
            ReplayEvent::Cancel { .. } => None,
        })
        .collect::<Vec<_>>();
    let expected_prompts_for_request = expected_prompts.clone();
    let (server_url, server_handle) =
        spawn_scripted_llama_cpp_completion_stream_server(4, move |request_index, payload| {
            let prompt = payload.get("prompt").expect("prompt should be present");
            assert!(
                expected_prompts_for_request
                    .iter()
                    .any(|candidate| prompt == &json!(candidate))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));

            if request_index % 2 == 0 {
                vec![
                    serde_json::json!({
                        "content": "",
                        "tokens": [],
                        "stop": false,
                        "prompt_progress": {
                            "total": 32,
                            "cache": 0,
                            "processed": 32,
                            "time_ms": 1.0
                        }
                    }),
                    serde_json::json!({
                        "content": "seed",
                        "tokens": [121],
                        "stop": true,
                        "stop_type": "limit"
                    }),
                ]
            } else {
                vec![
                    serde_json::json!({
                        "content": "",
                        "tokens": [],
                        "stop": false,
                        "prompt_progress": {
                            "total": 32,
                            "cache": 24,
                            "processed": 24,
                            "time_ms": 1.0
                        }
                    }),
                    serde_json::json!({
                        "content": "reuse",
                        "tokens": [122],
                        "stop": true,
                        "stop_type": "limit"
                    }),
                ]
            }
        });
    manifest.runtime.backend_adapter =
        Some(BackendAdapterManifest::LlamaCppServerCompletion { server_url });

    let result =
        execute_manifest_runtime(&manifest).expect("llama.cpp prefix-reuse replay should execute");
    server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert!(
        result.correctness.passed,
        "correctness reason: {:?}",
        result.correctness.reason
    );
    assert!(result.determinism.passed);
    assert!(result.observation.prefix_hit_rate() > 0.0);
    assert_eq!(
        result
            .observation
            .route_metadata
            .prefix_cache_path
            .as_deref(),
        Some("delegated_prompt_cache")
    );
    assert!(route_decision(&result.observation, "delegated_cached_tokens") > 0);
}

#[test]
fn llama_cpp_benchmark_rejects_cli_adapter_for_token_workloads() {
    let manifest = llama_cpp_cli_manifest();

    let error = execute_manifest_runtime(&manifest)
        .expect_err("llama.cpp CLI benchmark should fail closed");
    let CliError::Contract(message) = error else {
        panic!("llama.cpp CLI benchmark should return a contract error");
    };

    assert!(message.contains("llama.cpp"));
}

#[test]
fn regression_artifact_records_prefix_reuse_contract() {
    let baseline_metrics = json!({
        "run_id": "baseline-run",
        "metrics": {
            "ttft_ms": 10.0,
            "decode_tok_s": 100.0,
            "memory_peak_mb": 512.0,
            "prefix_hit_rate": 50.0
        }
    });
    let candidate_metrics = json!({
        "run_id": "candidate-run",
        "metrics": {
            "ttft_ms": 8.0,
            "decode_tok_s": 110.0,
            "memory_peak_mb": 500.0,
            "prefix_hit_rate": 55.0
        }
    });
    let mut baseline_runtime = llama_runtime_fixture("http://127.0.0.1:8081");
    baseline_runtime["max_batch_tokens"] = json!(2048);
    baseline_runtime["kv_total_blocks"] = json!(512);
    baseline_runtime["flags"] = json!({ "prefix_cache": false });
    let mut candidate_runtime = llama_runtime_fixture("http://127.0.0.1:8081");
    candidate_runtime["max_batch_tokens"] = json!(4096);
    candidate_runtime["kv_total_blocks"] = json!(1024);
    candidate_runtime["flags"] = json!({ "prefix_cache": true });
    let baseline_environment = json!({
        "software": {
            "tool_mode": "llama_cpp_stepwise_runtime"
        },
        "runtime": baseline_runtime,
        "route": {
            "prefix_cache_path": "delegated_prompt_cache",
            "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
            "prefix_reuse_provenance": "delegated_backend_prompt_cache",
            "backend_reported_cached_prompt_tokens": 96
        }
    });
    let candidate_environment = json!({
        "software": {
            "tool_mode": "llama_cpp_stepwise_runtime"
        },
        "runtime": candidate_runtime,
        "route": {
            "prefix_cache_path": "delegated_prompt_cache",
            "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
            "prefix_reuse_provenance": "delegated_backend_prompt_cache",
            "backend_reported_cached_prompt_tokens": 96
        }
    });

    let regression = build_regression_json(
        &baseline_metrics,
        &candidate_metrics,
        &baseline_environment,
        &candidate_environment,
        None,
    )
    .expect("regression artifact should build");

    assert_eq!(
        regression
            .get("summary")
            .and_then(|summary| summary.get("result"))
            .and_then(Value::as_str),
        Some("llama_cpp_stepwise_compare")
    );
    assert_eq!(
        regression
            .get("runtime")
            .and_then(|runtime| runtime.get("tool_mode"))
            .and_then(Value::as_str),
        Some("llama_cpp_stepwise_runtime")
    );
    assert_eq!(
        regression
            .get("runtime")
            .and_then(|runtime| runtime.get("selected_backend"))
            .and_then(Value::as_str),
        Some("llama_cpp")
    );
    assert_eq!(
        regression
            .get("summary")
            .and_then(|summary| summary.get("execution_semantics"))
            .and_then(Value::as_str),
        Some("delegated_backend_runtime")
    );
    assert_eq!(
        regression
            .get("summary")
            .and_then(|summary| summary.get("prefix_cache_path"))
            .and_then(Value::as_str),
        Some("delegated_prompt_cache")
    );
    assert_eq!(
        regression
            .get("summary")
            .and_then(|summary| summary.get("prefix_cache_evidence"))
            .and_then(Value::as_str),
        Some("backend_reported_cached_prompt_tokens")
    );
    assert_eq!(
        regression
            .get("summary")
            .and_then(|summary| summary.get("prefix_reuse_provenance"))
            .and_then(Value::as_str),
        Some("delegated_backend_prompt_cache")
    );
    assert_eq!(
        regression
            .get("summary")
            .and_then(|summary| summary.get("backend_reported_cached_prompt_tokens"))
            .and_then(Value::as_u64),
        Some(96)
    );
    assert_eq!(
        regression
            .get("contract")
            .and_then(|contract| contract.get("tool_mode"))
            .and_then(|field| field.get("baseline"))
            .and_then(Value::as_str),
        Some("llama_cpp_stepwise_runtime")
    );
    assert_eq!(
        regression
            .get("contract")
            .and_then(|contract| contract.get("runtime"))
            .and_then(|runtime| runtime.get("selected_backend"))
            .and_then(|field| field.get("baseline"))
            .and_then(Value::as_str),
        Some("llama_cpp")
    );
    assert_eq!(
        nested_value(
            &regression,
            &[
                "informational_diff",
                "runtime_tuning",
                "max_batch_tokens",
                "changed"
            ]
        )
        .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        nested_value(
            &regression,
            &[
                "informational_diff",
                "runtime_tuning",
                "prefix_cache",
                "candidate"
            ]
        )
        .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        regression
            .get("contract")
            .and_then(|contract| contract.get("prefix_cache_evidence"))
            .and_then(|field| field.get("baseline"))
            .and_then(Value::as_str),
        Some("backend_reported_cached_prompt_tokens")
    );
    assert_eq!(
        regression
            .get("contract")
            .and_then(|contract| contract.get("prefix_reuse_provenance"))
            .and_then(|field| field.get("baseline"))
            .and_then(Value::as_str),
        Some("delegated_backend_prompt_cache")
    );
    assert_eq!(
        regression
            .get("contract")
            .and_then(|contract| contract.get("backend_reported_cached_prompt_tokens"))
            .and_then(|field| field.get("baseline"))
            .and_then(Value::as_u64),
        Some(96)
    );
    assert_eq!(
        regression
            .get("readiness")
            .and_then(|readiness| readiness.get("baseline"))
            .and_then(|entry| entry.get("status"))
            .and_then(Value::as_str),
        Some("delegated_runtime")
    );
}

#[test]
fn compare_summary_uses_regression_prefix_reuse_provenance() {
    let baseline_metrics = json!({
        "run_id": "baseline-run"
    });
    let candidate_metrics = json!({
        "run_id": "candidate-run"
    });
    let regression = json!({
        "runtime": {
            "tool_mode": "llama_cpp_stepwise_runtime",
            "backend_adapter": {
                "kind": "llama_cpp_server_completion",
                "server_url": "http://127.0.0.1:8081"
            }
        },
        "comparison": {
            "ttft_ms_pct": -20.0,
            "decode_tok_s_pct": 10.0,
            "memory_peak_mb_pct": -2.0,
            "prefix_hit_rate_pct": 5.0
        },
        "summary": {
            "result": "llama_cpp_stepwise_compare",
            "tool_mode": "llama_cpp_stepwise_runtime",
            "selected_backend": "llama_cpp",
            "support_tier": "llama_cpp",
            "resolution_policy": "allow_llama_cpp",
            "prefix_cache_path": "delegated_prompt_cache",
            "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
            "prefix_reuse_provenance": "delegated_backend_prompt_cache",
            "backend_reported_cached_prompt_tokens": 144
        },
        "readiness": {
            "baseline": {
                "status": "delegated_runtime",
                "hot_path_cpu_fallback_free": false,
                "batched_direct_decode_logits_ready": false,
                "prefix_min_mlx_metal_dispatch_share": null,
                "direct_decode_min_mlx_metal_dispatch_share": null,
                "blockers": ["delegated_runtime_not_mlx"]
            },
            "candidate": {
                "status": "delegated_runtime",
                "hot_path_cpu_fallback_free": false,
                "batched_direct_decode_logits_ready": false,
                "prefix_min_mlx_metal_dispatch_share": null,
                "direct_decode_min_mlx_metal_dispatch_share": null,
                "blockers": ["delegated_runtime_not_mlx"]
            }
        }
    });

    let summary =
        build_compare_summary_markdown(&baseline_metrics, &candidate_metrics, &regression, None);

    assert!(summary.contains("delegated_backend_prompt_cache"));
    assert!(summary.contains("llama_cpp_stepwise_compare"));
    assert!(summary.contains("llama_cpp_stepwise_runtime"));
    assert!(summary.contains("selected_backend: `llama_cpp`"));
    assert!(summary.contains("backend_adapter: `{\"kind\":\"llama_cpp_server_completion\",\"server_url\":\"http://127.0.0.1:8081\"}`"));
    assert!(summary.contains("execution_semantics: `unknown`"));
    assert!(summary.contains("prefix_cache_evidence: `backend_reported_cached_prompt_tokens`"));
    assert!(summary.contains("backend_reported_cached_prompt_tokens: `144`"));
    assert!(summary.contains("baseline_mlx_metal_readiness: `delegated_runtime`"));
    assert!(
        summary
            .contains("candidate_mlx_metal_readiness_blockers: `[\"delegated_runtime_not_mlx\"]`")
    );
    assert!(!summary.contains("prefix_reuse_provenance: `unknown`"));
    assert!(!summary.contains("mode: `engine_bringup`"));
}

#[test]
fn regression_artifact_records_mlx_metal_readiness() {
    let baseline_metrics = json!({
        "run_id": "baseline-run",
        "metrics": {
            "ttft_ms": 10.0,
            "decode_tok_s": 100.0,
            "memory_peak_mb": 512.0,
            "prefix_hit_rate": 50.0
        }
    });
    let candidate_metrics = json!({
        "run_id": "candidate-run",
        "metrics": {
            "ttft_ms": 9.0,
            "decode_tok_s": 110.0,
            "memory_peak_mb": 500.0,
            "prefix_hit_rate": 55.0
        }
    });
    let baseline_environment = json!({
        "software": {
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_real_model_forward",
            "metal_direct_decode_batching_opportunity_observed": true,
            "metal_complete_model_forward_supported": true,
            "metal_real_model_forward": true,
            "metal_model_artifacts_validated": true,
            "mlx_metal_prefix_layers_attention": true,
            "metal_prefix_layers_cpu_reference": false,
            "metal_prefix_cpu_reference_dispatch_count": 0,
            "mlx_metal_prefix_projection_row_count": 2048,
            "metal_prefix_cpu_projection_row_count": 0,
            "mlx_metal_prefix_rms_norm_element_count": 1024,
            "metal_prefix_cpu_rms_norm_element_count": 0,
            "mlx_metal_prefix_ffn_activation_element_count": 1024,
            "metal_prefix_cpu_ffn_activation_element_count": 0,
            "metal_direct_decode_tokens": true,
            "metal_direct_decode_model_bound_ffn": true,
            "metal_direct_decode_batched_logits_group_count": 2,
            "metal_direct_decode_batched_logits_token_count": 2,
            "metal_direct_decode_batched_group_fallback_count": 0,
            "metal_direct_decode_batched_group_fallback_token_count": 0,
            "mlx_metal_direct_decode_projection_row_count": 4096,
            "metal_direct_decode_cpu_projection_row_count": 0,
            "mlx_metal_direct_decode_rms_norm_element_count": 1024,
            "metal_direct_decode_cpu_rms_norm_element_count": 0,
            "mlx_metal_direct_decode_ffn_activation_element_count": 1024,
            "metal_direct_decode_cpu_ffn_activation_element_count": 0,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed",
            "ax_mlx_kv_capacity_kib": 32,
            "ax_mlx_kv_sliding_reclaimable_capacity_kib": 8,
            "ax_mlx_kv_growth_count": 1
        }
    });
    let candidate_environment = json!({
        "software": {
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_real_model_forward",
            "metal_direct_decode_batching_opportunity_observed": true,
            "metal_complete_model_forward_supported": true,
            "metal_real_model_forward": true,
            "metal_model_artifacts_validated": true,
            "mlx_metal_prefix_layers_attention": true,
            "metal_prefix_layers_cpu_reference": false,
            "metal_prefix_cpu_reference_dispatch_count": 0,
            "mlx_metal_prefix_projection_row_count": 2048,
            "metal_prefix_cpu_projection_row_count": 0,
            "mlx_metal_prefix_rms_norm_element_count": 1024,
            "metal_prefix_cpu_rms_norm_element_count": 0,
            "mlx_metal_prefix_ffn_activation_element_count": 1024,
            "metal_prefix_cpu_ffn_activation_element_count": 0,
            "metal_direct_decode_tokens": true,
            "metal_direct_decode_model_bound_ffn": true,
            "metal_direct_decode_batched_logits_group_count": 1,
            "metal_direct_decode_batched_logits_token_count": 2,
            "metal_direct_decode_batched_group_fallback_count": 1,
            "metal_direct_decode_batched_group_fallback_token_count": 2,
            "mlx_metal_direct_decode_projection_row_count": 3072,
            "metal_direct_decode_cpu_projection_row_count": 0,
            "mlx_metal_direct_decode_rms_norm_element_count": 1024,
            "metal_direct_decode_cpu_rms_norm_element_count": 0,
            "mlx_metal_direct_decode_ffn_activation_element_count": 1024,
            "metal_direct_decode_cpu_ffn_activation_element_count": 0,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed",
            "ax_mlx_kv_capacity_kib": 48,
            "ax_mlx_kv_sliding_reclaimable_capacity_kib": 12,
            "ax_mlx_kv_growth_count": 2
        }
    });

    let regression = build_regression_json(
        &baseline_metrics,
        &candidate_metrics,
        &baseline_environment,
        &candidate_environment,
        None,
    )
    .expect("regression artifact should build");

    assert_eq!(
        regression
            .get("summary")
            .and_then(|summary| summary.get("metal_direct_decode_batching_opportunity_observed"))
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        regression
            .get("summary")
            .and_then(|summary| summary.get("mlx_metal_readiness"))
            .and_then(Value::as_str),
        Some("ready")
    );
    assert_eq!(
        regression
            .get("summary")
            .and_then(|summary| summary.get("mlx_metal_hot_path_cpu_fallback_free"))
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        regression
            .get("summary")
            .and_then(|summary| summary.get("mlx_metal_batched_direct_decode_logits_ready"))
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        regression
            .get("readiness")
            .and_then(|readiness| readiness.get("candidate"))
            .and_then(|entry| entry.get("status"))
            .and_then(Value::as_str),
        Some("coverage_gap")
    );
    assert_eq!(
        regression
            .get("readiness")
            .and_then(|readiness| readiness.get("candidate"))
            .and_then(|entry| entry.get("hot_path_cpu_fallback_free"))
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        regression
            .get("readiness")
            .and_then(|readiness| readiness.get("candidate"))
            .and_then(|entry| entry.get("blockers"))
            .and_then(Value::as_array)
            .map(|entries| { entries.iter().filter_map(Value::as_str).collect::<Vec<_>>() }),
        Some(vec![
            "batched_direct_decode_logits_not_observed",
            "batched_direct_decode_group_fallback_remaining"
        ])
    );
    assert_eq!(
        regression
            .get("summary")
            .and_then(|summary| {
                summary.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB)
            })
            .and_then(Value::as_u64),
        Some(32)
    );
    assert_eq!(
        regression
            .get("contract")
            .and_then(|contract| {
                contract.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB)
            })
            .and_then(|field| field.get("candidate"))
            .and_then(Value::as_u64),
        Some(48)
    );

    let summary =
        build_compare_summary_markdown(&baseline_metrics, &candidate_metrics, &regression, None);
    assert!(summary.contains("metal_direct_decode_batching_opportunity_observed: `true`"));
    assert!(summary.contains("baseline_ax_mlx_kv_capacity_kib: `32`"));
    assert!(summary.contains("candidate_ax_mlx_kv_capacity_kib: `48`"));
    assert!(summary.contains("baseline_ax_mlx_kv_sliding_reclaimable_capacity_kib: `8`"));
    assert!(summary.contains("candidate_ax_mlx_kv_sliding_reclaimable_capacity_kib: `12`"));
}

#[test]
fn mlx_metal_readiness_allows_single_request_decode_without_batching_opportunity() {
    let inputs = MlxMetalReadinessInputs {
        tool_mode: "engine_bringup_runtime",
        selected_backend: "mlx",
        native_dense_dequantized_source: false,
        native_quantized_projection_binding_count: 0,
        direct_decode_batching_opportunity_observed: false,
        metal_complete_model_forward_supported: true,
        metal_real_model_forward: true,
        metal_model_artifacts_validated: true,
        mlx_metal_prefix_layers_attention: true,
        metal_prefix_layers_cpu_reference: false,
        metal_prefix_cpu_reference_dispatch_count: 0,
        mlx_metal_prefix_projection_row_count: 2048,
        metal_prefix_cpu_projection_row_count: 0,
        mlx_metal_prefix_rms_norm_element_count: 1024,
        metal_prefix_cpu_rms_norm_element_count: 0,
        mlx_metal_prefix_ffn_activation_element_count: 1024,
        metal_prefix_cpu_ffn_activation_element_count: 0,
        mlx_metal_prefix_residual_add_element_count: 1024,
        metal_prefix_cpu_residual_add_element_count: 0,
        mlx_metal_prefix_scale_element_count: 1024,
        metal_prefix_cpu_scale_element_count: 0,
        metal_direct_decode_tokens: true,
        metal_direct_decode_model_bound_ffn: true,
        metal_direct_decode_batched_logits_group_count: 0,
        metal_direct_decode_batched_logits_token_count: 0,
        metal_direct_decode_batched_group_fallback_count: 0,
        metal_direct_decode_batched_group_fallback_token_count: 0,
        mlx_metal_direct_decode_projection_row_count: 4096,
        metal_direct_decode_cpu_projection_row_count: 0,
        mlx_metal_direct_decode_rms_norm_element_count: 1024,
        metal_direct_decode_cpu_rms_norm_element_count: 0,
        mlx_metal_direct_decode_ffn_activation_element_count: 1024,
        metal_direct_decode_cpu_ffn_activation_element_count: 0,
        mlx_metal_direct_decode_residual_add_element_count: 1024,
        metal_direct_decode_cpu_residual_add_element_count: 0,
        mlx_metal_direct_decode_scale_element_count: 1024,
        metal_direct_decode_cpu_scale_element_count: 0,
    };
    let readiness = mlx_metal_readiness(inputs);

    assert_eq!(readiness.status, "ready");
    assert!(readiness.hot_path_cpu_fallback_free);
    assert!(readiness.batched_direct_decode_logits_ready);
    assert!(readiness.blockers.is_empty());

    let dequantized_readiness = mlx_metal_readiness(MlxMetalReadinessInputs {
        native_dense_dequantized_source: true,
        ..inputs
    });
    assert_eq!(dequantized_readiness.status, "coverage_gap");
    assert_eq!(
        dequantized_readiness.blockers,
        vec![NATIVE_DENSE_DEQUANTIZED_SOURCE_BLOCKER]
    );

    let quantized_projection_readiness = mlx_metal_readiness(MlxMetalReadinessInputs {
        native_quantized_projection_binding_count: 1,
        ..inputs
    });
    assert_eq!(quantized_projection_readiness.status, "coverage_gap");
    assert_eq!(
        quantized_projection_readiness.blockers,
        vec!["native_quantized_projection_kernel_missing"]
    );
}

#[test]
fn annotate_route_json_with_decode_batching_opportunity_marks_multi_decode_steps() {
    let mut route_json = serialize_route_metadata(&RouteMetadata::empty());
    let step_trace = vec![StepTraceEntry {
        t_ms: 0,
        step_id: Some(StepId(1)),
        admitted_request_ids: Vec::new(),
        selected_request_ids: vec![RequestId(1), RequestId(2)],
        deferred_request_ids: Vec::new(),
        memory_blocked_request_ids: Vec::new(),
        cleanup_request_ids: Vec::new(),
        scheduled_tokens: 2,
        prefix_hits: 0,
        cpu_time_us: 0,
        runner_time_us: 0,
        kv_usage_blocks: 0,
        evictions: 0,
        preempted_requests: 0,
        preempted_tokens: 0,
        runner_executed: true,
        route_metadata: RouteMetadata::empty(),
        metal_dispatch: None,
        items: vec![
            StepTraceItem {
                request_id: RequestId(1),
                mode: ax_engine_core::ExecutionMode::Decode,
                position_start: 0,
                position_end_exclusive: 1,
                scheduled_token_count: 1,
                input_token_count: 1,
                prefix_tokens_reused: 0,
                prefix_blocks_reused: 0,
            },
            StepTraceItem {
                request_id: RequestId(2),
                mode: ax_engine_core::ExecutionMode::Decode,
                position_start: 0,
                position_end_exclusive: 1,
                scheduled_token_count: 1,
                input_token_count: 1,
                prefix_tokens_reused: 0,
                prefix_blocks_reused: 0,
            },
        ],
    }];

    annotate_route_json_with_decode_batching_opportunity(&mut route_json, &step_trace);

    assert_eq!(
        route_json.get("metal_direct_decode_batching_opportunity_observed"),
        Some(&Value::Bool(true))
    );
}

#[test]
fn trusted_baseline_summary_surfaces_mlx_kv_cache_counters() {
    let trusted_baseline = json!({
        "name": "MLX KV Baseline",
        "source_run_id": "run-kv",
        "source_result_dir": "/tmp/run-kv",
        "manifest": {
            "id": "chat_qwen_short",
            "class": "scenario",
            "scenario": "chat",
            "model_family": "qwen3"
        },
        "runtime": {
            "tool_mode": "engine_bringup_runtime",
            "selected_backend": "mlx",
            "support_tier": "mlx_preview",
            "resolution_policy": "mlx_only"
        },
        "route": {
            "execution_semantics": "metal_real_model_forward",
            "ax_mlx_kv_capacity_kib": 64,
            "ax_mlx_kv_sliding_reclaimable_capacity_kib": 8,
            "ax_mlx_kv_growth_count": 4
        },
        "metrics": {
            "ttft_ms": 10.0,
            "decode_tok_s": 100.0,
            "memory_peak_mb": 1024.0,
            "prefix_hit_rate": 0.0
        }
    });

    let summary = build_trusted_baseline_summary_markdown(&trusted_baseline);

    assert!(summary.contains("ax_mlx_kv_capacity_kib: `64`"));
    assert!(summary.contains("ax_mlx_kv_sliding_reclaimable_capacity_kib: `8`"));
    assert!(summary.contains("ax_mlx_kv_growth_count: `4`"));
}

#[test]
fn handle_baseline_snapshots_named_artifact_and_prevents_overwrite() {
    let root = unique_test_dir("trusted-baseline");
    let source_root = root.join("results");
    let baseline_root = root.join("baselines");
    fs::create_dir_all(&source_root).expect("source root should create");
    fs::create_dir_all(&baseline_root).expect("baseline root should create");

    let run_dir = write_execution_artifact_fixture(
        &source_root,
        "1760-chat-qwen-short",
        "run-a",
        10.0,
        120.0,
    );
    let args = vec![
        "--source".to_string(),
        run_dir.display().to_string(),
        "--name".to_string(),
        "Dense Qwen M4 Baseline".to_string(),
        "--output-root".to_string(),
        baseline_root.display().to_string(),
    ];

    handle_baseline(&args).expect("baseline command should succeed");

    let trusted_baseline_dir = baseline_root.join("Dense-Qwen-M4-Baseline");
    assert!(trusted_baseline_dir.is_dir());
    assert!(trusted_baseline_dir.join("manifest.json").is_file());
    assert!(trusted_baseline_dir.join("environment.json").is_file());
    assert!(trusted_baseline_dir.join("metrics.json").is_file());
    assert!(trusted_baseline_dir.join("summary.md").is_file());
    assert!(trusted_baseline_dir.join("routes.json").is_file());
    assert!(trusted_baseline_dir.join("trace.json").is_file());
    assert!(trusted_baseline_dir.join("trusted_baseline.md").is_file());

    let metadata = load_json_value(&trusted_baseline_dir.join("trusted_baseline.json"))
        .expect("trusted baseline metadata should load");
    assert_eq!(
        metadata.get("name").and_then(Value::as_str),
        Some("Dense Qwen M4 Baseline")
    );
    assert_eq!(
        metadata.get("slug").and_then(Value::as_str),
        Some("Dense-Qwen-M4-Baseline")
    );
    assert_eq!(
        metadata.get("source_run_id").and_then(Value::as_str),
        Some("run-a")
    );
    assert_eq!(
        nested_value(&metadata, &["manifest", "id"]).and_then(Value::as_str),
        Some("chat_qwen_short")
    );

    let error = handle_baseline(&args).expect_err("baseline overwrite should fail closed");
    let CliError::Contract(message) = error else {
        panic!("baseline overwrite should return a contract error");
    };
    assert!(message.contains("trusted baseline already exists"));

    fs::remove_dir_all(&root).expect("test directory should clean up");
}

#[test]
fn compare_artifacts_include_trusted_baseline_identity() {
    let root = unique_test_dir("compare-trusted-baseline");
    let runs_root = root.join("runs");
    let compare_root = root.join("compare");
    let baseline_root = root.join("baselines");
    fs::create_dir_all(&runs_root).expect("runs root should create");
    fs::create_dir_all(&compare_root).expect("compare root should create");
    fs::create_dir_all(&baseline_root).expect("baseline root should create");

    let baseline_run =
        write_execution_artifact_fixture(&runs_root, "baseline-run", "baseline-run", 10.0, 120.0);
    let candidate_run =
        write_execution_artifact_fixture(&runs_root, "candidate-run", "candidate-run", 9.0, 132.0);

    let baseline_args = vec![
        "--source".to_string(),
        baseline_run.display().to_string(),
        "--name".to_string(),
        "Dense Qwen Trusted".to_string(),
        "--output-root".to_string(),
        baseline_root.display().to_string(),
    ];
    handle_baseline(&baseline_args).expect("trusted baseline should snapshot");
    let trusted_baseline_dir = baseline_root.join("Dense-Qwen-Trusted");

    let compare_dir = write_compare_artifacts(&trusted_baseline_dir, &candidate_run, &compare_root)
        .expect("compare should succeed against trusted baseline");
    let regression = load_json_value(&compare_dir.join("regression.json"))
        .expect("regression artifact should load");
    let summary = fs::read_to_string(compare_dir.join("comparison.md"))
        .expect("comparison summary should load");

    assert_eq!(
        nested_value(&regression, &["trusted_baseline", "name"]).and_then(Value::as_str),
        Some("Dense Qwen Trusted")
    );
    assert_eq!(
        nested_value(&regression, &["trusted_baseline", "source_run_id"]).and_then(Value::as_str),
        Some("baseline-run")
    );
    assert!(summary.contains("trusted_baseline: `Dense Qwen Trusted`"));

    fs::remove_dir_all(&root).expect("test directory should clean up");
}

#[test]
fn handle_matrix_compare_writes_rollup_and_member_compares() {
    let root = unique_test_dir("matrix-compare");
    let runs_root = root.join("runs");
    let matrix_root = root.join("matrix");
    let compare_root = root.join("compare");
    fs::create_dir_all(&runs_root).expect("runs root should create");
    fs::create_dir_all(&matrix_root).expect("matrix root should create");
    fs::create_dir_all(&compare_root).expect("compare root should create");

    let baseline_chat =
        write_execution_artifact_fixture(&runs_root, "baseline-chat", "baseline-chat", 10.0, 120.0);
    let candidate_chat = write_execution_artifact_fixture(
        &runs_root,
        "candidate-chat",
        "candidate-chat",
        9.0,
        132.0,
    );
    let baseline_concurrent = write_execution_artifact_fixture(
        &runs_root,
        "baseline-concurrent",
        "baseline-concurrent",
        12.0,
        100.0,
    );
    let candidate_concurrent = write_execution_artifact_fixture(
        &runs_root,
        "candidate-concurrent",
        "candidate-concurrent",
        11.0,
        108.0,
    );

    let baseline_matrix = write_matrix_result_fixture(
        &matrix_root,
        "baseline-matrix",
        "baseline-matrix-run",
        &[
            json!({
                "label": "Chat Qwen Short",
                "manifest_id": "chat_qwen_short",
                "manifest_path": repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
                "scenario": "chat",
                "model_family": "qwen3",
                "status": "ok",
                "tool_mode": "engine_bringup_runtime",
                "selected_backend": "mlx",
                "support_tier": "mlx_preview",
                "resolution_policy": "mlx_only",
                "result_dir": baseline_chat.display().to_string(),
                "ttft_ms": 10.0,
                "decode_tok_s": 120.0,
                "prefix_hit_rate": 0.0
            }),
            json!({
                "label": "Concurrent Qwen Dual",
                "manifest_id": "concurrent_qwen_dual",
                "manifest_path": repo_manifest_path("benchmarks/manifests/scenario/concurrent_qwen_dual.json").as_str(),
                "scenario": "concurrent",
                "model_family": "qwen3",
                "status": "ok",
                "tool_mode": "engine_bringup_runtime",
                "selected_backend": "mlx",
                "support_tier": "mlx_preview",
                "resolution_policy": "mlx_only",
                "result_dir": baseline_concurrent.display().to_string(),
                "ttft_ms": 12.0,
                "decode_tok_s": 100.0,
                "prefix_hit_rate": 0.0
            }),
        ],
    );
    let candidate_matrix = write_matrix_result_fixture(
        &matrix_root,
        "candidate-matrix",
        "candidate-matrix-run",
        &[
            json!({
                "label": "Chat Qwen Short",
                "manifest_id": "chat_qwen_short",
                "manifest_path": repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
                "scenario": "chat",
                "model_family": "qwen3",
                "status": "ok",
                "tool_mode": "engine_bringup_runtime",
                "selected_backend": "mlx",
                "support_tier": "mlx_preview",
                "resolution_policy": "mlx_only",
                "result_dir": candidate_chat.display().to_string(),
                "ttft_ms": 9.0,
                "decode_tok_s": 132.0,
                "prefix_hit_rate": 0.0
            }),
            json!({
                "label": "Concurrent Qwen Dual",
                "manifest_id": "concurrent_qwen_dual",
                "manifest_path": repo_manifest_path("benchmarks/manifests/scenario/concurrent_qwen_dual.json").as_str(),
                "scenario": "concurrent",
                "model_family": "qwen3",
                "status": "ok",
                "tool_mode": "engine_bringup_runtime",
                "selected_backend": "mlx",
                "support_tier": "mlx_preview",
                "resolution_policy": "mlx_only",
                "result_dir": candidate_concurrent.display().to_string(),
                "ttft_ms": 11.0,
                "decode_tok_s": 108.0,
                "prefix_hit_rate": 0.0
            }),
        ],
    );

    let args = vec![
        "--baseline".to_string(),
        baseline_matrix.display().to_string(),
        "--candidate".to_string(),
        candidate_matrix.display().to_string(),
        "--output-root".to_string(),
        compare_root.display().to_string(),
    ];

    handle_matrix_compare(&args).expect("matrix compare should succeed");

    let run_dir = fs::read_dir(&compare_root)
        .expect("compare root should be readable")
        .map(|entry| entry.expect("directory entry should read").path())
        .next()
        .expect("matrix compare result should exist");
    let matrix_regression = load_json_value(&run_dir.join("matrix_regression.json"))
        .expect("matrix regression should load");
    let summary =
        fs::read_to_string(run_dir.join("summary.md")).expect("matrix compare summary should load");

    assert_eq!(
        matrix_regression.get("id").and_then(Value::as_str),
        Some("test_mlx_mlx_matrix")
    );
    assert_eq!(
        nested_value(&matrix_regression, &["summary", "member_count"]).and_then(Value::as_u64),
        Some(2)
    );
    assert_eq!(
        matrix_regression
            .get("members")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(2)
    );
    assert!(summary.contains("Benchmark Matrix Compare"));
    assert!(summary.contains("Chat Qwen Short"));
    assert!(summary.contains("Concurrent Qwen Dual"));

    fs::remove_dir_all(&root).expect("test directory should clean up");
}

#[test]
fn handle_matrix_compare_requires_matrix_result_directories() {
    let root = unique_test_dir("matrix-compare-invalid-path");
    let runs_root = root.join("runs");
    let matrix_root = root.join("matrix");
    let compare_root = root.join("compare");
    fs::create_dir_all(&runs_root).expect("runs root should create");
    fs::create_dir_all(&matrix_root).expect("matrix root should create");
    fs::create_dir_all(&compare_root).expect("compare root should create");

    let baseline_chat =
        write_execution_artifact_fixture(&runs_root, "baseline-chat", "baseline-chat", 10.0, 120.0);
    let candidate_chat = write_execution_artifact_fixture(
        &runs_root,
        "candidate-chat",
        "candidate-chat",
        9.0,
        132.0,
    );

    let baseline_matrix = write_matrix_result_fixture(
        &matrix_root,
        "baseline-matrix",
        "baseline-matrix-run",
        &[json!({
            "label": "Chat Qwen Short",
            "manifest_id": "chat_qwen_short",
            "manifest_path": repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
            "scenario": "chat",
            "model_family": "qwen3",
            "status": "ok",
            "tool_mode": "engine_bringup_runtime",
            "selected_backend": "mlx",
            "support_tier": "mlx_preview",
            "resolution_policy": "mlx_only",
            "result_dir": baseline_chat.display().to_string(),
            "ttft_ms": 10.0,
            "decode_tok_s": 120.0,
            "prefix_hit_rate": 0.0
        })],
    );
    let candidate_matrix = write_matrix_result_fixture(
        &matrix_root,
        "candidate-matrix",
        "candidate-matrix-run",
        &[json!({
            "label": "Chat Qwen Short",
            "manifest_id": "chat_qwen_short",
            "manifest_path": repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
            "scenario": "chat",
            "model_family": "qwen3",
            "status": "ok",
            "tool_mode": "engine_bringup_runtime",
            "selected_backend": "mlx",
            "support_tier": "mlx_preview",
            "resolution_policy": "mlx_only",
            "result_dir": candidate_chat.display().to_string(),
            "ttft_ms": 9.0,
            "decode_tok_s": 132.0,
            "prefix_hit_rate": 0.0
        })],
    );

    let args = vec![
        "--baseline".to_string(),
        baseline_matrix.join("matrix.json").display().to_string(),
        "--candidate".to_string(),
        candidate_matrix.join("matrix.json").display().to_string(),
        "--output-root".to_string(),
        compare_root.display().to_string(),
    ];

    let error = handle_matrix_compare(&args).expect_err("matrix compare should reject file paths");
    let CliError::Contract(message) = error else {
        panic!("matrix compare should return a contract error for non-directory inputs");
    };

    assert!(message.contains("required directory does not exist or is not a directory"));

    fs::remove_dir_all(&root).expect("test directory should clean up");
}

#[test]
fn handle_compare_rejects_contract_failure_artifact_directories() {
    let root = unique_test_dir("compare-contract-failure");
    let output_root = root.join("results");
    fs::create_dir_all(&output_root).expect("output root should create");

    let mut manifest = llama_cpp_scenario_manifest("http://127.0.0.1:8081");
    manifest.runtime.backend_adapter = None;
    let manifest_path = root.join("llama-cpp-missing-adapter.json");
    write_json_file(&manifest_path, &manifest).expect("manifest should write");

    let scenario_args = vec![
        "--manifest".to_string(),
        manifest_path.display().to_string(),
        "--output-root".to_string(),
        output_root.display().to_string(),
    ];
    let _ = handle_scenario(&scenario_args)
        .expect_err("shared-prefix scenario should emit contract failure artifacts");

    let run_dir = fs::read_dir(&output_root)
        .expect("output root should be readable")
        .map(|entry| entry.expect("directory entry should read").path())
        .next()
        .expect("contract failure run should exist");

    let compare_output = root.join("compare-results");
    fs::create_dir_all(&compare_output).expect("compare output should create");
    let compare_args = vec![
        "--baseline".to_string(),
        run_dir.display().to_string(),
        "--candidate".to_string(),
        run_dir.display().to_string(),
        "--output-root".to_string(),
        compare_output.display().to_string(),
    ];

    let error =
        handle_compare(&compare_args).expect_err("compare should reject contract-failure runs");
    let CliError::Contract(message) = error else {
        panic!("compare should return a contract error for contract-failure artifacts");
    };

    assert!(message.contains("contract-failure result"));
    assert!(message.contains("llama_backend_adapter_required"));

    fs::remove_dir_all(&root).expect("test directory should clean up");
}

#[test]
fn compare_validation_allows_runtime_tuning_drift() {
    let baseline = load_test_manifest_json(
        repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
    );
    let mut candidate = baseline.clone();
    candidate["runtime"]["max_batch_tokens"] = json!(4096);
    candidate["runtime"]["kv_total_blocks"] = json!(1024);
    candidate["runtime"]["flags"]["prefix_cache"] = json!(true);

    validate_comparable_manifests(&baseline, &candidate)
        .expect("compare validation should allow runtime tuning drift");
}

#[test]
fn compare_validation_rejects_runtime_identity_drift() {
    let baseline = load_test_manifest_json(
        repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
    );
    let mut candidate = baseline.clone();
    candidate["runtime"]["selected_backend"] = json!("llama_cpp");

    let error = validate_comparable_manifests(&baseline, &candidate)
        .expect_err("compare validation should reject runtime identity drift");
    let CliError::Contract(message) = error else {
        panic!("compare validation should return a contract error");
    };

    assert!(message.contains("runtime.selected_backend"));
}

#[test]
fn compare_validation_rejects_replay_event_drift() {
    let baseline = load_test_manifest_json(
        repo_manifest_path("benchmarks/manifests/replay/full_prefix_to_decode_branch.json")
            .as_str(),
    );
    let mut candidate = baseline.clone();
    candidate["events"][1]["t_ms"] = json!(2);

    let error = validate_comparable_manifests(&baseline, &candidate)
        .expect_err("compare validation should reject replay event drift");
    let CliError::Contract(message) = error else {
        panic!("compare validation should return a contract error");
    };

    assert!(message.contains("events"));
}

#[test]
fn compare_validation_rejects_environment_arch_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "mlx_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "x86_64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "mlx_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject environment drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("machine.arch"));
}

#[test]
fn compare_validation_rejects_environment_kernel_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "mlx_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.5.0",
            "metal_driver": "system-default",
            "tool_mode": "mlx_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject kernel drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("software.kernel_release"));
}

#[test]
fn compare_validation_rejects_metal_toolchain_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "mlx_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });
    let mut candidate_runtime = mlx_runtime_fixture();
    candidate_runtime["metal_toolchain"]["metal"]["available"] = json!(false);
    candidate_runtime["metal_toolchain"]["fully_available"] = json!(false);
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "mlx_runtime"
        },
        "runtime": candidate_runtime,
        "route": {
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject metal toolchain drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("runtime.metal_toolchain"));
}

#[test]
fn compare_validation_rejects_environment_backend_adapter_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "llama_cpp_stepwise_runtime"
        },
        "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
        "route": {
            "prefix_cache_path": "delegated_prompt_cache",
            "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
            "prefix_reuse_provenance": "delegated_backend_prompt_cache",
            "backend_reported_cached_prompt_tokens": 64
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "llama_cpp_stepwise_runtime"
        },
        "runtime": llama_runtime_fixture("http://127.0.0.1:8082"),
        "route": {
            "prefix_cache_path": "delegated_prompt_cache",
            "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
            "prefix_reuse_provenance": "delegated_backend_prompt_cache",
            "backend_reported_cached_prompt_tokens": 64
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject backend adapter drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("runtime.backend_adapter"));
}

#[test]
fn compare_validation_rejects_environment_native_model_drift() {
    let mut baseline_runtime = mlx_runtime_fixture();
    baseline_runtime["mlx_model"] = json!({
        "artifacts_source": "explicit_config",
        "model_family": "qwen3",
        "tensor_format": "safetensors",
        "layer_count": 36,
        "tensor_count": 402,
        "tie_word_embeddings": false,
        "bindings_prepared": false,
        "buffers_bound": false,
        "buffer_count": 0,
        "buffer_bytes": 0
    });
    let mut candidate_runtime = baseline_runtime.clone();
    candidate_runtime["mlx_model"]["tensor_count"] = json!(401);

    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "mlx_runtime"
        },
        "runtime": baseline_runtime,
        "route": {
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "mlx_runtime"
        },
        "runtime": candidate_runtime,
        "route": {
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject native model drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("runtime.mlx_model.tensor_count"));
}

#[test]
fn compare_validation_rejects_route_execution_semantics_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_numeric_scaffold_only",
            "metal_numeric_scaffold_only": true,
            "metal_real_model_forward": false,
            "metal_model_conditioned_inputs": false,
            "metal_real_model_tensor_inputs": false,
            "metal_model_artifacts_validated": false,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_real_model_forward",
            "metal_numeric_scaffold_only": false,
            "metal_real_model_forward": true,
            "metal_model_conditioned_inputs": true,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject route execution semantics drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("route.execution_semantics"));
}

#[test]
fn compare_validation_rejects_complete_model_forward_support_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_real_model_tensor_inputs",
            "metal_numeric_scaffold_only": false,
            "metal_complete_model_forward_supported": false,
            "metal_real_model_forward": false,
            "metal_model_conditioned_inputs": true,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_real_model_tensor_inputs",
            "metal_numeric_scaffold_only": false,
            "metal_complete_model_forward_supported": true,
            "metal_real_model_forward": false,
            "metal_model_conditioned_inputs": true,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject complete-model-forward support drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("route.metal_complete_model_forward_supported"));
}

#[test]
fn compare_validation_rejects_route_direct_decode_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_real_model_tensor_inputs",
            "metal_numeric_scaffold_only": false,
            "metal_real_model_forward": false,
            "metal_model_conditioned_inputs": true,
            "metal_direct_decode_tokens": false,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_real_model_tensor_inputs",
            "metal_numeric_scaffold_only": false,
            "metal_real_model_forward": false,
            "metal_model_conditioned_inputs": true,
            "metal_direct_decode_tokens": true,
            "metal_direct_decode_checksum_lo": 42,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject route direct decode drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("route.metal_direct_decode_tokens"));
}

#[test]
fn compare_validation_rejects_route_prefix_attention_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_multilayer_native_prefix_attention",
            "metal_numeric_scaffold_only": false,
            "metal_real_model_forward": false,
            "metal_model_conditioned_inputs": true,
            "mlx_metal_prefix_layers_attention": true,
            "metal_prefix_layers_cpu_reference": false,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_multilayer_native_prefix_attention",
            "metal_numeric_scaffold_only": false,
            "metal_real_model_forward": false,
            "metal_model_conditioned_inputs": true,
            "mlx_metal_prefix_layers_attention": false,
            "metal_prefix_layers_cpu_reference": true,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject route prefix-attention drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("route.mlx_metal_prefix_layers_attention"));
}

#[test]
fn compare_validation_rejects_route_prefix_dispatch_count_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_multilayer_mixed_prefix_attention",
            "metal_numeric_scaffold_only": false,
            "metal_real_model_forward": false,
            "metal_model_conditioned_inputs": true,
            "mlx_metal_prefix_layers_attention": true,
            "metal_prefix_layers_cpu_reference": true,
            "mlx_metal_prefix_dispatch_count": 35,
            "metal_prefix_cpu_reference_dispatch_count": 1,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_multilayer_mixed_prefix_attention",
            "metal_numeric_scaffold_only": false,
            "metal_real_model_forward": false,
            "metal_model_conditioned_inputs": true,
            "mlx_metal_prefix_layers_attention": true,
            "metal_prefix_layers_cpu_reference": true,
            "mlx_metal_prefix_dispatch_count": 36,
            "metal_prefix_cpu_reference_dispatch_count": 0,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject route prefix dispatch-count drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("route.mlx_metal_prefix_dispatch_count"));
}

#[test]
fn compare_validation_rejects_route_native_dense_projection_coverage_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_real_model_forward",
            "metal_numeric_scaffold_only": false,
            "metal_complete_model_forward_supported": true,
            "metal_real_model_forward": true,
            "metal_model_conditioned_inputs": true,
            "mlx_metal_prefix_layers_attention": true,
            "metal_prefix_layers_cpu_reference": false,
            "mlx_metal_prefix_dispatch_count": 35,
            "metal_prefix_cpu_reference_dispatch_count": 0,
            "metal_direct_decode_tokens": true,
            "metal_direct_decode_model_bound_ffn": true,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "mlx_metal_projection_f32_binding_count": 0,
            "mlx_metal_projection_f16_binding_count": 28,
            "mlx_metal_projection_bf16_binding_count": 0,
            "mlx_metal_projection_unsupported_binding_count": 0,
            "mlx_metal_rms_norm_f32_binding_count": 0,
            "mlx_metal_rms_norm_f16_binding_count": 28,
            "mlx_metal_rms_norm_bf16_binding_count": 0,
            "mlx_metal_rms_norm_unsupported_binding_count": 0,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_real_model_forward",
            "metal_numeric_scaffold_only": false,
            "metal_complete_model_forward_supported": true,
            "metal_real_model_forward": true,
            "metal_model_conditioned_inputs": true,
            "mlx_metal_prefix_layers_attention": true,
            "metal_prefix_layers_cpu_reference": false,
            "mlx_metal_prefix_dispatch_count": 35,
            "metal_prefix_cpu_reference_dispatch_count": 0,
            "metal_direct_decode_tokens": true,
            "metal_direct_decode_model_bound_ffn": true,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "mlx_metal_projection_f32_binding_count": 0,
            "mlx_metal_projection_f16_binding_count": 27,
            "mlx_metal_projection_bf16_binding_count": 0,
            "mlx_metal_projection_unsupported_binding_count": 1,
            "mlx_metal_rms_norm_f32_binding_count": 0,
            "mlx_metal_rms_norm_f16_binding_count": 28,
            "mlx_metal_rms_norm_bf16_binding_count": 0,
            "mlx_metal_rms_norm_unsupported_binding_count": 0,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject native dense projection coverage drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("route.mlx_metal_projection_f16_binding_count"));
}

#[test]
fn compare_validation_rejects_route_direct_decode_ffn_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_real_model_tensor_inputs",
            "metal_numeric_scaffold_only": false,
            "metal_real_model_forward": false,
            "metal_model_conditioned_inputs": true,
            "metal_direct_decode_tokens": true,
            "metal_direct_decode_model_bound_ffn": false,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "engine_bringup_runtime"
        },
        "runtime": mlx_runtime_fixture(),
        "route": {
            "execution_semantics": "metal_real_model_tensor_inputs",
            "metal_numeric_scaffold_only": false,
            "metal_real_model_forward": false,
            "metal_model_conditioned_inputs": true,
            "metal_direct_decode_tokens": true,
            "metal_direct_decode_model_bound_ffn": true,
            "metal_real_model_tensor_inputs": true,
            "metal_model_artifacts_validated": true,
            "prefix_cache_path": "metadata_lookup",
            "prefix_cache_evidence": "none_observed",
            "prefix_reuse_provenance": "none_observed"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "scenario"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject route direct decode ffn drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("route.metal_direct_decode_model_bound_ffn"));
}

#[test]
fn compare_validation_rejects_prefix_reuse_provenance_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "llama_cpp_stepwise_runtime"
        },
        "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
        "route": {
            "prefix_cache_path": "delegated_prompt_cache",
            "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
            "prefix_reuse_provenance": "delegated_backend_prompt_cache",
            "backend_reported_cached_prompt_tokens": 64
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "replay"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "llama_cpp_stepwise_runtime"
        },
        "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
        "route": {
            "prefix_cache_path": "live_request_share",
            "prefix_cache_evidence": "engine_live_request_scheduler_reuse",
            "prefix_reuse_provenance": "mlx_live_request_share"
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "replay"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject prefix reuse provenance drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(
        message.contains("route.prefix_cache_path")
            || message.contains("route.prefix_reuse_provenance")
    );
}

#[test]
fn compare_validation_rejects_prefix_cache_evidence_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "llama_cpp_stepwise_runtime"
        },
        "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
        "route": {
            "prefix_cache_path": "delegated_prompt_cache",
            "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
            "prefix_reuse_provenance": "delegated_backend_prompt_cache",
            "backend_reported_cached_prompt_tokens": 64
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "replay"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "llama_cpp_stepwise_runtime"
        },
        "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
        "route": {
            "prefix_cache_path": "delegated_prompt_cache",
            "prefix_cache_evidence": "engine_live_request_scheduler_reuse",
            "prefix_reuse_provenance": "delegated_backend_prompt_cache",
            "backend_reported_cached_prompt_tokens": 64
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "replay"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject prefix-cache evidence drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("route.prefix_cache_evidence"));
}

#[test]
fn compare_validation_rejects_backend_reported_cached_prompt_tokens_drift() {
    let baseline = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "llama_cpp_stepwise_runtime"
        },
        "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
        "route": {
            "prefix_cache_path": "delegated_prompt_cache",
            "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
            "prefix_reuse_provenance": "delegated_backend_prompt_cache",
            "backend_reported_cached_prompt_tokens": 64
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "replay"
        }
    });
    let candidate = json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "machine": {
            "arch": "aarch64",
            "system_model": "Mac17,6",
            "soc": "Apple M5 Max",
            "memory_bytes": 137438953472_u64
        },
        "software": {
            "os_family": "macos",
            "os_version": "26.4.1",
            "os_build": "25E253",
            "kernel_release": "25.4.0",
            "metal_driver": "system-default",
            "tool_mode": "llama_cpp_stepwise_runtime"
        },
        "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
        "route": {
            "prefix_cache_path": "delegated_prompt_cache",
            "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
            "prefix_reuse_provenance": "delegated_backend_prompt_cache",
            "backend_reported_cached_prompt_tokens": 32
        },
        "benchmark": {
            "schema_family": "ax.engine_bench.v1",
            "subcommand": "replay"
        }
    });

    let error = validate_comparable_environments(&baseline, &candidate)
        .expect_err("compare validation should reject delegated cached-token drift");
    let CliError::Contract(message) = error else {
        panic!("environment validation should return a contract error");
    };

    assert!(message.contains("route.backend_reported_cached_prompt_tokens"));
}
