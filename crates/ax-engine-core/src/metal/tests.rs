use super::*;
use crate::ids::{BlockId, CacheGroupId, RequestId, StepId};
use crate::kv::BlockTableView;
use crate::scheduler::{
    ExecutionBatch, ExecutionItem, ExecutionMode, PositionRange, RouteMetadata,
};
use std::ffi::OsString;
use std::os::unix::fs::PermissionsExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

struct Phase1Fixture {
    root: PathBuf,
    build_dir: PathBuf,
}

impl Phase1Fixture {
    fn cleanup(self) {
        let _ = fs::remove_dir_all(self.root);
    }
}

type SimulatedNumericPath = ReferenceNumericPath;

fn simulated_numeric_path(workload: &MetalDispatchWorkload) -> SimulatedNumericPath {
    reference_numeric_path(workload)
}

fn simulated_numeric_trace(reference: &SimulatedNumericPath) -> MetalDispatchNumericTrace {
    MetalDispatchNumericTrace {
        attention_output_bits: reference
            .attention_output
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        key_cache_checksum: checksum_f32_slice(&reference.key_cache),
        attention_output_checksum: checksum_f32_slice(&reference.attention_output),
        gather_output_checksum: checksum_f32_pair(&reference.gather_key, &reference.gather_value),
        copy_output_checksum: checksum_f32_pair(&reference.copy_key, &reference.copy_value),
        validation: None,
    }
}

#[test]
fn pipeline_lookup_index_preserves_first_position_for_duplicate_kernel_names() {
    let lookup = pipeline_lookup_index(&[
        "rms_norm_f32".to_string(),
        "decode_logits_projection_f32".to_string(),
        "rms_norm_f32".to_string(),
    ]);

    assert_eq!(lookup.get("rms_norm_f32"), Some(&0));
    assert_eq!(lookup.get("decode_logits_projection_f32"), Some(&1));
    assert!(!lookup.contains_key("missing_kernel"));
}

#[test]
fn optional_kernel_dispatch_plan_tracks_available_hot_path_kernels() {
    let lookup = pipeline_lookup_index(&[
        "vector_add_f32".to_string(),
        "decode_logits_projection_f32".to_string(),
        "decode_logits_projection_batched_f16".to_string(),
        "gather_embedding_rows_bf16".to_string(),
        "rms_norm_f32".to_string(),
        "rms_norm_batched_bf16".to_string(),
        "logits_argmax_f32".to_string(),
        "sample_argmax_logprob_f32".to_string(),
        "apply_rope_batched_f32".to_string(),
        "expand_grouped_kv_heads_f32".to_string(),
        "ffn_gate_gelu_approx_product_f32".to_string(),
    ]);

    let plan = build_optional_kernel_dispatch_plan(&lookup);

    assert_eq!(
        plan.vector_add_kernel().map(|(name, _)| name),
        Some("vector_add_f32")
    );
    assert_eq!(
        plan.projection_kernel(NativeTensorDataType::F32)
            .map(|(name, _)| name),
        Some("decode_logits_projection_f32")
    );
    assert_eq!(plan.projection_kernel(NativeTensorDataType::F16), None);
    assert_eq!(
        plan.batched_projection_kernel(NativeTensorDataType::F16)
            .map(|(name, _)| name),
        Some("decode_logits_projection_batched_f16")
    );
    assert_eq!(
        plan.embedding_gather_kernel(NativeTensorDataType::Bf16)
            .map(|(name, _)| name),
        Some("gather_embedding_rows_bf16")
    );
    assert_eq!(
        plan.rms_norm_kernel(NativeTensorDataType::F32)
            .map(|(name, _)| name),
        Some("rms_norm_f32")
    );
    assert_eq!(
        plan.batched_rms_norm_kernel(NativeTensorDataType::Bf16)
            .map(|(name, _)| name),
        Some("rms_norm_batched_bf16")
    );
    assert!(plan.logits_argmax_f32.is_some());
    assert!(plan.logits_argmax_batched_f32.is_none());
    assert!(plan.sample_argmax_logprob_f32.is_some());
    assert!(plan.sample_argmax_logprob_batched_f32.is_none());
    assert!(plan.apply_rope_f32.is_none());
    assert!(plan.apply_rope_batched_f32.is_some());
    assert!(plan.expand_grouped_kv_heads_f32.is_some());
    assert_eq!(
        plan.ffn_gate_product_kernel(ModelFfnActivation::GeluApprox)
            .map(|(name, _)| name),
        Some("ffn_gate_gelu_approx_product_f32")
    );
    assert_eq!(plan.ffn_gate_product_kernel(ModelFfnActivation::Silu), None);
}

#[cfg(target_os = "macos")]
#[test]
fn optional_kernel_feedback_disables_kernel_after_threshold_failures() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let kernel_name = "decode_logits_projection_f32";

    for attempt in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        assert!(optional_kernel_allowed_in_feedback_state(
            &feedback,
            kernel_name
        ));
        record_optional_kernel_feedback_state(&mut feedback, kernel_name, false);
        let expected_failures = attempt + 1;
        assert_eq!(
            feedback.consecutive_failures_by_kernel.get(kernel_name),
            Some(&expected_failures)
        );
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        kernel_name
    ));
    assert!(feedback.disabled_kernels.contains(kernel_name));
}

#[cfg(target_os = "macos")]
#[test]
fn optional_kernel_feedback_success_resets_consecutive_failures() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let kernel_name = "rms_norm_f32";

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, kernel_name, false);
    }
    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        kernel_name
    ));

    record_optional_kernel_feedback_state(&mut feedback, kernel_name, true);

    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        kernel_name
    ));
    assert!(!feedback
        .consecutive_failures_by_kernel
        .contains_key(kernel_name));
    assert!(!feedback.disabled_kernels.contains(kernel_name));
}

#[cfg(target_os = "macos")]
#[test]
fn batched_group_feedback_keys_disable_only_the_failing_group_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let sampler_group_key = sampler_batched_group_feedback_key(4, 32_000);
    let different_sampler_group_key = sampler_batched_group_feedback_key(2, 32_000);
    let different_sampler_width_key = sampler_batched_group_feedback_key(4, 256_000);
    let decode_dims = ModelBoundDecodeDims {
        input_width: 8,
        hidden_dim: 16,
        intermediate_dim: 32,
        vocab_rows: 64,
    };
    let decode_group_key = direct_decode_batched_group_feedback_key(4, decode_dims);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &sampler_group_key, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &sampler_group_key
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_sampler_group_key
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_sampler_width_key
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &decode_group_key
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn prefix_attention_group_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill workload should resolve")
            .with_numeric_layout(MetalDispatchNumericLayout::new(4, 8));
    let different_group_size = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("mixed workload should resolve")
        .with_numeric_layout(MetalDispatchNumericLayout::new(4, 8));
    let different_layout =
        failing_workload.with_numeric_layout(MetalDispatchNumericLayout::new(2, 16));

    let failing_key = prefix_attention_group_feedback_key(&failing_workload);
    let different_group_size_key = prefix_attention_group_feedback_key(&different_group_size);
    let different_layout_key = prefix_attention_group_feedback_key(&different_layout);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_key, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_key
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_group_size_key
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_layout_key
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn batched_sampler_group_output_validation_requires_matching_cardinality() {
    let (valid_output, valid_success) =
        validate_batched_sampler_group_output(Some(vec![1_u32, 2_u32]), 2);
    assert_eq!(valid_output, Some(vec![1_u32, 2_u32]));
    assert!(valid_success);

    let (missing_output, missing_success) =
        validate_batched_sampler_group_output(Some(vec![1_u32]), 2);
    assert_eq!(missing_output, None);
    assert!(!missing_success);

    let (none_output, none_success) = validate_batched_sampler_group_output::<u32>(None, 2);
    assert_eq!(none_output, None);
    assert!(!none_success);
}

#[cfg(target_os = "macos")]
#[test]
fn direct_decode_group_output_validation_requires_complete_request_coverage() {
    let expected_request_ids = vec![RequestId(3), RequestId(5)];
    let valid_result = ModelBoundDirectDecodeResult {
        tokens: vec![(RequestId(3), 3), (RequestId(5), 5)],
        logits_outputs: vec![
            RequestLogitsOutput {
                request_id: RequestId(3),
                logits: vec![3.0],
            },
            RequestLogitsOutput {
                request_id: RequestId(5),
                logits: vec![5.0],
            },
        ],
        model_bound_ffn_decode: true,
        native_logits_projection_decode: true,
        execution_tally: PrefixAttentionExecutionTally::default(),
        native_dense_tally: DirectDecodeNativeDenseTally::default(),
    };
    let (valid_output, valid_success) = validate_model_bound_direct_decode_group_output(
        Some(valid_result.clone()),
        &expected_request_ids,
    );
    assert_eq!(valid_output, Some(valid_result));
    assert!(valid_success);

    let incomplete_result = ModelBoundDirectDecodeResult {
        tokens: vec![(RequestId(3), 3)],
        logits_outputs: vec![RequestLogitsOutput {
            request_id: RequestId(3),
            logits: vec![3.0],
        }],
        model_bound_ffn_decode: true,
        native_logits_projection_decode: true,
        execution_tally: PrefixAttentionExecutionTally::default(),
        native_dense_tally: DirectDecodeNativeDenseTally::default(),
    };
    let (incomplete_output, incomplete_success) = validate_model_bound_direct_decode_group_output(
        Some(incomplete_result),
        &expected_request_ids,
    );
    assert_eq!(incomplete_output, None);
    assert!(!incomplete_success);
}

#[cfg(target_os = "macos")]
#[test]
fn prefix_attention_group_split_policy_only_splits_when_batch_salvage_is_possible() {
    assert!(prefix_attention_group_should_split(4, true, false, false));
    assert!(prefix_attention_group_should_split(4, true, true, true));
    assert!(!prefix_attention_group_should_split(4, true, true, false));
    assert!(!prefix_attention_group_should_split(1, true, false, false));
    assert!(!prefix_attention_group_should_split(1, true, true, true));
    assert!(!prefix_attention_group_should_split(4, false, false, false));
    assert!(!prefix_attention_group_should_split(4, false, true, true));
}

#[cfg(target_os = "macos")]
#[test]
fn batched_projection_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape = batched_projection_feedback_key(
        "decode_logits_projection_batched_f16",
        4,
        4096,
        4096,
        4096,
        4096,
    );
    let different_row_count = batched_projection_feedback_key(
        "decode_logits_projection_batched_f16",
        2,
        4096,
        4096,
        4096,
        4096,
    );
    let different_output_dim = batched_projection_feedback_key(
        "decode_logits_projection_batched_f16",
        4,
        8192,
        4096,
        4096,
        4096,
    );
    let different_hidden_stride = batched_projection_feedback_key(
        "decode_logits_projection_batched_f16",
        4,
        4096,
        4096,
        8192,
        4096,
    );

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_row_count
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_output_dim
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_hidden_stride
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn batched_logits_argmax_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape = batched_logits_argmax_feedback_key("logits_argmax_batched_f32", 4, 256_000);
    let different_row_count =
        batched_logits_argmax_feedback_key("logits_argmax_batched_f32", 2, 256_000);
    let different_vocab_rows =
        batched_logits_argmax_feedback_key("logits_argmax_batched_f32", 4, 32_000);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_row_count
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_vocab_rows
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn batched_ffn_gate_product_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape = batched_ffn_gate_product_feedback_key("ffn_gate_silu_product_f32", 8, 4096);
    let different_row_count =
        batched_ffn_gate_product_feedback_key("ffn_gate_silu_product_f32", 4, 4096);
    let different_row_width =
        batched_ffn_gate_product_feedback_key("ffn_gate_silu_product_f32", 8, 2048);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_row_count
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_row_width
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn ffn_gate_product_feedback_keys_disable_only_the_failing_width() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_width = ffn_gate_product_feedback_key("ffn_gate_silu_product_f32", 4096);
    let different_width = ffn_gate_product_feedback_key("ffn_gate_silu_product_f32", 2048);
    let different_kernel = ffn_gate_product_feedback_key("ffn_gate_gelu_approx_product_f32", 4096);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_width, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_width
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_width
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_kernel
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn embedding_gather_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape =
        embedding_gather_feedback_key("gather_embedding_rows_f16", 128, 256_000, 4096);
    let different_token_count =
        embedding_gather_feedback_key("gather_embedding_rows_f16", 64, 256_000, 4096);
    let different_hidden_dim =
        embedding_gather_feedback_key("gather_embedding_rows_f16", 128, 256_000, 2048);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_token_count
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_hidden_dim
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn batched_grouped_kv_expand_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape =
        batched_grouped_kv_expand_feedback_key("expand_grouped_kv_heads_f32", 8, 8, 2, 128);
    let different_token_count =
        batched_grouped_kv_expand_feedback_key("expand_grouped_kv_heads_f32", 4, 8, 2, 128);
    let different_head_dim =
        batched_grouped_kv_expand_feedback_key("expand_grouped_kv_heads_f32", 8, 8, 2, 64);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_token_count
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_head_dim
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn batched_rope_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape = batched_rope_feedback_key(
        "apply_rope_batched_f32",
        8,
        8,
        2,
        128,
        ModelStageRopeStyle::Neox,
    );
    let different_token_count = batched_rope_feedback_key(
        "apply_rope_batched_f32",
        4,
        8,
        2,
        128,
        ModelStageRopeStyle::Neox,
    );
    let different_rope_style = batched_rope_feedback_key(
        "apply_rope_batched_f32",
        8,
        8,
        2,
        128,
        ModelStageRopeStyle::Interleaved,
    );

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_token_count
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_rope_style
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn projection_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape = projection_feedback_key("decode_logits_projection_f16", 4096, 4096, 4096);
    let different_output_dim =
        projection_feedback_key("decode_logits_projection_f16", 2048, 4096, 4096);
    let different_input_width =
        projection_feedback_key("decode_logits_projection_f16", 4096, 2048, 4096);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_output_dim
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_input_width
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn sampler_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape = sampler_feedback_key("sample_argmax_logprob_f32", 256_000);
    let different_logits_width = sampler_feedback_key("sample_argmax_logprob_f32", 32_000);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_logits_width
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn batched_sampler_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape =
        batched_sampler_feedback_key("sample_argmax_logprob_batched_f32", 8, 256_000);
    let different_row_count =
        batched_sampler_feedback_key("sample_argmax_logprob_batched_f32", 4, 256_000);
    let different_logits_width =
        batched_sampler_feedback_key("sample_argmax_logprob_batched_f32", 8, 32_000);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_row_count
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_logits_width
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn logits_argmax_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape = logits_argmax_feedback_key("logits_argmax_f32", 256_000);
    let different_vocab_rows = logits_argmax_feedback_key("logits_argmax_f32", 32_000);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_vocab_rows
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn rope_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape = rope_feedback_key("apply_rope_f32", 8, 2, 128, ModelStageRopeStyle::Neox);
    let different_head_dim =
        rope_feedback_key("apply_rope_f32", 8, 2, 64, ModelStageRopeStyle::Neox);
    let different_rope_style = rope_feedback_key(
        "apply_rope_f32",
        8,
        2,
        128,
        ModelStageRopeStyle::Interleaved,
    );

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_head_dim
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_rope_style
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn grouped_kv_expand_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape = grouped_kv_expand_feedback_key("expand_grouped_kv_heads_f32", 8, 2, 128);
    let different_q_heads =
        grouped_kv_expand_feedback_key("expand_grouped_kv_heads_f32", 4, 2, 128);
    let different_head_dim =
        grouped_kv_expand_feedback_key("expand_grouped_kv_heads_f32", 8, 2, 64);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_q_heads
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_head_dim
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn rms_norm_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape = rms_norm_feedback_key("rms_norm_f16", 4096);
    let different_value_count = rms_norm_feedback_key("rms_norm_f16", 2048);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_value_count
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn batched_rms_norm_feedback_keys_disable_only_the_failing_shape() {
    let mut feedback = MetalOptionalKernelFeedbackState::default();
    let failing_shape = batched_rms_norm_feedback_key("rms_norm_batched_f16", 8, 256);
    let different_row_count = batched_rms_norm_feedback_key("rms_norm_batched_f16", 4, 256);
    let different_row_width = batched_rms_norm_feedback_key("rms_norm_batched_f16", 8, 128);

    for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        record_optional_kernel_feedback_state(&mut feedback, &failing_shape, false);
    }

    assert!(!optional_kernel_allowed_in_feedback_state(
        &feedback,
        &failing_shape
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_row_count
    ));
    assert!(optional_kernel_allowed_in_feedback_state(
        &feedback,
        &different_row_width
    ));
}

#[test]
fn grouped_sampler_request_indices_by_logits_width_preserves_request_order() {
    let requests = vec![
        SamplerRequest {
            request_id: RequestId(1),
            previous_token: 9,
            logits: Some(vec![0.1, 0.9, -0.5]),
            generated_len: 0,
            max_output_tokens: 4,
            sampling_params: crate::sampling::SamplingParams::default(),
        },
        SamplerRequest {
            request_id: RequestId(2),
            previous_token: 19,
            logits: Some(vec![0.2, 0.3]),
            generated_len: 0,
            max_output_tokens: 4,
            sampling_params: crate::sampling::SamplingParams::default(),
        },
        SamplerRequest {
            request_id: RequestId(3),
            previous_token: 29,
            logits: None,
            generated_len: 0,
            max_output_tokens: 4,
            sampling_params: crate::sampling::SamplingParams::default(),
        },
        SamplerRequest {
            request_id: RequestId(4),
            previous_token: 39,
            logits: Some(vec![1.2, 0.3, -0.2]),
            generated_len: 0,
            max_output_tokens: 4,
            sampling_params: crate::sampling::SamplingParams::default(),
        },
    ];

    let grouped = grouped_sampler_request_indices_by_logits_width(&requests)
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(grouped, vec![(2, vec![1]), (3, vec![0, 3])]);
}

#[cfg(target_os = "macos")]
#[test]
fn grouped_sampler_results_recursively_preserve_batched_subgroups() {
    let indices = vec![0_usize, 2, 4, 6];
    let mut attempted_group_sizes = Vec::new();

    let results = collect_grouped_sampler_results_with_item_fallback(
        &indices,
        &mut |group_indices| {
            attempted_group_sizes.push(group_indices.len());
            if group_indices.len() > 2 {
                return None;
            }
            Some(
                group_indices
                    .iter()
                    .map(|request_index| {
                        (
                            u32::try_from(*request_index).unwrap_or(u32::MAX),
                            -(*request_index as f32),
                        )
                    })
                    .collect(),
            )
        },
        &mut |request_index| {
            Some((
                u32::try_from(request_index).unwrap_or(u32::MAX),
                request_index as f32,
            ))
        },
    )
    .expect("recursive sampler subgroup fallback should succeed");

    assert_eq!(attempted_group_sizes, vec![4, 2, 2]);
    assert_eq!(
        results,
        vec![
            (0, (0, -0.0)),
            (2, (2, -2.0)),
            (4, (4, -4.0)),
            (6, (6, -6.0)),
        ]
    );
}

#[allow(dead_code)]
fn write_valid_native_model_fixture() -> PathBuf {
    let root_dir = unique_test_dir("native-model-fixture");
    fs::create_dir_all(&root_dir).expect("native model fixture directory should create");
    fs::write(root_dir.join("model.safetensors"), vec![0_u8; 4096])
        .expect("native model weights should write");

    // Dimensions are deliberately tiny so that every tensor fits within the
    // 32-byte (= 16 × f16) limit imposed by `native_model_tensor`.
    //
    // hidden_size=2, vocab=4, q_heads=1, kv_heads=1, head_dim=2 gives:
    //   embed_tokens  [4, 2]  =  8 f16 = 16 B
    //   qkv_proj      [6, 2]  = 12 f16 = 24 B  (packed: (q+2k) heads × head_dim rows)
    //   gate_up_proj  [4, 2]  =  8 f16 = 16 B  (intermediate_size=2, gate+up packed)
    //   all 1-D norms  [2]    =  2 f16 =  4 B
    //   all 2-D mats  [2, 2] =  4 f16 =  8 B
    // Token IDs 1–4 map to rows 1, 2, 3, 0 (mod 4) — all within the 4-row embedding.
    let manifest = crate::model::NativeModelManifest {
        schema_version: crate::model::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: "qwen3_dense".to_string(),
        tensor_format: crate::model::NativeTensorFormat::Safetensors,
        layer_count: 1,
        hidden_size: 2,
        attention_head_count: 1,
        attention_head_dim: 2,
        kv_head_count: 1,
        vocab_size: 4,
        tie_word_embeddings: false,
        rope_theta: None,
        query_pre_attn_scalar: None,
        attention_logit_softcap: None,
        attention_value_from_key_layers: Vec::new(),
        attention_v_norm_no_scale_layers: Vec::new(),
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        tensors: vec![
            native_model_tensor(
                "model.embed_tokens.weight",
                NativeTensorRole::TokenEmbedding,
                None,
                vec![4, 2],
            ),
            native_model_tensor(
                "model.norm.weight",
                NativeTensorRole::FinalNorm,
                None,
                vec![2],
            ),
            native_model_tensor("lm_head.weight", NativeTensorRole::LmHead, None, vec![4, 2]),
            native_model_tensor(
                "model.layers.0.input_layernorm.weight",
                NativeTensorRole::AttentionNorm,
                Some(0),
                vec![2],
            ),
            native_model_tensor(
                "model.layers.0.self_attn.qkv_proj.weight",
                NativeTensorRole::AttentionQkvPacked,
                Some(0),
                // (q_heads + 2 * kv_heads) * head_dim = (1 + 2) * 2 = 6 rows
                vec![6, 2],
            ),
            native_model_tensor(
                "model.layers.0.self_attn.o_proj.weight",
                NativeTensorRole::AttentionO,
                Some(0),
                vec![2, 2],
            ),
            native_model_tensor(
                "model.layers.0.post_attention_layernorm.weight",
                NativeTensorRole::FfnNorm,
                Some(0),
                vec![2],
            ),
            native_model_tensor(
                "model.layers.0.mlp.gate_up_proj.weight",
                NativeTensorRole::FfnGateUpPacked,
                Some(0),
                // 2 * intermediate_size = 2 * 2 = 4 rows
                vec![4, 2],
            ),
            native_model_tensor(
                "model.layers.0.mlp.down_proj.weight",
                NativeTensorRole::FfnDown,
                Some(0),
                vec![2, 2],
            ),
        ],
    };

    fs::write(
        root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).expect("native model manifest should serialize"),
    )
    .expect("native model manifest should write");

    root_dir
}

#[allow(dead_code)]
fn native_model_tensor(
    name: &str,
    role: NativeTensorRole,
    layer_index: Option<u32>,
    shape: Vec<u64>,
) -> crate::model::NativeTensorSpec {
    crate::model::NativeTensorSpec {
        name: name.to_string(),
        role,
        layer_index,
        dtype: crate::model::NativeTensorDataType::F16,
        shape,
        file: PathBuf::from("model.safetensors"),
        offset_bytes: 0,
        length_bytes: 32,
    }
}

fn native_model_tensor_with_file(
    name: &str,
    role: NativeTensorRole,
    layer_index: Option<u32>,
    shape: &[u64],
    file: &str,
    length_bytes: u64,
) -> crate::model::NativeTensorSpec {
    crate::model::NativeTensorSpec {
        name: name.to_string(),
        role,
        layer_index,
        dtype: crate::model::NativeTensorDataType::F32,
        shape: shape.to_vec(),
        file: PathBuf::from(file),
        offset_bytes: 0,
        length_bytes,
    }
}

fn write_f32_tensor_file(root_dir: &Path, file_name: &str, values: &[f32]) {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(root_dir.join(file_name), bytes).expect("tensor bytes should write");
}

#[cfg(target_os = "macos")]
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

    let matrix_bytes = (64 * size_of::<f32>()) as u64;
    let vector_bytes = (8 * size_of::<f32>()) as u64;
    let embedding_bytes = (embedding.len() * size_of::<f32>()) as u64;
    let manifest = crate::model::NativeModelManifest {
        schema_version: crate::model::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: "qwen3_dense".to_string(),
        tensor_format: crate::model::NativeTensorFormat::Safetensors,
        layer_count: 1,
        hidden_size: 8,
        attention_head_count: 2,
        attention_head_dim: 4,
        kv_head_count: 2,
        vocab_size: 5,
        tie_word_embeddings: false,
        rope_theta: None,
        query_pre_attn_scalar: None,
        attention_logit_softcap: None,
        attention_value_from_key_layers: Vec::new(),
        attention_v_norm_no_scale_layers: Vec::new(),
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        tensors: vec![
            native_model_tensor_with_file(
                "model.embed_tokens.weight",
                NativeTensorRole::TokenEmbedding,
                None,
                &[5, 8],
                "embed.bin",
                embedding_bytes,
            ),
            native_model_tensor_with_file(
                "model.norm.weight",
                NativeTensorRole::FinalNorm,
                None,
                &[8],
                "final_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "lm_head.weight",
                NativeTensorRole::LmHead,
                None,
                &[5, 8],
                "lm_head.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.input_layernorm.weight",
                NativeTensorRole::AttentionNorm,
                Some(0),
                &[8],
                "attn_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(0),
                &[8, 8],
                "attn_q.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(0),
                &[8, 8],
                "attn_k.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(0),
                &[8, 8],
                "attn_v.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.o_proj.weight",
                NativeTensorRole::AttentionO,
                Some(0),
                &[8, 8],
                "attn_o.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.post_attention_layernorm.weight",
                NativeTensorRole::FfnNorm,
                Some(0),
                &[8],
                "ffn_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.gate_proj.weight",
                NativeTensorRole::FfnGate,
                Some(0),
                &[8, 8],
                "ffn_gate.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.up_proj.weight",
                NativeTensorRole::FfnUp,
                Some(0),
                &[8, 8],
                "ffn_up.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.down_proj.weight",
                NativeTensorRole::FfnDown,
                Some(0),
                &[8, 8],
                "ffn_down.bin",
                matrix_bytes,
            ),
        ],
    };

    fs::write(
        root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
    )
    .expect("projection manifest should write");

    root_dir
}

#[cfg(target_os = "macos")]
fn write_projection_qk_norm_native_model_fixture() -> PathBuf {
    let root_dir = write_projection_native_model_fixture();
    let q_norm = vec![2.0_f32, 1.0, 1.0, 1.0];
    let k_norm = vec![3.0_f32, 1.0, 1.0, 1.0];
    let norm_bytes = (q_norm.len() * size_of::<f32>()) as u64;
    write_f32_tensor_file(&root_dir, "attn_q_norm.bin", &q_norm);
    write_f32_tensor_file(&root_dir, "attn_k_norm.bin", &k_norm);

    let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("projection manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("projection manifest should parse");
    manifest.tensors.push(native_model_tensor_with_file(
        "model.layers.0.self_attn.q_norm.weight",
        NativeTensorRole::AttentionQNorm,
        Some(0),
        &[4],
        "attn_q_norm.bin",
        norm_bytes,
    ));
    manifest.tensors.push(native_model_tensor_with_file(
        "model.layers.0.self_attn.k_norm.weight",
        NativeTensorRole::AttentionKNorm,
        Some(0),
        &[4],
        "attn_k_norm.bin",
        norm_bytes,
    ));
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
    )
    .expect("projection manifest should rewrite");

    root_dir
}

#[cfg(target_os = "macos")]
fn write_projection_value_from_key_native_model_fixture(apply_v_norm_no_scale: bool) -> PathBuf {
    let root_dir = write_projection_qk_norm_native_model_fixture();
    let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("projection manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("projection manifest should parse");
    manifest.model_family = "llama3_dense".to_string();
    manifest.attention_value_from_key_layers = vec![0];
    manifest.attention_v_norm_no_scale_layers = if apply_v_norm_no_scale {
        vec![0]
    } else {
        Vec::new()
    };
    manifest.tensors.retain(|tensor| {
        !(tensor.layer_index == Some(0) && tensor.role == NativeTensorRole::AttentionV)
    });
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
    )
    .expect("projection manifest should write");

    root_dir
}

#[cfg(target_os = "macos")]
fn write_projection_custom_rope_native_model_fixture(rope_theta: u32) -> PathBuf {
    let root_dir = write_projection_native_model_fixture();
    let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("projection manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("projection manifest should parse");
    manifest.rope_theta = Some(rope_theta);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
    )
    .expect("projection manifest should rewrite");

    root_dir
}

#[cfg(target_os = "macos")]
fn write_gemma_projection_native_model_fixture() -> PathBuf {
    let root_dir = write_projection_native_model_fixture();
    let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("projection manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("projection manifest should parse");
    manifest.model_family = "gemma2_dense".to_string();
    fs::write(
        root_dir.join("attn_norm.bin"),
        vec![0_u8; 8 * size_of::<f32>()],
    )
    .expect("gemma attention norm should write");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
    )
    .expect("projection manifest should rewrite");

    root_dir
}

#[cfg(target_os = "macos")]
fn write_gemma_projection_custom_rope_native_model_fixture(rope_theta: u32) -> PathBuf {
    let root_dir = write_gemma_projection_native_model_fixture();
    let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("gemma projection manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("gemma projection manifest should parse");
    manifest.rope_theta = Some(rope_theta);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("gemma projection manifest should serialize"),
    )
    .expect("gemma projection manifest should rewrite");

    root_dir
}

#[cfg(target_os = "macos")]
fn write_gemma_projection_attention_config_native_model_fixture(
    query_pre_attn_scalar: u32,
    attention_logit_softcap: u32,
) -> PathBuf {
    let root_dir = write_gemma_projection_native_model_fixture();
    let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("gemma projection manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("gemma projection manifest should parse");
    manifest.query_pre_attn_scalar = Some(query_pre_attn_scalar);
    manifest.attention_logit_softcap = Some(attention_logit_softcap);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("gemma projection manifest should serialize"),
    )
    .expect("gemma projection manifest should rewrite");

    root_dir
}

#[cfg(target_os = "macos")]
fn write_grouped_projection_native_model_fixture() -> PathBuf {
    let root_dir = unique_test_dir("native-model-grouped-projection");
    fs::create_dir_all(&root_dir).expect("grouped projection fixture directory should create");

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
    let grouped_k = [
        2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let grouped_v = [
        3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let zero_matrix = vec![0.0_f32; 64];

    write_f32_tensor_file(&root_dir, "embed.bin", &embedding);
    write_f32_tensor_file(&root_dir, "final_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "lm_head.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "attn_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "attn_q.bin", &identity);
    write_f32_tensor_file(&root_dir, "attn_k.bin", &grouped_k);
    write_f32_tensor_file(&root_dir, "attn_v.bin", &grouped_v);
    write_f32_tensor_file(&root_dir, "attn_o.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "ffn_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "ffn_gate.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "ffn_up.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "ffn_down.bin", &zero_matrix);

    let full_matrix_bytes = (64 * size_of::<f32>()) as u64;
    let grouped_matrix_bytes = (32 * size_of::<f32>()) as u64;
    let vector_bytes = (8 * size_of::<f32>()) as u64;
    let embedding_bytes = (embedding.len() * size_of::<f32>()) as u64;
    let manifest = crate::model::NativeModelManifest {
        schema_version: crate::model::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: "qwen3_dense".to_string(),
        tensor_format: crate::model::NativeTensorFormat::Safetensors,
        layer_count: 1,
        hidden_size: 8,
        attention_head_count: 4,
        attention_head_dim: 2,
        kv_head_count: 2,
        vocab_size: 5,
        tie_word_embeddings: false,
        rope_theta: None,
        query_pre_attn_scalar: None,
        attention_logit_softcap: None,
        attention_value_from_key_layers: Vec::new(),
        attention_v_norm_no_scale_layers: Vec::new(),
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        tensors: vec![
            native_model_tensor_with_file(
                "model.embed_tokens.weight",
                NativeTensorRole::TokenEmbedding,
                None,
                &[5, 8],
                "embed.bin",
                embedding_bytes,
            ),
            native_model_tensor_with_file(
                "model.norm.weight",
                NativeTensorRole::FinalNorm,
                None,
                &[8],
                "final_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "lm_head.weight",
                NativeTensorRole::LmHead,
                None,
                &[5, 8],
                "lm_head.bin",
                full_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.input_layernorm.weight",
                NativeTensorRole::AttentionNorm,
                Some(0),
                &[8],
                "attn_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(0),
                &[8, 8],
                "attn_q.bin",
                full_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(0),
                &[4, 8],
                "attn_k.bin",
                grouped_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(0),
                &[4, 8],
                "attn_v.bin",
                grouped_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.o_proj.weight",
                NativeTensorRole::AttentionO,
                Some(0),
                &[8, 8],
                "attn_o.bin",
                full_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.post_attention_layernorm.weight",
                NativeTensorRole::FfnNorm,
                Some(0),
                &[8],
                "ffn_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.gate_proj.weight",
                NativeTensorRole::FfnGate,
                Some(0),
                &[8, 8],
                "ffn_gate.bin",
                full_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.up_proj.weight",
                NativeTensorRole::FfnUp,
                Some(0),
                &[8, 8],
                "ffn_up.bin",
                full_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.down_proj.weight",
                NativeTensorRole::FfnDown,
                Some(0),
                &[8, 8],
                "ffn_down.bin",
                full_matrix_bytes,
            ),
        ],
    };

    fs::write(
        root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).expect("grouped projection manifest should serialize"),
    )
    .expect("grouped projection manifest should write");

    root_dir
}

#[cfg(target_os = "macos")]
fn write_multilayer_projection_native_model_fixture() -> PathBuf {
    let root_dir = write_projection_native_model_fixture();
    let zero_matrix = vec![0.0_f32; 64];
    write_f32_tensor_file(&root_dir, "attn_q_l1.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "attn_k_l1.bin", &zero_matrix);
    write_f32_tensor_file(&root_dir, "attn_v_l1.bin", &zero_matrix);

    let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("projection manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("projection manifest should parse");
    let vector_bytes = (8 * size_of::<f32>()) as u64;
    let matrix_bytes = (64 * size_of::<f32>()) as u64;
    manifest.layer_count = 2;
    manifest.tensors.extend([
        native_model_tensor_with_file(
            "model.layers.1.input_layernorm.weight",
            NativeTensorRole::AttentionNorm,
            Some(1),
            &[8],
            "attn_norm.bin",
            vector_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.self_attn.q_proj.weight",
            NativeTensorRole::AttentionQ,
            Some(1),
            &[8, 8],
            "attn_q_l1.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.self_attn.k_proj.weight",
            NativeTensorRole::AttentionK,
            Some(1),
            &[8, 8],
            "attn_k_l1.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.self_attn.v_proj.weight",
            NativeTensorRole::AttentionV,
            Some(1),
            &[8, 8],
            "attn_v_l1.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.self_attn.o_proj.weight",
            NativeTensorRole::AttentionO,
            Some(1),
            &[8, 8],
            "attn_o.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.post_attention_layernorm.weight",
            NativeTensorRole::FfnNorm,
            Some(1),
            &[8],
            "ffn_norm.bin",
            vector_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.mlp.gate_proj.weight",
            NativeTensorRole::FfnGate,
            Some(1),
            &[8, 8],
            "ffn_gate.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.mlp.up_proj.weight",
            NativeTensorRole::FfnUp,
            Some(1),
            &[8, 8],
            "ffn_up.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.1.mlp.down_proj.weight",
            NativeTensorRole::FfnDown,
            Some(1),
            &[8, 8],
            "ffn_down.bin",
            matrix_bytes,
        ),
    ]);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
    )
    .expect("projection manifest should write");

    root_dir
}

#[cfg(target_os = "macos")]
fn tail_projector_matrix(
    output_rows: usize,
    hidden_size: usize,
    tail_size: usize,
    scale: f32,
) -> Vec<f32> {
    let mut matrix = vec![0.0_f32; output_rows * hidden_size];
    let tail_start = hidden_size.saturating_sub(tail_size);
    for row in 0..output_rows.min(tail_size) {
        matrix[row * hidden_size + tail_start + row] = scale;
    }
    matrix
}

#[cfg(target_os = "macos")]
fn write_wide_projection_native_model_fixture() -> PathBuf {
    let root_dir = unique_test_dir("native-model-wide-projection");
    fs::create_dir_all(&root_dir).expect("wide projection fixture directory should create");

    let hidden_size = 40_usize;
    let head_width = 8_usize;
    let vocab_size = 5_usize;
    let tail_size = 8_usize;
    let mut embedding = vec![0.0_f32; vocab_size * hidden_size];
    for token in 1..vocab_size {
        let base = token * hidden_size + (hidden_size - tail_size);
        for lane in 0..tail_size {
            embedding[base + lane] = token as f32 + lane as f32;
        }
    }
    let ones = vec![1.0_f32; hidden_size];
    let q_proj = tail_projector_matrix(head_width, hidden_size, tail_size, 1.0);
    let k_proj = tail_projector_matrix(head_width, hidden_size, tail_size, 2.0);
    let v_proj = tail_projector_matrix(head_width, hidden_size, tail_size, 3.0);
    let attention_o = vec![0.0_f32; hidden_size * head_width];
    let zero_square = vec![0.0_f32; hidden_size * hidden_size];
    let zero_lm_head = vec![0.0_f32; vocab_size * hidden_size];

    write_f32_tensor_file(&root_dir, "embed.bin", &embedding);
    write_f32_tensor_file(&root_dir, "final_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "lm_head.bin", &zero_lm_head);
    write_f32_tensor_file(&root_dir, "attn_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "attn_q.bin", &q_proj);
    write_f32_tensor_file(&root_dir, "attn_k.bin", &k_proj);
    write_f32_tensor_file(&root_dir, "attn_v.bin", &v_proj);
    write_f32_tensor_file(&root_dir, "attn_o.bin", &attention_o);
    write_f32_tensor_file(&root_dir, "ffn_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "ffn_gate.bin", &zero_square);
    write_f32_tensor_file(&root_dir, "ffn_up.bin", &zero_square);
    write_f32_tensor_file(&root_dir, "ffn_down.bin", &zero_square);

    let hidden_vector_bytes = (hidden_size * size_of::<f32>()) as u64;
    let qkv_matrix_bytes = (head_width * hidden_size * size_of::<f32>()) as u64;
    let attention_o_bytes = (hidden_size * head_width * size_of::<f32>()) as u64;
    let square_matrix_bytes = (hidden_size * hidden_size * size_of::<f32>()) as u64;
    let embedding_bytes = (embedding.len() * size_of::<f32>()) as u64;
    let lm_head_bytes = (zero_lm_head.len() * size_of::<f32>()) as u64;

    let manifest = crate::model::NativeModelManifest {
        schema_version: crate::model::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: "qwen3_dense".to_string(),
        tensor_format: crate::model::NativeTensorFormat::Safetensors,
        layer_count: 1,
        hidden_size: hidden_size as u32,
        attention_head_count: 2,
        attention_head_dim: 4,
        kv_head_count: 2,
        vocab_size: vocab_size as u32,
        tie_word_embeddings: false,
        rope_theta: None,
        query_pre_attn_scalar: None,
        attention_logit_softcap: None,
        attention_value_from_key_layers: Vec::new(),
        attention_v_norm_no_scale_layers: Vec::new(),
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        tensors: vec![
            native_model_tensor_with_file(
                "model.embed_tokens.weight",
                NativeTensorRole::TokenEmbedding,
                None,
                &[vocab_size as u64, hidden_size as u64],
                "embed.bin",
                embedding_bytes,
            ),
            native_model_tensor_with_file(
                "model.norm.weight",
                NativeTensorRole::FinalNorm,
                None,
                &[hidden_size as u64],
                "final_norm.bin",
                hidden_vector_bytes,
            ),
            native_model_tensor_with_file(
                "lm_head.weight",
                NativeTensorRole::LmHead,
                None,
                &[vocab_size as u64, hidden_size as u64],
                "lm_head.bin",
                lm_head_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.input_layernorm.weight",
                NativeTensorRole::AttentionNorm,
                Some(0),
                &[hidden_size as u64],
                "attn_norm.bin",
                hidden_vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(0),
                &[head_width as u64, hidden_size as u64],
                "attn_q.bin",
                qkv_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(0),
                &[head_width as u64, hidden_size as u64],
                "attn_k.bin",
                qkv_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(0),
                &[head_width as u64, hidden_size as u64],
                "attn_v.bin",
                qkv_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.o_proj.weight",
                NativeTensorRole::AttentionO,
                Some(0),
                &[hidden_size as u64, head_width as u64],
                "attn_o.bin",
                attention_o_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.post_attention_layernorm.weight",
                NativeTensorRole::FfnNorm,
                Some(0),
                &[hidden_size as u64],
                "ffn_norm.bin",
                hidden_vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.gate_proj.weight",
                NativeTensorRole::FfnGate,
                Some(0),
                &[hidden_size as u64, hidden_size as u64],
                "ffn_gate.bin",
                square_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.up_proj.weight",
                NativeTensorRole::FfnUp,
                Some(0),
                &[hidden_size as u64, hidden_size as u64],
                "ffn_up.bin",
                square_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.down_proj.weight",
                NativeTensorRole::FfnDown,
                Some(0),
                &[hidden_size as u64, hidden_size as u64],
                "ffn_down.bin",
                square_matrix_bytes,
            ),
        ],
    };

    fs::write(
        root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).expect("wide projection manifest should serialize"),
    )
    .expect("wide projection manifest should write");

    root_dir
}

#[cfg(target_os = "macos")]
fn write_wide_direct_decode_native_model_fixture() -> PathBuf {
    let root_dir = unique_test_dir("native-model-wide-direct-decode");
    fs::create_dir_all(&root_dir).expect("wide direct decode fixture directory should create");

    let hidden_size = 40_usize;
    let head_width = 8_usize;
    let vocab_size = 5_usize;
    let tail_size = 8_usize;
    let mut embedding = vec![0.0_f32; vocab_size * hidden_size];
    let token_four_base = 4 * hidden_size + (hidden_size - tail_size);
    for lane in 0..tail_size {
        embedding[token_four_base + lane] = (lane + 1) as f32;
    }
    let ones = vec![1.0_f32; hidden_size];
    let zero_qkv = vec![0.0_f32; head_width * hidden_size];
    let zero_attention_o = vec![0.0_f32; hidden_size * head_width];
    let zero_square = vec![0.0_f32; hidden_size * hidden_size];
    let mut lm_head = vec![0.0_f32; vocab_size * hidden_size];
    let lm_head_token_two_base = 2 * hidden_size + (hidden_size - tail_size);
    for lane in 0..tail_size {
        lm_head[lm_head_token_two_base + lane] = 1.0;
    }

    write_f32_tensor_file(&root_dir, "embed.bin", &embedding);
    write_f32_tensor_file(&root_dir, "final_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "lm_head.bin", &lm_head);
    write_f32_tensor_file(&root_dir, "attn_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "attn_q.bin", &zero_qkv);
    write_f32_tensor_file(&root_dir, "attn_k.bin", &zero_qkv);
    write_f32_tensor_file(&root_dir, "attn_v.bin", &zero_qkv);
    write_f32_tensor_file(&root_dir, "attn_o.bin", &zero_attention_o);
    write_f32_tensor_file(&root_dir, "ffn_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "ffn_gate.bin", &zero_square);
    write_f32_tensor_file(&root_dir, "ffn_up.bin", &zero_square);
    write_f32_tensor_file(&root_dir, "ffn_down.bin", &zero_square);

    let hidden_vector_bytes = (hidden_size * size_of::<f32>()) as u64;
    let qkv_matrix_bytes = (head_width * hidden_size * size_of::<f32>()) as u64;
    let attention_o_bytes = (hidden_size * head_width * size_of::<f32>()) as u64;
    let square_matrix_bytes = (hidden_size * hidden_size * size_of::<f32>()) as u64;
    let embedding_bytes = (embedding.len() * size_of::<f32>()) as u64;
    let lm_head_bytes = (lm_head.len() * size_of::<f32>()) as u64;

    let manifest = crate::model::NativeModelManifest {
        schema_version: crate::model::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: "qwen3_dense".to_string(),
        tensor_format: crate::model::NativeTensorFormat::Safetensors,
        layer_count: 1,
        hidden_size: hidden_size as u32,
        attention_head_count: 2,
        attention_head_dim: 4,
        kv_head_count: 2,
        vocab_size: vocab_size as u32,
        tie_word_embeddings: false,
        rope_theta: None,
        query_pre_attn_scalar: None,
        attention_logit_softcap: None,
        attention_value_from_key_layers: Vec::new(),
        attention_v_norm_no_scale_layers: Vec::new(),
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        tensors: vec![
            native_model_tensor_with_file(
                "model.embed_tokens.weight",
                NativeTensorRole::TokenEmbedding,
                None,
                &[vocab_size as u64, hidden_size as u64],
                "embed.bin",
                embedding_bytes,
            ),
            native_model_tensor_with_file(
                "model.norm.weight",
                NativeTensorRole::FinalNorm,
                None,
                &[hidden_size as u64],
                "final_norm.bin",
                hidden_vector_bytes,
            ),
            native_model_tensor_with_file(
                "lm_head.weight",
                NativeTensorRole::LmHead,
                None,
                &[vocab_size as u64, hidden_size as u64],
                "lm_head.bin",
                lm_head_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.input_layernorm.weight",
                NativeTensorRole::AttentionNorm,
                Some(0),
                &[hidden_size as u64],
                "attn_norm.bin",
                hidden_vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(0),
                &[head_width as u64, hidden_size as u64],
                "attn_q.bin",
                qkv_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(0),
                &[head_width as u64, hidden_size as u64],
                "attn_k.bin",
                qkv_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(0),
                &[head_width as u64, hidden_size as u64],
                "attn_v.bin",
                qkv_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.o_proj.weight",
                NativeTensorRole::AttentionO,
                Some(0),
                &[hidden_size as u64, head_width as u64],
                "attn_o.bin",
                attention_o_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.post_attention_layernorm.weight",
                NativeTensorRole::FfnNorm,
                Some(0),
                &[hidden_size as u64],
                "ffn_norm.bin",
                hidden_vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.gate_proj.weight",
                NativeTensorRole::FfnGate,
                Some(0),
                &[hidden_size as u64, hidden_size as u64],
                "ffn_gate.bin",
                square_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.up_proj.weight",
                NativeTensorRole::FfnUp,
                Some(0),
                &[hidden_size as u64, hidden_size as u64],
                "ffn_up.bin",
                square_matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.down_proj.weight",
                NativeTensorRole::FfnDown,
                Some(0),
                &[hidden_size as u64, hidden_size as u64],
                "ffn_down.bin",
                square_matrix_bytes,
            ),
        ],
    };

    fs::write(
        root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).expect("wide direct decode manifest should serialize"),
    )
    .expect("wide direct decode manifest should write");

    root_dir
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectDecodeFixtureGateUpLayout {
    Split,
    Packed,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectDecodeFixtureVariant {
    ProjectionOnly,
    FfnContinuation,
}

#[cfg(target_os = "macos")]
fn write_direct_decode_native_model_fixture_with_variant(
    tie_word_embeddings: bool,
    gate_up_layout: DirectDecodeFixtureGateUpLayout,
    variant: DirectDecodeFixtureVariant,
) -> PathBuf {
    let root_dir = unique_test_dir("native-model-direct-decode");
    fs::create_dir_all(&root_dir).expect("decode fixture directory should create");

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
    let projection_lm_head = embedding.clone();
    let continuation_lm_head = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
    ];
    let (ffn_gate, ffn_up, ffn_down, lm_head) = match variant {
        DirectDecodeFixtureVariant::ProjectionOnly => (
            zero_matrix.clone(),
            zero_matrix.clone(),
            zero_matrix.clone(),
            projection_lm_head,
        ),
        DirectDecodeFixtureVariant::FfnContinuation => {
            let mut gate = zero_matrix.clone();
            let mut up = zero_matrix.clone();
            let mut down = zero_matrix.clone();
            gate[0] = 1.0;
            up[0] = 0.5;
            down[2 * 8] = 3.0;
            (gate, up, down, continuation_lm_head)
        }
    };
    let packed_gate_up = ffn_gate
        .iter()
        .chain(ffn_up.iter())
        .copied()
        .collect::<Vec<_>>();

    write_f32_tensor_file(&root_dir, "embed.bin", &embedding);
    write_f32_tensor_file(&root_dir, "final_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "attn_norm.bin", &ones);
    write_f32_tensor_file(&root_dir, "attn_q.bin", &identity);
    write_f32_tensor_file(&root_dir, "attn_k.bin", &double_identity);
    write_f32_tensor_file(&root_dir, "attn_v.bin", &triple_identity);
    write_f32_tensor_file(&root_dir, "attn_o.bin", &identity);
    write_f32_tensor_file(&root_dir, "ffn_norm.bin", &ones);
    match gate_up_layout {
        DirectDecodeFixtureGateUpLayout::Split => {
            write_f32_tensor_file(&root_dir, "ffn_gate.bin", &ffn_gate);
            write_f32_tensor_file(&root_dir, "ffn_up.bin", &ffn_up);
        }
        DirectDecodeFixtureGateUpLayout::Packed => {
            write_f32_tensor_file(&root_dir, "ffn_gate_up.bin", &packed_gate_up);
        }
    }
    write_f32_tensor_file(&root_dir, "ffn_down.bin", &ffn_down);
    if !tie_word_embeddings {
        write_f32_tensor_file(&root_dir, "lm_head.bin", &lm_head);
    }

    let matrix_bytes = (64 * size_of::<f32>()) as u64;
    let packed_matrix_bytes = (128 * size_of::<f32>()) as u64;
    let vector_bytes = (8 * size_of::<f32>()) as u64;
    let embedding_bytes = (embedding.len() * size_of::<f32>()) as u64;
    let mut tensors = vec![
        native_model_tensor_with_file(
            "model.embed_tokens.weight",
            NativeTensorRole::TokenEmbedding,
            None,
            &[5, 8],
            "embed.bin",
            embedding_bytes,
        ),
        native_model_tensor_with_file(
            "model.norm.weight",
            NativeTensorRole::FinalNorm,
            None,
            &[8],
            "final_norm.bin",
            vector_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.0.input_layernorm.weight",
            NativeTensorRole::AttentionNorm,
            Some(0),
            &[8],
            "attn_norm.bin",
            vector_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.0.self_attn.q_proj.weight",
            NativeTensorRole::AttentionQ,
            Some(0),
            &[8, 8],
            "attn_q.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.0.self_attn.k_proj.weight",
            NativeTensorRole::AttentionK,
            Some(0),
            &[8, 8],
            "attn_k.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.0.self_attn.v_proj.weight",
            NativeTensorRole::AttentionV,
            Some(0),
            &[8, 8],
            "attn_v.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.0.self_attn.o_proj.weight",
            NativeTensorRole::AttentionO,
            Some(0),
            &[8, 8],
            "attn_o.bin",
            matrix_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.0.post_attention_layernorm.weight",
            NativeTensorRole::FfnNorm,
            Some(0),
            &[8],
            "ffn_norm.bin",
            vector_bytes,
        ),
        native_model_tensor_with_file(
            "model.layers.0.mlp.down_proj.weight",
            NativeTensorRole::FfnDown,
            Some(0),
            &[8, 8],
            "ffn_down.bin",
            matrix_bytes,
        ),
    ];
    match gate_up_layout {
        DirectDecodeFixtureGateUpLayout::Split => {
            tensors.push(native_model_tensor_with_file(
                "model.layers.0.mlp.gate_proj.weight",
                NativeTensorRole::FfnGate,
                Some(0),
                &[8, 8],
                "ffn_gate.bin",
                matrix_bytes,
            ));
            tensors.push(native_model_tensor_with_file(
                "model.layers.0.mlp.up_proj.weight",
                NativeTensorRole::FfnUp,
                Some(0),
                &[8, 8],
                "ffn_up.bin",
                matrix_bytes,
            ));
        }
        DirectDecodeFixtureGateUpLayout::Packed => {
            tensors.push(native_model_tensor_with_file(
                "model.layers.0.mlp.gate_up_proj.weight",
                NativeTensorRole::FfnGateUpPacked,
                Some(0),
                &[16, 8],
                "ffn_gate_up.bin",
                packed_matrix_bytes,
            ));
        }
    }
    if !tie_word_embeddings {
        tensors.insert(
            2,
            native_model_tensor_with_file(
                "lm_head.weight",
                NativeTensorRole::LmHead,
                None,
                &[5, 8],
                "lm_head.bin",
                embedding_bytes,
            ),
        );
    }

    let manifest = crate::model::NativeModelManifest {
        schema_version: crate::model::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: "qwen3_dense".to_string(),
        tensor_format: crate::model::NativeTensorFormat::Safetensors,
        layer_count: 1,
        hidden_size: 8,
        attention_head_count: 2,
        attention_head_dim: 4,
        kv_head_count: 2,
        vocab_size: 5,
        tie_word_embeddings,
        rope_theta: None,
        query_pre_attn_scalar: None,
        attention_logit_softcap: None,
        attention_value_from_key_layers: Vec::new(),
        attention_v_norm_no_scale_layers: Vec::new(),
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        tensors,
    };

    fs::write(
        root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).expect("decode manifest should serialize"),
    )
    .expect("decode manifest should write");

    root_dir
}

#[cfg(target_os = "macos")]
fn write_direct_decode_native_model_fixture(tie_word_embeddings: bool) -> PathBuf {
    write_direct_decode_native_model_fixture_with_variant(
        tie_word_embeddings,
        DirectDecodeFixtureGateUpLayout::Split,
        DirectDecodeFixtureVariant::ProjectionOnly,
    )
}

#[cfg(target_os = "macos")]
fn write_gemma_direct_decode_native_model_fixture(tie_word_embeddings: bool) -> PathBuf {
    let root_dir = write_direct_decode_native_model_fixture(tie_word_embeddings);
    let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("decode manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("decode manifest should parse");
    manifest.model_family = "gemma2_dense".to_string();
    fs::write(
        root_dir.join("final_norm.bin"),
        vec![0_u8; 8 * size_of::<f32>()],
    )
    .expect("gemma final norm should write");
    fs::write(
        root_dir.join("ffn_norm.bin"),
        vec![0_u8; 8 * size_of::<f32>()],
    )
    .expect("gemma ffn norm should write");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("decode manifest should serialize"),
    )
    .expect("decode manifest should rewrite");

    root_dir
}

#[cfg(target_os = "macos")]
fn write_ffn_decode_native_model_fixture(
    gate_up_layout: DirectDecodeFixtureGateUpLayout,
) -> PathBuf {
    write_direct_decode_native_model_fixture_with_variant(
        false,
        gate_up_layout,
        DirectDecodeFixtureVariant::FfnContinuation,
    )
}

#[cfg(target_os = "macos")]
fn write_multilayer_direct_decode_native_model_fixture() -> PathBuf {
    let root_dir = write_direct_decode_native_model_fixture(false);
    let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("manifest should parse");
    let layer_zero_tensors = manifest
        .tensors
        .iter()
        .filter(|tensor| tensor.layer_index == Some(0))
        .cloned()
        .collect::<Vec<_>>();

    for mut tensor in layer_zero_tensors {
        tensor.layer_index = Some(1);
        tensor.name = tensor.name.replace("layers.0", "layers.1");
        manifest.tensors.push(tensor);
    }
    manifest.layer_count = 2;

    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
    )
    .expect("manifest should write");

    root_dir
}

#[cfg(target_os = "macos")]
fn assert_f32_slice_close(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= epsilon,
            "index {index}: actual={actual} expected={expected} epsilon={epsilon}"
        );
    }
}

#[cfg(target_os = "macos")]
fn neox_rotate_reference(values: &[f32], position: f32) -> Vec<f32> {
    neox_rotate_reference_with_base(values, position, PHASE1_MODEL_STAGE_ROPE_FREQ_BASE)
}

#[cfg(target_os = "macos")]
fn neox_rotate_reference_with_base(values: &[f32], position: f32, freq_base: f32) -> Vec<f32> {
    let half = values.len() / 2;
    let neg_log_base = -(freq_base.ln());
    let dim_inv = 2.0 / values.len() as f32;
    let mut rotated = values.to_vec();

    for index in 0..half {
        let freq = (neg_log_base * index as f32 * dim_inv).exp();
        let theta = position * freq;
        let (sin_theta, cos_theta) = theta.sin_cos();
        let low = values[index];
        let high = values[index + half];
        rotated[index] = low * cos_theta - high * sin_theta;
        rotated[index + half] = low * sin_theta + high * cos_theta;
    }

    rotated
}

#[cfg(target_os = "macos")]
fn interleaved_rotate_reference(values: &[f32], position: f32) -> Vec<f32> {
    interleaved_rotate_reference_with_base(values, position, PHASE1_MODEL_STAGE_ROPE_FREQ_BASE)
}

#[cfg(target_os = "macos")]
fn interleaved_rotate_reference_with_base(
    values: &[f32],
    position: f32,
    freq_base: f32,
) -> Vec<f32> {
    let half = values.len() / 2;
    let neg_log_base = -(freq_base.ln());
    let dim_inv = 2.0 / values.len() as f32;
    let mut rotated = values.to_vec();

    for index in 0..half {
        let freq = (neg_log_base * index as f32 * dim_inv).exp();
        let theta = position * freq;
        let (sin_theta, cos_theta) = theta.sin_cos();
        let pair_base = index * 2;
        let even = values[pair_base];
        let odd = values[pair_base + 1];
        rotated[pair_base] = even * cos_theta - odd * sin_theta;
        rotated[pair_base + 1] = odd * cos_theta + even * sin_theta;
    }

    rotated
}

#[cfg(target_os = "macos")]
fn rms_normalize_reference(values: &[f32], epsilon: f32) -> Vec<f32> {
    let mean_square = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let denom = (mean_square + epsilon).sqrt();
    values.iter().map(|value| value / denom).collect()
}

#[cfg(target_os = "macos")]
fn scale_reference(values: &[f32], factor: f32) -> Vec<f32> {
    values.iter().map(|value| value * factor).collect()
}

#[cfg(target_os = "macos")]
fn manual_attention_head_output(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    token_index: usize,
    head_index: usize,
    attention_config: ReferenceAttentionConfig,
) -> Vec<f32> {
    let reference = reference_numeric_path_with_inputs(workload, staged_inputs);
    let head_size = workload.numeric_layout.head_size() as usize;
    let head_dim = workload.numeric_layout.head_dim as usize;
    let batch_id = batch_id_for_token(
        &workload.kv_metadata.scheduled_cu_seq_lens,
        token_index as u32,
    );
    let context_begin = workload.kv_metadata.cu_seq_lens[batch_id] as usize;
    let context_end = workload.kv_metadata.cu_seq_lens[batch_id + 1] as usize;
    let query_base = token_index * head_size + head_index * head_dim;
    let query = &staged_inputs.query[query_base..query_base + head_dim];

    let mut scores = Vec::new();
    for context_index in context_begin..context_end {
        let context_base = context_index * head_size + head_index * head_dim;
        let key = &reference.gather_key[context_base..context_base + head_dim];
        scores.push(configured_attention_score(query, key, attention_config));
    }

    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut accum = vec![0.0_f32; head_dim];
    let mut weight_sum = 0.0_f32;
    for (relative_index, context_index) in (context_begin..context_end).enumerate() {
        let weight = (scores[relative_index] - max_score).exp();
        let context_base = context_index * head_size + head_index * head_dim;
        let value = &reference.gather_value[context_base..context_base + head_dim];
        weight_sum += weight;
        for lane in 0..head_dim {
            accum[lane] += weight * value[lane];
        }
    }
    for lane in &mut accum {
        *lane /= weight_sum.max(0.000001);
    }

    accum
}

#[cfg(target_os = "macos")]
#[test]
fn model_gate_up_product_uses_family_specific_activation() {
    let qwen_dir = write_projection_native_model_fixture();
    let qwen_artifacts =
        NativeModelArtifacts::from_dir(&qwen_dir).expect("qwen artifacts should load");
    let gemma_dir = write_gemma_projection_native_model_fixture();
    let gemma_artifacts =
        NativeModelArtifacts::from_dir(&gemma_dir).expect("gemma artifacts should load");

    let gate = vec![-1.0_f32, 0.5, 1.0];
    let up = vec![2.0_f32, 3.0, 4.0];
    let mut qwen_gate = gate.clone();
    let mut gemma_gate = gate.clone();

    apply_model_gate_up_product(&qwen_artifacts, &mut qwen_gate, &up);
    apply_model_gate_up_product(&gemma_artifacts, &mut gemma_gate, &up);

    let expected_qwen = gate
        .iter()
        .zip(&up)
        .map(|(gate_value, up_value)| silu(*gate_value) * *up_value)
        .collect::<Vec<_>>();
    let expected_gemma = gate
        .iter()
        .zip(&up)
        .map(|(gate_value, up_value)| gelu_approx(*gate_value) * *up_value)
        .collect::<Vec<_>>();

    assert_f32_slice_close(&qwen_gate, &expected_qwen, 1e-6);
    assert_f32_slice_close(&gemma_gate, &expected_gemma, 1e-6);
    assert_ne!(qwen_gate, gemma_gate);

    let _ = fs::remove_dir_all(qwen_dir);
    let _ = fs::remove_dir_all(gemma_dir);
}

#[cfg(target_os = "macos")]
fn per_head_rms_norm_reference(values: &[f32], head_dim: usize, weights: &[f32]) -> Vec<f32> {
    values
        .chunks_exact(head_dim)
        .flat_map(|head| {
            rms_normalize_reference(head, 1e-6)
                .into_iter()
                .zip(weights.iter().copied())
                .map(|(value, weight)| value * weight)
                .collect::<Vec<_>>()
        })
        .collect()
}

#[cfg(target_os = "macos")]
#[test]
fn native_dense_shadow_bytes_promote_byte_tensors_to_f32() {
    let u8_spec = NativeTensorSpec {
        name: "u8_weight".to_string(),
        role: NativeTensorRole::AttentionO,
        layer_index: Some(0),
        dtype: NativeTensorDataType::U8,
        shape: vec![2, 2],
        file: PathBuf::from("weights.bin"),
        offset_bytes: 0,
        length_bytes: 4,
    };
    let (u8_dtype, u8_shadow) = native_dense_shadow_bytes(&u8_spec, &[1, 2, 3, 4])
        .expect("u8 weights should promote into f32 shadow bytes");

    assert_eq!(u8_dtype, NativeTensorDataType::F32);
    assert_eq!(
        u8_shadow
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk width should match")))
            .collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    let i8_spec = NativeTensorSpec {
        name: "i8_weight".to_string(),
        role: NativeTensorRole::FfnNorm,
        layer_index: Some(0),
        dtype: NativeTensorDataType::I8,
        shape: vec![4],
        file: PathBuf::from("weights.bin"),
        offset_bytes: 0,
        length_bytes: 4,
    };
    let (i8_dtype, i8_shadow) = native_dense_shadow_bytes(&i8_spec, &[255, 0, 1, 127])
        .expect("i8 weights should promote into f32 shadow bytes");

    assert_eq!(i8_dtype, NativeTensorDataType::F32);
    assert_eq!(
        i8_shadow
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk width should match")))
            .collect::<Vec<_>>(),
        vec![-1.0, 0.0, 1.0, 127.0]
    );
}

#[cfg(target_os = "macos")]
#[test]
fn native_dense_kernel_coverage_counts_byte_tensors_as_promoted_f32_support() {
    let byte_projection = MetalNativeTensorBinding {
        spec: NativeTensorSpec {
            name: "projection".to_string(),
            role: NativeTensorRole::AttentionO,
            layer_index: Some(0),
            dtype: NativeTensorDataType::U8,
            shape: vec![2, 2],
            file: PathBuf::from("projection.bin"),
            offset_bytes: 0,
            length_bytes: 4,
        },
        resolved_path: PathBuf::from("/tmp/projection.bin"),
    };
    let byte_norm = MetalNativeTensorBinding {
        spec: NativeTensorSpec {
            name: "norm".to_string(),
            role: NativeTensorRole::FfnNorm,
            layer_index: Some(0),
            dtype: NativeTensorDataType::I8,
            shape: vec![4],
            file: PathBuf::from("norm.bin"),
            offset_bytes: 0,
            length_bytes: 4,
        },
        resolved_path: PathBuf::from("/tmp/norm.bin"),
    };
    let mut coverage = MetalNativeDenseKernelCoverage::default();

    record_projection_binding_coverage(&mut coverage, &byte_projection);
    record_rms_norm_binding_coverage(&mut coverage, &byte_norm);

    assert_eq!(coverage.projection_f32_binding_count, 1);
    assert_eq!(coverage.rms_norm_f32_binding_count, 1);
    assert_eq!(coverage.projection_unsupported_binding_count, 0);
    assert_eq!(coverage.rms_norm_unsupported_binding_count, 0);
}

#[test]
fn metal_assets_load_compiled_build_and_resolve_required_kernels() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);

    let assets = MetalKernelAssets::from_build_dir(&fixture.build_dir).expect("assets should load");

    assert_eq!(assets.build_status(), MetalBuildStatus::Compiled);
    assert_eq!(assets.manifest().library_name, PHASE1_METAL_LIBRARY_NAME);
    assert_eq!(
        assets.default_block_size_tokens(),
        PHASE1_DEFAULT_BLOCK_SIZE_TOKENS
    );
    assert_eq!(
        assets.supported_block_size_tokens(),
        PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS
    );
    assert_eq!(
        assets
            .required_kernel("reshape_and_cache")
            .expect("required kernel should resolve")
            .tier,
        MetalKernelTier::Required
    );
    assert_eq!(
        assets.kernel("kv_scale_update").map(|kernel| kernel.tier),
        Some(MetalKernelTier::Optional)
    );
    assert_eq!(
        assets.kernel("swap_blocks").map(|kernel| kernel.tier),
        Some(MetalKernelTier::Deferred)
    );
    assert!(assets.compiled_metallib_path().is_some());
    assert!(!assets
        .compiled_metallib_bytes()
        .expect("compiled metallib should load")
        .is_empty());

    fixture.cleanup();
}

#[test]
fn native_model_bindings_cover_all_manifest_tensors() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");

    assert_eq!(
        bindings.flattened_tensor_bindings().len() as u32,
        artifacts.summary().tensor_count
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn native_model_buffers_bind_real_tensor_bytes_into_metal_shared_buffers() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");

    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let stats = buffers.stats();

    // The projection fixture uses real tensor data with varying per-tensor sizes
    // (embedding: 160 B, 3 vectors: 32 B each, 8 matrices: 256 B each = 2304 B total).
    let expected_bytes: u64 = artifacts
        .manifest()
        .tensors
        .iter()
        .map(|t| t.length_bytes)
        .sum();
    assert!(stats.buffers_bound);
    assert_eq!(stats.buffer_count, artifacts.summary().tensor_count);
    assert_eq!(stats.buffer_bytes, expected_bytes);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_bringup_runner_executes_single_layer_fixture_with_real_build_artifacts() {
    let Some(build_dir) = compiled_repo_metal_build_dir() else {
        return;
    };
    let model_dir = write_projection_native_model_fixture();
    let runner =
        MetalBringupRunner::from_build_dir_and_model_artifacts(&build_dir, Some(&model_dir))
            .expect("metal bring-up runner should initialize");

    let output = runner.run(sample_runner_input());

    assert_eq!(
        output.execution_status,
        ExecutionStatus::Success,
        "{output:?}"
    );
    assert!(output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_runtime_real_model_tensor_inputs" && *value > 0
        ));
    assert!(output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_runtime_complete_model_forward_supported"
                && *value > 0
        ));
    assert!(output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(|(key, value)| key == "metal_dispatch_real_model_forward" && *value > 0));
    assert!(output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_direct_decode_native_logits_projection"
                && *value > 0
        ));
    assert!(output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_direct_decode_native_projection_row_count"
                && *value > 0
        ));
    assert!(output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_direct_decode_cpu_projection_row_count"
                && *value == 0
        ));

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_bringup_runner_reuses_multilayer_prefix_cache_for_native_decode_continuation() {
    let Some(build_dir) = compiled_repo_metal_build_dir() else {
        return;
    };
    let model_dir = write_multilayer_direct_decode_native_model_fixture();
    let runner =
        MetalBringupRunner::from_build_dir_and_model_artifacts(&build_dir, Some(&model_dir))
            .expect("metal bring-up runner should initialize");

    let prefill_output = runner.run(sample_prefill_only_runner_input());

    assert_eq!(
        prefill_output.execution_status,
        ExecutionStatus::Success,
        "{prefill_output:?}"
    );
    assert!(prefill_output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(|(key, value)| key == "metal_dispatch_prefix_layers_native_attention" && *value > 0));
    assert!(prefill_output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_prefix_cpu_reference_dispatch_count"
                && *value == 0
        ));
    assert!(prefill_output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_runtime_complete_model_forward_supported"
                && *value > 0
        ));

    let continuation_output = runner.run(sample_decode_continuation_runner_input());

    assert_eq!(
        continuation_output.execution_status,
        ExecutionStatus::Success,
        "{continuation_output:?}"
    );
    assert!(continuation_output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(|(key, value)| key == "metal_dispatch_prefix_layers_native_attention" && *value > 0));
    assert!(continuation_output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_prefix_cpu_reference_dispatch_count"
                && *value == 0
        ));
    assert!(continuation_output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_runtime_complete_model_forward_supported"
                && *value > 0
        ));
    assert!(continuation_output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(|(key, value)| key == "metal_dispatch_real_model_forward" && *value > 0));
    assert!(continuation_output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_direct_decode_native_logits_projection"
                && *value > 0
        ));
    assert!(continuation_output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_direct_decode_cpu_projection_row_count"
                && *value == 0
        ));
    assert!(continuation_output
        .logits_outputs
        .iter()
        .any(|output| output.request_id == RequestId(17) && !output.logits.is_empty()));

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_bringup_runner_batches_direct_decode_logits_for_multiple_requests() {
    let Some(build_dir) = compiled_repo_metal_build_dir() else {
        return;
    };
    let model_dir = write_direct_decode_native_model_fixture(false);
    let runner =
        MetalBringupRunner::from_build_dir_and_model_artifacts(&build_dir, Some(&model_dir))
            .expect("metal bring-up runner should initialize");

    let output = runner.run(sample_decode_only_runner_input());

    assert_eq!(
        output.execution_status,
        ExecutionStatus::Success,
        "{output:?}"
    );
    assert!(output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(|(key, value)| key == "metal_dispatch_real_model_forward" && *value > 0));
    assert!(output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_direct_decode_batched_logits_group_count"
                && *value > 0
        ));
    assert!(output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_direct_decode_batched_logits_token_count"
                && *value >= 2
        ));
    assert!(output
        .route_metadata
        .crossover_decisions
        .iter()
        .any(
            |(key, value)| key == "metal_dispatch_direct_decode_cpu_projection_row_count"
                && *value == 0
        ));
    assert_eq!(output.logits_outputs.len(), 2);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_staged_inputs_use_bound_tensor_bytes() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let input = sample_runner_input();
    let staged = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("projection-backed staged inputs should resolve");
    let synthetic = synthetic_staged_inputs(&workload);

    assert_eq!(
        staged.source,
        MetalStagedInputSource::ModelConditionedMiniProjection
    );
    let expected_hidden = rms_normalize_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1e-6);
    assert_f32_slice_close(&staged.query[..8], &expected_hidden, 1e-6);
    assert_f32_slice_close(
        &staged.key[..8],
        &scale_reference(&expected_hidden, 2.0),
        1e-6,
    );
    assert_f32_slice_close(
        &staged.value[..8],
        &scale_reference(&expected_hidden, 3.0),
        1e-6,
    );
    assert_ne!(staged.query, synthetic.query);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn batched_row_rms_norm_matches_per_row_reference_path() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let attention_norm = buffers
        .binding_for(&bindings.layers[0].attention_norm)
        .expect("attention norm binding should resolve");
    let row_width = tensor_element_count(&attention_norm.meta.spec)
        .expect("attention norm tensor should have a shape");
    let epsilon = native_model_rms_norm_epsilon(&artifacts);
    let weight_offset = native_model_rms_norm_weight_offset(&artifacts);
    let weights =
        tensor_prefix_f32(attention_norm, row_width).expect("attention norm weights exist");
    let mut rows = vec![
        (0..row_width)
            .map(|index| index as f32 + 1.0)
            .collect::<Vec<_>>(),
        (0..row_width)
            .map(|index| (index as f32 + 1.0) * 0.5)
            .collect::<Vec<_>>(),
    ];
    let mut expected = rows.clone();
    for row in &mut expected {
        apply_rms_norm_with_weights_in_place(row, &weights, epsilon, weight_offset)
            .expect("per-row reference rms norm should succeed");
    }

    let tally = apply_batched_row_rms_norm_with_binding_in_place_with_tally(
        &mut rows,
        row_width,
        attention_norm,
        epsilon,
        weight_offset,
        None,
    )
    .expect("batched row rms norm should succeed");

    assert_eq!(rows, expected);
    assert_eq!(tally.native_rms_norm_element_count(), 0);
    assert_eq!(
        tally.cpu_rms_norm_element_count(),
        saturating_usize_to_u32(row_width * 2)
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_staged_inputs_apply_optional_qk_head_norms() {
    let model_dir = write_projection_qk_norm_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let input = sample_runner_input();
    let staged = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("qk norm staged inputs should resolve");

    let expected_hidden = rms_normalize_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1e-6);
    let expected_query = per_head_rms_norm_reference(&expected_hidden, 4, &[2.0, 1.0, 1.0, 1.0]);
    let expected_key = per_head_rms_norm_reference(
        &scale_reference(&expected_hidden, 2.0),
        4,
        &[3.0, 1.0, 1.0, 1.0],
    );

    assert_f32_slice_close(&staged.query[..8], &expected_query, 1e-6);
    assert_f32_slice_close(&staged.key[..8], &expected_key, 1e-6);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_staged_inputs_reuse_key_projection_for_missing_value_weights() {
    let model_dir = write_projection_value_from_key_native_model_fixture(false);
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let token_embedding = buffers
        .binding_for(&bindings.token_embedding)
        .expect("token embedding binding should resolve");
    let hidden_states = initial_model_hidden_states_cpu(
        token_embedding,
        &workload,
        8,
        native_model_embedding_scale(&artifacts),
    )
    .expect("hidden states should resolve");
    let layer = bindings
        .layers
        .first()
        .expect("fixture should include one layer");
    let attention_norm = buffers
        .binding_for(&layer.attention_norm)
        .expect("attention norm should bind");
    let attention_k_norm = layer
        .attention_k_norm
        .as_ref()
        .and_then(|binding| buffers.binding_for(binding))
        .expect("attention k norm should bind");

    let mut normalized_hidden_states = hidden_states.clone();
    apply_batched_row_rms_norm_with_binding_in_place_with_tally(
        &mut normalized_hidden_states,
        8,
        attention_norm,
        native_model_rms_norm_epsilon(&artifacts),
        native_model_rms_norm_weight_offset(&artifacts),
        None,
    )
    .expect("attention norm should apply");
    let stage_dims =
        resolved_model_stage_dims_for_input_width(&artifacts, layer, attention_norm, &buffers, 8)
            .expect("stage dims should resolve");
    let (_, mut expected_key_rows, expected_value_rows, _) =
        project_batched_attention_qkv_with_dims_and_tally(
            &artifacts,
            layer
                .attention_qkv
                .as_ref()
                .expect("attention_qkv should exist"),
            &buffers,
            &normalized_hidden_states,
            8,
            stage_dims,
            None,
        )
        .expect("qkv projection should resolve");
    apply_batched_per_head_rms_norm_rows_with_tally(
        &mut expected_key_rows,
        stage_dims.kv_heads,
        stage_dims.head_dim,
        attention_k_norm,
        native_model_rms_norm_epsilon(&artifacts),
        native_model_rms_norm_weight_offset(&artifacts),
        None,
    )
    .expect("k norm should apply");

    let staged =
        stage_model_layer_qkv_inputs(&artifacts, layer, &buffers, &workload, &hidden_states, None)
            .expect("staged inputs should resolve");

    assert_f32_slice_close(&staged.key[..8], &expected_key_rows[0], 1e-6);
    assert_f32_slice_close(&staged.value[..8], &expected_value_rows[0], 1e-6);
    assert_ne!(&staged.key[..8], &staged.value[..8]);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_staged_inputs_apply_no_scale_value_norm_for_missing_value_weights() {
    let model_dir = write_projection_value_from_key_native_model_fixture(true);
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let token_embedding = buffers
        .binding_for(&bindings.token_embedding)
        .expect("token embedding binding should resolve");
    let hidden_states = initial_model_hidden_states_cpu(
        token_embedding,
        &workload,
        8,
        native_model_embedding_scale(&artifacts),
    )
    .expect("hidden states should resolve");
    let layer = bindings
        .layers
        .first()
        .expect("fixture should include one layer");
    let attention_norm = buffers
        .binding_for(&layer.attention_norm)
        .expect("attention norm should bind");

    let mut normalized_hidden_states = hidden_states.clone();
    apply_batched_row_rms_norm_with_binding_in_place_with_tally(
        &mut normalized_hidden_states,
        8,
        attention_norm,
        native_model_rms_norm_epsilon(&artifacts),
        native_model_rms_norm_weight_offset(&artifacts),
        None,
    )
    .expect("attention norm should apply");
    let stage_dims =
        resolved_model_stage_dims_for_input_width(&artifacts, layer, attention_norm, &buffers, 8)
            .expect("stage dims should resolve");
    let (_, _, mut expected_value_rows, _) = project_batched_attention_qkv_with_dims_and_tally(
        &artifacts,
        layer
            .attention_qkv
            .as_ref()
            .expect("attention_qkv should exist"),
        &buffers,
        &normalized_hidden_states,
        8,
        stage_dims,
        None,
    )
    .expect("qkv projection should resolve");
    apply_batched_per_head_rms_norm_rows_without_weights_with_tally(
        &mut expected_value_rows,
        stage_dims.kv_heads,
        stage_dims.head_dim,
        native_model_rms_norm_epsilon(&artifacts),
        None,
    )
    .expect("value norm should apply");

    let staged =
        stage_model_layer_qkv_inputs(&artifacts, layer, &buffers, &workload, &hidden_states, None)
            .expect("staged inputs should resolve");

    assert_f32_slice_close(&staged.value[..8], &expected_value_rows[0], 1e-6);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_staged_inputs_apply_gemma_weight_offset_norms() {
    let model_dir = write_gemma_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let input = sample_runner_input();
    let staged = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("gemma staged inputs should resolve");

    let expected_hidden = rms_normalize_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1e-6);
    assert_f32_slice_close(&staged.query[..8], &expected_hidden, 1e-6);
    assert_f32_slice_close(
        &staged.key[..8],
        &scale_reference(&expected_hidden, 2.0),
        1e-6,
    );
    assert_f32_slice_close(
        &staged.value[..8],
        &scale_reference(&expected_hidden, 3.0),
        1e-6,
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_hidden_states_before_final_layer_apply_gemma_embedding_scale() {
    let model_dir = write_gemma_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let workload =
        MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");

    let (hidden_states, final_layer_index, _) = model_hidden_states_before_final_layer(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("gemma hidden states should resolve");

    let scale = (8.0_f32).sqrt();
    assert_eq!(final_layer_index, 0);
    assert_f32_slice_close(
        &hidden_states[0],
        &scale_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], scale),
        1e-6,
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_staged_inputs_expand_grouped_kv_heads_to_query_layout() {
    let model_dir = write_grouped_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let input = sample_runner_input();
    let staged = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("grouped projection staged inputs should resolve");

    assert_eq!(staged.layout, MetalDispatchNumericLayout::new(4, 2));
    let expected_hidden = rms_normalize_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1e-6);
    assert_f32_slice_close(&staged.query[..8], &expected_hidden, 1e-6);
    assert_f32_slice_close(
        &staged.key[..8],
        &[
            expected_hidden[0] * 2.0,
            expected_hidden[1] * 2.0,
            expected_hidden[0] * 2.0,
            expected_hidden[1] * 2.0,
            expected_hidden[2] * 2.0,
            expected_hidden[3] * 2.0,
            expected_hidden[2] * 2.0,
            expected_hidden[3] * 2.0,
        ],
        1e-6,
    );
    assert_f32_slice_close(
        &staged.value[..8],
        &[
            expected_hidden[0] * 3.0,
            expected_hidden[1] * 3.0,
            expected_hidden[0] * 3.0,
            expected_hidden[1] * 3.0,
            expected_hidden[2] * 3.0,
            expected_hidden[3] * 3.0,
            expected_hidden[2] * 3.0,
            expected_hidden[3] * 3.0,
        ],
        1e-6,
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_staged_inputs_apply_qwen_rope_for_nonzero_positions() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let input = sample_runner_input();
    let staged = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("projection-backed staged inputs should resolve");
    let normalized_hidden =
        rms_normalize_reference(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1e-6);
    let expected_query = [
        neox_rotate_reference(&normalized_hidden[..4], 1.0),
        neox_rotate_reference(&normalized_hidden[4..8], 1.0),
    ]
    .concat();
    let doubled_hidden = scale_reference(&normalized_hidden, 2.0);
    let expected_key = [
        neox_rotate_reference(&doubled_hidden[..4], 1.0),
        neox_rotate_reference(&doubled_hidden[4..8], 1.0),
    ]
    .concat();

    assert_eq!(workload.scheduled_positions, vec![0, 1, 2, 3]);
    assert_f32_slice_close(&staged.query[8..16], &expected_query, 1e-5);
    assert_f32_slice_close(&staged.key[8..16], &expected_key, 1e-5);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_staged_inputs_apply_gemma_interleaved_rope_for_nonzero_positions() {
    let model_dir = write_gemma_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let input = sample_runner_input();
    let staged = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("gemma projection-backed staged inputs should resolve");
    let normalized_hidden =
        rms_normalize_reference(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1e-6);
    let expected_query = [
        interleaved_rotate_reference(&normalized_hidden[..4], 1.0),
        interleaved_rotate_reference(&normalized_hidden[4..8], 1.0),
    ]
    .concat();
    let doubled_hidden = scale_reference(&normalized_hidden, 2.0);
    let expected_key = [
        interleaved_rotate_reference(&doubled_hidden[..4], 1.0),
        interleaved_rotate_reference(&doubled_hidden[4..8], 1.0),
    ]
    .concat();

    assert_eq!(workload.scheduled_positions, vec![0, 1, 2, 3]);
    assert_f32_slice_close(&staged.query[8..16], &expected_query, 1e-5);
    assert_f32_slice_close(&staged.key[8..16], &expected_key, 1e-5);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_staged_inputs_apply_manifest_rope_theta() {
    let model_dir = write_projection_custom_rope_native_model_fixture(100);
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let input = sample_runner_input();
    let staged = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("custom rope staged inputs should resolve");
    let normalized_hidden =
        rms_normalize_reference(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1e-6);
    let expected_query = [
        neox_rotate_reference_with_base(&normalized_hidden[..4], 1.0, 100.0),
        neox_rotate_reference_with_base(&normalized_hidden[4..8], 1.0, 100.0),
    ]
    .concat();
    let doubled_hidden = scale_reference(&normalized_hidden, 2.0);
    let expected_key = [
        neox_rotate_reference_with_base(&doubled_hidden[..4], 1.0, 100.0),
        neox_rotate_reference_with_base(&doubled_hidden[4..8], 1.0, 100.0),
    ]
    .concat();

    assert_f32_slice_close(&staged.query[8..16], &expected_query, 1e-5);
    assert_f32_slice_close(&staged.key[8..16], &expected_key, 1e-5);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_gemma_staged_inputs_apply_manifest_rope_theta() {
    let model_dir = write_gemma_projection_custom_rope_native_model_fixture(100);
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let input = sample_runner_input();
    let staged = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("custom gemma rope staged inputs should resolve");
    let normalized_hidden =
        rms_normalize_reference(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1e-6);
    let expected_query = [
        interleaved_rotate_reference_with_base(&normalized_hidden[..4], 1.0, 100.0),
        interleaved_rotate_reference_with_base(&normalized_hidden[4..8], 1.0, 100.0),
    ]
    .concat();
    let doubled_hidden = scale_reference(&normalized_hidden, 2.0);
    let expected_key = [
        interleaved_rotate_reference_with_base(&doubled_hidden[..4], 1.0, 100.0),
        interleaved_rotate_reference_with_base(&doubled_hidden[4..8], 1.0, 100.0),
    ]
    .concat();

    assert_f32_slice_close(&staged.query[8..16], &expected_query, 1e-5);
    assert_f32_slice_close(&staged.key[8..16], &expected_key, 1e-5);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn gemma_attention_output_uses_manifest_scale_and_softcap() {
    let default_model_dir = write_gemma_projection_native_model_fixture();
    let configured_model_dir = write_gemma_projection_attention_config_native_model_fixture(16, 1);

    let default_artifacts = NativeModelArtifacts::from_dir(&default_model_dir)
        .expect("default gemma artifacts should load");
    let configured_artifacts = NativeModelArtifacts::from_dir(&configured_model_dir)
        .expect("configured gemma artifacts should load");
    let default_bindings = MetalNativeModelBindings::from_artifacts(&default_artifacts)
        .expect("default bindings should load");
    let configured_bindings = MetalNativeModelBindings::from_artifacts(&configured_artifacts)
        .expect("configured bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let default_buffers =
        MetalNativeModelBufferBindings::from_model_bindings(&device, &default_bindings)
            .expect("default buffers should bind");
    let configured_buffers =
        MetalNativeModelBufferBindings::from_model_bindings(&device, &configured_bindings)
            .expect("configured buffers should bind");
    let input = sample_runner_input();
    let workload =
        MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
    let default_staged = model_conditioned_staged_inputs(
        &default_artifacts,
        &default_bindings,
        &default_buffers,
        &input,
        &workload,
        None,
        None,
        None,
    )
    .expect("default staged inputs should resolve");
    let configured_staged = model_conditioned_staged_inputs(
        &configured_artifacts,
        &configured_bindings,
        &configured_buffers,
        &input,
        &workload,
        None,
        None,
        None,
    )
    .expect("configured staged inputs should resolve");

    assert_f32_slice_close(&default_staged.query, &configured_staged.query, 1e-6);
    assert_f32_slice_close(&default_staged.key, &configured_staged.key, 1e-6);
    assert_f32_slice_close(&default_staged.value, &configured_staged.value, 1e-6);

    let (_, default_used_native) = attention_output_from_model_layer(
        &default_artifacts,
        &workload,
        &default_staged,
        None,
        None,
        true,
    )
    .expect("default attention output should resolve");
    let (configured_attention, configured_used_native) = attention_output_from_model_layer(
        &configured_artifacts,
        &workload,
        &configured_staged,
        None,
        None,
        true,
    )
    .expect("configured attention output should resolve");
    assert!(!default_used_native);
    assert!(!configured_used_native);

    let default_attention = attention_output_from_model_layer(
        &default_artifacts,
        &workload,
        &default_staged,
        None,
        None,
        true,
    )
    .expect("default attention output should resolve")
    .0;
    assert!(
        default_attention
            .iter()
            .zip(configured_attention.iter())
            .any(|(default, configured)| (default - configured).abs() > 1e-5),
        "configured Gemma attention should differ when manifest scale/softcap change"
    );

    let numeric_workload = workload.with_numeric_layout(configured_staged.layout);
    let attention_config = native_model_reference_attention_config(
        &configured_artifacts,
        configured_staged.layout.head_dim as usize,
    );
    let expected_head = manual_attention_head_output(
        &numeric_workload,
        &configured_staged,
        1,
        0,
        attention_config,
    );
    let head_size = numeric_workload.numeric_layout.head_size() as usize;
    let head_dim = numeric_workload.numeric_layout.head_dim as usize;
    let token_base = head_size;
    assert_f32_slice_close(
        &configured_attention[token_base..token_base + head_dim],
        &expected_head,
        1e-6,
    );

    let _ = fs::remove_dir_all(default_model_dir);
    let _ = fs::remove_dir_all(configured_model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_staged_inputs_use_final_layer_qkv_for_multilayer_fixture() {
    let model_dir = write_multilayer_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let input = sample_runner_input();
    let staged = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("projection-backed staged inputs should resolve");

    assert_eq!(
        staged.source,
        MetalStagedInputSource::ModelConditionedCpuPrefixAttention
    );
    assert_eq!(staged.prefix_attention_tally.native_dispatch_count(), 0);
    // 2-layer model: only layer 0 runs through advance_hidden_states_through_model_layer
    // (final_layer_index=1, loop is layers[..1]), producing 1 CPU reference dispatch.
    assert_eq!(
        staged.prefix_attention_tally.cpu_reference_dispatch_count(),
        1
    );
    assert!(staged.query.iter().all(|value| value.abs() <= 1e-6));
    assert!(staged.key.iter().all(|value| value.abs() <= 1e-6));
    assert!(staged.value.iter().all(|value| value.abs() <= 1e-6));

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_staged_inputs_match_explicit_prefix_cache_for_multilayer_fixture() {
    let model_dir = write_multilayer_direct_decode_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let workload =
        MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
    let prefix_layer_caches = vec![Mutex::new(MetalPersistentLayerKvCache::default())];

    let staged_without_cache = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("projection-backed staged inputs should resolve without explicit cache");
    let staged_with_cache = model_conditioned_staged_inputs(
        &artifacts,
        &bindings,
        &buffers,
        &input,
        &workload,
        Some(prefix_layer_caches.as_slice()),
        None,
        None,
    )
    .expect("projection-backed staged inputs should resolve with explicit cache");

    assert_eq!(staged_without_cache.source, staged_with_cache.source);
    assert_eq!(
        staged_without_cache.prefix_attention_tally,
        staged_with_cache.prefix_attention_tally
    );
    assert_f32_slice_close(&staged_without_cache.query, &staged_with_cache.query, 1e-6);
    assert_f32_slice_close(&staged_without_cache.key, &staged_with_cache.key, 1e-6);
    assert_f32_slice_close(&staged_without_cache.value, &staged_with_cache.value, 1e-6);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn batched_layer_qkv_staging_matches_per_item_staging_for_mixed_batch() {
    let model_dir = write_multilayer_direct_decode_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let workload =
        MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
    let token_embedding = buffers
        .binding_for(&bindings.token_embedding)
        .expect("token embedding binding should resolve");
    let (_, embedding_cols) = tensor_matrix_dimensions(&token_embedding.meta.spec)
        .expect("token embedding dimensions should resolve");
    let hidden_states = initial_model_hidden_states_cpu(
        token_embedding,
        &workload,
        (artifacts.manifest().hidden_size as usize).min(embedding_cols),
        native_model_embedding_scale(&artifacts),
    )
    .expect("initial hidden states should resolve");
    let layer = bindings
        .layers
        .first()
        .expect("multilayer fixture should expose at least one layer");
    let full_staged =
        stage_model_layer_qkv_inputs(&artifacts, layer, &buffers, &workload, &hidden_states, None)
            .expect("full staged inputs should resolve");
    let mut merged_item_tally = PrefixAttentionExecutionTally::default();
    let mut token_cursor = 0_usize;

    for item in &input.execution_batch.items {
        let token_end = token_cursor + item.scheduled_token_count as usize;
        let item_input =
            runner_input_for_execution_item(&input, item).expect("item input should resolve");
        let item_workload = MetalDispatchWorkload::from_runner_input(&item_input)
            .expect("item workload should resolve");
        let item_hidden_states = hidden_states
            .get(token_cursor..token_end)
            .expect("item hidden states should slice");
        let item_staged = stage_model_layer_qkv_inputs(
            &artifacts,
            layer,
            &buffers,
            &item_workload,
            item_hidden_states,
            None,
        )
        .expect("per-item staged inputs should resolve");
        let sliced = slice_staged_inputs_for_token_range(&full_staged, token_cursor..token_end)
            .expect("full staged inputs should slice");

        assert_eq!(sliced.layout, item_staged.layout);
        assert_f32_slice_close(&sliced.query, &item_staged.query, 1e-6);
        assert_f32_slice_close(&sliced.key, &item_staged.key, 1e-6);
        assert_f32_slice_close(&sliced.value, &item_staged.value, 1e-6);
        merged_item_tally = merged_item_tally.merge(item_staged.prefix_attention_tally);
        token_cursor = token_end;
    }

    assert_eq!(token_cursor, hidden_states.len());
    assert_eq!(
        full_staged
            .prefix_attention_tally
            .cpu_reference_dispatch_count(),
        merged_item_tally.cpu_reference_dispatch_count()
    );
    assert_eq!(
        full_staged
            .prefix_attention_tally
            .cpu_projection_row_count(),
        merged_item_tally.cpu_projection_row_count()
    );
    assert_eq!(
        full_staged
            .prefix_attention_tally
            .cpu_rms_norm_element_count(),
        merged_item_tally.cpu_rms_norm_element_count()
    );
    assert_eq!(
        full_staged
            .prefix_attention_tally
            .cpu_ffn_activation_element_count(),
        merged_item_tally.cpu_ffn_activation_element_count()
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn batched_layer_continuation_matches_per_item_continuation_for_mixed_batch() {
    let model_dir = write_multilayer_direct_decode_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let workload =
        MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
    let token_embedding = buffers
        .binding_for(&bindings.token_embedding)
        .expect("token embedding binding should resolve");
    let (_, embedding_cols) = tensor_matrix_dimensions(&token_embedding.meta.spec)
        .expect("token embedding dimensions should resolve");
    let hidden_states = initial_model_hidden_states_cpu(
        token_embedding,
        &workload,
        (artifacts.manifest().hidden_size as usize).min(embedding_cols),
        native_model_embedding_scale(&artifacts),
    )
    .expect("initial hidden states should resolve");
    let layer = bindings
        .layers
        .first()
        .expect("multilayer fixture should expose at least one layer");
    let attention_o = buffers
        .binding_for(
            layer
                .attention_o
                .as_ref()
                .expect("attention_o should exist"),
        )
        .expect("attention_o binding should resolve");
    let ffn_norm = buffers
        .binding_for(&layer.ffn_norm)
        .expect("ffn_norm binding should resolve");
    let ffn_down = buffers
        .binding_for(&layer.ffn_down)
        .expect("ffn_down binding should resolve");
    let staged =
        stage_model_layer_qkv_inputs(&artifacts, layer, &buffers, &workload, &hidden_states, None)
            .expect("full staged inputs should resolve");
    let head_size = staged.layout.head_size() as usize;
    let attention_output = (0..hidden_states.len().checked_mul(head_size).unwrap_or(0))
        .map(|index| index as f32 * 0.03125 + 0.5)
        .collect::<Vec<_>>();
    let (batched_hidden_states, batched_tally) = project_hidden_states_from_layer_attention_output(
        &artifacts,
        layer,
        &buffers,
        attention_o,
        ffn_norm,
        ffn_down,
        &hidden_states,
        &attention_output,
        head_size,
        None,
    )
    .expect("batched continuation should resolve");
    let mut token_cursor = 0_usize;
    let mut per_item_hidden_states = Vec::new();
    let mut per_item_tally = PrefixAttentionExecutionTally::default();

    for item in &input.execution_batch.items {
        let token_end = token_cursor + item.scheduled_token_count as usize;
        let item_hidden_states = hidden_states
            .get(token_cursor..token_end)
            .expect("item hidden states should slice");
        let attention_base = token_cursor.checked_mul(head_size).expect("attention base");
        let attention_end = token_end.checked_mul(head_size).expect("attention end");
        let item_attention_output = attention_output
            .get(attention_base..attention_end)
            .expect("item attention output should slice");
        let (item_next_hidden_states, item_tally) =
            project_hidden_states_from_layer_attention_output(
                &artifacts,
                layer,
                &buffers,
                attention_o,
                ffn_norm,
                ffn_down,
                item_hidden_states,
                item_attention_output,
                head_size,
                None,
            )
            .expect("per-item continuation should resolve");
        per_item_hidden_states.extend(item_next_hidden_states);
        per_item_tally = per_item_tally.merge(item_tally);
        token_cursor = token_end;
    }

    assert_eq!(token_cursor, hidden_states.len());
    assert_eq!(batched_hidden_states.len(), per_item_hidden_states.len());
    for (batched_hidden, per_item_hidden) in batched_hidden_states
        .iter()
        .zip(per_item_hidden_states.iter())
    {
        assert_f32_slice_close(batched_hidden, per_item_hidden, 1e-6);
    }
    assert_eq!(
        batched_tally.cpu_projection_row_count(),
        per_item_tally.cpu_projection_row_count()
    );
    assert_eq!(
        batched_tally.cpu_rms_norm_element_count(),
        per_item_tally.cpu_rms_norm_element_count()
    );
    assert_eq!(
        batched_tally.cpu_ffn_activation_element_count(),
        per_item_tally.cpu_ffn_activation_element_count()
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn grouped_prefix_attention_matches_per_item_outputs_for_mixed_batch() {
    let model_dir = write_multilayer_direct_decode_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let workload =
        MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
    let token_embedding = buffers
        .binding_for(&bindings.token_embedding)
        .expect("token embedding binding should resolve");
    let (_, embedding_cols) = tensor_matrix_dimensions(&token_embedding.meta.spec)
        .expect("token embedding dimensions should resolve");
    let hidden_states = initial_model_hidden_states_cpu(
        token_embedding,
        &workload,
        (artifacts.manifest().hidden_size as usize).min(embedding_cols),
        native_model_embedding_scale(&artifacts),
    )
    .expect("initial hidden states should resolve");
    let layer = bindings
        .layers
        .first()
        .expect("multilayer fixture should expose at least one layer");
    let staged =
        stage_model_layer_qkv_inputs(&artifacts, layer, &buffers, &workload, &hidden_states, None)
            .expect("full staged inputs should resolve");
    let item_token_ranges =
        execution_item_token_ranges(&input).expect("item token ranges should resolve");
    let (grouped_attention_output, grouped_tally) =
        collect_prefix_attention_outputs_with_item_fallback(
            &artifacts,
            &input,
            &staged,
            &item_token_ranges,
            0..input.execution_batch.items.len(),
            None,
            None,
        )
        .expect("grouped prefix attention should resolve");
    let mut token_cursor = 0_usize;
    let mut per_item_attention_output = Vec::new();
    let mut per_item_tally = PrefixAttentionExecutionTally::default();

    for item in &input.execution_batch.items {
        let token_end = token_cursor + item.scheduled_token_count as usize;
        let item_input =
            runner_input_for_execution_item(&input, item).expect("item input should resolve");
        let item_workload = MetalDispatchWorkload::from_runner_input(&item_input)
            .expect("item workload should resolve");
        let item_staged = slice_staged_inputs_for_token_range(&staged, token_cursor..token_end)
            .expect("item staged inputs should slice");
        let (item_attention_output, used_native_dispatch) = attention_output_from_model_layer(
            &artifacts,
            &item_workload,
            &item_staged,
            None,
            None,
            true,
        )
        .expect("per-item attention output should resolve");
        assert!(!used_native_dispatch);
        per_item_attention_output.extend(item_attention_output);
        per_item_tally = per_item_tally.record(used_native_dispatch);
        token_cursor = token_end;
    }

    assert_eq!(token_cursor, hidden_states.len());
    assert_f32_slice_close(&grouped_attention_output, &per_item_attention_output, 1e-6);
    assert_eq!(grouped_tally.native_dispatch_count(), 0);
    assert_eq!(grouped_tally.cpu_reference_dispatch_count(), 1);
    assert_eq!(
        per_item_tally.cpu_reference_dispatch_count(),
        input.execution_batch.items.len() as u32
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_bound_direct_decode_reuses_cached_multilayer_hidden_states() {
    let model_dir = write_multilayer_direct_decode_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let workload =
        MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
    let staged = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("multilayer staged inputs should resolve");
    let hidden_state_cache = staged
        .final_layer_hidden_state_cache
        .as_ref()
        .expect("model-conditioned staged inputs should retain hidden states");
    let mut attention_output_bits =
        vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];
    let decode_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
    for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .iter()
        .enumerate()
    {
        attention_output_bits[decode_base + index] = value.to_bits();
    }

    let replayed = derive_model_bound_direct_decode_result(
        &input,
        &attention_output_bits,
        Some(&artifacts),
        Some(&bindings),
        Some(&buffers),
        None,
        None,
        None,
    );
    let reused = derive_model_bound_direct_decode_result(
        &input,
        &attention_output_bits,
        Some(&artifacts),
        Some(&bindings),
        Some(&buffers),
        Some(hidden_state_cache),
        None,
        None,
    );

    assert_eq!(reused.tokens, replayed.tokens);
    assert_eq!(reused.logits_outputs, replayed.logits_outputs);
    assert_eq!(
        reused.model_bound_ffn_decode,
        replayed.model_bound_ffn_decode
    );
    assert_eq!(
        reused.native_logits_projection_decode,
        replayed.native_logits_projection_decode
    );
    assert_eq!(reused.execution_tally, replayed.execution_tally);
    assert_eq!(reused.native_dense_tally, replayed.native_dense_tally);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_bound_direct_decode_batches_multiple_decode_items_without_changing_outputs() {
    let model_dir = write_direct_decode_native_model_fixture(false);
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_decode_only_runner_input();
    let workload =
        MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
    let (hidden_states, final_layer_index, _) = model_hidden_states_before_final_layer(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("hidden states before final layer should resolve");
    let final_layer = bindings
        .layers
        .get(final_layer_index)
        .expect("final layer should resolve");
    let token_width = PHASE1_NUMERIC_HEAD_SIZE as usize;
    let mut attention_output_bits =
        vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * token_width];
    let token_patterns = [
        [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [8.0_f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ];
    for (token_index, pattern) in token_patterns.iter().enumerate() {
        let token_base = token_index * token_width;
        for (lane, value) in pattern.iter().enumerate() {
            attention_output_bits[token_base + lane] = value.to_bits();
        }
    }

    let batched = derive_model_bound_direct_decode_result(
        &input,
        &attention_output_bits,
        Some(&artifacts),
        Some(&bindings),
        Some(&buffers),
        None,
        None,
        None,
    );

    let mut expected_tokens = Vec::new();
    let mut expected_logits_outputs = Vec::new();
    let mut expected_model_bound_ffn_decode = false;
    let mut expected_native_logits_projection_decode = false;
    let mut expected_execution_tally = PrefixAttentionExecutionTally::default();
    let mut expected_native_dense_tally = DirectDecodeNativeDenseTally::default();
    let mut attention_index = 0_usize;
    for item in &input.execution_batch.items {
        if item.mode == ExecutionMode::Decode {
            let token_base = attention_index * token_width;
            let token_end = token_base + token_width;
            let hidden_index = attention_index + item.scheduled_token_count as usize - 1;
            let (
                logits,
                token_id,
                used_model_bound_ffn,
                vocab_rows_scanned,
                used_native_logits_projection,
                decode_native_dense_tally,
            ) = decode_logits_from_model_attention_output_with_metadata(
                &artifacts,
                &bindings,
                &buffers,
                final_layer,
                hidden_states
                    .get(hidden_index)
                    .expect("decode hidden state should resolve"),
                &attention_output_bits[token_base..token_end],
                None,
            )
            .expect("per-item decode logits should resolve");
            expected_tokens.push((item.request_id, token_id));
            expected_logits_outputs.push(RequestLogitsOutput {
                request_id: item.request_id,
                logits,
            });
            expected_model_bound_ffn_decode |= used_model_bound_ffn;
            expected_native_logits_projection_decode |= used_native_logits_projection;
            expected_execution_tally = expected_execution_tally
                .record_layer_continuation_tokens(1)
                .record_logits_projection(1, vocab_rows_scanned);
            expected_native_dense_tally =
                expected_native_dense_tally.merge(decode_native_dense_tally);
        }
        attention_index += item.scheduled_token_count as usize;
    }
    if expected_native_logits_projection_decode {
        expected_native_dense_tally =
            expected_native_dense_tally.record_batched_logits_group(expected_tokens.len());
    } else if expected_tokens.len() > 1 {
        expected_native_dense_tally =
            expected_native_dense_tally.record_batched_group_fallback(expected_tokens.len());
    }

    assert_eq!(batched.tokens, expected_tokens);
    assert_eq!(batched.logits_outputs, expected_logits_outputs);
    assert_eq!(
        batched.model_bound_ffn_decode,
        expected_model_bound_ffn_decode
    );
    assert_eq!(
        batched.native_logits_projection_decode,
        expected_native_logits_projection_decode
    );
    assert_eq!(batched.execution_tally, expected_execution_tally);
    assert_eq!(batched.native_dense_tally, expected_native_dense_tally);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn grouped_direct_decode_results_fall_back_to_singletons_when_group_processing_fails() {
    let dims = ModelBoundDecodeDims {
        input_width: 8,
        hidden_dim: 8,
        intermediate_dim: 16,
        vocab_rows: 32,
    };
    let groups = vec![vec![
        PreparedDirectDecodeItem {
            request_id: RequestId(9),
            dims,
            hidden: vec![0.0; dims.hidden_dim],
            attention_input: vec![0.0; dims.input_width],
        },
        PreparedDirectDecodeItem {
            request_id: RequestId(11),
            dims,
            hidden: vec![0.0; dims.hidden_dim],
            attention_input: vec![0.0; dims.input_width],
        },
    ]];
    let mut attempted_group_sizes = Vec::new();

    let collected =
        collect_model_bound_direct_decode_group_results_with_item_fallback(groups, |group| {
            attempted_group_sizes.push(group.len());
            if group.len() > 1 {
                return None;
            }
            let item = group.first()?;
            Some(ModelBoundDirectDecodeResult {
                tokens: vec![(
                    item.request_id,
                    u32::try_from(item.request_id.0).unwrap_or(u32::MAX),
                )],
                logits_outputs: vec![RequestLogitsOutput {
                    request_id: item.request_id,
                    logits: vec![item.request_id.0 as f32],
                }],
                model_bound_ffn_decode: true,
                native_logits_projection_decode: true,
                execution_tally: PrefixAttentionExecutionTally::default()
                    .record_layer_continuation_tokens(1),
                native_dense_tally: DirectDecodeNativeDenseTally::default()
                    .record_batched_logits_group(1),
            })
        })
        .expect("singleton fallback should recover grouped processing");

    assert_eq!(attempted_group_sizes, vec![2, 1, 1]);
    assert_eq!(collected.len(), 2);
    assert_eq!(collected[0].tokens, vec![(RequestId(9), 9)]);
    assert_eq!(collected[1].tokens, vec![(RequestId(11), 11)]);
    let merged_tally = collected
        .iter()
        .fold(DirectDecodeNativeDenseTally::default(), |tally, result| {
            tally.merge(result.native_dense_tally)
        });
    assert_eq!(merged_tally.batched_group_fallback_count, 1);
    assert_eq!(merged_tally.batched_group_fallback_token_count, 2);
    assert!(collected
        .iter()
        .all(|result| result.native_logits_projection_decode));
}

#[cfg(target_os = "macos")]
#[test]
fn grouped_direct_decode_results_recursively_preserve_batched_subgroups() {
    let dims = ModelBoundDecodeDims {
        input_width: 8,
        hidden_dim: 8,
        intermediate_dim: 16,
        vocab_rows: 32,
    };
    let groups = vec![vec![
        PreparedDirectDecodeItem {
            request_id: RequestId(3),
            dims,
            hidden: vec![0.0; dims.hidden_dim],
            attention_input: vec![0.0; dims.input_width],
        },
        PreparedDirectDecodeItem {
            request_id: RequestId(5),
            dims,
            hidden: vec![0.0; dims.hidden_dim],
            attention_input: vec![0.0; dims.input_width],
        },
        PreparedDirectDecodeItem {
            request_id: RequestId(7),
            dims,
            hidden: vec![0.0; dims.hidden_dim],
            attention_input: vec![0.0; dims.input_width],
        },
        PreparedDirectDecodeItem {
            request_id: RequestId(9),
            dims,
            hidden: vec![0.0; dims.hidden_dim],
            attention_input: vec![0.0; dims.input_width],
        },
    ]];
    let mut attempted_group_sizes = Vec::new();

    let collected =
        collect_model_bound_direct_decode_group_results_with_item_fallback(groups, |group| {
            attempted_group_sizes.push(group.len());
            if group.len() > 2 {
                return None;
            }
            let tokens = group
                .iter()
                .map(|item| {
                    (
                        item.request_id,
                        u32::try_from(item.request_id.0).unwrap_or(u32::MAX),
                    )
                })
                .collect::<Vec<_>>();
            let logits_outputs = group
                .iter()
                .map(|item| RequestLogitsOutput {
                    request_id: item.request_id,
                    logits: vec![item.request_id.0 as f32],
                })
                .collect::<Vec<_>>();
            Some(ModelBoundDirectDecodeResult {
                tokens,
                logits_outputs,
                model_bound_ffn_decode: true,
                native_logits_projection_decode: true,
                execution_tally: PrefixAttentionExecutionTally::default()
                    .record_layer_continuation_tokens(
                        u32::try_from(group.len()).unwrap_or(u32::MAX),
                    ),
                native_dense_tally: DirectDecodeNativeDenseTally::default()
                    .record_batched_logits_group(group.len()),
            })
        })
        .expect("recursive subgroup fallback should preserve grouped execution");

    assert_eq!(attempted_group_sizes, vec![4, 2, 2]);
    assert_eq!(collected.len(), 2);
    assert_eq!(
        collected[0].tokens,
        vec![(RequestId(3), 3), (RequestId(5), 5)]
    );
    assert_eq!(
        collected[1].tokens,
        vec![(RequestId(7), 7), (RequestId(9), 9)]
    );
    let merged_tally = collected
        .iter()
        .fold(DirectDecodeNativeDenseTally::default(), |tally, result| {
            tally.merge(result.native_dense_tally)
        });
    assert_eq!(merged_tally.native_batched_logits_group_count, 2);
    assert_eq!(merged_tally.native_batched_logits_token_count, 4);
    assert_eq!(merged_tally.batched_group_fallback_count, 1);
    assert_eq!(merged_tally.batched_group_fallback_token_count, 4);
}

#[cfg(target_os = "macos")]
#[test]
fn model_conditioned_staged_inputs_use_full_hidden_width_beyond_32d_cap() {
    let model_dir = write_wide_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_prefill_only_runner_input();
    let workload =
        MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");

    let staged = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("wide projection staged inputs should resolve");

    assert_eq!(
        staged.source,
        MetalStagedInputSource::ModelConditionedMiniProjection
    );
    let mut wide_hidden = vec![0.0_f32; 40];
    wide_hidden[32..40].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let expected_hidden = rms_normalize_reference(&wide_hidden, 1e-6);
    assert_f32_slice_close(&staged.query[..8], &expected_hidden[32..40], 1e-6);
    assert_f32_slice_close(
        &staged.key[..8],
        &scale_reference(&expected_hidden[32..40], 2.0),
        1e-6,
    );
    assert_f32_slice_close(
        &staged.value[..8],
        &scale_reference(&expected_hidden[32..40], 3.0),
        1e-6,
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn resolve_runtime_staged_inputs_fails_closed_when_model_projection_is_invalid() {
    let model_dir = write_projection_native_model_fixture();
    let manifest_path = model_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("manifest should parse");
    manifest.attention_head_dim = 3;
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
    )
    .expect("manifest should write");

    let error = NativeModelArtifacts::from_dir(&model_dir)
        .expect_err("invalid projection manifest should fail closed at load time");
    let crate::model::NativeModelError::InvalidManifest { message } = error else {
        panic!("expected invalid manifest");
    };
    assert!(message.contains("attention_q rows 8 must be divisible by head_dim 3"));

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn numeric_trace_validation_accepts_model_conditioned_reference_path() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");
    let input = sample_runner_input();
    let staged = model_conditioned_staged_inputs(
        &artifacts, &bindings, &buffers, &input, &workload, None, None, None,
    )
    .expect("projection-backed staged inputs should resolve");
    let simulated = reference_numeric_path_with_inputs(&workload, &staged);
    let trace = simulated_numeric_trace(&simulated);

    let summary = validate_numeric_trace_against_inputs(&workload, &staged, &trace)
        .expect("model-conditioned reference trace should validate");

    assert_eq!(
        summary.expected_key_cache_checksum,
        trace.key_cache_checksum
    );
    assert_eq!(
        summary.expected_attention_output_checksum,
        trace.attention_output_checksum
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_bound_decode_tokens_use_lm_head_projection_for_decode_items() {
    let model_dir = write_direct_decode_native_model_fixture(false);
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let mut attention_output_bits =
        vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];
    let decode_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
    for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .iter()
        .enumerate()
    {
        attention_output_bits[decode_base + index] = value.to_bits();
    }

    let direct_decode = derive_model_bound_direct_decode_result(
        &input,
        &attention_output_bits,
        Some(&artifacts),
        Some(&bindings),
        Some(&buffers),
        None,
        None,
        None,
    );

    assert_eq!(direct_decode.tokens, vec![(RequestId(9), 4)]);
    assert_eq!(direct_decode.logits_outputs.len(), 1);
    assert_eq!(direct_decode.logits_outputs[0].request_id, RequestId(9));
    assert_eq!(
        direct_decode.logits_outputs[0]
            .logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32),
        Some(4)
    );
    assert!(!direct_decode.model_bound_ffn_decode);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_bound_decode_tokens_apply_gemma_weight_offset_norms() {
    let model_dir = write_gemma_direct_decode_native_model_fixture(false);
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let mut attention_output_bits =
        vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];
    let decode_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
    for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .iter()
        .enumerate()
    {
        attention_output_bits[decode_base + index] = value.to_bits();
    }

    let direct_decode_tokens = derive_model_bound_decode_tokens(
        &input,
        &attention_output_bits,
        Some(&artifacts),
        Some(&bindings),
        Some(&buffers),
        None,
        None,
        None,
    );

    assert_eq!(direct_decode_tokens, vec![(RequestId(9), 4)]);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_bound_decode_tokens_fall_back_to_tied_embeddings_when_lm_head_is_absent() {
    let model_dir = write_direct_decode_native_model_fixture(true);
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let mut attention_output_bits =
        vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];
    let decode_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
    for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .iter()
        .enumerate()
    {
        attention_output_bits[decode_base + index] = value.to_bits();
    }

    let direct_decode_tokens = derive_model_bound_decode_tokens(
        &input,
        &attention_output_bits,
        Some(&artifacts),
        Some(&bindings),
        Some(&buffers),
        None,
        None,
        None,
    );

    assert_eq!(direct_decode_tokens, vec![(RequestId(9), 4)]);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_bound_decode_tokens_support_multilayer_cpu_prefix_forward() {
    let model_dir = write_multilayer_direct_decode_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let mut attention_output_bits =
        vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];
    let decode_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
    for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .iter()
        .enumerate()
    {
        attention_output_bits[decode_base + index] = value.to_bits();
    }

    let direct_decode_tokens = derive_model_bound_decode_tokens(
        &input,
        &attention_output_bits,
        Some(&artifacts),
        Some(&bindings),
        Some(&buffers),
        None,
        None,
        None,
    );

    assert_eq!(direct_decode_tokens.len(), 1);
    assert_eq!(direct_decode_tokens[0].0, RequestId(9));
    assert!(direct_decode_tokens[0].1 > 0);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_bound_direct_decode_matches_explicit_prefix_cache_for_multilayer_fixture() {
    let model_dir = write_multilayer_direct_decode_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let prefix_layer_caches = vec![Mutex::new(MetalPersistentLayerKvCache::default())];
    let mut attention_output_bits =
        vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];
    let decode_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
    for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .iter()
        .enumerate()
    {
        attention_output_bits[decode_base + index] = value.to_bits();
    }

    let direct_decode_without_cache = derive_model_bound_direct_decode_result(
        &input,
        &attention_output_bits,
        Some(&artifacts),
        Some(&bindings),
        Some(&buffers),
        None,
        None,
        None,
    );
    let direct_decode_with_cache = derive_model_bound_direct_decode_result(
        &input,
        &attention_output_bits,
        Some(&artifacts),
        Some(&bindings),
        Some(&buffers),
        None,
        Some(prefix_layer_caches.as_slice()),
        None,
    );

    assert_eq!(
        direct_decode_without_cache.tokens,
        direct_decode_with_cache.tokens
    );
    assert_eq!(
        direct_decode_without_cache.model_bound_ffn_decode,
        direct_decode_with_cache.model_bound_ffn_decode
    );
    assert_eq!(
        direct_decode_without_cache.logits_outputs,
        direct_decode_with_cache.logits_outputs
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_bound_decode_tokens_apply_split_ffn_continuation_before_projection() {
    let model_dir = write_ffn_decode_native_model_fixture(DirectDecodeFixtureGateUpLayout::Split);
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let attention_output_bits =
        vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];

    let direct_decode = derive_model_bound_direct_decode_result(
        &input,
        &attention_output_bits,
        Some(&artifacts),
        Some(&bindings),
        Some(&buffers),
        None,
        None,
        None,
    );

    assert_eq!(direct_decode.tokens, vec![(RequestId(9), 4)]);
    assert_eq!(direct_decode.logits_outputs.len(), 1);
    assert!(direct_decode.model_bound_ffn_decode);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn model_bound_decode_tokens_apply_packed_ffn_continuation_before_projection() {
    let model_dir = write_ffn_decode_native_model_fixture(DirectDecodeFixtureGateUpLayout::Packed);
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let attention_output_bits =
        vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];

    let direct_decode = derive_model_bound_direct_decode_result(
        &input,
        &attention_output_bits,
        Some(&artifacts),
        Some(&bindings),
        Some(&buffers),
        None,
        None,
        None,
    );

    assert_eq!(direct_decode.tokens, vec![(RequestId(9), 4)]);
    assert_eq!(direct_decode.logits_outputs.len(), 1);
    assert!(direct_decode.model_bound_ffn_decode);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn batched_ffn_gate_up_paths_match_rowwise_reference_for_split_and_packed_layouts() {
    for layout in [
        DirectDecodeFixtureGateUpLayout::Split,
        DirectDecodeFixtureGateUpLayout::Packed,
    ] {
        let model_dir = write_ffn_decode_native_model_fixture(layout);
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let layer = bindings
            .layers
            .first()
            .expect("fixture should contain one layer");
        let input_rows = vec![
            vec![1.0, 0.5, -0.5, 0.25, 0.0, 0.75, -1.0, 0.5],
            vec![0.2, -0.1, 0.4, -0.3, 0.6, -0.5, 0.8, -0.7],
        ];

        let mut expected_gate_rows = Vec::new();
        let mut expected_up_rows = Vec::new();
        let mut expected_projection_tally = DirectDecodeNativeDenseTally::default();
        let mut expected_activation_tally = DirectDecodeNativeDenseTally::default();
        for input in &input_rows {
            let (mut gate, up, projection_tally) =
                project_ffn_gate_up_with_coverage(&layer.ffn_gate_up, &buffers, 8, input, None)
                    .expect("rowwise gate/up projection should succeed");
            expected_projection_tally = expected_projection_tally.merge(projection_tally);
            let used_native_activation =
                apply_model_gate_up_product_with_path(&artifacts, &mut gate, &up, None)
                    .expect("rowwise gate/up product should succeed");
            expected_activation_tally = expected_activation_tally
                .record_ffn_activation_elements(gate.len(), used_native_activation);
            expected_gate_rows.push(gate);
            expected_up_rows.push(up);
        }

        let (mut actual_gate_rows, actual_up_rows, actual_projection_tally) =
            project_batched_ffn_gate_up_with_tally(
                &layer.ffn_gate_up,
                &buffers,
                8,
                &input_rows,
                8,
                None,
            )
            .expect("batched gate/up projection should succeed");
        let actual_activation_tally = apply_batched_model_gate_up_product_in_place_with_tally(
            &artifacts,
            &mut actual_gate_rows,
            &actual_up_rows,
            8,
            None,
        )
        .expect("batched gate/up product should succeed");

        assert_eq!(actual_projection_tally, expected_projection_tally);
        assert_eq!(actual_activation_tally, expected_activation_tally);
        for (actual, expected) in actual_up_rows.iter().zip(expected_up_rows.iter()) {
            assert_f32_slice_close(actual, expected, 1e-6);
        }
        for (actual, expected) in actual_gate_rows.iter().zip(expected_gate_rows.iter()) {
            assert_f32_slice_close(actual, expected, 1e-6);
        }

        let _ = fs::remove_dir_all(model_dir);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn model_bound_decode_tokens_use_hidden_dimensions_beyond_32d_cap() {
    let model_dir = write_wide_direct_decode_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("native model buffers should bind");
    let input = sample_runner_input();
    let attention_output_bits =
        vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];

    let direct_decode_tokens = derive_model_bound_decode_tokens(
        &input,
        &attention_output_bits,
        Some(&artifacts),
        Some(&bindings),
        Some(&buffers),
        None,
        None,
        None,
    );

    assert_eq!(direct_decode_tokens, vec![(RequestId(9), 2)]);

    let _ = fs::remove_dir_all(model_dir);
}

#[test]
fn annotate_staged_input_source_marks_model_conditioned_inputs() {
    let mut route_metadata = RouteMetadata::empty();

    annotate_staged_input_source(
        &mut route_metadata,
        MetalStagedInputSource::ModelConditionedMiniProjection,
    );

    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_model_conditioned_inputs".to_string(), 1,)));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_token_only_inputs".to_string(), 0,)));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_layers_native_attention".to_string(),
        0,
    )));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_prefix_layers_cpu_reference".to_string(), 0,)));
    assert!(!route_metadata
        .crossover_decisions
        .iter()
        .any(|(key, _)| key == "metal_dispatch_real_model_tensor_inputs"));
}

#[test]
fn annotate_staged_input_source_marks_cpu_prefix_attention() {
    let mut route_metadata = RouteMetadata::empty();

    annotate_staged_input_source(
        &mut route_metadata,
        MetalStagedInputSource::ModelConditionedCpuPrefixAttention,
    );

    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_layers_native_attention".to_string(),
        0,
    )));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_prefix_layers_cpu_reference".to_string(), 1,)));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_model_conditioned_inputs".to_string(), 1,)));
}

#[test]
fn annotate_staged_input_source_marks_native_prefix_attention() {
    let mut route_metadata = RouteMetadata::empty();

    annotate_staged_input_source(
        &mut route_metadata,
        MetalStagedInputSource::ModelConditionedNativePrefixAttention,
    );

    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_layers_native_attention".to_string(),
        1,
    )));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_prefix_layers_cpu_reference".to_string(), 0,)));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_model_conditioned_inputs".to_string(), 1,)));
}

#[test]
fn annotate_staged_input_source_marks_mixed_prefix_attention() {
    let mut route_metadata = RouteMetadata::empty();

    annotate_staged_input_source(
        &mut route_metadata,
        MetalStagedInputSource::ModelConditionedMixedPrefixAttention,
    );

    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_layers_native_attention".to_string(),
        1,
    )));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_prefix_layers_cpu_reference".to_string(), 1,)));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_model_conditioned_inputs".to_string(), 1,)));
}

#[test]
fn complete_model_forward_support_requires_model_conditioned_source() {
    let single_layer_model = NativeModelArtifactsSummary {
        model_family: "qwen3_dense".to_string(),
        tensor_format: crate::model::NativeTensorFormat::Safetensors,
        layer_count: 1,
        tensor_count: 9,
        tie_word_embeddings: false,
    };
    let multilayer_model = NativeModelArtifactsSummary {
        model_family: "qwen3_dense".to_string(),
        tensor_format: crate::model::NativeTensorFormat::Safetensors,
        layer_count: 2,
        tensor_count: 18,
        tie_word_embeddings: false,
    };

    assert!(!complete_model_forward_support_for_source(
        Some(&single_layer_model),
        None
    ));
    assert!(!complete_model_forward_support_for_source(
        Some(&single_layer_model),
        Some(MetalStagedInputSource::SyntheticTokenIds)
    ));
    assert!(complete_model_forward_support_for_source(
        Some(&single_layer_model),
        Some(MetalStagedInputSource::ModelConditionedMiniProjection)
    ));
    assert!(!complete_model_forward_support_for_source(
        Some(&multilayer_model),
        Some(MetalStagedInputSource::ModelConditionedMiniProjection)
    ));
    assert!(complete_model_forward_support_for_source(
        Some(&multilayer_model),
        Some(MetalStagedInputSource::ModelConditionedNativePrefixAttention)
    ));
    assert!(complete_model_forward_support_for_source(
        Some(&multilayer_model),
        Some(MetalStagedInputSource::ModelConditionedMixedPrefixAttention)
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn native_prefix_attention_support_requires_prefill_only_self_contained_context() {
    let prefill_workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill-only workload should resolve");
    let mixed_workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("mixed workload should resolve");

    assert!(workload_supports_native_prefix_attention(
        &prefill_workload,
        None
    ));
    assert!(!workload_supports_native_prefix_attention(
        &mixed_workload,
        None
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn mixed_batch_prefill_segment_retains_native_prefix_attention_eligibility() {
    let input = sample_runner_input();
    let prefill_item = input
        .execution_batch
        .items
        .iter()
        .find(|item| item.mode == ExecutionMode::Prefill)
        .expect("mixed batch should contain a prefill item");
    let decode_item = input
        .execution_batch
        .items
        .iter()
        .find(|item| item.mode == ExecutionMode::Decode)
        .expect("mixed batch should contain a decode item");

    let prefill_input =
        runner_input_for_execution_item(&input, prefill_item).expect("prefill item should slice");
    let decode_input =
        runner_input_for_execution_item(&input, decode_item).expect("decode item should slice");
    let prefill_workload = MetalDispatchWorkload::from_runner_input(&prefill_input)
        .expect("prefill slice workload should resolve");
    let decode_workload = MetalDispatchWorkload::from_runner_input(&decode_input)
        .expect("decode slice workload should resolve");

    assert!(workload_supports_native_prefix_attention(
        &prefill_workload,
        None
    ));
    assert!(!workload_supports_native_prefix_attention(
        &decode_workload,
        None
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn native_prefix_attention_viability_matches_group_support_and_cache_state() {
    let mixed_workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("mixed workload should resolve");
    let prefill_workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill-only workload should resolve");
    let decode_workload =
        MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
            .expect("decode continuation workload should resolve");
    let layer_cache = Mutex::new(MetalPersistentLayerKvCache::default());

    assert!(!native_prefix_attention_is_viable(
        &mixed_workload,
        true,
        Some(&layer_cache)
    ));
    assert!(native_prefix_attention_is_viable(
        &prefill_workload,
        true,
        Some(&layer_cache)
    ));
    assert!(!native_prefix_attention_is_viable(
        &decode_workload,
        true,
        Some(&layer_cache)
    ));

    let prefill_staged = synthetic_staged_inputs(&prefill_workload);
    let prefill_reference = reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
    layer_cache
        .lock()
        .expect("metal prefix layer cache mutex should not be poisoned")
        .apply_snapshot(
            &prefill_workload,
            &MetalDispatchKvCacheSnapshot::from_reference_for_workload(
                &prefill_workload,
                &prefill_reference,
            ),
        );

    assert!(native_prefix_attention_is_viable(
        &decode_workload,
        true,
        Some(&layer_cache)
    ));
    assert!(!native_prefix_attention_is_viable(
        &prefill_workload,
        false,
        Some(&layer_cache)
    ));
}

#[test]
fn seeded_reference_numeric_path_reuses_prior_layer_cache_for_decode() {
    let prefill_workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill workload should resolve");
    let decode_workload =
        MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
            .expect("decode continuation workload should resolve");
    let prefill_staged = synthetic_staged_inputs(&prefill_workload);
    let prefill_reference = reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
    let mut layer_cache = MetalPersistentLayerKvCache::default();
    layer_cache.apply_snapshot(
        &prefill_workload,
        &MetalDispatchKvCacheSnapshot::from_reference_for_workload(
            &prefill_workload,
            &prefill_reference,
        ),
    );
    let decode_staged = synthetic_staged_inputs(&decode_workload);
    let seeded_reference = {
        let cache_seed = layer_cache.seed_for_workload(&decode_workload);
        reference_numeric_path_with_inputs_and_cache_seed(
            &decode_workload,
            &decode_staged,
            Some(cache_seed),
        )
    };
    let baseline_reference = reference_numeric_path_with_inputs(&decode_workload, &decode_staged);

    assert_ne!(
        checksum_f32_slice(&seeded_reference.gather_key),
        checksum_f32_slice(&baseline_reference.gather_key)
    );
    assert_ne!(
        checksum_f32_slice(&seeded_reference.attention_output),
        checksum_f32_slice(&baseline_reference.attention_output)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn initialized_layer_cache_allows_decode_native_prefix_attention() {
    let prefill_workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill workload should resolve");
    let decode_workload =
        MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
            .expect("decode continuation workload should resolve");
    let layer_cache = Mutex::new(MetalPersistentLayerKvCache::default());

    assert!(!workload_supports_native_prefix_attention(
        &decode_workload,
        Some(&layer_cache)
    ));

    let prefill_staged = synthetic_staged_inputs(&prefill_workload);
    let prefill_reference = reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
    layer_cache
        .lock()
        .expect("metal prefix layer cache mutex should not be poisoned")
        .apply_snapshot(
            &prefill_workload,
            &MetalDispatchKvCacheSnapshot::from_reference_for_workload(
                &prefill_workload,
                &prefill_reference,
            ),
        );

    assert!(workload_supports_native_prefix_attention(
        &decode_workload,
        Some(&layer_cache)
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn prefix_support_probe_preserves_initialized_slots_across_layout_mismatch() {
    let prefill_workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill workload should resolve")
            .with_numeric_layout(MetalDispatchNumericLayout::new(4, 8));
    let decode_workload =
        MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
            .expect("decode continuation workload should resolve");
    let decode_numeric_workload =
        decode_workload.with_numeric_layout(MetalDispatchNumericLayout::new(4, 8));
    let layer_cache = Mutex::new(MetalPersistentLayerKvCache::default());
    layer_cache
        .lock()
        .expect("metal prefix layer cache mutex should not be poisoned")
        .apply_snapshot(
            &prefill_workload,
            &MetalDispatchKvCacheSnapshot {
                key_cache: vec![1.0; prefill_workload.slot_numeric_capacity() as usize],
                value_cache: vec![2.0; prefill_workload.slot_numeric_capacity() as usize],
            },
        );

    assert!(workload_supports_native_prefix_attention(
        &decode_numeric_workload,
        Some(&layer_cache)
    ));
    assert!(workload_supports_native_prefix_attention(
        &decode_workload,
        Some(&layer_cache)
    ));
    assert!(workload_supports_native_prefix_attention(
        &decode_numeric_workload,
        Some(&layer_cache)
    ));

    let layer_cache = layer_cache
        .lock()
        .expect("metal prefix layer cache mutex should not be poisoned");
    for slot in 0..4 {
        assert!(layer_cache.slot_initialized(slot));
    }
}

#[cfg(target_os = "macos")]
#[test]
fn prefix_reuse_warmup_is_needed_until_all_prefix_layer_caches_are_initialized() {
    let prefill_workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill workload should resolve");
    let decode_workload =
        MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
            .expect("decode continuation workload should resolve");
    let prefix_layer_caches = vec![
        Mutex::new(MetalPersistentLayerKvCache::default()),
        Mutex::new(MetalPersistentLayerKvCache::default()),
    ];
    let prefill_staged = synthetic_staged_inputs(&prefill_workload);
    let prefill_reference = reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
    let prefill_snapshot = MetalDispatchKvCacheSnapshot::from_reference_for_workload(
        &prefill_workload,
        &prefill_reference,
    );

    assert!(prefix_reuse_warmup_needed(
        &decode_workload,
        prefix_layer_caches.as_slice(),
        2,
    ));

    prefix_layer_caches[0]
        .lock()
        .expect("metal prefix layer cache mutex should not be poisoned")
        .apply_snapshot(&prefill_workload, &prefill_snapshot);

    assert!(prefix_reuse_warmup_needed(
        &decode_workload,
        prefix_layer_caches.as_slice(),
        2,
    ));

    prefix_layer_caches[1]
        .lock()
        .expect("metal prefix layer cache mutex should not be poisoned")
        .apply_snapshot(&prefill_workload, &prefill_snapshot);

    assert!(!prefix_reuse_warmup_needed(
        &decode_workload,
        prefix_layer_caches.as_slice(),
        2,
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn copied_prefix_blocks_persist_into_layer_cache_for_future_native_decode() {
    let mut prefill_workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill workload should resolve");
    prefill_workload.kv_slot_capacity = 32;
    prefill_workload.kv_block_capacity = 32;
    prefill_workload.kv_metadata.copy_block_mapping = vec![[0, 16]];
    let prefill_staged = synthetic_staged_inputs(&prefill_workload);
    let prefill_reference = reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
    let prefill_snapshot = MetalDispatchKvCacheSnapshot::from_reference_for_workload(
        &prefill_workload,
        &prefill_reference,
    );
    let head_size = prefill_workload.numeric_layout.head_size() as usize;
    let copied_block_base = 16_usize * head_size;
    let copied_block_end = copied_block_base + 4 * head_size;

    assert_eq!(
        &prefill_snapshot.key_cache[copied_block_base..copied_block_end],
        &prefill_reference.key_cache[..4 * head_size]
    );
    assert_eq!(
        &prefill_snapshot.value_cache[copied_block_base..copied_block_end],
        &prefill_reference.value_cache[..4 * head_size]
    );

    let decode_input = RunnerInput {
        block_size_tokens: 16,
        execution_batch: ExecutionBatch {
            step_id: StepId(7),
            model_id: "qwen3_dense".into(),
            execution_plan_ref: Some("phase1.qwen3_dense.decode_copied_prefix".into()),
            items: vec![ExecutionItem {
                request_id: RequestId(29),
                mode: ExecutionMode::Decode,
                input_token_slice: vec![5],
                reused_prefix_token_slice: Vec::new(),
                position_range: PositionRange {
                    start: 4,
                    end_exclusive: 5,
                },
                scheduled_token_count: 1,
                block_table_ref: RequestId(29),
                prefix_tokens_reused: 0,
                prefix_blocks_reused: 0,
            }],
            total_scheduled_tokens: 1,
            route_metadata: RouteMetadata::empty(),
        },
        block_tables: vec![crate::runner::ResolvedBlockTable {
            request_id: RequestId(29),
            block_table: BlockTableView {
                cache_group_id: CacheGroupId(1),
                block_ids: vec![BlockId(1)],
            },
        }],
    };
    let decode_workload = MetalDispatchWorkload::from_runner_input(&decode_input)
        .expect("decode workload should resolve");
    let layer_cache = Mutex::new(MetalPersistentLayerKvCache::default());

    assert!(!workload_supports_native_prefix_attention(
        &decode_workload,
        Some(&layer_cache)
    ));

    layer_cache
        .lock()
        .expect("metal prefix layer cache mutex should not be poisoned")
        .apply_snapshot(&prefill_workload, &prefill_snapshot);

    assert!(workload_supports_native_prefix_attention(
        &decode_workload,
        Some(&layer_cache)
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn copied_prefix_snapshot_only_marks_offsets_backed_by_real_prefix_tokens() {
    let mut prefill_workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill workload should resolve");
    prefill_workload.kv_slot_capacity = 32;
    prefill_workload.kv_block_capacity = 32;
    prefill_workload.kv_metadata.copy_block_mapping = vec![[0, 16]];
    let prefill_staged = synthetic_staged_inputs(&prefill_workload);
    let prefill_reference = reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
    let prefill_snapshot = MetalDispatchKvCacheSnapshot::from_reference_for_workload(
        &prefill_workload,
        &prefill_reference,
    );
    let mut layer_cache = MetalPersistentLayerKvCache::default();

    layer_cache.apply_snapshot(&prefill_workload, &prefill_snapshot);

    for slot in 16..20 {
        assert!(layer_cache.slot_initialized(slot));
    }
    for slot in 20..32 {
        assert!(!layer_cache.slot_initialized(slot));
    }
}

#[cfg(target_os = "macos")]
#[test]
fn copied_prefix_sources_allow_native_prefix_attention_before_target_slots_are_marked() {
    let prefill_workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill workload should resolve");
    let prefill_staged = synthetic_staged_inputs(&prefill_workload);
    let prefill_reference = reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
    let layer_cache = Mutex::new(MetalPersistentLayerKvCache::default());
    layer_cache
        .lock()
        .expect("metal prefix layer cache mutex should not be poisoned")
        .apply_snapshot(
            &prefill_workload,
            &MetalDispatchKvCacheSnapshot::from_reference_for_workload(
                &prefill_workload,
                &prefill_reference,
            ),
        );

    let mut decode_workload =
        MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
            .expect("decode continuation workload should resolve");
    decode_workload.kv_slot_capacity = 32;
    decode_workload.kv_block_capacity = 32;
    decode_workload.kv_metadata.gather_block_table = vec![16];
    decode_workload.kv_metadata.gather_block_table_stride = 1;

    assert!(!workload_supports_native_prefix_attention(
        &decode_workload,
        Some(&layer_cache)
    ));

    decode_workload.kv_metadata.copy_block_mapping = vec![[0, 16]];

    assert!(workload_supports_native_prefix_attention(
        &decode_workload,
        Some(&layer_cache)
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn materialized_seed_cache_applies_copy_block_mapping_to_target_slots() {
    let mut workload =
        MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
            .expect("decode continuation workload should resolve");
    workload.kv_slot_capacity = 32;
    workload.kv_block_capacity = 32;
    workload.kv_metadata.copy_block_mapping = vec![[0, 16]];
    let slot_capacity = workload.slot_numeric_capacity() as usize;
    let head_size = workload.numeric_layout.head_size() as usize;
    let block_width = workload.block_numeric_elements() as usize;
    let mut key_cache = vec![0.0_f32; slot_capacity];
    let mut value_cache = vec![0.0_f32; slot_capacity];
    for index in 0..block_width {
        key_cache[index] = index as f32 + 1.0;
        value_cache[index] = index as f32 + 101.0;
    }

    let owned_seed = materialize_copy_targets_into_owned_cache_seed(
        &workload,
        MetalDispatchKvCacheSeed {
            key_cache: &key_cache,
            value_cache: &value_cache,
        },
    );
    let copied_block_base = 16 * head_size;
    let copied_block_end = copied_block_base + block_width;

    assert_eq!(
        &owned_seed.key_cache[copied_block_base..copied_block_end],
        &key_cache[..block_width]
    );
    assert_eq!(
        &owned_seed.value_cache[copied_block_base..copied_block_end],
        &value_cache[..block_width]
    );
}

#[cfg(target_os = "macos")]
#[test]
fn materialized_seed_cache_applies_transitive_copy_block_mapping_regardless_of_order() {
    let mut workload =
        MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
            .expect("decode continuation workload should resolve");
    workload.kv_slot_capacity = 48;
    workload.kv_block_capacity = 48;
    workload.kv_metadata.copy_block_mapping = vec![[16, 32], [0, 16]];
    let slot_capacity = workload.slot_numeric_capacity() as usize;
    let head_size = workload.numeric_layout.head_size() as usize;
    let block_width = workload.block_numeric_elements() as usize;
    let mut key_cache = vec![0.0_f32; slot_capacity];
    let mut value_cache = vec![0.0_f32; slot_capacity];
    for index in 0..block_width {
        key_cache[index] = index as f32 + 1.0;
        value_cache[index] = index as f32 + 101.0;
    }

    let owned_seed = materialize_copy_targets_into_owned_cache_seed(
        &workload,
        MetalDispatchKvCacheSeed {
            key_cache: &key_cache,
            value_cache: &value_cache,
        },
    );
    let copied_block_16_base = 16 * head_size;
    let copied_block_32_base = 32 * head_size;
    let copied_block_16_end = copied_block_16_base + block_width;
    let copied_block_32_end = copied_block_32_base + block_width;

    assert_eq!(
        &owned_seed.key_cache[copied_block_16_base..copied_block_16_end],
        &key_cache[..block_width]
    );
    assert_eq!(
        &owned_seed.value_cache[copied_block_16_base..copied_block_16_end],
        &value_cache[..block_width]
    );
    assert_eq!(
        &owned_seed.key_cache[copied_block_32_base..copied_block_32_end],
        &key_cache[..block_width]
    );
    assert_eq!(
        &owned_seed.value_cache[copied_block_32_base..copied_block_32_end],
        &value_cache[..block_width]
    );
}

#[cfg(target_os = "macos")]
#[test]
fn self_contained_seed_from_staged_inputs_populates_scheduled_and_copied_slots() {
    let mut workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill workload should resolve");
    workload.kv_slot_capacity = 32;
    workload.kv_block_capacity = 32;
    workload.kv_metadata.copy_block_mapping = vec![[0, 16]];
    let staged_inputs = synthetic_staged_inputs(&workload);
    let owned_seed = self_contained_owned_cache_seed_from_staged_inputs(&workload, &staged_inputs)
        .expect("self-contained seed should build");
    let head_size = workload.numeric_layout.head_size() as usize;
    let populated_width = workload.scheduled_numeric_elements() as usize;
    let block_width = workload.block_numeric_elements() as usize;
    let copied_block_base = 16 * head_size;
    let copied_block_end = copied_block_base + block_width;

    assert_eq!(
        &owned_seed.key_cache[..populated_width],
        &staged_inputs.key[..populated_width]
    );
    assert_eq!(
        &owned_seed.value_cache[..populated_width],
        &staged_inputs.value[..populated_width]
    );
    assert_eq!(
        &owned_seed.key_cache[copied_block_base..copied_block_base + populated_width],
        &staged_inputs.key[..populated_width]
    );
    assert_eq!(
        &owned_seed.value_cache[copied_block_base..copied_block_base + populated_width],
        &staged_inputs.value[..populated_width]
    );
    assert!(
        owned_seed.key_cache[copied_block_base + populated_width..copied_block_end]
            .iter()
            .all(|value| *value == 0.0)
    );
    assert!(
        owned_seed.value_cache[copied_block_base + populated_width..copied_block_end]
            .iter()
            .all(|value| *value == 0.0)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn composed_seed_overrides_stale_cache_with_current_staged_inputs_before_copying() {
    let mut workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill workload should resolve");
    workload.kv_slot_capacity = 32;
    workload.kv_block_capacity = 32;
    workload.kv_metadata.copy_block_mapping = vec![[0, 16]];
    let staged_inputs = synthetic_staged_inputs(&workload);
    let slot_capacity = workload.slot_numeric_capacity() as usize;
    let head_size = workload.numeric_layout.head_size() as usize;
    let populated_width = workload.scheduled_numeric_elements() as usize;
    let mut stale_key_cache = vec![-1.0_f32; slot_capacity];
    let mut stale_value_cache = vec![-2.0_f32; slot_capacity];
    let copied_block_base = 16 * head_size;

    for value in &mut stale_key_cache[copied_block_base..copied_block_base + populated_width] {
        *value = -9.0;
    }
    for value in &mut stale_value_cache[copied_block_base..copied_block_base + populated_width] {
        *value = -10.0;
    }

    let owned_seed = owned_cache_seed_with_staged_inputs_and_copy_targets(
        &workload,
        &staged_inputs,
        Some(MetalDispatchKvCacheSeed {
            key_cache: &stale_key_cache,
            value_cache: &stale_value_cache,
        }),
    )
    .expect("composed seed should build");

    assert_eq!(
        &owned_seed.key_cache[..populated_width],
        &staged_inputs.key[..populated_width]
    );
    assert_eq!(
        &owned_seed.value_cache[..populated_width],
        &staged_inputs.value[..populated_width]
    );
    assert_eq!(
        &owned_seed.key_cache[copied_block_base..copied_block_base + populated_width],
        &staged_inputs.key[..populated_width]
    );
    assert_eq!(
        &owned_seed.value_cache[copied_block_base..copied_block_base + populated_width],
        &staged_inputs.value[..populated_width]
    );
}

#[cfg(target_os = "macos")]
#[test]
fn reference_snapshot_and_layer_cache_materialize_transitive_copy_targets() {
    let mut prefill_workload =
        MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
            .expect("prefill workload should resolve");
    prefill_workload.kv_slot_capacity = 48;
    prefill_workload.kv_block_capacity = 48;
    prefill_workload.kv_metadata.copy_block_mapping = vec![[16, 32], [0, 16]];
    let prefill_staged = synthetic_staged_inputs(&prefill_workload);
    let prefill_reference = reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
    let prefill_snapshot = MetalDispatchKvCacheSnapshot::from_reference_for_workload(
        &prefill_workload,
        &prefill_reference,
    );
    let head_size = prefill_workload.numeric_layout.head_size() as usize;
    let populated_width = prefill_workload.scheduled_numeric_elements() as usize;
    let copied_block_16_base = 16 * head_size;
    let copied_block_32_base = 32 * head_size;

    assert_eq!(
        &prefill_snapshot.key_cache[copied_block_16_base..copied_block_16_base + populated_width],
        &prefill_staged.key[..populated_width]
    );
    assert_eq!(
        &prefill_snapshot.value_cache[copied_block_16_base..copied_block_16_base + populated_width],
        &prefill_staged.value[..populated_width]
    );
    assert_eq!(
        &prefill_snapshot.key_cache[copied_block_32_base..copied_block_32_base + populated_width],
        &prefill_staged.key[..populated_width]
    );
    assert_eq!(
        &prefill_snapshot.value_cache[copied_block_32_base..copied_block_32_base + populated_width],
        &prefill_staged.value[..populated_width]
    );

    let mut layer_cache = MetalPersistentLayerKvCache::default();
    layer_cache.apply_snapshot(&prefill_workload, &prefill_snapshot);
    for slot in 16..20 {
        assert!(layer_cache.slot_initialized(slot));
    }
    for slot in 32..36 {
        assert!(layer_cache.slot_initialized(slot));
    }

    let mut decode_workload =
        MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
            .expect("decode continuation workload should resolve");
    decode_workload.kv_slot_capacity = 48;
    decode_workload.kv_block_capacity = 48;
    decode_workload.kv_metadata.slot_mapping = vec![36];
    decode_workload.kv_metadata.attention_block_table = vec![36];
    decode_workload.kv_metadata.gather_block_table = vec![32];
    decode_workload.kv_metadata.gather_block_table_stride = 1;

    let layer_cache = Mutex::new(layer_cache);
    assert!(workload_supports_native_prefix_attention(
        &decode_workload,
        Some(&layer_cache)
    ));
}

#[test]
fn annotate_bringup_execution_flags_clears_numeric_scaffold_when_runtime_uses_model_tensors() {
    let mut route_metadata = RouteMetadata::empty();
    let runtime = MetalDispatchRuntimeInfo {
        device_name: "Apple M4 Max".to_string(),
        required_pipeline_count: 4,
        max_thread_execution_width: 64,
        binary_archive: MetalBinaryArchiveInfo {
            path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
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
        model_buffer_count: 4,
        model_buffer_bytes: 4096,
        native_dense_kernel_coverage: MetalNativeDenseKernelCoverage {
            projection_f32_binding_count: 1,
            projection_f16_binding_count: 2,
            projection_bf16_binding_count: 3,
            projection_unsupported_binding_count: 4,
            rms_norm_f32_binding_count: 5,
            rms_norm_f16_binding_count: 6,
            rms_norm_bf16_binding_count: 7,
            rms_norm_unsupported_binding_count: 8,
        },
        model: Some(NativeModelArtifactsSummary {
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 1,
            tensor_count: 9,
            tie_word_embeddings: false,
        }),
    };

    annotate_runtime_summary(&mut route_metadata, &runtime);
    annotate_bringup_execution_flags(
        &mut route_metadata,
        &runtime,
        &[(RequestId(9), 4)],
        MetalBringupExecutionFlags {
            model_bound_ffn_decode: false,
            native_logits_projection_decode: true,
            complete_model_forward_supported: true,
            real_model_forward: false,
            prefix_attention_tally: PrefixAttentionExecutionTally {
                native_dispatches: 1,
                cpu_reference_dispatches: 0,
                qkv_projection_tokens: 3,
                layer_continuation_tokens: 2,
                logits_projection_tokens: 1,
                logits_vocab_scan_rows: 5,
                native_projection_rows: 31,
                cpu_projection_rows: 37,
                native_rms_norm_elements: 41,
                cpu_rms_norm_elements: 43,
                native_ffn_activation_elements: 47,
                cpu_ffn_activation_elements: 53,
                native_residual_add_elements: 59,
                cpu_residual_add_elements: 61,
            },
            direct_decode_native_dense_tally: DirectDecodeNativeDenseTally {
                native_projection_rows: 17,
                cpu_projection_rows: 19,
                native_rms_norm_elements: 23,
                cpu_rms_norm_elements: 29,
                native_ffn_activation_elements: 31,
                cpu_ffn_activation_elements: 37,
                native_residual_add_elements: 41,
                cpu_residual_add_elements: 43,
                native_batched_logits_group_count: 1,
                native_batched_logits_token_count: 2,
                batched_group_fallback_count: 0,
                batched_group_fallback_token_count: 0,
            },
        },
    );

    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_numeric_scaffold_only".to_string(), 0,)));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_complete_model_forward_supported".to_string(),
        1,
    )));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_real_model_forward".to_string(), 0,)));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_direct_decode_tokens".to_string(), 1,)));
    assert!(route_metadata
        .crossover_decisions
        .iter()
        .any(|(key, value)| { key == "metal_dispatch_direct_decode_checksum_lo" && *value > 0 }));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_model_bound_ffn".to_string(),
        0,
    )));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_prefix_native_dispatch_count".to_string(), 1,)));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_cpu_reference_dispatch_count".to_string(),
        0,
    )));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_qkv_projection_token_count".to_string(), 3,)));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_layer_continuation_token_count".to_string(),
        2,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_logits_projection_token_count".to_string(),
        1,
    )));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_logits_vocab_scan_row_count".to_string(), 5,)));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_native_projection_row_count".to_string(),
        31,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_cpu_projection_row_count".to_string(),
        37,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_native_rms_norm_element_count".to_string(),
        41,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_cpu_rms_norm_element_count".to_string(),
        43,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_native_ffn_activation_element_count".to_string(),
        47,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_cpu_ffn_activation_element_count".to_string(),
        53,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_native_residual_add_element_count".to_string(),
        59,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_prefix_cpu_residual_add_element_count".to_string(),
        61,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_native_logits_projection".to_string(),
        1,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_native_projection_row_count".to_string(),
        17,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_cpu_projection_row_count".to_string(),
        19,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_native_rms_norm_element_count".to_string(),
        23,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_cpu_rms_norm_element_count".to_string(),
        29,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_native_ffn_activation_element_count".to_string(),
        31,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_cpu_ffn_activation_element_count".to_string(),
        37,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_native_residual_add_element_count".to_string(),
        41,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_cpu_residual_add_element_count".to_string(),
        43,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_batched_logits_group_count".to_string(),
        1,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_batched_logits_token_count".to_string(),
        2,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_batched_group_fallback_count".to_string(),
        0,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_direct_decode_batched_group_fallback_token_count".to_string(),
        0,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_native_projection_bf16_binding_count".to_string(),
        3,
    )));
    assert!(route_metadata.crossover_decisions.contains(&(
        "metal_dispatch_native_rms_norm_unsupported_binding_count".to_string(),
        8,
    )));
}

#[test]
fn completed_real_model_forward_step_marks_pure_decode_batch_with_no_remaining_logits() {
    let input = sample_decode_only_runner_input();
    let runtime = MetalDispatchRuntimeInfo {
        device_name: "Apple M4 Max".to_string(),
        required_pipeline_count: 4,
        max_thread_execution_width: 64,
        binary_archive: MetalBinaryArchiveInfo {
            path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
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
        model_buffer_count: 4,
        model_buffer_bytes: 4096,
        native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
        model: Some(NativeModelArtifactsSummary {
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 1,
            tensor_count: 9,
            tie_word_embeddings: false,
        }),
    };
    let mut output = successful_runner_output_from_input(&input);
    let direct_decode_tokens = vec![(RequestId(9), 17), (RequestId(11), 23)];

    let direct_logits_outputs = direct_decode_tokens
        .iter()
        .map(|(request_id, token_id)| RequestLogitsOutput {
            request_id: *request_id,
            logits: vec![0.0, *token_id as f32],
        })
        .collect::<Vec<_>>();

    apply_direct_decode_logits_to_runner_output(&mut output, &direct_logits_outputs);

    assert!(completed_real_model_forward_step(
        &input,
        &output,
        &runtime,
        &direct_decode_tokens,
    ));

    let mut route_metadata = RouteMetadata::empty();
    annotate_bringup_execution_flags(
        &mut route_metadata,
        &runtime,
        &direct_decode_tokens,
        MetalBringupExecutionFlags {
            model_bound_ffn_decode: true,
            native_logits_projection_decode: true,
            complete_model_forward_supported: true,
            real_model_forward: true,
            prefix_attention_tally: PrefixAttentionExecutionTally {
                native_dispatches: 2,
                cpu_reference_dispatches: 0,
                qkv_projection_tokens: 4,
                layer_continuation_tokens: 3,
                logits_projection_tokens: 1,
                logits_vocab_scan_rows: 7,
                native_projection_rows: 11,
                cpu_projection_rows: 13,
                native_rms_norm_elements: 17,
                cpu_rms_norm_elements: 19,
                native_ffn_activation_elements: 23,
                cpu_ffn_activation_elements: 29,
                native_residual_add_elements: 31,
                cpu_residual_add_elements: 37,
            },
            direct_decode_native_dense_tally: DirectDecodeNativeDenseTally::default(),
        },
    );
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_real_model_forward".to_string(), 1,)));
}

#[test]
fn completed_real_model_forward_step_accepts_mixed_prefill_decode_batches_when_decode_items_resolve(
) {
    let input = sample_runner_input();
    let runtime = MetalDispatchRuntimeInfo {
        device_name: "Apple M4 Max".to_string(),
        required_pipeline_count: 4,
        max_thread_execution_width: 64,
        binary_archive: MetalBinaryArchiveInfo {
            path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
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
        model_buffer_count: 4,
        model_buffer_bytes: 4096,
        native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
        model: Some(NativeModelArtifactsSummary {
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 1,
            tensor_count: 9,
            tie_word_embeddings: false,
        }),
    };
    let mut output = successful_runner_output_from_input(&input);
    let direct_decode_tokens = vec![(RequestId(9), 17)];

    let direct_logits_outputs = direct_decode_tokens
        .iter()
        .map(|(request_id, token_id)| RequestLogitsOutput {
            request_id: *request_id,
            logits: vec![0.0, *token_id as f32],
        })
        .collect::<Vec<_>>();

    apply_direct_decode_logits_to_runner_output(&mut output, &direct_logits_outputs);

    assert!(completed_real_model_forward_step(
        &input,
        &output,
        &runtime,
        &direct_decode_tokens,
    ));
}

#[test]
fn completed_real_model_forward_step_marks_prefill_only_batch_without_remaining_logits() {
    let input = sample_prefill_only_runner_input();
    let runtime = MetalDispatchRuntimeInfo {
        device_name: "Apple M4 Max".to_string(),
        required_pipeline_count: 4,
        max_thread_execution_width: 64,
        binary_archive: MetalBinaryArchiveInfo {
            path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
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
        model_buffer_count: 4,
        model_buffer_bytes: 4096,
        native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
        model: Some(NativeModelArtifactsSummary {
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 1,
            tensor_count: 9,
            tie_word_embeddings: false,
        }),
    };
    let output = successful_runner_output_from_input(&input);

    assert!(completed_real_model_forward_step(
        &input,
        &output,
        &runtime,
        &[],
    ));
}

#[test]
fn completed_real_model_forward_step_accepts_multilayer_runtime_when_prefix_attention_is_native() {
    let input = sample_decode_only_runner_input();
    let runtime = MetalDispatchRuntimeInfo {
        device_name: "Apple M4 Max".to_string(),
        required_pipeline_count: 4,
        max_thread_execution_width: 64,
        binary_archive: MetalBinaryArchiveInfo {
            path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
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
        model_buffer_count: 4,
        model_buffer_bytes: 4096,
        native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
        model: Some(NativeModelArtifactsSummary {
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 2,
            tensor_count: 18,
            tie_word_embeddings: false,
        }),
    };
    let unresolved_output = successful_runner_output_from_input(&input);
    let direct_decode_tokens = vec![(RequestId(9), 17), (RequestId(11), 23)];

    assert!(!completed_real_model_forward_step(
        &input,
        &unresolved_output,
        &runtime,
        &[],
    ));

    let mut output = unresolved_output;
    let direct_logits_outputs = direct_decode_tokens
        .iter()
        .map(|(request_id, token_id)| RequestLogitsOutput {
            request_id: *request_id,
            logits: vec![0.0, *token_id as f32],
        })
        .collect::<Vec<_>>();

    apply_direct_decode_logits_to_runner_output(&mut output, &direct_logits_outputs);

    assert!(completed_real_model_forward_step(
        &input,
        &output,
        &runtime,
        &direct_decode_tokens,
    ));
}

#[test]
fn completed_real_model_forward_step_rejects_multilayer_runtime_when_prefix_attention_is_cpu_reference(
) {
    let input = sample_decode_only_runner_input();
    let runtime = MetalDispatchRuntimeInfo {
        device_name: "Apple M4 Max".to_string(),
        required_pipeline_count: 4,
        max_thread_execution_width: 64,
        binary_archive: MetalBinaryArchiveInfo {
            path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
            state: MetalBinaryArchiveState::Loaded,
            attached_pipeline_count: 4,
            serialized: true,
            note: None,
        },
        command_queue_ready: true,
        model_conditioned_inputs: true,
        real_model_tensor_inputs: true,
        complete_model_forward_supported: false,
        model_bindings_prepared: true,
        model_buffers_bound: true,
        model_buffer_count: 4,
        model_buffer_bytes: 4096,
        native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
        model: Some(NativeModelArtifactsSummary {
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 2,
            tensor_count: 18,
            tie_word_embeddings: false,
        }),
    };
    let mut output = successful_runner_output_from_input(&input);
    let direct_decode_tokens = vec![(RequestId(9), 17), (RequestId(11), 23)];

    let direct_logits_outputs = direct_decode_tokens
        .iter()
        .map(|(request_id, token_id)| RequestLogitsOutput {
            request_id: *request_id,
            logits: vec![0.0, *token_id as f32],
        })
        .collect::<Vec<_>>();

    apply_direct_decode_logits_to_runner_output(&mut output, &direct_logits_outputs);

    assert!(!completed_real_model_forward_step(
        &input,
        &output,
        &runtime,
        &direct_decode_tokens,
    ));
}

#[test]
fn completed_real_model_forward_step_accepts_multilayer_runtime_when_prefix_attention_is_mixed() {
    let input = sample_decode_only_runner_input();
    let runtime = MetalDispatchRuntimeInfo {
        device_name: "Apple M4 Max".to_string(),
        required_pipeline_count: 4,
        max_thread_execution_width: 64,
        binary_archive: MetalBinaryArchiveInfo {
            path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
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
        model_buffer_count: 4,
        model_buffer_bytes: 4096,
        native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
        model: Some(NativeModelArtifactsSummary {
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 2,
            tensor_count: 18,
            tie_word_embeddings: false,
        }),
    };
    let mut output = successful_runner_output_from_input(&input);
    let direct_decode_tokens = vec![(RequestId(9), 17), (RequestId(11), 23)];
    let direct_logits_outputs = direct_decode_tokens
        .iter()
        .map(|(request_id, token_id)| RequestLogitsOutput {
            request_id: *request_id,
            logits: vec![0.0, *token_id as f32],
        })
        .collect::<Vec<_>>();

    apply_direct_decode_logits_to_runner_output(&mut output, &direct_logits_outputs);

    assert!(completed_real_model_forward_step(
        &input,
        &output,
        &runtime,
        &direct_decode_tokens,
    ));
}

#[test]
fn metal_assets_reject_missing_required_kernel() {
    let fixture = write_phase1_fixture(
        MetalBuildStatus::Compiled,
        Some(|manifest: &mut MetalKernelManifest| {
            manifest
                .kernels
                .retain(|kernel| kernel.name != "copy_blocks");
        }),
    );

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject missing required kernel");
    let MetalRuntimeError::InvalidManifest { message } = error else {
        panic!("expected invalid manifest error");
    };
    assert!(message.contains("missing required kernel copy_blocks"));

    fixture.cleanup();
}

#[test]
fn metal_assets_reject_missing_deferred_kernel() {
    let fixture = write_phase1_fixture(
        MetalBuildStatus::Compiled,
        Some(|manifest: &mut MetalKernelManifest| {
            manifest
                .kernels
                .retain(|kernel| kernel.name != "swap_blocks");
        }),
    );

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject missing deferred kernel inventory");
    let MetalRuntimeError::InvalidManifest { message } = error else {
        panic!("expected invalid manifest error");
    };
    assert!(message.contains("missing deferred kernel swap_blocks"));

    fixture.cleanup();
}

#[test]
fn metal_assets_reject_non_phase1_block_size_policy() {
    let fixture = write_phase1_fixture(
        MetalBuildStatus::Compiled,
        Some(|manifest: &mut MetalKernelManifest| {
            manifest.supported_block_size_tokens = vec![8, 16];
        }),
    );

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject unsupported block size policy");
    let MetalRuntimeError::InvalidBuildReport { message } = error else {
        panic!("expected invalid build report error");
    };
    assert!(message.contains("supported_block_size_tokens must be multiples of"));

    fixture.cleanup();
}

#[test]
fn metal_asset_validator_loads_metallib_and_preserves_asset_contract() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);
    let validator =
        MetalAssetValidator::from_build_dir(&fixture.build_dir).expect("validator should load");

    validator
        .validate_block_size_tokens(PHASE1_DEFAULT_BLOCK_SIZE_TOKENS)
        .expect("validator should accept supported native block size");
    assert_eq!(
        validator
            .metallib()
            .path
            .file_name()
            .and_then(|name| name.to_str()),
        Some("ax_phase1_dense_path.metallib")
    );
    assert_eq!(
        validator.resolved_kernel_names().len(),
        PHASE1_REQUIRED_METAL_KERNELS.len()
    );

    fixture.cleanup();
}

#[test]
fn metal_asset_validator_fails_closed_on_unsupported_native_block_size() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);
    let validator =
        MetalAssetValidator::from_build_dir(&fixture.build_dir).expect("validator should load");

    let error = validator
        .validate_block_size_tokens(8)
        .expect_err("validator should reject unsupported native block size");
    let MetalRuntimeError::UnsupportedNativeBlockSize {
        block_size_tokens,
        default_block_size_tokens,
        supported_block_size_tokens,
    } = error
    else {
        panic!("expected unsupported native block size error");
    };
    assert_eq!(block_size_tokens, 8);
    assert_eq!(default_block_size_tokens, PHASE1_DEFAULT_BLOCK_SIZE_TOKENS);
    assert_eq!(
        supported_block_size_tokens,
        PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS
    );

    fixture.cleanup();
}

#[test]
fn metal_dispatch_workload_tracks_tokens_blocks_and_modes() {
    let input = sample_runner_input();

    let workload =
        MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");

    assert_eq!(workload.scheduled_requests, 2);
    assert_eq!(workload.prefill_requests, 1);
    assert_eq!(workload.decode_requests, 1);
    assert_eq!(workload.scheduled_tokens, 4);
    assert_eq!(workload.scheduled_token_ids, vec![1, 2, 3, 4]);
    assert_eq!(workload.scheduled_positions, vec![0, 1, 2, 3]);
    assert_eq!(workload.resolved_blocks, 3);
    assert_eq!(workload.token_elements, 4);
    assert_eq!(workload.block_elements, 3);
    assert_eq!(workload.scratch_elements, 4);
    assert_eq!(workload.scheduled_numeric_elements(), 32);
    assert_eq!(workload.gather_numeric_elements(), 56);
    assert_eq!(workload.slot_numeric_capacity(), 384);
    assert_eq!(workload.kv_slot_capacity, 48);
    assert_eq!(workload.kv_block_capacity, 48);
    assert_eq!(workload.kv_metadata.slot_mapping, vec![0, 1, 2, 35]);
    assert_eq!(
        workload.kv_metadata.attention_block_table,
        vec![0, 1, 2, 35]
    );
    assert_eq!(workload.kv_metadata.gather_block_table, vec![0, 16, 32, 0]);
    assert_eq!(workload.kv_metadata.gather_block_table_stride, 2);
    assert_eq!(workload.kv_metadata.copy_block_mapping, vec![[0, 0]]);
    assert_eq!(workload.kv_metadata.seq_lens, vec![3, 4]);
    assert_eq!(workload.kv_metadata.cu_seq_lens, vec![0, 3, 7]);
    assert_eq!(workload.kv_metadata.scheduled_cu_seq_lens, vec![0, 3, 4]);
}

#[test]
fn metal_dispatch_trace_builder_respects_pipeline_widths_and_kernel_kinds() {
    let workload = MetalDispatchWorkload {
        scheduled_requests: 2,
        prefill_requests: 1,
        decode_requests: 1,
        scheduled_tokens: 8,
        scheduled_token_ids: vec![11, 12, 13, 14, 15, 16, 17, 18],
        scheduled_positions: vec![0, 1, 2, 3, 4, 5, 6, 7],
        resolved_blocks: 2,
        token_elements: 8,
        block_elements: 2,
        scratch_elements: 8,
        kv_slot_capacity: 8,
        kv_block_capacity: 24,
        numeric_layout: MetalDispatchNumericLayout::default(),
        kv_metadata: MetalDispatchKvMetadata {
            block_size_tokens: 16,
            slot_mapping: vec![0, 1, 2, 3, 4, 5, 6, 7],
            attention_block_table: vec![0, 1, 2, 3, 20, 21, 22, 23],
            gather_block_table: vec![0, 16],
            gather_block_table_stride: 2,
            copy_block_mapping: vec![[0, 0], [16, 16]],
            seq_lens: vec![8],
            cu_seq_lens: vec![0, 8],
            scheduled_cu_seq_lens: vec![0, 8],
        },
    };
    let traces = build_dispatch_traces(
        &workload,
        &[
            MetalComputePipelineInfo {
                function_name: "reshape_and_cache".to_string(),
                thread_execution_width: 32,
                max_total_threads_per_threadgroup: 128,
                static_threadgroup_memory_length: 0,
            },
            MetalComputePipelineInfo {
                function_name: "copy_blocks".to_string(),
                thread_execution_width: 64,
                max_total_threads_per_threadgroup: 64,
                static_threadgroup_memory_length: 0,
            },
        ],
    );

    assert_eq!(traces.len(), 2);
    assert_eq!(traces[0].function_name, "reshape_and_cache");
    assert_eq!(traces[0].element_count, 64);
    assert_eq!(traces[0].threads_per_grid.width, 64);
    assert_eq!(traces[0].threads_per_threadgroup.width, 32);
    assert_eq!(traces[1].function_name, "copy_blocks");
    assert_eq!(traces[1].element_count, workload.copy_numeric_elements());
    assert_eq!(
        traces[1].threads_per_grid.width,
        u64::from(workload.copy_numeric_elements())
    );
    assert_eq!(traces[1].threads_per_threadgroup.width, 64);
}

#[test]
fn metal_dispatch_trace_builder_runs_gather_before_attention() {
    let workload =
        MetalDispatchWorkload::from_runner_input(&sample_runner_input()).expect("workload");
    let traces = build_dispatch_traces(
        &workload,
        &[
            MetalComputePipelineInfo {
                function_name: "paged_decode_attention".to_string(),
                thread_execution_width: 32,
                max_total_threads_per_threadgroup: 128,
                static_threadgroup_memory_length: 0,
            },
            MetalComputePipelineInfo {
                function_name: "copy_blocks".to_string(),
                thread_execution_width: 32,
                max_total_threads_per_threadgroup: 128,
                static_threadgroup_memory_length: 0,
            },
            MetalComputePipelineInfo {
                function_name: "gather_kv_cache".to_string(),
                thread_execution_width: 32,
                max_total_threads_per_threadgroup: 128,
                static_threadgroup_memory_length: 0,
            },
            MetalComputePipelineInfo {
                function_name: "reshape_and_cache".to_string(),
                thread_execution_width: 32,
                max_total_threads_per_threadgroup: 128,
                static_threadgroup_memory_length: 0,
            },
        ],
    );

    assert_eq!(
        traces
            .iter()
            .map(|trace| trace.function_name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "reshape_and_cache",
            "gather_kv_cache",
            "paged_decode_attention",
            "copy_blocks",
        ]
    );
    assert_eq!(traces[1].element_count, workload.gather_numeric_elements());
    assert_eq!(
        traces[2].element_count,
        workload.attention_numeric_elements()
    );
}

#[test]
fn metal_dispatch_workload_rejects_request_without_required_block_table_span() {
    let mut input = sample_runner_input();
    input.block_tables[0].block_table.block_ids.clear();

    let error = MetalDispatchWorkload::from_runner_input(&input)
        .expect_err("workload should reject missing block coverage");
    let MetalRuntimeError::InvalidDispatchInput { message } = error else {
        panic!("expected invalid dispatch input");
    };
    assert!(message.contains("requires block index 0"));
}

#[test]
fn failed_metal_runner_output_marks_all_requests_as_errors() {
    let input = sample_runner_input();
    let workload =
        MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
    let runtime = MetalDispatchRuntimeInfo {
        device_name: "Apple M4 Max".to_string(),
        required_pipeline_count: 4,
        max_thread_execution_width: 64,
        binary_archive: MetalBinaryArchiveInfo {
            path: PathBuf::from("/tmp/ax_phase1_dense_path.binary_archive.metallib"),
            state: MetalBinaryArchiveState::Loaded,
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
        native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
        model: None,
    };

    let output = failed_runner_output_from_input(
        &input,
        "metal dispatch exploded".to_string(),
        &workload,
        Some(&runtime),
    );

    assert_eq!(output.execution_status, ExecutionStatus::Failed);
    assert!(output.logits_handles.is_empty());
    assert!(output.logits_outputs.is_empty());
    assert_eq!(output.kv_write_summary.tokens_written, 0);
    assert_eq!(output.kv_write_summary.blocks_touched, 0);
    assert!(output
        .route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_failed".to_string(), 1)));
    assert!(output
        .route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_runtime_required_pipelines".to_string(), 4)));
    assert!(output
        .route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_binary_archive_state".to_string(), 2)));
    assert_eq!(output.request_updates.len(), 2);
    assert!(output
        .request_updates
        .iter()
        .all(|update| update.stop_reason == Some(StopReason::Error)
            && update.error.as_deref() == Some("metal dispatch exploded")));
}

#[test]
fn derive_metal_decode_tokens_uses_attention_output_bits_for_decode_items() {
    let input = sample_runner_input();
    let direct_decode_tokens = derive_metal_decode_tokens_from_attention_bits(
        &input,
        &[
            1001.0_f32.to_bits(),
            1001.0_f32.to_bits(),
            1001.0_f32.to_bits(),
            1001.0_f32.to_bits(),
            1001.0_f32.to_bits(),
            1001.0_f32.to_bits(),
            1001.0_f32.to_bits(),
            1001.0_f32.to_bits(),
            1002.0_f32.to_bits(),
            1002.0_f32.to_bits(),
            1002.0_f32.to_bits(),
            1002.0_f32.to_bits(),
            1002.0_f32.to_bits(),
            1002.0_f32.to_bits(),
            1002.0_f32.to_bits(),
            1002.0_f32.to_bits(),
            1003.0_f32.to_bits(),
            1003.0_f32.to_bits(),
            1003.0_f32.to_bits(),
            1003.0_f32.to_bits(),
            1003.0_f32.to_bits(),
            1003.0_f32.to_bits(),
            1003.0_f32.to_bits(),
            1003.0_f32.to_bits(),
            3015.0_f32.to_bits(),
            3015.0_f32.to_bits(),
            3015.0_f32.to_bits(),
            3015.0_f32.to_bits(),
            3015.0_f32.to_bits(),
            3015.0_f32.to_bits(),
            3015.0_f32.to_bits(),
            3015.0_f32.to_bits(),
        ],
    );

    assert_eq!(direct_decode_tokens, vec![(RequestId(9), 3015)]);
}

#[test]
fn apply_direct_decode_logits_clears_logits_handles_for_resolved_requests() {
    let input = sample_runner_input();
    let mut output = successful_runner_output_from_input(&input);

    apply_direct_decode_logits_to_runner_output(
        &mut output,
        &[RequestLogitsOutput {
            request_id: RequestId(9),
            logits: vec![0.1, 3015.0],
        }],
    );

    assert!(output.logits_handles.is_empty());
    assert_eq!(output.logits_outputs.len(), 1);
    assert_eq!(
        output
            .logits_outputs
            .iter()
            .find(|output| output.request_id == RequestId(9))
            .map(|output| output.logits.clone()),
        Some(vec![0.1, 3015.0])
    );
}

#[test]
fn metal_dispatch_execution_info_tracks_direct_decode_resolution() {
    let input = sample_runner_input();
    let mut output = successful_runner_output_from_input(&input);
    let direct_decode_tokens = vec![(RequestId(9), 3015)];

    apply_direct_decode_logits_to_runner_output(
        &mut output,
        &[RequestLogitsOutput {
            request_id: RequestId(9),
            logits: vec![0.1, 3015.0],
        }],
    );

    let execution = metal_dispatch_execution_info(
        &output,
        &direct_decode_tokens,
        true,
        true,
        PrefixAttentionExecutionTally {
            native_dispatches: 2,
            cpu_reference_dispatches: 1,
            qkv_projection_tokens: 6,
            layer_continuation_tokens: 4,
            logits_projection_tokens: 1,
            logits_vocab_scan_rows: 9,
            native_projection_rows: 23,
            cpu_projection_rows: 29,
            native_rms_norm_elements: 31,
            cpu_rms_norm_elements: 37,
            native_ffn_activation_elements: 41,
            cpu_ffn_activation_elements: 43,
            native_residual_add_elements: 47,
            cpu_residual_add_elements: 53,
        },
        DirectDecodeNativeDenseTally {
            native_projection_rows: 13,
            cpu_projection_rows: 17,
            native_rms_norm_elements: 19,
            cpu_rms_norm_elements: 23,
            native_ffn_activation_elements: 29,
            cpu_ffn_activation_elements: 31,
            native_residual_add_elements: 37,
            cpu_residual_add_elements: 41,
            native_batched_logits_group_count: 1,
            native_batched_logits_token_count: 2,
            batched_group_fallback_count: 0,
            batched_group_fallback_token_count: 0,
        },
    );

    assert_eq!(execution.direct_decode_token_count, 1);
    assert!(execution.direct_decode_checksum_lo > 0);
    assert_eq!(execution.logits_output_count, 1);
    assert_eq!(execution.remaining_logits_handle_count, 0);
    assert!(execution.model_bound_ffn_decode);
    assert!(execution.real_model_forward_completed);
    assert_eq!(execution.prefix_native_dispatch_count, 2);
    assert_eq!(execution.prefix_cpu_reference_dispatch_count, 1);
    assert_eq!(execution.qkv_projection_token_count, 6);
    assert_eq!(execution.layer_continuation_token_count, 4);
    assert_eq!(execution.logits_projection_token_count, 1);
    assert_eq!(execution.logits_vocab_scan_row_count, 9);
    assert_eq!(execution.prefix_native_projection_row_count, 23);
    assert_eq!(execution.prefix_cpu_projection_row_count, 29);
    assert_eq!(execution.prefix_native_rms_norm_element_count, 31);
    assert_eq!(execution.prefix_cpu_rms_norm_element_count, 37);
    assert_eq!(execution.prefix_native_ffn_activation_element_count, 41);
    assert_eq!(execution.prefix_cpu_ffn_activation_element_count, 43);
    assert_eq!(execution.prefix_native_residual_add_element_count, 47);
    assert_eq!(execution.prefix_cpu_residual_add_element_count, 53);
    assert_eq!(execution.direct_decode_native_projection_row_count, 13);
    assert_eq!(execution.direct_decode_cpu_projection_row_count, 17);
    assert_eq!(execution.direct_decode_native_rms_norm_element_count, 19);
    assert_eq!(execution.direct_decode_cpu_rms_norm_element_count, 23);
    assert_eq!(
        execution.direct_decode_native_ffn_activation_element_count,
        29
    );
    assert_eq!(execution.direct_decode_cpu_ffn_activation_element_count, 31);
    assert_eq!(
        execution.direct_decode_native_residual_add_element_count,
        37
    );
    assert_eq!(execution.direct_decode_cpu_residual_add_element_count, 41);
    assert_eq!(execution.direct_decode_batched_logits_group_count, 1);
    assert_eq!(execution.direct_decode_batched_logits_token_count, 2);
    assert_eq!(execution.direct_decode_batched_group_fallback_count, 0);
    assert_eq!(
        execution.direct_decode_batched_group_fallback_token_count,
        0
    );
}

#[test]
fn staged_numeric_values_follow_scheduled_token_ids() {
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let staged_key = staged_key_values(&workload);
    let staged_value = staged_value_values(&workload);
    let staged_query = staged_query_values(&workload);

    assert_eq!(staged_key.len(), 32);
    assert_eq!(staged_value.len(), 32);
    assert_eq!(staged_query.len(), 32);
    assert_eq!(
        &staged_key[..8],
        &[0.5, 0.53125, 0.5625, 0.59375, 0.75, 0.78125, 0.8125, 0.84375]
    );
    assert_eq!(
        &staged_value[8..16],
        &[2.0, 2.015625, 2.03125, 2.046875, 2.0625, 2.078125, 2.09375, 2.109375,]
    );
    assert_eq!(
        &staged_query[24..32],
        &[2.015625, 2.046875, 2.078125, 2.109375, 2.265625, 2.296875, 2.328125, 2.359375,]
    );
}

#[test]
fn simulated_numeric_path_keeps_vectorized_kv_and_decode_contract() {
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");
    let simulated = simulated_numeric_path(&workload);

    let decode_slot_base = 35 * PHASE1_NUMERIC_HEAD_SIZE as usize;
    assert_eq!(
        &simulated.key_cache[decode_slot_base..decode_slot_base + 8],
        &[2.0, 2.03125, 2.0625, 2.09375, 2.25, 2.28125, 2.3125, 2.34375]
    );
    assert_eq!(simulated.gather_key.len(), 56);
    assert_eq!(simulated.gather_value.len(), 56);
    assert_eq!(simulated.attention_output.len(), 32);
    assert!(simulated
        .attention_output
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0));
    let decode_output_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
    assert_eq!(
        decode_token_from_attention_bits(
            &simulated.attention_output[decode_output_base..decode_output_base + 8]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        ),
        4
    );
    // copy_block_mapping for sample_runner_input is the default dummy [[0, 0]],
    // so only block 0 is copied. Verify the copied block matches key_cache at that range.
    let block_width = workload.block_numeric_elements() as usize;
    assert_eq!(
        &simulated.copy_key[..block_width],
        &simulated.key_cache[..block_width]
    );
    assert_eq!(
        &simulated.copy_value[..block_width],
        &simulated.value_cache[..block_width]
    );
    // Remaining slots in copy buffers should be zero (no other blocks were copied).
    assert!(simulated.copy_key[block_width..].iter().all(|v| *v == 0.0));
    assert!(simulated.copy_value[block_width..]
        .iter()
        .all(|v| *v == 0.0));
}

#[cfg(target_os = "macos")]
#[test]
fn numeric_trace_reference_validation_accepts_reference_path() {
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");
    let simulated = simulated_numeric_path(&workload);
    let trace = simulated_numeric_trace(&simulated);

    let summary = validate_numeric_trace_against_reference(&workload, &trace)
        .expect("reference trace should validate");
    assert_eq!(
        summary.expected_key_cache_checksum,
        trace.key_cache_checksum
    );
    assert_eq!(
        summary.expected_attention_output_checksum,
        trace.attention_output_checksum
    );
    assert_eq!(
        summary.expected_gather_output_checksum,
        trace.gather_output_checksum
    );
    assert_eq!(
        summary.expected_copy_output_checksum,
        trace.copy_output_checksum
    );
    assert_eq!(summary.attention_max_abs_diff_microunits, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn numeric_trace_reference_validation_rejects_attention_drift() {
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");
    let simulated = simulated_numeric_path(&workload);
    let mut trace = simulated_numeric_trace(&simulated);
    trace.attention_output_bits[7] =
        (f32::from_bits(trace.attention_output_bits[7]) + 0.25).to_bits();

    let error = validate_numeric_trace_against_reference(&workload, &trace)
        .expect_err("drifted attention output should fail validation");
    let MetalRuntimeError::NumericValidationMismatch { stage, message } = error else {
        panic!("expected numeric validation mismatch");
    };
    assert_eq!(stage, "attention_output");
    assert!(message.contains("value drift"));
}

#[cfg(target_os = "macos")]
#[test]
fn numeric_trace_reference_validation_rejects_copy_checksum_drift() {
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");
    let simulated = simulated_numeric_path(&workload);
    let mut trace = simulated_numeric_trace(&simulated);
    trace.copy_output_checksum ^= 0x55;

    let error = validate_numeric_trace_against_reference(&workload, &trace)
        .expect_err("copy checksum drift should fail validation");
    let MetalRuntimeError::NumericValidationMismatch { stage, message } = error else {
        panic!("expected numeric validation mismatch");
    };
    assert_eq!(stage, "copy_blocks");
    assert!(message.contains("checksum mismatch"));
}

#[test]
fn annotate_successful_dispatch_surfaces_validation_summary_metrics() {
    let input = sample_runner_input();
    let workload = MetalDispatchWorkload::from_runner_input(&input).expect("workload");
    let expected_copy_workload_elements = workload.copy_numeric_elements();
    let simulated = simulated_numeric_path(&workload);
    let mut trace = simulated_numeric_trace(&simulated);
    trace.validation = Some(
        validate_numeric_trace_against_reference(&workload, &trace)
            .expect("reference summary should exist"),
    );

    let mut route_metadata = RouteMetadata::empty();
    annotate_successful_dispatch(
        &mut route_metadata,
        &MetalDispatchTrace {
            command_queue_label: "test.queue".to_string(),
            command_buffer_label: "test.buffer".to_string(),
            command_buffer_status: MetalCommandBufferStatus::Completed,
            runtime: MetalDispatchRuntimeInfo {
                device_name: "Apple M4 Max".to_string(),
                required_pipeline_count: 4,
                max_thread_execution_width: 64,
                binary_archive: MetalBinaryArchiveInfo {
                    path: PathBuf::from("/tmp/ax_phase1_dense_path.binary_archive.metallib"),
                    state: MetalBinaryArchiveState::Loaded,
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
                native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
                model: None,
            },
            workload,
            arena: MetalDispatchArenaInfo {
                token_capacity: 8,
                slot_capacity: 64,
                attention_ref_capacity: 8,
                gather_ref_capacity: 8,
                gather_output_capacity: 8,
                copy_pair_capacity: 4,
                sequence_capacity: 4,
                reused_existing: false,
                grew_existing: false,
            },
            execution: MetalDispatchExecutionInfo::default(),
            kernels: Vec::new(),
            numeric: trace,
        },
    );

    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_numeric_reference_validated".to_string(), 1,)));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_runtime_required_pipelines".to_string(), 4,)));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_binary_archive_state".to_string(), 2,)));
    assert!(route_metadata
        .crossover_decisions
        .contains(&("metal_dispatch_binary_archive_serialized".to_string(), 1,)));
    assert!(route_metadata
        .crossover_decisions
        .iter()
        .any(|(key, value)| {
            key == "metal_dispatch_copy_workload_elements"
                && *value == expected_copy_workload_elements
        }));
    assert!(route_metadata
        .crossover_decisions
        .iter()
        .any(|(key, value)| {
            key == "metal_dispatch_attention_max_abs_diff_microunits" && *value == 0
        }));
}

#[test]
fn metal_runtime_bringup_requires_compiled_build_report() {
    let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);

    let error = MetalRuntimeBringup::from_build_dir(&fixture.build_dir)
        .expect_err("bring-up should require compiled build artifacts");
    let MetalRuntimeError::BuildNotCompiled { status } = error else {
        panic!("expected build-not-compiled error");
    };
    assert_eq!(status, MetalBuildStatus::SkippedToolchainUnavailable);

    fixture.cleanup();
}

#[cfg(target_os = "macos")]
#[test]
fn metal_runtime_bringup_rejects_invalid_metallib_bytes() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);

    let error = MetalRuntimeBringup::from_build_dir(&fixture.build_dir)
        .expect_err("bring-up should reject fake metallib bytes");
    let MetalRuntimeError::LoadCompiledLibrary { path, message } = error else {
        panic!("expected metallib load error");
    };
    assert_eq!(
        path.file_name().and_then(|name| name.to_str()),
        Some("ax_phase1_dense_path.metallib")
    );
    assert!(!message.is_empty());

    fixture.cleanup();
}

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_runtime_bringup_requires_macos_host() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);

    let error = MetalRuntimeBringup::from_build_dir(&fixture.build_dir)
        .expect_err("bring-up should reject non-macos hosts");
    let MetalRuntimeError::UnsupportedPlatform { host_os } = error else {
        panic!("expected unsupported platform error");
    };
    assert_eq!(host_os, std::env::consts::OS);

    fixture.cleanup();
}

#[test]
fn metal_kernel_builder_skips_when_toolchain_is_unavailable() {
    let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);
    let request = MetalKernelBuildRequest {
        manifest_path: fixture.root.join("metal/phase1-kernels.json"),
        output_dir: fixture.build_dir.clone(),
        doctor: sample_build_doctor(false, false),
    };

    let artifacts =
        build_phase1_kernel_artifacts(&request).expect("builder should emit skipped report");

    assert_eq!(
        artifacts.build_status(),
        MetalBuildStatus::SkippedToolchainUnavailable
    );
    assert!(artifacts.build_report.compile_commands.is_empty());
    assert!(artifacts.build_report.outputs.air.is_none());
    assert!(artifacts.build_report.outputs.metalar.is_none());
    assert!(artifacts.build_report.outputs.metallib.is_none());
    assert!(artifacts.doctor_path.is_file());
    assert!(artifacts.build_report_path.is_file());
    let summary =
        fs::read_to_string(&artifacts.summary_path).expect("summary file should be readable");
    assert!(summary.contains("skipped_toolchain_unavailable"));

    fixture.cleanup();
}

#[test]
fn metal_kernel_builder_fails_closed_on_source_manifest_drift() {
    let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);
    let source_path = fixture.root.join("metal/kernels/phase1_dense_path.metal");
    fs::write(
        &source_path,
        r#"
kernel void reshape_and_cache() {}
kernel void paged_decode_attention() {}
kernel void gather_kv_cache() {}
kernel void copy_blocks() {}
kernel void swap_blocks() {}
"#,
    )
    .expect("drifted source should write");

    let request = MetalKernelBuildRequest {
        manifest_path: fixture.root.join("metal/phase1-kernels.json"),
        output_dir: fixture.build_dir.clone(),
        doctor: sample_build_doctor(true, true),
    };

    let artifacts =
        build_phase1_kernel_artifacts(&request).expect("builder should emit failed report");

    assert_eq!(artifacts.build_status(), MetalBuildStatus::FailedCompile);
    assert!(artifacts.build_report.compile_commands.is_empty());
    assert!(artifacts
        .build_report
        .reason
        .as_deref()
        .is_some_and(|reason| reason.contains("kv_scale_update")));
    assert!(artifacts.build_report.outputs.metallib.is_none());

    fixture.cleanup();
}

#[test]
fn metal_kernel_builder_compiles_with_fake_xcrun_toolchain() {
    let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);
    let bin_dir = fixture.root.join("fake-bin");
    fs::create_dir_all(&bin_dir).expect("fake bin directory should create");
    let fake_xcrun = bin_dir.join("xcrun");
    fs::write(&fake_xcrun, fake_xcrun_script()).expect("fake xcrun should write");
    let mut permissions = fs::metadata(&fake_xcrun)
        .expect("fake xcrun metadata should load")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_xcrun, permissions).expect("fake xcrun should be executable");

    let request = MetalKernelBuildRequest {
        manifest_path: fixture.root.join("metal/phase1-kernels.json"),
        output_dir: fixture.build_dir.clone(),
        doctor: sample_build_doctor(true, true),
    };

    let _guard = env_lock().lock().expect("env lock should acquire");
    let original_path = std::env::var_os("PATH");
    let fake_path = prepend_to_path(&bin_dir, original_path.as_ref());
    std::env::set_var("PATH", &fake_path);

    let artifacts = build_phase1_kernel_artifacts(&request)
        .expect("builder should compile through fake toolchain");

    if let Some(path) = original_path {
        std::env::set_var("PATH", path);
    } else {
        std::env::remove_var("PATH");
    }

    assert_eq!(artifacts.build_status(), MetalBuildStatus::Compiled);
    assert!(!artifacts.reused_existing_artifacts());
    assert_eq!(artifacts.build_report.compile_commands.len(), 3);
    assert!(artifacts
        .build_report
        .outputs
        .air
        .as_deref()
        .is_some_and(Path::is_file));
    assert!(artifacts
        .build_report
        .outputs
        .metalar
        .as_deref()
        .is_some_and(Path::is_file));
    assert!(artifacts
        .build_report
        .outputs
        .metallib
        .as_deref()
        .is_some_and(Path::is_file));
    assert_eq!(
        artifacts
            .build_report
            .outputs
            .metallib_sha256
            .as_deref()
            .map(str::len),
        Some(64)
    );

    fixture.cleanup();
}

#[test]
fn metal_kernel_builder_reuses_valid_compiled_artifacts_without_recompiling() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);
    let bin_dir = fixture.root.join("fake-bin");
    fs::create_dir_all(&bin_dir).expect("fake bin directory should create");
    let fake_xcrun = bin_dir.join("xcrun");
    fs::write(&fake_xcrun, fake_failing_xcrun_script()).expect("fake failing xcrun should write");
    let mut permissions = fs::metadata(&fake_xcrun)
        .expect("fake failing xcrun metadata should load")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_xcrun, permissions).expect("fake failing xcrun should be executable");

    let mut doctor = sample_build_doctor(true, true);
    doctor.metal_toolchain.metal.version = Some("Apple metal version 99999.1".to_string());

    let request = MetalKernelBuildRequest {
        manifest_path: fixture.root.join("metal/phase1-kernels.json"),
        output_dir: fixture.build_dir.clone(),
        doctor: doctor.clone(),
    };

    let _guard = env_lock().lock().expect("env lock should acquire");
    let original_path = std::env::var_os("PATH");
    let fake_path = prepend_to_path(&bin_dir, original_path.as_ref());
    std::env::set_var("PATH", &fake_path);

    let artifacts = build_phase1_kernel_artifacts(&request)
        .expect("builder should reuse validated compiled artifacts");

    if let Some(path) = original_path {
        std::env::set_var("PATH", path);
    } else {
        std::env::remove_var("PATH");
    }

    assert_eq!(artifacts.build_status(), MetalBuildStatus::Compiled);
    assert!(artifacts.reused_existing_artifacts());
    assert_eq!(artifacts.build_report.doctor, doctor);
    assert!(artifacts
        .build_report
        .outputs
        .metallib
        .as_deref()
        .is_some_and(Path::is_file));

    fixture.cleanup();
}

fn write_phase1_fixture(
    status: MetalBuildStatus,
    manifest_edit: Option<fn(&mut MetalKernelManifest)>,
) -> Phase1Fixture {
    let root = unique_test_dir("metal-fixture");
    let metal_dir = root.join("metal");
    let kernels_dir = metal_dir.join("kernels");
    let build_dir = root.join("build").join("metal");
    fs::create_dir_all(&kernels_dir).expect("kernels directory should create");
    fs::create_dir_all(&build_dir).expect("build directory should create");

    let source_path = kernels_dir.join("phase1_dense_path.metal");
    let source_text = phase1_source_text();
    fs::write(&source_path, source_text.as_bytes()).expect("source file should write");

    let manifest_path = metal_dir.join("phase1-kernels.json");
    let mut manifest = MetalKernelManifest {
        schema_version: PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION.to_string(),
        native_target: PHASE1_METAL_NATIVE_TARGET.to_string(),
        metal_language_standard: PHASE1_METAL_LANGUAGE_STANDARD.to_string(),
        library_name: PHASE1_METAL_LIBRARY_NAME.to_string(),
        default_block_size_tokens: PHASE1_DEFAULT_BLOCK_SIZE_TOKENS,
        supported_block_size_tokens: PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS.to_vec(),
        source_file: PathBuf::from("metal/kernels/phase1_dense_path.metal"),
        toolchain_requirements: REQUIRED_TOOLCHAIN_REQUIREMENTS
            .iter()
            .map(|tool| (*tool).to_string())
            .collect(),
        build_gate: PHASE1_METAL_BUILD_GATE.to_string(),
        kernels: phase1_kernel_specs(),
    };
    if let Some(edit) = manifest_edit {
        edit(&mut manifest);
    }
    write_json_file(&manifest_path, &manifest);

    let air_path = build_dir.join("ax_phase1_dense_path.air");
    let metalar_path = build_dir.join("ax_phase1_dense_path.metalar");
    let metallib_path = build_dir.join("ax_phase1_dense_path.metallib");
    let mut outputs = MetalBuildOutputs {
        air: None,
        metalar: None,
        metallib: None,
        air_sha256: None,
        metalar_sha256: None,
        metallib_sha256: None,
    };
    let reason = match status {
        MetalBuildStatus::Compiled => {
            fs::write(&air_path, b"fake-air").expect("air file should write");
            fs::write(&metalar_path, b"fake-metalar").expect("metalar should write");
            fs::write(&metallib_path, b"fake-metallib").expect("metallib should write");
            outputs.air = Some(air_path.clone());
            outputs.metalar = Some(metalar_path.clone());
            outputs.metallib = Some(metallib_path.clone());
            outputs.air_sha256 = Some(sha256_hex(b"fake-air"));
            outputs.metalar_sha256 = Some(sha256_hex(b"fake-metalar"));
            outputs.metallib_sha256 = Some(sha256_hex(b"fake-metallib"));
            None
        }
        MetalBuildStatus::SkippedToolchainUnavailable => {
            Some("Metal toolchain is incomplete on this machine".to_string())
        }
        MetalBuildStatus::SkippedNotReady => {
            Some("AX native bring-up is not allowed on this machine without override".to_string())
        }
        MetalBuildStatus::FailedCompile => Some("command failed with exit code 1".to_string()),
        MetalBuildStatus::Unknown => None,
    };
    let build_report = MetalBuildReport {
        schema_version: PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION.to_string(),
        manifest_path: manifest_path.clone(),
        source_file: source_path.clone(),
        native_target: manifest.native_target.clone(),
        metal_language_standard: manifest.metal_language_standard.clone(),
        library_name: manifest.library_name.clone(),
        default_block_size_tokens: manifest.default_block_size_tokens,
        supported_block_size_tokens: manifest.supported_block_size_tokens.clone(),
        toolchain_requirements: manifest.toolchain_requirements.clone(),
        doctor: MetalBuildDoctorReport {
            status: match status {
                MetalBuildStatus::Compiled => "ready".to_string(),
                MetalBuildStatus::SkippedToolchainUnavailable => "not_ready".to_string(),
                MetalBuildStatus::SkippedNotReady => "not_ready".to_string(),
                MetalBuildStatus::FailedCompile => "bringup_only".to_string(),
                MetalBuildStatus::Unknown => "not_ready".to_string(),
            },
            bringup_allowed: matches!(
                status,
                MetalBuildStatus::Compiled | MetalBuildStatus::FailedCompile
            ),
            native_runtime_ready: status == MetalBuildStatus::Compiled,
            metal_toolchain_fully_available: !matches!(
                status,
                MetalBuildStatus::SkippedToolchainUnavailable
            ),
            host: MetalBuildHostReport {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
                detected_soc: Some("Apple M4 Max".to_string()),
                supported_native_runtime: true,
                unsupported_host_override_active: false,
            },
            metal_toolchain: MetalBuildToolchainReport {
                fully_available: !matches!(status, MetalBuildStatus::SkippedToolchainUnavailable),
                metal: MetalBuildToolStatus {
                    available: !matches!(status, MetalBuildStatus::SkippedToolchainUnavailable),
                    version: Some("Apple metal version 36000.4".to_string()),
                },
                metallib: MetalBuildToolStatus {
                    available: !matches!(status, MetalBuildStatus::SkippedToolchainUnavailable),
                    version: Some("Apple metallib version 36000.4".to_string()),
                },
                metal_ar: MetalBuildToolStatus {
                    available: !matches!(status, MetalBuildStatus::SkippedToolchainUnavailable),
                    version: Some("Apple metal-ar version 36000.4".to_string()),
                },
            },
        },
        kernels: manifest.kernels.clone(),
        source_sha256: sha256_hex(source_text.as_bytes()),
        outputs,
        compile_commands: if status == MetalBuildStatus::Compiled {
            vec![
                vec!["xcrun".to_string(), "metal".to_string()],
                vec!["xcrun".to_string(), "metal-ar".to_string()],
                vec!["xcrun".to_string(), "metallib".to_string()],
            ]
        } else {
            Vec::new()
        },
        status,
        reason,
    };
    write_json_file(&build_dir.join("build_report.json"), &build_report);

    Phase1Fixture { root, build_dir }
}

fn phase1_source_text() -> &'static str {
    r#"
// The parser should ignore commented kernels such as:
// kernel void commented_line_kernel() {}
/*
kernel void commented_block_kernel() {}
*/
kernel void reshape_and_cache() {}
kernel void paged_decode_attention() {}
kernel void gather_kv_cache() {}
kernel void copy_blocks() {}
kernel void swap_blocks() {}
kernel void kv_scale_update() {}
kernel void gather_embedding_rows_f32() {}
kernel void gather_embedding_rows_f16() {}
kernel void gather_embedding_rows_bf16() {}
kernel void decode_logits_projection_f32() {}
kernel void decode_logits_projection_f16() {}
kernel void decode_logits_projection_bf16() {}
kernel void decode_logits_projection_batched_f32() {}
kernel void decode_logits_projection_batched_f16() {}
kernel void decode_logits_projection_batched_bf16() {}
kernel void logits_argmax_f32() {}
kernel void logits_argmax_batched_f32() {}
kernel void sample_argmax_logprob_f32() {}
kernel void sample_argmax_logprob_batched_f32() {}
kernel void rms_norm_f32() {}
kernel void rms_norm_f16() {}
kernel void rms_norm_bf16() {}
kernel void rms_norm_batched_f32() {}
kernel void rms_norm_batched_f16() {}
kernel void rms_norm_batched_bf16() {}
kernel void ffn_gate_silu_product_f32() {}
kernel void ffn_gate_gelu_approx_product_f32() {}
kernel void apply_rope_f32() {}
kernel void expand_grouped_kv_heads_f32() {}
"#
}

fn phase1_kernel_specs() -> Vec<MetalKernelSpec> {
    vec![
        MetalKernelSpec {
            name: "reshape_and_cache".to_string(),
            tier: MetalKernelTier::Required,
            purpose: "paged KV writes".to_string(),
        },
        MetalKernelSpec {
            name: "paged_decode_attention".to_string(),
            tier: MetalKernelTier::Required,
            purpose: "decode attention".to_string(),
        },
        MetalKernelSpec {
            name: "gather_kv_cache".to_string(),
            tier: MetalKernelTier::Required,
            purpose: "KV gather".to_string(),
        },
        MetalKernelSpec {
            name: "copy_blocks".to_string(),
            tier: MetalKernelTier::Required,
            purpose: "block copy".to_string(),
        },
        MetalKernelSpec {
            name: "swap_blocks".to_string(),
            tier: MetalKernelTier::Deferred,
            purpose: "future block swap".to_string(),
        },
        MetalKernelSpec {
            name: "kv_scale_update".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "quantized KV scaling".to_string(),
        },
        MetalKernelSpec {
            name: "gather_embedding_rows_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native embedding row gather for f32 token embeddings".to_string(),
        },
        MetalKernelSpec {
            name: "gather_embedding_rows_f16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native embedding row gather for f16 token embeddings".to_string(),
        },
        MetalKernelSpec {
            name: "gather_embedding_rows_bf16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native embedding row gather for bf16 token embeddings".to_string(),
        },
        MetalKernelSpec {
            name: "decode_logits_projection_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "f32 decode logits projection".to_string(),
        },
        MetalKernelSpec {
            name: "decode_logits_projection_f16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "f16 decode logits projection".to_string(),
        },
        MetalKernelSpec {
            name: "decode_logits_projection_bf16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "bf16 decode logits projection".to_string(),
        },
        MetalKernelSpec {
            name: "decode_logits_projection_batched_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "batched f32 decode logits projection".to_string(),
        },
        MetalKernelSpec {
            name: "decode_logits_projection_batched_f16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "batched f16 decode logits projection".to_string(),
        },
        MetalKernelSpec {
            name: "decode_logits_projection_batched_bf16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "batched bf16 decode logits projection".to_string(),
        },
        MetalKernelSpec {
            name: "logits_argmax_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native top-1 logits scan".to_string(),
        },
        MetalKernelSpec {
            name: "logits_argmax_batched_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native batched top-1 logits scan".to_string(),
        },
        MetalKernelSpec {
            name: "sample_argmax_logprob_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native deterministic top-1 sampling with logprob".to_string(),
        },
        MetalKernelSpec {
            name: "sample_argmax_logprob_batched_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native batched deterministic top-1 sampling with logprob".to_string(),
        },
        MetalKernelSpec {
            name: "rms_norm_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native RMSNorm".to_string(),
        },
        MetalKernelSpec {
            name: "rms_norm_f16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native f16 RMSNorm".to_string(),
        },
        MetalKernelSpec {
            name: "rms_norm_bf16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native bf16 RMSNorm".to_string(),
        },
        MetalKernelSpec {
            name: "rms_norm_batched_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native batched f32 RMSNorm".to_string(),
        },
        MetalKernelSpec {
            name: "rms_norm_batched_f16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native batched f16 RMSNorm".to_string(),
        },
        MetalKernelSpec {
            name: "rms_norm_batched_bf16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native batched bf16 RMSNorm".to_string(),
        },
        MetalKernelSpec {
            name: "ffn_gate_silu_product_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native SiLU gate-up activation product".to_string(),
        },
        MetalKernelSpec {
            name: "ffn_gate_gelu_approx_product_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native GELU-approx gate-up activation product".to_string(),
        },
        MetalKernelSpec {
            name: "apply_rope_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native Q/K rotary position embedding application".to_string(),
        },
        MetalKernelSpec {
            name: "expand_grouped_kv_heads_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native grouped-KV expansion into query-head layout".to_string(),
        },
    ]
}

fn sample_runner_input() -> RunnerInput {
    RunnerInput {
        block_size_tokens: 16,
        execution_batch: ExecutionBatch {
            step_id: StepId(3),
            model_id: "qwen3_dense".into(),
            execution_plan_ref: Some("phase1.qwen3_dense.dense_prefill".into()),
            items: vec![
                ExecutionItem {
                    request_id: RequestId(7),
                    mode: ExecutionMode::Prefill,
                    input_token_slice: vec![1, 2, 3],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: 0,
                        end_exclusive: 3,
                    },
                    scheduled_token_count: 3,
                    block_table_ref: RequestId(7),
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                },
                ExecutionItem {
                    request_id: RequestId(9),
                    mode: ExecutionMode::Decode,
                    input_token_slice: vec![4],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: 3,
                        end_exclusive: 4,
                    },
                    scheduled_token_count: 1,
                    block_table_ref: RequestId(9),
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                },
            ],
            total_scheduled_tokens: 4,
            route_metadata: RouteMetadata {
                execution_plan: Some("phase1.qwen3_dense.dense_prefill".into()),
                attention_route: Some("qwen3_dense_prefill".into()),
                kv_mode: Some("paged_metadata".into()),
                prefix_cache_path: Some("metadata_lookup".into()),
                barrier_mode: Some("serial".into()),
                crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
            },
        },
        block_tables: vec![
            crate::runner::ResolvedBlockTable {
                request_id: RequestId(7),
                block_table: BlockTableView {
                    cache_group_id: CacheGroupId(1),
                    block_ids: vec![BlockId(0), BlockId(1)],
                },
            },
            crate::runner::ResolvedBlockTable {
                request_id: RequestId(9),
                block_table: BlockTableView {
                    cache_group_id: CacheGroupId(1),
                    block_ids: vec![BlockId(2)],
                },
            },
        ],
    }
}

fn sample_decode_only_runner_input() -> RunnerInput {
    RunnerInput {
        block_size_tokens: 16,
        execution_batch: ExecutionBatch {
            step_id: StepId(4),
            model_id: "qwen3_dense".into(),
            execution_plan_ref: Some("phase1.qwen3_dense.decode_only".into()),
            items: vec![
                ExecutionItem {
                    request_id: RequestId(9),
                    mode: ExecutionMode::Decode,
                    input_token_slice: vec![4],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: 3,
                        end_exclusive: 4,
                    },
                    scheduled_token_count: 1,
                    block_table_ref: RequestId(9),
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                },
                ExecutionItem {
                    request_id: RequestId(11),
                    mode: ExecutionMode::Decode,
                    input_token_slice: vec![8],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: 5,
                        end_exclusive: 6,
                    },
                    scheduled_token_count: 1,
                    block_table_ref: RequestId(11),
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                },
            ],
            total_scheduled_tokens: 2,
            route_metadata: RouteMetadata {
                execution_plan: Some("phase1.qwen3_dense.decode_only".into()),
                attention_route: Some("qwen3_dense_decode".into()),
                kv_mode: Some("paged_metadata".into()),
                prefix_cache_path: Some("metadata_lookup".into()),
                barrier_mode: Some("serial".into()),
                crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
            },
        },
        block_tables: vec![
            crate::runner::ResolvedBlockTable {
                request_id: RequestId(9),
                block_table: BlockTableView {
                    cache_group_id: CacheGroupId(1),
                    block_ids: vec![BlockId(2)],
                },
            },
            crate::runner::ResolvedBlockTable {
                request_id: RequestId(11),
                block_table: BlockTableView {
                    cache_group_id: CacheGroupId(1),
                    block_ids: vec![BlockId(3)],
                },
            },
        ],
    }
}

#[cfg(target_os = "macos")]
fn compiled_repo_metal_build_dir() -> Option<PathBuf> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .parent()?
        .to_path_buf();
    let build_dir = repo_root.join("build/metal");
    MetalKernelAssets::from_build_dir(&build_dir).ok()?;
    Some(build_dir)
}

fn sample_prefill_only_runner_input() -> RunnerInput {
    RunnerInput {
        block_size_tokens: 16,
        execution_batch: ExecutionBatch {
            step_id: StepId(5),
            model_id: "qwen3_dense".into(),
            execution_plan_ref: Some("phase1.qwen3_dense.prefill_only".into()),
            items: vec![ExecutionItem {
                request_id: RequestId(17),
                mode: ExecutionMode::Prefill,
                input_token_slice: vec![1, 2, 3, 4],
                reused_prefix_token_slice: Vec::new(),
                position_range: PositionRange {
                    start: 0,
                    end_exclusive: 4,
                },
                scheduled_token_count: 4,
                block_table_ref: RequestId(17),
                prefix_tokens_reused: 0,
                prefix_blocks_reused: 0,
            }],
            total_scheduled_tokens: 4,
            route_metadata: RouteMetadata {
                execution_plan: Some("phase1.qwen3_dense.prefill_only".into()),
                attention_route: Some("qwen3_dense_prefill".into()),
                kv_mode: Some("paged_metadata".into()),
                prefix_cache_path: Some("metadata_lookup".into()),
                barrier_mode: Some("serial".into()),
                crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
            },
        },
        block_tables: vec![crate::runner::ResolvedBlockTable {
            request_id: RequestId(17),
            block_table: BlockTableView {
                cache_group_id: CacheGroupId(1),
                block_ids: vec![BlockId(0)],
            },
        }],
    }
}

fn sample_decode_continuation_runner_input() -> RunnerInput {
    RunnerInput {
        block_size_tokens: 16,
        execution_batch: ExecutionBatch {
            step_id: StepId(6),
            model_id: "qwen3_dense".into(),
            execution_plan_ref: Some("phase1.qwen3_dense.decode_continuation".into()),
            items: vec![ExecutionItem {
                request_id: RequestId(17),
                mode: ExecutionMode::Decode,
                input_token_slice: vec![4],
                reused_prefix_token_slice: Vec::new(),
                position_range: PositionRange {
                    start: 4,
                    end_exclusive: 5,
                },
                scheduled_token_count: 1,
                block_table_ref: RequestId(17),
                prefix_tokens_reused: 0,
                prefix_blocks_reused: 0,
            }],
            total_scheduled_tokens: 1,
            route_metadata: RouteMetadata {
                execution_plan: Some("phase1.qwen3_dense.decode_continuation".into()),
                attention_route: Some("qwen3_dense_decode".into()),
                kv_mode: Some("paged_metadata".into()),
                prefix_cache_path: Some("metadata_lookup".into()),
                barrier_mode: Some("serial".into()),
                crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
            },
        },
        block_tables: vec![crate::runner::ResolvedBlockTable {
            request_id: RequestId(17),
            block_table: BlockTableView {
                cache_group_id: CacheGroupId(1),
                block_ids: vec![BlockId(0)],
            },
        }],
    }
}

#[test]
fn metal_assets_reject_unknown_build_status() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Unknown, None);

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject unknown build status");
    let MetalRuntimeError::InvalidBuildReport { message } = error else {
        panic!("expected invalid build report error");
    };
    assert!(message.contains("status unknown is not allowed"));

    fixture.cleanup();
}

#[test]
fn successful_metal_runner_output_matches_expected_decode_sampling_contract() {
    let input = sample_runner_input();

    let output = successful_runner_output_from_input(&input);

    assert_eq!(output.execution_status, ExecutionStatus::Success);
    assert_eq!(output.request_updates.len(), 2);
    assert_eq!(output.request_updates[0].tokens_executed, 3);
    assert_eq!(output.request_updates[1].tokens_executed, 1);
    assert_eq!(output.logits_handles, vec![RequestId(9)]);
    assert!(output.logits_outputs.is_empty());
    assert_eq!(output.kv_write_summary.tokens_written, 4);
    assert_eq!(output.kv_write_summary.blocks_touched, 3);
    assert_eq!(output.route_metadata, input.execution_batch.route_metadata);
}

#[test]
fn metal_assets_reject_unknown_build_status_before_manifest_resolution() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Unknown, None);
    let manifest_path = fixture.root.join("metal/phase1-kernels.json");
    fs::remove_file(&manifest_path).expect("manifest file should be removable");

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject unknown status before reading manifest");
    let MetalRuntimeError::InvalidBuildReport { message } = error else {
        panic!("expected invalid build report error");
    };
    assert!(message.contains("status unknown is not allowed"));

    fixture.cleanup();
}

#[test]
fn metal_assets_reject_source_kernel_manifest_drift() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);
    let source_path = fixture.root.join("metal/kernels/phase1_dense_path.metal");
    let drifted_source = r#"
kernel void reshape_and_cache() {}
kernel void paged_decode_attention() {}
kernel void gather_kv_cache() {}
kernel void copy_blocks() {}
kernel void swap_blocks() {}
"#;
    fs::write(&source_path, drifted_source).expect("source file should be updated");

    let build_report_path = fixture.build_dir.join("build_report.json");
    let mut build_report: MetalBuildReport =
        read_json_file(&build_report_path).expect("build report should load");
    build_report.source_sha256 = sha256_hex(drifted_source.as_bytes());
    write_json_file(&build_report_path, &build_report);

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject source/manifest drift");
    let MetalRuntimeError::InvalidBuildReport { message } = error else {
        panic!("expected invalid build report error");
    };
    assert!(message.contains("source kernel declarations do not match manifest"));
    assert!(message.contains("kv_scale_update"));

    fixture.cleanup();
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) {
    let json = serde_json::to_vec_pretty(value).expect("json should serialize");
    fs::write(path, json).expect("json file should write");
}

fn sample_build_doctor(
    metal_toolchain_fully_available: bool,
    bringup_allowed: bool,
) -> MetalBuildDoctorReport {
    MetalBuildDoctorReport {
        status: if metal_toolchain_fully_available && bringup_allowed {
            "ready".to_string()
        } else if bringup_allowed {
            "bringup_only".to_string()
        } else {
            "not_ready".to_string()
        },
        bringup_allowed,
        native_runtime_ready: metal_toolchain_fully_available && bringup_allowed,
        metal_toolchain_fully_available,
        host: MetalBuildHostReport {
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
            detected_soc: Some("Apple M4 Max".to_string()),
            supported_native_runtime: true,
            unsupported_host_override_active: false,
        },
        metal_toolchain: MetalBuildToolchainReport {
            fully_available: metal_toolchain_fully_available,
            metal: MetalBuildToolStatus {
                available: metal_toolchain_fully_available,
                version: Some("Apple metal version 36000.4".to_string()),
            },
            metallib: MetalBuildToolStatus {
                available: metal_toolchain_fully_available,
                version: Some("Apple metallib version 36000.4".to_string()),
            },
            metal_ar: MetalBuildToolStatus {
                available: metal_toolchain_fully_available,
                version: Some("Apple metal-ar version 36000.4".to_string()),
            },
        },
    }
}

fn fake_xcrun_script() -> &'static str {
    r#"#!/bin/sh
set -eu

while [ "$#" -gt 0 ]; do
  if [ "$1" = "--sdk" ]; then
    shift 2
    continue
  fi
  tool="$1"
  shift
  break
done

case "${tool:-}" in
  metal)
    out=""
    src=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -o)
          out="$2"
          shift 2
          ;;
        -c|-O3|-Wall|-Wextra)
          shift
          ;;
        -std=*)
          shift
          ;;
        *)
          src="$1"
          shift
          ;;
      esac
    done
    cp "$src" "$out"
    ;;
  metal-ar)
    archive=""
    input=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -q)
          shift
          ;;
        *)
          if [ -z "$archive" ]; then
            archive="$1"
          else
            input="$1"
          fi
          shift
          ;;
      esac
    done
    cp "$input" "$archive"
    ;;
  metallib)
    input=""
    out=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -o)
          out="$2"
          shift 2
          ;;
        *)
          input="$1"
          shift
          ;;
      esac
    done
    cp "$input" "$out"
    ;;
  *)
    echo "unexpected tool" >&2
    exit 64
    ;;
esac
"#
}

fn fake_failing_xcrun_script() -> &'static str {
    r#"#!/bin/sh
set -eu
echo "unexpected xcrun invocation" >&2
exit 99
"#
}

fn env_lock() -> &'static Mutex<()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}

fn prepend_to_path(dir: &Path, existing: Option<&OsString>) -> OsString {
    let mut paths = vec![dir.to_path_buf()];
    if let Some(existing) = existing {
        paths.extend(std::env::split_paths(existing));
    }
    std::env::join_paths(paths).expect("PATH should join")
}

// -------------------------------------------------------------------------
// Tier 1: apply_rms_norm_with_weights_in_place (no Metal device required)
// -------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[test]
fn rms_norm_with_weights_in_place_normalizes_and_scales_by_weights() {
    let mut values = vec![3.0_f32, 4.0];
    let weights = vec![2.0_f32, 0.5];
    let epsilon = 1e-6_f32;

    apply_rms_norm_with_weights_in_place(&mut values, &weights, epsilon, 0.0)
        .expect("rms norm should succeed");

    // mean_square = (9 + 16) / 2 = 12.5
    let mean_square = (9.0_f32 + 16.0) / 2.0;
    let denom = (mean_square + epsilon).sqrt();
    let expected = vec![(3.0 / denom) * 2.0, (4.0 / denom) * 0.5];
    assert_f32_slice_close(&values, &expected, 1e-5);
}

#[cfg(target_os = "macos")]
#[test]
fn rms_norm_with_weights_in_place_applies_weight_offset() {
    // weight_offset=1.0 is the Gemma style: effective_weight = weight + 1.0
    let mut values = vec![1.0_f32, 0.0];
    let weights = vec![0.0_f32, 0.0];
    let epsilon = 1e-6_f32;

    apply_rms_norm_with_weights_in_place(&mut values, &weights, epsilon, 1.0)
        .expect("rms norm with weight_offset should succeed");

    // effective weight = 0 + 1.0 = 1.0 for all; mean_square = 0.5
    let mean_square = 0.5_f32;
    let denom = (mean_square + epsilon).sqrt();
    let expected = vec![1.0 / denom, 0.0 / denom];
    assert_f32_slice_close(&values, &expected, 1e-5);
}

#[cfg(target_os = "macos")]
#[test]
fn rms_norm_with_weights_in_place_rejects_mismatched_lengths() {
    let mut values = vec![1.0_f32, 2.0, 3.0];
    let weights = vec![1.0_f32, 1.0]; // wrong length

    assert!(
        apply_rms_norm_with_weights_in_place(&mut values, &weights, 1e-6, 0.0).is_none(),
        "mismatched lengths should return None"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rms_norm_with_weights_in_place_rejects_nonpositive_epsilon() {
    let mut values = vec![1.0_f32, 2.0];
    let weights = vec![1.0_f32, 1.0];

    assert!(
        apply_rms_norm_with_weights_in_place(&mut values, &weights, 0.0, 0.0).is_none(),
        "zero epsilon should return None"
    );
    assert!(
        apply_rms_norm_with_weights_in_place(&mut values, &weights, -1e-6, 0.0).is_none(),
        "negative epsilon should return None"
    );
}

// -------------------------------------------------------------------------
// Tier 1: expand_grouped_kv_heads_cpu (no Metal device required)
// -------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[test]
fn expand_grouped_kv_heads_cpu_repeats_kv_head_to_fill_query_heads() {
    // 1 kv head, 4 q heads, head_dim=3 → each q head gets the same kv head
    let kv = vec![1.0_f32, 2.0, 3.0];
    let expanded =
        expand_grouped_kv_heads_cpu(&kv, 4, 1, 3).expect("single kv head expansion should succeed");

    assert_eq!(
        expanded,
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[cfg(target_os = "macos")]
#[test]
fn expand_grouped_kv_heads_cpu_is_identity_when_q_heads_equals_kv_heads() {
    let kv = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expanded =
        expand_grouped_kv_heads_cpu(&kv, 2, 2, 3).expect("equal-head expansion should succeed");

    assert_eq!(expanded, kv);
}

#[cfg(target_os = "macos")]
#[test]
fn expand_grouped_kv_heads_cpu_handles_multiple_kv_heads_correctly() {
    // 2 kv heads, 4 q heads, head_dim=2
    // kv: [head0: 1,2], [head1: 3,4]
    // expected q: [1,2, 1,2, 3,4, 3,4]
    let kv = vec![1.0_f32, 2.0, 3.0, 4.0];
    let expanded =
        expand_grouped_kv_heads_cpu(&kv, 4, 2, 2).expect("multi kv head expansion should succeed");

    assert_eq!(expanded, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
}

#[cfg(target_os = "macos")]
#[test]
fn expand_grouped_kv_heads_cpu_rejects_q_heads_not_divisible_by_kv_heads() {
    let kv = vec![1.0_f32, 2.0, 3.0, 4.0];
    assert!(
        expand_grouped_kv_heads_cpu(&kv, 3, 2, 2).is_none(),
        "3 q_heads / 2 kv_heads is not integer, should return None"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn expand_grouped_kv_heads_cpu_rejects_zero_head_dim() {
    let kv = vec![1.0_f32, 2.0];
    assert!(
        expand_grouped_kv_heads_cpu(&kv, 2, 1, 0).is_none(),
        "zero head_dim should return None"
    );
}

// -------------------------------------------------------------------------
// Tier 2: apply_model_stage_rope_cpu (needs artifacts on disk, no GPU ops)
// -------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[test]
fn rope_cpu_is_identity_at_position_zero_for_qwen() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");
    let stage_dims = ModelStageDims {
        input_dim: 8,
        q_heads: 2,
        kv_heads: 2,
        head_dim: 4,
    };
    let original_query = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let original_key = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let mut query = original_query.clone();
    let mut key = original_key.clone();

    apply_model_stage_rope_cpu(&artifacts, &mut query, &mut key, 0.0, stage_dims);

    // At position 0: theta = 0 for all pairs → cos(0)=1, sin(0)=0 → identity
    assert_f32_slice_close(&query, &original_query, 1e-5);
    assert_f32_slice_close(&key, &original_key, 1e-5);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn rope_cpu_result_matches_neox_rotate_reference_for_qwen() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");
    let stage_dims = ModelStageDims {
        input_dim: 8,
        q_heads: 2,
        kv_heads: 2,
        head_dim: 4,
    };
    let original_query = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let original_key = vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
    let mut query = original_query.clone();
    let mut key = original_key.clone();

    apply_model_stage_rope_cpu(&artifacts, &mut query, &mut key, 2.0, stage_dims);

    // Qwen uses NeoX style: each head independently rotated
    let expected_query: Vec<f32> = neox_rotate_reference(&original_query[..4], 2.0)
        .into_iter()
        .chain(neox_rotate_reference(&original_query[4..], 2.0))
        .collect();
    let expected_key: Vec<f32> = neox_rotate_reference(&original_key[..4], 2.0)
        .into_iter()
        .chain(neox_rotate_reference(&original_key[4..], 2.0))
        .collect();

    assert_f32_slice_close(&query, &expected_query, 1e-5);
    assert_f32_slice_close(&key, &expected_key, 1e-5);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn rope_cpu_result_matches_interleaved_rotate_reference_for_gemma() {
    let model_dir = write_gemma_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("gemma artifacts should load");
    let stage_dims = ModelStageDims {
        input_dim: 8,
        q_heads: 2,
        kv_heads: 2,
        head_dim: 4,
    };
    let original_query = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let original_key = vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
    let mut query = original_query.clone();
    let mut key = original_key.clone();

    apply_model_stage_rope_cpu(&artifacts, &mut query, &mut key, 1.0, stage_dims);

    // Gemma uses Interleaved style: consecutive element-pairs rotated
    let expected_query: Vec<f32> = interleaved_rotate_reference(&original_query[..4], 1.0)
        .into_iter()
        .chain(interleaved_rotate_reference(&original_query[4..], 1.0))
        .collect();
    let expected_key: Vec<f32> = interleaved_rotate_reference(&original_key[..4], 1.0)
        .into_iter()
        .chain(interleaved_rotate_reference(&original_key[4..], 1.0))
        .collect();

    assert_f32_slice_close(&query, &expected_query, 1e-5);
    assert_f32_slice_close(&key, &expected_key, 1e-5);

    let _ = fs::remove_dir_all(model_dir);
}

// -------------------------------------------------------------------------
// Tier 2: *_with_path fallback wiring — verify CPU is used when bringup=None
// -------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[test]
fn rope_with_path_matches_rope_cpu_when_bringup_is_none() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");
    let stage_dims = ModelStageDims {
        input_dim: 8,
        q_heads: 2,
        kv_heads: 2,
        head_dim: 4,
    };
    let input_query = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_key = vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];

    let mut cpu_query = input_query.clone();
    let mut cpu_key = input_key.clone();
    apply_model_stage_rope_cpu(&artifacts, &mut cpu_query, &mut cpu_key, 3.0, stage_dims);

    let mut with_path_query = input_query.clone();
    let mut with_path_key = input_key.clone();
    apply_model_stage_rope_with_path(
        &artifacts,
        &mut with_path_query,
        &mut with_path_key,
        3.0,
        stage_dims,
        None, // bringup=None → forces CPU
    );

    assert_f32_slice_close(&with_path_query, &cpu_query, 1e-6);
    assert_f32_slice_close(&with_path_key, &cpu_key, 1e-6);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn batched_rope_with_path_matches_per_row_reference_when_bringup_is_none() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");
    let stage_dims = ModelStageDims {
        input_dim: 8,
        q_heads: 2,
        kv_heads: 2,
        head_dim: 4,
    };
    let mut batched_query = vec![
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![0.5_f32, -1.0, 2.0, -0.5, 3.0, 1.5, -2.0, 4.0],
    ];
    let mut batched_key = vec![
        vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        vec![-0.25_f32, 0.75, -1.5, 2.5, 1.0, -3.0, 0.5, 4.5],
    ];
    let positions = vec![1_u32, 3_u32];

    let mut reference_query = batched_query.clone();
    let mut reference_key = batched_key.clone();
    for ((query, key), position) in reference_query
        .iter_mut()
        .zip(reference_key.iter_mut())
        .zip(positions.iter().copied())
    {
        apply_model_stage_rope_with_path(&artifacts, query, key, position as f32, stage_dims, None);
    }

    apply_batched_model_stage_rope_with_path(
        &artifacts,
        &mut batched_query,
        &mut batched_key,
        &positions,
        stage_dims,
        None,
    )
    .expect("batched rope path should succeed");

    assert_eq!(batched_query, reference_query);
    assert_eq!(batched_key, reference_key);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn gate_up_product_with_path_reports_cpu_path_and_matches_cpu_when_bringup_is_none() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");

    let original_gate = vec![1.0_f32, -1.0, 2.0, 0.5];
    let up = vec![1.0_f32, 2.0, 0.5, -1.0];

    let mut cpu_gate = original_gate.clone();
    apply_model_gate_up_product(&artifacts, &mut cpu_gate, &up);

    let mut with_path_gate = original_gate.clone();
    let used_native =
        apply_model_gate_up_product_with_path(&artifacts, &mut with_path_gate, &up, None)
            .expect("gate_up_product_with_path should succeed");

    assert!(!used_native, "should report CPU path when bringup is None");
    assert_f32_slice_close(&with_path_gate, &cpu_gate, 1e-6);

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn expand_grouped_kv_heads_with_path_falls_back_to_cpu_when_bringup_is_none() {
    // 2 kv heads, 4 q heads, head_dim=2
    let kv = vec![1.0_f32, 2.0, 3.0, 4.0];
    let with_path = expand_grouped_kv_heads_with_path(&kv, 4, 2, 2, None)
        .expect("expand_grouped_kv_heads_with_path should succeed");
    let cpu = expand_grouped_kv_heads_cpu(&kv, 4, 2, 2)
        .expect("expand_grouped_kv_heads_cpu should succeed");

    assert_eq!(with_path, cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn expand_batched_grouped_kv_heads_with_path_matches_row_wise_cpu_reference() {
    let rows = vec![
        vec![1.0_f32, 2.0, 3.0, 4.0],
        vec![5.0_f32, 6.0, 7.0, 8.0],
        vec![9.0_f32, 10.0, 11.0, 12.0],
    ];

    let with_path = expand_batched_grouped_kv_heads_with_path(&rows, 4, 2, 2, None)
        .expect("expand_batched_grouped_kv_heads_with_path should succeed");
    let cpu_reference: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| {
            expand_grouped_kv_heads_cpu(row, 4, 2, 2)
                .expect("expand_grouped_kv_heads_cpu should succeed")
        })
        .collect();

    assert_eq!(with_path, cpu_reference);
}

#[cfg(target_os = "macos")]
#[test]
fn project_batched_attention_qkv_with_dims_matches_per_row_reference() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("buffers should bind");
    let layer = &bindings.layers[0];
    let attention_norm = buffers
        .binding_for(&layer.attention_norm)
        .expect("attention norm binding should resolve");
    let stage_dims =
        resolved_model_stage_dims_for_input_width(&artifacts, layer, attention_norm, &buffers, 8)
            .expect("stage dims should resolve");
    let input_rows = vec![
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![0.5_f32, -1.0, 2.0, -0.5, 3.0, 1.5, -2.0, 4.0],
        vec![3.0_f32, 0.0, -1.0, 2.0, -4.0, 1.0, 0.5, -0.25],
    ];

    let (batched_query, batched_key, batched_value, batched_tally) =
        project_batched_attention_qkv_with_dims_and_tally(
            &artifacts,
            layer
                .attention_qkv
                .as_ref()
                .expect("attention_qkv should exist"),
            &buffers,
            &input_rows,
            8,
            stage_dims,
            None,
        )
        .expect("batched attention qkv projection should succeed");

    let mut reference_query = Vec::with_capacity(input_rows.len());
    let mut reference_key = Vec::with_capacity(input_rows.len());
    let mut reference_value = Vec::with_capacity(input_rows.len());
    let mut reference_tally = PrefixAttentionExecutionTally::default();
    for input in &input_rows {
        let (query, key, value, tally) = project_attention_qkv_with_dims_and_tally(
            &artifacts,
            layer
                .attention_qkv
                .as_ref()
                .expect("attention_qkv should exist"),
            &buffers,
            input,
            stage_dims,
            None,
        )
        .expect("row-wise attention qkv projection should succeed");
        reference_query.push(query);
        reference_key.push(key);
        reference_value.push(value);
        reference_tally = reference_tally.merge(tally);
    }

    assert_eq!(batched_query, reference_query);
    assert_eq!(batched_key, reference_key);
    assert_eq!(batched_value, reference_value);
    assert_eq!(
        batched_tally.cpu_projection_row_count(),
        reference_tally.cpu_projection_row_count()
    );
    assert_eq!(
        batched_tally.native_projection_row_count(),
        reference_tally.native_projection_row_count()
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn read_shared_buffer_prefix_reads_large_buffers_in_texture_sized_chunks() {
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let element_count = SHARED_BUFFER_READBACK_MAX_TEXTURE_WIDTH as usize + 19;
    let values = (0..element_count)
        .map(|index| index as f32 * 0.5 - 7.0)
        .collect::<Vec<_>>();
    let buffer = new_shared_buffer_with_data(&device, &values);

    let restored = read_shared_buffer_prefix(&buffer, saturating_usize_to_u32(values.len()));

    assert_eq!(restored.len(), values.len());
    assert_f32_slice_close(&restored, &values, 1e-6);
}

#[cfg(target_os = "macos")]
#[test]
fn read_shared_u32_buffer_prefix_reads_large_buffers_in_texture_sized_chunks() {
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let element_count = SHARED_BUFFER_READBACK_MAX_TEXTURE_WIDTH as usize + 23;
    let values = (0..element_count)
        .map(|index| (index as u32).wrapping_mul(17))
        .collect::<Vec<_>>();
    let buffer = new_shared_buffer_with_data(&device, &values);

    let restored = read_shared_u32_buffer_prefix(&buffer, saturating_usize_to_u32(values.len()));

    assert_eq!(restored, values);
}

// -------------------------------------------------------------------------
// Tier 3: per_head_rms_norm_with_tally (needs Metal device for buffer)
// -------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[test]
fn per_head_rms_norm_with_tally_result_matches_per_head_reference() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("buffers should bind");
    let attention_norm = buffers
        .binding_for(&bindings.layers[0].attention_norm)
        .expect("attention norm binding should resolve");
    let head_count = 2_usize;
    let head_dim = 4_usize;
    let epsilon = native_model_rms_norm_epsilon(&artifacts);
    let weight_offset = native_model_rms_norm_weight_offset(&artifacts);
    let weights =
        tensor_prefix_f32(attention_norm, head_dim).expect("attention norm weights should exist");

    let original: Vec<f32> = (1..=8).map(|i| i as f32).collect();

    // Reference: apply rms norm to each head slice independently
    let mut expected = original.clone();
    for head in expected.chunks_exact_mut(head_dim) {
        apply_rms_norm_with_weights_in_place(head, &weights, epsilon, weight_offset)
            .expect("per-head reference norm should succeed");
    }

    let mut actual = original.clone();
    let tally = apply_per_head_rms_norm_with_binding_in_place_with_tally(
        &mut actual,
        head_count,
        head_dim,
        attention_norm,
        epsilon,
        weight_offset,
        None, // bringup=None → CPU path
    )
    .expect("per_head_rms_norm_with_tally should succeed");

    assert_f32_slice_close(&actual, &expected, 1e-6);
    assert_eq!(tally.native_rms_norm_element_count(), 0);
    assert_eq!(
        tally.cpu_rms_norm_element_count(),
        (head_count * head_dim) as u32
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn per_head_rms_norm_with_tally_normalizes_heads_independently_not_jointly() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("buffers should bind");
    let attention_norm = buffers
        .binding_for(&bindings.layers[0].attention_norm)
        .expect("attention norm binding should resolve");
    // Use uniform weights=1.0 so we can reason about the norms directly
    let head_count = 2_usize;
    let head_dim = 4_usize;
    let epsilon = 1e-6_f32;

    // head 0: [1, 1, 1, 1] — small magnitude
    // head 1: [10, 10, 10, 10] — large magnitude
    // If normalized jointly the ratio would be 1:1; if independently each head
    // normalizes to roughly unit RMS
    let mut values = vec![1.0_f32, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0];
    apply_per_head_rms_norm_with_binding_in_place_with_tally(
        &mut values,
        head_count,
        head_dim,
        attention_norm,
        epsilon,
        0.0,
        None,
    )
    .expect("per_head_rms_norm should succeed");

    // After independent per-head normalization both heads should have ~equal RMS
    // (since uniform-magnitude heads both normalize to ~1/sqrt(head_dim))
    let rms0 = (values[..head_dim].iter().map(|v| v * v).sum::<f32>() / head_dim as f32).sqrt();
    let rms1 = (values[head_dim..].iter().map(|v| v * v).sum::<f32>() / head_dim as f32).sqrt();
    // Both heads should have RMS ≈ weights / sqrt(head_dim), not wildly different
    assert!(
        (rms0 - rms1).abs() < 0.01,
        "independently normalized heads should have similar RMS; got rms0={rms0}, rms1={rms1}"
    );

    let _ = fs::remove_dir_all(model_dir);
}

// -------------------------------------------------------------------------
// Tier 3: project_decode_logits_cpu (needs Metal device for buffer)
// -------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[test]
fn project_decode_logits_cpu_returns_correct_argmax_token() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("buffers should bind");
    // The fixture lm_head is a zero matrix ([5, 8]) so all dot products are 0.
    // Token 0 (first max) is chosen as argmax.
    let lm_head_binding = bindings.lm_head.as_ref().expect("lm_head should exist");
    let lm_head = buffers
        .binding_for(lm_head_binding)
        .expect("lm_head buffer should bind");
    let hidden = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let (logits, token_id) = project_decode_logits_cpu(lm_head, 8, &hidden)
        .expect("project_decode_logits_cpu should succeed with zero lm_head");

    assert_eq!(logits.len(), 5, "vocab_size=5 → 5 logits");
    assert!(
        logits.iter().all(|v| v.abs() < 1e-6),
        "zero lm_head → all logits should be ~0"
    );
    assert_eq!(token_id, 0, "argmax of all-zero logits should be token 0");

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn project_decode_logits_cpu_selects_highest_scoring_token() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("buffers should bind");
    // The fixture attention Q matrix is an identity matrix ([8, 8]).
    // Use it as a stand-in projection to get non-zero logits.
    // With identity weights, logit[i] = hidden[i].
    // If we set hidden = [0, 0, 0, 0, 0, 0, 0, 99], logit for row i = hidden[i].
    // But vocab is only 5 rows, so logits = [hidden·row0, ..., hidden·row4].
    // Row 0 of identity = [1,0,0,...], so logit[0] = hidden[0] = 0.
    // ...hidden[7] is not reached because vocab_size=5.
    // To get a clear winner, set hidden[2]=10 so logit[2]=10 is the max.
    let layer = &bindings.layers[0];
    let attn_q_binding = match layer
        .attention_qkv
        .as_ref()
        .expect("attention_qkv should exist")
    {
        MetalAttentionQkvBindings::Split { q, .. } => q,
        MetalAttentionQkvBindings::Packed(_) => return, // fixture uses split
    };
    let attn_q = buffers
        .binding_for(attn_q_binding)
        .expect("attn_q buffer should bind");
    // hidden[2] = 10 → row 2 of identity → logit[2] = 10, all others ≤ 5
    let hidden = vec![1.0_f32, 2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let (logits, token_id) = project_decode_logits_cpu(attn_q, 8, &hidden)
        .expect("project_decode_logits_cpu should succeed");

    // attn_q has 8 rows (not 5), so logits.len() = 8
    // logit[2] = dot([0,0,1,0,...], hidden) = 10.0 → argmax
    assert!(logits.len() >= 3, "should have at least 3 logits");
    assert_eq!(token_id, 2, "token 2 should win with logit 10.0");
    assert_f32_slice_close(&logits[..3], &[1.0, 2.0, 10.0], 1e-5);

    let _ = fs::remove_dir_all(model_dir);
}

// -------------------------------------------------------------------------
// Tier 3: initial_model_hidden_states_cpu (needs Metal device + workload)
// -------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[test]
fn initial_model_hidden_states_cpu_returns_one_vector_per_scheduled_token() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("buffers should bind");
    let token_embedding = buffers
        .binding_for(&bindings.token_embedding)
        .expect("token_embedding binding should resolve");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve from sample runner input");

    let hidden_states = initial_model_hidden_states_cpu(token_embedding, &workload, 8, 1.0)
        .expect("initial_model_hidden_states_cpu should succeed");

    let expected_count = workload.scheduled_token_ids.len();
    assert_eq!(
        hidden_states.len(),
        expected_count,
        "should produce one hidden state per scheduled token"
    );
    assert!(
        hidden_states.iter().all(|h| h.len() == 8),
        "each hidden state should have hidden_dim=8 elements"
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn initial_model_hidden_states_cpu_gathers_correct_embedding_rows() {
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("buffers should bind");
    let token_embedding = buffers
        .binding_for(&bindings.token_embedding)
        .expect("token_embedding binding should resolve");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    // The fixture embedding rows are:
    //   row 0: [0,0,0,0,0,0,0,0]
    //   row 1: [1,2,3,4,5,6,7,8]
    //   row 2: [2,3,4,5,6,7,8,9]
    //   row 3: [3,4,5,6,7,8,9,10]
    //   row 4: [4,5,6,7,8,9,10,11]
    // sample_runner_input schedules token IDs [1,2,3,4]
    let hidden_states = initial_model_hidden_states_cpu(token_embedding, &workload, 8, 1.0)
        .expect("initial_model_hidden_states_cpu should succeed");

    assert_f32_slice_close(
        &hidden_states[0],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        1e-6,
    );
    assert_f32_slice_close(
        &hidden_states[1],
        &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        1e-6,
    );

    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(target_os = "macos")]
#[test]
fn initial_model_hidden_states_cpu_applies_embedding_scale() {
    let model_dir = write_gemma_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("gemma artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("buffers should bind");
    let token_embedding = buffers
        .binding_for(&bindings.token_embedding)
        .expect("token_embedding binding should resolve");
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");

    let scale_1x = initial_model_hidden_states_cpu(token_embedding, &workload, 8, 1.0)
        .expect("scale 1x should succeed");
    let scale_2x = initial_model_hidden_states_cpu(token_embedding, &workload, 8, 2.0)
        .expect("scale 2x should succeed");

    for (unscaled, scaled) in scale_1x.iter().zip(scale_2x.iter()) {
        let expected: Vec<f32> = unscaled.iter().map(|v| v * 2.0).collect();
        assert_f32_slice_close(scaled, &expected, 1e-6);
    }

    let _ = fs::remove_dir_all(model_dir);
}

// -------------------------------------------------------------------------
// Tier 4: GPU vs CPU correctness — each test compiles real Metal kernels via
// xcrun (skipped gracefully when xcrun / Metal toolchain is unavailable) and
// then asserts that the GPU path produces the same numerical result as the
// corresponding CPU reference implementation.
// -------------------------------------------------------------------------

/// Compile the real phase1 Metal kernels from the workspace source and load
/// a `MetalRuntimeBringup`.  Returns `None` when xcrun is unavailable, when
/// the workspace `metal/phase1-kernels.json` manifest cannot be found, or
/// when the Metal device is absent.  The caller receives the temp output
/// directory and is responsible for cleaning it up.
#[cfg(target_os = "macos")]
fn try_compile_real_bringup() -> Option<(MetalRuntimeBringup, PathBuf)> {
    // CARGO_MANIFEST_DIR points at .../crates/ax-engine-core.
    // The workspace root is two levels up.
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = crate_dir.parent()?.parent()?;
    let manifest_path = workspace_root.join("metal").join("phase1-kernels.json");
    if !manifest_path.is_file() {
        return None; // manifest not found in workspace
    }

    let output_dir = unique_test_dir("real-bringup");
    let request = MetalKernelBuildRequest {
        manifest_path,
        output_dir: output_dir.clone(),
        doctor: sample_build_doctor(true, true),
    };

    let artifacts = build_phase1_kernel_artifacts(&request).ok()?;
    if artifacts.build_status() != MetalBuildStatus::Compiled {
        let _ = fs::remove_dir_all(&output_dir);
        return None; // xcrun failed or toolchain unavailable
    }

    // Wrap Metal initialization in autoreleasepool so that internally
    // autoreleased objects created by the Metal framework during pipeline
    // compilation are drained while the bringup (and all its Metal objects)
    // is still alive.  Moving bringup out of the closure ensures it outlives
    // the pool drain, preventing double-release on thread exit.
    let bringup =
        autoreleasepool(|| MetalRuntimeBringup::from_build_dir(&artifacts.output_dir).ok())?;
    Some((bringup, output_dir))
}

// --- GPU test: expand_grouped_kv_heads_f32 --------------------------------

#[cfg(target_os = "macos")]
#[test]
fn expand_grouped_kv_heads_f32_gpu_output_matches_cpu_reference() {
    let Some((bringup, output_dir)) = try_compile_real_bringup() else {
        return; // xcrun / Metal toolchain unavailable — skip
    };

    // 2 kv heads, 4 q heads, head_dim=3
    let kv = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cpu = expand_grouped_kv_heads_cpu(&kv, 4, 2, 3).expect("cpu expand should succeed");
    let gpu = expand_grouped_kv_heads_with_path(&kv, 4, 2, 3, Some(&bringup))
        .expect("gpu expand should succeed");

    assert_f32_slice_close(&gpu, &cpu, 1e-5);
    let _ = fs::remove_dir_all(output_dir);
}

// --- GPU test: ffn_gate_silu_product_f32 ----------------------------------

#[cfg(target_os = "macos")]
#[test]
fn ffn_gate_silu_product_f32_gpu_output_matches_cpu_reference() {
    let Some((bringup, output_dir)) = try_compile_real_bringup() else {
        return;
    };
    let model_dir = write_projection_native_model_fixture(); // Qwen → silu
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");

    let gate_input = vec![1.0_f32, -1.0, 2.0, 0.5, -0.5, 3.0, -3.0, 0.0];
    let up = vec![0.5_f32, 2.0, -1.0, 1.5, 1.0, 0.25, -0.5, 1.0];

    // CPU reference
    let mut cpu_gate = gate_input.clone();
    apply_model_gate_up_product(&artifacts, &mut cpu_gate, &up);

    // GPU path
    let mut gpu_gate = gate_input.clone();
    let used_native =
        apply_model_gate_up_product_with_path(&artifacts, &mut gpu_gate, &up, Some(&bringup))
            .expect("gate_up_product_with_path should succeed");

    assert!(used_native, "GPU path should be taken when bringup is Some");
    assert_f32_slice_close(&gpu_gate, &cpu_gate, 1e-5);

    let _ = fs::remove_dir_all(&model_dir);
    let _ = fs::remove_dir_all(output_dir);
}

// --- GPU test: ffn_gate_gelu_approx_product_f32 ---------------------------

#[cfg(target_os = "macos")]
#[test]
fn ffn_gate_gelu_approx_product_f32_gpu_output_matches_cpu_reference() {
    let Some((bringup, output_dir)) = try_compile_real_bringup() else {
        return;
    };
    let model_dir = write_gemma_projection_native_model_fixture(); // Gemma → gelu_approx
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("gemma artifacts should load");

    let gate_input = vec![0.5_f32, -0.5, 1.0, -1.0, 2.0, -2.0, 0.25, -0.25];
    let up = vec![1.0_f32, 1.0, 0.5, 0.5, 2.0, 2.0, -1.0, -1.0];

    let mut cpu_gate = gate_input.clone();
    apply_model_gate_up_product(&artifacts, &mut cpu_gate, &up);

    let mut gpu_gate = gate_input.clone();
    let used_native =
        apply_model_gate_up_product_with_path(&artifacts, &mut gpu_gate, &up, Some(&bringup))
            .expect("gate_up_product_with_path should succeed");

    assert!(used_native, "GPU path should be taken when bringup is Some");
    // gelu_approx uses a polynomial approximation; tolerance is slightly wider
    assert_f32_slice_close(&gpu_gate, &cpu_gate, 1e-4);

    let _ = fs::remove_dir_all(&model_dir);
    let _ = fs::remove_dir_all(output_dir);
}

// --- GPU test: apply_rope_f32 (NeoX style, Qwen) --------------------------

#[cfg(target_os = "macos")]
#[test]
fn apply_rope_f32_gpu_output_matches_cpu_reference_qwen() {
    let Some((bringup, output_dir)) = try_compile_real_bringup() else {
        return;
    };
    let model_dir = write_projection_native_model_fixture(); // Qwen → NeoX
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");
    let stage_dims = ModelStageDims {
        input_dim: 8,
        q_heads: 2,
        kv_heads: 2,
        head_dim: 4,
    };
    let input_query = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_key = vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];

    // CPU reference
    let mut cpu_query = input_query.clone();
    let mut cpu_key = input_key.clone();
    apply_model_stage_rope_cpu(&artifacts, &mut cpu_query, &mut cpu_key, 3.0, stage_dims);

    // GPU path (dispatches to GPU when bringup is Some)
    let mut gpu_query = input_query.clone();
    let mut gpu_key = input_key.clone();
    apply_model_stage_rope_with_path(
        &artifacts,
        &mut gpu_query,
        &mut gpu_key,
        3.0,
        stage_dims,
        Some(&bringup),
    );

    assert_f32_slice_close(&gpu_query, &cpu_query, 1e-4);
    assert_f32_slice_close(&gpu_key, &cpu_key, 1e-4);

    let _ = fs::remove_dir_all(&model_dir);
    let _ = fs::remove_dir_all(output_dir);
}

// --- GPU test: apply_rope_f32 (Interleaved style, Gemma) ------------------

#[cfg(target_os = "macos")]
#[test]
fn apply_rope_f32_gpu_output_matches_cpu_reference_gemma() {
    let Some((bringup, output_dir)) = try_compile_real_bringup() else {
        return;
    };
    let model_dir = write_gemma_projection_native_model_fixture();
    let artifacts =
        NativeModelArtifacts::from_dir(&model_dir).expect("gemma artifacts should load");
    let stage_dims = ModelStageDims {
        input_dim: 8,
        q_heads: 2,
        kv_heads: 2,
        head_dim: 4,
    };
    let input_query = vec![1.0_f32, 0.5, -1.0, 2.0, -2.0, 0.25, 0.75, -0.5];
    let input_key = vec![0.1_f32, -0.1, 0.9, -0.9, 1.1, -1.1, 0.3, -0.3];

    let mut cpu_query = input_query.clone();
    let mut cpu_key = input_key.clone();
    apply_model_stage_rope_cpu(&artifacts, &mut cpu_query, &mut cpu_key, 1.0, stage_dims);

    let mut gpu_query = input_query.clone();
    let mut gpu_key = input_key.clone();
    apply_model_stage_rope_with_path(
        &artifacts,
        &mut gpu_query,
        &mut gpu_key,
        1.0,
        stage_dims,
        Some(&bringup),
    );

    assert_f32_slice_close(&gpu_query, &cpu_query, 1e-4);
    assert_f32_slice_close(&gpu_key, &cpu_key, 1e-4);

    let _ = fs::remove_dir_all(&model_dir);
    let _ = fs::remove_dir_all(output_dir);
}

// --- GPU test: rms_norm_batched_f32 (per-head norm) -----------------------

#[cfg(target_os = "macos")]
#[test]
fn rms_norm_batched_f32_gpu_output_matches_cpu_reference() {
    let Some((bringup, output_dir)) = try_compile_real_bringup() else {
        return;
    };
    let model_dir = write_projection_native_model_fixture();
    let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
    let bindings =
        MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
    let device = Device::system_default().expect("Metal device should exist on macOS");
    let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
        .expect("buffers should bind");
    let attention_norm = buffers
        .binding_for(&bindings.layers[0].attention_norm)
        .expect("attention norm binding should resolve");

    let head_count = 2_usize;
    let head_dim = 4_usize;
    let epsilon = native_model_rms_norm_epsilon(&artifacts);
    let weight_offset = native_model_rms_norm_weight_offset(&artifacts);

    let original: Vec<f32> = (1..=8).map(|i| i as f32 * 0.5).collect();

    // CPU reference: per-head norm via scalar function
    let weights =
        tensor_prefix_f32(attention_norm, head_dim).expect("attention norm weights should read");
    let mut cpu_out = original.clone();
    for head in cpu_out.chunks_exact_mut(head_dim) {
        apply_rms_norm_with_weights_in_place(head, &weights, epsilon, weight_offset)
            .expect("cpu per-head norm should succeed");
    }

    // GPU path via batched tally function
    let mut gpu_out = original.clone();
    let tally = apply_per_head_rms_norm_with_binding_in_place_with_tally(
        &mut gpu_out,
        head_count,
        head_dim,
        attention_norm,
        epsilon,
        weight_offset,
        Some(&bringup),
    )
    .expect("per_head_rms_norm_with_tally should succeed");

    assert_f32_slice_close(&gpu_out, &cpu_out, 1e-4);
    // When the batched GPU kernel fires, native element count is non-zero
    assert!(
        tally.native_rms_norm_element_count() > 0,
        "native element count should be > 0 when GPU path was taken"
    );

    let _ = fs::remove_dir_all(&model_dir);
    let _ = fs::remove_dir_all(output_dir);
}

// --- GPU test: reshape_and_cache (required kernel) ------------------------

/// Runs the full required-kernel dispatch and verifies that `reshape_and_cache`
/// wrote each token's key and value vectors to the correct paged KV cache slot.
/// Failure indicates that the `reshape_and_cache` Metal kernel is producing
/// wrong slot assignments or corrupted values (validation stage: "key_cache").
#[cfg(target_os = "macos")]
#[test]
fn reshape_and_cache_gpu_writes_kv_to_correct_cache_slots() {
    let Some((bringup, output_dir)) = try_compile_real_bringup() else {
        return; // xcrun / Metal toolchain unavailable — skip
    };
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");
    let staged_inputs = synthetic_staged_inputs(&workload);

    let (_trace, cache_snapshot) = bringup
        .dispatch_numeric_workload_ephemeral_with_cache_snapshot_and_attention_config(
            &workload,
            &staged_inputs,
            None,
        )
        .expect(
            "reshape_and_cache: GPU should write key/value token vectors to the \
                 correct paged KV cache slots (numeric validation stage: key_cache)",
        );

    // Compare the GPU cache snapshot against the CPU reference.  The snapshot
    // incorporates copy_blocks destinations via merge_copy_targets so we build
    // the matching CPU snapshot the same way.
    let reference = reference_numeric_path_with_inputs(&workload, &staged_inputs);
    let reference_snapshot =
        MetalDispatchKvCacheSnapshot::from_reference_for_workload(&workload, &reference);
    assert_f32_slice_close(
        &cache_snapshot.key_cache,
        &reference_snapshot.key_cache,
        1e-5,
    );
    assert_f32_slice_close(
        &cache_snapshot.value_cache,
        &reference_snapshot.value_cache,
        1e-5,
    );

    let _ = fs::remove_dir_all(output_dir);
}

// --- GPU test: paged_decode_attention (required kernel) -------------------

/// Runs the full required-kernel dispatch and verifies that `paged_decode_attention`
/// produces attention outputs that match the CPU softmax-attention reference.
/// Failure indicates the attention kernel has wrong softmax scaling, causal masking,
/// or output accumulation (validation stage: "attention_output").
#[cfg(target_os = "macos")]
#[test]
fn paged_decode_attention_gpu_output_matches_cpu_softmax_reference() {
    let Some((bringup, output_dir)) = try_compile_real_bringup() else {
        return;
    };
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");
    let staged_inputs = synthetic_staged_inputs(&workload);

    let (trace, _cache_snapshot) = bringup
        .dispatch_numeric_workload_ephemeral_with_cache_snapshot_and_attention_config(
            &workload,
            &staged_inputs,
            None,
        )
        .expect(
            "paged_decode_attention: GPU attention output should match CPU softmax \
                 reference (numeric validation stage: attention_output)",
        );

    let reference = reference_numeric_path_with_inputs(&workload, &staged_inputs);
    let gpu_attn: Vec<f32> = trace
        .numeric
        .attention_output_bits
        .iter()
        .map(|&bits| f32::from_bits(bits))
        .collect();
    assert_f32_slice_close(&gpu_attn, &reference.attention_output, 1e-4);

    let _ = fs::remove_dir_all(output_dir);
}

// --- GPU test: gather_kv_cache (required kernel) --------------------------

/// Runs the full required-kernel dispatch and verifies that `gather_kv_cache`
/// assembles the correct KV slices from paged block-table metadata.
/// Failure indicates the gather kernel is reading from wrong block offsets or
/// producing a mismatched KV layout for attention (validation stage: "gather_kv").
#[cfg(target_os = "macos")]
#[test]
fn gather_kv_cache_gpu_output_passes_validation_against_cpu_reference() {
    let Some((bringup, output_dir)) = try_compile_real_bringup() else {
        return;
    };
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");
    let staged_inputs = synthetic_staged_inputs(&workload);

    let (trace, _cache_snapshot) = bringup
        .dispatch_numeric_workload_ephemeral_with_cache_snapshot_and_attention_config(
            &workload,
            &staged_inputs,
            None,
        )
        .expect(
            "gather_kv_cache: GPU gather of block-table KV slices should match CPU \
                 reference checksum (numeric validation stage: gather_kv)",
        );

    let summary = trace
        .numeric
        .validation
        .expect("validation summary should be present on successful dispatch");
    assert_eq!(
        trace.numeric.gather_output_checksum, summary.expected_gather_output_checksum,
        "gather_kv_cache: GPU gather checksum should equal CPU reference checksum"
    );

    let _ = fs::remove_dir_all(output_dir);
}

// --- GPU test: copy_blocks (required kernel) ------------------------------

/// Runs the full required-kernel dispatch and verifies that `copy_blocks`
/// moves KV block data to the correct destination slots.
/// Failure indicates the copy kernel is writing to wrong destinations or
/// corrupting block contents during movement (validation stage: "copy_blocks").
#[cfg(target_os = "macos")]
#[test]
fn copy_blocks_gpu_output_passes_validation_against_cpu_reference() {
    let Some((bringup, output_dir)) = try_compile_real_bringup() else {
        return;
    };
    let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
        .expect("workload should resolve");
    let staged_inputs = synthetic_staged_inputs(&workload);

    let (trace, _cache_snapshot) = bringup
        .dispatch_numeric_workload_ephemeral_with_cache_snapshot_and_attention_config(
            &workload,
            &staged_inputs,
            None,
        )
        .expect(
            "copy_blocks: GPU block-movement primitive should write matching KV data \
                 to copy destinations (numeric validation stage: copy_blocks)",
        );

    let summary = trace
        .numeric
        .validation
        .expect("validation summary should be present on successful dispatch");
    assert_eq!(
        trace.numeric.copy_output_checksum, summary.expected_copy_output_checksum,
        "copy_blocks: GPU copy checksum should equal CPU reference checksum"
    );

    let _ = fs::remove_dir_all(output_dir);
}

fn unique_test_dir(label: &str) -> PathBuf {
    static NEXT_SUFFIX: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    let suffix = NEXT_SUFFIX.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ax-engine-core-{label}-{}-{nanos}-{suffix}",
        std::process::id()
    ))
}
