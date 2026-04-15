use super::*;
use crate::backend::Qwen3_5RecurrentStateBatch;
use crate::compute::rms_norm;
use crate::gguf::MappedModel;
use crate::gguf::tensor::GgmlType;
use crate::model::WeightStore;
use crate::model::qwen35::{
    Qwen3_5Forward, Qwen3_5NativeRecurrentCachedWeights, Qwen3_5NativeRecurrentDtypes,
    Qwen3_5NativeRecurrentProjectionScratch,
};
use ax_engine_metal::MatvecProfileVariant;
use std::path::PathBuf;

fn lock_env_test() -> std::sync::MutexGuard<'static, ()> {
    static ENV_TEST_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    ENV_TEST_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn workspace_model_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(file_name)
}

fn is_active_layer_weight_name(name: &str) -> bool {
    const LAYER_SUFFIXES: &[&str] = &[
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ];

    name.starts_with("blk.") && LAYER_SUFFIXES.iter().any(|suffix| name.ends_with(suffix))
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(previous) => unsafe { std::env::set_var(self.key, previous) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

fn test_fingerprint() -> ModelFingerprint {
    ModelFingerprint {
        model_name: "Qwen3-8B".to_string(),
        architecture: "qwen3".to_string(),
        family: "qwen3".to_string(),
        size_label: "8b".to_string(),
        n_layers: 36,
        n_heads: 32,
        n_kv_heads: 8,
        embedding_dim: 4096,
        head_dim: 128,
        intermediate_dim: 14336,
        context_length: 4096,
        sliding_window_size: None,
        sliding_window_pattern: None,
        n_expert: None,
        n_expert_used: None,
        qwen35_full_attention_interval: None,
        total_tensor_bytes: 0,
        predominant_quant: "Q4_K".to_string(),
        predominant_layer_quant: "Q4_K".to_string(),
        lm_head_quant: None,
        layer_quant_histogram: vec![],
        has_mixed_layer_quants: false,
        has_q4k_layer_weights: true,
        has_q5k_layer_weights: false,
        has_q6k_layer_weights: false,
        has_q8_layer_weights: false,
        has_f32_layer_weights: false,
    }
}

#[test]
fn test_validate_moe_mul_mat_id_dtype_accepts_supported_routed_quants() {
    MetalOps::validate_moe_mul_mat_id_dtype(GgmlType::Q4K, "expert").unwrap();
    MetalOps::validate_moe_mul_mat_id_dtype(GgmlType::Q5_1, "expert").unwrap();
    MetalOps::validate_moe_mul_mat_id_dtype(GgmlType::Q5K, "expert").unwrap();
    MetalOps::validate_moe_mul_mat_id_dtype(GgmlType::Q6K, "expert").unwrap();
    MetalOps::validate_moe_mul_mat_id_dtype(GgmlType::Q8_0, "expert").unwrap();
}

#[test]
fn test_validate_moe_selected_dtype_accepts_supported_quantized_experts() {
    MetalOps::validate_moe_selected_dtype(GgmlType::Q4K, "expert").unwrap();
    MetalOps::validate_moe_selected_dtype(GgmlType::Q5K, "expert").unwrap();
    MetalOps::validate_moe_selected_dtype(GgmlType::Q6K, "expert").unwrap();
    MetalOps::validate_moe_selected_dtype(GgmlType::Q8_0, "expert").unwrap();
}

#[test]
fn test_moe_blocks_per_expert_requires_aligned_q4k_stride() {
    assert_eq!(
        MetalOps::moe_blocks_per_expert(288, GgmlType::Q4K, "expert").unwrap(),
        2
    );

    let err = MetalOps::moe_blocks_per_expert(145, GgmlType::Q4K, "expert")
        .expect_err("misaligned Q4_K stride should fail");
    assert!(err.to_string().contains("stride 145"));
}

#[test]
fn test_supports_q8_kv_requires_supported_alignment_and_head_dim() {
    let fingerprint = test_fingerprint();
    assert!(MetalOps::supports_q8_kv(&fingerprint));

    let mut unsupported_head_dim = fingerprint.clone();
    unsupported_head_dim.head_dim = 96;
    assert!(!MetalOps::supports_q8_kv(&unsupported_head_dim));

    let mut unsupported_stride = fingerprint;
    unsupported_stride.n_kv_heads = 0;
    assert!(!MetalOps::supports_q8_kv(&unsupported_stride));
}

#[test]
fn test_quant_from_profile_label_parses_common_forms() {
    assert_eq!(quant_from_profile_label("Q4_K"), Some(GgmlType::Q4K));
    assert_eq!(quant_from_profile_label("Q4_K_M"), Some(GgmlType::Q4K));
    assert_eq!(quant_from_profile_label("q5-k"), Some(GgmlType::Q5K));
    assert_eq!(quant_from_profile_label("q5_k_s"), Some(GgmlType::Q5K));
    assert_eq!(quant_from_profile_label("q6k"), Some(GgmlType::Q6K));
    assert_eq!(quant_from_profile_label("Q8"), Some(GgmlType::Q8_0));
    assert_eq!(quant_from_profile_label("Q8_0"), Some(GgmlType::Q8_0));
    assert_eq!(quant_from_profile_label("Q8_1"), None);
    assert_eq!(quant_from_profile_label("unknown"), None);
}

#[test]
fn test_preferred_batch_prefill_quant_accepts_normalized_layer_label_forms() {
    let backend = MetalBackend::new().unwrap();
    let mut fingerprint = test_fingerprint();
    fingerprint.predominant_layer_quant = "q5-k".to_string();
    fingerprint.has_q4k_layer_weights = false;
    fingerprint.has_q5k_layer_weights = false;

    assert_eq!(
        backend.ops.preferred_batch_prefill_quant(&fingerprint),
        Some(GgmlType::Q5K)
    );
}

#[test]
fn test_preferred_batch_prefill_quant_falls_back_to_predominant_quant() {
    let backend = MetalBackend::new().unwrap();
    let mut fingerprint = test_fingerprint();
    fingerprint.predominant_layer_quant = "unknown".to_string();
    fingerprint.predominant_quant = "Q6K".to_string();
    fingerprint.has_q4k_layer_weights = false;
    fingerprint.has_q6k_layer_weights = false;

    assert_eq!(
        backend.ops.preferred_batch_prefill_quant(&fingerprint),
        Some(GgmlType::Q6K)
    );
}

#[test]
fn test_estimated_attention_kv_bytes_respects_sparse_attention_layers() {
    let mut fingerprint = test_fingerprint();
    fingerprint.architecture = "qwen35".to_string();
    fingerprint.n_layers = 28;
    fingerprint.context_length = 8192;
    fingerprint.qwen35_full_attention_interval = Some(4);

    let estimated =
        MetalOps::estimated_attention_kv_bytes(&fingerprint, KvPrecisionPolicy::ForceQ8_0).unwrap();
    let kv_stride = fingerprint.n_kv_heads as u64 * fingerprint.head_dim as u64;
    let row_bytes = (kv_stride / 32) * 34;
    let expected_layers = 7u64;
    let expected = expected_layers * fingerprint.context_length as u64 * row_bytes * 2;

    assert_eq!(
        MetalOps::estimated_attention_kv_layers(&fingerprint),
        expected_layers
    );
    assert_eq!(estimated, expected);
}

#[test]
fn test_qwen35moe_scratch_dims_cover_q_projection_and_recurrent_inner_size() {
    let config = ModelConfig {
        architecture: "qwen35moe".into(),
        n_layers: 40,
        n_heads: 16,
        n_kv_heads: 2,
        embedding_dim: 2048,
        head_dim: 256,
        intermediate_dim: 1024,
        context_length: 128,
        vocab_size: 32_000,
        rms_norm_eps: 1e-6,
        rope_freq_base: 1_000_000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: crate::model::config::GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: Some(256),
        n_expert_used: Some(8),
        expert_intermediate_dim: Some(512),
        qwen35_full_attention_interval: Some(4),
        qwen35_ssm_conv_kernel: Some(4),
        qwen35_ssm_inner_size: Some(8192),
        qwen35_ssm_state_size: Some(128),
        qwen35_ssm_time_step_rank: Some(32),
        qwen35_ssm_group_count: Some(16),
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let q_dim = config.n_heads as usize * config.head_dim as usize;
    let (gate_dim, up_dim) =
        qwen35_gate_up_scratch_dims(&config, q_dim, config.intermediate_dim as usize);

    assert_eq!(gate_dim, 8192);
    assert_eq!(up_dim, 8192);
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs_diff_with_index(a: &[f32], b: &[f32]) -> (usize, f32) {
    a.iter()
        .zip(b.iter())
        .enumerate()
        .map(|(idx, (x, y))| (idx, (x - y).abs()))
        .max_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap())
        .unwrap_or((0, 0.0))
}

fn q4k_block_first128_constant(nibble: u8) -> [u8; 144] {
    let mut block = [0u8; 144];
    let d = half::f16::from_f32(1.0).to_le_bytes();
    block[0] = d[0];
    block[1] = d[1];
    block[2] = 0;
    block[3] = 0;
    block[4] = 1;
    block[5] = 1;
    block[6] = 1;
    block[7] = 1;
    block[16..144].fill((nibble & 0x0f) | ((nibble & 0x0f) << 4));
    block
}

fn q5k_block_first128_scaled(scale: f32) -> [u8; 176] {
    let mut block = [0u8; 176];
    let d = half::f16::from_f32(scale).to_le_bytes();
    block[0] = d[0];
    block[1] = d[1];
    block[4] = 1;
    block[5] = 1;
    block[6] = 1;
    block[7] = 1;
    block[48..112].fill(0x55);
    block
}

#[test]
fn test_metal_backend_init() {
    let _backend = MetalBackend::new().unwrap();
}

#[test]
fn test_metal_backend_matmul_small() {
    let backend = MetalBackend::new().unwrap();
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let mut c = [0.0f32; 4];
    backend.matmul(&a, &b, &mut c, 2, 2, 2);
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    assert!((c[0] - 19.0).abs() < 1e-3);
    assert!((c[1] - 22.0).abs() < 1e-3);
    assert!((c[2] - 43.0).abs() < 1e-3);
    assert!((c[3] - 50.0).abs() < 1e-3);
}

#[test]
fn test_metal_backend_matvec() {
    let backend = MetalBackend::new().unwrap();
    // A: 3×4, x: 4×1
    #[rustfmt::skip]
    let a = [
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ];
    let x = [1.0, 2.0, 3.0, 4.0];
    let mut y = [0.0f32; 3];
    backend.matmul(&a, &x, &mut y, 3, 1, 4);
    // row 0: 1+4+9+16 = 30
    // row 1: 5+12+21+32 = 70
    // row 2: 9+20+33+48 = 110
    assert!((y[0] - 30.0).abs() < 1e-3);
    assert!((y[1] - 70.0).abs() < 1e-3);
    assert!((y[2] - 110.0).abs() < 1e-3);
}

#[test]
fn test_metal_backend_dequant_matmul_f32() {
    // dequant_matmul with F32 dtype should work (passthrough)
    let backend = MetalBackend::new().unwrap();
    let a_f32: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let a_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(a_f32.as_ptr() as *const u8, std::mem::size_of_val(&a_f32))
    };
    let b = [1.0f32, 0.0, 0.0, 1.0];
    let mut c = [0.0f32; 4];
    backend.dequant_matmul(a_bytes, GgmlType::F32, &b, &mut c, 2, 2, 2);
    // Identity multiplication: [[1,2],[3,4]] * I = [[1,2],[3,4]]
    assert!((c[0] - 1.0).abs() < 1e-3);
    assert!((c[1] - 2.0).abs() < 1e-3);
    assert!((c[2] - 3.0).abs() < 1e-3);
    assert!((c[3] - 4.0).abs() < 1e-3);
}

#[test]
fn test_prepare_multi_token_gdn_bh_buffers_matches_legacy_pack_path() {
    let n_tokens = 3;
    let group_count = 2;
    let time_step_rank = 4;
    let state_size = 4;
    let key_dim = group_count * state_size;
    let value_dim = time_step_rank * state_size;
    let conv_dim = 2 * key_dim + value_dim;
    let rms_norm_eps = 1e-5;

    let conv_out: Vec<f32> = (0..n_tokens * conv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let alpha: Vec<f32> = (0..n_tokens * time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let beta: Vec<f32> = (0..n_tokens * time_step_rank)
        .map(|i| 0.2 + (i % 5) as f32 * 0.03)
        .collect();

    let mut expected_q = vec![0.0f32; n_tokens * value_dim];
    let mut expected_k = vec![0.0f32; n_tokens * value_dim];
    let mut expected_v = vec![0.0f32; n_tokens * value_dim];

    for token_idx in 0..n_tokens {
        let conv_start = token_idx * conv_dim;
        let conv_end = conv_start + conv_dim;
        let conv_token = &conv_out[conv_start..conv_end];
        let mut q_lin = conv_token[..key_dim].to_vec();
        let mut k_lin = conv_token[key_dim..2 * key_dim].to_vec();
        let v_lin = &conv_token[2 * key_dim..2 * key_dim + value_dim];

        crate::compute::gdn::l2_norm_heads(&mut q_lin, group_count, state_size, rms_norm_eps);
        crate::compute::gdn::l2_norm_heads(&mut k_lin, group_count, state_size, rms_norm_eps);

        let q_rep =
            crate::compute::gdn::repeat_heads(&q_lin, group_count, time_step_rank, state_size);
        let k_rep =
            crate::compute::gdn::repeat_heads(&k_lin, group_count, time_step_rank, state_size);
        let out_start = token_idx * value_dim;
        let out_end = out_start + value_dim;
        expected_q[out_start..out_end].copy_from_slice(&q_rep);
        expected_k[out_start..out_end].copy_from_slice(&k_rep);
        expected_v[out_start..out_end].copy_from_slice(v_lin);
    }

    let expected_q = pack_token_major_to_bhsk(&expected_q, n_tokens, time_step_rank, state_size);
    let expected_k = pack_token_major_to_bhsk(&expected_k, n_tokens, time_step_rank, state_size);
    let expected_v = pack_token_major_to_bhsk(&expected_v, n_tokens, time_step_rank, state_size);
    let expected_gate = pack_token_major_scalars_to_bhs(&alpha, n_tokens, time_step_rank);
    let expected_beta = pack_token_major_scalars_to_bhs(&beta, n_tokens, time_step_rank);

    let mut actual_q = vec![0.0f32; n_tokens * value_dim];
    let mut actual_k = vec![0.0f32; n_tokens * value_dim];
    let mut actual_v = vec![0.0f32; n_tokens * value_dim];
    let mut actual_gate = vec![0.0f32; n_tokens * time_step_rank];
    let mut actual_beta = vec![0.0f32; n_tokens * time_step_rank];

    prepare_multi_token_gdn_bh_buffers(
        &conv_out,
        &alpha,
        &beta,
        &mut actual_q,
        &mut actual_k,
        &mut actual_v,
        &mut actual_gate,
        &mut actual_beta,
        n_tokens,
        group_count,
        time_step_rank,
        state_size,
        rms_norm_eps,
    );

    for (actual, expected) in actual_q.iter().zip(expected_q.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
    for (actual, expected) in actual_k.iter().zip(expected_k.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
    for (actual, expected) in actual_v.iter().zip(expected_v.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (actual, expected) in actual_gate.iter().zip(expected_gate.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (actual, expected) in actual_beta.iter().zip(expected_beta.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn test_metal_gdn_prepare_multi_token_qkv_matches_cpu_pack_path() {
    let backend = MetalBackend::new().unwrap();
    let n_tokens = 3usize;
    let group_count = 2usize;
    let time_step_rank = 4usize;
    let state_size = 4usize;
    let key_dim = group_count * state_size;
    let value_dim = time_step_rank * state_size;
    let conv_dim = 2 * key_dim + value_dim;
    let rms_norm_eps = 1e-5f32;

    let conv_out: Vec<f32> = (0..n_tokens * conv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let alpha: Vec<f32> = (0..n_tokens * time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let beta: Vec<f32> = (0..n_tokens * time_step_rank)
        .map(|i| 0.2 + (i % 5) as f32 * 0.03)
        .collect();

    let mut expected_q = vec![0.0f32; n_tokens * value_dim];
    let mut expected_k = vec![0.0f32; n_tokens * value_dim];
    let mut expected_v = vec![0.0f32; n_tokens * value_dim];
    let mut expected_gate = vec![0.0f32; n_tokens * time_step_rank];
    let mut expected_beta = vec![0.0f32; n_tokens * time_step_rank];
    prepare_multi_token_gdn_bh_buffers(
        &conv_out,
        &alpha,
        &beta,
        &mut expected_q,
        &mut expected_k,
        &mut expected_v,
        &mut expected_gate,
        &mut expected_beta,
        n_tokens,
        group_count,
        time_step_rank,
        state_size,
        rms_norm_eps,
    );

    let conv_buf = MetalBuffer::from_slice(backend.device.device(), &conv_out).unwrap();
    let q_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * value_dim * size_of::<f32>(),
    )
    .unwrap();
    let k_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * value_dim * size_of::<f32>(),
    )
    .unwrap();
    let v_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * value_dim * size_of::<f32>(),
    )
    .unwrap();
    let gate_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * time_step_rank * size_of::<f32>(),
    )
    .unwrap();
    let beta_out_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * time_step_rank * size_of::<f32>(),
    )
    .unwrap();

    assert!(
        backend
            .qwen35_prepare_multi_token_qkv_sync(
                &conv_buf,
                &alpha,
                &beta,
                &q_buf,
                &k_buf,
                &v_buf,
                &gate_buf,
                &beta_out_buf,
                n_tokens as u32,
                group_count as u32,
                time_step_rank as u32,
                state_size as u32,
                rms_norm_eps,
            )
            .unwrap()
    );

    let actual_q = unsafe { q_buf.as_slice::<f32>()[..n_tokens * value_dim].to_vec() };
    let actual_k = unsafe { k_buf.as_slice::<f32>()[..n_tokens * value_dim].to_vec() };
    let actual_v = unsafe { v_buf.as_slice::<f32>()[..n_tokens * value_dim].to_vec() };
    let actual_gate = unsafe { gate_buf.as_slice::<f32>()[..n_tokens * time_step_rank].to_vec() };
    let actual_beta =
        unsafe { beta_out_buf.as_slice::<f32>()[..n_tokens * time_step_rank].to_vec() };

    for (actual, expected) in actual_q.iter().zip(expected_q.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
    for (actual, expected) in actual_k.iter().zip(expected_k.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
    for (actual, expected) in actual_v.iter().zip(expected_v.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (actual, expected) in actual_gate.iter().zip(expected_gate.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (actual, expected) in actual_beta.iter().zip(expected_beta.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn test_metal_gdn_prepare_multi_token_qkv_matches_cpu_pack_path_with_f16_alpha_beta_storage() {
    let backend = MetalBackend::new().unwrap();
    let n_tokens = 3usize;
    let group_count = 2usize;
    let time_step_rank = 4usize;
    let state_size = 4usize;
    let key_dim = group_count * state_size;
    let value_dim = time_step_rank * state_size;
    let conv_dim = 2 * key_dim + value_dim;
    let rms_norm_eps = 1e-5f32;

    let conv_out: Vec<f32> = (0..n_tokens * conv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let alpha: Vec<f32> = (0..n_tokens * time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let beta: Vec<f32> = (0..n_tokens * time_step_rank)
        .map(|i| 0.2 + (i % 5) as f32 * 0.03)
        .collect();
    let alpha_f16: Vec<half::f16> = alpha.iter().copied().map(half::f16::from_f32).collect();
    let beta_f16: Vec<half::f16> = beta.iter().copied().map(half::f16::from_f32).collect();

    let mut expected_q = vec![0.0f32; n_tokens * value_dim];
    let mut expected_k = vec![0.0f32; n_tokens * value_dim];
    let mut expected_v = vec![0.0f32; n_tokens * value_dim];
    let mut expected_gate = vec![0.0f32; n_tokens * time_step_rank];
    let mut expected_beta = vec![0.0f32; n_tokens * time_step_rank];
    prepare_multi_token_gdn_bh_buffers(
        &conv_out,
        &alpha,
        &beta,
        &mut expected_q,
        &mut expected_k,
        &mut expected_v,
        &mut expected_gate,
        &mut expected_beta,
        n_tokens,
        group_count,
        time_step_rank,
        state_size,
        rms_norm_eps,
    );

    let conv_buf = MetalBuffer::from_slice(backend.device.device(), &conv_out).unwrap();
    let alpha_buf = MetalBuffer::from_slice(backend.device.device(), &alpha_f16).unwrap();
    let beta_buf = MetalBuffer::from_slice(backend.device.device(), &beta_f16).unwrap();
    let q_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * value_dim * size_of::<f32>(),
    )
    .unwrap();
    let k_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * value_dim * size_of::<f32>(),
    )
    .unwrap();
    let v_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * value_dim * size_of::<f32>(),
    )
    .unwrap();
    let gate_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * time_step_rank * size_of::<f32>(),
    )
    .unwrap();
    let beta_out_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * time_step_rank * size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            assert!(
                backend
                    .gdn_kernels
                    .encode_prepare_multi_token_qkv_alpha_beta_f16(
                        encoder,
                        &conv_buf,
                        &alpha_buf,
                        &beta_buf,
                        &q_buf,
                        &k_buf,
                        &v_buf,
                        &gate_buf,
                        &beta_out_buf,
                        n_tokens as u32,
                        group_count as u32,
                        time_step_rank as u32,
                        state_size as u32,
                        rms_norm_eps,
                    )
            );
            Ok(())
        })
        .unwrap();

    let actual_q = unsafe { q_buf.as_slice::<f32>()[..n_tokens * value_dim].to_vec() };
    let actual_k = unsafe { k_buf.as_slice::<f32>()[..n_tokens * value_dim].to_vec() };
    let actual_v = unsafe { v_buf.as_slice::<f32>()[..n_tokens * value_dim].to_vec() };
    let actual_gate = unsafe { gate_buf.as_slice::<f32>()[..n_tokens * time_step_rank].to_vec() };
    let actual_beta =
        unsafe { beta_out_buf.as_slice::<f32>()[..n_tokens * time_step_rank].to_vec() };

    for (actual, expected) in actual_q.iter().zip(expected_q.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
    for (actual, expected) in actual_k.iter().zip(expected_k.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
    for (actual, expected) in actual_v.iter().zip(expected_v.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (actual, expected) in actual_gate.iter().zip(expected_gate.iter()) {
        assert!((actual - expected).abs() < 5e-4);
    }
    for (actual, expected) in actual_beta.iter().zip(expected_beta.iter()) {
        assert!((actual - expected).abs() < 5e-4);
    }
}

#[test]
fn test_metal_gdn_unpack_bhsk_to_token_major_matches_cpu() {
    let backend = MetalBackend::new().unwrap();
    let n_tokens = 3usize;
    let n_heads = 4usize;
    let head_dim = 4usize;
    let input: Vec<f32> = (0..n_tokens * n_heads * head_dim)
        .map(|i| i as f32 * 0.25 - 3.0)
        .collect();
    let mut expected = vec![0.0f32; input.len()];
    unpack_bhsk_to_token_major(&input, &mut expected, n_tokens, n_heads, head_dim);

    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let output_buf =
        MetalBuffer::new(backend.device.device(), input.len() * size_of::<f32>()).unwrap();
    backend
        .gdn_kernels
        .unpack_bhsk_to_token_major(
            &backend.device,
            &input_buf,
            &output_buf,
            n_tokens as u32,
            n_heads as u32,
            head_dim as u32,
        )
        .unwrap();
    let actual = unsafe { output_buf.as_slice::<f32>()[..input.len()].to_vec() };

    assert_eq!(actual, expected);
}

#[test]
fn test_metal_backend_batch_dequant_matvec_mixed_dtypes_share_command_buffer() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let k = 32;
    // Q8_0 block: 2 bytes f16 scale + 32 i8 values = 34 bytes, 32 values
    let mut q8_block = [0u8; 34];
    let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
    q8_block[0] = d_bytes[0];
    q8_block[1] = d_bytes[1];
    // Fill i8 quantized values with 1 (signed)
    q8_block[2..34].fill(1);

    let f32_weight = [0.5f32; 32];
    let f32_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            f32_weight.as_ptr() as *const u8,
            std::mem::size_of_val(&f32_weight),
        )
    };

    let x = [1.0f32; 32];
    let mut q8_out = [0.0f32; 1];
    let mut f32_out = [0.0f32; 1];
    let mut expected_q8 = [0.0f32; 1];
    let mut expected_f32 = [0.0f32; 1];

    cpu.dequant_matmul(&q8_block, GgmlType::Q8_0, &x, &mut expected_q8, 1, 1, k);
    cpu.dequant_matmul(f32_bytes, GgmlType::F32, &x, &mut expected_f32, 1, 1, k);

    backend.device.reset_perf_counters();
    backend.batch_dequant_matvec(
        &[
            (&q8_block, GgmlType::Q8_0, 1),
            (f32_bytes, GgmlType::F32, 1),
        ],
        &x,
        k,
        &mut [&mut q8_out, &mut f32_out],
    );
    let counters = backend.device.perf_counters();

    assert!((q8_out[0] - expected_q8[0]).abs() < 1e-2);
    assert!((f32_out[0] - expected_f32[0]).abs() < 1e-3);
    assert_eq!(
        counters.command_buffers, 1,
        "mixed-dtype batch matvec should use one command buffer"
    );
}

#[test]
fn test_metal_backend_fused_decode_after_dense_cache_uses_quant_weight_cache() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let k = 32;
    let mut q8_block = [0u8; 34];
    let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
    q8_block[0] = d_bytes[0];
    q8_block[1] = d_bytes[1];
    q8_block[2..34].fill(1);

    // Prime the dense cache with the multi-token prefill route first.
    let prefill_x = [1.0f32; 64];
    let mut prefill_out = [0.0f32; 2];
    backend.dequant_matmul(
        &q8_block,
        GgmlType::Q8_0,
        &prefill_x,
        &mut prefill_out,
        1,
        2,
        k,
    );

    let x = [1.0f32; 32];
    let mut actual = [0.0f32; 1];
    let mut expected = [0.0f32; 1];
    cpu.dequant_matmul(&q8_block, GgmlType::Q8_0, &x, &mut expected, 1, 1, k);
    backend.dequant_matmul(&q8_block, GgmlType::Q8_0, &x, &mut actual, 1, 1, k);

    assert!(
        (actual[0] - expected[0]).abs() < 1e-2,
        "fused decode should read raw quant blocks after dense-cache priming: actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn test_metal_backend_batch_matvec_after_dense_cache_uses_quant_weight_cache() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let k = 32;
    let mut q8_block = [0u8; 34];
    let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
    q8_block[0] = d_bytes[0];
    q8_block[1] = d_bytes[1];
    q8_block[2..34].fill(1);

    let f32_weight = [0.5f32; 32];
    let f32_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            f32_weight.as_ptr() as *const u8,
            std::mem::size_of_val(&f32_weight),
        )
    };

    // Prime the dense cache with the multi-token prefill route first.
    let prefill_x = [1.0f32; 64];
    let mut prefill_out = [0.0f32; 2];
    backend.dequant_matmul(
        &q8_block,
        GgmlType::Q8_0,
        &prefill_x,
        &mut prefill_out,
        1,
        2,
        k,
    );

    let x = [1.0f32; 32];
    let mut q8_out = [0.0f32; 1];
    let mut f32_out = [0.0f32; 1];
    let mut expected_q8 = [0.0f32; 1];
    let mut expected_f32 = [0.0f32; 1];

    cpu.dequant_matmul(&q8_block, GgmlType::Q8_0, &x, &mut expected_q8, 1, 1, k);
    cpu.dequant_matmul(f32_bytes, GgmlType::F32, &x, &mut expected_f32, 1, 1, k);

    backend.batch_dequant_matvec(
        &[
            (&q8_block, GgmlType::Q8_0, 1),
            (f32_bytes, GgmlType::F32, 1),
        ],
        &x,
        k,
        &mut [&mut q8_out, &mut f32_out],
    );

    assert!(
        (q8_out[0] - expected_q8[0]).abs() < 1e-2,
        "batch decode should keep quantized weights separate from dense cache: actual={q8_out:?}, expected={expected_q8:?}"
    );
    assert!((f32_out[0] - expected_f32[0]).abs() < 1e-3);
}

#[test]
fn test_metal_backend_quant_cache_keys_are_disjoint_from_f32_keys() {
    let backend = MetalBackend::new().unwrap();
    let backing = [0u32; 8];
    let f32_weight =
        unsafe { std::slice::from_raw_parts(backing.as_ptr() as *const f32, backing.len()) };
    let quant_raw = unsafe {
        std::slice::from_raw_parts(
            backing.as_ptr() as *const u8,
            std::mem::size_of_val(&backing),
        )
    };

    let dense_key = backend.ops.ensure_f32_cached(f32_weight);
    let quant_key = backend.ops.ensure_quant_cached(quant_raw);
    let raw_key = quant_raw.as_ptr() as usize;

    assert_ne!(
        dense_key, quant_key,
        "resident quant weights must not share the dense/f32 key space"
    );
    assert_eq!(dense_key, f32_weight.as_ptr() as usize);
    assert_eq!(quant_key, MetalOps::quant_view_cache_key(raw_key));
    assert_ne!(quant_key, raw_key);

    let weight_cache = backend.ops.lock_weight_cache();
    assert!(weight_cache.contains_key(&dense_key));
    assert!(weight_cache.contains_key(&quant_key));
}

#[test]
fn test_metal_backend_safe_batch_dequant_matvec_mixed_dtypes_share_command_buffer() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let k = 32;
    // Q8_0 block: 2 bytes f16 scale + 32 i8 values = 34 bytes, 32 values
    let mut q8_block = [0u8; 34];
    let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
    q8_block[0] = d_bytes[0];
    q8_block[1] = d_bytes[1];
    // Fill i8 quantized values with 1 (signed)
    q8_block[2..34].fill(1);

    let f32_weight = [0.5f32; 32];
    let f32_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            f32_weight.as_ptr() as *const u8,
            std::mem::size_of_val(&f32_weight),
        )
    };

    let x = [1.0f32; 32];
    let mut q8_out = [0.0f32; 1];
    let mut f32_out = [0.0f32; 1];
    let mut expected_q8 = [0.0f32; 1];
    let mut expected_f32 = [0.0f32; 1];

    cpu.dequant_matmul(&q8_block, GgmlType::Q8_0, &x, &mut expected_q8, 1, 1, k);
    cpu.dequant_matmul(f32_bytes, GgmlType::F32, &x, &mut expected_f32, 1, 1, k);

    backend.device.reset_perf_counters();
    backend.safe_batch_dequant_matvec(
        &[
            (&q8_block, GgmlType::Q8_0, 1),
            (f32_bytes, GgmlType::F32, 1),
        ],
        &x,
        k,
        &mut [&mut q8_out, &mut f32_out],
    );
    let counters = backend.device.perf_counters();

    assert!((q8_out[0] - expected_q8[0]).abs() < 1e-2);
    assert!((f32_out[0] - expected_f32[0]).abs() < 1e-3);
    assert_eq!(
        counters.command_buffers, 1,
        "safe mixed-dtype batch matvec should use one command buffer"
    );
}

#[test]
fn test_metal_backend_real_qwen35_35b_a3b_recurrent_q8_input_projections_match_cpu() {
    let _env_lock = lock_env_test();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let weights = WeightStore::new(&model);
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let dim = 2048usize;
    let token_id = 42usize;
    let mut hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id, &mut hidden)
        .unwrap();
    let norm_w = weights.f32_slice("blk.0.attn_norm.weight").unwrap();
    let mut normed = vec![0.0f32; dim];
    rms_norm::rms_norm_out(&hidden, norm_w, &mut normed, 1e-6);

    for (tensor_name, rows) in [
        ("blk.0.attn_qkv.weight", 8192usize),
        ("blk.0.attn_gate.weight", 4096usize),
        ("blk.0.ssm_beta.weight", 32usize),
        ("blk.0.ssm_alpha.weight", 32usize),
    ] {
        let (raw, dtype) = weights.raw_with_dtype(tensor_name).unwrap();
        assert_eq!(
            dtype,
            GgmlType::Q8_0,
            "{tensor_name} dtype changed; update this regression"
        );

        let mut expected = vec![0.0f32; rows];
        let mut actual = vec![0.0f32; rows];
        cpu.dequant_matmul(raw, dtype, &normed, &mut expected, rows, 1, dim);
        backend.dequant_matmul(raw, dtype, &normed, &mut actual, rows, 1, dim);

        let max_diff = expected
            .iter()
            .zip(actual.iter())
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 5e-2,
            "{tensor_name} mismatch for real Qwen3.5-35B-A3B recurrent projection: max_diff={max_diff}"
        );
    }
}

#[test]
fn test_metal_backend_real_qwen35_35b_a3b_recurrent_sequence_matches_cpu() {
    let _env_lock = lock_env_test();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let weights = WeightStore::new(&model);
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let dim = 2048usize;
    let conv_dim = 8192usize;
    let time_step_rank = 32usize;
    let state_size = 128usize;
    let group_count = 16usize;
    let conv_cache_len = 3usize;
    let tokens_per_slot = 1usize;
    let slot_indices = [0usize];
    let layer_idx = 0usize;

    let mut hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", 42, &mut hidden)
        .unwrap();
    let norm_w = weights.f32_slice("blk.0.attn_norm.weight").unwrap();
    let mut normed = vec![0.0f32; dim];
    rms_norm::rms_norm_out(&hidden, norm_w, &mut normed, 1e-6);

    let mut qkv = vec![0.0f32; conv_dim];
    let mut beta = vec![0.0f32; time_step_rank];
    let mut alpha = vec![0.0f32; time_step_rank];
    cpu.dequant_matmul(
        weights.raw("blk.0.attn_qkv.weight").unwrap(),
        GgmlType::Q8_0,
        &normed,
        &mut qkv,
        conv_dim,
        1,
        dim,
    );
    cpu.dequant_matmul(
        weights.raw("blk.0.ssm_beta.weight").unwrap(),
        GgmlType::Q8_0,
        &normed,
        &mut beta,
        time_step_rank,
        1,
        dim,
    );
    cpu.dequant_matmul(
        weights.raw("blk.0.ssm_alpha.weight").unwrap(),
        GgmlType::Q8_0,
        &normed,
        &mut alpha,
        time_step_rank,
        1,
        dim,
    );

    let dt_bias = weights.f32_slice("blk.0.ssm_dt.bias").unwrap().to_vec();
    let a = weights.f32_slice("blk.0.ssm_a").unwrap().to_vec();
    let conv_kernel = weights
        .f32_slice("blk.0.ssm_conv1d.weight")
        .unwrap()
        .to_vec();
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len,
        conv_dim,
        group_count,
        state_size,
        time_step_rank,
        rms_norm_eps: 1e-6,
    };
    let conv_state_stride = conv_cache_len * conv_dim;
    let recurrent_state_stride = time_step_rank * state_size * state_size;

    let mut cpu_conv_state = vec![0.0f32; conv_state_stride];
    let mut cpu_recurrent_state = vec![0.0f32; recurrent_state_stride];
    let mut metal_conv_state = cpu_conv_state.clone();
    let mut metal_recurrent_state = cpu_recurrent_state.clone();
    let mut cpu_beta = beta.clone();
    let mut cpu_alpha = alpha.clone();
    let mut metal_beta = beta;
    let mut metal_alpha = alpha;
    let mut expected = vec![0.0f32; cfg.value_dim()];
    let mut actual = vec![0.0f32; cfg.value_dim()];

    {
        let mut state_batch = Qwen3_5RecurrentStateBatch::new(
            layer_idx,
            &slot_indices,
            &mut cpu_conv_state,
            &mut cpu_recurrent_state,
            conv_state_stride,
            recurrent_state_stride,
        );
        cpu.qwen35_recurrent_sequence(
            &qkv,
            &mut cpu_beta,
            &mut cpu_alpha,
            &dt_bias,
            &a,
            &conv_kernel,
            &mut state_batch,
            &mut expected,
            tokens_per_slot,
            cfg,
        );
    }
    {
        let mut state_batch = Qwen3_5RecurrentStateBatch::new(
            layer_idx,
            &slot_indices,
            &mut metal_conv_state,
            &mut metal_recurrent_state,
            conv_state_stride,
            recurrent_state_stride,
        );
        backend.qwen35_recurrent_sequence(
            &qkv,
            &mut metal_beta,
            &mut metal_alpha,
            &dt_bias,
            &a,
            &conv_kernel,
            &mut state_batch,
            &mut actual,
            tokens_per_slot,
            cfg,
        );
    }

    let output_diff = max_abs_diff(&actual, &expected);
    let alpha_diff = max_abs_diff(&metal_alpha, &cpu_alpha);
    let beta_diff = max_abs_diff(&metal_beta, &cpu_beta);
    let conv_diff = max_abs_diff(&metal_conv_state, &cpu_conv_state);
    let recurrent_diff = max_abs_diff(&metal_recurrent_state, &cpu_recurrent_state);

    assert!(
        output_diff < 1e-4,
        "real Qwen3.5-35B-A3B recurrent output mismatch: max_diff={output_diff}"
    );
    assert!(alpha_diff < 1e-5, "alpha mismatch: max_diff={alpha_diff}");
    assert!(beta_diff < 1e-5, "beta mismatch: max_diff={beta_diff}");
    assert!(
        conv_diff < 1e-5,
        "conv state mismatch: max_diff={conv_diff}"
    );
    assert!(
        recurrent_diff < 1e-4,
        "recurrent state mismatch: max_diff={recurrent_diff}"
    );
}

#[test]
fn test_metal_backend_real_qwen35_35b_a3b_layer0_attn_norm_matches_cpu() {
    let _env_lock = lock_env_test();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let weights = WeightStore::new(&model);
    let backend = MetalBackend::new().unwrap();

    let dim = 2048usize;
    let mut hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", 42, &mut hidden)
        .unwrap();
    let norm_w = weights.f32_slice("blk.0.attn_norm.weight").unwrap();
    let mut expected = vec![0.0f32; dim];
    rms_norm::rms_norm_out(&hidden, norm_w, &mut expected, 1e-6);

    let hidden_buf = MetalBuffer::from_slice(backend.device.device(), &hidden).unwrap();
    let norm_w_buf = MetalBuffer::from_slice(backend.device.device(), norm_w).unwrap();
    let out_buf =
        MetalBuffer::new(backend.device.device(), dim * std::mem::size_of::<f32>()).unwrap();
    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.elementwise.encode_rms_norm_out(
                encoder,
                &hidden_buf,
                &norm_w_buf,
                &out_buf,
                dim as u32,
                1e-6,
            );
            Ok(())
        })
        .unwrap();
    let actual = unsafe { out_buf.as_slice::<f32>()[..dim].to_vec() };
    let max_diff = max_abs_diff(&actual, &expected);
    assert!(
        max_diff < 1e-5,
        "real Qwen3.5-35B-A3B layer-0 attn_norm mismatch: max_diff={max_diff}"
    );
}

#[test]
fn test_metal_backend_real_qwen35_35b_a3b_layer3_full_attention_prep_matches_cpu() {
    let _env_lock = lock_env_test();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = crate::model::ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let layer_idx = 3usize;
    let token_id = 42usize;
    let position = 5usize;
    let dim = cfg.embedding_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let mut hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id, &mut hidden)
        .unwrap();
    let attn_norm_w = weights
        .f32_slice(&format!("blk.{layer_idx}.attn_norm.weight"))
        .unwrap();
    let mut normed = vec![0.0f32; dim];
    rms_norm::rms_norm_out(&hidden, attn_norm_w, &mut normed, cfg.rms_norm_eps);

    let q_weight_name = format!("blk.{layer_idx}.attn_q.weight");
    let k_weight_name = format!("blk.{layer_idx}.attn_k.weight");
    let v_weight_name = format!("blk.{layer_idx}.attn_v.weight");
    let (wq_raw, wq_dtype) = weights.raw_with_dtype(&q_weight_name).unwrap();
    let (wk_raw, wk_dtype) = weights.raw_with_dtype(&k_weight_name).unwrap();
    let (wv_raw, wv_dtype) = weights.raw_with_dtype(&v_weight_name).unwrap();

    let mut expected_q_gate_proj = vec![0.0f32; q_dim * 2];
    let mut expected_k_proj = vec![0.0f32; kv_dim];
    let mut expected_v_proj = vec![0.0f32; kv_dim];
    cpu.dequant_matmul(
        wq_raw,
        wq_dtype,
        &normed,
        &mut expected_q_gate_proj,
        q_dim * 2,
        1,
        dim,
    );
    cpu.dequant_matmul(
        wk_raw,
        wk_dtype,
        &normed,
        &mut expected_k_proj,
        kv_dim,
        1,
        dim,
    );
    cpu.dequant_matmul(
        wv_raw,
        wv_dtype,
        &normed,
        &mut expected_v_proj,
        kv_dim,
        1,
        dim,
    );

    let mut actual_q_gate_proj = vec![0.0f32; q_dim * 2];
    let mut actual_k_proj = vec![0.0f32; kv_dim];
    let mut actual_v_proj = vec![0.0f32; kv_dim];
    backend.dequant_matmul(
        wq_raw,
        wq_dtype,
        &normed,
        &mut actual_q_gate_proj,
        q_dim * 2,
        1,
        dim,
    );
    backend.dequant_matmul(
        wk_raw,
        wk_dtype,
        &normed,
        &mut actual_k_proj,
        kv_dim,
        1,
        dim,
    );
    backend.dequant_matmul(
        wv_raw,
        wv_dtype,
        &normed,
        &mut actual_v_proj,
        kv_dim,
        1,
        dim,
    );

    let q_proj_diff = max_abs_diff(&actual_q_gate_proj, &expected_q_gate_proj);
    let k_proj_diff = max_abs_diff(&actual_k_proj, &expected_k_proj);
    let v_proj_diff = max_abs_diff(&actual_v_proj, &expected_v_proj);
    assert!(
        q_proj_diff < 5e-2,
        "layer-3 q projection mismatch: max_diff={q_proj_diff}"
    );
    assert!(
        k_proj_diff < 5e-2,
        "layer-3 k projection mismatch: max_diff={k_proj_diff}"
    );
    assert!(
        v_proj_diff < 5e-2,
        "layer-3 v projection mismatch: max_diff={v_proj_diff}"
    );

    let mut expected_q = vec![0.0f32; q_dim];
    let mut expected_gate = vec![0.0f32; q_dim];
    for head in 0..n_heads {
        let src_head = head * head_dim * 2;
        let dst_head = head * head_dim;
        expected_q[dst_head..dst_head + head_dim]
            .copy_from_slice(&expected_q_gate_proj[src_head..src_head + head_dim]);
        expected_gate[dst_head..dst_head + head_dim]
            .copy_from_slice(&expected_q_gate_proj[src_head + head_dim..src_head + 2 * head_dim]);
    }
    let mut expected_k = expected_k_proj.clone();
    let q_norm_w = weights
        .f32_slice(&format!("blk.{layer_idx}.attn_q_norm.weight"))
        .unwrap();
    let k_norm_w = weights
        .f32_slice(&format!("blk.{layer_idx}.attn_k_norm.weight"))
        .unwrap();
    let mut q_head_norm = vec![0.0f32; head_dim];
    let mut k_head_norm = vec![0.0f32; head_dim];
    for head in 0..n_heads {
        let start = head * head_dim;
        let end = start + head_dim;
        rms_norm::rms_norm_out(
            &expected_q[start..end],
            q_norm_w,
            &mut q_head_norm,
            cfg.rms_norm_eps,
        );
        expected_q[start..end].copy_from_slice(&q_head_norm);
    }
    for head in 0..n_kv_heads {
        let start = head * head_dim;
        let end = start + head_dim;
        rms_norm::rms_norm_out(
            &expected_k[start..end],
            k_norm_w,
            &mut k_head_norm,
            cfg.rms_norm_eps,
        );
        expected_k[start..end].copy_from_slice(&k_head_norm);
    }
    let expected_q_normed = expected_q.clone();
    let expected_k_normed = expected_k.clone();
    crate::compute::rope::apply_rope_multi_head_neox_partial_scaled(
        &mut expected_q,
        &mut expected_k,
        n_heads,
        n_kv_heads,
        head_dim,
        head_dim.min(64),
        cfg.rope_scaling.scaled_position(position),
        cfg.rope_freq_base,
    );

    let q_gate_buf =
        MetalBuffer::from_slice(backend.device.device(), &expected_q_gate_proj).unwrap();
    let k_proj_buf = MetalBuffer::from_slice(backend.device.device(), &expected_k_proj).unwrap();
    let q_norm_w_buf = MetalBuffer::from_slice(backend.device.device(), q_norm_w).unwrap();
    let k_norm_w_buf = MetalBuffer::from_slice(backend.device.device(), k_norm_w).unwrap();
    let q_buf =
        MetalBuffer::new(backend.device.device(), q_dim * std::mem::size_of::<f32>()).unwrap();
    let gate_buf =
        MetalBuffer::new(backend.device.device(), q_dim * std::mem::size_of::<f32>()).unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.elementwise.encode_split_qgate_batch(
                encoder,
                &q_gate_buf,
                &q_buf,
                &gate_buf,
                1,
                q_dim as u32,
                head_dim as u32,
            );
            backend.ops.elementwise.encode_per_head_rms_norm_batch(
                encoder,
                &q_buf,
                &q_norm_w_buf,
                1,
                n_heads as u32,
                head_dim as u32,
                cfg.rms_norm_eps,
            );
            backend.ops.elementwise.encode_per_head_rms_norm_batch(
                encoder,
                &k_proj_buf,
                &k_norm_w_buf,
                1,
                n_kv_heads as u32,
                head_dim as u32,
                cfg.rms_norm_eps,
            );
            Ok(())
        })
        .unwrap();

    let actual_q_normed = unsafe { q_buf.as_slice::<f32>()[..q_dim].to_vec() };
    let actual_gate = unsafe { gate_buf.as_slice::<f32>()[..q_dim].to_vec() };
    let actual_k_normed = unsafe { k_proj_buf.as_slice::<f32>()[..kv_dim].to_vec() };

    let q_norm_diff = max_abs_diff(&actual_q_normed, &expected_q_normed);
    let gate_diff = max_abs_diff(&actual_gate, &expected_gate);
    let k_norm_diff = max_abs_diff(&actual_k_normed, &expected_k_normed);
    assert!(
        gate_diff < 1e-5,
        "layer-3 gate split mismatch: q_norm_diff={q_norm_diff} gate_diff={gate_diff} k_norm_diff={k_norm_diff}"
    );
    assert!(
        k_norm_diff < 1e-4,
        "layer-3 k norm mismatch: q_norm_diff={q_norm_diff} gate_diff={gate_diff} k_norm_diff={k_norm_diff}"
    );
    assert!(
        q_norm_diff < 1e-4,
        "layer-3 q norm mismatch: q_norm_diff={q_norm_diff} gate_diff={gate_diff} k_norm_diff={k_norm_diff}"
    );

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.elementwise.encode_rope_batch_neox_partial(
                encoder,
                &q_buf,
                &k_proj_buf,
                1,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                (head_dim as u32).min(64),
                cfg.rope_scaling.scaled_position(position),
                0.0,
                cfg.rope_freq_base,
            );
            Ok(())
        })
        .unwrap();

    let actual_q = unsafe { q_buf.as_slice::<f32>()[..q_dim].to_vec() };
    let actual_k = unsafe { k_proj_buf.as_slice::<f32>()[..kv_dim].to_vec() };

    let q_diff = max_abs_diff(&actual_q, &expected_q);
    let k_diff = max_abs_diff(&actual_k, &expected_k);
    assert!(
        k_diff < 1e-4,
        "layer-3 k prep mismatch: q_diff={q_diff} gate_diff={gate_diff} k_diff={k_diff}"
    );
    assert!(
        q_diff < 1e-4,
        "layer-3 q prep mismatch: q_diff={q_diff} gate_diff={gate_diff} k_diff={k_diff}"
    );
}

#[test]
fn test_metal_backend_qwen35_layer3_attention_decode_after_gpu_kv_append_matches_cpu() {
    let _env_lock = lock_env_test();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = crate::model::ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let layer_idx = 3usize;
    let token_id = 42usize;
    let position = 5usize;
    let dim = cfg.embedding_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let mut hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id, &mut hidden)
        .unwrap();
    let attn_norm_w = weights
        .f32_slice(&format!("blk.{layer_idx}.attn_norm.weight"))
        .unwrap();
    let mut normed = vec![0.0f32; dim];
    rms_norm::rms_norm_out(&hidden, attn_norm_w, &mut normed, cfg.rms_norm_eps);

    let q_weight_name = format!("blk.{layer_idx}.attn_q.weight");
    let k_weight_name = format!("blk.{layer_idx}.attn_k.weight");
    let v_weight_name = format!("blk.{layer_idx}.attn_v.weight");
    let (wq_raw, wq_dtype) = weights.raw_with_dtype(&q_weight_name).unwrap();
    let (wk_raw, wk_dtype) = weights.raw_with_dtype(&k_weight_name).unwrap();
    let (wv_raw, wv_dtype) = weights.raw_with_dtype(&v_weight_name).unwrap();

    let mut q_gate_proj = vec![0.0f32; q_dim * 2];
    let mut k_proj = vec![0.0f32; kv_dim];
    let mut v_proj = vec![0.0f32; kv_dim];
    cpu.dequant_matmul(
        wq_raw,
        wq_dtype,
        &normed,
        &mut q_gate_proj,
        q_dim * 2,
        1,
        dim,
    );
    cpu.dequant_matmul(wk_raw, wk_dtype, &normed, &mut k_proj, kv_dim, 1, dim);
    cpu.dequant_matmul(wv_raw, wv_dtype, &normed, &mut v_proj, kv_dim, 1, dim);

    let mut q = vec![0.0f32; q_dim];
    let mut gate = vec![0.0f32; q_dim];
    for head in 0..n_heads {
        let src_head = head * head_dim * 2;
        let dst_head = head * head_dim;
        q[dst_head..dst_head + head_dim]
            .copy_from_slice(&q_gate_proj[src_head..src_head + head_dim]);
        gate[dst_head..dst_head + head_dim]
            .copy_from_slice(&q_gate_proj[src_head + head_dim..src_head + 2 * head_dim]);
    }
    let q_norm_w = weights
        .f32_slice(&format!("blk.{layer_idx}.attn_q_norm.weight"))
        .unwrap();
    let k_norm_w = weights
        .f32_slice(&format!("blk.{layer_idx}.attn_k_norm.weight"))
        .unwrap();
    let mut head_tmp = vec![0.0f32; head_dim];
    for head in 0..n_heads {
        let start = head * head_dim;
        let end = start + head_dim;
        rms_norm::rms_norm_out(&q[start..end], q_norm_w, &mut head_tmp, cfg.rms_norm_eps);
        q[start..end].copy_from_slice(&head_tmp);
    }
    for head in 0..n_kv_heads {
        let start = head * head_dim;
        let end = start + head_dim;
        rms_norm::rms_norm_out(
            &k_proj[start..end],
            k_norm_w,
            &mut head_tmp,
            cfg.rms_norm_eps,
        );
        k_proj[start..end].copy_from_slice(&head_tmp);
    }
    crate::compute::rope::apply_rope_multi_head_neox_partial_scaled(
        &mut q,
        &mut k_proj,
        n_heads,
        n_kv_heads,
        head_dim,
        head_dim.min(64),
        cfg.rope_scaling.scaled_position(position),
        cfg.rope_freq_base,
    );

    let prefix_k: Vec<f32> = (0..kv_dim)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
        .collect();
    let prefix_v: Vec<f32> = (0..kv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let mut all_k = prefix_k.clone();
    all_k.extend_from_slice(&k_proj);
    let mut all_v = prefix_v.clone();
    all_v.extend_from_slice(&v_proj);

    let params = crate::compute::attention::AttentionParams::new(n_heads, n_kv_heads, head_dim);
    let mut expected_attn = vec![0.0f32; q_dim];
    crate::compute::attention::multi_head_attention(
        &q,
        &all_k,
        &all_v,
        &mut expected_attn,
        &params,
        2,
    );
    for (out, gate_val) in expected_attn.iter_mut().zip(gate.iter()) {
        *out /= 1.0 + (-*gate_val).exp();
    }

    backend.ops.init_scratches(&cfg);
    let scratch_guard = backend.ops.scratches();
    let scratches = scratch_guard.as_ref().unwrap();
    let q_buf = MetalBuffer::from_slice(backend.device.device(), &q).unwrap();
    let gate_buf = MetalBuffer::from_slice(backend.device.device(), &gate).unwrap();
    let current_k_buf = MetalBuffer::from_slice(backend.device.device(), &k_proj).unwrap();
    let current_v_buf = MetalBuffer::from_slice(backend.device.device(), &v_proj).unwrap();
    let mut k_cache_init = prefix_k.clone();
    k_cache_init.resize(2 * kv_dim, 0.0);
    let mut v_cache_init = prefix_v.clone();
    v_cache_init.resize(2 * kv_dim, 0.0);
    let k_cache_buf = MetalBuffer::from_slice(backend.device.device(), &k_cache_init).unwrap();
    let v_cache_buf = MetalBuffer::from_slice(backend.device.device(), &v_cache_init).unwrap();
    let out_buf =
        MetalBuffer::new(backend.device.device(), q_dim * std::mem::size_of::<f32>()).unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.elementwise.encode_kv_append(
                encoder,
                &current_k_buf,
                &k_cache_buf,
                false,
                kv_dim as u32,
                kv_dim as u32,
            );
            backend.ops.elementwise.encode_kv_append(
                encoder,
                &current_v_buf,
                &v_cache_buf,
                false,
                kv_dim as u32,
                kv_dim as u32,
            );
            ax_engine_metal::barrier_buffers(encoder);
            backend
                .ops
                .attention
                .encode_attention_decode_with_scratch_and_config(
                    encoder,
                    &q_buf,
                    &k_cache_buf,
                    &v_cache_buf,
                    &out_buf,
                    &scratches.splitk_partial_out,
                    &scratches.splitk_partial_lse,
                    false,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    0,
                    2,
                    backend.ops.attention_dispatch_config(),
                );
            ax_engine_metal::barrier_buffers(encoder);
            backend.ops.elementwise.encode_sigmoid_elementwise_mul(
                encoder,
                &gate_buf,
                &out_buf,
                q_dim as u32,
            );
            Ok(())
        })
        .unwrap();

    let actual_attn = unsafe { out_buf.as_slice::<f32>()[..q_dim].to_vec() };
    let diff = max_abs_diff(&actual_attn, &expected_attn);
    assert!(
        diff < 1e-4,
        "layer-3 attention decode after kv append mismatch: max_diff={diff}"
    );
}

#[test]
fn test_metal_backend_sigmoid_scalar_mul_inplace_matches_cpu() {
    let backend = MetalBackend::new().unwrap();
    let gate = [1.75f32];
    let mut expected = vec![0.5f32, -1.25, 2.0, -0.75, 3.5, -2.25, 1.0, 4.0];
    let scale = 1.0f32 / (1.0 + (-gate[0]).exp());
    for value in &mut expected {
        *value *= scale;
    }

    let gate_buf =
        ax_engine_metal::MetalBuffer::from_slice(backend.ops.device.device(), &gate).unwrap();
    let out_buf = ax_engine_metal::MetalBuffer::from_slice(
        backend.ops.device.device(),
        &[0.5f32, -1.25, 2.0, -0.75, 3.5, -2.25, 1.0, 4.0],
    )
    .unwrap();

    backend
        .ops
        .device
        .execute_sync(|encoder| -> anyhow::Result<()> {
            backend.ops.elementwise.encode_sigmoid_scalar_mul_inplace(
                encoder,
                &gate_buf,
                &out_buf,
                expected.len() as u32,
            );
            Ok(())
        })
        .unwrap();

    let actual = unsafe { out_buf.as_slice::<f32>()[..expected.len()].to_vec() };
    let diff = max_abs_diff(&actual, &expected);
    assert!(
        diff < 1e-6,
        "sigmoid scalar mul mismatch: max_diff={diff}, actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn test_metal_backend_moe_weighted_reduce_slots8_add_matches_cpu() {
    let backend = MetalBackend::new().unwrap();

    let n_tokens = 1usize;
    let n_expert_used = 8usize;
    let dim = 256usize;

    let mut src = vec![0.0f32; n_tokens * n_expert_used * dim];
    for slot in 0..n_expert_used {
        for d in 0..dim {
            src[slot * dim + d] = (slot as f32 + 1.0) * ((d % 17) as f32 - 8.0) * 0.03125;
        }
    }
    let weights = vec![0.03f32, 0.07, 0.11, 0.13, 0.17, 0.19, 0.23, 0.07];
    let mut expected = vec![0.0f32; n_tokens * dim];
    for d in 0..dim {
        let mut acc = 0.0f32;
        for slot in 0..n_expert_used {
            acc += weights[slot] * src[slot * dim + d];
        }
        expected[d] = acc;
    }

    let src_buf = MetalBuffer::from_slice(backend.device.device(), &src).unwrap();
    let weights_buf = MetalBuffer::from_slice(backend.device.device(), &weights).unwrap();
    let dst_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.elementwise.encode_moe_weighted_reduce_slots(
                encoder,
                &src_buf,
                &weights_buf,
                &dst_buf,
                dim as u32,
                n_tokens as u32,
                n_expert_used as u32,
            );
            Ok(())
        })
        .unwrap();

    let actual = unsafe { dst_buf.as_slice::<f32>()[..dim].to_vec() };
    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-5,
        "weighted reduce slots8 mismatch: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual[..8],
        &expected[..8],
    );
}

#[test]
fn test_qwen35_selected_weighted_down_defaults_follow_layout() {
    let _env_lock = lock_env_test();
    let _guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_WEIGHTED_DOWN",
        previous: std::env::var_os("AX_QWEN35_SELECTED_WEIGHTED_DOWN"),
    };
    unsafe { std::env::remove_var("AX_QWEN35_SELECTED_WEIGHTED_DOWN") };

    assert!(qwen35_selected_weighted_down_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q4K,
    ));
    assert!(!qwen35_selected_weighted_down_enabled(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q5K,
    ));
    assert!(qwen35_selected_weighted_down_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q6K,
    ));
    assert!(!qwen35_selected_weighted_down_enabled(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q6K,
    ));
    assert!(qwen35_selected_weighted_down_enabled(
        GgmlType::Q6K,
        GgmlType::Q6K,
        GgmlType::Q6K,
    ));
}

#[test]
fn test_qwen35_selected_weighted_down_env_override_applies_to_q4k_and_q5k() {
    let _env_lock = lock_env_test();

    let _on = EnvVarGuard::set("AX_QWEN35_SELECTED_WEIGHTED_DOWN", "1");
    assert!(qwen35_selected_weighted_down_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q4K,
    ));
    assert!(qwen35_selected_weighted_down_enabled(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q5K,
    ));
    drop(_on);

    let _off = EnvVarGuard::set("AX_QWEN35_SELECTED_WEIGHTED_DOWN", "0");
    assert!(!qwen35_selected_weighted_down_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q4K,
    ));
    assert!(!qwen35_selected_weighted_down_enabled(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q5K,
    ));
}

#[test]
fn test_qwen35_selected_fused_silu_down_q5_k_defaults_on_and_can_be_disabled() {
    let _env_lock = lock_env_test();
    let _guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K",
        previous: std::env::var_os("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K"),
    };
    unsafe { std::env::remove_var("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K") };
    assert!(qwen35_selected_fused_silu_down_q5_k_enabled());

    let _off = EnvVarGuard::set("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K", "0");
    assert!(!qwen35_selected_fused_silu_down_q5_k_enabled());
}

#[test]
fn test_qwen35_selected_fused_silu_down_q5_k_slots8_defaults_on_and_can_be_disabled() {
    let _env_lock = lock_env_test();
    let _guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K_SLOTS8",
        previous: std::env::var_os("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K_SLOTS8"),
    };
    unsafe { std::env::remove_var("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K_SLOTS8") };
    assert!(qwen35_selected_fused_silu_down_q5_k_slots8_enabled());

    let _off = EnvVarGuard::set("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K_SLOTS8", "0");
    assert!(!qwen35_selected_fused_silu_down_q5_k_slots8_enabled());
}

#[test]
fn test_qwen35_selected_fused_silu_down_q5_k_nr2_defaults_off_and_can_be_enabled() {
    let _env_lock = lock_env_test();
    let _guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K_NR2",
        previous: std::env::var_os("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K_NR2"),
    };
    unsafe { std::env::remove_var("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K_NR2") };
    assert!(!qwen35_selected_fused_silu_down_q5_k_nr2_enabled());

    let _on = EnvVarGuard::set("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K_NR2", "1");
    assert!(qwen35_selected_fused_silu_down_q5_k_nr2_enabled());
}

#[test]
fn test_qwen35_selected_q4_k_matvec_defaults_on_and_can_be_disabled() {
    let _env_lock = lock_env_test();
    let _guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_Q4K_MATVEC",
        previous: std::env::var_os("AX_QWEN35_SELECTED_Q4K_MATVEC"),
    };
    unsafe { std::env::remove_var("AX_QWEN35_SELECTED_Q4K_MATVEC") };
    assert!(qwen35_selected_q4_k_matvec_enabled());

    let _off = EnvVarGuard::set("AX_QWEN35_SELECTED_Q4K_MATVEC", "0");
    assert!(!qwen35_selected_q4_k_matvec_enabled());
}

#[test]
fn test_qwen35_selected_q5_k_matvec_defaults_off_and_can_be_enabled() {
    let _env_lock = lock_env_test();
    let _guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_Q5K_MATVEC",
        previous: std::env::var_os("AX_QWEN35_SELECTED_Q5K_MATVEC"),
    };
    unsafe { std::env::remove_var("AX_QWEN35_SELECTED_Q5K_MATVEC") };
    assert!(!qwen35_selected_q5_k_matvec_enabled());

    let _on = EnvVarGuard::set("AX_QWEN35_SELECTED_Q5K_MATVEC", "1");
    assert!(qwen35_selected_q5_k_matvec_enabled());
}

#[test]
fn test_qwen35_selected_pair_q4_k_matvec_defaults_on_and_can_be_disabled() {
    let _env_lock = lock_env_test();
    let _guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC",
        previous: std::env::var_os("AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC"),
    };
    unsafe { std::env::remove_var("AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC") };
    assert!(qwen35_selected_pair_q4_k_matvec_enabled());

    let _off = EnvVarGuard::set("AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC", "0");
    assert!(!qwen35_selected_pair_q4_k_matvec_enabled());
}

#[test]
fn test_qwen35_selected_pair_q5_k_matvec_defaults_off_and_can_be_enabled() {
    let _env_lock = lock_env_test();
    let _guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC",
        previous: std::env::var_os("AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC"),
    };
    unsafe { std::env::remove_var("AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC") };
    assert!(!qwen35_selected_pair_q5_k_matvec_enabled_for_layout(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q8_0,
    ));

    let _on = EnvVarGuard::set("AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC", "1");
    assert!(qwen35_selected_pair_q5_k_matvec_enabled_for_layout(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q8_0,
    ));
}

#[test]
fn test_qwen35_selected_pair_q5_k_matvec_layout_default_targets_q5_q5_q6() {
    let _env_lock = lock_env_test();
    let _guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC",
        previous: std::env::var_os("AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC"),
    };
    unsafe { std::env::remove_var("AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC") };

    assert!(qwen35_selected_pair_q5_k_matvec_enabled_for_layout(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q6K,
    ));
    assert!(!qwen35_selected_pair_q5_k_matvec_enabled_for_layout(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q8_0,
    ));

    let _off = EnvVarGuard::set("AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC", "0");
    assert!(!qwen35_selected_pair_q5_k_matvec_enabled_for_layout(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q6K,
    ));
}

#[test]
fn test_qwen35_selected_single_token_gate_up_path_supports_mixed_qwen3_coder_layouts() {
    assert!(qwen35_selected_single_token_gate_up_path_supported(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q6K,
    ));
    assert!(qwen35_selected_single_token_gate_up_path_supported(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q8_0,
    ));
    assert!(qwen35_selected_single_token_gate_up_path_supported(
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
    ));
    assert!(qwen35_selected_single_token_gate_up_path_supported(
        GgmlType::Q6K,
        GgmlType::Q5K,
        GgmlType::Q6K,
    ));
    assert!(qwen35_selected_single_token_gate_up_path_supported(
        GgmlType::Q4K,
        GgmlType::Q6K,
        GgmlType::Q6K,
    ));
    assert!(qwen35_selected_single_token_gate_up_path_supported(
        GgmlType::Q6K,
        GgmlType::Q6K,
        GgmlType::Q6K,
    ));
    assert!(qwen35_selected_single_token_gate_up_path_supported(
        GgmlType::Q8_0,
        GgmlType::Q8_0,
        GgmlType::Q8_0,
    ));
    assert!(!qwen35_selected_single_token_gate_up_path_supported(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::F32,
    ));
}

#[test]
fn test_qwen35_selected_single_token_down_support_distinguishes_selected_vs_fallback() {
    assert!(qwen35_selected_single_token_down_supported(GgmlType::Q4K));
    assert!(qwen35_selected_single_token_down_supported(GgmlType::Q5K));
    assert!(qwen35_selected_single_token_down_supported(GgmlType::Q6K));
    assert!(qwen35_selected_single_token_down_supported(GgmlType::Q8_0));

    assert!(!qwen35_selected_single_token_down_falls_back_to_mul_mat_id(
        GgmlType::Q4K,
    ));
    assert!(!qwen35_selected_single_token_down_falls_back_to_mul_mat_id(
        GgmlType::Q5K,
    ));
    assert!(qwen35_selected_single_token_down_falls_back_to_mul_mat_id(
        GgmlType::Q6K,
    ));
    assert!(qwen35_selected_single_token_down_falls_back_to_mul_mat_id(
        GgmlType::Q8_0,
    ));
}

#[test]
fn test_qwen35_selected_single_token_default_enables_mixed_down_with_weighted_path() {
    let _env_lock = lock_env_test();
    let _pair_q4_guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC",
        previous: std::env::var_os("AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC"),
    };
    let _pair_q5_guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC",
        previous: std::env::var_os("AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC"),
    };
    unsafe { std::env::remove_var("AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC") };
    unsafe { std::env::remove_var("AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC") };

    assert!(qwen35_selected_single_token_default_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q6K,
    ));
    assert!(qwen35_selected_single_token_default_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q8_0,
    ));
    assert!(qwen35_selected_single_token_default_enabled(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q5K,
    ));
    assert!(qwen35_selected_single_token_default_enabled(
        GgmlType::Q6K,
        GgmlType::Q6K,
        GgmlType::Q6K,
    ));
    assert!(qwen35_selected_single_token_default_enabled(
        GgmlType::Q8_0,
        GgmlType::Q8_0,
        GgmlType::Q8_0,
    ));
    assert!(qwen35_selected_single_token_default_enabled(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q6K,
    ));
    assert!(!qwen35_selected_single_token_default_enabled(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q8_0,
    ));
}

#[test]
fn test_qwen35_selected_weighted_down_enabled_defaults_include_q6k_and_q8_0() {
    let _env_lock = lock_env_test();
    let _guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_WEIGHTED_DOWN",
        previous: std::env::var_os("AX_QWEN35_SELECTED_WEIGHTED_DOWN"),
    };
    unsafe { std::env::remove_var("AX_QWEN35_SELECTED_WEIGHTED_DOWN") };

    assert!(qwen35_selected_weighted_down_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q4K,
    ));
    assert!(!qwen35_selected_weighted_down_enabled(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q5K,
    ));
    assert!(qwen35_selected_weighted_down_enabled(
        GgmlType::Q6K,
        GgmlType::Q6K,
        GgmlType::Q6K,
    ));
    assert!(qwen35_selected_weighted_down_enabled(
        GgmlType::Q8_0,
        GgmlType::Q8_0,
        GgmlType::Q8_0,
    ));
}

#[test]
fn test_qwen35_selected_single_token_default_stays_on_when_q4_pair_env_is_off() {
    let _env_lock = lock_env_test();
    let _off = EnvVarGuard::set("AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC", "0");
    assert!(qwen35_selected_single_token_default_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q6K,
    ));
}

#[test]
fn test_qwen35_selected_expert_pair_defaults_on_for_q4k_gate_up_when_q5k_fused_down_is_on() {
    let _env_lock = lock_env_test();
    let _pair_guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_EXPERT_PAIR",
        previous: std::env::var_os("AX_QWEN35_SELECTED_EXPERT_PAIR"),
    };
    let _fused_guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K",
        previous: std::env::var_os("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K"),
    };
    let _pair_q4_guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC",
        previous: std::env::var_os("AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC"),
    };
    unsafe {
        std::env::remove_var("AX_QWEN35_SELECTED_EXPERT_PAIR");
        std::env::remove_var("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K");
        std::env::remove_var("AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC");
    }

    assert!(qwen35_selected_expert_pair_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q5K,
    ));
    assert!(qwen35_selected_expert_pair_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q4K,
    ));

    let _off = EnvVarGuard::set("AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC", "0");
    assert!(!qwen35_selected_expert_pair_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q5K,
    ));
}

#[test]
fn test_qwen35_selected_expert_pair_env_override_still_applies() {
    let _env_lock = lock_env_test();
    let _pair_guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_EXPERT_PAIR",
        previous: std::env::var_os("AX_QWEN35_SELECTED_EXPERT_PAIR"),
    };
    let _fused_guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K",
        previous: std::env::var_os("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K"),
    };
    unsafe {
        std::env::remove_var("AX_QWEN35_SELECTED_EXPERT_PAIR");
        std::env::remove_var("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K");
    }

    let _on = EnvVarGuard::set("AX_QWEN35_SELECTED_EXPERT_PAIR", "1");
    assert!(qwen35_selected_expert_pair_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q5K,
    ));
    drop(_on);

    let _off = EnvVarGuard::set("AX_QWEN35_SELECTED_EXPERT_PAIR", "0");
    assert!(!qwen35_selected_expert_pair_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q4K,
    ));
}

#[test]
fn test_qwen35_selected_expert_pair_defaults_on_for_matching_full_quant_gate_up() {
    assert!(qwen35_selected_expert_pair_enabled(
        GgmlType::Q6K,
        GgmlType::Q6K,
        GgmlType::Q6K,
    ));
    assert!(qwen35_selected_expert_pair_enabled(
        GgmlType::Q8_0,
        GgmlType::Q8_0,
        GgmlType::Q8_0,
    ));
    assert!(!qwen35_selected_expert_pair_enabled(
        GgmlType::Q6K,
        GgmlType::Q8_0,
        GgmlType::Q8_0,
    ));
}

#[test]
fn test_qwen35_shared_gate_inp_fused_defaults_on_and_can_be_disabled() {
    let _env_lock = lock_env_test();
    let _guard = EnvVarGuard {
        key: "AX_QWEN35_SHARED_GATE_INP_FUSED",
        previous: std::env::var_os("AX_QWEN35_SHARED_GATE_INP_FUSED"),
    };
    unsafe { std::env::remove_var("AX_QWEN35_SHARED_GATE_INP_FUSED") };
    assert!(qwen35_shared_gate_inp_fused_enabled());

    let _off = EnvVarGuard::set("AX_QWEN35_SHARED_GATE_INP_FUSED", "0");
    assert!(!qwen35_shared_gate_inp_fused_enabled());
}

#[test]
fn test_metal_backend_real_qwen35_35b_a3b_layer3_moe_resident_matches_cpu() {
    let _env_lock = lock_env_test();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = crate::model::ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let layer_idx = 3usize;
    let prefix = format!("blk.{layer_idx}");
    let token_id = 42usize;
    let position = 5usize;
    let dim = cfg.embedding_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let n_expert = cfg.n_expert.unwrap() as usize;
    let n_expert_used = cfg.n_expert_used.unwrap() as usize;

    let tensor_output_rows = |name: &str| -> usize {
        match weights.info(name).unwrap().shape.as_slice() {
            [_input_dim] => 1,
            [_input_dim, output_dim, ..] => *output_dim as usize,
            other => panic!("unexpected tensor shape for {name}: {other:?}"),
        }
    };
    let apply_shared_gate = |down: &mut [f32], gate: &[f32], gate_name: &str| match gate.len() {
        0 => {}
        1 => {
            let scale = 1.0 / (1.0 + (-gate[0]).exp());
            for value in down.iter_mut() {
                *value *= scale;
            }
        }
        len if len == down.len() => {
            for (value, gate_value) in down.iter_mut().zip(gate.iter()) {
                *value *= 1.0 / (1.0 + (-*gate_value).exp());
            }
        }
        other => panic!(
            "unsupported shared expert gate width for {gate_name}: expected 1 or {}, got {other}",
            down.len()
        ),
    };

    let mut hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id, &mut hidden)
        .unwrap();
    let attn_norm_w = weights
        .f32_slice(&format!("{prefix}.attn_norm.weight"))
        .unwrap();
    let mut normed = vec![0.0f32; dim];
    rms_norm::rms_norm_out(&hidden, attn_norm_w, &mut normed, cfg.rms_norm_eps);

    let (wq_raw, wq_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
        .unwrap();
    let (wk_raw, wk_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
        .unwrap();
    let (wv_raw, wv_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_v.weight"))
        .unwrap();

    let mut q_gate_proj = vec![0.0f32; q_dim * 2];
    let mut k_proj = vec![0.0f32; kv_dim];
    let mut v_proj = vec![0.0f32; kv_dim];
    cpu.dequant_matmul(
        wq_raw,
        wq_dtype,
        &normed,
        &mut q_gate_proj,
        q_dim * 2,
        1,
        dim,
    );
    cpu.dequant_matmul(wk_raw, wk_dtype, &normed, &mut k_proj, kv_dim, 1, dim);
    cpu.dequant_matmul(wv_raw, wv_dtype, &normed, &mut v_proj, kv_dim, 1, dim);

    let mut q = vec![0.0f32; q_dim];
    let mut gate = vec![0.0f32; q_dim];
    for head in 0..n_heads {
        let src_head = head * head_dim * 2;
        let dst_head = head * head_dim;
        q[dst_head..dst_head + head_dim]
            .copy_from_slice(&q_gate_proj[src_head..src_head + head_dim]);
        gate[dst_head..dst_head + head_dim]
            .copy_from_slice(&q_gate_proj[src_head + head_dim..src_head + 2 * head_dim]);
    }
    let q_norm_w = weights
        .f32_slice(&format!("{prefix}.attn_q_norm.weight"))
        .unwrap();
    let k_norm_w = weights
        .f32_slice(&format!("{prefix}.attn_k_norm.weight"))
        .unwrap();
    let mut head_tmp = vec![0.0f32; head_dim];
    for head in 0..n_heads {
        let start = head * head_dim;
        let end = start + head_dim;
        rms_norm::rms_norm_out(&q[start..end], q_norm_w, &mut head_tmp, cfg.rms_norm_eps);
        q[start..end].copy_from_slice(&head_tmp);
    }
    for head in 0..n_kv_heads {
        let start = head * head_dim;
        let end = start + head_dim;
        rms_norm::rms_norm_out(
            &k_proj[start..end],
            k_norm_w,
            &mut head_tmp,
            cfg.rms_norm_eps,
        );
        k_proj[start..end].copy_from_slice(&head_tmp);
    }
    crate::compute::rope::apply_rope_multi_head_neox_partial_scaled(
        &mut q,
        &mut k_proj,
        n_heads,
        n_kv_heads,
        head_dim,
        head_dim.min(64),
        cfg.rope_scaling.scaled_position(position),
        cfg.rope_freq_base,
    );

    let prefix_k: Vec<f32> = (0..kv_dim)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
        .collect();
    let prefix_v: Vec<f32> = (0..kv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let mut all_k = prefix_k.clone();
    all_k.extend_from_slice(&k_proj);
    let mut all_v = prefix_v.clone();
    all_v.extend_from_slice(&v_proj);

    let params = crate::compute::attention::AttentionParams::new(n_heads, n_kv_heads, head_dim);
    let mut attn_out = vec![0.0f32; q_dim];
    crate::compute::attention::multi_head_attention(&q, &all_k, &all_v, &mut attn_out, &params, 2);
    for (out, gate_val) in attn_out.iter_mut().zip(gate.iter()) {
        *out *= 1.0 / (1.0 + (-*gate_val).exp());
    }

    let (wo_raw, wo_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_output.weight"))
        .unwrap();
    let mut proj = vec![0.0f32; dim];
    cpu.dequant_matmul(wo_raw, wo_dtype, &attn_out, &mut proj, dim, 1, q_dim);

    let mut hidden_after_attn = hidden.clone();
    for (dst, src) in hidden_after_attn.iter_mut().zip(proj.iter()) {
        *dst += *src;
    }

    let ffn_norm_w = weights
        .f32_slice(&format!("{prefix}.post_attention_norm.weight"))
        .unwrap();
    let router_name = format!("{prefix}.ffn_gate_inp.weight");
    let gate_name = format!("{prefix}.ffn_gate_exps.weight");
    let up_name = format!("{prefix}.ffn_up_exps.weight");
    let down_name = format!("{prefix}.ffn_down_exps.weight");
    let shared_gate_name = format!("{prefix}.ffn_gate_shexp.weight");
    let shared_up_name = format!("{prefix}.ffn_up_shexp.weight");
    let shared_down_name = format!("{prefix}.ffn_down_shexp.weight");
    let shared_gate_inp_name = format!("{prefix}.ffn_gate_inp_shexp.weight");

    let expert_inter_dim = tensor_output_rows(&gate_name);
    let (router_raw, router_dtype) = weights.raw_with_dtype(&router_name).unwrap();
    let (gate_raw, gate_dtype) = weights.raw_with_dtype(&gate_name).unwrap();
    let (up_raw, up_dtype) = weights.raw_with_dtype(&up_name).unwrap();
    let (down_raw, down_dtype) = weights.raw_with_dtype(&down_name).unwrap();
    let gate_stride =
        crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
    let up_stride = crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
    let down_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);

    let mut norm_buf = vec![0.0f32; dim];
    rms_norm::rms_norm_out(
        &hidden_after_attn,
        ffn_norm_w,
        &mut norm_buf,
        cfg.rms_norm_eps,
    );
    let mut router_logits = vec![0.0f32; n_expert];
    cpu.dequant_matmul(
        router_raw,
        router_dtype,
        &norm_buf,
        &mut router_logits,
        n_expert,
        1,
        dim,
    );
    let (expert_ids, expert_weights) =
        crate::model::moe_utils::top_k_softmax(&router_logits, n_expert_used);

    let shared_inter_dim = if weights.has(&shared_gate_name) {
        tensor_output_rows(&shared_gate_name)
    } else {
        1usize
    };
    let max_inter_dim = expert_inter_dim.max(shared_inter_dim);
    let mut gate_work = vec![0.0f32; max_inter_dim];
    let mut up_work = vec![0.0f32; max_inter_dim];
    let mut down_work = vec![0.0f32; dim];
    let mut expert_accum = vec![0.0f32; dim];

    for (slot, &eid) in expert_ids.iter().enumerate() {
        let gate_slice = &gate_raw[eid * gate_stride..(eid + 1) * gate_stride];
        let up_slice = &up_raw[eid * up_stride..(eid + 1) * up_stride];
        let down_slice = &down_raw[eid * down_stride..(eid + 1) * down_stride];
        cpu.dequant_matmul(
            gate_slice,
            gate_dtype,
            &norm_buf,
            &mut gate_work[..expert_inter_dim],
            expert_inter_dim,
            1,
            dim,
        );
        cpu.dequant_matmul(
            up_slice,
            up_dtype,
            &norm_buf,
            &mut up_work[..expert_inter_dim],
            expert_inter_dim,
            1,
            dim,
        );
        crate::compute::silu::silu_elementwise_mul(
            &mut gate_work[..expert_inter_dim],
            &up_work[..expert_inter_dim],
        );
        cpu.dequant_matmul(
            down_slice,
            down_dtype,
            &gate_work[..expert_inter_dim],
            &mut down_work,
            dim,
            1,
            expert_inter_dim,
        );
        for (dst, src) in expert_accum.iter_mut().zip(down_work.iter()) {
            *dst += expert_weights[slot] * *src;
        }
    }

    if weights.has(&shared_gate_name) {
        let (shared_gate_raw, shared_gate_dtype) =
            weights.raw_with_dtype(&shared_gate_name).unwrap();
        let (shared_up_raw, shared_up_dtype) = weights.raw_with_dtype(&shared_up_name).unwrap();
        let (shared_down_raw, shared_down_dtype) =
            weights.raw_with_dtype(&shared_down_name).unwrap();
        cpu.dequant_matmul(
            shared_gate_raw,
            shared_gate_dtype,
            &norm_buf,
            &mut gate_work[..shared_inter_dim],
            shared_inter_dim,
            1,
            dim,
        );
        cpu.dequant_matmul(
            shared_up_raw,
            shared_up_dtype,
            &norm_buf,
            &mut up_work[..shared_inter_dim],
            shared_inter_dim,
            1,
            dim,
        );
        crate::compute::silu::silu_elementwise_mul(
            &mut gate_work[..shared_inter_dim],
            &up_work[..shared_inter_dim],
        );
        cpu.dequant_matmul(
            shared_down_raw,
            shared_down_dtype,
            &gate_work[..shared_inter_dim],
            &mut down_work,
            dim,
            1,
            shared_inter_dim,
        );
        if weights.has(&shared_gate_inp_name) {
            let gate_rows = tensor_output_rows(&shared_gate_inp_name);
            let (shared_gate_inp_raw, shared_gate_inp_dtype) =
                weights.raw_with_dtype(&shared_gate_inp_name).unwrap();
            let mut shared_gate = vec![0.0f32; gate_rows];
            cpu.dequant_matmul(
                shared_gate_inp_raw,
                shared_gate_inp_dtype,
                &norm_buf,
                &mut shared_gate,
                gate_rows,
                1,
                dim,
            );
            apply_shared_gate(&mut down_work, &shared_gate, &shared_gate_inp_name);
        }
        for (dst, src) in expert_accum.iter_mut().zip(down_work.iter()) {
            *dst += *src;
        }
    }

    let mut expected_hidden = hidden_after_attn.clone();
    for (dst, src) in expected_hidden.iter_mut().zip(expert_accum.iter()) {
        *dst += *src;
    }

    let hidden_buf = MetalBuffer::from_slice(backend.device.device(), &hidden_after_attn).unwrap();
    let ffn_norm_w_buf = MetalBuffer::from_slice(backend.device.device(), ffn_norm_w).unwrap();
    let router_buf = MetalBuffer::from_bytes(backend.device.device(), router_raw).unwrap();
    let gate_buf = MetalBuffer::from_bytes(backend.device.device(), gate_raw).unwrap();
    let up_buf = MetalBuffer::from_bytes(backend.device.device(), up_raw).unwrap();
    let down_buf = MetalBuffer::from_bytes(backend.device.device(), down_raw).unwrap();
    let shared_gate_buf = weights.has(&shared_gate_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_gate_name).unwrap();
        MetalBuffer::from_bytes(backend.device.device(), raw).unwrap()
    });
    let shared_up_buf = weights.has(&shared_up_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_up_name).unwrap();
        MetalBuffer::from_bytes(backend.device.device(), raw).unwrap()
    });
    let shared_down_buf = weights.has(&shared_down_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_down_name).unwrap();
        MetalBuffer::from_bytes(backend.device.device(), raw).unwrap()
    });
    let shared_gate_inp_buf = weights.has(&shared_gate_inp_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_gate_inp_name).unwrap();
        MetalBuffer::from_bytes(backend.device.device(), raw).unwrap()
    });
    let shared_expert = if let (Some(gate), Some(up), Some(down)) = (
        shared_gate_buf.as_ref(),
        shared_up_buf.as_ref(),
        shared_down_buf.as_ref(),
    ) {
        let (_, shared_dtype) = weights.raw_with_dtype(&shared_gate_name).unwrap();
        let gate_inp_dtype = if weights.has(&shared_gate_inp_name) {
            Some(weights.raw_with_dtype(&shared_gate_inp_name).unwrap().1)
        } else {
            None
        };
        Some(SharedExpertCachedBuffers {
            gate,
            up,
            down,
            gate_inp: shared_gate_inp_buf.as_ref(),
            gate_inp_dtype,
            dtype: shared_dtype,
            inter_dim: shared_inter_dim,
            gate_inp_rows: if weights.has(&shared_gate_inp_name) {
                tensor_output_rows(&shared_gate_inp_name)
            } else {
                0
            },
        })
    } else {
        None
    };

    backend.ops.init_batch_scratches(&cfg, 1);
    backend
        .ops
        .moe_ffn_gpu_resident_cached(
            &hidden_buf,
            &ffn_norm_w_buf,
            &router_buf,
            router_dtype,
            &gate_buf,
            gate_dtype,
            &up_buf,
            up_dtype,
            &down_buf,
            down_dtype,
            1,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            gate_stride,
            up_stride,
            down_stride,
            cfg.rms_norm_eps,
            shared_expert.as_ref(),
        )
        .unwrap();

    let actual_hidden = unsafe { hidden_buf.as_slice::<f32>()[..dim].to_vec() };
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 5e-3,
        "real Qwen3.5-35B-A3B layer-3 resident MoE mismatch: rel_diff={}, max_diff={diff}, expert_ids={expert_ids:?}, expert_weights={expert_weights:?}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );
}

#[test]
fn test_metal_backend_real_qwen35_35b_a3b_layer4_native_recurrent_projection_matches_cpu() {
    let _env_lock = lock_env_test();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = crate::model::ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let layer_idx = 4usize;
    assert!(cfg.qwen35_is_recurrent_layer(layer_idx));

    let dims = Qwen3_5Forward::recurrent_dims(&cfg).unwrap();
    let dim = cfg.embedding_dim as usize;
    let recurrent_slot = 0usize;
    let prefix = format!("blk.{layer_idx}");
    let recurrent_slot_indices = [recurrent_slot];
    let dequant_dispatch = backend.ops.dequant_dispatch_config();

    let mut norm_buf = vec![0.0f32; dim];
    for (i, value) in norm_buf.iter_mut().enumerate() {
        *value = ((i % 31) as f32 - 15.0) * 0.041;
    }

    let make_kv = || {
        crate::kv::Qwen3_5Kv::new(
            cfg.n_layers as usize,
            cfg.n_kv_heads as usize,
            cfg.head_dim as usize,
            8,
            cfg.qwen35_full_attention_interval.unwrap() as usize,
            cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
            cfg.qwen35_ssm_inner_size.unwrap() as usize,
            cfg.qwen35_ssm_state_size.unwrap() as usize,
            cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
            cfg.qwen35_ssm_group_count.unwrap() as usize,
        )
    };

    let mut expected_kv = make_kv();
    let mut actual_kv = make_kv();
    let mut conv_state = vec![
        0.0f32;
        expected_kv
            .conv_state_for_slot(recurrent_slot, layer_idx)
            .len()
    ];
    for (i, value) in conv_state.iter_mut().enumerate() {
        *value = ((i % 19) as f32 - 9.0) * 0.017;
    }
    let mut recurrent_state = vec![
        0.0f32;
        expected_kv
            .recurrent_state_for_slot(recurrent_slot, layer_idx)
            .len()
    ];
    for (i, value) in recurrent_state.iter_mut().enumerate() {
        *value = ((i % 23) as f32 - 11.0) * 0.009;
    }
    expected_kv
        .conv_state_for_slot_mut(recurrent_slot, layer_idx)
        .copy_from_slice(&conv_state);
    expected_kv
        .recurrent_state_for_slot_mut(recurrent_slot, layer_idx)
        .copy_from_slice(&recurrent_state);
    actual_kv
        .conv_state_for_slot_mut(recurrent_slot, layer_idx)
        .copy_from_slice(&conv_state);
    actual_kv
        .recurrent_state_for_slot_mut(recurrent_slot, layer_idx)
        .copy_from_slice(&recurrent_state);

    let input_ops = Qwen3_5Forward::recurrent_input_ops(&weights, &prefix, dims).unwrap();
    let mut expected_qkv = vec![0.0f32; dims.conv_dim()];
    let mut expected_z = vec![0.0f32; dims.inner_size];
    let mut expected_beta = vec![0.0f32; dims.time_step_rank];
    let mut expected_alpha = vec![0.0f32; dims.time_step_rank];
    {
        let mut outputs = [
            &mut expected_qkv[..],
            &mut expected_z[..],
            &mut expected_beta[..],
            &mut expected_alpha[..],
        ];
        Qwen3_5Forward::decode_project_ops_gpu_safe(&cpu, &input_ops, &norm_buf, dim, &mut outputs);
    }
    let projected_beta = expected_beta.clone();
    let projected_alpha = expected_alpha.clone();

    let mut expected_rec_out_raw = vec![0.0f32; dims.inner_size];
    let runtime = Qwen3_5Forward::run_recurrent_sequence(
        &cpu,
        &weights,
        &prefix,
        &mut expected_kv,
        layer_idx,
        &recurrent_slot_indices,
        dims,
        cfg.rms_norm_eps,
        &expected_qkv,
        &mut expected_beta,
        &mut expected_alpha,
        &mut expected_rec_out_raw,
        1,
    )
    .unwrap();

    let recurrent_dtypes = Qwen3_5NativeRecurrentDtypes {
        wqkv: input_ops[0].1,
        wgate: input_ops[1].1,
        wbeta: input_ops[2].1,
        walpha: input_ops[3].1,
        wssm_out: Qwen3_5Forward::recurrent_output_op(&weights, &prefix, dim)
            .unwrap()
            .1,
    };

    let mut expected_rec_out_normed = expected_rec_out_raw.clone();
    for head in 0..dims.time_step_rank {
        let start = head * dims.state_size;
        let end = start + dims.state_size;
        rms_norm::rms_norm(
            &mut expected_rec_out_normed[start..end],
            runtime.ssm_norm,
            cfg.rms_norm_eps,
        );
    }
    let mut expected_rec_out_gated = expected_rec_out_normed.clone();
    let mut z_gate = expected_z.clone();
    crate::compute::silu::silu(&mut z_gate);
    crate::compute::silu::elementwise_mul(&mut expected_rec_out_gated, &z_gate);

    let (wssm_out_raw, wssm_out_dtype, _) =
        Qwen3_5Forward::recurrent_output_op(&weights, &prefix, dim).unwrap();
    let mut expected_proj = vec![0.0f32; dim];
    cpu.dequant_matmul(
        wssm_out_raw,
        wssm_out_dtype,
        &expected_rec_out_gated,
        &mut expected_proj,
        dim,
        1,
        dims.inner_size,
    );

    let mut safe_kv = make_kv();
    safe_kv
        .conv_state_for_slot_mut(recurrent_slot, layer_idx)
        .copy_from_slice(&conv_state);
    safe_kv
        .recurrent_state_for_slot_mut(recurrent_slot, layer_idx)
        .copy_from_slice(&recurrent_state);
    let mut safe_beta = projected_beta.clone();
    let mut safe_alpha = projected_alpha.clone();
    let mut safe_rec_out = vec![0.0f32; dims.inner_size];
    let conv_cache_len = actual_kv.conv_cache_len();
    backend.qwen35_recurrent_sequence_for_kv(
        &expected_qkv,
        &mut safe_beta,
        &mut safe_alpha,
        runtime.dt_bias,
        runtime.a,
        runtime.conv_kernel,
        &mut safe_kv,
        layer_idx,
        &recurrent_slot_indices,
        &mut safe_rec_out,
        1,
        crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len,
            conv_dim: dims.conv_dim(),
            group_count: dims.group_count,
            state_size: dims.state_size,
            time_step_rank: dims.time_step_rank,
            rms_norm_eps: cfg.rms_norm_eps,
        },
    );

    let (wqkv_raw, wqkv_dtype, _) = input_ops[0];
    let (wgate_raw, wgate_dtype, _) = input_ops[1];
    let (wbeta_raw, wbeta_dtype, _) = input_ops[2];
    let (walpha_raw, walpha_dtype, _) = input_ops[3];
    assert_eq!(wqkv_dtype, GgmlType::Q8_0);
    assert_eq!(wgate_dtype, GgmlType::Q8_0);
    assert_eq!(wbeta_dtype, GgmlType::Q8_0);
    assert_eq!(walpha_dtype, GgmlType::Q8_0);

    let norm_gpu = MetalBuffer::from_slice(backend.device.device(), &norm_buf).unwrap();
    let qkv_gpu = MetalBuffer::new(
        backend.device.device(),
        dims.conv_dim() * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let z_gpu = MetalBuffer::new(
        backend.device.device(),
        dims.inner_size * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let beta_gpu = MetalBuffer::new(
        backend.device.device(),
        dims.time_step_rank * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let alpha_gpu = MetalBuffer::new(
        backend.device.device(),
        dims.time_step_rank * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let recurrent_out_gpu = MetalBuffer::new(
        backend.device.device(),
        dims.inner_size * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let proj_gpu =
        MetalBuffer::new(backend.device.device(), dim * std::mem::size_of::<f32>()).unwrap();

    let wqkv_buf = MetalBuffer::from_bytes(backend.device.device(), wqkv_raw).unwrap();
    let wgate_buf = MetalBuffer::from_bytes(backend.device.device(), wgate_raw).unwrap();
    let wbeta_buf = MetalBuffer::from_bytes(backend.device.device(), wbeta_raw).unwrap();
    let walpha_buf = MetalBuffer::from_bytes(backend.device.device(), walpha_raw).unwrap();
    let conv_kernel_buf =
        MetalBuffer::from_slice(backend.device.device(), runtime.conv_kernel).unwrap();
    let ssm_norm_buf = MetalBuffer::from_slice(backend.device.device(), runtime.ssm_norm).unwrap();
    let dt_bias_buf = MetalBuffer::from_slice(backend.device.device(), runtime.dt_bias).unwrap();
    let ssm_a_buf = MetalBuffer::from_slice(backend.device.device(), runtime.a).unwrap();
    let wssm_out_buf = MetalBuffer::from_bytes(backend.device.device(), wssm_out_raw).unwrap();
    let recurrent_weights = Qwen3_5NativeRecurrentCachedWeights {
        wqkv: &wqkv_buf,
        wgate: &wgate_buf,
        wbeta: &wbeta_buf,
        walpha: &walpha_buf,
        conv_kernel: &conv_kernel_buf,
        ssm_norm: &ssm_norm_buf,
        dt_bias: &dt_bias_buf,
        ssm_a: &ssm_a_buf,
        wssm_out: &wssm_out_buf,
    };
    let recurrent_scratch = Qwen3_5NativeRecurrentProjectionScratch {
        qkv: &qkv_gpu,
        z: &z_gpu,
        beta: &beta_gpu,
        alpha: &alpha_gpu,
    };

    let conv_state_stride = actual_kv.conv_cache_len() * actual_kv.conv_dim();
    let recurrent_state_stride = actual_kv.recurrent_state_len();
    let _ = backend
        .ops
        .sync_qwen35_slot_buffers_from_kv(&actual_kv, layer_idx, recurrent_slot);

    backend
        .device
        .execute_sync(|encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(
                encoder,
                crate::model::execution_plan::DecodeBarrierPlan::Explicit,
            );
            backend.ops.with_qwen35_recurrent_slot_buffer_for_kv(
                &actual_kv,
                layer_idx,
                recurrent_slot,
                conv_state_stride,
                recurrent_state_stride,
                |slot_buffers| -> anyhow::Result<()> {
                    Qwen3_5Forward::encode_qwen35_native_recurrent_inputs(
                        &backend.ops,
                        encoder,
                        &barrier,
                        recurrent_weights,
                        recurrent_scratch,
                        &norm_gpu,
                        dims,
                        dim,
                        recurrent_dtypes,
                        dequant_dispatch,
                    );
                    Qwen3_5Forward::encode_qwen35_single_token_recurrent_core(
                        &backend.ops,
                        encoder,
                        &barrier,
                        recurrent_scratch.qkv,
                        recurrent_scratch.alpha,
                        recurrent_scratch.beta,
                        recurrent_weights.conv_kernel,
                        slot_buffers,
                        &recurrent_out_gpu,
                        conv_cache_len,
                        dims,
                        cfg.rms_norm_eps,
                    )?;
                    barrier.flush();
                    Ok(())
                },
            )
        })
        .unwrap();

    let actual_qkv = unsafe { qkv_gpu.as_slice::<f32>()[..dims.conv_dim()].to_vec() };
    let actual_beta = unsafe { beta_gpu.as_slice::<f32>()[..dims.time_step_rank].to_vec() };
    let actual_alpha = unsafe { alpha_gpu.as_slice::<f32>()[..dims.time_step_rank].to_vec() };
    let actual_recurrent_core =
        unsafe { recurrent_out_gpu.as_slice::<f32>()[..dims.inner_size].to_vec() };

    backend
        .device
        .execute_sync(|encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(
                encoder,
                crate::model::execution_plan::DecodeBarrierPlan::Explicit,
            );
            backend.ops.elementwise.encode_per_head_rms_norm_batch(
                encoder,
                &recurrent_out_gpu,
                recurrent_weights.ssm_norm,
                1,
                dims.time_step_rank as u32,
                dims.state_size as u32,
                cfg.rms_norm_eps,
            );
            barrier.step(encoder);
            backend.ops.elementwise.encode_silu_elementwise_mul_batch(
                encoder,
                &z_gpu,
                &recurrent_out_gpu,
                dims.inner_size as u32,
                1,
            );
            barrier.step(encoder);
            Qwen3_5Forward::encode_qwen35_native_recurrent_out_proj(
                &backend.ops,
                encoder,
                &barrier,
                recurrent_weights,
                &z_gpu,
                &proj_gpu,
                dim,
                dims,
                recurrent_dtypes,
                dequant_dispatch,
            );
            barrier.flush();
            Ok(())
        })
        .unwrap();

    let actual_z = unsafe { z_gpu.as_slice::<f32>()[..dims.inner_size].to_vec() };
    let actual_recurrent_out =
        unsafe { recurrent_out_gpu.as_slice::<f32>()[..dims.inner_size].to_vec() };
    let actual_proj = unsafe { proj_gpu.as_slice::<f32>()[..dim].to_vec() };

    let qkv_diff = max_abs_diff(&actual_qkv, &expected_qkv);
    let recurrent_core_diff = max_abs_diff(&actual_recurrent_core, &expected_rec_out_raw);
    let safe_core_diff = max_abs_diff(&safe_rec_out, &expected_rec_out_raw);
    let native_vs_safe_core_diff = max_abs_diff(&actual_recurrent_core, &safe_rec_out);
    let z_diff = max_abs_diff(&actual_z, &expected_rec_out_gated);
    let beta_diff = max_abs_diff(&actual_beta, &expected_beta);
    let alpha_diff = max_abs_diff(&actual_alpha, &expected_alpha);
    let recurrent_out_diff = max_abs_diff(&actual_recurrent_out, &expected_rec_out_normed);
    let proj_diff = max_abs_diff(&actual_proj, &expected_proj);

    assert!(qkv_diff < 5e-2, "layer-4 qkv mismatch: max_diff={qkv_diff}");
    assert!(
        beta_diff < 5e-3,
        "layer-4 beta mismatch: max_diff={beta_diff}"
    );
    assert!(
        alpha_diff < 5e-3,
        "layer-4 alpha mismatch: max_diff={alpha_diff}"
    );
    assert!(
        safe_core_diff < 5e-3,
        "layer-4 safe recurrent core mismatch: max_diff={safe_core_diff}"
    );
    assert!(
        recurrent_core_diff < 5e-3,
        "layer-4 native recurrent core mismatch: max_diff={recurrent_core_diff}, native_vs_safe={native_vs_safe_core_diff}"
    );
    assert!(
        recurrent_out_diff < 5e-3,
        "layer-4 recurrent output mismatch: max_diff={recurrent_out_diff}"
    );
    assert!(
        z_diff < 5e-3,
        "layer-4 gated recurrent mismatch: max_diff={z_diff}"
    );
    assert!(
        proj_diff < 5e-2,
        "layer-4 proj mismatch: max_diff={proj_diff}"
    );
}

#[test]
fn test_hybrid_backend_matvec() {
    let backend = HybridBackend::new().unwrap();
    // N=1 matvec through Hybrid decode path (CPU by default).
    let a = [1.0, 2.0, 3.0, 4.0];
    let x = [1.0, 2.0];
    let mut y = [0.0f32; 2];
    backend.matmul(&a, &x, &mut y, 2, 1, 2);
    // row 0: 1+4 = 5, row 1: 3+8 = 11
    assert!((y[0] - 5.0).abs() < 1e-3);
    assert!((y[1] - 11.0).abs() < 1e-3);
}

#[test]
fn test_hybrid_backend_routes_matmul_to_metal() {
    let backend = HybridBackend::new().unwrap();
    // N=2 (prefill-style) should use Metal path.
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let mut c = [0.0f32; 4];
    backend.matmul(&a, &b, &mut c, 2, 2, 2);
    assert!((c[0] - 19.0).abs() < 1e-3);
    assert!((c[1] - 22.0).abs() < 1e-3);
    assert!((c[2] - 43.0).abs() < 1e-3);
    assert!((c[3] - 50.0).abs() < 1e-3);
}

#[test]
fn test_metal_backend_fused_q5_k_matvec() {
    let backend = MetalBackend::new().unwrap();

    let m = 4;
    let k = 512;
    let blocks_per_row = k / 256;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; 176];
            let d_val = (row as f32 + 1.0) * 0.05 + blk as f32 * 0.02;
            let d_bytes = half::f16::from_f32(d_val).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            let dmin_val = (blk as f32) * 0.01;
            let dmin_bytes = half::f16::from_f32(dmin_val).to_le_bytes();
            block[2] = dmin_bytes[0];
            block[3] = dmin_bytes[1];
            for i in 0..8 {
                block[4 + (i % 4)] = ((row + i) % 8 + 1) as u8;
                block[8 + (i % 4)] = ((blk + i) % 4) as u8;
            }
            for (i, b) in block[16..48].iter_mut().enumerate() {
                *b = ((row * 5 + blk * 3 + i) % 256) as u8;
            }
            for (i, b) in block[48..176].iter_mut().enumerate() {
                *b = ((row * 11 + blk * 7 + i) % 256) as u8;
            }
            quant_data.extend(block);
        }
    }

    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.01) - 2.56).collect();

    let mut weights = vec![0.0f32; m * k];
    crate::quant::q5_k::dequantize(&quant_data, &mut weights);
    let mut expected = vec![0.0f32; m];
    crate::compute::matmul::matmul_f32(&weights, &x, &mut expected, m, 1, k);

    let mut result = vec![0.0f32; m];
    backend.dequant_matmul(&quant_data, GgmlType::Q5K, &x, &mut result, m, 1, k);

    let diff = max_abs_diff(&result, &expected);
    assert!(
        diff < 0.5,
        "Fused Q5_K matvec mismatch: max_diff={diff}, result={result:?}, expected={expected:?}"
    );
}

#[test]
fn test_metal_backend_fused_q5_1_matvec() {
    let backend = MetalBackend::new().unwrap();

    let m = 6usize;
    let k = 256usize;
    let blocks_per_row = k / 32;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; 24];
            let d_val = 0.03 * (row as f32 + 1.0) + 0.01 * blk as f32;
            let m_val = -0.25 * row as f32 + 0.05 * blk as f32;
            let d_bytes = half::f16::from_f32(d_val).to_le_bytes();
            let m_bytes = half::f16::from_f32(m_val).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            block[2] = m_bytes[0];
            block[3] = m_bytes[1];

            let qh = 0xA5A5_5A5Au32.rotate_left(((row * 3 + blk) % 32) as u32);
            block[4..8].copy_from_slice(&qh.to_le_bytes());
            for (i, byte) in block[8..24].iter_mut().enumerate() {
                *byte = (((row * 17 + blk * 11 + i) % 16) as u8)
                    | ((((row * 7 + blk * 5 + i * 3) % 16) as u8) << 4);
            }
            quant_data.extend(block);
        }
    }

    let x: Vec<f32> = (0..k).map(|i| ((i % 23) as f32 - 11.0) * 0.019).collect();

    let mut weights = vec![0.0f32; m * k];
    crate::quant::q5_1::dequantize(&quant_data, &mut weights);
    let mut expected = vec![0.0f32; m];
    crate::compute::matmul::matmul_f32(&weights, &x, &mut expected, m, 1, k);

    let mut result = vec![0.0f32; m];
    backend.dequant_matmul(&quant_data, GgmlType::Q5_1, &x, &mut result, m, 1, k);

    let diff = max_abs_diff(&result, &expected);
    assert!(
        diff < 1e-3,
        "Fused Q5_1 matvec mismatch: max_diff={diff}, result={result:?}, expected={expected:?}"
    );
}

#[test]
fn test_real_gemma4_q5km_q5_1_tensor_matches_cpu_matvec() {
    let _env_lock = lock_env_test();
    let path = workspace_model_path("gemma-4-26B-A4B-it-Q5_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let backend = MetalBackend::new().unwrap();
    let info = model
        .tensors
        .iter()
        .filter(|tensor| {
            tensor.dtype == GgmlType::Q5_1 && is_active_layer_weight_name(&tensor.name)
        })
        .min_by_key(|tensor| tensor.n_elements())
        .expect("expected at least one active Q5_1 layer tensor in Gemma4 Q5_K_M model");
    let raw = model.tensor_data(info).unwrap();
    let k = info.shape[0] as usize;
    let m = info.n_elements() as usize / k;

    let x: Vec<f32> = (0..k).map(|i| ((i % 29) as f32 - 14.0) * 0.021).collect();
    let mut weights = vec![0.0f32; m * k];
    crate::quant::dequantize(info.dtype, raw, &mut weights);
    let mut expected = vec![0.0f32; m];
    crate::compute::matmul::matmul_f32(&weights, &x, &mut expected, m, 1, k);

    let mut actual = vec![0.0f32; m];
    backend.dequant_matmul(raw, info.dtype, &x, &mut actual, m, 1, k);

    let diff = max_abs_diff(&actual, &expected);
    assert!(
        diff < 5e-2,
        "Gemma4 Q5_1 tensor '{}' mismatched CPU reference: max_diff={diff}",
        info.name,
    );
}

#[test]
fn test_real_gemma4_q5km_down_moe_mul_mat_id_matches_cpu_hid_major_input() {
    let _env_lock = lock_env_test();
    let path = workspace_model_path("gemma-4-26B-A4B-it-Q5_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let (prefix, down_raw, down_dtype) = (0..cfg.n_layers as usize)
        .map(|layer| format!("blk.{layer}"))
        .find_map(|prefix| {
            let (down_raw, down_dtype) = weights
                .raw_with_dtype(&format!("{prefix}.ffn_down_exps.weight"))
                .ok()?;
            if down_dtype == GgmlType::Q5_1 {
                Some((prefix, down_raw, down_dtype))
            } else {
                None
            }
        })
        .expect("expected Gemma4 Q5_K_M fixture to include at least one Q5_1 MoE down layer");

    assert_eq!(
        down_dtype,
        GgmlType::Q5_1,
        "expected Gemma4 Q5_K_M routed down weights to use Q5_1 in {prefix}",
    );

    let n_tokens = 1usize;
    let n_expert = cfg.n_expert.unwrap_or(0) as usize;
    let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
    let dim = cfg.embedding_dim as usize;
    let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;
    let expert_weight_name = format!("{prefix}.ffn_down_exps.weight");
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, down_dtype, "real gemma4 q5km down")
            .unwrap();
    let active_list: Vec<usize> = (0..n_expert_used).map(|slot| n_expert - 1 - slot).collect();
    let input: Vec<f32> = (0..n_expert_used * expert_inter_dim)
        .map(|i| ((i * 17 % 257) as f32 - 128.0) * 0.01)
        .collect();

    let mut tpe = vec![0u32; n_expert];
    let mut hids = vec![0i32; n_expert * n_tokens];
    for (slot, &expert) in active_list.iter().enumerate() {
        tpe[expert] = 1;
        hids[expert * n_tokens] = slot as i32;
    }
    let mut active = vec![0u32; 1 + n_expert_used];
    active[0] = active_list.len() as u32;
    for (slot, &expert) in active_list.iter().enumerate() {
        active[1 + slot] = expert as u32;
    }

    let weights_buf = MetalBuffer::from_slice(backend.device.device(), down_raw).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let tpe_buf = MetalBuffer::from_slice(backend.device.device(), &tpe).unwrap();
    let hids_buf = MetalBuffer::from_slice(backend.device.device(), &hids).unwrap();
    let active_buf = MetalBuffer::from_slice(backend.device.device(), &active).unwrap();
    let output_buf = MetalBuffer::new(
        backend.device.device(),
        n_expert_used * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_id(
                encoder,
                &weights_buf,
                &input_buf,
                &tpe_buf,
                &hids_buf,
                &output_buf,
                down_dtype,
                dim as u32,
                expert_inter_dim as u32,
                n_tokens as u32,
                n_expert_used as u32,
                n_expert as u32,
                blocks_per_expert,
                &active_buf,
                active_list.len() as u32,
                false,
                true,
            )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..n_expert_used * dim].to_vec() };
    let mut expected = vec![0.0f32; n_expert_used * dim];
    for (slot, &expert) in active_list.iter().enumerate() {
        let expert_weights = crate::model::moe_utils::expert_quant_slice(
            down_raw,
            expert_stride,
            expert,
            &expert_weight_name,
        )
        .unwrap();
        cpu.dequant_matmul(
            expert_weights,
            down_dtype,
            &input[slot * expert_inter_dim..(slot + 1) * expert_inter_dim],
            &mut expected[slot * dim..(slot + 1) * dim],
            dim,
            1,
            expert_inter_dim,
        );
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 3e-3,
        "real Gemma4 Q5_K_M down moe_mul_mat_id mismatch in {prefix}: rel_diff={} max_diff={diff}",
        diff / scale,
    );
}

#[test]
fn test_real_gemma4_q6_routed_moe_helper_matches_cpu_reference() {
    let _env_lock = lock_env_test();
    let path = workspace_model_path("gemma-4-26B-A4B-it-Q6_K.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let prefix = (0..cfg.n_layers as usize)
        .map(|layer| format!("blk.{layer}"))
        .find(|prefix| weights.has(&format!("{prefix}.ffn_gate_inp.weight")))
        .expect("expected Gemma4 Q6_K fixture to include at least one MoE layer");

    let n_tokens = 2usize;
    let n_expert = cfg.n_expert.unwrap_or(0) as usize;
    let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
    let dim = cfg.embedding_dim as usize;
    let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;
    let eps = cfg.rms_norm_eps;
    let router_input_scale = (dim as f32).sqrt().recip();
    let hidden: Vec<f32> = (0..n_tokens * dim)
        .map(|i| ((i * 31 % 997) as f32 - 498.0) * 0.002)
        .collect();

    let router_scale = weights
        .f32_slice(&format!("{prefix}.ffn_gate_inp.scale"))
        .unwrap();
    let pre_ff2_w = weights
        .f32_slice(&format!("{prefix}.pre_ffw_norm_2.weight"))
        .unwrap();
    let post_ff2_w = weights
        .f32_slice(&format!("{prefix}.post_ffw_norm_2.weight"))
        .unwrap();
    let expert_scales = weights
        .f32_slice(&format!("{prefix}.ffn_down_exps.scale"))
        .unwrap();
    let (router_raw, router_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_gate_inp.weight"))
        .unwrap();
    let (gate_up_raw, gate_up_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_gate_up_exps.weight"))
        .unwrap();
    let (down_raw, down_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_down_exps.weight"))
        .unwrap();
    let fused_dim = 2 * expert_inter_dim;
    let gate_up_stride =
        crate::model::moe_utils::expert_byte_stride(gate_up_dtype, fused_dim * dim);
    let down_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);

    let mut expert_input = vec![0.0f32; n_tokens * dim];
    let mut router_input = vec![0.0f32; n_tokens * dim];
    let mut expert_ids = vec![-1i32; n_tokens * n_expert_used];
    let mut expert_weights = vec![0.0f32; n_tokens * n_expert_used];
    let mut expected = vec![0.0f32; n_tokens * dim];
    let mut fused_buf = vec![0.0f32; fused_dim];
    let mut down_buf = vec![0.0f32; dim];
    let mut router_logits = vec![0.0f32; n_expert];
    let gate_up_name = format!("{prefix}.ffn_gate_up_exps.weight");
    let down_name = format!("{prefix}.ffn_down_exps.weight");

    for token in 0..n_tokens {
        let start = token * dim;
        let end = start + dim;
        let hidden_token = &hidden[start..end];
        let expert_input_token = &mut expert_input[start..end];
        rms_norm::rms_norm_out(hidden_token, pre_ff2_w, expert_input_token, eps);

        let router_input_token = &mut router_input[start..end];
        router_input_token.copy_from_slice(hidden_token);
        rms_norm::rms_norm_no_weight(router_input_token, eps);
        for ((value, &scale),) in router_input_token
            .iter_mut()
            .zip(router_scale.iter())
            .map(|pair| (pair,))
        {
            *value *= router_input_scale * scale;
        }

        cpu.dequant_matmul(
            router_raw,
            router_dtype,
            router_input_token,
            &mut router_logits,
            n_expert,
            1,
            dim,
        );
        let (top_indices, mut top_weights) =
            crate::model::moe_utils::top_k_softmax(&router_logits, n_expert_used);
        for (weight, &expert_idx) in top_weights.iter_mut().zip(top_indices.iter()) {
            *weight *= expert_scales[expert_idx];
        }

        let slot_base = token * n_expert_used;
        let expected_token = &mut expected[start..end];
        for (slot_idx, (&expert_idx, &weight)) in
            top_indices.iter().zip(top_weights.iter()).enumerate()
        {
            expert_ids[slot_base + slot_idx] = expert_idx as i32;
            expert_weights[slot_base + slot_idx] = weight;

            let expert_gate_up = crate::model::moe_utils::expert_quant_slice(
                gate_up_raw,
                gate_up_stride,
                expert_idx,
                &gate_up_name,
            )
            .unwrap();
            cpu.dequant_matmul(
                expert_gate_up,
                gate_up_dtype,
                expert_input_token,
                &mut fused_buf,
                fused_dim,
                1,
                dim,
            );
            let (gate_half, up_half) = fused_buf.split_at_mut(expert_inter_dim);
            crate::compute::gelu::gelu_elementwise_mul(gate_half, up_half);

            let expert_down = crate::model::moe_utils::expert_quant_slice(
                down_raw,
                down_stride,
                expert_idx,
                &down_name,
            )
            .unwrap();
            cpu.dequant_matmul(
                expert_down,
                down_dtype,
                gate_half,
                &mut down_buf,
                dim,
                1,
                expert_inter_dim,
            );
            for (dst, &src) in expected_token.iter_mut().zip(down_buf.iter()) {
                *dst += weight * src;
            }
        }
    }

    for token in 0..n_tokens {
        let start = token * dim;
        rms_norm::rms_norm(&mut expected[start..start + dim], post_ff2_w, eps);
    }

    let mut expected_via_old_helper = vec![0.0f32; n_tokens * dim];
    backend
        .ops
        .moe_fused_gate_up_gelu_dispatch(
            &expert_input,
            &mut expected_via_old_helper,
            &expert_ids,
            &expert_weights,
            gate_up_raw,
            down_raw,
            gate_up_dtype,
            down_dtype,
            n_tokens,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            gate_up_stride,
            down_stride,
        )
        .unwrap();
    for token in 0..n_tokens {
        let start = token * dim;
        rms_norm::rms_norm(
            &mut expected_via_old_helper[start..start + dim],
            post_ff2_w,
            eps,
        );
    }

    let old_diff = max_abs_diff(&expected_via_old_helper, &expected);
    let old_scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        old_diff / old_scale < 3e-3,
        "real Gemma4 Q6 old fused gate_up helper drifted from CPU in {prefix}: rel_diff={} max_diff={old_diff}",
        old_diff / old_scale,
    );

    backend.ops.init_batch_scratches(&cfg, n_tokens);
    let router_scale_key = backend.ops.ensure_f32_cached(router_scale);
    let pre_ff2_key = backend.ops.ensure_f32_cached(pre_ff2_w);
    let post_ff2_key = backend.ops.ensure_f32_cached(post_ff2_w);
    let expert_scales_key = backend.ops.ensure_f32_cached(expert_scales);
    let router_key = backend.ops.ensure_moe_quant_cached(router_raw);
    let gate_up_key = backend.ops.ensure_moe_quant_cached(gate_up_raw);
    let down_key = backend.ops.ensure_moe_quant_cached(down_raw);

    let weight_cache = backend.ops.lock_weight_cache();
    let moe_weight_cache = backend.ops.lock_moe_weight_cache();
    let router_scale_buf = weight_cache.get(&router_scale_key).unwrap();
    let pre_ff2_buf = weight_cache.get(&pre_ff2_key).unwrap();
    let post_ff2_buf = weight_cache.get(&post_ff2_key).unwrap();
    let expert_scales_buf = weight_cache.get(&expert_scales_key).unwrap();
    let router_buf = moe_weight_cache.get(&router_key).unwrap();
    let gate_up_buf = moe_weight_cache.get(&gate_up_key).unwrap();
    let down_buf_gpu = moe_weight_cache.get(&down_key).unwrap();

    let mut batch_guard = backend.ops.batch_scratches();
    let bs = batch_guard.as_mut().unwrap();
    unsafe {
        bs.hidden.as_mut_slice::<f32>()[..hidden.len()].copy_from_slice(&hidden);
    }

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_gemma4_routed_moe_fused_gate_up(
                encoder,
                bs,
                router_scale_buf,
                router_buf,
                router_dtype,
                pre_ff2_buf,
                post_ff2_buf,
                expert_scales_buf,
                gate_up_buf,
                gate_up_dtype,
                down_buf_gpu,
                down_dtype,
                n_tokens,
                n_expert,
                n_expert_used,
                dim,
                expert_inter_dim,
                gate_up_stride,
                down_stride,
                eps,
            )
        })
        .unwrap();

    let actual =
        unsafe { bs.moe_accum.as_ref().unwrap().as_slice::<f32>()[..n_tokens * dim].to_vec() };
    let actual_ids = unsafe {
        bs.moe_expert_ids.as_ref().unwrap().as_slice::<i32>()[..n_tokens * n_expert_used].to_vec()
    };
    let actual_weights = unsafe {
        bs.moe_expert_weights.as_ref().unwrap().as_slice::<f32>()[..n_tokens * n_expert_used]
            .to_vec()
    };

    assert_eq!(
        actual_ids, expert_ids,
        "real Gemma4 Q6 routed helper selected different experts in {prefix}",
    );

    let router_weight_diff = max_abs_diff(&actual_weights, &expert_weights);
    let router_weight_scale = expert_weights
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        router_weight_diff / router_weight_scale < 3e-3,
        "real Gemma4 Q6 routed helper produced different expert weights in {prefix}: rel_diff={} max_diff={router_weight_diff}",
        router_weight_diff / router_weight_scale,
    );

    let diff = max_abs_diff(&actual, &expected_via_old_helper);
    let scale = expected_via_old_helper
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 3e-3,
        "real Gemma4 Q6 routed helper mismatch in {prefix}: rel_diff={} max_diff={diff}",
        diff / scale,
    );
}

#[test]
fn test_real_gemma4_q6_shared_ffn_path_matches_cpu_reference() {
    let _env_lock = lock_env_test();
    let path = workspace_model_path("gemma-4-26B-A4B-it-Q6_K.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let prefix = (0..cfg.n_layers as usize)
        .map(|layer| format!("blk.{layer}"))
        .find(|prefix| weights.has(&format!("{prefix}.ffn_gate_inp.weight")))
        .expect("expected Gemma4 Q6_K fixture to include at least one MoE layer");

    let tensor_output_rows = |name: &str| -> usize {
        match weights.info(name).unwrap().shape.as_slice() {
            [_input_dim] => 1,
            [_input_dim, output_dim, ..] => *output_dim as usize,
            other => panic!("unexpected tensor shape for {name}: {other:?}"),
        }
    };

    let n_tokens = 2usize;
    let dim = cfg.embedding_dim as usize;
    let inter_dim = tensor_output_rows(&format!("{prefix}.ffn_gate.weight"));
    let eps = cfg.rms_norm_eps;
    let hidden: Vec<f32> = (0..n_tokens * dim)
        .map(|i| ((i * 37 % 1021) as f32 - 510.0) * 0.0015)
        .collect();

    let ffn_norm_w = weights
        .f32_slice(&format!("{prefix}.ffn_norm.weight"))
        .unwrap();
    let post_ff1_w = weights
        .f32_slice(&format!("{prefix}.post_ffw_norm_1.weight"))
        .unwrap();
    let (wg_raw, wg_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))
        .unwrap();
    let (wu_raw, wu_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_up.weight"))
        .unwrap();
    let (wd_raw, wd_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_down.weight"))
        .unwrap();
    let wu_rows = tensor_output_rows(&format!("{prefix}.ffn_up.weight"));
    let wd_input_rows = match weights
        .info(&format!("{prefix}.ffn_down.weight"))
        .unwrap()
        .shape
        .as_slice()
    {
        [input_dim, _output_dim, ..] => *input_dim as usize,
        other => panic!("unexpected tensor shape for {prefix}.ffn_down.weight: {other:?}"),
    };
    assert_eq!(
        wu_rows, inter_dim,
        "Gemma4 shared FFN gate/up rows diverged in {prefix}: gate_rows={inter_dim} up_rows={wu_rows}",
    );
    assert_eq!(
        wd_input_rows, inter_dim,
        "Gemma4 shared FFN down input rows diverged in {prefix}: gate_rows={inter_dim} down_input_rows={wd_input_rows}",
    );

    let q6_blocks_per_row = dim / 256;
    let q6_row_bytes = q6_blocks_per_row * 210;
    assert_eq!(
        wg_raw.len(),
        inter_dim * q6_row_bytes,
        "Gemma4 Q6 shared gate raw bytes are not tightly packed rows in {prefix}: len={} expected={}",
        wg_raw.len(),
        inter_dim * q6_row_bytes,
    );
    let q6_first_row = &wg_raw[..q6_row_bytes];
    let mut q6_first_row_cpu = vec![0.0f32; dim];
    crate::quant::q6_k::dequantize(q6_first_row, &mut q6_first_row_cpu);
    let q6_first_row_buf = MetalBuffer::from_bytes(backend.device.device(), q6_first_row).unwrap();
    let q6_first_row_gpu_buf =
        MetalBuffer::new(backend.device.device(), dim * std::mem::size_of::<f32>()).unwrap();
    backend
        .ops
        .dequant
        .dequant_q6_k(
            &backend.device,
            &q6_first_row_buf,
            &q6_first_row_gpu_buf,
            q6_blocks_per_row as u32,
        )
        .unwrap();
    let q6_first_row_gpu = unsafe { q6_first_row_gpu_buf.as_slice::<f32>()[..dim].to_vec() };
    let q6_row_scale = q6_first_row_cpu
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let q6_row_diff = max_abs_diff(&q6_first_row_gpu, &q6_first_row_cpu);
    assert!(
        q6_row_diff / q6_row_scale < 3e-3,
        "real Gemma4 Q6 first-row dequant mismatch in {prefix}: rel_diff={} max_diff={q6_row_diff}",
        q6_row_diff / q6_row_scale,
    );

    let mut norm = vec![0.0f32; n_tokens * dim];
    let mut expected_gate = vec![0.0f32; n_tokens * inter_dim];
    let mut expected_up = vec![0.0f32; n_tokens * inter_dim];
    let mut expected = vec![0.0f32; n_tokens * dim];
    for token in 0..n_tokens {
        let src = &hidden[token * dim..(token + 1) * dim];
        let dst = &mut norm[token * dim..(token + 1) * dim];
        rms_norm::rms_norm_out(src, ffn_norm_w, dst, eps);
    }
    cpu.dequant_matmul_token_major(
        wg_raw,
        wg_dtype,
        &norm,
        &mut expected_gate,
        n_tokens,
        inter_dim,
        dim,
    );
    cpu.dequant_matmul_token_major(
        wu_raw,
        wu_dtype,
        &norm,
        &mut expected_up,
        n_tokens,
        inter_dim,
        dim,
    );
    let expected_gate_raw = expected_gate.clone();
    for token in 0..n_tokens {
        let gate_row = &mut expected_gate[token * inter_dim..(token + 1) * inter_dim];
        let up_row = &expected_up[token * inter_dim..(token + 1) * inter_dim];
        crate::compute::gelu::gelu_elementwise_mul(gate_row, up_row);
    }
    cpu.dequant_matmul_token_major(
        wd_raw,
        wd_dtype,
        &expected_gate,
        &mut expected,
        n_tokens,
        dim,
        inter_dim,
    );
    for token in 0..n_tokens {
        let dst = &mut expected[token * dim..(token + 1) * dim];
        rms_norm::rms_norm(dst, post_ff1_w, eps);
    }

    let norm_buf = MetalBuffer::from_slice(backend.device.device(), &norm).unwrap();
    let gate_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * inter_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let up_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * inter_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let fused_out_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let sep_out_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let norm_token0_buf = MetalBuffer::from_slice(backend.device.device(), &norm[..dim]).unwrap();
    let gate_matvec_buf = MetalBuffer::new(
        backend.device.device(),
        inter_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let gate_matvec_base_buf = MetalBuffer::new(
        backend.device.device(),
        inter_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let gate_matvec_nr2_buf = MetalBuffer::new(
        backend.device.device(),
        inter_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let gate_matvec_ilp4_buf = MetalBuffer::new(
        backend.device.device(),
        inter_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let gate_matvec_row0_buf =
        MetalBuffer::new(backend.device.device(), std::mem::size_of::<f32>()).unwrap();
    let gate_matvec_row4_buf =
        MetalBuffer::new(backend.device.device(), 4 * std::mem::size_of::<f32>()).unwrap();
    let norm_f16_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * dim * std::mem::size_of::<half::f16>(),
    )
    .unwrap();
    let gate_f16in_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * inter_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let wg_buf = MetalBuffer::from_bytes(backend.device.device(), wg_raw).unwrap();
    let wu_buf = MetalBuffer::from_bytes(backend.device.device(), wu_raw).unwrap();
    let wd_buf = MetalBuffer::from_bytes(backend.device.device(), wd_raw).unwrap();
    let post_ff1_buf = MetalBuffer::from_slice(backend.device.device(), post_ff1_w).unwrap();

    if wd_dtype == GgmlType::Q6K {
        backend
            .device
            .execute_sync(|encoder| {
                backend.ops.encode_quant_batch_matmul(
                    encoder,
                    &wg_buf,
                    &norm_buf,
                    &gate_buf,
                    inter_dim as u32,
                    n_tokens as u32,
                    dim as u32,
                    wg_dtype,
                )?;
                backend.ops.encode_quant_batch_matmul(
                    encoder,
                    &wu_buf,
                    &norm_buf,
                    &up_buf,
                    inter_dim as u32,
                    n_tokens as u32,
                    dim as u32,
                    wu_dtype,
                )?;
                backend.ops.dequant.encode_fused_batch_q6_k_gelu(
                    encoder,
                    &wd_buf,
                    &gate_buf,
                    &up_buf,
                    &fused_out_buf,
                    dim as u32,
                    n_tokens as u32,
                    inter_dim as u32,
                );
                backend.ops.elementwise.encode_rms_norm_batch(
                    encoder,
                    &fused_out_buf,
                    &post_ff1_buf,
                    dim as u32,
                    n_tokens as u32,
                    eps,
                );
                Ok(())
            })
            .unwrap();
    }

    backend
        .device
        .execute_sync(|encoder| {
            let base_config = ax_engine_metal::DequantDispatchConfig {
                q6_k_variant: Some(MatvecProfileVariant::Base),
                ..backend.ops.dequant_dispatch_config()
            };
            let nr2_config = ax_engine_metal::DequantDispatchConfig {
                q6_k_variant: Some(MatvecProfileVariant::Nr2),
                ..backend.ops.dequant_dispatch_config()
            };
            let ilp4_config = ax_engine_metal::DequantDispatchConfig {
                q6_k_variant: Some(MatvecProfileVariant::Ilp4),
                ..backend.ops.dequant_dispatch_config()
            };
            backend.ops.elementwise.encode_cast_f32_to_f16(
                encoder,
                &norm_buf,
                &norm_f16_buf,
                (n_tokens * dim) as u32,
            );
            backend.ops.encode_quant_matvec(
                encoder,
                &wg_buf,
                &norm_token0_buf,
                &gate_matvec_buf,
                inter_dim as u32,
                dim as u32,
                wg_dtype,
            )?;
            backend.ops.dequant.encode_fused_matvec_q6_k_with_config(
                encoder,
                &wg_buf,
                &norm_token0_buf,
                &gate_matvec_base_buf,
                inter_dim as u32,
                dim as u32,
                base_config,
            );
            backend.ops.dequant.encode_fused_matvec_q6_k_with_config(
                encoder,
                &wg_buf,
                &norm_token0_buf,
                &gate_matvec_nr2_buf,
                inter_dim as u32,
                dim as u32,
                nr2_config,
            );
            backend.ops.dequant.encode_fused_matvec_q6_k_with_config(
                encoder,
                &wg_buf,
                &norm_token0_buf,
                &gate_matvec_ilp4_buf,
                inter_dim as u32,
                dim as u32,
                ilp4_config,
            );
            backend.ops.dequant.encode_fused_matvec_q6_k_with_config(
                encoder,
                &wg_buf,
                &norm_token0_buf,
                &gate_matvec_row0_buf,
                1,
                dim as u32,
                base_config,
            );
            backend.ops.dequant.encode_fused_matvec_q6_k_with_config(
                encoder,
                &wg_buf,
                &norm_token0_buf,
                &gate_matvec_row4_buf,
                4,
                dim as u32,
                base_config,
            );
            backend.ops.encode_quant_batch_matmul(
                encoder,
                &wg_buf,
                &norm_buf,
                &gate_buf,
                inter_dim as u32,
                n_tokens as u32,
                dim as u32,
                wg_dtype,
            )?;
            backend
                .ops
                .dequant
                .encode_fused_batch_q6_k_f16in_with_config(
                    encoder,
                    &wg_buf,
                    &norm_f16_buf,
                    &gate_f16in_buf,
                    inter_dim as u32,
                    n_tokens as u32,
                    dim as u32,
                    backend.ops.dequant_dispatch_config(),
                );
            backend.ops.encode_quant_batch_matmul(
                encoder,
                &wu_buf,
                &norm_buf,
                &up_buf,
                inter_dim as u32,
                n_tokens as u32,
                dim as u32,
                wu_dtype,
            )?;
            backend.ops.elementwise.encode_gelu_elementwise_mul_batch(
                encoder,
                &gate_buf,
                &up_buf,
                inter_dim as u32,
                n_tokens as u32,
            );
            backend.ops.encode_quant_batch_matmul(
                encoder,
                &wd_buf,
                &gate_buf,
                &sep_out_buf,
                dim as u32,
                n_tokens as u32,
                inter_dim as u32,
                wd_dtype,
            )?;
            backend.ops.elementwise.encode_rms_norm_batch(
                encoder,
                &sep_out_buf,
                &post_ff1_buf,
                dim as u32,
                n_tokens as u32,
                eps,
            );
            Ok(())
        })
        .unwrap();

    let sep_actual = unsafe { sep_out_buf.as_slice::<f32>()[..n_tokens * dim].to_vec() };
    let gate_actual = unsafe { gate_buf.as_slice::<f32>()[..n_tokens * inter_dim].to_vec() };
    let up_actual = unsafe { up_buf.as_slice::<f32>()[..n_tokens * inter_dim].to_vec() };
    let gate_matvec_actual = unsafe { gate_matvec_buf.as_slice::<f32>()[..inter_dim].to_vec() };
    let gate_matvec_base = unsafe { gate_matvec_base_buf.as_slice::<f32>()[..inter_dim].to_vec() };
    let gate_matvec_nr2 = unsafe { gate_matvec_nr2_buf.as_slice::<f32>()[..inter_dim].to_vec() };
    let gate_matvec_ilp4 = unsafe { gate_matvec_ilp4_buf.as_slice::<f32>()[..inter_dim].to_vec() };
    let gate_matvec_row0 = unsafe { gate_matvec_row0_buf.as_slice::<f32>()[0] };
    let gate_matvec_row4 = unsafe { gate_matvec_row4_buf.as_slice::<f32>()[..4].to_vec() };
    let gate_f16in_actual =
        unsafe { gate_f16in_buf.as_slice::<f32>()[..n_tokens * inter_dim].to_vec() };

    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let gate_proj_scale = expected_gate_raw
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let gate_scale = expected_gate
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let up_scale = expected_up
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let gate_diff = max_abs_diff(&gate_actual, &expected_gate);
    let gate_matvec_diff = max_abs_diff(&gate_matvec_actual, &expected_gate_raw[..inter_dim]);
    let gate_matvec_base_diff = max_abs_diff(&gate_matvec_base, &expected_gate_raw[..inter_dim]);
    let gate_matvec_nr2_diff = max_abs_diff(&gate_matvec_nr2, &expected_gate_raw[..inter_dim]);
    let gate_matvec_ilp4_diff = max_abs_diff(&gate_matvec_ilp4, &expected_gate_raw[..inter_dim]);
    let gate_matvec_row0_diff = (gate_matvec_row0 - expected_gate[0]).abs();
    let gate_matvec_row4_diff = max_abs_diff(&gate_matvec_row4, &expected_gate_raw[..4]);
    let gate_f16in_diff = max_abs_diff(&gate_f16in_actual, &expected_gate_raw);
    let up_diff = max_abs_diff(&up_actual, &expected_up);
    let gate_matvec_max_row = gate_matvec_actual
        .iter()
        .zip(expected_gate_raw[..inter_dim].iter())
        .enumerate()
        .max_by(
            |(_, (lhs_actual, lhs_expected)), (_, (rhs_actual, rhs_expected))| {
                (*lhs_actual - *lhs_expected)
                    .abs()
                    .partial_cmp(&(*rhs_actual - *rhs_expected).abs())
                    .unwrap()
            },
        )
        .map(|(row, _)| row)
        .unwrap();
    let gate_matvec_standalone_buf = MetalBuffer::new(
        backend.device.device(),
        inter_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    backend
        .ops
        .dequant
        .fused_matvec_q6_k_with_config(
            &backend.device,
            &wg_buf,
            &norm_token0_buf,
            &gate_matvec_standalone_buf,
            inter_dim as u32,
            dim as u32,
            ax_engine_metal::DequantDispatchConfig {
                q6_k_variant: Some(MatvecProfileVariant::Base),
                ..backend.ops.dequant_dispatch_config()
            },
        )
        .unwrap();
    let gate_matvec_standalone =
        unsafe { gate_matvec_standalone_buf.as_slice::<f32>()[..inter_dim].to_vec() };
    let gate_matvec_standalone_diff =
        max_abs_diff(&gate_matvec_standalone, &expected_gate_raw[..inter_dim]);
    let gate_batch_standalone_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * inter_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_quant_batch_matmul(
                encoder,
                &wg_buf,
                &norm_buf,
                &gate_batch_standalone_buf,
                inter_dim as u32,
                n_tokens as u32,
                dim as u32,
                wg_dtype,
            )?;
            Ok(())
        })
        .unwrap();
    let gate_batch_standalone =
        unsafe { gate_batch_standalone_buf.as_slice::<f32>()[..n_tokens * inter_dim].to_vec() };
    let gate_batch_standalone_diff = max_abs_diff(&gate_batch_standalone, &expected_gate_raw);
    let probe_single_row = |row: usize| -> f32 {
        let row_start = row * q6_row_bytes;
        let row_end = row_start + q6_row_bytes;
        let row_buf =
            MetalBuffer::from_bytes(backend.device.device(), &wg_raw[row_start..row_end]).unwrap();
        let out_buf =
            MetalBuffer::new(backend.device.device(), std::mem::size_of::<f32>()).unwrap();
        backend
            .ops
            .dequant
            .fused_matvec_q6_k_with_config(
                &backend.device,
                &row_buf,
                &norm_token0_buf,
                &out_buf,
                1,
                dim as u32,
                ax_engine_metal::DequantDispatchConfig {
                    q6_k_variant: Some(MatvecProfileVariant::Base),
                    ..backend.ops.dequant_dispatch_config()
                },
            )
            .unwrap();
        let actual = unsafe { out_buf.as_slice::<f32>()[0] };
        (actual - expected_gate_raw[row]).abs()
    };
    let gate_matvec_mid_single_row_diff = probe_single_row(inter_dim / 2);
    let gate_matvec_last_single_row_diff = probe_single_row(inter_dim - 1);
    let gate_matvec_max_row_single_row_diff = probe_single_row(gate_matvec_max_row);
    let max_row_start = gate_matvec_max_row * q6_row_bytes;
    let max_row_end = max_row_start + q6_row_bytes;
    let max_row_raw = &wg_raw[max_row_start..max_row_end];
    let mut max_row_cpu = vec![0.0f32; dim];
    crate::quant::q6_k::dequantize(max_row_raw, &mut max_row_cpu);
    let max_row_src_buf = MetalBuffer::from_bytes(backend.device.device(), max_row_raw).unwrap();
    let max_row_gpu_buf =
        MetalBuffer::new(backend.device.device(), dim * std::mem::size_of::<f32>()).unwrap();
    backend
        .ops
        .dequant
        .dequant_q6_k(
            &backend.device,
            &max_row_src_buf,
            &max_row_gpu_buf,
            q6_blocks_per_row as u32,
        )
        .unwrap();
    let max_row_gpu = unsafe { max_row_gpu_buf.as_slice::<f32>()[..dim].to_vec() };
    let max_row_dequant_scale = max_row_cpu
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let max_row_dequant_diff = max_abs_diff(&max_row_gpu, &max_row_cpu);
    let mut max_row_block_idx = 0usize;
    let mut max_row_block_diff = 0.0f32;
    let mut max_row_fullk_block_idx = 0usize;
    let mut max_row_fullk_block_diff = 0.0f32;
    let mut max_row_abs_block_sum = 0.0f32;
    let mut first_bad_prefix_blocks = 0usize;
    let mut worst_prefix_blocks = 0usize;
    let mut worst_prefix_diff = 0.0f32;
    let run_sparse_fullk = |block_indices: &[usize]| -> f32 {
        let mut input_sparse = vec![0.0f32; dim];
        let mut expected = 0.0f32;
        for &block_idx in block_indices {
            let input_start = block_idx * 256;
            let input_end = input_start + 256;
            let input_block = &norm[input_start..input_end];
            input_sparse[input_start..input_end].copy_from_slice(input_block);

            let block_start = block_idx * 210;
            let block_end = block_start + 210;
            let block_raw = &max_row_raw[block_start..block_end];
            let mut cpu_block = vec![0.0f32; 256];
            crate::quant::q6_k::dequantize(block_raw, &mut cpu_block);
            expected += cpu_block
                .iter()
                .zip(input_block.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
        }

        let input_sparse_buf =
            MetalBuffer::from_slice(backend.device.device(), &input_sparse).unwrap();
        let out_buf =
            MetalBuffer::new(backend.device.device(), std::mem::size_of::<f32>()).unwrap();
        backend
            .ops
            .dequant
            .fused_matvec_q6_k_with_config(
                &backend.device,
                &max_row_src_buf,
                &input_sparse_buf,
                &out_buf,
                1,
                dim as u32,
                ax_engine_metal::DequantDispatchConfig {
                    q6_k_variant: Some(MatvecProfileVariant::Base),
                    ..backend.ops.dequant_dispatch_config()
                },
            )
            .unwrap();
        let actual = unsafe { out_buf.as_slice::<f32>()[0] };
        (actual - expected).abs()
    };
    for block_idx in 0..q6_blocks_per_row {
        let block_start = block_idx * 210;
        let block_end = block_start + 210;
        let block_raw = &max_row_raw[block_start..block_end];
        let input_start = block_idx * 256;
        let input_end = input_start + 256;
        let input_block = &norm[..dim][input_start..input_end];
        let mut cpu_block = vec![0.0f32; 256];
        crate::quant::q6_k::dequantize(block_raw, &mut cpu_block);
        let expected_block = cpu_block
            .iter()
            .zip(input_block.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>();
        max_row_abs_block_sum += expected_block.abs();
        let block_weights_buf =
            MetalBuffer::from_bytes(backend.device.device(), block_raw).unwrap();
        let block_input_buf =
            MetalBuffer::from_slice(backend.device.device(), input_block).unwrap();
        let block_out_buf =
            MetalBuffer::new(backend.device.device(), std::mem::size_of::<f32>()).unwrap();
        backend
            .ops
            .dequant
            .fused_matvec_q6_k_with_config(
                &backend.device,
                &block_weights_buf,
                &block_input_buf,
                &block_out_buf,
                1,
                256,
                ax_engine_metal::DequantDispatchConfig {
                    q6_k_variant: Some(MatvecProfileVariant::Base),
                    ..backend.ops.dequant_dispatch_config()
                },
            )
            .unwrap();
        let actual_block = unsafe { block_out_buf.as_slice::<f32>()[0] };
        let block_diff = (actual_block - expected_block).abs();
        if block_diff > max_row_block_diff {
            max_row_block_diff = block_diff;
            max_row_block_idx = block_idx;
        }

        let mut input_sparse = vec![0.0f32; dim];
        input_sparse[input_start..input_end].copy_from_slice(input_block);
        let input_sparse_buf =
            MetalBuffer::from_slice(backend.device.device(), &input_sparse).unwrap();
        let fullk_out_buf =
            MetalBuffer::new(backend.device.device(), std::mem::size_of::<f32>()).unwrap();
        backend
            .ops
            .dequant
            .fused_matvec_q6_k_with_config(
                &backend.device,
                &max_row_src_buf,
                &input_sparse_buf,
                &fullk_out_buf,
                1,
                dim as u32,
                ax_engine_metal::DequantDispatchConfig {
                    q6_k_variant: Some(MatvecProfileVariant::Base),
                    ..backend.ops.dequant_dispatch_config()
                },
            )
            .unwrap();
        let actual_fullk_block = unsafe { fullk_out_buf.as_slice::<f32>()[0] };
        let fullk_block_diff = (actual_fullk_block - expected_block).abs();
        if fullk_block_diff > max_row_fullk_block_diff {
            max_row_fullk_block_diff = fullk_block_diff;
            max_row_fullk_block_idx = block_idx;
        }
    }
    for prefix_blocks in 1..=q6_blocks_per_row {
        let prefix_dim = prefix_blocks * 256;
        let prefix_row_raw = &max_row_raw[..prefix_blocks * 210];
        let prefix_input = &norm[..prefix_dim];
        let mut prefix_row_cpu = vec![0.0f32; prefix_dim];
        crate::quant::q6_k::dequantize(prefix_row_raw, &mut prefix_row_cpu);
        let prefix_expected = prefix_row_cpu
            .iter()
            .zip(prefix_input.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>();
        let prefix_row_buf =
            MetalBuffer::from_bytes(backend.device.device(), prefix_row_raw).unwrap();
        let prefix_input_buf =
            MetalBuffer::from_slice(backend.device.device(), prefix_input).unwrap();
        let prefix_out_buf =
            MetalBuffer::new(backend.device.device(), std::mem::size_of::<f32>()).unwrap();
        backend
            .ops
            .dequant
            .fused_matvec_q6_k_with_config(
                &backend.device,
                &prefix_row_buf,
                &prefix_input_buf,
                &prefix_out_buf,
                1,
                prefix_dim as u32,
                ax_engine_metal::DequantDispatchConfig {
                    q6_k_variant: Some(MatvecProfileVariant::Base),
                    ..backend.ops.dequant_dispatch_config()
                },
            )
            .unwrap();
        let prefix_actual = unsafe { prefix_out_buf.as_slice::<f32>()[0] };
        let prefix_diff = (prefix_actual - prefix_expected).abs();
        if first_bad_prefix_blocks == 0 && prefix_diff > 1e-2 {
            first_bad_prefix_blocks = prefix_blocks;
        }
        if prefix_diff > worst_prefix_diff {
            worst_prefix_diff = prefix_diff;
            worst_prefix_blocks = prefix_blocks;
        }
    }
    let same_group_two_block_diff = run_sparse_fullk(&[0, 4]);
    let diff_group_two_block_diff = run_sparse_fullk(&[0, 1]);
    assert!(
        gate_matvec_diff / gate_proj_scale < 3e-3,
        "real Gemma4 Q6 shared gate matvec mismatch in {prefix} (wg={wg_dtype:?}): default_rel_diff={} max_diff={gate_matvec_diff} base_rel_diff={} nr2_rel_diff={} ilp4_rel_diff={} standalone_matvec_rel_diff={} row0_rel_diff={} row4_rel_diff={} mid_single_row_rel_diff={} last_single_row_rel_diff={} max_row={} max_row_single_row_rel_diff={} max_row_dequant_rel_diff={} max_row_abs_block_sum={} max_row_block={} max_row_block_abs_diff={} max_row_fullk_block={} max_row_fullk_block_abs_diff={} first_bad_prefix_blocks={} worst_prefix_blocks={} worst_prefix_abs_diff={} same_group_two_block_abs_diff={} diff_group_two_block_abs_diff={} batch_f32_rel_diff={} standalone_batch_rel_diff={} batch_f16in_rel_diff={}",
        gate_matvec_diff / gate_proj_scale,
        gate_matvec_base_diff / gate_proj_scale,
        gate_matvec_nr2_diff / gate_proj_scale,
        gate_matvec_ilp4_diff / gate_proj_scale,
        gate_matvec_standalone_diff / gate_proj_scale,
        gate_matvec_row0_diff / gate_proj_scale,
        gate_matvec_row4_diff / gate_proj_scale,
        gate_matvec_mid_single_row_diff / gate_proj_scale,
        gate_matvec_last_single_row_diff / gate_proj_scale,
        gate_matvec_max_row,
        gate_matvec_max_row_single_row_diff / gate_proj_scale,
        max_row_dequant_diff / max_row_dequant_scale,
        max_row_abs_block_sum,
        max_row_block_idx,
        max_row_block_diff,
        max_row_fullk_block_idx,
        max_row_fullk_block_diff,
        first_bad_prefix_blocks,
        worst_prefix_blocks,
        worst_prefix_diff,
        same_group_two_block_diff,
        diff_group_two_block_diff,
        gate_diff / gate_scale,
        gate_batch_standalone_diff / gate_proj_scale,
        gate_f16in_diff / gate_proj_scale,
    );
    assert!(
        gate_diff / gate_scale < 3e-3,
        "real Gemma4 Q6 shared gate projection mismatch in {prefix} (wg={wg_dtype:?}): rel_diff={} max_diff={gate_diff} f16in_rel_diff={}",
        gate_diff / gate_scale,
        gate_f16in_diff / gate_scale,
    );
    assert!(
        up_diff / up_scale < 3e-3,
        "real Gemma4 Q6 shared up projection mismatch in {prefix} (wu={wu_dtype:?}): rel_diff={} max_diff={up_diff}",
        up_diff / up_scale,
    );
    let sep_diff = max_abs_diff(&sep_actual, &expected);
    assert!(
        sep_diff / scale < 3e-3,
        "real Gemma4 Q6 shared FFN path mismatch in {prefix} (wg={wg_dtype:?} wu={wu_dtype:?} wd={wd_dtype:?}): rel_diff={} max_diff={sep_diff}",
        sep_diff / scale,
    );
    if wd_dtype == GgmlType::Q6K {
        let fused_actual = unsafe { fused_out_buf.as_slice::<f32>()[..n_tokens * dim].to_vec() };
        let fused_diff = max_abs_diff(&fused_actual, &expected);
        assert!(
            fused_diff / scale < 3e-3,
            "real Gemma4 Q6 shared FFN fused path mismatch in {prefix}: rel_diff={} max_diff={fused_diff}",
            fused_diff / scale,
        );
    }
}

#[test]
fn test_metal_backend_moe_down_projection_reads_input_by_hid_for_blocked_q4k() {
    let backend = MetalBackend::new().unwrap();

    let n_tokens = 1usize;
    let n_expert = 2usize;
    let n_expert_used = 2usize;
    let m = 64usize;
    let k = 256usize;
    let blocks_per_row = 1usize;
    let blocks_per_expert = m * blocks_per_row;

    let mut weights = Vec::with_capacity(n_expert * blocks_per_expert * 144);
    for _expert in 0..n_expert {
        for _row in 0..m {
            weights.extend_from_slice(&q4k_block_first128_constant(1));
        }
    }

    let mut input = vec![0.0f32; n_tokens * n_expert_used * k];
    input[..128].fill(1.0);
    input[k..k + 128].fill(2.0);

    let tpe = [1u32, 1u32];
    let hids = [0i32, 1i32];
    let active = [2u32, 0u32, 1u32];

    let weights_buf = MetalBuffer::from_slice(backend.device.device(), &weights).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let tpe_buf = MetalBuffer::from_slice(backend.device.device(), &tpe).unwrap();
    let hids_buf = MetalBuffer::from_slice(backend.device.device(), &hids).unwrap();
    let active_buf = MetalBuffer::from_slice(backend.device.device(), &active).unwrap();
    let output_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * n_expert_used * m * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .ops
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_id(
                encoder,
                &weights_buf,
                &input_buf,
                &tpe_buf,
                &hids_buf,
                &output_buf,
                GgmlType::Q4K,
                m as u32,
                k as u32,
                n_tokens as u32,
                n_expert_used as u32,
                n_expert as u32,
                blocks_per_expert as u32,
                &active_buf,
                active[0],
                false,
                true,
            )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..n_expert_used * m].to_vec() };
    for row in 0..m {
        assert!(
            (actual[row] - 128.0).abs() < 1e-2,
            "expert slot 0 row {row} expected 128, got {}",
            actual[row]
        );
        assert!(
            (actual[m + row] - 256.0).abs() < 1e-2,
            "expert slot 1 row {row} expected 256, got {}",
            actual[m + row]
        );
    }
}

#[test]
fn test_metal_backend_moe_weighted_reduce_slots_accumulates_per_token() {
    let backend = MetalBackend::new().unwrap();

    let n_tokens = 2u32;
    let n_expert_used = 2u32;
    let dim = 4u32;
    let src = [
        1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0,
    ];
    let weights = [0.25f32, 0.75, 0.4, 0.6];

    let src_buf = MetalBuffer::from_slice(backend.device.device(), &src).unwrap();
    let weights_buf = MetalBuffer::from_slice(backend.device.device(), &weights).unwrap();
    let mut dst_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens as usize * dim as usize * std::mem::size_of::<f32>(),
    )
    .unwrap();
    unsafe {
        dst_buf.as_mut_slice::<f32>().fill(0.0);
    }

    backend
        .ops
        .device
        .execute_sync(|encoder| {
            backend.ops.elementwise.encode_moe_weighted_reduce_slots(
                encoder,
                &src_buf,
                &weights_buf,
                &dst_buf,
                dim,
                n_tokens,
                n_expert_used,
            );
            Ok(())
        })
        .unwrap();

    let actual = unsafe { dst_buf.as_slice::<f32>()[..n_tokens as usize * dim as usize].to_vec() };
    let expected = [7.75f32, 15.5, 23.25, 31.0, 32.0, 38.4, 44.8, 51.2];
    for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "reduced slot output mismatch at {idx}: got {got}, want {want}"
        );
    }
}

#[test]
fn test_metal_backend_moe_softmax_topk_matches_cpu_reference() {
    let backend = MetalBackend::new().unwrap();

    let n_tokens = 3u32;
    let n_expert = 8u32;
    let n_expert_used = 3u32;
    let router_logits = [
        1.0f32, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 0.0, -4.0, -1.0, -3.0, -2.0, -8.0, -5.0, -7.0, -6.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    let router_buf = MetalBuffer::from_slice(backend.device.device(), &router_logits).unwrap();
    let ids_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens as usize * n_expert_used as usize * std::mem::size_of::<i32>(),
    )
    .unwrap();
    let weights_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens as usize * n_expert_used as usize * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .ops
        .device
        .execute_sync(|encoder| {
            backend.ops.elementwise.encode_moe_softmax_topk(
                encoder,
                &router_buf,
                &ids_buf,
                &weights_buf,
                n_tokens,
                n_expert,
                n_expert_used,
            );
            Ok(())
        })
        .unwrap();

    let actual_ids =
        unsafe { ids_buf.as_slice::<i32>()[..n_tokens as usize * n_expert_used as usize].to_vec() };
    let actual_weights = unsafe {
        weights_buf.as_slice::<f32>()[..n_tokens as usize * n_expert_used as usize].to_vec()
    };

    for token in 0..n_tokens as usize {
        let start = token * n_expert as usize;
        let end = start + n_expert as usize;
        let (expected_ids, expected_weights) = crate::model::moe_utils::top_k_softmax(
            &router_logits[start..end],
            n_expert_used as usize,
        );
        let actual_ids_token =
            &actual_ids[token * n_expert_used as usize..(token + 1) * n_expert_used as usize];
        let actual_weights_token =
            &actual_weights[token * n_expert_used as usize..(token + 1) * n_expert_used as usize];

        assert_eq!(
            actual_ids_token,
            expected_ids
                .iter()
                .map(|&id| id as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            "expert ids mismatch for token {token}"
        );
        for (slot, (&got, &want)) in actual_weights_token
            .iter()
            .zip(expected_weights.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-6,
                "expert weight mismatch for token {token} slot {slot}: got {got}, want {want}"
            );
        }
    }
}

#[test]
fn test_real_qwen3_coder_q8_0_gate_moe_mul_mat_id_matches_cpu_token_major_input() {
    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 4usize;
    let n_expert = cfg.n_expert.unwrap_or(0) as usize;
    let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
    let dim = cfg.embedding_dim as usize;
    let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;

    let (gate_raw, gate_dtype) = weights
        .raw_with_dtype("blk.0.ffn_gate_exps.weight")
        .unwrap();
    assert_eq!(gate_dtype, GgmlType::Q8_0);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, gate_dtype, "real qwen3coder q8 gate")
            .unwrap();

    let input: Vec<f32> = (0..n_tokens * dim)
        .map(|i| ((i * 11 % 101) as f32 - 50.0) * 0.0125)
        .collect();

    let active_list = [0usize, 1usize];
    let mut tpe = vec![0u32; n_expert];
    let mut hids = vec![0i32; n_expert * n_tokens];
    let assignments = [
        (0usize, [0usize, n_expert_used + 2, 2 * n_expert_used + 1]),
        (1usize, [1usize, 2 * n_expert_used + 3, 3 * n_expert_used]),
    ];
    for (expert, expert_hids) in assignments {
        tpe[expert] = expert_hids.len() as u32;
        let dst = &mut hids[expert * n_tokens..expert * n_tokens + expert_hids.len()];
        for (slot, &hid) in expert_hids.iter().enumerate() {
            dst[slot] = hid as i32;
        }
    }
    let mut active = vec![0u32; 1 + n_expert];
    active[0] = active_list.len() as u32;
    for (slot, &expert) in active_list.iter().enumerate() {
        active[1 + slot] = expert as u32;
    }

    let weights_buf = MetalBuffer::from_slice(backend.device.device(), gate_raw).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let tpe_buf = MetalBuffer::from_slice(backend.device.device(), &tpe).unwrap();
    let hids_buf = MetalBuffer::from_slice(backend.device.device(), &hids).unwrap();
    let active_buf = MetalBuffer::from_slice(backend.device.device(), &active).unwrap();
    let output_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * n_expert_used * expert_inter_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_id(
                encoder,
                &weights_buf,
                &input_buf,
                &tpe_buf,
                &hids_buf,
                &output_buf,
                gate_dtype,
                expert_inter_dim as u32,
                dim as u32,
                n_tokens as u32,
                n_expert_used as u32,
                n_expert as u32,
                blocks_per_expert,
                &active_buf,
                active_list.len() as u32,
                false,
                false,
            )
        })
        .unwrap();

    let actual = unsafe {
        output_buf.as_slice::<f32>()[..n_tokens * n_expert_used * expert_inter_dim].to_vec()
    };
    let mut expected = vec![0.0f32; n_tokens * n_expert_used * expert_inter_dim];
    for (expert, expert_hids) in assignments {
        let expert_weights = crate::model::moe_utils::expert_quant_slice(
            gate_raw,
            expert_stride,
            expert,
            "blk.0.ffn_gate_exps.weight",
        )
        .unwrap();
        for &hid in &expert_hids {
            let token = hid / n_expert_used;
            cpu.dequant_matmul(
                expert_weights,
                gate_dtype,
                &input[token * dim..(token + 1) * dim],
                &mut expected[hid * expert_inter_dim..(hid + 1) * expert_inter_dim],
                expert_inter_dim,
                1,
                dim,
            );
        }
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "real Qwen3-Coder Q8_0 gate moe_mul_mat_id mismatch: rel_diff={} max_diff={diff}",
        diff / scale,
    );
}

#[test]
fn test_real_qwen3_coder_q8_0_down_moe_mul_mat_id_matches_cpu_hid_major_input() {
    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 4usize;
    let n_expert = cfg.n_expert.unwrap_or(0) as usize;
    let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
    let dim = cfg.embedding_dim as usize;
    let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;

    let (down_raw, down_dtype) = weights
        .raw_with_dtype("blk.0.ffn_down_exps.weight")
        .unwrap();
    assert_eq!(down_dtype, GgmlType::Q8_0);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, down_dtype, "real qwen3coder q8 down")
            .unwrap();

    let input: Vec<f32> = (0..n_tokens * n_expert_used * expert_inter_dim)
        .map(|i| ((i * 5 % 89) as f32 - 44.0) * 0.01)
        .collect();

    let active_list = [0usize, 1usize];
    let mut tpe = vec![0u32; n_expert];
    let mut hids = vec![0i32; n_expert * n_tokens];
    let assignments = [
        (0usize, [0usize, n_expert_used + 2, 2 * n_expert_used + 1]),
        (1usize, [1usize, 2 * n_expert_used + 3, 3 * n_expert_used]),
    ];
    for (expert, expert_hids) in assignments {
        tpe[expert] = expert_hids.len() as u32;
        let dst = &mut hids[expert * n_tokens..expert * n_tokens + expert_hids.len()];
        for (slot, &hid) in expert_hids.iter().enumerate() {
            dst[slot] = hid as i32;
        }
    }
    let mut active = vec![0u32; 1 + n_expert];
    active[0] = active_list.len() as u32;
    for (slot, &expert) in active_list.iter().enumerate() {
        active[1 + slot] = expert as u32;
    }

    let weights_buf = MetalBuffer::from_slice(backend.device.device(), down_raw).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let tpe_buf = MetalBuffer::from_slice(backend.device.device(), &tpe).unwrap();
    let hids_buf = MetalBuffer::from_slice(backend.device.device(), &hids).unwrap();
    let active_buf = MetalBuffer::from_slice(backend.device.device(), &active).unwrap();
    let output_buf = MetalBuffer::new(
        backend.device.device(),
        n_tokens * n_expert_used * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_id(
                encoder,
                &weights_buf,
                &input_buf,
                &tpe_buf,
                &hids_buf,
                &output_buf,
                down_dtype,
                dim as u32,
                expert_inter_dim as u32,
                n_tokens as u32,
                n_expert_used as u32,
                n_expert as u32,
                blocks_per_expert,
                &active_buf,
                active_list.len() as u32,
                false,
                true,
            )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..n_tokens * n_expert_used * dim].to_vec() };
    let mut expected = vec![0.0f32; n_tokens * n_expert_used * dim];
    for (expert, expert_hids) in assignments {
        let expert_weights = crate::model::moe_utils::expert_quant_slice(
            down_raw,
            expert_stride,
            expert,
            "blk.0.ffn_down_exps.weight",
        )
        .unwrap();
        for &hid in &expert_hids {
            cpu.dequant_matmul(
                expert_weights,
                down_dtype,
                &input[hid * expert_inter_dim..(hid + 1) * expert_inter_dim],
                &mut expected[hid * dim..(hid + 1) * dim],
                dim,
                1,
                expert_inter_dim,
            );
        }
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "real Qwen3-Coder Q8_0 down moe_mul_mat_id mismatch: rel_diff={} max_diff={diff}",
        diff / scale,
    );
}

fn run_real_qwen3_coder_q5_layer6_down_moe_mul_mat_id_test(allow_blocked_input_is_hid: bool) {
    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 1usize;
    let n_expert = cfg.n_expert.unwrap_or(0) as usize;
    let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
    let dim = cfg.embedding_dim as usize;
    let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;
    assert_eq!(n_expert_used, 8, "expected Qwen3-Coder top-k=8");

    let (_, _, down_dtype) =
        crate::model::shared::routed_moe_expert_dtypes(&weights, "blk.6").unwrap();
    let (down_raw, raw_down_dtype) = weights
        .raw_with_dtype("blk.6.ffn_down_exps.weight")
        .unwrap();
    assert_eq!(raw_down_dtype, down_dtype);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);
    let blocks_per_expert = MetalOps::moe_blocks_per_expert(
        expert_stride,
        down_dtype,
        "real qwen3coder q5 layer6 down",
    )
    .unwrap();

    let active_list = [103usize, 112, 32, 127, 28, 44, 27, 48];
    let input: Vec<f32> = (0..n_expert_used * expert_inter_dim)
        .map(|i| ((i * 13 % 257) as f32 - 128.0) * 0.01)
        .collect();

    let mut tpe = vec![0u32; n_expert];
    let mut hids = vec![0i32; n_expert * n_tokens];
    for (slot, &expert) in active_list.iter().enumerate() {
        tpe[expert] = 1;
        hids[expert * n_tokens] = slot as i32;
    }
    let mut active = vec![0u32; 1 + n_expert];
    active[0] = active_list.len() as u32;
    for (slot, &expert) in active_list.iter().enumerate() {
        active[1 + slot] = expert as u32;
    }

    let weights_buf = MetalBuffer::from_slice(backend.device.device(), down_raw).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let tpe_buf = MetalBuffer::from_slice(backend.device.device(), &tpe).unwrap();
    let hids_buf = MetalBuffer::from_slice(backend.device.device(), &hids).unwrap();
    let active_buf = MetalBuffer::from_slice(backend.device.device(), &active).unwrap();
    let output_buf = MetalBuffer::new(
        backend.device.device(),
        n_expert_used * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_id(
                encoder,
                &weights_buf,
                &input_buf,
                &tpe_buf,
                &hids_buf,
                &output_buf,
                down_dtype,
                dim as u32,
                expert_inter_dim as u32,
                n_tokens as u32,
                n_expert_used as u32,
                n_expert as u32,
                blocks_per_expert,
                &active_buf,
                active_list.len() as u32,
                allow_blocked_input_is_hid,
                true,
            )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..n_expert_used * dim].to_vec() };
    let actual_nonfinite = actual
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite());
    assert!(
        actual_nonfinite.is_none(),
        "real Qwen3-Coder Q5_K_M layer6 down moe_mul_mat_id produced non-finite output with dtype={down_dtype:?} blocked_input_is_hid={allow_blocked_input_is_hid}: {actual_nonfinite:?}"
    );

    let mut expected = vec![0.0f32; n_expert_used * dim];
    for (slot, &expert) in active_list.iter().enumerate() {
        let expert_weights = crate::model::moe_utils::expert_quant_slice(
            down_raw,
            expert_stride,
            expert,
            "blk.6.ffn_down_exps.weight",
        )
        .unwrap();
        cpu.dequant_matmul(
            expert_weights,
            down_dtype,
            &input[slot * expert_inter_dim..(slot + 1) * expert_inter_dim],
            &mut expected[slot * dim..(slot + 1) * dim],
            dim,
            1,
            expert_inter_dim,
        );
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 2e-3,
        "real Qwen3-Coder Q5_K_M layer6 down moe_mul_mat_id mismatch with dtype={down_dtype:?} blocked_input_is_hid={allow_blocked_input_is_hid}: rel_diff={} max_diff={diff}",
        diff / scale,
    );
}

#[test]
fn test_real_qwen3_coder_q5_layer6_down_moe_mul_mat_id_matches_cpu_hid_major_input_f32_path() {
    run_real_qwen3_coder_q5_layer6_down_moe_mul_mat_id_test(false);
}

#[test]
fn test_real_qwen3_coder_q5_layer6_down_moe_mul_mat_id_matches_cpu_hid_major_input_blocked_path() {
    run_real_qwen3_coder_q5_layer6_down_moe_mul_mat_id_test(true);
}

#[test]
fn test_real_qwen3_coder_q8_0_batch_matmul_token_major_matches_cpu_for_attention_and_output() {
    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let metal = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 8usize;
    let dim = cfg.embedding_dim as usize;
    let q_dim = cfg.n_heads as usize * cfg.head_dim as usize;
    let kv_dim = cfg.n_kv_heads as usize * cfg.head_dim as usize;

    let hidden_input: Vec<f32> = (0..n_tokens * dim)
        .map(|i| ((i * 17 % 113) as f32 - 56.0) * 0.01)
        .collect();
    let attn_input: Vec<f32> = (0..n_tokens * q_dim)
        .map(|i| ((i * 7 % 79) as f32 - 39.0) * 0.0125)
        .collect();

    for (tensor_name, input, out_dim, in_dim) in [
        ("blk.0.attn_q.weight", hidden_input.as_slice(), q_dim, dim),
        ("blk.0.attn_k.weight", hidden_input.as_slice(), kv_dim, dim),
        ("blk.0.attn_v.weight", hidden_input.as_slice(), kv_dim, dim),
        (
            "blk.0.attn_output.weight",
            attn_input.as_slice(),
            dim,
            q_dim,
        ),
    ] {
        let (raw, dtype) = weights.raw_with_dtype(tensor_name).unwrap();
        assert_eq!(dtype, GgmlType::Q8_0, "{tensor_name} should be Q8_0");

        let mut expected = vec![0.0f32; n_tokens * out_dim];
        let mut actual = vec![0.0f32; n_tokens * out_dim];

        cpu.dequant_matmul_token_major(raw, dtype, input, &mut expected, n_tokens, out_dim, in_dim);
        metal.dequant_matmul_token_major(raw, dtype, input, &mut actual, n_tokens, out_dim, in_dim);

        let diff = max_abs_diff(&actual, &expected);
        let scale = expected
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1.0);
        assert!(
            diff / scale < 1e-3,
            "{tensor_name} token-major batch matmul mismatch: rel_diff={} max_diff={diff}",
            diff / scale,
        );
    }
}

#[test]
fn test_metal_backend_moe_ffn_gpu_resident_cached_matches_cpu_for_routed_experts() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 1usize;
    let n_expert = 2usize;
    let n_expert_used = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;
    let eps = 1e-5f32;

    let hidden: Vec<f32> = (0..dim)
        .map(|i| {
            if i < 128 {
                0.01
            } else {
                0.005 + (i - 128) as f32 * 0.00001
            }
        })
        .collect();
    let norm_weights = vec![1.0f32; dim];

    let mut router_weights = vec![0.0f32; n_expert * dim];
    router_weights[..128].fill(0.02);
    router_weights[dim..dim + 128].fill(0.01);

    let build_q4k_expert_tensor = |expert_nibbles: &[u8], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_nibbles.len() * rows * 144);
        for &nibble in expert_nibbles {
            for _ in 0..rows {
                bytes.extend_from_slice(&q4k_block_first128_constant(nibble));
            }
        }
        bytes
    };

    let gate_weights = build_q4k_expert_tensor(&[1, 2], expert_inter_dim);
    let up_weights = build_q4k_expert_tensor(&[2, 1], expert_inter_dim);
    let down_weights = build_q4k_expert_tensor(&[1, 2], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q4K, dim * expert_inter_dim);
    let config = crate::model::config::ModelConfig {
        architecture: "qwen35".to_string(),
        n_layers: 1,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: dim as u32,
        head_dim: dim as u32,
        intermediate_dim: expert_inter_dim as u32,
        context_length: 16,
        vocab_size: 32,
        rms_norm_eps: eps,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: crate::model::config::GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: Some(n_expert as u32),
        n_expert_used: Some(n_expert_used as u32),
        expert_intermediate_dim: Some(expert_inter_dim as u32),
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let mean_sq = hidden.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
    let norm: Vec<f32> = hidden
        .iter()
        .zip(norm_weights.iter())
        .map(|(&x, &w)| x * inv_rms * w)
        .collect();

    let mut router_logits = vec![0.0f32; n_expert];
    cpu.matmul(
        &router_weights,
        &norm,
        &mut router_logits,
        n_expert,
        n_tokens,
        dim,
    );
    let (expert_ids, expert_weights) =
        crate::model::moe_utils::top_k_softmax(&router_logits, n_expert_used);

    let mut expected_accum = vec![0.0f32; dim];
    let mut gate_buf = vec![0.0f32; expert_inter_dim];
    let mut up_buf = vec![0.0f32; expert_inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    for (slot, &eid) in expert_ids.iter().enumerate() {
        let gate_slice = &gate_weights[eid * expert_stride..(eid + 1) * expert_stride];
        let up_slice = &up_weights[eid * expert_stride..(eid + 1) * expert_stride];
        let down_slice = &down_weights[eid * expert_stride..(eid + 1) * expert_stride];

        cpu.dequant_matmul(
            gate_slice,
            GgmlType::Q4K,
            &norm,
            &mut gate_buf,
            expert_inter_dim,
            n_tokens,
            dim,
        );
        cpu.dequant_matmul(
            up_slice,
            GgmlType::Q4K,
            &norm,
            &mut up_buf,
            expert_inter_dim,
            n_tokens,
            dim,
        );
        crate::compute::silu::silu_elementwise_mul(&mut gate_buf, &up_buf);
        cpu.dequant_matmul(
            down_slice,
            GgmlType::Q4K,
            &gate_buf,
            &mut down_buf,
            dim,
            n_tokens,
            expert_inter_dim,
        );
        for (dst, src) in expected_accum.iter_mut().zip(down_buf.iter()) {
            *dst += expert_weights[slot] * *src;
        }
    }

    let mut expected_hidden = hidden.clone();
    for (dst, src) in expected_hidden.iter_mut().zip(expected_accum.iter()) {
        *dst += *src;
    }

    let hidden_buf = MetalBuffer::from_slice(backend.device.device(), &hidden).unwrap();
    let norm_buf = MetalBuffer::from_slice(backend.device.device(), &norm_weights).unwrap();
    let router_buf = MetalBuffer::from_slice(backend.device.device(), &router_weights).unwrap();
    let gate_buf = MetalBuffer::from_slice(backend.device.device(), &gate_weights).unwrap();
    let up_buf = MetalBuffer::from_slice(backend.device.device(), &up_weights).unwrap();
    let down_buf = MetalBuffer::from_slice(backend.device.device(), &down_weights).unwrap();

    backend.ops.init_batch_scratches(&config, n_tokens);
    backend
        .ops
        .moe_ffn_gpu_resident_cached(
            &hidden_buf,
            &norm_buf,
            &router_buf,
            GgmlType::F32,
            &gate_buf,
            GgmlType::Q4K,
            &up_buf,
            GgmlType::Q4K,
            &down_buf,
            GgmlType::Q4K,
            n_tokens,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            expert_stride,
            expert_stride,
            expert_stride,
            eps,
            None,
        )
        .unwrap();

    let actual_hidden = unsafe { hidden_buf.as_slice::<f32>()[..dim].to_vec() };
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "resident MoE routed path mismatch: max_diff={diff}, rel_diff={}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );
}

#[test]
fn test_metal_backend_moe_ffn_gpu_resident_cached_resets_accumulator_between_in_encoder_invocations()
 {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 1usize;
    let n_expert = 2usize;
    let n_expert_used = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;
    let eps = 1e-5f32;

    let hidden_a: Vec<f32> = (0..dim)
        .map(|i| {
            if i < 128 {
                0.01
            } else {
                0.005 + (i - 128) as f32 * 0.00001
            }
        })
        .collect();
    let hidden_b: Vec<f32> = (0..dim)
        .map(|i| {
            if i < 128 {
                0.02
            } else {
                0.004 + (i - 128) as f32 * 0.00002
            }
        })
        .collect();
    let norm_weights = vec![1.0f32; dim];

    let mut router_weights = vec![0.0f32; n_expert * dim];
    router_weights[..128].fill(0.02);
    router_weights[dim..dim + 128].fill(0.01);

    let build_q4k_expert_tensor = |expert_nibbles: &[u8], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_nibbles.len() * rows * 144);
        for &nibble in expert_nibbles {
            for _ in 0..rows {
                bytes.extend_from_slice(&q4k_block_first128_constant(nibble));
            }
        }
        bytes
    };

    let gate_weights = build_q4k_expert_tensor(&[1, 2], expert_inter_dim);
    let up_weights = build_q4k_expert_tensor(&[2, 1], expert_inter_dim);
    let down_weights = build_q4k_expert_tensor(&[1, 2], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q4K, dim * expert_inter_dim);
    let config = crate::model::config::ModelConfig {
        architecture: "qwen35".to_string(),
        n_layers: 1,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: dim as u32,
        head_dim: dim as u32,
        intermediate_dim: expert_inter_dim as u32,
        context_length: 16,
        vocab_size: 32,
        rms_norm_eps: eps,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: crate::model::config::GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: Some(n_expert as u32),
        n_expert_used: Some(n_expert_used as u32),
        expert_intermediate_dim: Some(expert_inter_dim as u32),
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let expected_hidden = |hidden: &[f32]| {
        let mean_sq = hidden.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
        let norm: Vec<f32> = hidden
            .iter()
            .zip(norm_weights.iter())
            .map(|(&x, &w)| x * inv_rms * w)
            .collect();
        let mut router_logits = vec![0.0f32; n_expert];
        cpu.matmul(
            &router_weights,
            &norm,
            &mut router_logits,
            n_expert,
            n_tokens,
            dim,
        );
        let (expert_ids, expert_weights) =
            crate::model::moe_utils::top_k_softmax(&router_logits, n_expert_used);
        let mut expected_accum = vec![0.0f32; dim];
        let mut gate_buf = vec![0.0f32; expert_inter_dim];
        let mut up_buf = vec![0.0f32; expert_inter_dim];
        let mut down_buf = vec![0.0f32; dim];
        for (slot, &eid) in expert_ids.iter().enumerate() {
            let gate_slice = &gate_weights[eid * expert_stride..(eid + 1) * expert_stride];
            let up_slice = &up_weights[eid * expert_stride..(eid + 1) * expert_stride];
            let down_slice = &down_weights[eid * expert_stride..(eid + 1) * expert_stride];
            cpu.dequant_matmul(
                gate_slice,
                GgmlType::Q4K,
                &norm,
                &mut gate_buf,
                expert_inter_dim,
                n_tokens,
                dim,
            );
            cpu.dequant_matmul(
                up_slice,
                GgmlType::Q4K,
                &norm,
                &mut up_buf,
                expert_inter_dim,
                n_tokens,
                dim,
            );
            crate::compute::silu::silu_elementwise_mul(&mut gate_buf, &up_buf);
            cpu.dequant_matmul(
                down_slice,
                GgmlType::Q4K,
                &gate_buf,
                &mut down_buf,
                dim,
                n_tokens,
                expert_inter_dim,
            );
            for (dst, src) in expected_accum.iter_mut().zip(down_buf.iter()) {
                *dst += expert_weights[slot] * *src;
            }
        }
        let mut expected_hidden = hidden.to_vec();
        for (dst, src) in expected_hidden.iter_mut().zip(expected_accum.iter()) {
            *dst += *src;
        }
        expected_hidden
    };

    let expected_hidden_a = expected_hidden(&hidden_a);
    let expected_hidden_b = expected_hidden(&hidden_b);

    let hidden_a_buf = MetalBuffer::from_slice(backend.device.device(), &hidden_a).unwrap();
    let hidden_b_buf = MetalBuffer::from_slice(backend.device.device(), &hidden_b).unwrap();
    let norm_buf = MetalBuffer::from_slice(backend.device.device(), &norm_weights).unwrap();
    let router_buf = MetalBuffer::from_slice(backend.device.device(), &router_weights).unwrap();
    let gate_buf = MetalBuffer::from_slice(backend.device.device(), &gate_weights).unwrap();
    let up_buf = MetalBuffer::from_slice(backend.device.device(), &up_weights).unwrap();
    let down_buf = MetalBuffer::from_slice(backend.device.device(), &down_weights).unwrap();

    backend.ops.init_batch_scratches(&config, n_tokens);
    let scratch = backend.ops.moe_batch_scratch_view().unwrap();
    backend
        .ops
        .device
        .execute_sync(|encoder| {
            backend
                .ops
                .encode_moe_ffn_gpu_resident_cached_with_scratch(
                    encoder,
                    scratch,
                    &hidden_a_buf,
                    &norm_buf,
                    &router_buf,
                    GgmlType::F32,
                    &gate_buf,
                    GgmlType::Q4K,
                    &up_buf,
                    GgmlType::Q4K,
                    &down_buf,
                    GgmlType::Q4K,
                    n_tokens,
                    n_expert,
                    n_expert_used,
                    dim,
                    expert_inter_dim,
                    expert_stride,
                    expert_stride,
                    expert_stride,
                    eps,
                    None,
                    true,
                )?;
            backend
                .ops
                .encode_moe_ffn_gpu_resident_cached_with_scratch(
                    encoder,
                    scratch,
                    &hidden_b_buf,
                    &norm_buf,
                    &router_buf,
                    GgmlType::F32,
                    &gate_buf,
                    GgmlType::Q4K,
                    &up_buf,
                    GgmlType::Q4K,
                    &down_buf,
                    GgmlType::Q4K,
                    n_tokens,
                    n_expert,
                    n_expert_used,
                    dim,
                    expert_inter_dim,
                    expert_stride,
                    expert_stride,
                    expert_stride,
                    eps,
                    None,
                    true,
                )?;
            Ok(())
        })
        .unwrap();

    let actual_hidden_a = unsafe { hidden_a_buf.as_slice::<f32>()[..dim].to_vec() };
    let actual_hidden_b = unsafe { hidden_b_buf.as_slice::<f32>()[..dim].to_vec() };
    let diff_a = max_abs_diff(&actual_hidden_a, &expected_hidden_a);
    let scale_a = expected_hidden_a
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let diff_b = max_abs_diff(&actual_hidden_b, &expected_hidden_b);
    let scale_b = expected_hidden_b
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff_a / scale_a < 5e-3,
        "resident MoE first in-encoder invocation mismatch: rel_diff={}, max_diff={diff_a}",
        diff_a / scale_a,
    );
    assert!(
        diff_b / scale_b < 5e-3,
        "resident MoE second in-encoder invocation mismatch: rel_diff={}, max_diff={diff_b}",
        diff_b / scale_b,
    );
}

#[test]
fn test_metal_backend_moe_ffn_gpu_resident_cached_matches_cpu_for_multitoken_routed_experts() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 3usize;
    let n_expert = 4usize;
    let n_expert_used = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;
    let eps = 1e-5f32;

    let hidden: Vec<f32> = (0..n_tokens)
        .flat_map(|token| {
            (0..dim).map(move |i| {
                let bucket = i / 64;
                let base = 0.01 * (token + 1) as f32;
                base + bucket as f32 * 0.002 + (i % 64) as f32 * 0.00001
            })
        })
        .collect();
    let norm_weights = vec![1.0f32; dim];

    let mut router_weights = vec![0.0f32; n_expert * dim];
    for expert in 0..n_expert {
        let start = expert * 64;
        let end = start + 64;
        router_weights[expert * dim + start..expert * dim + end].fill(0.01 + expert as f32 * 0.002);
    }

    let build_q4k_expert_tensor = |expert_nibbles: &[u8], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_nibbles.len() * rows * 144);
        for &nibble in expert_nibbles {
            for _ in 0..rows {
                bytes.extend_from_slice(&q4k_block_first128_constant(nibble));
            }
        }
        bytes
    };

    let gate_weights = build_q4k_expert_tensor(&[1, 2, 3, 4], expert_inter_dim);
    let up_weights = build_q4k_expert_tensor(&[4, 3, 2, 1], expert_inter_dim);
    let down_weights = build_q4k_expert_tensor(&[1, 2, 3, 4], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q4K, dim * expert_inter_dim);
    let config = crate::model::config::ModelConfig {
        architecture: "qwen35".to_string(),
        n_layers: 1,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: dim as u32,
        head_dim: dim as u32,
        intermediate_dim: expert_inter_dim as u32,
        context_length: 16,
        vocab_size: 32,
        rms_norm_eps: eps,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: crate::model::config::GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: Some(n_expert as u32),
        n_expert_used: Some(n_expert_used as u32),
        expert_intermediate_dim: Some(expert_inter_dim as u32),
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let mut norm = vec![0.0f32; n_tokens * dim];
    for token in 0..n_tokens {
        let src = &hidden[token * dim..(token + 1) * dim];
        let dst = &mut norm[token * dim..(token + 1) * dim];
        let mean_sq = src.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
        for ((dst, &x), &w) in dst.iter_mut().zip(src.iter()).zip(norm_weights.iter()) {
            *dst = x * inv_rms * w;
        }
    }

    let mut router_logits = vec![0.0f32; n_tokens * n_expert];
    cpu.matmul(
        &router_weights,
        &norm,
        &mut router_logits,
        n_expert,
        n_tokens,
        dim,
    );

    let mut expected_accum = vec![0.0f32; n_tokens * dim];
    let mut gate_buf = vec![0.0f32; expert_inter_dim];
    let mut up_buf = vec![0.0f32; expert_inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    for token in 0..n_tokens {
        let token_norm = &norm[token * dim..(token + 1) * dim];
        let (expert_ids, expert_weights) = crate::model::moe_utils::top_k_softmax(
            &router_logits[token * n_expert..(token + 1) * n_expert],
            n_expert_used,
        );
        for (slot, &eid) in expert_ids.iter().enumerate() {
            let gate_slice = &gate_weights[eid * expert_stride..(eid + 1) * expert_stride];
            let up_slice = &up_weights[eid * expert_stride..(eid + 1) * expert_stride];
            let down_slice = &down_weights[eid * expert_stride..(eid + 1) * expert_stride];

            cpu.dequant_matmul(
                gate_slice,
                GgmlType::Q4K,
                token_norm,
                &mut gate_buf,
                expert_inter_dim,
                1,
                dim,
            );
            cpu.dequant_matmul(
                up_slice,
                GgmlType::Q4K,
                token_norm,
                &mut up_buf,
                expert_inter_dim,
                1,
                dim,
            );
            crate::compute::silu::silu_elementwise_mul(&mut gate_buf, &up_buf);
            cpu.dequant_matmul(
                down_slice,
                GgmlType::Q4K,
                &gate_buf,
                &mut down_buf,
                dim,
                1,
                expert_inter_dim,
            );
            let dst = &mut expected_accum[token * dim..(token + 1) * dim];
            for (dst, src) in dst.iter_mut().zip(down_buf.iter()) {
                *dst += expert_weights[slot] * *src;
            }
        }
    }

    let mut expected_hidden = hidden.clone();
    for (dst, src) in expected_hidden.iter_mut().zip(expected_accum.iter()) {
        *dst += *src;
    }

    let hidden_buf = MetalBuffer::from_slice(backend.device.device(), &hidden).unwrap();
    let norm_buf = MetalBuffer::from_slice(backend.device.device(), &norm_weights).unwrap();
    let router_buf = MetalBuffer::from_slice(backend.device.device(), &router_weights).unwrap();
    let gate_buf = MetalBuffer::from_slice(backend.device.device(), &gate_weights).unwrap();
    let up_buf = MetalBuffer::from_slice(backend.device.device(), &up_weights).unwrap();
    let down_buf = MetalBuffer::from_slice(backend.device.device(), &down_weights).unwrap();

    backend.ops.init_batch_scratches(&config, n_tokens);
    backend
        .ops
        .moe_ffn_gpu_resident_cached(
            &hidden_buf,
            &norm_buf,
            &router_buf,
            GgmlType::F32,
            &gate_buf,
            GgmlType::Q4K,
            &up_buf,
            GgmlType::Q4K,
            &down_buf,
            GgmlType::Q4K,
            n_tokens,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            expert_stride,
            expert_stride,
            expert_stride,
            eps,
            None,
        )
        .unwrap();

    let actual_hidden = unsafe { hidden_buf.as_slice::<f32>()[..n_tokens * dim].to_vec() };
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 2e-3,
        "resident MoE multitoken routed path mismatch: max_diff={diff}, rel_diff={}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );
}

#[test]
fn test_metal_backend_moe_ffn_gpu_resident_cached_matches_cpu_for_q5_down_experts() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 1usize;
    let n_expert = 2usize;
    let n_expert_used = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;
    let eps = 1e-5f32;

    let hidden: Vec<f32> = (0..dim)
        .map(|i| {
            if i < 128 {
                0.01
            } else {
                0.005 + (i - 128) as f32 * 0.00001
            }
        })
        .collect();
    let norm_weights = vec![1.0f32; dim];

    let mut router_weights = vec![0.0f32; n_expert * dim];
    router_weights[..128].fill(0.02);
    router_weights[dim..dim + 128].fill(0.01);

    let build_q4k_expert_tensor = |expert_nibbles: &[u8], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_nibbles.len() * rows * 144);
        for &nibble in expert_nibbles {
            for _ in 0..rows {
                bytes.extend_from_slice(&q4k_block_first128_constant(nibble));
            }
        }
        bytes
    };
    let build_q5k_expert_tensor = |expert_scales: &[f32], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_scales.len() * rows * 176);
        for &scale in expert_scales {
            for _ in 0..rows {
                bytes.extend_from_slice(&q5k_block_first128_scaled(scale));
            }
        }
        bytes
    };

    let gate_weights = build_q4k_expert_tensor(&[1, 2], expert_inter_dim);
    let up_weights = build_q4k_expert_tensor(&[2, 1], expert_inter_dim);
    let down_weights = build_q5k_expert_tensor(&[0.2, 0.4], dim);
    let q4_expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q4K, dim * expert_inter_dim);
    let q5_expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q5K, dim * expert_inter_dim);
    let config = crate::model::config::ModelConfig {
        architecture: "qwen35".to_string(),
        n_layers: 1,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: dim as u32,
        head_dim: dim as u32,
        intermediate_dim: expert_inter_dim as u32,
        context_length: 16,
        vocab_size: 32,
        rms_norm_eps: eps,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: crate::model::config::GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: Some(n_expert as u32),
        n_expert_used: Some(n_expert_used as u32),
        expert_intermediate_dim: Some(expert_inter_dim as u32),
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let mean_sq = hidden.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
    let norm: Vec<f32> = hidden
        .iter()
        .zip(norm_weights.iter())
        .map(|(&x, &w)| x * inv_rms * w)
        .collect();

    let mut router_logits = vec![0.0f32; n_expert];
    cpu.matmul(
        &router_weights,
        &norm,
        &mut router_logits,
        n_expert,
        n_tokens,
        dim,
    );
    let (expert_ids, expert_weights) =
        crate::model::moe_utils::top_k_softmax(&router_logits, n_expert_used);

    let mut expected_accum = vec![0.0f32; dim];
    let mut gate_buf = vec![0.0f32; expert_inter_dim];
    let mut up_buf = vec![0.0f32; expert_inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    for (slot, &eid) in expert_ids.iter().enumerate() {
        let gate_slice = &gate_weights[eid * q4_expert_stride..(eid + 1) * q4_expert_stride];
        let up_slice = &up_weights[eid * q4_expert_stride..(eid + 1) * q4_expert_stride];
        let down_slice = &down_weights[eid * q5_expert_stride..(eid + 1) * q5_expert_stride];

        cpu.dequant_matmul(
            gate_slice,
            GgmlType::Q4K,
            &norm,
            &mut gate_buf,
            expert_inter_dim,
            n_tokens,
            dim,
        );
        cpu.dequant_matmul(
            up_slice,
            GgmlType::Q4K,
            &norm,
            &mut up_buf,
            expert_inter_dim,
            n_tokens,
            dim,
        );
        crate::compute::silu::silu_elementwise_mul(&mut gate_buf, &up_buf);
        cpu.dequant_matmul(
            down_slice,
            GgmlType::Q5K,
            &gate_buf,
            &mut down_buf,
            dim,
            n_tokens,
            expert_inter_dim,
        );
        for (dst, src) in expected_accum.iter_mut().zip(down_buf.iter()) {
            *dst += expert_weights[slot] * *src;
        }
    }

    let mut expected_hidden = hidden.clone();
    for (dst, src) in expected_hidden.iter_mut().zip(expected_accum.iter()) {
        *dst += *src;
    }

    let hidden_buf = MetalBuffer::from_slice(backend.device.device(), &hidden).unwrap();
    let norm_buf = MetalBuffer::from_slice(backend.device.device(), &norm_weights).unwrap();
    let router_buf = MetalBuffer::from_slice(backend.device.device(), &router_weights).unwrap();
    let gate_buf = MetalBuffer::from_slice(backend.device.device(), &gate_weights).unwrap();
    let up_buf = MetalBuffer::from_slice(backend.device.device(), &up_weights).unwrap();
    let down_buf = MetalBuffer::from_slice(backend.device.device(), &down_weights).unwrap();

    backend.ops.init_batch_scratches(&config, n_tokens);
    backend
        .ops
        .moe_ffn_gpu_resident_cached(
            &hidden_buf,
            &norm_buf,
            &router_buf,
            GgmlType::F32,
            &gate_buf,
            GgmlType::Q4K,
            &up_buf,
            GgmlType::Q4K,
            &down_buf,
            GgmlType::Q5K,
            n_tokens,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            q4_expert_stride,
            q4_expert_stride,
            q5_expert_stride,
            eps,
            None,
        )
        .unwrap();

    let actual_hidden = unsafe { hidden_buf.as_slice::<f32>()[..dim].to_vec() };
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "resident MoE routed Q5-down path mismatch: max_diff={diff}, rel_diff={}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );
}

/// Build a Q6_K block where all 256 values dequantize to approx `d * 1 * (-31)`.
/// ql bytes = 0x11 (low nibble 1 for both halves), qh = 0, scales = 1.
fn q6k_block_constant(d_scale: f32) -> [u8; 210] {
    let mut block = [0u8; 210];
    // ql[0..128] = 0x11 → low nibble 1 for all quants
    block[0..128].fill(0x11);
    // qh[128..192] = 0x00 → no upper bits
    // scales[192..208] = 1 (signed i8)
    block[192..208].fill(1);
    // d (half) at [208..210]
    let d = half::f16::from_f32(d_scale).to_le_bytes();
    block[208] = d[0];
    block[209] = d[1];
    block
}

fn q8_0_block_constant(d_scale: f32) -> [u8; 34] {
    let mut block = [0u8; 34];
    let d = half::f16::from_f32(d_scale).to_le_bytes();
    block[0] = d[0];
    block[1] = d[1];
    block[2..].fill(1);
    block
}

#[test]
fn test_metal_backend_moe_ffn_gpu_resident_cached_matches_cpu_for_q6_down_experts() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    // Use realistic Qwen3 Coder dimensions to catch size-dependent bugs.
    // Keep dim=256 to avoid half-precision overflow in test data, but use
    // non-trivial K=768 to exercise multi-block-per-row Q6K dequant.
    let n_tokens = 1usize;
    let n_expert = 2usize;
    let n_expert_used = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 768usize;
    let eps = 1e-5f32;

    let hidden: Vec<f32> = (0..dim)
        .map(|i| {
            if i < 128 {
                0.01
            } else {
                0.005 + (i - 128) as f32 * 0.00001
            }
        })
        .collect();
    let norm_weights = vec![1.0f32; dim];

    let mut router_weights = vec![0.0f32; n_expert * dim];
    router_weights[..128].fill(0.02);
    router_weights[dim..dim + 128].fill(0.01);

    // Each expert weight matrix has `rows` rows and `cols` columns.
    // Each row has ceil(cols/256) Q4K blocks or ceil(cols/256) Q6K blocks.
    let gate_k = dim; // gate: [expert_inter_dim, dim]
    let gate_blocks_per_row = gate_k.div_ceil(256);
    let gate_weights: Vec<u8> = [1u8, 2u8]
        .iter()
        .flat_map(|&nibble| {
            (0..expert_inter_dim).flat_map(move |_| {
                (0..gate_blocks_per_row)
                    .flat_map(move |_| q4k_block_first128_constant(nibble).into_iter())
            })
        })
        .collect();
    let up_weights: Vec<u8> = [2u8, 1u8]
        .iter()
        .flat_map(|&nibble| {
            (0..expert_inter_dim).flat_map(move |_| {
                (0..gate_blocks_per_row)
                    .flat_map(move |_| q4k_block_first128_constant(nibble).into_iter())
            })
        })
        .collect();
    let down_k = expert_inter_dim; // down: [dim, expert_inter_dim]
    let down_blocks_per_row = down_k.div_ceil(256);
    let down_weights: Vec<u8> = [0.02f32, 0.04f32]
        .iter()
        .flat_map(|&scale| {
            (0..dim).flat_map(move |_| {
                (0..down_blocks_per_row).flat_map(move |_| q6k_block_constant(scale).into_iter())
            })
        })
        .collect();
    let q4_expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q4K, dim * expert_inter_dim);
    let q6_expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q6K, dim * expert_inter_dim);

    let config = crate::model::config::ModelConfig {
        architecture: "qwen35".to_string(),
        n_layers: 1,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: dim as u32,
        head_dim: dim as u32,
        intermediate_dim: expert_inter_dim as u32,
        context_length: 16,
        vocab_size: 32,
        rms_norm_eps: eps,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: crate::model::config::GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: Some(n_expert as u32),
        n_expert_used: Some(n_expert_used as u32),
        expert_intermediate_dim: Some(expert_inter_dim as u32),
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let mean_sq = hidden.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
    let norm: Vec<f32> = hidden
        .iter()
        .zip(norm_weights.iter())
        .map(|(&x, &w)| x * inv_rms * w)
        .collect();

    let mut router_logits = vec![0.0f32; n_expert];
    cpu.matmul(
        &router_weights,
        &norm,
        &mut router_logits,
        n_expert,
        n_tokens,
        dim,
    );
    let (expert_ids, expert_weights) =
        crate::model::moe_utils::top_k_softmax(&router_logits, n_expert_used);

    let mut expected_accum = vec![0.0f32; dim];
    let mut gate_buf = vec![0.0f32; expert_inter_dim];
    let mut up_buf = vec![0.0f32; expert_inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    for (slot, &eid) in expert_ids.iter().enumerate() {
        let gate_slice = &gate_weights[eid * q4_expert_stride..(eid + 1) * q4_expert_stride];
        let up_slice = &up_weights[eid * q4_expert_stride..(eid + 1) * q4_expert_stride];
        let down_slice = &down_weights[eid * q6_expert_stride..(eid + 1) * q6_expert_stride];

        cpu.dequant_matmul(
            gate_slice,
            GgmlType::Q4K,
            &norm,
            &mut gate_buf,
            expert_inter_dim,
            n_tokens,
            dim,
        );
        cpu.dequant_matmul(
            up_slice,
            GgmlType::Q4K,
            &norm,
            &mut up_buf,
            expert_inter_dim,
            n_tokens,
            dim,
        );
        crate::compute::silu::silu_elementwise_mul(&mut gate_buf, &up_buf);
        cpu.dequant_matmul(
            down_slice,
            GgmlType::Q6K,
            &gate_buf,
            &mut down_buf,
            dim,
            n_tokens,
            expert_inter_dim,
        );
        for (dst, src) in expected_accum.iter_mut().zip(down_buf.iter()) {
            *dst += expert_weights[slot] * *src;
        }
    }

    let mut expected_hidden = hidden.clone();
    for (dst, src) in expected_hidden.iter_mut().zip(expected_accum.iter()) {
        *dst += *src;
    }

    let hidden_buf = MetalBuffer::from_slice(backend.device.device(), &hidden).unwrap();
    let norm_buf = MetalBuffer::from_slice(backend.device.device(), &norm_weights).unwrap();
    let router_buf = MetalBuffer::from_slice(backend.device.device(), &router_weights).unwrap();
    let gate_buf = MetalBuffer::from_slice(backend.device.device(), &gate_weights).unwrap();
    let up_buf = MetalBuffer::from_slice(backend.device.device(), &up_weights).unwrap();
    let down_buf = MetalBuffer::from_slice(backend.device.device(), &down_weights).unwrap();

    backend.ops.init_batch_scratches(&config, n_tokens);
    backend
        .ops
        .moe_ffn_gpu_resident_cached(
            &hidden_buf,
            &norm_buf,
            &router_buf,
            GgmlType::F32,
            &gate_buf,
            GgmlType::Q4K,
            &up_buf,
            GgmlType::Q4K,
            &down_buf,
            GgmlType::Q6K,
            n_tokens,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            q4_expert_stride,
            q4_expert_stride,
            q6_expert_stride,
            eps,
            None,
        )
        .unwrap();

    let actual_hidden = unsafe { hidden_buf.as_slice::<f32>()[..dim].to_vec() };
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 5e-3,
        "resident MoE routed Q6K-down path mismatch: max_diff={diff}, rel_diff={}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );
}

#[test]
fn test_metal_backend_moe_mul_mat_selected_single_token_down_uses_slot_major_input_rows() {
    let _env_lock = lock_env_test();
    let _on = EnvVarGuard::set("AX_QWEN35_SELECTED_Q5K_MATVEC", "1");
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;

    let build_q5k_expert_tensor = |expert_scales: &[f32], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_scales.len() * rows * 176);
        for &scale in expert_scales {
            for _ in 0..rows {
                bytes.extend_from_slice(&q5k_block_first128_scaled(scale));
            }
        }
        bytes
    };

    let weights = build_q5k_expert_tensor(&[0.2, 0.2], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q5K, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, GgmlType::Q5K, "selected down").unwrap();

    let mut input = vec![0.0f32; n_selected * expert_inter_dim];
    input[..128].fill(1.0);
    input[expert_inter_dim..expert_inter_dim + 128].fill(2.0);

    let selected_experts = vec![0i32, 1i32];
    let weights_buf = MetalBuffer::from_slice(backend.device.device(), &weights).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let output_buf = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights_buf,
                &input_buf,
                &selected_buf,
                &output_buf,
                GgmlType::Q5K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                true,
            )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let mut expected = vec![0.0f32; n_selected * dim];
    for slot in 0..n_selected {
        let weight_slice = &weights[slot * expert_stride..(slot + 1) * expert_stride];
        cpu.dequant_matmul(
            weight_slice,
            GgmlType::Q5K,
            &input[slot * expert_inter_dim..(slot + 1) * expert_inter_dim],
            &mut expected[slot * dim..(slot + 1) * dim],
            dim,
            1,
            expert_inter_dim,
        );
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "selected single-token Q5K down mismatch: rel_diff={}, max_diff={diff}, actual_slot0[0..8]={:?}, expected_slot0[0..8]={:?}, actual_slot1[0..8]={:?}, expected_slot1[0..8]={:?}",
        diff / scale,
        &actual[..8],
        &expected[..8],
        &actual[dim..dim + 8],
        &expected[dim..dim + 8],
    );
}

#[test]
fn test_metal_backend_moe_mul_mat_selected_single_token_matches_cpu_q4_k() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;

    let build_q4k_expert_tensor = |expert_nibbles: &[u8], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_nibbles.len() * rows * 144);
        for &nibble in expert_nibbles {
            for _ in 0..rows {
                bytes.extend_from_slice(&q4k_block_first128_constant(nibble));
            }
        }
        bytes
    };

    let weights = build_q4k_expert_tensor(&[1, 2], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q4K, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, GgmlType::Q4K, "selected q4k").unwrap();

    let mut input = vec![0.0f32; n_selected * expert_inter_dim];
    input[..128].fill(1.0);
    input[expert_inter_dim..expert_inter_dim + 128].fill(2.0);

    let selected_experts = vec![0i32, 1i32];
    let weights_buf = MetalBuffer::from_slice(backend.device.device(), &weights).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let output_buf = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights_buf,
                &input_buf,
                &selected_buf,
                &output_buf,
                GgmlType::Q4K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                true,
            )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let mut expected = vec![0.0f32; n_selected * dim];
    for slot in 0..n_selected {
        let weight_slice = &weights[slot * expert_stride..(slot + 1) * expert_stride];
        cpu.dequant_matmul(
            weight_slice,
            GgmlType::Q4K,
            &input[slot * expert_inter_dim..(slot + 1) * expert_inter_dim],
            &mut expected[slot * dim..(slot + 1) * dim],
            dim,
            1,
            expert_inter_dim,
        );
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "selected single-token Q4K mismatch: rel_diff={}, max_diff={diff}, actual_slot0[0..8]={:?}, expected_slot0[0..8]={:?}, actual_slot1[0..8]={:?}, expected_slot1[0..8]={:?}",
        diff / scale,
        &actual[..8],
        &expected[..8],
        &actual[dim..dim + 8],
        &expected[dim..dim + 8],
    );
}

#[test]
fn test_metal_backend_moe_mul_mat_selected_single_token_matches_cpu_q5_k() {
    let _env_lock = lock_env_test();
    let _on = EnvVarGuard::set("AX_QWEN35_SELECTED_Q5K_MATVEC", "1");
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;

    let build_q5k_expert_tensor = |expert_scales: &[f32], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_scales.len() * rows * 176);
        for &scale in expert_scales {
            for _ in 0..rows {
                bytes.extend_from_slice(&q5k_block_first128_scaled(scale));
            }
        }
        bytes
    };

    let weights = build_q5k_expert_tensor(&[0.2, 0.35], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q5K, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, GgmlType::Q5K, "selected q5k").unwrap();

    let mut input = vec![0.0f32; expert_inter_dim];
    input[..128].fill(1.0);

    let selected_experts = vec![0i32, 1i32];
    let weights_buf = MetalBuffer::from_slice(backend.device.device(), &weights).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let output_buf = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights_buf,
                &input_buf,
                &selected_buf,
                &output_buf,
                GgmlType::Q5K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                false,
            )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let mut expected = vec![0.0f32; n_selected * dim];
    for slot in 0..n_selected {
        let eid = selected_experts[slot] as usize;
        let weight_slice = &weights[eid * expert_stride..(eid + 1) * expert_stride];
        cpu.dequant_matmul(
            weight_slice,
            GgmlType::Q5K,
            &input,
            &mut expected[slot * dim..(slot + 1) * dim],
            dim,
            1,
            expert_inter_dim,
        );
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "selected single-token Q5K mismatch: rel_diff={}, max_diff={diff}, actual_slot0[0..8]={:?}, expected_slot0[0..8]={:?}, actual_slot1[0..8]={:?}, expected_slot1[0..8]={:?}",
        diff / scale,
        &actual[..8],
        &expected[..8],
        &actual[dim..dim + 8],
        &expected[dim..dim + 8],
    );
}

#[test]
fn test_metal_backend_moe_mul_mat_selected_single_token_matches_cpu_q6_k() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;

    let build_q6k_expert_tensor = |expert_scales: &[f32], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_scales.len() * rows * 210);
        for &scale in expert_scales {
            for _ in 0..rows {
                bytes.extend_from_slice(&q6k_block_constant(scale));
            }
        }
        bytes
    };

    let weights = build_q6k_expert_tensor(&[0.02, 0.04], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q6K, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, GgmlType::Q6K, "selected q6k").unwrap();

    let mut input = vec![0.0f32; expert_inter_dim];
    input[..128].fill(0.01);

    let selected_experts = vec![0i32, 1i32];
    let weights_buf = MetalBuffer::from_slice(backend.device.device(), &weights).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let output_buf = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights_buf,
                &input_buf,
                &selected_buf,
                &output_buf,
                GgmlType::Q6K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                false,
            )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let mut expected = vec![0.0f32; n_selected * dim];
    for slot in 0..n_selected {
        let eid = selected_experts[slot] as usize;
        let weight_slice = &weights[eid * expert_stride..(eid + 1) * expert_stride];
        cpu.dequant_matmul(
            weight_slice,
            GgmlType::Q6K,
            &input,
            &mut expected[slot * dim..(slot + 1) * dim],
            dim,
            1,
            expert_inter_dim,
        );
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "selected single-token Q6K mismatch: rel_diff={}, max_diff={diff}, actual_slot0[0..8]={:?}, expected_slot0[0..8]={:?}, actual_slot1[0..8]={:?}, expected_slot1[0..8]={:?}",
        diff / scale,
        &actual[..8],
        &expected[..8],
        &actual[dim..dim + 8],
        &expected[dim..dim + 8],
    );
}

#[test]
fn test_metal_backend_moe_mul_mat_selected_single_token_matches_cpu_q8_0() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;
    let q8_blocks_per_row = expert_inter_dim.div_ceil(32);

    let build_q8_0_expert_tensor = |expert_scales: &[f32], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_scales.len() * rows * q8_blocks_per_row * 34);
        for &scale in expert_scales {
            for _ in 0..rows {
                for _ in 0..q8_blocks_per_row {
                    bytes.extend_from_slice(&q8_0_block_constant(scale));
                }
            }
        }
        bytes
    };

    let weights = build_q8_0_expert_tensor(&[0.02, 0.05], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q8_0, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, GgmlType::Q8_0, "selected q8_0").unwrap();

    let mut input = vec![0.0f32; expert_inter_dim];
    input[..128].fill(0.5);

    let selected_experts = vec![0i32, 1i32];
    let weights_buf = MetalBuffer::from_slice(backend.device.device(), &weights).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let output_buf = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights_buf,
                &input_buf,
                &selected_buf,
                &output_buf,
                GgmlType::Q8_0,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                false,
            )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let mut expected = vec![0.0f32; n_selected * dim];
    for slot in 0..n_selected {
        let eid = selected_experts[slot] as usize;
        let weight_slice = &weights[eid * expert_stride..(eid + 1) * expert_stride];
        cpu.dequant_matmul(
            weight_slice,
            GgmlType::Q8_0,
            &input,
            &mut expected[slot * dim..(slot + 1) * dim],
            dim,
            1,
            expert_inter_dim,
        );
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "selected single-token Q8_0 mismatch: rel_diff={}, max_diff={diff}, actual_slot0[0..8]={:?}, expected_slot0[0..8]={:?}, actual_slot1[0..8]={:?}, expected_slot1[0..8]={:?}",
        diff / scale,
        &actual[..8],
        &expected[..8],
        &actual[dim..dim + 8],
        &expected[dim..dim + 8],
    );
}

#[test]
fn test_metal_backend_moe_mul_mat_selected_single_token_weighted_matches_cpu_q5_k() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;

    let build_q5k_expert_tensor = |expert_scales: &[f32], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_scales.len() * rows * 176);
        for &scale in expert_scales {
            for _ in 0..rows {
                bytes.extend_from_slice(&q5k_block_first128_scaled(scale));
            }
        }
        bytes
    };

    let weights = build_q5k_expert_tensor(&[0.2, 0.35], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q5K, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, GgmlType::Q5K, "selected weighted").unwrap();

    let mut input = vec![0.0f32; n_selected * expert_inter_dim];
    input[..128].fill(1.0);
    input[expert_inter_dim..expert_inter_dim + 128].fill(2.0);

    let selected_experts = vec![0i32, 1i32];
    let expert_weights = vec![0.25f32, 0.75f32];

    let weights_buf = MetalBuffer::from_slice(backend.device.device(), &weights).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let expert_weights_buf =
        MetalBuffer::from_slice(backend.device.device(), &expert_weights).unwrap();
    let output_buf =
        MetalBuffer::new(backend.device.device(), dim * std::mem::size_of::<f32>()).unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend
                .ops
                .encode_moe_mul_mat_selected_single_token_weighted(
                    encoder,
                    &weights_buf,
                    &input_buf,
                    &selected_buf,
                    &expert_weights_buf,
                    &output_buf,
                    GgmlType::Q5K,
                    dim as u32,
                    expert_inter_dim as u32,
                    n_selected as u32,
                    blocks_per_expert,
                )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..dim].to_vec() };
    let mut expected = vec![0.0f32; dim];
    let mut scratch = vec![0.0f32; dim];
    for slot in 0..n_selected {
        let weight_slice = &weights[slot * expert_stride..(slot + 1) * expert_stride];
        scratch.fill(0.0);
        cpu.dequant_matmul(
            weight_slice,
            GgmlType::Q5K,
            &input[slot * expert_inter_dim..(slot + 1) * expert_inter_dim],
            &mut scratch,
            dim,
            1,
            expert_inter_dim,
        );
        for (dst, value) in expected.iter_mut().zip(&scratch) {
            *dst += expert_weights[slot] * *value;
        }
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "selected weighted Q5K mismatch: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual[..8],
        &expected[..8],
    );
}

#[test]
fn test_metal_backend_moe_mul_mat_selected_single_token_weighted_matches_cpu_q6_k() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;

    let build_q6k_expert_tensor = |expert_scales: &[f32], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_scales.len() * rows * 210);
        for &scale in expert_scales {
            for _ in 0..rows {
                bytes.extend_from_slice(&q6k_block_constant(scale));
            }
        }
        bytes
    };

    let weights = build_q6k_expert_tensor(&[0.02, 0.04], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q6K, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, GgmlType::Q6K, "selected weighted").unwrap();

    let mut input = vec![0.0f32; n_selected * expert_inter_dim];
    input[..128].fill(0.01);
    input[expert_inter_dim..expert_inter_dim + 128].fill(0.02);

    let selected_experts = vec![0i32, 1i32];
    let expert_weights = vec![0.25f32, 0.75f32];

    let weights_buf = MetalBuffer::from_slice(backend.device.device(), &weights).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let expert_weights_buf =
        MetalBuffer::from_slice(backend.device.device(), &expert_weights).unwrap();
    let output_buf =
        MetalBuffer::new(backend.device.device(), dim * std::mem::size_of::<f32>()).unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend
                .ops
                .encode_moe_mul_mat_selected_single_token_weighted(
                    encoder,
                    &weights_buf,
                    &input_buf,
                    &selected_buf,
                    &expert_weights_buf,
                    &output_buf,
                    GgmlType::Q6K,
                    dim as u32,
                    expert_inter_dim as u32,
                    n_selected as u32,
                    blocks_per_expert,
                )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..dim].to_vec() };
    let mut expected = vec![0.0f32; dim];
    let mut scratch = vec![0.0f32; dim];
    for slot in 0..n_selected {
        let weight_slice = &weights[slot * expert_stride..(slot + 1) * expert_stride];
        scratch.fill(0.0);
        cpu.dequant_matmul(
            weight_slice,
            GgmlType::Q6K,
            &input[slot * expert_inter_dim..(slot + 1) * expert_inter_dim],
            &mut scratch,
            dim,
            1,
            expert_inter_dim,
        );
        for (dst, value) in expected.iter_mut().zip(&scratch) {
            *dst += expert_weights[slot] * *value;
        }
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "selected weighted Q6K mismatch: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual[..8],
        &expected[..8],
    );
}

#[test]
fn test_metal_backend_moe_mul_mat_selected_single_token_weighted_matches_cpu_q8_0() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;
    let q8_blocks_per_row = expert_inter_dim.div_ceil(32);

    let build_q8_0_expert_tensor = |expert_scales: &[f32], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_scales.len() * rows * q8_blocks_per_row * 34);
        for &scale in expert_scales {
            for _ in 0..rows {
                for _ in 0..q8_blocks_per_row {
                    bytes.extend_from_slice(&q8_0_block_constant(scale));
                }
            }
        }
        bytes
    };

    let weights = build_q8_0_expert_tensor(&[0.02, 0.05], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q8_0, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, GgmlType::Q8_0, "selected weighted")
            .unwrap();

    let mut input = vec![0.0f32; n_selected * expert_inter_dim];
    input[..128].fill(0.5);
    input[expert_inter_dim..expert_inter_dim + 128].fill(1.0);

    let selected_experts = vec![0i32, 1i32];
    let expert_weights = vec![0.4f32, 0.6f32];

    let weights_buf = MetalBuffer::from_slice(backend.device.device(), &weights).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let expert_weights_buf =
        MetalBuffer::from_slice(backend.device.device(), &expert_weights).unwrap();
    let output_buf =
        MetalBuffer::new(backend.device.device(), dim * std::mem::size_of::<f32>()).unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend
                .ops
                .encode_moe_mul_mat_selected_single_token_weighted(
                    encoder,
                    &weights_buf,
                    &input_buf,
                    &selected_buf,
                    &expert_weights_buf,
                    &output_buf,
                    GgmlType::Q8_0,
                    dim as u32,
                    expert_inter_dim as u32,
                    n_selected as u32,
                    blocks_per_expert,
                )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..dim].to_vec() };
    let mut expected = vec![0.0f32; dim];
    let mut scratch = vec![0.0f32; dim];
    for slot in 0..n_selected {
        let weight_slice = &weights[slot * expert_stride..(slot + 1) * expert_stride];
        scratch.fill(0.0);
        cpu.dequant_matmul(
            weight_slice,
            GgmlType::Q8_0,
            &input[slot * expert_inter_dim..(slot + 1) * expert_inter_dim],
            &mut scratch,
            dim,
            1,
            expert_inter_dim,
        );
        for (dst, value) in expected.iter_mut().zip(&scratch) {
            *dst += expert_weights[slot] * *value;
        }
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "selected weighted Q8_0 mismatch: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual[..8],
        &expected[..8],
    );
}

#[test]
fn test_metal_backend_moe_fused_silu_down_selected_weighted_matches_cpu_q5_k() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;

    let build_q5k_expert_tensor = |expert_scales: &[f32], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_scales.len() * rows * 176);
        for &scale in expert_scales {
            for _ in 0..rows {
                bytes.extend_from_slice(&q5k_block_first128_scaled(scale));
            }
        }
        bytes
    };

    let weights = build_q5k_expert_tensor(&[0.2, 0.35], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q5K, dim * expert_inter_dim);
    let blocks_per_expert = MetalOps::moe_blocks_per_expert(
        expert_stride,
        GgmlType::Q5K,
        "selected fused silu weighted",
    )
    .unwrap();

    let mut gate = vec![0.0f32; n_selected * expert_inter_dim];
    let mut up = vec![0.0f32; n_selected * expert_inter_dim];
    gate[..128].fill(0.5);
    up[..128].fill(1.25);
    gate[expert_inter_dim..expert_inter_dim + 128].fill(-0.75);
    up[expert_inter_dim..expert_inter_dim + 128].fill(0.8);

    let selected_experts = vec![0i32, 1i32];
    let expert_weights = vec![0.25f32, 0.75f32];

    let weights_buf = MetalBuffer::from_slice(backend.device.device(), &weights).unwrap();
    let gate_buf = MetalBuffer::from_slice(backend.device.device(), &gate).unwrap();
    let up_buf = MetalBuffer::from_slice(backend.device.device(), &up).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let expert_weights_buf =
        MetalBuffer::from_slice(backend.device.device(), &expert_weights).unwrap();
    let output_buf =
        MetalBuffer::new(backend.device.device(), dim * std::mem::size_of::<f32>()).unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend
                .ops
                .encode_moe_fused_silu_down_selected_single_token_q5_k(
                    encoder,
                    &weights_buf,
                    &gate_buf,
                    &up_buf,
                    &selected_buf,
                    &expert_weights_buf,
                    &output_buf,
                    dim as u32,
                    expert_inter_dim as u32,
                    n_selected as u32,
                    blocks_per_expert,
                )
        })
        .unwrap();

    let actual = unsafe { output_buf.as_slice::<f32>()[..dim].to_vec() };
    let mut expected = vec![0.0f32; dim];
    let mut scratch_in = vec![0.0f32; expert_inter_dim];
    let mut scratch_out = vec![0.0f32; dim];
    for slot in 0..n_selected {
        let weight_slice = &weights[slot * expert_stride..(slot + 1) * expert_stride];
        for i in 0..expert_inter_dim {
            let gate_v = gate[slot * expert_inter_dim + i];
            let up_v = up[slot * expert_inter_dim + i];
            scratch_in[i] = (gate_v / (1.0 + (-gate_v).exp())) * up_v;
        }
        scratch_out.fill(0.0);
        cpu.dequant_matmul(
            weight_slice,
            GgmlType::Q5K,
            &scratch_in,
            &mut scratch_out,
            dim,
            1,
            expert_inter_dim,
        );
        for (dst, value) in expected.iter_mut().zip(&scratch_out) {
            *dst += expert_weights[slot] * *value;
        }
    }

    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "selected fused silu weighted Q5K mismatch: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual[..8],
        &expected[..8],
    );
}

#[test]
fn test_metal_backend_dense_row_dot_sigmoid_mul_inplace_matches_cpu() {
    let backend = MetalBackend::new().unwrap();

    let k = 256usize;
    let n = 512usize;
    let mut row = vec![0.0f32; k];
    let mut x = vec![0.0f32; k];
    let mut out = vec![0.0f32; n];
    for i in 0..k {
        row[i] = ((i % 13) as f32 - 6.0) * 0.0625;
        x[i] = ((i % 7) as f32 - 3.0) * 0.125;
    }
    for (i, v) in out.iter_mut().enumerate() {
        *v = ((i % 17) as f32 - 8.0) * 0.5;
    }

    let gate: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
    let scale = 1.0f32 / (1.0 + (-gate).exp());
    let expected: Vec<f32> = out.iter().map(|v| v * scale).collect();

    let row_buf = MetalBuffer::from_slice(backend.device.device(), &row).unwrap();
    let x_buf = MetalBuffer::from_slice(backend.device.device(), &x).unwrap();
    let out_buf = MetalBuffer::from_slice(backend.device.device(), &out).unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend
                .ops
                .elementwise
                .encode_dense_row_dot_sigmoid_mul_inplace(
                    encoder, &row_buf, &x_buf, &out_buf, k as u32, n as u32,
                );
            Ok(())
        })
        .unwrap();

    let actual = unsafe { out_buf.as_slice::<f32>()[..n].to_vec() };
    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-5,
        "dense_row_dot_sigmoid_mul mismatch: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual[..8],
        &expected[..8],
    );
}

#[test]
fn test_metal_backend_moe_mul_mat_selected_single_token_pair_matches_separate_q5_k() {
    let _env_lock = lock_env_test();
    let _guard = EnvVarGuard {
        key: "AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC",
        previous: std::env::var_os("AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC"),
    };
    unsafe { std::env::set_var("AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC", "1") };

    let backend = MetalBackend::new().unwrap();

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;

    let build_q5k_expert_tensor = |expert_scales: &[f32], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_scales.len() * rows * 176);
        for &scale in expert_scales {
            for _ in 0..rows {
                bytes.extend_from_slice(&q5k_block_first128_scaled(scale));
            }
        }
        bytes
    };

    let weights0 = build_q5k_expert_tensor(&[0.2, 0.35], dim);
    let weights1 = build_q5k_expert_tensor(&[0.45, 0.6], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q5K, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, GgmlType::Q5K, "selected pair").unwrap();

    let mut input = vec![0.0f32; expert_inter_dim];
    input[..128].fill(1.0);

    let selected_experts = vec![0i32, 1i32];
    let weights0_buf = MetalBuffer::from_slice(backend.device.device(), &weights0).unwrap();
    let weights1_buf = MetalBuffer::from_slice(backend.device.device(), &weights1).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let pair_out0 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let pair_out1 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let sep_out0 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let sep_out1 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token_pair(
                encoder,
                &weights0_buf,
                &weights1_buf,
                &input_buf,
                &selected_buf,
                &pair_out0,
                &pair_out1,
                GgmlType::Q5K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                blocks_per_expert,
                false,
                true,
            )
        })
        .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights0_buf,
                &input_buf,
                &selected_buf,
                &sep_out0,
                GgmlType::Q5K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                false,
            )?;
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights1_buf,
                &input_buf,
                &selected_buf,
                &sep_out1,
                GgmlType::Q5K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                false,
            )
        })
        .unwrap();

    let pair0 = unsafe { pair_out0.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let pair1 = unsafe { pair_out1.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let sep0 = unsafe { sep_out0.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let sep1 = unsafe { sep_out1.as_slice::<f32>()[..n_selected * dim].to_vec() };

    let (idx0, diff0) = max_abs_diff_with_index(&pair0, &sep0);
    let (idx1, diff1) = max_abs_diff_with_index(&pair1, &sep1);
    let scale0 = sep0
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let scale1 = sep1
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff0 / scale0 < 1e-3 && diff1 / scale1 < 1e-3,
        "selected pair path mismatch: idx0={idx0} slot0={} row0={} diff0={diff0}, idx1={idx1} slot1={} row1={} diff1={diff1}, pair0[0..8]={:?}, sep0[0..8]={:?}, pair1[0..8]={:?}, sep1[0..8]={:?}",
        idx0 / dim,
        idx0 % dim,
        idx1 / dim,
        idx1 % dim,
        &pair0[..8],
        &sep0[..8],
        &pair1[..8],
        &sep1[..8],
    );
}

#[test]
#[ignore = "Q4_K selected pair kernel is still experimental and not used by the default decode path"]
fn test_metal_backend_moe_mul_mat_selected_single_token_pair_matches_separate_q4_k() {
    let backend = MetalBackend::new().unwrap();

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;

    let build_q4k_expert_tensor = |expert_nibbles: &[u8], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_nibbles.len() * rows * 144);
        for &nibble in expert_nibbles {
            for _ in 0..rows {
                bytes.extend_from_slice(&q4k_block_first128_constant(nibble));
            }
        }
        bytes
    };

    let weights0 = build_q4k_expert_tensor(&[1, 2], dim);
    let weights1 = build_q4k_expert_tensor(&[3, 4], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q4K, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, GgmlType::Q4K, "selected pair").unwrap();

    let mut input = vec![0.0f32; expert_inter_dim];
    input[..128].fill(1.0);

    let selected_experts = vec![0i32, 1i32];
    let weights0_buf = MetalBuffer::from_slice(backend.device.device(), &weights0).unwrap();
    let weights1_buf = MetalBuffer::from_slice(backend.device.device(), &weights1).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let pair_out0 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let pair_out1 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let sep_out0 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let sep_out1 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token_pair(
                encoder,
                &weights0_buf,
                &weights1_buf,
                &input_buf,
                &selected_buf,
                &pair_out0,
                &pair_out1,
                GgmlType::Q4K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                blocks_per_expert,
                false,
                false,
            )
        })
        .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights0_buf,
                &input_buf,
                &selected_buf,
                &sep_out0,
                GgmlType::Q4K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                false,
            )?;
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights1_buf,
                &input_buf,
                &selected_buf,
                &sep_out1,
                GgmlType::Q4K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                false,
            )
        })
        .unwrap();

    let pair0 = unsafe { pair_out0.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let pair1 = unsafe { pair_out1.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let sep0 = unsafe { sep_out0.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let sep1 = unsafe { sep_out1.as_slice::<f32>()[..n_selected * dim].to_vec() };

    let (idx0, diff0) = max_abs_diff_with_index(&pair0, &sep0);
    let (idx1, diff1) = max_abs_diff_with_index(&pair1, &sep1);
    let scale0 = sep0
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let scale1 = sep1
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff0 / scale0 < 1e-3 && diff1 / scale1 < 1e-3,
        "selected Q4_K pair path mismatch: idx0={idx0} slot0={} row0={} diff0={diff0}, idx1={idx1} slot1={} row1={} diff1={diff1}, pair0[0..8]={:?}, sep0[0..8]={:?}, pair1[0..8]={:?}, sep1[0..8]={:?}",
        idx0 / dim,
        idx0 % dim,
        idx1 / dim,
        idx1 % dim,
        &pair0[..8],
        &sep0[..8],
        &pair1[..8],
        &sep1[..8],
    );
}

#[test]
fn test_metal_backend_moe_mul_mat_selected_single_token_pair_matches_separate_q6_k() {
    let backend = MetalBackend::new().unwrap();

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;

    let build_q6k_expert_tensor = |expert_scales: &[f32], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_scales.len() * rows * 210);
        for &scale in expert_scales {
            for _ in 0..rows {
                bytes.extend_from_slice(&q6k_block_constant(scale));
            }
        }
        bytes
    };

    let weights0 = build_q6k_expert_tensor(&[0.04, 0.07], dim);
    let weights1 = build_q6k_expert_tensor(&[0.09, 0.12], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q6K, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, GgmlType::Q6K, "selected pair").unwrap();

    let mut input = vec![0.0f32; expert_inter_dim];
    input[..128].fill(1.0);

    let selected_experts = vec![0i32, 1i32];
    let weights0_buf = MetalBuffer::from_slice(backend.device.device(), &weights0).unwrap();
    let weights1_buf = MetalBuffer::from_slice(backend.device.device(), &weights1).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let pair_out0 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let pair_out1 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let sep_out0 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let sep_out1 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token_pair(
                encoder,
                &weights0_buf,
                &weights1_buf,
                &input_buf,
                &selected_buf,
                &pair_out0,
                &pair_out1,
                GgmlType::Q6K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                blocks_per_expert,
                false,
                false,
            )
        })
        .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights0_buf,
                &input_buf,
                &selected_buf,
                &sep_out0,
                GgmlType::Q6K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                false,
            )?;
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights1_buf,
                &input_buf,
                &selected_buf,
                &sep_out1,
                GgmlType::Q6K,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                false,
            )
        })
        .unwrap();

    let pair0 = unsafe { pair_out0.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let pair1 = unsafe { pair_out1.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let sep0 = unsafe { sep_out0.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let sep1 = unsafe { sep_out1.as_slice::<f32>()[..n_selected * dim].to_vec() };

    let (idx0, diff0) = max_abs_diff_with_index(&pair0, &sep0);
    let (idx1, diff1) = max_abs_diff_with_index(&pair1, &sep1);
    let scale0 = sep0
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let scale1 = sep1
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff0 / scale0 < 1e-3 && diff1 / scale1 < 1e-3,
        "selected pair Q6_K path mismatch: idx0={idx0} slot0={} row0={} diff0={diff0}, idx1={idx1} slot1={} row1={} diff1={diff1}",
        idx0 / dim,
        idx0 % dim,
        idx1 / dim,
        idx1 % dim,
    );
}

#[test]
fn test_metal_backend_moe_mul_mat_selected_single_token_pair_matches_separate_q8_0() {
    let backend = MetalBackend::new().unwrap();

    let n_selected = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;
    let q8_blocks_per_row = expert_inter_dim.div_ceil(32);

    let build_q8_0_expert_tensor = |expert_scales: &[f32], rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expert_scales.len() * rows * q8_blocks_per_row * 34);
        for &scale in expert_scales {
            for _ in 0..rows {
                for _ in 0..q8_blocks_per_row {
                    bytes.extend_from_slice(&q8_0_block_constant(scale));
                }
            }
        }
        bytes
    };

    let weights0 = build_q8_0_expert_tensor(&[0.04, 0.07], dim);
    let weights1 = build_q8_0_expert_tensor(&[0.09, 0.12], dim);
    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q8_0, dim * expert_inter_dim);
    let blocks_per_expert =
        MetalOps::moe_blocks_per_expert(expert_stride, GgmlType::Q8_0, "selected pair").unwrap();

    let mut input = vec![0.0f32; expert_inter_dim];
    input[..128].fill(1.0);

    let selected_experts = vec![0i32, 1i32];
    let weights0_buf = MetalBuffer::from_slice(backend.device.device(), &weights0).unwrap();
    let weights1_buf = MetalBuffer::from_slice(backend.device.device(), &weights1).unwrap();
    let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
    let selected_buf = MetalBuffer::from_slice(backend.device.device(), &selected_experts).unwrap();
    let pair_out0 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let pair_out1 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let sep_out0 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let sep_out1 = MetalBuffer::new(
        backend.device.device(),
        n_selected * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token_pair(
                encoder,
                &weights0_buf,
                &weights1_buf,
                &input_buf,
                &selected_buf,
                &pair_out0,
                &pair_out1,
                GgmlType::Q8_0,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                blocks_per_expert,
                false,
                false,
            )
        })
        .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights0_buf,
                &input_buf,
                &selected_buf,
                &sep_out0,
                GgmlType::Q8_0,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                false,
            )?;
            backend.ops.encode_moe_mul_mat_selected_single_token(
                encoder,
                &weights1_buf,
                &input_buf,
                &selected_buf,
                &sep_out1,
                GgmlType::Q8_0,
                dim as u32,
                expert_inter_dim as u32,
                n_selected as u32,
                blocks_per_expert,
                false,
            )
        })
        .unwrap();

    let pair0 = unsafe { pair_out0.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let pair1 = unsafe { pair_out1.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let sep0 = unsafe { sep_out0.as_slice::<f32>()[..n_selected * dim].to_vec() };
    let sep1 = unsafe { sep_out1.as_slice::<f32>()[..n_selected * dim].to_vec() };

    let (idx0, diff0) = max_abs_diff_with_index(&pair0, &sep0);
    let (idx1, diff1) = max_abs_diff_with_index(&pair1, &sep1);
    let scale0 = sep0
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let scale1 = sep1
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff0 / scale0 < 1e-3 && diff1 / scale1 < 1e-3,
        "selected pair Q8_0 path mismatch: idx0={idx0} slot0={} row0={} diff0={diff0}, idx1={idx1} slot1={} row1={} diff1={diff1}",
        idx0 / dim,
        idx0 % dim,
        idx1 / dim,
        idx1 % dim,
    );
}

#[test]
fn test_metal_backend_moe_ffn_gpu_resident_cached_matches_cpu_for_shared_expert_scalar_gate() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 1usize;
    let n_expert = 2usize;
    let n_expert_used = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;
    let shared_inter_dim = 64usize;
    let eps = 1e-5f32;

    let hidden: Vec<f32> = (0..dim)
        .map(|i| {
            if i < 128 {
                0.01
            } else {
                0.005 + (i - 128) as f32 * 0.00001
            }
        })
        .collect();
    let norm_weights = vec![1.0f32; dim];

    let mut router_weights = vec![0.0f32; n_expert * dim];
    router_weights[..128].fill(0.02);
    router_weights[dim..dim + 128].fill(0.01);

    let mut routed_zero_weights = Vec::with_capacity(n_expert * expert_inter_dim * 144);
    for _expert in 0..n_expert {
        for _row in 0..expert_inter_dim {
            routed_zero_weights.extend_from_slice(&q4k_block_first128_constant(0));
        }
    }

    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q4K, dim * expert_inter_dim);
    let config = crate::model::config::ModelConfig {
        architecture: "qwen35".to_string(),
        n_layers: 1,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: dim as u32,
        head_dim: dim as u32,
        intermediate_dim: shared_inter_dim as u32,
        context_length: 16,
        vocab_size: 32,
        rms_norm_eps: eps,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: crate::model::config::GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: Some(n_expert as u32),
        n_expert_used: Some(n_expert_used as u32),
        expert_intermediate_dim: Some(expert_inter_dim as u32),
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let mean_sq = hidden.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
    let norm: Vec<f32> = hidden
        .iter()
        .zip(norm_weights.iter())
        .map(|(&x, &w)| x * inv_rms * w)
        .collect();

    let mut shared_gate = vec![0.0f32; shared_inter_dim * dim];
    let mut shared_up = vec![0.0f32; shared_inter_dim * dim];
    let mut shared_down = vec![0.0f32; dim * shared_inter_dim];
    let mut shared_gate_inp = vec![0.0f32; dim];
    for row in 0..shared_inter_dim {
        shared_gate[row * dim..row * dim + 128].fill(0.004 + row as f32 * 0.00001);
        shared_up[row * dim..row * dim + 128].fill(0.003 + row as f32 * 0.00001);
    }
    for row in 0..dim {
        shared_down[row * shared_inter_dim..(row + 1) * shared_inter_dim]
            .fill(0.002 + row as f32 * 0.000001);
    }
    shared_gate_inp[..128].fill(0.05);

    let mut gate_buf = vec![0.0f32; shared_inter_dim];
    let mut up_buf = vec![0.0f32; shared_inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    cpu.matmul(
        &shared_gate,
        &norm,
        &mut gate_buf,
        shared_inter_dim,
        n_tokens,
        dim,
    );
    cpu.matmul(
        &shared_up,
        &norm,
        &mut up_buf,
        shared_inter_dim,
        n_tokens,
        dim,
    );
    crate::compute::silu::silu_elementwise_mul(&mut gate_buf, &up_buf);
    cpu.matmul(
        &shared_down,
        &gate_buf,
        &mut down_buf,
        dim,
        n_tokens,
        shared_inter_dim,
    );
    let mut gate_scalar = [0.0f32; 1];
    cpu.matmul(&shared_gate_inp, &norm, &mut gate_scalar, 1, n_tokens, dim);
    let gate_scale = 1.0 / (1.0 + (-gate_scalar[0]).exp());
    for value in &mut down_buf {
        *value *= gate_scale;
    }

    let mut expected_hidden = hidden.clone();
    for (dst, src) in expected_hidden.iter_mut().zip(down_buf.iter()) {
        *dst += *src;
    }

    let hidden_buf = MetalBuffer::from_slice(backend.device.device(), &hidden).unwrap();
    let norm_buf = MetalBuffer::from_slice(backend.device.device(), &norm_weights).unwrap();
    let router_buf = MetalBuffer::from_slice(backend.device.device(), &router_weights).unwrap();
    let gate_buf = MetalBuffer::from_slice(backend.device.device(), &routed_zero_weights).unwrap();
    let up_buf = MetalBuffer::from_slice(backend.device.device(), &routed_zero_weights).unwrap();
    let down_buf = MetalBuffer::from_slice(backend.device.device(), &routed_zero_weights).unwrap();
    let shared_gate_buf = MetalBuffer::from_slice(backend.device.device(), &shared_gate).unwrap();
    let shared_up_buf = MetalBuffer::from_slice(backend.device.device(), &shared_up).unwrap();
    let shared_down_buf = MetalBuffer::from_slice(backend.device.device(), &shared_down).unwrap();
    let shared_gate_inp_buf =
        MetalBuffer::from_slice(backend.device.device(), &shared_gate_inp).unwrap();
    let shared_expert = SharedExpertCachedBuffers {
        gate: &shared_gate_buf,
        up: &shared_up_buf,
        down: &shared_down_buf,
        gate_inp: Some(&shared_gate_inp_buf),
        gate_inp_dtype: Some(GgmlType::F32),
        dtype: GgmlType::F32,
        inter_dim: shared_inter_dim,
        gate_inp_rows: 1,
    };

    backend.ops.init_batch_scratches(&config, n_tokens);
    backend
        .ops
        .moe_ffn_gpu_resident_cached(
            &hidden_buf,
            &norm_buf,
            &router_buf,
            GgmlType::F32,
            &gate_buf,
            GgmlType::Q4K,
            &up_buf,
            GgmlType::Q4K,
            &down_buf,
            GgmlType::Q4K,
            n_tokens,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            expert_stride,
            expert_stride,
            expert_stride,
            eps,
            Some(&shared_expert),
        )
        .unwrap();

    let actual_hidden = unsafe { hidden_buf.as_slice::<f32>()[..dim].to_vec() };
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "resident MoE shared path mismatch: max_diff={diff}, rel_diff={}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );
}

#[test]
fn test_metal_backend_moe_ffn_gpu_resident_cached_matches_cpu_for_shared_expert_q8_scalar_gate() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 1usize;
    let n_expert = 2usize;
    let n_expert_used = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;
    let shared_inter_dim = 64usize;
    let eps = 1e-5f32;

    let hidden: Vec<f32> = (0..dim)
        .map(|i| {
            if i < 128 {
                0.01
            } else {
                0.005 + (i - 128) as f32 * 0.00001
            }
        })
        .collect();
    let norm_weights = vec![1.0f32; dim];

    let mut router_weights = vec![0.0f32; n_expert * dim];
    router_weights[..128].fill(0.02);
    router_weights[dim..dim + 128].fill(0.01);

    let mut routed_zero_weights = Vec::with_capacity(n_expert * expert_inter_dim * 144);
    for _expert in 0..n_expert {
        for _row in 0..expert_inter_dim {
            routed_zero_weights.extend_from_slice(&q4k_block_first128_constant(0));
        }
    }

    let build_q8_0_tensor =
        |rows: usize, cols: usize, scale_for_row: &dyn Fn(usize) -> f32, q: i8| -> Vec<u8> {
            let blocks_per_row = cols / 32;
            let mut bytes = Vec::with_capacity(rows * blocks_per_row * 34);
            for row in 0..rows {
                let d = half::f16::from_f32(scale_for_row(row)).to_le_bytes();
                for _ in 0..blocks_per_row {
                    bytes.push(d[0]);
                    bytes.push(d[1]);
                    bytes.extend(std::iter::repeat_n(q as u8, 32));
                }
            }
            bytes
        };

    let shared_gate = build_q8_0_tensor(
        shared_inter_dim,
        dim,
        &|row| 0.004 + row as f32 * 0.00001,
        1,
    );
    let shared_up = build_q8_0_tensor(
        shared_inter_dim,
        dim,
        &|row| 0.003 + row as f32 * 0.00001,
        2,
    );
    let shared_down = build_q8_0_tensor(
        dim,
        shared_inter_dim,
        &|row| 0.002 + row as f32 * 0.000001,
        1,
    );
    let mut shared_gate_inp = vec![0.0f32; dim];
    shared_gate_inp[..128].fill(0.05);

    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q4K, dim * expert_inter_dim);
    let config = crate::model::config::ModelConfig {
        architecture: "qwen35".to_string(),
        n_layers: 1,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: dim as u32,
        head_dim: dim as u32,
        intermediate_dim: shared_inter_dim as u32,
        context_length: 16,
        vocab_size: 32,
        rms_norm_eps: eps,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: crate::model::config::GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: Some(n_expert as u32),
        n_expert_used: Some(n_expert_used as u32),
        expert_intermediate_dim: Some(expert_inter_dim as u32),
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let mean_sq = hidden.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
    let norm: Vec<f32> = hidden
        .iter()
        .zip(norm_weights.iter())
        .map(|(&x, &w)| x * inv_rms * w)
        .collect();

    let mut gate_buf = vec![0.0f32; shared_inter_dim];
    let mut up_buf = vec![0.0f32; shared_inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    cpu.dequant_matmul(
        &shared_gate,
        GgmlType::Q8_0,
        &norm,
        &mut gate_buf,
        shared_inter_dim,
        n_tokens,
        dim,
    );
    cpu.dequant_matmul(
        &shared_up,
        GgmlType::Q8_0,
        &norm,
        &mut up_buf,
        shared_inter_dim,
        n_tokens,
        dim,
    );
    crate::compute::silu::silu_elementwise_mul(&mut gate_buf, &up_buf);
    cpu.dequant_matmul(
        &shared_down,
        GgmlType::Q8_0,
        &gate_buf,
        &mut down_buf,
        dim,
        n_tokens,
        shared_inter_dim,
    );
    let mut gate_scalar = [0.0f32; 1];
    cpu.matmul(&shared_gate_inp, &norm, &mut gate_scalar, 1, n_tokens, dim);
    let gate_scale = 1.0 / (1.0 + (-gate_scalar[0]).exp());
    for value in &mut down_buf {
        *value *= gate_scale;
    }

    let mut expected_hidden = hidden.clone();
    for (dst, src) in expected_hidden.iter_mut().zip(down_buf.iter()) {
        *dst += *src;
    }

    let hidden_buf = MetalBuffer::from_slice(backend.device.device(), &hidden).unwrap();
    let norm_buf = MetalBuffer::from_slice(backend.device.device(), &norm_weights).unwrap();
    let router_buf = MetalBuffer::from_slice(backend.device.device(), &router_weights).unwrap();
    let gate_buf = MetalBuffer::from_slice(backend.device.device(), &routed_zero_weights).unwrap();
    let up_buf = MetalBuffer::from_slice(backend.device.device(), &routed_zero_weights).unwrap();
    let down_buf = MetalBuffer::from_slice(backend.device.device(), &routed_zero_weights).unwrap();
    let shared_gate_buf = MetalBuffer::from_slice(backend.device.device(), &shared_gate).unwrap();
    let shared_up_buf = MetalBuffer::from_slice(backend.device.device(), &shared_up).unwrap();
    let shared_down_buf = MetalBuffer::from_slice(backend.device.device(), &shared_down).unwrap();
    let shared_gate_inp_buf =
        MetalBuffer::from_slice(backend.device.device(), &shared_gate_inp).unwrap();
    let shared_expert = SharedExpertCachedBuffers {
        gate: &shared_gate_buf,
        up: &shared_up_buf,
        down: &shared_down_buf,
        gate_inp: Some(&shared_gate_inp_buf),
        gate_inp_dtype: Some(GgmlType::F32),
        dtype: GgmlType::Q8_0,
        inter_dim: shared_inter_dim,
        gate_inp_rows: 1,
    };

    backend.ops.init_batch_scratches(&config, n_tokens);
    backend
        .ops
        .moe_ffn_gpu_resident_cached(
            &hidden_buf,
            &norm_buf,
            &router_buf,
            GgmlType::F32,
            &gate_buf,
            GgmlType::Q4K,
            &up_buf,
            GgmlType::Q4K,
            &down_buf,
            GgmlType::Q4K,
            n_tokens,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            expert_stride,
            expert_stride,
            expert_stride,
            eps,
            Some(&shared_expert),
        )
        .unwrap();

    let actual_hidden = unsafe { hidden_buf.as_slice::<f32>()[..dim].to_vec() };
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "resident MoE shared Q8 path mismatch: max_diff={diff}, rel_diff={}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );
}

#[test]
fn test_metal_backend_moe_ffn_gpu_resident_cached_matches_cpu_for_multitoken_shared_expert_vector_gate()
 {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 4usize;
    let n_expert = 2usize;
    let n_expert_used = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;
    let shared_inter_dim = 64usize;
    let eps = 1e-5f32;

    let hidden: Vec<f32> = (0..n_tokens)
        .flat_map(|token| {
            (0..dim).map(move |i| {
                let lane = i / 64;
                0.01 * (token + 1) as f32 + lane as f32 * 0.001 + (i % 64) as f32 * 0.00002
            })
        })
        .collect();
    let norm_weights = vec![1.0f32; dim];

    let mut router_weights = vec![0.0f32; n_expert * dim];
    router_weights[..128].fill(0.02);
    router_weights[dim..dim + 128].fill(0.01);

    let mut routed_zero_weights = Vec::with_capacity(n_expert * expert_inter_dim * 144);
    for _expert in 0..n_expert {
        for _row in 0..expert_inter_dim {
            routed_zero_weights.extend_from_slice(&q4k_block_first128_constant(0));
        }
    }

    let mut shared_gate = vec![0.0f32; shared_inter_dim * dim];
    let mut shared_up = vec![0.0f32; shared_inter_dim * dim];
    let mut shared_down = vec![0.0f32; dim * shared_inter_dim];
    let mut shared_gate_inp = vec![0.0f32; dim * dim];
    for row in 0..shared_inter_dim {
        shared_gate[row * dim..row * dim + 128].fill(0.004 + row as f32 * 0.00001);
        shared_up[row * dim..row * dim + 128].fill(0.003 + row as f32 * 0.00002);
    }
    for row in 0..dim {
        shared_down[row * shared_inter_dim..(row + 1) * shared_inter_dim]
            .fill(0.002 + row as f32 * 0.000001);
        shared_gate_inp[row * dim + row] = 0.05 + row as f32 * 0.00001;
    }

    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q4K, dim * expert_inter_dim);
    let config = crate::model::config::ModelConfig {
        architecture: "qwen35".to_string(),
        n_layers: 1,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: dim as u32,
        head_dim: dim as u32,
        intermediate_dim: shared_inter_dim as u32,
        context_length: 16,
        vocab_size: 32,
        rms_norm_eps: eps,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: crate::model::config::GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: Some(n_expert as u32),
        n_expert_used: Some(n_expert_used as u32),
        expert_intermediate_dim: Some(expert_inter_dim as u32),
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let mut norm = vec![0.0f32; n_tokens * dim];
    for token in 0..n_tokens {
        let src = &hidden[token * dim..(token + 1) * dim];
        let dst = &mut norm[token * dim..(token + 1) * dim];
        let mean_sq = src.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
        for ((dst, &x), &w) in dst.iter_mut().zip(src.iter()).zip(norm_weights.iter()) {
            *dst = x * inv_rms * w;
        }
    }

    let mut expected_hidden = hidden.clone();
    let mut gate_buf = vec![0.0f32; shared_inter_dim];
    let mut up_buf = vec![0.0f32; shared_inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    let mut gate_inp_buf = vec![0.0f32; dim];
    for token in 0..n_tokens {
        let token_norm = &norm[token * dim..(token + 1) * dim];
        cpu.matmul(
            &shared_gate,
            token_norm,
            &mut gate_buf,
            shared_inter_dim,
            1,
            dim,
        );
        cpu.matmul(
            &shared_up,
            token_norm,
            &mut up_buf,
            shared_inter_dim,
            1,
            dim,
        );
        crate::compute::silu::silu_elementwise_mul(&mut gate_buf, &up_buf);
        cpu.matmul(
            &shared_down,
            &gate_buf,
            &mut down_buf,
            dim,
            1,
            shared_inter_dim,
        );
        cpu.matmul(&shared_gate_inp, token_norm, &mut gate_inp_buf, dim, 1, dim);
        for ((dst, src), gate) in expected_hidden[token * dim..(token + 1) * dim]
            .iter_mut()
            .zip(down_buf.iter())
            .zip(gate_inp_buf.iter())
        {
            *dst += *src * (1.0 / (1.0 + (-*gate).exp()));
        }
    }

    let hidden_buf = MetalBuffer::from_slice(backend.device.device(), &hidden).unwrap();
    let norm_buf = MetalBuffer::from_slice(backend.device.device(), &norm_weights).unwrap();
    let router_buf = MetalBuffer::from_slice(backend.device.device(), &router_weights).unwrap();
    let gate_buf = MetalBuffer::from_slice(backend.device.device(), &routed_zero_weights).unwrap();
    let up_buf = MetalBuffer::from_slice(backend.device.device(), &routed_zero_weights).unwrap();
    let down_buf = MetalBuffer::from_slice(backend.device.device(), &routed_zero_weights).unwrap();
    let shared_gate_buf = MetalBuffer::from_slice(backend.device.device(), &shared_gate).unwrap();
    let shared_up_buf = MetalBuffer::from_slice(backend.device.device(), &shared_up).unwrap();
    let shared_down_buf = MetalBuffer::from_slice(backend.device.device(), &shared_down).unwrap();
    let shared_gate_inp_buf =
        MetalBuffer::from_slice(backend.device.device(), &shared_gate_inp).unwrap();
    let shared_expert = SharedExpertCachedBuffers {
        gate: &shared_gate_buf,
        up: &shared_up_buf,
        down: &shared_down_buf,
        gate_inp: Some(&shared_gate_inp_buf),
        gate_inp_dtype: Some(GgmlType::F32),
        dtype: GgmlType::F32,
        inter_dim: shared_inter_dim,
        gate_inp_rows: dim,
    };

    backend.ops.init_batch_scratches(&config, n_tokens);
    backend
        .ops
        .moe_ffn_gpu_resident_cached(
            &hidden_buf,
            &norm_buf,
            &router_buf,
            GgmlType::F32,
            &gate_buf,
            GgmlType::Q4K,
            &up_buf,
            GgmlType::Q4K,
            &down_buf,
            GgmlType::Q4K,
            n_tokens,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            expert_stride,
            expert_stride,
            expert_stride,
            eps,
            Some(&shared_expert),
        )
        .unwrap();

    let actual_hidden = unsafe { hidden_buf.as_slice::<f32>()[..n_tokens * dim].to_vec() };
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 2e-3,
        "resident MoE multitoken shared-vector path mismatch: max_diff={diff}, rel_diff={}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );
}

#[test]
fn test_metal_backend_moe_ffn_gpu_resident_cached_matches_cpu_for_multitoken_shared_expert_without_gate()
 {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 3usize;
    let n_expert = 2usize;
    let n_expert_used = 2usize;
    let dim = 256usize;
    let expert_inter_dim = 256usize;
    let shared_inter_dim = 64usize;
    let eps = 1e-5f32;

    let hidden: Vec<f32> = (0..n_tokens)
        .flat_map(|token| {
            (0..dim).map(move |i| 0.01 * (token + 1) as f32 + (i / 64) as f32 * 0.001)
        })
        .collect();
    let norm_weights = vec![1.0f32; dim];

    let mut router_weights = vec![0.0f32; n_expert * dim];
    router_weights[..128].fill(0.02);
    router_weights[dim..dim + 128].fill(0.01);

    let mut routed_zero_weights = Vec::with_capacity(n_expert * expert_inter_dim * 144);
    for _expert in 0..n_expert {
        for _row in 0..expert_inter_dim {
            routed_zero_weights.extend_from_slice(&q4k_block_first128_constant(0));
        }
    }

    let mut shared_gate = vec![0.0f32; shared_inter_dim * dim];
    let mut shared_up = vec![0.0f32; shared_inter_dim * dim];
    let mut shared_down = vec![0.0f32; dim * shared_inter_dim];
    for row in 0..shared_inter_dim {
        shared_gate[row * dim..row * dim + 128].fill(0.004 + row as f32 * 0.00001);
        shared_up[row * dim..row * dim + 128].fill(0.003 + row as f32 * 0.00002);
    }
    for row in 0..dim {
        shared_down[row * shared_inter_dim..(row + 1) * shared_inter_dim]
            .fill(0.002 + row as f32 * 0.000001);
    }

    let expert_stride =
        crate::model::moe_utils::expert_byte_stride(GgmlType::Q4K, dim * expert_inter_dim);
    let config = crate::model::config::ModelConfig {
        architecture: "qwen35".to_string(),
        n_layers: 1,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: dim as u32,
        head_dim: dim as u32,
        intermediate_dim: shared_inter_dim as u32,
        context_length: 16,
        vocab_size: 32,
        rms_norm_eps: eps,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: crate::model::config::GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: Some(n_expert as u32),
        n_expert_used: Some(n_expert_used as u32),
        expert_intermediate_dim: Some(expert_inter_dim as u32),
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let mut norm = vec![0.0f32; n_tokens * dim];
    for token in 0..n_tokens {
        let src = &hidden[token * dim..(token + 1) * dim];
        let dst = &mut norm[token * dim..(token + 1) * dim];
        let mean_sq = src.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
        for ((dst, &x), &w) in dst.iter_mut().zip(src.iter()).zip(norm_weights.iter()) {
            *dst = x * inv_rms * w;
        }
    }

    let mut expected_hidden = hidden.clone();
    let mut gate_buf = vec![0.0f32; shared_inter_dim];
    let mut up_buf = vec![0.0f32; shared_inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    for token in 0..n_tokens {
        let token_norm = &norm[token * dim..(token + 1) * dim];
        cpu.matmul(
            &shared_gate,
            token_norm,
            &mut gate_buf,
            shared_inter_dim,
            1,
            dim,
        );
        cpu.matmul(
            &shared_up,
            token_norm,
            &mut up_buf,
            shared_inter_dim,
            1,
            dim,
        );
        crate::compute::silu::silu_elementwise_mul(&mut gate_buf, &up_buf);
        cpu.matmul(
            &shared_down,
            &gate_buf,
            &mut down_buf,
            dim,
            1,
            shared_inter_dim,
        );
        for (dst, src) in expected_hidden[token * dim..(token + 1) * dim]
            .iter_mut()
            .zip(down_buf.iter())
        {
            *dst += *src;
        }
    }

    let hidden_buf = MetalBuffer::from_slice(backend.device.device(), &hidden).unwrap();
    let norm_buf = MetalBuffer::from_slice(backend.device.device(), &norm_weights).unwrap();
    let router_buf = MetalBuffer::from_slice(backend.device.device(), &router_weights).unwrap();
    let gate_buf = MetalBuffer::from_slice(backend.device.device(), &routed_zero_weights).unwrap();
    let up_buf = MetalBuffer::from_slice(backend.device.device(), &routed_zero_weights).unwrap();
    let down_buf = MetalBuffer::from_slice(backend.device.device(), &routed_zero_weights).unwrap();
    let shared_gate_buf = MetalBuffer::from_slice(backend.device.device(), &shared_gate).unwrap();
    let shared_up_buf = MetalBuffer::from_slice(backend.device.device(), &shared_up).unwrap();
    let shared_down_buf = MetalBuffer::from_slice(backend.device.device(), &shared_down).unwrap();
    let shared_expert = SharedExpertCachedBuffers {
        gate: &shared_gate_buf,
        up: &shared_up_buf,
        down: &shared_down_buf,
        gate_inp: None,
        gate_inp_dtype: None,
        dtype: GgmlType::F32,
        inter_dim: shared_inter_dim,
        gate_inp_rows: 0,
    };

    backend.ops.init_batch_scratches(&config, n_tokens);
    backend
        .ops
        .moe_ffn_gpu_resident_cached(
            &hidden_buf,
            &norm_buf,
            &router_buf,
            GgmlType::F32,
            &gate_buf,
            GgmlType::Q4K,
            &up_buf,
            GgmlType::Q4K,
            &down_buf,
            GgmlType::Q4K,
            n_tokens,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            expert_stride,
            expert_stride,
            expert_stride,
            eps,
            Some(&shared_expert),
        )
        .unwrap();

    let actual_hidden = unsafe { hidden_buf.as_slice::<f32>()[..n_tokens * dim].to_vec() };
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 2e-3,
        "resident MoE multitoken shared path without gate mismatch: max_diff={diff}, rel_diff={}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );
}

#[test]
fn test_metal_backend_fused_q6_k_matvec() {
    let backend = MetalBackend::new().unwrap();

    // Create Q6_K data: 4 rows × 512 cols (2 blocks per row)
    let m = 4;
    let k = 512;
    let block_bytes = 210;
    let blocks_per_row = k / 256;

    // Use a simple LCG for deterministic pseudo-random data
    let mut rng_state: u64 = 42;
    let mut next_u8 = || -> u8 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as u8
    };

    let mut quant_data = Vec::new();
    for row in 0..m {
        for _blk in 0..blocks_per_row {
            let mut block = vec![0u8; block_bytes];

            // d = varied scale
            let d_val = (row as f32 + 1.0) * 0.05;
            let d_bytes = half::f16::from_f32(d_val).to_le_bytes();
            block[208] = d_bytes[0];
            block[209] = d_bytes[1];

            // Varied scales (signed i8)
            for i in 0..16 {
                block[192 + i] = ((i as i8 % 7) - 3) as u8; // range -3..3
            }

            // Varied ql and qh
            for b in block[..128].iter_mut() {
                *b = next_u8();
            }
            for b in block[128..192].iter_mut() {
                *b = next_u8();
            }

            quant_data.extend(block);
        }
    }

    // Input vector
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.01) - 2.56).collect();

    // CPU reference
    let mut weights = vec![0.0f32; m * k];
    crate::quant::q6_k::dequantize(&quant_data, &mut weights);
    let mut expected = vec![0.0f32; m];
    crate::compute::matmul::matmul_f32(&weights, &x, &mut expected, m, 1, k);

    // GPU fused dequant+matvec
    let mut result = vec![0.0f32; m];
    backend.dequant_matmul(&quant_data, GgmlType::Q6K, &x, &mut result, m, 1, k);

    let diff = max_abs_diff(&result, &expected);
    assert!(
        diff < 0.5,
        "Fused Q6_K matvec mismatch: max_diff={diff}, result={result:?}, expected={expected:?}"
    );
}

#[test]
fn test_metal_matches_cpu_larger() {
    let cpu = super::super::cpu::CpuBackend;
    let metal = MetalBackend::new().unwrap();

    let m = 64;
    let n = 32;
    let k = 48;

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

    let mut cpu_c = vec![0.0f32; m * n];
    let mut metal_c = vec![0.0f32; m * n];

    cpu.matmul(&a, &b, &mut cpu_c, m, n, k);
    metal.matmul(&a, &b, &mut metal_c, m, n, k);

    let diff = max_abs_diff(&cpu_c, &metal_c);
    assert!(diff < 0.1, "CPU vs Metal mismatch: max_diff={diff}");
}

#[test]
fn test_metal_backend_attention_prefill() {
    let backend = MetalBackend::new().unwrap();

    // 4 tokens, 2 heads, 2 KV heads, head_dim=4
    let n_tokens = 4;
    let n_heads = 2;
    let n_kv_heads = 2;
    let head_dim = 4;
    let q_size = n_tokens * n_heads * head_dim;
    let kv_size = n_tokens * n_kv_heads * head_dim;

    let q: Vec<f32> = (0..q_size).map(|i| ((i % 5) as f32 - 2.0) * 0.2).collect();
    let k: Vec<f32> = (0..kv_size).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
    let v: Vec<f32> = (0..kv_size).map(|i| ((i % 3) as f32 - 1.0) * 0.5).collect();

    // CPU reference
    let cpu = super::super::cpu::CpuBackend;
    let mut expected = vec![0.0f32; q_size];
    cpu.attention_prefill(
        &q,
        &k,
        &v,
        &mut expected,
        n_tokens,
        n_heads,
        n_kv_heads,
        head_dim,
    );

    // Metal
    let mut result = vec![0.0f32; q_size];
    backend.attention_prefill(
        &q,
        &k,
        &v,
        &mut result,
        n_tokens,
        n_heads,
        n_kv_heads,
        head_dim,
    );

    let diff = max_abs_diff(&result, &expected);
    assert!(
        diff < 1e-3,
        "Metal attention vs CPU mismatch: max_diff={diff}"
    );
}

#[test]
#[ignore = "parallel Metal backend suite is nondeterministic for this kernel; run standalone"]
fn test_metal_backend_qwen35_causal_conv_sequence_matches_cpu() {
    let _env_lock = lock_env_test();
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 7;
    let conv_cache_len = 3;
    let conv_dim = 96;
    let kernel_size = conv_cache_len + 1;

    let input: Vec<f32> = (0..n_tokens * conv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let kernel: Vec<f32> = (0..kernel_size * conv_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.03)
        .collect();
    let mut cpu_state: Vec<f32> = (0..conv_cache_len * conv_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.02)
        .collect();
    let mut metal_state = cpu_state.clone();
    let mut expected = vec![0.0f32; n_tokens * conv_dim];
    let mut result = vec![0.0f32; n_tokens * conv_dim];
    let mut warmup_state = cpu_state.clone();
    let mut warmup_result = vec![0.0f32; n_tokens * conv_dim];

    cpu.qwen35_causal_conv_sequence(
        &input,
        &kernel,
        &mut cpu_state,
        &mut expected,
        n_tokens,
        conv_cache_len,
        conv_dim,
    );
    backend.qwen35_causal_conv_sequence(
        &input,
        &kernel,
        &mut warmup_state,
        &mut warmup_result,
        n_tokens,
        conv_cache_len,
        conv_dim,
    );
    backend.qwen35_causal_conv_sequence(
        &input,
        &kernel,
        &mut metal_state,
        &mut result,
        n_tokens,
        conv_cache_len,
        conv_dim,
    );

    let output_diff = max_abs_diff(&result, &expected);
    let state_diff = max_abs_diff(&metal_state, &cpu_state);
    assert!(
        output_diff < 1e-5,
        "Metal qwen35 causal conv output mismatch: max_diff={output_diff}"
    );
    assert!(
        state_diff < 1e-6,
        "Metal qwen35 causal conv state mismatch: max_diff={state_diff}"
    );
}

#[test]
fn test_metal_backend_qwen35_single_token_fused_gated_delta_matches_cpu() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let group_count = 2usize;
    let time_step_rank = 4usize;
    let state_size = 128usize;
    let key_dim = group_count * state_size;
    let value_dim = time_step_rank * state_size;
    let conv_dim = 2 * key_dim + value_dim;
    let state_len = time_step_rank * state_size * state_size;
    let eps = 1e-5f32;

    let conv_out: Vec<f32> = (0..conv_dim)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
        .collect();
    let gate: Vec<f32> = (0..time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.2)
        .collect();
    let beta: Vec<f32> = (0..time_step_rank)
        .map(|i| 0.1 + (i % 5) as f32 * 0.04)
        .collect();
    let initial_state: Vec<f32> = (0..state_len)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let mut cpu_state = initial_state.clone();
    let mut q = vec![0.0f32; value_dim];
    let mut k = vec![0.0f32; value_dim];
    let mut v = vec![0.0f32; value_dim];
    let mut expected = vec![0.0f32; value_dim];

    prepare_single_token_gdn_qkv(
        &conv_out,
        &mut q,
        &mut k,
        &mut v,
        group_count,
        time_step_rank,
        state_size,
        eps,
    );
    cpu.qwen35_gated_delta_sequence(
        &q,
        &k,
        &v,
        &gate,
        &beta,
        &mut cpu_state,
        &mut expected,
        1,
        time_step_rank,
        state_size,
    );

    let conv_out_buf = MetalBuffer::from_slice(backend.device.device(), &conv_out).unwrap();
    let gate_buf = MetalBuffer::from_slice(backend.device.device(), &gate).unwrap();
    let beta_buf = MetalBuffer::from_slice(backend.device.device(), &beta).unwrap();
    let state_buf = MetalBuffer::from_slice(backend.device.device(), &initial_state).unwrap();
    let output_buf = MetalBuffer::new(
        backend.device.device(),
        value_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    backend
        .device
        .execute_sync(|encoder| {
            anyhow::ensure!(
                backend.gdn_kernels.encode_single_token_gated_delta_fused(
                    encoder,
                    &conv_out_buf,
                    &gate_buf,
                    &beta_buf,
                    &state_buf,
                    &output_buf,
                    group_count as u32,
                    time_step_rank as u32,
                    state_size as u32,
                    eps,
                ),
                "single-token fused gated-delta kernel should support head_dim={state_size}"
            );
            Ok(())
        })
        .unwrap();

    let result = unsafe { output_buf.as_slice::<f32>()[..value_dim].to_vec() };
    let metal_state = unsafe { state_buf.as_slice::<f32>()[..state_len].to_vec() };

    let output_diff = max_abs_diff(&result, &expected);
    let state_diff = max_abs_diff(&metal_state, &cpu_state);
    assert!(
        output_diff < 1e-4,
        "Metal qwen35 single-token fused gated delta output mismatch: max_diff={output_diff}"
    );
    assert!(
        state_diff < 1e-4,
        "Metal qwen35 single-token fused gated delta state mismatch: max_diff={state_diff}"
    );
}

#[test]
fn test_metal_backend_qwen35_gated_delta_sequence_matches_cpu() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 3;
    let n_heads = 2;
    let head_dim = 128;
    let value_dim = n_heads * head_dim;
    let state_len = n_heads * head_dim * head_dim;

    let q_batch: Vec<f32> = (0..n_tokens * value_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.04)
        .collect();
    let k_batch: Vec<f32> = (0..n_tokens * value_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.03)
        .collect();
    let v_batch: Vec<f32> = (0..n_tokens * value_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let gate_batch: Vec<f32> = (0..n_tokens * n_heads)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.2)
        .collect();
    let beta_batch: Vec<f32> = (0..n_tokens * n_heads)
        .map(|i| 0.1 + (i % 7) as f32 * 0.05)
        .collect();
    let mut cpu_state: Vec<f32> = (0..state_len)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
        .collect();
    let mut metal_state = cpu_state.clone();
    let mut expected = vec![0.0f32; n_tokens * value_dim];
    let mut result = vec![0.0f32; n_tokens * value_dim];

    cpu.qwen35_gated_delta_sequence(
        &q_batch,
        &k_batch,
        &v_batch,
        &gate_batch,
        &beta_batch,
        &mut cpu_state,
        &mut expected,
        n_tokens,
        n_heads,
        head_dim,
    );
    backend.qwen35_gated_delta_sequence(
        &q_batch,
        &k_batch,
        &v_batch,
        &gate_batch,
        &beta_batch,
        &mut metal_state,
        &mut result,
        n_tokens,
        n_heads,
        head_dim,
    );

    let output_diff = max_abs_diff(&result, &expected);
    let state_diff = max_abs_diff(&metal_state, &cpu_state);
    assert!(
        output_diff < 1e-4,
        "Metal qwen35 gated delta output mismatch: max_diff={output_diff}"
    );
    assert!(
        state_diff < 1e-4,
        "Metal qwen35 gated delta state mismatch: max_diff={state_diff}"
    );
}

#[test]
fn test_metal_backend_qwen35_gated_delta_sequence_chunked_matches_cpu() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;

    let n_tokens = 96;
    let n_heads = 2;
    let head_dim = 128;
    let value_dim = n_heads * head_dim;
    let state_len = n_heads * head_dim * head_dim;

    let q_batch: Vec<f32> = (0..n_tokens * value_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.04)
        .collect();
    let k_batch: Vec<f32> = (0..n_tokens * value_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.03)
        .collect();
    let v_batch: Vec<f32> = (0..n_tokens * value_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let gate_batch: Vec<f32> = (0..n_tokens * n_heads)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.2)
        .collect();
    let beta_batch: Vec<f32> = (0..n_tokens * n_heads)
        .map(|i| 0.1 + (i % 7) as f32 * 0.05)
        .collect();
    let mut cpu_state: Vec<f32> = (0..state_len)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
        .collect();
    let mut metal_state = cpu_state.clone();
    let mut expected = vec![0.0f32; n_tokens * value_dim];
    let mut result = vec![0.0f32; n_tokens * value_dim];

    cpu.qwen35_gated_delta_sequence(
        &q_batch,
        &k_batch,
        &v_batch,
        &gate_batch,
        &beta_batch,
        &mut cpu_state,
        &mut expected,
        n_tokens,
        n_heads,
        head_dim,
    );
    backend.qwen35_gated_delta_sequence(
        &q_batch,
        &k_batch,
        &v_batch,
        &gate_batch,
        &beta_batch,
        &mut metal_state,
        &mut result,
        n_tokens,
        n_heads,
        head_dim,
    );

    let output_diff = max_abs_diff(&result, &expected);
    let state_diff = max_abs_diff(&metal_state, &cpu_state);
    assert!(
        output_diff < 1e-4,
        "Metal qwen35 chunked gated delta output mismatch: max_diff={output_diff}"
    );
    assert!(
        state_diff < 1e-4,
        "Metal qwen35 chunked gated delta state mismatch: max_diff={state_diff}"
    );
}

#[test]
fn test_metal_backend_qwen35_recurrent_sequence_reuses_slot_buffers() {
    let _env_lock = lock_env_test();
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;
    let slot_indices = [7usize];
    let tokens_per_slot = 2;
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 2,
        conv_dim: 192,
        group_count: 1,
        state_size: 64,
        time_step_rank: 1,
        rms_norm_eps: 1e-5,
    };
    let conv_state_stride = cfg.conv_cache_len * cfg.conv_dim;
    let recurrent_state_stride = cfg.time_step_rank * cfg.state_size * cfg.state_size;
    let total_tokens = slot_indices.len() * tokens_per_slot;
    let kernel_len = (cfg.conv_cache_len + 1) * cfg.conv_dim;
    let slot_key = Qwen35RecurrentSlotBufferKey {
        layer_idx: 3,
        slot_idx: slot_indices[0],
        conv_state_stride,
        recurrent_state_stride,
    };

    let qkv: Vec<f32> = (0..total_tokens * cfg.conv_dim)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.02)
        .collect();
    let alpha_input: Vec<f32> = (0..total_tokens * cfg.time_step_rank)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
        .collect();
    let beta_input: Vec<f32> = (0..total_tokens * cfg.time_step_rank)
        .map(|i| 0.1 + (i % 7) as f32 * 0.03)
        .collect();
    let dt_bias = vec![0.05f32; cfg.time_step_rank];
    let a = vec![0.02f32; cfg.time_step_rank];
    let kernel: Vec<f32> = (0..kernel_len)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();

    let mut cpu_conv_state = vec![0.0f32; conv_state_stride];
    let mut cpu_recurrent_state = vec![0.0f32; recurrent_state_stride];
    let mut metal_conv_state = cpu_conv_state.clone();
    let mut metal_recurrent_state = cpu_recurrent_state.clone();
    let mut cpu_alpha = alpha_input.clone();
    let mut cpu_beta = beta_input.clone();
    let mut metal_alpha = alpha_input.clone();
    let mut metal_beta = beta_input.clone();
    let mut expected = vec![0.0f32; total_tokens * cfg.value_dim()];
    let mut result = vec![0.0f32; total_tokens * cfg.value_dim()];

    {
        let mut cpu_state_batch = Qwen3_5RecurrentStateBatch::new(
            slot_key.layer_idx,
            &slot_indices,
            &mut cpu_conv_state,
            &mut cpu_recurrent_state,
            conv_state_stride,
            recurrent_state_stride,
        );
        cpu.qwen35_recurrent_sequence(
            &qkv,
            &mut cpu_beta,
            &mut cpu_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut cpu_state_batch,
            &mut expected,
            tokens_per_slot,
            cfg,
        );
    }
    {
        let mut metal_state_batch = Qwen3_5RecurrentStateBatch::new(
            slot_key.layer_idx,
            &slot_indices,
            &mut metal_conv_state,
            &mut metal_recurrent_state,
            conv_state_stride,
            recurrent_state_stride,
        );
        backend.qwen35_recurrent_sequence(
            &qkv,
            &mut metal_beta,
            &mut metal_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut metal_state_batch,
            &mut result,
            tokens_per_slot,
            cfg,
        );
    }

    assert!(max_abs_diff(&result, &expected) < 1e-4);
    assert!(max_abs_diff(&metal_alpha, &cpu_alpha) < 1e-5);
    assert!(max_abs_diff(&metal_beta, &cpu_beta) < 1e-5);
    assert!(max_abs_diff(&metal_conv_state, &cpu_conv_state) < 1e-5);
    assert!(max_abs_diff(&metal_recurrent_state, &cpu_recurrent_state) < 1e-4);

    {
        let cache = backend.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
        assert_eq!(
            cache.len(),
            0,
            "state-batch Metal recurrent path should alias CPU state directly without slot buffers"
        );
    }
    let first_scratch_ptrs = {
        let cache = backend.qwen35_recurrent_scratch_buffers.lock().unwrap();
        assert_eq!(cache.len(), 1);
        let scratch_key = Qwen35RecurrentScratchBufferKey {
            tokens_per_slot,
            conv_dim: cfg.conv_dim,
            time_step_rank: cfg.time_step_rank,
            state_size: cfg.state_size,
        };
        let buffers = cache.get(&scratch_key).unwrap();
        (
            buffers.input.ptr_id(),
            buffers.conv_out.ptr_id(),
            buffers.q.ptr_id(),
            buffers.output.ptr_id(),
        )
    };

    {
        let mut metal_state_batch = Qwen3_5RecurrentStateBatch::new(
            slot_key.layer_idx,
            &slot_indices,
            &mut metal_conv_state,
            &mut metal_recurrent_state,
            conv_state_stride,
            recurrent_state_stride,
        );
        backend.qwen35_recurrent_sequence(
            &qkv,
            &mut metal_beta,
            &mut metal_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut metal_state_batch,
            &mut result,
            tokens_per_slot,
            cfg,
        );
    }
    let cache = backend.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
    assert_eq!(cache.len(), 0);
    drop(cache);

    let cache = backend.qwen35_recurrent_scratch_buffers.lock().unwrap();
    assert_eq!(cache.len(), 1);
    let scratch_key = Qwen35RecurrentScratchBufferKey {
        tokens_per_slot,
        conv_dim: cfg.conv_dim,
        time_step_rank: cfg.time_step_rank,
        state_size: cfg.state_size,
    };
    let buffers = cache.get(&scratch_key).unwrap();
    assert_eq!(buffers.input.ptr_id(), first_scratch_ptrs.0);
    assert_eq!(buffers.conv_out.ptr_id(), first_scratch_ptrs.1);
    assert_eq!(buffers.q.ptr_id(), first_scratch_ptrs.2);
    assert_eq!(buffers.output.ptr_id(), first_scratch_ptrs.3);
}

#[test]
fn test_metal_backend_qwen35_recurrent_sequence_for_kv_single_slot_matches_cpu() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 0usize;
    let tokens_per_slot = 1usize;
    let slot_indices = [0usize];
    let mut expected_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let mut actual_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    expected_kv
        .conv_state_for_slot_mut(0, layer_idx)
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = 0.1 + i as f32 * 0.001);
    expected_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = 0.2 + i as f32 * 0.002);
    actual_kv
        .conv_state_for_slot_mut(0, layer_idx)
        .copy_from_slice(expected_kv.conv_state_for_slot(0, layer_idx));
    actual_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .copy_from_slice(expected_kv.recurrent_state_for_slot(0, layer_idx));

    let qkv: Vec<f32> = (0..cfg.conv_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.03)
        .collect();
    let mut expected_alpha: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.07)
        .collect();
    let mut expected_beta: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| 0.1 + i as f32 * 0.03)
        .collect();
    let mut actual_alpha = expected_alpha.clone();
    let mut actual_beta = expected_beta.clone();
    let dt_bias = vec![0.03, 0.04, 0.05, 0.06];
    let a = vec![0.2, 0.25, 0.3, 0.35];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let mut expected_out = vec![0.0f32; cfg.value_dim()];
    let mut actual_out = vec![0.0f32; cfg.value_dim()];

    cpu.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut expected_beta,
        &mut expected_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &slot_indices,
        &mut expected_out,
        tokens_per_slot,
        cfg,
    );
    backend.device.reset_perf_counters();
    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut actual_beta,
        &mut actual_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut actual_out,
        tokens_per_slot,
        cfg,
    );
    let counters = backend.device.perf_counters();

    assert!(max_abs_diff(&actual_alpha, &expected_alpha) < 1e-5);
    assert!(max_abs_diff(&actual_beta, &expected_beta) < 1e-5);
    assert!(max_abs_diff(&actual_out, &expected_out) < 1e-4);
    assert!(
        actual_kv.recurrent_state_cpu_stale(0, layer_idx),
        "single-token qwen35 recurrent state should stay backend-owned until an explicit sync"
    );
    backend.sync_qwen35_kv(&mut actual_kv);
    assert!(
        max_abs_diff(
            actual_kv.conv_state_for_slot(0, layer_idx),
            expected_kv.conv_state_for_slot(0, layer_idx),
        ) < 1e-5
    );
    assert!(
        max_abs_diff(
            actual_kv.recurrent_state_for_slot(0, layer_idx),
            expected_kv.recurrent_state_for_slot(0, layer_idx),
        ) < 1e-4
    );
    assert_eq!(
        counters.command_buffers, 1,
        "single-token qwen35 recurrent KV path should only dispatch gated-delta on Metal"
    );
    let slot_key = Qwen35RecurrentSlotBufferKey {
        layer_idx,
        slot_idx: 0,
        conv_state_stride: actual_kv.conv_cache_len() * actual_kv.conv_dim(),
        recurrent_state_stride: actual_kv.recurrent_state_len(),
    };
    let slot_cache = backend.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
    assert_eq!(slot_cache.len(), 1);
    let slot_buffers = slot_cache.get(&slot_key).unwrap();
    assert_eq!(
        slot_buffers.recurrent_synced_generation,
        Some(actual_kv.recurrent_state_generation(0, layer_idx))
    );
    drop(slot_cache);

    let scratch_key = Qwen35RecurrentScratchBufferKey {
        tokens_per_slot,
        conv_dim: cfg.conv_dim,
        time_step_rank: cfg.time_step_rank,
        state_size: cfg.state_size,
    };
    assert_eq!(
        backend
            .qwen35_recurrent_scratch_buffers
            .lock()
            .unwrap()
            .len(),
        1
    );
    assert!(
        backend
            .qwen35_recurrent_scratch_buffers
            .lock()
            .unwrap()
            .contains_key(&scratch_key)
    );
}

#[test]
fn test_metal_ops_qwen35_recurrent_projection_scratch_reuses_buffers_by_shape() {
    let backend = MetalBackend::new().unwrap();

    let first_ptrs =
        backend
            .ops
            .with_qwen35_recurrent_projection_scratch(64, 4096, 2048, 128, |scratch| {
                (
                    scratch.qkv.ptr_id(),
                    scratch.z.ptr_id(),
                    scratch.beta.ptr_id(),
                    scratch.alpha.ptr_id(),
                )
            });
    let second_ptrs =
        backend
            .ops
            .with_qwen35_recurrent_projection_scratch(64, 4096, 2048, 128, |scratch| {
                (
                    scratch.qkv.ptr_id(),
                    scratch.z.ptr_id(),
                    scratch.beta.ptr_id(),
                    scratch.alpha.ptr_id(),
                )
            });

    assert_eq!(first_ptrs, second_ptrs);

    let third_ptrs =
        backend
            .ops
            .with_qwen35_recurrent_projection_scratch(128, 4096, 2048, 128, |scratch| {
                (
                    scratch.qkv.ptr_id(),
                    scratch.z.ptr_id(),
                    scratch.beta.ptr_id(),
                    scratch.alpha.ptr_id(),
                )
            });

    assert_ne!(first_ptrs, third_ptrs);
    let cache = backend
        .ops
        .qwen35_recurrent_projection_scratch_buffers
        .lock()
        .unwrap();
    assert_eq!(cache.len(), 2);
}

#[test]
fn test_metal_ops_qwen35_batch_projection_scratch_reuses_buffers_by_shape() {
    let backend = MetalBackend::new().unwrap();

    let first_ptrs =
        backend
            .ops
            .with_qwen35_batch_projection_scratch(64, &[4096, 4096, 1024], |scratch| {
                scratch
                    .outputs
                    .iter()
                    .map(MetalBuffer::ptr_id)
                    .collect::<Vec<_>>()
            });
    let second_ptrs =
        backend
            .ops
            .with_qwen35_batch_projection_scratch(64, &[4096, 4096, 1024], |scratch| {
                scratch
                    .outputs
                    .iter()
                    .map(MetalBuffer::ptr_id)
                    .collect::<Vec<_>>()
            });

    assert_eq!(first_ptrs, second_ptrs);

    let third_ptrs =
        backend
            .ops
            .with_qwen35_batch_projection_scratch(128, &[4096, 4096, 1024], |scratch| {
                scratch
                    .outputs
                    .iter()
                    .map(MetalBuffer::ptr_id)
                    .collect::<Vec<_>>()
            });

    assert_ne!(first_ptrs, third_ptrs);
    let cache = backend
        .ops
        .qwen35_batch_projection_scratch_buffers
        .lock()
        .unwrap();
    assert_eq!(cache.len(), 2);
}

#[test]
fn test_metal_ops_qwen35_batch_logits_scratch_reuses_buffers_by_shape() {
    let backend = MetalBackend::new().unwrap();

    let first_ptrs = backend
        .ops
        .with_qwen35_batch_logits_scratch(4096, 32000, |scratch| {
            vec![
                scratch.hidden.ptr_id(),
                scratch.hidden_f16.ptr_id(),
                scratch.logits.ptr_id(),
            ]
        });
    let second_ptrs = backend
        .ops
        .with_qwen35_batch_logits_scratch(4096, 32000, |scratch| {
            vec![
                scratch.hidden.ptr_id(),
                scratch.hidden_f16.ptr_id(),
                scratch.logits.ptr_id(),
            ]
        });
    let third_ptrs = backend
        .ops
        .with_qwen35_batch_logits_scratch(8192, 32000, |scratch| {
            vec![
                scratch.hidden.ptr_id(),
                scratch.hidden_f16.ptr_id(),
                scratch.logits.ptr_id(),
            ]
        });

    assert_eq!(first_ptrs, second_ptrs);
    assert_ne!(first_ptrs, third_ptrs);
    let cache = backend
        .ops
        .qwen35_batch_logits_scratch_buffers
        .lock()
        .unwrap();
    assert_eq!(cache.len(), 2);
}

#[test]
#[ignore = "parallel Metal backend suite is nondeterministic for this recurrent path; run standalone"]
fn test_metal_backend_qwen35_recurrent_sequence_for_kv_multi_token_keeps_backend_owned_state() {
    let _env_lock = lock_env_test();
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 0usize;
    let tokens_per_slot = 5usize;
    let slot_indices = [0usize];
    let mut expected_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let mut actual_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);

    expected_kv
        .conv_state_for_slot_mut(0, layer_idx)
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = -0.05 + i as f32 * 0.002);
    expected_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = 0.07 + i as f32 * 0.0015);
    actual_kv
        .conv_state_for_slot_mut(0, layer_idx)
        .copy_from_slice(expected_kv.conv_state_for_slot(0, layer_idx));
    actual_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .copy_from_slice(expected_kv.recurrent_state_for_slot(0, layer_idx));

    let qkv: Vec<f32> = (0..tokens_per_slot * cfg.conv_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.03)
        .collect();
    let mut expected_alpha: Vec<f32> = (0..tokens_per_slot * cfg.time_step_rank)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.07)
        .collect();
    let mut expected_beta: Vec<f32> = (0..tokens_per_slot * cfg.time_step_rank)
        .map(|i| 0.1 + (i % 11) as f32 * 0.02)
        .collect();
    let mut actual_alpha = expected_alpha.clone();
    let mut actual_beta = expected_beta.clone();
    let dt_bias = vec![0.03, 0.04, 0.05, 0.06];
    let a = vec![0.2, 0.25, 0.3, 0.35];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let mut expected_out = vec![0.0f32; tokens_per_slot * cfg.value_dim()];
    let mut actual_out = vec![0.0f32; tokens_per_slot * cfg.value_dim()];

    cpu.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut expected_beta,
        &mut expected_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &slot_indices,
        &mut expected_out,
        tokens_per_slot,
        cfg,
    );
    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut actual_beta,
        &mut actual_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut actual_out,
        tokens_per_slot,
        cfg,
    );

    assert!(max_abs_diff(&actual_alpha, &expected_alpha) < 1e-5);
    assert!(max_abs_diff(&actual_beta, &expected_beta) < 1e-5);
    assert!(max_abs_diff(&actual_out, &expected_out) < 1e-4);
    assert!(
        actual_kv.conv_state_cpu_stale(0, layer_idx),
        "multi-token qwen35 recurrent KV path should keep conv state backend-owned",
    );
    assert!(
        actual_kv.recurrent_state_cpu_stale(0, layer_idx),
        "multi-token qwen35 recurrent KV path should keep recurrent state backend-owned",
    );
    assert!(
        !backend
            .ops
            .qwen35_recurrent_slot_buffers
            .lock()
            .unwrap()
            .is_empty(),
        "multi-token qwen35 recurrent KV path should populate cached Metal slot buffers",
    );

    backend.sync_qwen35_kv(&mut actual_kv);

    assert!(
        max_abs_diff(
            actual_kv.conv_state_for_slot(0, layer_idx),
            expected_kv.conv_state_for_slot(0, layer_idx),
        ) < 1e-5
    );
    assert!(
        max_abs_diff(
            actual_kv.recurrent_state_for_slot(0, layer_idx),
            expected_kv.recurrent_state_for_slot(0, layer_idx),
        ) < 1e-4
    );
}

#[test]
fn test_metal_backend_qwen35_recurrent_sequence_for_kv_multi_token_then_single_token_matches_cpu() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 0usize;
    let slot_indices = [0usize];
    let warmup_tokens_per_slot = 4usize;
    let decode_tokens_per_slot = 1usize;
    let mut expected_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let mut actual_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);

    expected_kv
        .conv_state_for_slot_mut(0, layer_idx)
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = -0.03 + i as f32 * 0.0017);
    expected_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = 0.05 + i as f32 * 0.0013);
    actual_kv
        .conv_state_for_slot_mut(0, layer_idx)
        .copy_from_slice(expected_kv.conv_state_for_slot(0, layer_idx));
    actual_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .copy_from_slice(expected_kv.recurrent_state_for_slot(0, layer_idx));

    let warmup_qkv: Vec<f32> = (0..warmup_tokens_per_slot * cfg.conv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.028)
        .collect();
    let mut warmup_expected_alpha: Vec<f32> = (0..warmup_tokens_per_slot * cfg.time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.045)
        .collect();
    let mut warmup_expected_beta: Vec<f32> = (0..warmup_tokens_per_slot * cfg.time_step_rank)
        .map(|i| 0.09 + (i % 9) as f32 * 0.017)
        .collect();
    let mut warmup_actual_alpha = warmup_expected_alpha.clone();
    let mut warmup_actual_beta = warmup_expected_beta.clone();
    let dt_bias = vec![0.03, 0.04, 0.05, 0.06];
    let a = vec![0.2, 0.25, 0.3, 0.35];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let mut warmup_expected_out = vec![0.0f32; warmup_tokens_per_slot * cfg.value_dim()];
    let mut warmup_actual_out = vec![0.0f32; warmup_tokens_per_slot * cfg.value_dim()];

    cpu.qwen35_recurrent_sequence_for_kv(
        &warmup_qkv,
        &mut warmup_expected_beta,
        &mut warmup_expected_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &slot_indices,
        &mut warmup_expected_out,
        warmup_tokens_per_slot,
        cfg,
    );
    backend.qwen35_recurrent_sequence_for_kv(
        &warmup_qkv,
        &mut warmup_actual_beta,
        &mut warmup_actual_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut warmup_actual_out,
        warmup_tokens_per_slot,
        cfg,
    );

    assert!(max_abs_diff(&warmup_actual_alpha, &warmup_expected_alpha) < 1e-5);
    assert!(max_abs_diff(&warmup_actual_beta, &warmup_expected_beta) < 1e-5);
    assert!(max_abs_diff(&warmup_actual_out, &warmup_expected_out) < 1e-4);
    assert!(
        actual_kv.conv_state_cpu_stale(0, layer_idx),
        "multi-token warmup should leave conv state backend-owned",
    );
    assert!(
        actual_kv.recurrent_state_cpu_stale(0, layer_idx),
        "multi-token warmup should leave recurrent state backend-owned",
    );

    let decode_qkv: Vec<f32> = (0..cfg.conv_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.031)
        .collect();
    let mut decode_expected_alpha: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.052)
        .collect();
    let mut decode_expected_beta: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| 0.07 + i as f32 * 0.019)
        .collect();
    let mut decode_actual_alpha = decode_expected_alpha.clone();
    let mut decode_actual_beta = decode_expected_beta.clone();
    let mut decode_expected_out = vec![0.0f32; cfg.value_dim()];
    let mut decode_actual_out = vec![0.0f32; cfg.value_dim()];

    cpu.qwen35_recurrent_sequence_for_kv(
        &decode_qkv,
        &mut decode_expected_beta,
        &mut decode_expected_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &slot_indices,
        &mut decode_expected_out,
        decode_tokens_per_slot,
        cfg,
    );
    backend.qwen35_recurrent_sequence_for_kv(
        &decode_qkv,
        &mut decode_actual_beta,
        &mut decode_actual_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut decode_actual_out,
        decode_tokens_per_slot,
        cfg,
    );

    assert!(max_abs_diff(&decode_actual_alpha, &decode_expected_alpha) < 1e-5);
    assert!(max_abs_diff(&decode_actual_beta, &decode_expected_beta) < 1e-5);
    assert!(max_abs_diff(&decode_actual_out, &decode_expected_out) < 1e-4);
    backend.sync_qwen35_kv(&mut actual_kv);
    assert!(
        max_abs_diff(
            actual_kv.conv_state_for_slot(0, layer_idx),
            expected_kv.conv_state_for_slot(0, layer_idx),
        ) < 1e-5
    );
    assert!(
        max_abs_diff(
            actual_kv.recurrent_state_for_slot(0, layer_idx),
            expected_kv.recurrent_state_for_slot(0, layer_idx),
        ) < 1e-4
    );
}

#[test]
fn test_metal_backend_qwen35_recurrent_sequence_for_kv_multi_token_then_single_token_matches_cpu_with_gpu_resident_kv()
 {
    let _env_lock = lock_env_test();
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 0usize;
    let slot_indices = [0usize];
    let warmup_tokens_per_slot = 4usize;
    let decode_tokens_per_slot = 1usize;
    let mut expected_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let mut actual_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    actual_kv
        .enable_gpu_recurrent_state(&backend.ops.device)
        .unwrap();

    expected_kv
        .conv_state_for_slot_mut(0, layer_idx)
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = -0.03 + i as f32 * 0.0017);
    expected_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = 0.05 + i as f32 * 0.0013);
    actual_kv
        .conv_state_for_slot_mut(0, layer_idx)
        .copy_from_slice(expected_kv.conv_state_for_slot(0, layer_idx));
    actual_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .copy_from_slice(expected_kv.recurrent_state_for_slot(0, layer_idx));

    let warmup_qkv: Vec<f32> = (0..warmup_tokens_per_slot * cfg.conv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.028)
        .collect();
    let mut warmup_expected_alpha: Vec<f32> = (0..warmup_tokens_per_slot * cfg.time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.045)
        .collect();
    let mut warmup_expected_beta: Vec<f32> = (0..warmup_tokens_per_slot * cfg.time_step_rank)
        .map(|i| 0.09 + (i % 9) as f32 * 0.017)
        .collect();
    let mut warmup_actual_alpha = warmup_expected_alpha.clone();
    let mut warmup_actual_beta = warmup_expected_beta.clone();
    let dt_bias = vec![0.03, 0.04, 0.05, 0.06];
    let a = vec![0.2, 0.25, 0.3, 0.35];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let mut warmup_expected_out = vec![0.0f32; warmup_tokens_per_slot * cfg.value_dim()];
    let mut warmup_actual_out = vec![0.0f32; warmup_tokens_per_slot * cfg.value_dim()];

    cpu.qwen35_recurrent_sequence_for_kv(
        &warmup_qkv,
        &mut warmup_expected_beta,
        &mut warmup_expected_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &slot_indices,
        &mut warmup_expected_out,
        warmup_tokens_per_slot,
        cfg,
    );
    backend.qwen35_recurrent_sequence_for_kv(
        &warmup_qkv,
        &mut warmup_actual_beta,
        &mut warmup_actual_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut warmup_actual_out,
        warmup_tokens_per_slot,
        cfg,
    );

    assert!(max_abs_diff(&warmup_actual_alpha, &warmup_expected_alpha) < 1e-5);
    assert!(max_abs_diff(&warmup_actual_beta, &warmup_expected_beta) < 1e-5);
    assert!(max_abs_diff(&warmup_actual_out, &warmup_expected_out) < 1e-4);

    let decode_qkv: Vec<f32> = (0..cfg.conv_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.031)
        .collect();
    let mut decode_expected_alpha: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.052)
        .collect();
    let mut decode_expected_beta: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| 0.07 + i as f32 * 0.019)
        .collect();
    let mut decode_actual_alpha = decode_expected_alpha.clone();
    let mut decode_actual_beta = decode_expected_beta.clone();
    let mut decode_expected_out = vec![0.0f32; cfg.value_dim()];
    let mut decode_actual_out = vec![0.0f32; cfg.value_dim()];

    cpu.qwen35_recurrent_sequence_for_kv(
        &decode_qkv,
        &mut decode_expected_beta,
        &mut decode_expected_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &slot_indices,
        &mut decode_expected_out,
        decode_tokens_per_slot,
        cfg,
    );
    backend.qwen35_recurrent_sequence_for_kv(
        &decode_qkv,
        &mut decode_actual_beta,
        &mut decode_actual_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut decode_actual_out,
        decode_tokens_per_slot,
        cfg,
    );

    assert!(max_abs_diff(&decode_actual_alpha, &decode_expected_alpha) < 1e-5);
    assert!(max_abs_diff(&decode_actual_beta, &decode_expected_beta) < 1e-5);
    assert!(max_abs_diff(&decode_actual_out, &decode_expected_out) < 1e-4);
    backend.sync_qwen35_kv(&mut actual_kv);
    assert!(
        max_abs_diff(
            actual_kv.conv_state_for_slot(0, layer_idx),
            expected_kv.conv_state_for_slot(0, layer_idx),
        ) < 1e-5
    );
    assert!(
        max_abs_diff(
            actual_kv.recurrent_state_for_slot(0, layer_idx),
            expected_kv.recurrent_state_for_slot(0, layer_idx),
        ) < 1e-4
    );
}

#[test]
fn test_metal_backend_qwen35_recurrent_sequence_for_kv_multi_slot_multi_token_keeps_backend_owned_state()
 {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 0usize;
    let warmup_tokens_per_slot = 1usize;
    let tokens_per_slot = 2usize;
    let dt_bias = vec![0.03, 0.04, 0.05, 0.06];
    let a = vec![0.2, 0.25, 0.3, 0.35];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();

    let mut expected_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let mut actual_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let slot1 = expected_kv.allocate_recurrent_slot();
    let actual_slot1 = actual_kv.allocate_recurrent_slot();
    assert_eq!(slot1, actual_slot1);

    let slot0_qkv: Vec<f32> = (0..cfg.conv_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
        .collect();
    let slot1_qkv: Vec<f32> = (0..cfg.conv_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.035)
        .collect();
    let slot0_alpha: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let slot1_alpha: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.06)
        .collect();
    let slot0_beta: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| 0.08 + i as f32 * 0.02)
        .collect();
    let slot1_beta: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| 0.11 + i as f32 * 0.015)
        .collect();

    let mut warmup_slot0_alpha_cpu = slot0_alpha.clone();
    let mut warmup_slot0_beta_cpu = slot0_beta.clone();
    let mut warmup_slot0_alpha_metal = slot0_alpha.clone();
    let mut warmup_slot0_beta_metal = slot0_beta.clone();
    let mut warmup_slot1_alpha_cpu = slot1_alpha.clone();
    let mut warmup_slot1_beta_cpu = slot1_beta.clone();
    let mut warmup_slot1_alpha_metal = slot1_alpha.clone();
    let mut warmup_slot1_beta_metal = slot1_beta.clone();
    let mut warmup_out_cpu = vec![0.0f32; cfg.value_dim()];
    let mut warmup_out_metal = vec![0.0f32; cfg.value_dim()];

    cpu.qwen35_recurrent_sequence_for_kv(
        &slot0_qkv,
        &mut warmup_slot0_beta_cpu,
        &mut warmup_slot0_alpha_cpu,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &[0usize],
        &mut warmup_out_cpu,
        warmup_tokens_per_slot,
        cfg,
    );
    backend.qwen35_recurrent_sequence_for_kv(
        &slot0_qkv,
        &mut warmup_slot0_beta_metal,
        &mut warmup_slot0_alpha_metal,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &[0usize],
        &mut warmup_out_metal,
        warmup_tokens_per_slot,
        cfg,
    );

    cpu.qwen35_recurrent_sequence_for_kv(
        &slot1_qkv,
        &mut warmup_slot1_beta_cpu,
        &mut warmup_slot1_alpha_cpu,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &[slot1],
        &mut warmup_out_cpu,
        warmup_tokens_per_slot,
        cfg,
    );
    backend.qwen35_recurrent_sequence_for_kv(
        &slot1_qkv,
        &mut warmup_slot1_beta_metal,
        &mut warmup_slot1_alpha_metal,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &[slot1],
        &mut warmup_out_metal,
        warmup_tokens_per_slot,
        cfg,
    );

    assert!(
        actual_kv.recurrent_state_cpu_stale(0, layer_idx),
        "slot 0 warmup should leave recurrent state backend-owned"
    );
    assert!(
        actual_kv.recurrent_state_cpu_stale(slot1, layer_idx),
        "slot 1 warmup should leave recurrent state backend-owned"
    );

    let slot_indices = [0usize, slot1];
    let total_tokens = slot_indices.len() * tokens_per_slot;
    let qkv: Vec<f32> = (0..total_tokens * cfg.conv_dim)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.025)
        .collect();
    let mut expected_alpha: Vec<f32> = (0..total_tokens * cfg.time_step_rank)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.04)
        .collect();
    let mut expected_beta: Vec<f32> = (0..total_tokens * cfg.time_step_rank)
        .map(|i| 0.09 + (i % 13) as f32 * 0.015)
        .collect();
    let mut actual_alpha = expected_alpha.clone();
    let mut actual_beta = expected_beta.clone();
    let mut expected_out = vec![0.0f32; total_tokens * cfg.value_dim()];
    let mut actual_out = vec![0.0f32; total_tokens * cfg.value_dim()];

    cpu.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut expected_beta,
        &mut expected_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &slot_indices,
        &mut expected_out,
        tokens_per_slot,
        cfg,
    );
    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut actual_beta,
        &mut actual_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut actual_out,
        tokens_per_slot,
        cfg,
    );

    assert!(max_abs_diff(&actual_alpha, &expected_alpha) < 1e-5);
    assert!(max_abs_diff(&actual_beta, &expected_beta) < 1e-5);
    assert!(max_abs_diff(&actual_out, &expected_out) < 1e-4);
    assert!(
        actual_kv.conv_state_cpu_stale(0, layer_idx),
        "multi-token slot 0 path should keep conv state backend-owned",
    );
    assert!(
        actual_kv.recurrent_state_cpu_stale(0, layer_idx),
        "multi-token slot 0 path should keep recurrent state backend-owned",
    );
    assert!(
        actual_kv.conv_state_cpu_stale(slot1, layer_idx),
        "multi-token slot 1 path should keep conv state backend-owned",
    );
    assert!(
        actual_kv.recurrent_state_cpu_stale(slot1, layer_idx),
        "multi-token slot 1 path should keep recurrent state backend-owned",
    );

    backend.sync_qwen35_kv(&mut actual_kv);

    assert!(
        max_abs_diff(
            actual_kv.conv_state_for_slot(0, layer_idx),
            expected_kv.conv_state_for_slot(0, layer_idx),
        ) < 1e-5
    );
    assert!(
        max_abs_diff(
            actual_kv.recurrent_state_for_slot(0, layer_idx),
            expected_kv.recurrent_state_for_slot(0, layer_idx),
        ) < 1e-4
    );
    assert!(
        max_abs_diff(
            actual_kv.conv_state_for_slot(slot1, layer_idx),
            expected_kv.conv_state_for_slot(slot1, layer_idx),
        ) < 1e-5
    );
    assert!(
        max_abs_diff(
            actual_kv.recurrent_state_for_slot(slot1, layer_idx),
            expected_kv.recurrent_state_for_slot(slot1, layer_idx),
        ) < 1e-4
    );
}

#[test]
fn test_metal_backend_qwen35_recurrent_sequence_for_kv_reloads_after_cpu_mutation() {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 0usize;
    let tokens_per_slot = 1usize;
    let slot_indices = [0usize];
    let mut expected_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let mut actual_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);

    let qkv: Vec<f32> = (0..cfg.conv_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
        .collect();
    let expected_alpha: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let expected_beta: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| 0.08 + i as f32 * 0.02)
        .collect();
    let mut actual_alpha = expected_alpha.clone();
    let mut actual_beta = expected_beta.clone();
    let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
    let a = vec![0.11, 0.13, 0.17, 0.19];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
        .collect();
    let mut warmup_out = vec![0.0f32; cfg.value_dim()];

    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut actual_beta,
        &mut actual_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut warmup_out,
        tokens_per_slot,
        cfg,
    );
    assert!(
        actual_kv.recurrent_state_cpu_stale(0, layer_idx),
        "single-token warmup should leave recurrent state backend-owned"
    );
    backend.sync_qwen35_kv(&mut actual_kv);
    expected_kv
        .conv_state_for_slot_mut(0, layer_idx)
        .copy_from_slice(actual_kv.conv_state_for_slot(0, layer_idx));
    expected_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .copy_from_slice(actual_kv.recurrent_state_for_slot(0, layer_idx));

    actual_kv.conv_state_for_slot_mut(0, layer_idx).fill(0.33);
    actual_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .fill(-0.27);
    expected_kv.conv_state_for_slot_mut(0, layer_idx).fill(0.33);
    expected_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .fill(-0.27);

    let cpu_generation = actual_kv.recurrent_state_generation(0, layer_idx);
    let mut expected_alpha_2 = expected_alpha.clone();
    let mut expected_beta_2 = expected_beta.clone();
    let mut actual_alpha_2 = expected_alpha.clone();
    let mut actual_beta_2 = expected_beta.clone();
    let mut expected_out = vec![0.0f32; cfg.value_dim()];
    let mut actual_out = vec![0.0f32; cfg.value_dim()];

    cpu.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut expected_beta_2,
        &mut expected_alpha_2,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &slot_indices,
        &mut expected_out,
        tokens_per_slot,
        cfg,
    );
    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut actual_beta_2,
        &mut actual_alpha_2,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut actual_out,
        tokens_per_slot,
        cfg,
    );

    assert!(max_abs_diff(&actual_alpha_2, &expected_alpha_2) < 1e-5);
    assert!(max_abs_diff(&actual_beta_2, &expected_beta_2) < 1e-5);
    assert!(max_abs_diff(&actual_out, &expected_out) < 1e-4);
    assert!(
        actual_kv.recurrent_state_cpu_stale(0, layer_idx),
        "single-token reload path should again leave recurrent state backend-owned"
    );
    backend.sync_qwen35_kv(&mut actual_kv);
    assert!(
        max_abs_diff(
            actual_kv.conv_state_for_slot(0, layer_idx),
            expected_kv.conv_state_for_slot(0, layer_idx),
        ) < 1e-5
    );
    assert!(
        max_abs_diff(
            actual_kv.recurrent_state_for_slot(0, layer_idx),
            expected_kv.recurrent_state_for_slot(0, layer_idx),
        ) < 1e-4
    );

    let slot_key = Qwen35RecurrentSlotBufferKey {
        layer_idx,
        slot_idx: 0,
        conv_state_stride: actual_kv.conv_cache_len() * actual_kv.conv_dim(),
        recurrent_state_stride: actual_kv.recurrent_state_len(),
    };
    let slot_cache = backend.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
    let slot_buffers = slot_cache.get(&slot_key).unwrap();
    assert!(
        actual_kv.recurrent_state_generation(0, layer_idx) > cpu_generation,
        "qwen35 single-token reload path should advance CPU state generation"
    );
    assert_eq!(
        slot_buffers.recurrent_synced_generation,
        Some(actual_kv.recurrent_state_generation(0, layer_idx))
    );
}

#[test]
fn test_metal_backend_qwen35_recurrent_sequence_for_kv_keeps_backend_recurrent_state_on_conv_only_cpu_mutation()
 {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 0usize;
    let tokens_per_slot = 1usize;
    let slot_indices = [0usize];
    let mut expected_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let mut actual_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);

    let qkv: Vec<f32> = (0..cfg.conv_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
        .collect();
    let expected_alpha: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let expected_beta: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| 0.08 + i as f32 * 0.02)
        .collect();
    let mut actual_alpha = expected_alpha.clone();
    let mut actual_beta = expected_beta.clone();
    let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
    let a = vec![0.11, 0.13, 0.17, 0.19];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
        .collect();
    let mut warmup_alpha_cpu = expected_alpha.clone();
    let mut warmup_beta_cpu = expected_beta.clone();
    let mut warmup_out_cpu = vec![0.0f32; cfg.value_dim()];
    let mut warmup_out_metal = vec![0.0f32; cfg.value_dim()];

    cpu.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut warmup_beta_cpu,
        &mut warmup_alpha_cpu,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &slot_indices,
        &mut warmup_out_cpu,
        tokens_per_slot,
        cfg,
    );
    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut actual_beta,
        &mut actual_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut warmup_out_metal,
        tokens_per_slot,
        cfg,
    );

    assert!(
        actual_kv.recurrent_state_cpu_stale(0, layer_idx),
        "warmup should leave recurrent state backend-owned"
    );

    expected_kv.conv_state_for_slot_mut(0, layer_idx).fill(0.41);
    actual_kv.conv_state_for_slot_mut(0, layer_idx).fill(0.41);

    let mut expected_alpha_2 = expected_alpha.clone();
    let mut expected_beta_2 = expected_beta.clone();
    let mut actual_alpha_2 = expected_alpha.clone();
    let mut actual_beta_2 = expected_beta.clone();
    let mut expected_out = vec![0.0f32; cfg.value_dim()];
    let mut actual_out = vec![0.0f32; cfg.value_dim()];

    cpu.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut expected_beta_2,
        &mut expected_alpha_2,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &slot_indices,
        &mut expected_out,
        tokens_per_slot,
        cfg,
    );
    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut actual_beta_2,
        &mut actual_alpha_2,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut actual_out,
        tokens_per_slot,
        cfg,
    );

    assert!(max_abs_diff(&actual_alpha_2, &expected_alpha_2) < 1e-5);
    assert!(max_abs_diff(&actual_beta_2, &expected_beta_2) < 1e-5);
    assert!(max_abs_diff(&actual_out, &expected_out) < 1e-4);
    backend.sync_qwen35_kv(&mut actual_kv);
    assert!(
        max_abs_diff(
            actual_kv.conv_state_for_slot(0, layer_idx),
            expected_kv.conv_state_for_slot(0, layer_idx),
        ) < 1e-5
    );
    assert!(
        max_abs_diff(
            actual_kv.recurrent_state_for_slot(0, layer_idx),
            expected_kv.recurrent_state_for_slot(0, layer_idx),
        ) < 1e-4
    );
}

#[test]
fn test_metal_backend_qwen35_recurrent_sequence_for_kv_reloads_cpu_materialized_state_across_fresh_kv_instances()
 {
    let backend = MetalBackend::new().unwrap();
    let cpu = super::super::cpu::CpuBackend;
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 0usize;
    let tokens_per_slot = 1usize;
    let slot_indices = [0usize];

    let qkv: Vec<f32> = (0..cfg.conv_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
        .collect();
    let alpha: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let beta: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| 0.08 + i as f32 * 0.02)
        .collect();
    let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
    let a = vec![0.11, 0.13, 0.17, 0.19];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
        .collect();

    let mut stale_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let mut stale_alpha = alpha.clone();
    let mut stale_beta = beta.clone();
    let mut stale_out = vec![0.0f32; cfg.value_dim()];
    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut stale_beta,
        &mut stale_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut stale_kv,
        layer_idx,
        &slot_indices,
        &mut stale_out,
        tokens_per_slot,
        cfg,
    );
    assert!(
        stale_kv.recurrent_state_cpu_stale(0, layer_idx),
        "warmup should leave recurrent state backend-owned"
    );

    let mut expected_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let mut actual_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    expected_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .fill(-0.27);
    actual_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .fill(-0.27);
    expected_kv.conv_state_for_slot_mut(0, layer_idx).fill(0.41);
    actual_kv.conv_state_for_slot_mut(0, layer_idx).fill(0.41);

    let mut expected_alpha = alpha.clone();
    let mut expected_beta = beta.clone();
    let mut actual_alpha = alpha.clone();
    let mut actual_beta = beta.clone();
    let mut expected_out = vec![0.0f32; cfg.value_dim()];
    let mut actual_out = vec![0.0f32; cfg.value_dim()];

    cpu.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut expected_beta,
        &mut expected_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut expected_kv,
        layer_idx,
        &slot_indices,
        &mut expected_out,
        tokens_per_slot,
        cfg,
    );
    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut actual_beta,
        &mut actual_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut actual_out,
        tokens_per_slot,
        cfg,
    );

    assert!(max_abs_diff(&actual_alpha, &expected_alpha) < 1e-5);
    assert!(max_abs_diff(&actual_beta, &expected_beta) < 1e-5);
    assert!(max_abs_diff(&actual_out, &expected_out) < 1e-4);
    backend.sync_qwen35_kv(&mut actual_kv);
    assert!(
        max_abs_diff(
            actual_kv.recurrent_state_for_slot(0, layer_idx),
            expected_kv.recurrent_state_for_slot(0, layer_idx),
        ) < 1e-4
    );
}

#[test]
fn test_metal_backend_sync_qwen35_kv_skips_slot_buffers_for_missing_slots() {
    let backend = MetalBackend::new().unwrap();
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 0usize;
    let tokens_per_slot = 1usize;
    let mut cached_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let slot1 = cached_kv.allocate_recurrent_slot();
    let slot_indices = [slot1];

    let qkv: Vec<f32> = (0..cfg.conv_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
        .collect();
    let mut alpha: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let mut beta: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| 0.08 + i as f32 * 0.02)
        .collect();
    let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
    let a = vec![0.11, 0.13, 0.17, 0.19];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
        .collect();
    let mut out = vec![0.0f32; cfg.value_dim()];

    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut beta,
        &mut alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut cached_kv,
        layer_idx,
        &slot_indices,
        &mut out,
        tokens_per_slot,
        cfg,
    );

    let mut fresh_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    backend.sync_qwen35_kv(&mut fresh_kv);
    assert_eq!(fresh_kv.active_slot(), 0);
    assert_eq!(fresh_kv.seq_len(), 0);
}

#[test]
fn test_metal_backend_sync_qwen35_kv_skips_slot_buffers_for_missing_layers() {
    let backend = MetalBackend::new().unwrap();
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 4usize;
    let tokens_per_slot = 1usize;
    let slot_indices = [0usize];
    let mut larger_kv = crate::kv::Qwen3_5Kv::new(8, 1, 2, 16, 4, 4, 8, 2, 4, 2);

    let qkv: Vec<f32> = (0..cfg.conv_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
        .collect();
    let mut alpha: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let mut beta: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| 0.08 + i as f32 * 0.02)
        .collect();
    let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
    let a = vec![0.11, 0.13, 0.17, 0.19];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
        .collect();
    let mut out = vec![0.0f32; cfg.value_dim()];

    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut beta,
        &mut alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut larger_kv,
        layer_idx,
        &slot_indices,
        &mut out,
        tokens_per_slot,
        cfg,
    );

    let mut smaller_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    backend.sync_qwen35_kv(&mut smaller_kv);
    assert_eq!(smaller_kv.active_slot(), 0);
    assert_eq!(smaller_kv.seq_len(), 0);
}

#[test]
fn test_metal_backend_sync_qwen35_kv_skips_slot_buffers_for_mismatched_shapes() {
    let backend = MetalBackend::new().unwrap();
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 0usize;
    let tokens_per_slot = 1usize;
    let slot_indices = [0usize];
    let mut old_shape_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);

    let qkv: Vec<f32> = (0..cfg.conv_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
        .collect();
    let mut alpha: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let mut beta: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| 0.08 + i as f32 * 0.02)
        .collect();
    let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
    let a = vec![0.11, 0.13, 0.17, 0.19];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
        .collect();
    let mut out = vec![0.0f32; cfg.value_dim()];

    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut beta,
        &mut alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut old_shape_kv,
        layer_idx,
        &slot_indices,
        &mut out,
        tokens_per_slot,
        cfg,
    );

    let mut different_shape_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 12, 3, 4, 2);
    let generation = different_shape_kv.note_backend_recurrent_state_update(0, layer_idx);
    assert_eq!(generation, 2, "fresh kv should advance to generation 2");
    assert!(different_shape_kv.recurrent_state_cpu_stale(0, layer_idx));

    backend.sync_qwen35_kv(&mut different_shape_kv);

    assert!(
        different_shape_kv.recurrent_state_cpu_stale(0, layer_idx),
        "mismatched cached Metal slot buffers must not sync into a different qwen35 shape"
    );
    assert!(
        different_shape_kv
            .recurrent_state_for_slot(0, layer_idx)
            .iter()
            .all(|&v| v == 0.0)
    );
}

#[test]
#[should_panic(
    expected = "cannot snapshot qwen35 recurrent slot while backend-owned recurrent state is not materialized on CPU"
)]
fn test_metal_backend_qwen35_snapshot_requires_sync_after_backend_owned_decode() {
    let backend = MetalBackend::new().unwrap();
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 0usize;
    let tokens_per_slot = 1usize;
    let slot_indices = [0usize];
    let mut qwen_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);

    let qkv: Vec<f32> = (0..cfg.conv_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
        .collect();
    let mut alpha: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let mut beta: Vec<f32> = (0..cfg.time_step_rank)
        .map(|i| 0.08 + i as f32 * 0.02)
        .collect();
    let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
    let a = vec![0.11, 0.13, 0.17, 0.19];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
        .collect();
    let mut out = vec![0.0f32; cfg.value_dim()];

    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut beta,
        &mut alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut qwen_kv,
        layer_idx,
        &slot_indices,
        &mut out,
        tokens_per_slot,
        cfg,
    );

    assert!(qwen_kv.recurrent_state_cpu_stale(0, layer_idx));
    let _ = qwen_kv.snapshot_active_slot();
}

#[test]
fn test_metal_backend_try_clone_qwen35_recurrent_slot_preserves_backend_owned_state() {
    let backend = MetalBackend::new().unwrap();
    let mut qwen_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let layer_idx = 0usize;
    let dst_slot = qwen_kv.allocate_recurrent_slot();
    let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
    let recurrent_state_stride = qwen_kv.recurrent_state_len();

    qwen_kv.conv_state_for_slot_mut(0, layer_idx).fill(-1.0);
    qwen_kv
        .recurrent_state_for_slot_mut(0, layer_idx)
        .fill(-2.0);

    let expected_conv: Vec<f32> = (0..conv_state_stride)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let expected_recurrent: Vec<f32> = (0..recurrent_state_stride)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let conv_generation = qwen_kv.note_backend_conv_state_update(0, layer_idx);
    let recurrent_generation = qwen_kv.note_backend_recurrent_state_update(0, layer_idx);
    let kv_identity = &qwen_kv as *const crate::kv::Qwen3_5Kv as usize;
    backend.ops.with_qwen35_recurrent_slot_buffer(
        layer_idx,
        0,
        conv_state_stride,
        recurrent_state_stride,
        |slot_buffers| {
            unsafe {
                slot_buffers.conv_state.as_mut_slice::<f32>()[..conv_state_stride]
                    .copy_from_slice(&expected_conv);
                slot_buffers.recurrent_state.as_mut_slice::<f32>()[..recurrent_state_stride]
                    .copy_from_slice(&expected_recurrent);
            }
            slot_buffers.conv_synced_generation = Some(conv_generation);
            slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
            slot_buffers.source_kv_identity = Some(kv_identity);
        },
    );

    assert!(qwen_kv.conv_state_cpu_stale(0, layer_idx));
    assert!(qwen_kv.recurrent_state_cpu_stale(0, layer_idx));
    assert!(backend.try_clone_qwen35_recurrent_slot_from_backend_owned(&mut qwen_kv, 0, dst_slot));
    assert!(qwen_kv.conv_state_cpu_stale(dst_slot, layer_idx));
    assert!(qwen_kv.recurrent_state_cpu_stale(dst_slot, layer_idx));
    assert_eq!(
        qwen_kv.conv_state_generation(dst_slot, layer_idx),
        conv_generation
    );
    assert_eq!(
        qwen_kv.recurrent_state_generation(dst_slot, layer_idx),
        recurrent_generation
    );

    backend.sync_qwen35_kv(&mut qwen_kv);

    assert_eq!(
        qwen_kv.conv_state_for_slot(dst_slot, layer_idx),
        expected_conv.as_slice()
    );
    assert_eq!(
        qwen_kv.recurrent_state_for_slot(dst_slot, layer_idx),
        expected_recurrent.as_slice()
    );
}

#[test]
fn test_sync_qwen35_slot_buffers_from_kv_zero_inits_pristine_cpu_state_without_copy() {
    let backend = MetalBackend::new().unwrap();
    let qwen_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let layer_idx = 0usize;
    let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
    let recurrent_state_stride = qwen_kv.recurrent_state_len();

    let outcome = backend
        .ops
        .sync_qwen35_slot_buffers_from_kv(&qwen_kv, layer_idx, 0);
    assert!(outcome.used_backend_zero_init);
    assert!(!outcome.used_cpu_materialization);
    assert!(!outcome.used_backend_carryover);

    backend.ops.with_qwen35_recurrent_slot_buffer(
        layer_idx,
        0,
        conv_state_stride,
        recurrent_state_stride,
        |slot_buffers| {
            let conv = unsafe { slot_buffers.conv_state.as_slice::<f32>() };
            let recurrent = unsafe { slot_buffers.recurrent_state.as_slice::<f32>() };
            assert!(conv[..conv_state_stride].iter().all(|&v| v == 0.0));
            assert!(
                recurrent[..recurrent_state_stride]
                    .iter()
                    .all(|&v| v == 0.0)
            );
            assert_eq!(
                slot_buffers.conv_synced_generation,
                Some(qwen_kv.conv_state_generation(0, layer_idx))
            );
            assert_eq!(
                slot_buffers.recurrent_synced_generation,
                Some(qwen_kv.recurrent_state_generation(0, layer_idx))
            );
        },
    );
}

#[test]
fn test_configure_for_model_updates_runtime_policy() {
    let backend = MetalBackend::new().unwrap();
    let before = backend.ops.runtime_policy();

    backend
        .configure_for_model("Qwen3-8B", "Q4_K", "qwen3")
        .unwrap();

    let after = backend.ops.runtime_policy();
    let expected = RuntimePolicy::for_model("Qwen3-8B", "Q4_K", "qwen3");

    assert_eq!(
        after.dequant_dispatch_config(),
        expected.dequant_dispatch_config()
    );
    assert_eq!(
        after.attention_dispatch_config(),
        expected.attention_dispatch_config()
    );
    assert_eq!(
        after.batch_prefill_prefers_f16_io(),
        expected.batch_prefill_prefers_f16_io()
    );
    assert_eq!(
        after.batch_prefill_prefers_pair_kernel(),
        expected.batch_prefill_prefers_pair_kernel()
    );
    assert_eq!(
        after.dequant_dispatch_config(),
        expected.dequant_dispatch_config(),
        "configure_for_model should resolve the same runtime policy as direct policy loading"
    );
    assert_eq!(
        before.attention_dispatch_config(),
        RuntimePolicy::resolved_defaults().attention_dispatch_config()
    );
    assert_eq!(
        after.gpu_kv_dtype(4096),
        expected.gpu_kv_dtype(4096),
        "configure_for_model should carry KV precision policy through the backend-local runtime policy"
    );
    assert_eq!(
        after.fused_qkv_prefill_enabled(),
        expected.fused_qkv_prefill_enabled()
    );
    assert_eq!(after.batch_simd_enabled(), expected.batch_simd_enabled());
}

#[test]
fn test_hybrid_backend_attention_prefill() {
    let backend = HybridBackend::new().unwrap();

    let n_tokens = 8;
    let n_heads = 4;
    let n_kv_heads = 2;
    let head_dim = 32;
    let q_size = n_tokens * n_heads * head_dim;
    let kv_size = n_tokens * n_kv_heads * head_dim;

    let q: Vec<f32> = (0..q_size)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
        .collect();
    let k: Vec<f32> = (0..kv_size)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.03)
        .collect();
    let v: Vec<f32> = (0..kv_size)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.08)
        .collect();

    // CPU reference
    let cpu = super::super::cpu::CpuBackend;
    let mut expected = vec![0.0f32; q_size];
    cpu.attention_prefill(
        &q,
        &k,
        &v,
        &mut expected,
        n_tokens,
        n_heads,
        n_kv_heads,
        head_dim,
    );

    // Hybrid should route to Metal
    let mut result = vec![0.0f32; q_size];
    backend.attention_prefill(
        &q,
        &k,
        &v,
        &mut result,
        n_tokens,
        n_heads,
        n_kv_heads,
        head_dim,
    );

    let diff = max_abs_diff(&result, &expected);
    assert!(
        diff < 1e-2,
        "Hybrid attention vs CPU mismatch: max_diff={diff}"
    );
}
