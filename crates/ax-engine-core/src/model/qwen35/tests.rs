use super::*;
use crate::backend::cpu::CpuBackend;
use crate::backend::metal::MetalBackend;
use crate::gguf::MetadataValue;
use crate::gguf::header::GgufHeader;
use crate::gguf::mmap::MappedModel;
use crate::gguf::tensor::GgmlType;
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

fn env_var_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct EnvVarRestore {
    key: &'static str,
    previous: Option<OsString>,
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        match &self.previous {
            Some(prev) => unsafe { std::env::set_var(self.key, prev) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

fn with_env_var<T>(key: &'static str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
    let _lock = env_var_lock()
        .lock()
        .expect("qwen35 test env mutex should not be poisoned");
    let _restore = EnvVarRestore {
        key,
        previous: std::env::var_os(key),
    };
    match value {
        Some(v) => unsafe { std::env::set_var(key, v) },
        None => unsafe { std::env::remove_var(key) },
    }
    f()
}

fn make_header(kv: Vec<(&str, MetadataValue)>) -> GgufHeader {
    let mut metadata = HashMap::new();
    for (k, v) in kv {
        metadata.insert(k.to_string(), v);
    }
    GgufHeader {
        version: 3,
        tensor_count: 0,
        metadata,
    }
}

fn align_to(offset: usize, alignment: usize) -> usize {
    offset.div_ceil(alignment) * alignment
}

fn push_string_metadata(buf: &mut Vec<u8>, key: &str, value: &str) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn push_u32_metadata(buf: &mut Vec<u8>, key: &str, value: u32) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_tensor_info(buf: &mut Vec<u8>, name: &str, shape: &[u64], dtype: GgmlType, offset: u64) {
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &dim in shape {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    buf.extend_from_slice(&(dtype as u32).to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn quantize_q8_0_rows(values: &[f32], row_width: usize) -> Vec<u8> {
    assert!(
        row_width.is_multiple_of(32),
        "Q8_0 row width must be a multiple of 32"
    );
    assert_eq!(
        values.len() % row_width,
        0,
        "Q8_0 input length must be divisible by row width"
    );

    let mut bytes = Vec::with_capacity(values.len() / 32 * 34);
    for row in values.chunks_exact(row_width) {
        for block in row.chunks_exact(32) {
            let max_abs = block
                .iter()
                .fold(0.0f32, |acc, &value| acc.max(value.abs()));
            let scale = if max_abs == 0.0 {
                1.0
            } else {
                (max_abs / 127.0).max(1.0)
            };
            bytes.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
            for &value in block {
                let quant = (value / scale).round().clamp(-127.0, 127.0) as i8;
                bytes.push(quant as u8);
            }
        }
    }
    bytes
}

fn build_qwen35_logits_test_gguf_with_dtype(
    output_norm: &[f32],
    output_weight_bytes: &[u8],
    output_weight_dtype: GgmlType,
    dim: usize,
    vocab_size: usize,
) -> Vec<u8> {
    let alignment = 32usize;
    let output_norm_bytes = f32_bytes(output_norm);
    let output_weight_offset = align_to(output_norm_bytes.len(), alignment);

    let mut buf = Vec::new();
    buf.extend_from_slice(&crate::gguf::GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&crate::gguf::GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    push_string_metadata(&mut buf, "general.architecture", "qwen35");
    push_u32_metadata(&mut buf, "general.alignment", alignment as u32);
    push_tensor_info(
        &mut buf,
        "output_norm.weight",
        &[dim as u64],
        GgmlType::F32,
        0,
    );
    push_tensor_info(
        &mut buf,
        "output.weight",
        &[dim as u64, vocab_size as u64],
        output_weight_dtype,
        output_weight_offset as u64,
    );
    let data_start = align_to(buf.len(), alignment);
    buf.resize(data_start, 0);
    buf.extend_from_slice(&output_norm_bytes);
    buf.resize(data_start + output_weight_offset, 0);
    buf.extend_from_slice(output_weight_bytes);
    buf
}

fn build_qwen35_logits_test_gguf(
    output_norm: &[f32],
    output_weight: &[f32],
    dim: usize,
    vocab_size: usize,
) -> Vec<u8> {
    let output_weight_bytes = f32_bytes(output_weight);
    build_qwen35_logits_test_gguf_with_dtype(
        output_norm,
        &output_weight_bytes,
        GgmlType::F32,
        dim,
        vocab_size,
    )
}

#[test]
fn test_qwen35_prefill_recurrent_state_mode_defaults_to_auto() {
    with_env_var("AX_QWEN35_PREFILL_RECURRENT_STATE_MODE", None, || {
        assert_eq!(
            Qwen35Forward::qwen35_prefill_recurrent_state_mode(),
            Qwen35PrefillRecurrentStateMode::Auto
        );
    });
}

#[test]
fn test_qwen35_prefill_recurrent_state_mode_parses_slot_buffer() {
    with_env_var(
        "AX_QWEN35_PREFILL_RECURRENT_STATE_MODE",
        Some("slot_buffer"),
        || {
            assert_eq!(
                Qwen35Forward::qwen35_prefill_recurrent_state_mode(),
                Qwen35PrefillRecurrentStateMode::SlotBuffer
            );
        },
    );
}

#[test]
fn test_qwen35_prefill_recurrent_state_mode_parses_backend_owned() {
    with_env_var(
        "AX_QWEN35_PREFILL_RECURRENT_STATE_MODE",
        Some("backend_owned"),
        || {
            assert_eq!(
                Qwen35Forward::qwen35_prefill_recurrent_state_mode(),
                Qwen35PrefillRecurrentStateMode::BackendOwned
            );
        },
    );
}

#[test]
fn test_qwen35_prefill_recurrent_state_mode_auto_is_prompt_aware() {
    with_env_var("AX_QWEN35_PREFILL_RECURRENT_STATE_MODE", None, || {
        assert_eq!(
            Qwen35Forward::qwen35_prefill_recurrent_state_mode_for_tokens(32),
            Qwen35PrefillRecurrentStateMode::BackendOwned
        );
        assert_eq!(
            Qwen35Forward::qwen35_prefill_recurrent_state_mode_for_tokens(64),
            Qwen35PrefillRecurrentStateMode::BackendOwned
        );
        assert_eq!(
            Qwen35Forward::qwen35_prefill_recurrent_state_mode_for_tokens(96),
            Qwen35PrefillRecurrentStateMode::BackendOwned
        );
        assert_eq!(
            Qwen35Forward::qwen35_prefill_recurrent_state_mode_for_tokens(128),
            Qwen35PrefillRecurrentStateMode::BackendOwned
        );
    });
}

#[test]
fn test_qwen35_prefill_recurrent_state_mode_auto_preserves_backend_owned_owner() {
    with_env_var("AX_QWEN35_PREFILL_RECURRENT_STATE_MODE", None, || {
        assert_eq!(
            Qwen35Forward::resolve_qwen35_prefill_recurrent_state_mode(
                64,
                crate::kv::Qwen35LayerStateOwner::BackendOwned,
            ),
            Qwen35PrefillRecurrentStateMode::BackendOwned
        );
        assert_eq!(
            Qwen35Forward::resolve_qwen35_prefill_recurrent_state_mode(
                64,
                crate::kv::Qwen35LayerStateOwner::Split,
            ),
            Qwen35PrefillRecurrentStateMode::BackendOwned
        );
    });
}

#[test]
fn test_qwen35_prefill_force_backend_state_batch_defaults_off() {
    with_env_var("AX_QWEN35_PREFILL_FORCE_BACKEND_STATE_BATCH", None, || {
        assert!(!Qwen35Forward::qwen35_prefill_force_backend_state_batch());
    });
}

#[test]
fn test_qwen35_prefill_force_backend_state_batch_parses_on() {
    with_env_var(
        "AX_QWEN35_PREFILL_FORCE_BACKEND_STATE_BATCH",
        Some("1"),
        || {
            assert!(Qwen35Forward::qwen35_prefill_force_backend_state_batch());
        },
    );
}

#[test]
fn test_qwen35_prefill_backend_state_batch_auto_prefers_backend_owned_layers() {
    with_env_var("AX_QWEN35_PREFILL_FORCE_BACKEND_STATE_BATCH", None, || {
        assert!(
            Qwen35Forward::qwen35_prefill_backend_state_batch_for_tokens(
                32,
                crate::kv::Qwen35LayerStateOwner::BackendOwned,
            )
        );
        assert!(
            Qwen35Forward::qwen35_prefill_backend_state_batch_for_tokens(
                64,
                crate::kv::Qwen35LayerStateOwner::Split,
            )
        );
        assert!(
            Qwen35Forward::qwen35_prefill_backend_state_batch_for_tokens(
                128,
                crate::kv::Qwen35LayerStateOwner::BackendOwned,
            )
        );
        assert!(
            Qwen35Forward::qwen35_prefill_backend_state_batch_for_tokens(
                256,
                crate::kv::Qwen35LayerStateOwner::Split,
            )
        );
        assert!(
            !Qwen35Forward::qwen35_prefill_backend_state_batch_for_tokens(
                64,
                crate::kv::Qwen35LayerStateOwner::CpuMaterialized,
            )
        );
    });
}

#[test]
fn test_qwen35_fused_recurrent_gpu_candidate_enabled_for_backend_state_batch() {
    assert!(Qwen35Forward::qwen35_fused_recurrent_gpu_candidate(
        1,
        true,
        &[2, 3],
        true,
        true,
    ));
}

#[test]
fn test_qwen35_fused_recurrent_gpu_candidate_enabled_when_eligible() {
    assert!(Qwen35Forward::qwen35_fused_recurrent_gpu_candidate(
        1,
        true,
        &[2, 3],
        true,
        true,
    ));
}

#[test]
fn test_qwen35_prefill_alpha_beta_storage_mode_defaults_to_auto() {
    with_env_var("AX_QWEN35_PREFILL_ALPHA_BETA_STORAGE_MODE", None, || {
        assert_eq!(
            Qwen35Forward::qwen35_prefill_alpha_beta_storage_mode(),
            Qwen35PrefillAlphaBetaStorageMode::Auto
        );
    });
}

#[test]
fn test_qwen35_prefill_alpha_beta_storage_mode_parses_f16() {
    with_env_var(
        "AX_QWEN35_PREFILL_ALPHA_BETA_STORAGE_MODE",
        Some("f16"),
        || {
            assert_eq!(
                Qwen35Forward::qwen35_prefill_alpha_beta_storage_mode(),
                Qwen35PrefillAlphaBetaStorageMode::F16
            );
        },
    );
}

#[test]
fn test_prepare_qwen35_handoff_alpha_beta_uses_projected_inputs() {
    let mut alpha = vec![0.0f32; 4];
    let mut beta = vec![0.0f32; 4];
    let alpha_src = vec![0.25f32, -0.5, 1.0, -1.5];
    let beta_src = vec![-2.0f32, -0.25, 0.5, 2.0];
    let dt_bias = vec![0.1f32, -0.2];
    let ssm_a = vec![0.5f32, 1.5];

    Qwen35Forward::prepare_qwen35_handoff_alpha_beta(
        &mut alpha, &mut beta, &alpha_src, &beta_src, &dt_bias, &ssm_a,
    );

    let mut expected_alpha = alpha_src.clone();
    let mut expected_beta = beta_src.clone();
    crate::compute::gdn::prepare_alpha_beta(
        &mut expected_alpha,
        &mut expected_beta,
        &dt_bias,
        &ssm_a,
    );

    assert_eq!(alpha, expected_alpha);
    assert_eq!(beta, expected_beta);
    assert_ne!(alpha, vec![0.0; 4], "alpha should reflect projected input");
    assert_ne!(beta, vec![0.0; 4], "beta should reflect projected input");
}

fn write_test_gguf_to_temp(data: &[u8]) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "ax-qwen35-logits-{}-{}.gguf",
        std::process::id(),
        unique
    ));
    std::fs::write(&path, data).unwrap();
    path
}

#[test]
fn test_qwen35_layer_pattern() {
    let header = make_header(vec![
        (
            "general.architecture",
            MetadataValue::String("qwen35".into()),
        ),
        ("qwen35.block_count", MetadataValue::Uint32(8)),
        ("qwen35.attention.head_count", MetadataValue::Uint32(16)),
        ("qwen35.attention.head_count_kv", MetadataValue::Uint32(8)),
        ("qwen35.embedding_length", MetadataValue::Uint32(2048)),
        ("qwen35.attention.key_length", MetadataValue::Uint32(128)),
        ("qwen35.feed_forward_length", MetadataValue::Uint32(8192)),
        ("qwen35.context_length", MetadataValue::Uint32(4096)),
        ("qwen35.full_attention_interval", MetadataValue::Uint32(4)),
        ("qwen35.ssm.conv_kernel", MetadataValue::Uint32(4)),
        ("qwen35.ssm.inner_size", MetadataValue::Uint32(1024)),
        ("qwen35.ssm.state_size", MetadataValue::Uint32(128)),
        ("qwen35.ssm.time_step_rank", MetadataValue::Uint32(8)),
        ("qwen35.ssm.group_count", MetadataValue::Uint32(2)),
    ]);
    let cfg = ModelConfig::from_gguf(&header).unwrap();
    assert!(cfg.qwen35_is_recurrent_layer(0));
    assert!(cfg.qwen35_is_recurrent_layer(1));
    assert!(cfg.qwen35_is_recurrent_layer(2));
    assert!(!cfg.qwen35_is_recurrent_layer(3));
}

#[test]
fn test_qwen35_validate_requires_recurrent_dims() {
    let fwd = Qwen35Forward;
    let cfg = ModelConfig {
        architecture: "qwen35".into(),
        n_layers: 4,
        n_heads: 16,
        n_kv_heads: 8,
        embedding_dim: 2048,
        head_dim: 128,
        intermediate_dim: 8192,
        context_length: 4096,
        vocab_size: 1000,
        rms_norm_eps: 1e-6,
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
        n_expert: None,
        n_expert_used: None,
        expert_intermediate_dim: None,
        qwen35_full_attention_interval: Some(4),
        qwen35_ssm_conv_kernel: Some(4),
        qwen35_ssm_inner_size: Some(1024),
        qwen35_ssm_state_size: Some(128),
        qwen35_ssm_time_step_rank: Some(8),
        qwen35_ssm_group_count: Some(2),
    };
    fwd.validate_config(&cfg).unwrap();
}

#[test]
fn test_qwen35_rope_position_honors_linear_scaling() {
    let cfg = ModelConfig {
        architecture: "qwen35".into(),
        n_layers: 4,
        n_heads: 16,
        n_kv_heads: 8,
        embedding_dim: 2048,
        head_dim: 128,
        intermediate_dim: 8192,
        context_length: 4096,
        vocab_size: 1000,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: crate::model::config::GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::Linear(8.0),
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: None,
        n_expert_used: None,
        expert_intermediate_dim: None,
        qwen35_full_attention_interval: Some(4),
        qwen35_ssm_conv_kernel: Some(4),
        qwen35_ssm_inner_size: Some(1024),
        qwen35_ssm_state_size: Some(128),
        qwen35_ssm_time_step_rank: Some(8),
        qwen35_ssm_group_count: Some(2),
    };

    assert!((Qwen35Forward::rope_position(&cfg, 16) - 2.0).abs() < 1e-6);
}

#[test]
fn test_qwen35_rope_position_uses_current_yarn_fallback_scaling() {
    let cfg = ModelConfig {
        architecture: "qwen35".into(),
        n_layers: 4,
        n_heads: 16,
        n_kv_heads: 8,
        embedding_dim: 2048,
        head_dim: 128,
        intermediate_dim: 8192,
        context_length: 4096,
        vocab_size: 1000,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: crate::model::config::GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::Yarn {
            factor: 8.0,
            ext_factor: 1.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            orig_ctx_len: 8192,
        },
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: None,
        n_expert_used: None,
        expert_intermediate_dim: None,
        qwen35_full_attention_interval: Some(4),
        qwen35_ssm_conv_kernel: Some(4),
        qwen35_ssm_inner_size: Some(1024),
        qwen35_ssm_state_size: Some(128),
        qwen35_ssm_time_step_rank: Some(8),
        qwen35_ssm_group_count: Some(2),
    };

    assert!((Qwen35Forward::rope_position(&cfg, 16) - 2.0).abs() < 1e-6);
}

#[test]
fn test_qwen35_apply_rope_batch_uses_absolute_positions() {
    let cfg = ModelConfig {
        architecture: "qwen35".into(),
        n_layers: 4,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: 8,
        head_dim: 4,
        intermediate_dim: 16,
        context_length: 4096,
        vocab_size: 1000,
        rms_norm_eps: 1e-6,
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
        n_expert: None,
        n_expert_used: None,
        expert_intermediate_dim: None,
        qwen35_full_attention_interval: Some(4),
        qwen35_ssm_conv_kernel: Some(4),
        qwen35_ssm_inner_size: Some(16),
        qwen35_ssm_state_size: Some(4),
        qwen35_ssm_time_step_rank: Some(4),
        qwen35_ssm_group_count: Some(1),
    };
    let n_tokens = 2usize;
    let start_position = 7usize;
    let q_dim = 4usize;
    let kv_dim = 4usize;
    let n_heads = 1usize;
    let n_kv_heads = 1usize;
    let head_dim = 4usize;
    let mut actual_q = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5];
    let mut actual_k = vec![4.0f32, 3.0, 2.0, 1.0, 3.5, 2.5, 1.5, 0.5];
    let mut expected_q = actual_q.clone();
    let mut expected_k = actual_k.clone();

    for token_idx in 0..n_tokens {
        let q_start = token_idx * q_dim;
        let k_start = token_idx * kv_dim;
        rope::apply_rope_multi_head_scaled(
            &mut expected_q[q_start..q_start + q_dim],
            &mut expected_k[k_start..k_start + kv_dim],
            n_heads,
            n_kv_heads,
            head_dim,
            Qwen35Forward::rope_position(&cfg, start_position + token_idx),
            cfg.rope_freq_base,
        );
    }

    Qwen35Forward::apply_rope_batch(
        &cfg,
        &mut actual_q,
        &mut actual_k,
        n_tokens,
        start_position,
        q_dim,
        kv_dim,
        n_heads,
        n_kv_heads,
        head_dim,
    );

    for (actual, expected) in actual_q.iter().zip(expected_q.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (actual, expected) in actual_k.iter().zip(expected_k.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn test_qwen35_prepare_full_attention_qk_batch_matches_staged_path() {
    let cfg = ModelConfig {
        architecture: "qwen35".into(),
        n_layers: 4,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: 8,
        head_dim: 4,
        intermediate_dim: 16,
        context_length: 4096,
        vocab_size: 1000,
        rms_norm_eps: 1e-6,
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
        n_expert: None,
        n_expert_used: None,
        expert_intermediate_dim: None,
        qwen35_full_attention_interval: Some(4),
        qwen35_ssm_conv_kernel: Some(4),
        qwen35_ssm_inner_size: Some(16),
        qwen35_ssm_state_size: Some(4),
        qwen35_ssm_time_step_rank: Some(4),
        qwen35_ssm_group_count: Some(1),
    };
    let q_gate_batch = vec![
        1.0f32, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4, 1.5, 2.5, 3.5, 4.5, 0.5, 0.6, 0.7, 0.8,
    ];
    let mut actual_q = vec![0.0f32; 8];
    let mut actual_k = vec![4.0f32, 3.0, 2.0, 1.0, 3.5, 2.5, 1.5, 0.5];
    let mut expected_q = actual_q.clone();
    let mut expected_k = actual_k.clone();
    let norm_weights = Qwen35AttentionNormWeights {
        q: &[1.0f32, 1.1, 1.2, 1.3],
        k: &[0.9f32, 1.0, 1.1, 1.2],
    };

    Qwen35Forward::extract_q_from_q_gate_batch(&q_gate_batch, &mut expected_q, 2, 4);
    Qwen35Forward::apply_attention_qk_norm_batch(
        &mut expected_q,
        &mut expected_k,
        2,
        4,
        4,
        1,
        1,
        4,
        norm_weights,
        cfg.rms_norm_eps,
    );
    Qwen35Forward::apply_rope_batch(&cfg, &mut expected_q, &mut expected_k, 2, 7, 4, 4, 1, 1, 4);

    Qwen35Forward::prepare_full_attention_qk_batch(
        &cfg,
        &q_gate_batch,
        &mut actual_q,
        &mut actual_k,
        2,
        7,
        4,
        4,
        1,
        1,
        4,
        Some(norm_weights),
        cfg.rms_norm_eps,
    );

    for (actual, expected) in actual_q.iter().zip(expected_q.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (actual, expected) in actual_k.iter().zip(expected_k.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn test_qwen35_full_attention_input_plan_fuses_matching_dtypes() {
    let wq = [1u8, 2, 3, 4];
    let wk = [5u8, 6];
    let wv = [7u8, 8, 9];
    let plan = Qwen35Forward::maybe_fused_full_attention_input_plan([
        (&wq, GgmlType::Q4K, 8),
        (&wk, GgmlType::Q4K, 4),
        (&wv, GgmlType::Q4K, 4),
    ]);

    match plan {
        Qwen35FullAttentionInputPlan::Fused { raw, dtype, rows } => {
            assert_eq!(dtype, GgmlType::Q4K);
            assert_eq!(rows, 16);
            assert_eq!(raw.as_ref(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        }
        Qwen35FullAttentionInputPlan::Split(_) => {
            panic!("expected fused full-attention input plan")
        }
    }
}

#[test]
fn test_qwen35_split_full_attention_fused_output_batch_layout() {
    let fused = vec![
        1.0f32, 2.0, 3.0, 4.0, 10.0, 11.0, 20.0, 21.0, 5.0, 6.0, 7.0, 8.0, 12.0, 13.0, 22.0, 23.0,
    ];
    let mut q_gate = vec![0.0f32; 8];
    let mut k = vec![0.0f32; 4];
    let mut v = vec![0.0f32; 4];

    Qwen35Forward::split_full_attention_fused_output_batch(&fused, &mut q_gate, &mut k, &mut v, 2);

    assert_eq!(q_gate, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(k, vec![10.0f32, 11.0, 12.0, 13.0]);
    assert_eq!(v, vec![20.0f32, 21.0, 22.0, 23.0]);
}

#[test]
fn test_qwen35_validate_rejects_incompatible_head_expansion() {
    let fwd = Qwen35Forward;
    let cfg = ModelConfig {
        architecture: "qwen35".into(),
        n_layers: 4,
        n_heads: 16,
        n_kv_heads: 8,
        embedding_dim: 2048,
        head_dim: 128,
        intermediate_dim: 8192,
        context_length: 4096,
        vocab_size: 1000,
        rms_norm_eps: 1e-6,
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
        n_expert: None,
        n_expert_used: None,
        expert_intermediate_dim: None,
        qwen35_full_attention_interval: Some(4),
        qwen35_ssm_conv_kernel: Some(4),
        qwen35_ssm_inner_size: Some(768),
        qwen35_ssm_state_size: Some(128),
        qwen35_ssm_time_step_rank: Some(6),
        qwen35_ssm_group_count: Some(4),
    };

    let err = fwd.validate_config(&cfg).unwrap_err();
    assert!(err.to_string().contains("multiple of group_count"));
}

#[test]
fn test_qwen35_write_all_batch_logits_matches_per_token_reference() {
    let dim = 4usize;
    let vocab_size = 3usize;
    let n_tokens = 2usize;
    let rms_norm_eps = 1e-6f32;
    let output_norm = [1.0f32, 0.5, 1.5, 0.75];
    let output_weight = [
        0.25f32, -0.5, 1.0, 0.75, -1.0, 0.5, 0.25, -0.75, 0.1, 0.2, -0.3, 0.4,
    ];
    let hidden = vec![1.0f32, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.5];
    let gguf = build_qwen35_logits_test_gguf(&output_norm, &output_weight, dim, vocab_size);
    let path = write_test_gguf_to_temp(&gguf);

    {
        let model = MappedModel::open(&path).unwrap();
        let weights = WeightStore::new(&model);
        let backend = CpuBackend;

        let mut actual = Vec::new();
        Qwen35Forward::write_all_batch_logits(
            &backend,
            &hidden,
            n_tokens,
            dim,
            vocab_size,
            rms_norm_eps,
            &weights,
            &mut actual,
        )
        .unwrap();

        let mut expected = vec![0.0f32; n_tokens * vocab_size];
        for token_idx in 0..n_tokens {
            let hidden_start = token_idx * dim;
            let logits_start = token_idx * vocab_size;
            let mut token_hidden = hidden[hidden_start..hidden_start + dim].to_vec();
            Qwen35Forward::write_single_logits(
                &backend,
                &mut token_hidden,
                dim,
                vocab_size,
                rms_norm_eps,
                &weights,
                &mut expected[logits_start..logits_start + vocab_size],
            )
            .unwrap();
        }

        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "logit {idx} mismatch: actual={actual}, expected={expected}"
            );
        }
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_qwen35_cpu_batch_fallback_scratch_reuses_capacity() {
    let mut scratch = Qwen35CpuBatchFallbackScratch::default();
    scratch.ensure_lengths(8, 64, 128, 64, 32, 2, 192, 96, 6);
    let hidden_capacity = scratch.hidden.capacity();
    let rec_qkv_capacity = scratch.rec_qkv_batch.capacity();
    let final_hidden_capacity = scratch.final_hidden.capacity();

    scratch.ensure_lengths(4, 64, 128, 64, 32, 2, 192, 96, 6);

    assert_eq!(scratch.hidden.len(), 4 * 64);
    assert_eq!(scratch.rec_qkv_batch.len(), 4 * 2 * 192);
    assert_eq!(scratch.final_hidden.len(), 4 * 64);
    assert!(scratch.hidden.capacity() >= hidden_capacity);
    assert!(scratch.rec_qkv_batch.capacity() >= rec_qkv_capacity);
    assert!(scratch.final_hidden.capacity() >= final_hidden_capacity);
}

#[test]
fn test_qwen35_write_all_batch_logits_gpu_q8_0_matches_cpu_reference() {
    let Ok(backend) = MetalBackend::new() else {
        return;
    };

    let dim = 32usize;
    let vocab_size = 2usize;
    let n_tokens = 3usize;
    let rms_norm_eps = 1e-6f32;
    let output_norm = vec![1.0f32; dim];
    let output_weight_f32 = vec![
        1.0, -2.0, 0.0, 3.0, -1.0, 2.0, -3.0, 1.0, 0.0, 1.0, -1.0, 2.0, 3.0, -2.0, 1.0, 0.0, -1.0,
        2.0, 1.0, -3.0, 2.0, 0.0, 1.0, -2.0, 3.0, 1.0, -1.0, 2.0, 0.0, -2.0, 1.0, 3.0, -1.0, 0.0,
        2.0, -2.0, 1.0, 3.0, -1.0, 0.0, 2.0, -3.0, 1.0, 1.0, -2.0, 0.0, 3.0, -1.0, 2.0, -1.0, 0.0,
        1.0, -3.0, 2.0, 1.0, -2.0, 3.0, 0.0, 1.0, -1.0, 2.0, -2.0, 1.0, 0.0,
    ];
    let hidden = vec![
        1.0, -1.0, 2.0, 0.0, 3.0, -2.0, 1.0, 0.0, -1.0, 2.0, 1.0, -3.0, 2.0, 0.0, 1.0, -2.0, 3.0,
        1.0, -1.0, 2.0, 0.0, -2.0, 1.0, 3.0, -1.0, 0.0, 2.0, -3.0, 1.0, 1.0, -2.0, 0.0, -1.0, 0.5,
        2.0, -2.0, 1.0, 3.0, -1.0, 0.0, 2.0, -3.0, 1.0, 1.0, -2.0, 0.0, 3.0, -1.0, 2.0, -1.0, 0.0,
        1.0, -3.0, 2.0, 1.0, -2.0, 3.0, 0.0, 1.0, -1.0, 2.0, -2.0, 1.0, 0.0, 0.25, 1.5, -0.5, 2.0,
        -1.0, 0.0, 1.0, -2.0, 3.0, -1.0, 0.0, 2.0, -3.0, 1.0, 1.0, -2.0, 0.0, 3.0, -1.0, 2.0, -1.0,
        0.0, 1.0, -3.0, 2.0, 1.0, -2.0, 3.0, 0.0, 1.0, -1.0, 2.0, -2.0, 1.0, 0.0,
    ];
    let output_weight_q8_0 = quantize_q8_0_rows(&output_weight_f32, dim);
    let gguf = build_qwen35_logits_test_gguf_with_dtype(
        &output_norm,
        &output_weight_q8_0,
        GgmlType::Q8_0,
        dim,
        vocab_size,
    );
    let path = write_test_gguf_to_temp(&gguf);

    {
        let model = MappedModel::open(&path).unwrap();
        let weights = WeightStore::new(&model);
        let cpu_backend = CpuBackend;

        let mut actual = Vec::new();
        Qwen35Forward::write_all_batch_logits(
            &backend,
            &hidden,
            n_tokens,
            dim,
            vocab_size,
            rms_norm_eps,
            &weights,
            &mut actual,
        )
        .unwrap();

        let mut expected = Vec::new();
        Qwen35Forward::write_all_batch_logits(
            &cpu_backend,
            &hidden,
            n_tokens,
            dim,
            vocab_size,
            rms_norm_eps,
            &weights,
            &mut expected,
        )
        .unwrap();

        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 5e-2,
                "q8_0 gpu batch logit {idx} mismatch: actual={actual}, expected={expected}"
            );
        }
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_qwen35_gpu_batch_logits_enabled_env_override() {
    with_env_var("AX_QWEN35_GPU_BATCH_LOGITS", None, || {
        assert!(Qwen35Forward::gpu_batch_logits_enabled());
    });
    with_env_var("AX_QWEN35_GPU_BATCH_LOGITS", Some("0"), || {
        assert!(!Qwen35Forward::gpu_batch_logits_enabled());
    });
    with_env_var("AX_QWEN35_GPU_BATCH_LOGITS", Some("off"), || {
        assert!(!Qwen35Forward::gpu_batch_logits_enabled());
    });
    with_env_var("AX_QWEN35_GPU_BATCH_LOGITS", Some("1"), || {
        assert!(Qwen35Forward::gpu_batch_logits_enabled());
    });
}
