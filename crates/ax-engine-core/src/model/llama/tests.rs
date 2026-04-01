use super::*;

fn tiny_config() -> ModelConfig {
    ModelConfig {
        architecture: "llama".into(),
        n_layers: 1,
        n_heads: 2,
        n_kv_heads: 2,
        embedding_dim: 8,
        head_dim: 4,
        intermediate_dim: 16,
        context_length: 32,
        vocab_size: 4,
        rms_norm_eps: 1e-5,
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
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        expert_intermediate_dim: None,
    }
}

#[test]
fn test_llama_model_creation() {
    let config = tiny_config();
    let model = LlamaModel::new(config.clone()).unwrap();
    assert_eq!(model.config.n_layers, 1);
    assert_eq!(model.attn_params.n_heads, 2);
    assert_eq!(model.attn_params.n_kv_heads, 2);
    assert_eq!(model.attn_params.head_dim, 4);
}

#[test]
fn test_llama_model_with_backend() {
    let config = tiny_config();
    let backend: Box<dyn Backend> = Box::new(CpuBackend);
    let model = LlamaModel::with_backend(config.clone(), backend).unwrap();
    assert_eq!(model.config.n_layers, 1);
}

#[test]
fn test_model_kv_creation_cpu() {
    let config = tiny_config();
    let model = LlamaModel::new(config).unwrap();
    let kv = model.create_model_kv();
    assert_eq!(kv.seq_len(), 0);
    assert!(!kv.is_gpu());
}

#[test]
fn test_model_kv_creation_hybrid_cpu_decode_uses_cpu_kv() {
    let Ok(backend) = crate::backend::metal::HybridCpuDecodeBackend::new() else {
        return;
    };
    let model = LlamaModel::with_backend(tiny_config(), Box::new(backend)).unwrap();
    let kv = model.create_model_kv();
    assert!(
        !kv.is_gpu(),
        "HybridCpuDecode must allocate a CPU KV cache because decode runs on CPU"
    );
}

#[test]
fn test_profiled_forward_same_signature() {
    let mut ops = crate::metrics::OpBreakdown::new();
    assert_eq!(ops.total(), std::time::Duration::ZERO);
    ops.matmul += std::time::Duration::from_nanos(1);
    assert!(ops.total() > std::time::Duration::ZERO);
}

#[test]
fn test_attention_params_from_config() {
    let config = ModelConfig {
        architecture: "llama".into(),
        n_layers: 32,
        n_heads: 32,
        n_kv_heads: 8,
        embedding_dim: 4096,
        head_dim: 128,
        intermediate_dim: 11008,
        context_length: 4096,
        vocab_size: 32000,
        rms_norm_eps: 1e-5,
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
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        expert_intermediate_dim: None,
    };
    let model = LlamaModel::new(config).unwrap();
    assert_eq!(model.attn_params.n_heads, 32);
    assert_eq!(model.attn_params.n_kv_heads, 8);
    assert_eq!(model.attn_params.head_dim, 128);
}

#[test]
fn test_llama_forward_arch_name() {
    let fwd = LlamaForward;
    assert_eq!(fwd.arch_name(), "llama");
}

#[test]
fn test_model_selects_llama_forward() {
    let config = tiny_config();
    let model = LlamaModel::new(config).unwrap();
    assert_eq!(model.arch_name(), "llama");
}

#[test]
fn test_model_rejects_invalid_gqa_shape() {
    let mut config = tiny_config();
    config.n_kv_heads = 3;
    let err = LlamaModel::new(config)
        .err()
        .expect("config should be rejected");
    assert!(err.to_string().contains("must be a multiple of n_kv_heads"));
}

#[test]
fn test_model_rejects_odd_head_dim() {
    let mut config = tiny_config();
    config.head_dim = 3;
    let err = LlamaModel::new(config)
        .err()
        .expect("config should be rejected");
    assert!(err.to_string().contains("must be even"));
}
