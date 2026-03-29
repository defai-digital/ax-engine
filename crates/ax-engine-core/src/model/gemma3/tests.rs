use super::*;

#[test]
fn test_gemma3_forward_arch_name() {
    let fwd = Gemma3Forward;
    assert_eq!(fwd.arch_name(), "gemma3");
}

#[test]
fn test_sliding_window_detection() {
    let mut config = ModelConfig {
        architecture: "gemma3".into(),
        n_layers: 12,
        n_heads: 8,
        n_kv_heads: 4,
        embedding_dim: 2560,
        head_dim: 256,
        intermediate_dim: 10240,
        context_length: 8192,
        vocab_size: 256000,
        rms_norm_eps: 1e-6,
        rope_freq_base: 1000000.0,
        has_qkv_bias: false,
        sliding_window_size: Some(1024),
        sliding_window_pattern: Some(6),
        gate_activation: crate::model::config::GateActivation::GELU,
        tie_word_embeddings: true,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: true,
        rope_freq_base_local: Some(10000.0),
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

    // Pattern=6: layers 0-4 = local (sliding window), layer 5 = global
    assert!(Gemma3Forward::use_sliding_window(0, &config)); // local
    assert!(Gemma3Forward::use_sliding_window(1, &config)); // local
    assert!(Gemma3Forward::use_sliding_window(2, &config)); // local
    assert!(Gemma3Forward::use_sliding_window(3, &config)); // local
    assert!(Gemma3Forward::use_sliding_window(4, &config)); // local
    assert!(!Gemma3Forward::use_sliding_window(5, &config)); // global (5 % 6 == 5 == 6-1)
    assert!(Gemma3Forward::use_sliding_window(6, &config)); // local
    assert!(!Gemma3Forward::use_sliding_window(11, &config)); // global (11 % 6 == 5 == 6-1)

    // No sliding window when config doesn't specify it
    config.sliding_window_size = None;
    config.sliding_window_pattern = None;
    assert!(!Gemma3Forward::use_sliding_window(0, &config));
    assert!(!Gemma3Forward::use_sliding_window(1, &config));
}

#[test]
fn test_gpu_prefill_chunk_len() {
    let mut config = ModelConfig {
        architecture: "gemma3".into(),
        n_layers: 12,
        n_heads: 8,
        n_kv_heads: 4,
        embedding_dim: 2560,
        head_dim: 256,
        intermediate_dim: 10240,
        context_length: 8192,
        vocab_size: 256000,
        rms_norm_eps: 1e-6,
        rope_freq_base: 1000000.0,
        has_qkv_bias: false,
        sliding_window_size: Some(1024),
        sliding_window_pattern: Some(6),
        gate_activation: crate::model::config::GateActivation::GELU,
        tie_word_embeddings: true,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: true,
        rope_freq_base_local: Some(10000.0),
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

    assert_eq!(Gemma3Forward::gpu_prefill_chunk_len(&config, 512), None);
    assert_eq!(Gemma3Forward::gpu_prefill_chunk_len(&config, 1024), None);
    assert_eq!(
        Gemma3Forward::gpu_prefill_chunk_len(&config, 2048),
        Some(1024)
    );

    config.sliding_window_size = None;
    assert_eq!(Gemma3Forward::gpu_prefill_chunk_len(&config, 4096), None);
}

#[test]
fn test_per_head_rms_norm() {
    // 2 heads, head_dim=4, weights all 1.0
    let mut buf = vec![2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];

    crate::model::shared::per_head_rms_norm(&mut buf, 2, 4, &weight, 1e-6);

    // Head 0: all 2.0 → RMS = 2.0, result ≈ [1,1,1,1]
    for (i, &v) in buf[..4].iter().enumerate() {
        assert!((v - 1.0).abs() < 0.01, "head0[{i}]: {v} != 1.0");
    }
    // Head 1: all 4.0 → RMS = 4.0, result ≈ [1,1,1,1]
    for (i, &v) in buf[4..8].iter().enumerate() {
        assert!((v - 1.0).abs() < 0.01, "head1[{i}]: {v} != 1.0");
    }
}
