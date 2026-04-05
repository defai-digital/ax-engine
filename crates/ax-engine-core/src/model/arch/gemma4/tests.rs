use super::*;

#[test]
fn test_use_sliding_window_pattern_6() {
    let cfg = ModelConfig {
        architecture: "gemma4".to_string(),
        sliding_window_size: Some(1024),
        sliding_window_pattern: Some(6),
        ..test_gemma4_config()
    };
    // Layers 0-4 are local (SWA), layer 5 is global
    for layer in 0..5 {
        assert!(
            Gemma4Forward::use_sliding_window(layer, &cfg),
            "layer {layer} should be SWA"
        );
    }
    assert!(
        !Gemma4Forward::use_sliding_window(5, &cfg),
        "layer 5 should be global"
    );
    assert!(
        Gemma4Forward::use_sliding_window(6, &cfg),
        "layer 6 should be SWA"
    );
    assert!(
        !Gemma4Forward::use_sliding_window(11, &cfg),
        "layer 11 should be global"
    );
}

#[test]
fn test_layer_dims_swa_vs_global() {
    let cfg = test_gemma4_config();

    // SWA layer (layer 0)
    let (hd, nkv, q_dim, kv_dim) = Gemma4Forward::layer_dims(0, &cfg);
    assert_eq!(hd, 256);
    assert_eq!(nkv, 16);
    assert_eq!(q_dim, 32 * 256);
    assert_eq!(kv_dim, 16 * 256);

    // Global layer (layer 5)
    let (hd, nkv, q_dim, kv_dim) = Gemma4Forward::layer_dims(5, &cfg);
    assert_eq!(hd, 512);
    assert_eq!(nkv, 4);
    assert_eq!(q_dim, 32 * 512);
    assert_eq!(kv_dim, 4 * 512);
}

fn test_gemma4_config() -> ModelConfig {
    ModelConfig {
        architecture: "gemma4".to_string(),
        n_layers: 60,
        n_heads: 32,
        n_kv_heads: 16, // SWA (primary for KV cache)
        embedding_dim: 5376,
        head_dim: 256, // SWA (primary for KV cache)
        intermediate_dim: 21504,
        context_length: 4096,
        vocab_size: 262144,
        rms_norm_eps: 1e-6,
        rope_freq_base: 1000000.0,
        has_qkv_bias: false,
        sliding_window_size: Some(1024),
        sliding_window_pattern: Some(6),
        gate_activation: crate::model::config::GateActivation::GELU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: crate::model::config::RopeScaling::None,
        embed_scale: true,
        rope_freq_base_local: Some(10000.0),
        n_expert: None,
        n_expert_used: None,
        expert_intermediate_dim: None,
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        gemma4_head_dim_swa: Some(256),
        gemma4_head_dim_global: Some(512),
        gemma4_n_kv_heads_swa: Some(16),
        gemma4_n_kv_heads_global: Some(4),
        gemma4_rope_dim_swa: Some(256),
        gemma4_rope_dim_global: Some(512),
        final_logit_softcapping: Some(30.0),
    }
}
