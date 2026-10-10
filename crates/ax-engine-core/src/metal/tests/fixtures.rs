use std::fs;
use std::path::PathBuf;

use crate::model::NativeTensorRole;

use super::unique_test_dir;

#[allow(dead_code)]
pub(super) fn write_valid_native_model_fixture() -> PathBuf {
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
        model_family: "qwen3".to_string(),
        tensor_format: crate::model::NativeTensorFormat::Safetensors,
        source_quantization: None,
        runtime_status: crate::model::NativeRuntimeStatus::default(),
        layer_count: 1,
        hidden_size: 2,
        intermediate_size: 0,
        attention_head_count: 1,
        attention_head_dim: 2,
        kv_head_count: 1,
        vocab_size: 4,
        tie_word_embeddings: false,
        rope_theta: None,
        rope_theta_swa: None,
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_low_freq_factor: None,
        rope_high_freq_factor: None,
        rope_original_context_len: None,
        rope_beta_fast: None,
        rope_beta_slow: None,
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
        global_kv_head_count: None,
        sliding_window_size: None,
        layer_types: Vec::new(),
        kv_shared_source_layers: Default::default(),
        final_logit_softcapping: None,
        final_logits_scale: None,
        attention_scale_multiplier: None,
        post_norm_eps: None,
        hidden_states_scale: None,
        moe_norm_topk_prob: false,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: None,
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        mla_attention: Default::default(),
        moe: crate::model::NativeMoeConfig::default(),
        glm_router: Default::default(),
        deepseek_v4: Default::default(),
        qwen4_exp: Default::default(),
        weight_sanitize: crate::model::WeightSanitize::None,
        think_start_token_id: None,
        think_end_token_id: None,
        diffusion: crate::model::NativeDiffusionConfig::default(),
        dropped_tensors: Default::default(),
        kv_cache_quantization: None,
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
pub(super) fn native_model_tensor(
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
