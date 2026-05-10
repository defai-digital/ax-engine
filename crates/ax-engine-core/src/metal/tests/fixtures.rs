use std::fs;
use std::mem::size_of;
use std::path::{Path, PathBuf};

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
        model_family: "qwen3_dense".to_string(),
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
        kv_shared_source_layers: Default::default(),
        final_logit_softcapping: None,
        hidden_states_scale: None,
        moe_norm_topk_prob: false,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: None,
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        mla_attention: Default::default(),
        moe: crate::model::NativeMoeConfig::default(),
        glm_router: Default::default(),
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

pub(super) fn native_model_tensor_with_file(
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

pub(super) fn write_f32_tensor_file(root_dir: &Path, file_name: &str, values: &[f32]) {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(root_dir.join(file_name), bytes).expect("tensor bytes should write");
}

#[cfg(target_os = "macos")]
pub(super) fn write_projection_native_model_fixture() -> PathBuf {
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
        source_quantization: None,
        runtime_status: crate::model::NativeRuntimeStatus::default(),
        layer_count: 1,
        hidden_size: 8,
        intermediate_size: 0,
        attention_head_count: 2,
        attention_head_dim: 4,
        kv_head_count: 2,
        vocab_size: 5,
        tie_word_embeddings: false,
        rope_theta: None,
        rope_theta_swa: None,
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
        kv_shared_source_layers: Default::default(),
        final_logit_softcapping: None,
        hidden_states_scale: None,
        moe_norm_topk_prob: false,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: None,
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        mla_attention: Default::default(),
        moe: crate::model::NativeMoeConfig::default(),
        glm_router: Default::default(),
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
pub(super) fn write_projection_qk_norm_native_model_fixture() -> PathBuf {
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
pub(super) fn write_projection_moe_native_model_fixture() -> PathBuf {
    let root_dir = write_projection_native_model_fixture();
    let router = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let packed_gate_up = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let down_experts = vec![
        1.0, //
        0.0, //
        0.0, //
        0.0, //
        0.0, //
        0.0, //
        0.0, //
        0.0, //
        0.0, //
        1.0, //
        0.0, //
        0.0, //
        0.0, //
        0.0, //
        0.0, //
        0.0,
    ];
    write_f32_tensor_file(&root_dir, "ffn_gate_inp.bin", &router);
    write_f32_tensor_file(&root_dir, "ffn_gate_up_exps.bin", &packed_gate_up);
    write_f32_tensor_file(&root_dir, "ffn_down_exps.bin", &down_experts);

    let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("projection manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("projection manifest should deserialize");
    manifest.moe = crate::model::NativeMoeConfig {
        expert_count: Some(2),
        experts_per_token: Some(1),
        expert_intermediate_size: Some(1),
    };
    manifest.tensors.extend([
        native_model_tensor_with_file(
            "model.layers.0.mlp.router.weight",
            NativeTensorRole::FfnGateInp,
            Some(0),
            &[2, 8],
            "ffn_gate_inp.bin",
            (router.len() * size_of::<f32>()) as u64,
        ),
        native_model_tensor_with_file(
            "model.layers.0.mlp.experts.gate_up_proj.weight",
            NativeTensorRole::FfnGateUpExpsPacked,
            Some(0),
            &[2, 2, 8],
            "ffn_gate_up_exps.bin",
            (packed_gate_up.len() * size_of::<f32>()) as u64,
        ),
        native_model_tensor_with_file(
            "model.layers.0.mlp.experts.down_proj.weight",
            NativeTensorRole::FfnDownExps,
            Some(0),
            &[2, 8, 1],
            "ffn_down_exps.bin",
            (down_experts.len() * size_of::<f32>()) as u64,
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
pub(super) fn write_projection_value_from_key_native_model_fixture(
    apply_v_norm_no_scale: bool,
) -> PathBuf {
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
pub(super) fn write_projection_custom_rope_native_model_fixture(rope_theta: u32) -> PathBuf {
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
pub(super) fn write_projection_partial_rotary_native_model_fixture(
    partial_rotary_factor: f32,
) -> PathBuf {
    let root_dir = write_projection_native_model_fixture();
    let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("projection manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("projection manifest should parse");
    manifest.partial_rotary_factor = Some(partial_rotary_factor);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
    )
    .expect("projection manifest should rewrite");

    root_dir
}

#[cfg(target_os = "macos")]
pub(super) fn write_gemma_projection_native_model_fixture() -> PathBuf {
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
pub(super) fn write_qwen35_projection_native_model_fixture() -> PathBuf {
    let root_dir = write_projection_native_model_fixture();
    let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).expect("projection manifest should read");
    let mut manifest = serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
        .expect("projection manifest should parse");
    manifest.model_family = "qwen35_dense".to_string();
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
    )
    .expect("projection manifest should rewrite");

    root_dir
}

#[cfg(target_os = "macos")]
pub(super) fn write_gemma_projection_custom_rope_native_model_fixture(rope_theta: u32) -> PathBuf {
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
pub(super) fn write_gemma_projection_attention_config_native_model_fixture(
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
pub(super) fn write_grouped_projection_native_model_fixture() -> PathBuf {
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
        source_quantization: None,
        runtime_status: crate::model::NativeRuntimeStatus::default(),
        layer_count: 1,
        hidden_size: 8,
        intermediate_size: 0,
        attention_head_count: 4,
        attention_head_dim: 2,
        kv_head_count: 2,
        vocab_size: 5,
        tie_word_embeddings: false,
        rope_theta: None,
        rope_theta_swa: None,
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
        kv_shared_source_layers: Default::default(),
        final_logit_softcapping: None,
        hidden_states_scale: None,
        moe_norm_topk_prob: false,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: None,
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        mla_attention: Default::default(),
        moe: crate::model::NativeMoeConfig::default(),
        glm_router: Default::default(),
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
pub(super) fn write_multilayer_projection_native_model_fixture() -> PathBuf {
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
pub(super) fn tail_projector_matrix(
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
pub(super) fn write_wide_projection_native_model_fixture() -> PathBuf {
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
        source_quantization: None,
        runtime_status: crate::model::NativeRuntimeStatus::default(),
        layer_count: 1,
        hidden_size: hidden_size as u32,
        intermediate_size: 0,
        attention_head_count: 2,
        attention_head_dim: 4,
        kv_head_count: 2,
        vocab_size: vocab_size as u32,
        tie_word_embeddings: false,
        rope_theta: None,
        rope_theta_swa: None,
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
        kv_shared_source_layers: Default::default(),
        final_logit_softcapping: None,
        hidden_states_scale: None,
        moe_norm_topk_prob: false,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: None,
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        mla_attention: Default::default(),
        moe: crate::model::NativeMoeConfig::default(),
        glm_router: Default::default(),
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
pub(super) fn write_wide_direct_decode_native_model_fixture() -> PathBuf {
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
        source_quantization: None,
        runtime_status: crate::model::NativeRuntimeStatus::default(),
        layer_count: 1,
        hidden_size: hidden_size as u32,
        intermediate_size: 0,
        attention_head_count: 2,
        attention_head_dim: 4,
        kv_head_count: 2,
        vocab_size: vocab_size as u32,
        tie_word_embeddings: false,
        rope_theta: None,
        rope_theta_swa: None,
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
        kv_shared_source_layers: Default::default(),
        final_logit_softcapping: None,
        hidden_states_scale: None,
        moe_norm_topk_prob: false,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: None,
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        mla_attention: Default::default(),
        moe: crate::model::NativeMoeConfig::default(),
        glm_router: Default::default(),
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
pub(super) enum DirectDecodeFixtureGateUpLayout {
    Split,
    Packed,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DirectDecodeFixtureVariant {
    ProjectionOnly,
    FfnContinuation,
}

#[cfg(target_os = "macos")]
pub(super) fn write_direct_decode_native_model_fixture_with_variant(
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
        source_quantization: None,
        runtime_status: crate::model::NativeRuntimeStatus::default(),
        layer_count: 1,
        hidden_size: 8,
        intermediate_size: 0,
        attention_head_count: 2,
        attention_head_dim: 4,
        kv_head_count: 2,
        vocab_size: 5,
        tie_word_embeddings,
        rope_theta: None,
        rope_theta_swa: None,
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
        kv_shared_source_layers: Default::default(),
        final_logit_softcapping: None,
        hidden_states_scale: None,
        moe_norm_topk_prob: false,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: None,
        linear_attention: crate::model::NativeLinearAttentionConfig::default(),
        mla_attention: Default::default(),
        moe: crate::model::NativeMoeConfig::default(),
        glm_router: Default::default(),
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
pub(super) fn write_direct_decode_native_model_fixture(tie_word_embeddings: bool) -> PathBuf {
    write_direct_decode_native_model_fixture_with_variant(
        tie_word_embeddings,
        DirectDecodeFixtureGateUpLayout::Split,
        DirectDecodeFixtureVariant::ProjectionOnly,
    )
}

#[cfg(target_os = "macos")]
pub(super) fn write_gemma_direct_decode_native_model_fixture(tie_word_embeddings: bool) -> PathBuf {
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
pub(super) fn write_ffn_decode_native_model_fixture(
    gate_up_layout: DirectDecodeFixtureGateUpLayout,
) -> PathBuf {
    write_direct_decode_native_model_fixture_with_variant(
        false,
        gate_up_layout,
        DirectDecodeFixtureVariant::FfnContinuation,
    )
}

#[cfg(target_os = "macos")]
pub(super) fn write_multilayer_direct_decode_native_model_fixture() -> PathBuf {
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
