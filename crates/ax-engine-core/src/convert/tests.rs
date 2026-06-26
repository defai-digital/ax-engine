use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::{
    ConvertError, NativeTensorDataType, NativeTensorRole, compute_attention_value_from_key_layers,
    convert_hf_model_dir, match_tensor, model_family_for_type, moe_config, parse_layer_types,
    parse_rope_params, validate_glm4_moe_lite_rope_scaling, validate_qwen_rope_scaling,
    with_real_model_manifest_lock, write_manifest,
};

fn write_fake_safetensors(dir: &Path, filename: &str, tensors: &[(&str, &str, &[u64])]) {
    let mut header = BTreeMap::new();
    let mut offset = 0u64;
    for (name, dtype, shape) in tensors {
        let elem_size: u64 = match *dtype {
            "F16" | "BF16" => 2,
            "F32" | "U32" => 4,
            _ => 1,
        };
        let num_elements: u64 = shape.iter().product();
        let byte_len = num_elements * elem_size;
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + byte_len],
            }),
        );
        offset += byte_len;
    }

    let header_json = serde_json::to_vec(&header).unwrap();
    let header_size = header_json.len() as u64;
    let data = vec![0u8; offset as usize];

    let path = dir.join(filename);
    let mut file = fs::File::create(&path).unwrap();
    file.write_all(&header_size.to_le_bytes()).unwrap();
    file.write_all(&header_json).unwrap();
    file.write_all(&data).unwrap();
}

fn write_config(dir: &Path, config: serde_json::Value) {
    let path = dir.join("config.json");
    fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
}

fn unique_test_dir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("ax-convert-{label}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn converts_gemma4_assistant_q_only_external_kv_contract() {
    let dir = unique_test_dir("gemma4_assistant");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 16,
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine"
            },
            "text_config": {
                "model_type": "gemma4_assistant",
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "num_hidden_layers": 2,
                "num_kv_shared_layers": 2,
                "vocab_size": 64,
                "intermediate_size": 32,
                "hidden_size_per_layer_input": 0,
                "vocab_size_per_layer_input": 0,
                "sliding_window_pattern": 2
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "BF16", &[64, 8]),
            ("model.norm.weight", "BF16", &[8]),
            ("lm_head.weight", "BF16", &[64, 8]),
            ("pre_projection.weight", "U32", &[8, 4]),
            ("post_projection.weight", "U32", &[16, 1]),
            ("model.layers.0.input_layernorm.weight", "BF16", &[8]),
            ("model.layers.0.self_attn.q_proj.weight", "BF16", &[8, 8]),
            ("model.layers.0.self_attn.o_proj.weight", "BF16", &[8, 8]),
            ("model.layers.0.self_attn.q_norm.weight", "BF16", &[4]),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "BF16",
                &[8],
            ),
            ("model.layers.0.mlp.gate_proj.weight", "BF16", &[32, 8]),
            ("model.layers.0.mlp.up_proj.weight", "BF16", &[32, 8]),
            ("model.layers.0.mlp.down_proj.weight", "BF16", &[8, 32]),
            ("model.layers.1.input_layernorm.weight", "BF16", &[8]),
            ("model.layers.1.self_attn.q_proj.weight", "BF16", &[8, 8]),
            ("model.layers.1.self_attn.o_proj.weight", "BF16", &[8, 8]),
            ("model.layers.1.self_attn.q_norm.weight", "BF16", &[4]),
            (
                "model.layers.1.post_attention_layernorm.weight",
                "BF16",
                &[8],
            ),
            ("model.layers.1.mlp.gate_proj.weight", "BF16", &[32, 8]),
            ("model.layers.1.mlp.up_proj.weight", "BF16", &[32, 8]),
            ("model.layers.1.mlp.down_proj.weight", "BF16", &[8, 32]),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("Gemma4 assistant conversion should succeed");
    assert_eq!(manifest.model_family, "gemma4_assistant");
    assert_eq!(manifest.hidden_size_per_layer_input, 0);
    assert_eq!(manifest.vocab_size_per_layer_input, None);
    assert_eq!(
        manifest.layer_types,
        vec![
            "sliding_attention".to_string(),
            "full_attention".to_string()
        ]
    );
    assert!(manifest.kv_shared_source_layers.is_empty());
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::AssistantPreProjection)
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::AssistantPostProjection)
    );
    assert!(!manifest.tensors.iter().any(|tensor| matches!(
        tensor.role,
        NativeTensorRole::AttentionK | NativeTensorRole::AttentionV
    )));
    crate::model::NativeModelArtifacts::from_manifest_and_root(dir, manifest)
        .expect("Gemma4 assistant manifest should validate");
}

#[test]
fn gemma4_moe_expert_names_map_to_unambiguous_roles() {
    let empty_config = serde_json::json!({});
    let family =
        model_family_for_type("gemma4", &empty_config).expect("gemma4 should be supported");

    assert_eq!(
        match_tensor(
            "language_model.model.layers.0.experts.gate_up_proj.weight",
            &family,
        ),
        Some((NativeTensorRole::FfnGateUpExpsPacked, Some(0)))
    );
    assert_eq!(
        match_tensor(
            "language_model.model.layers.0.experts.switch_glu.gate_proj.weight",
            &family,
        ),
        Some((NativeTensorRole::FfnGateExps, Some(0)))
    );
    assert_eq!(
        match_tensor(
            "language_model.model.layers.0.experts.switch_glu.up_proj.weight",
            &family,
        ),
        Some((NativeTensorRole::FfnUpExps, Some(0)))
    );
    assert_eq!(
        match_tensor(
            "language_model.model.layers.0.experts.switch_glu.down_proj.weight",
            &family,
        ),
        Some((NativeTensorRole::FfnDownExps, Some(0)))
    );
}

#[test]
fn converts_gemma4_default_layer_type_pattern_for_k_eq_v() {
    let dir = unique_test_dir("gemma4_default_layer_types");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "gemma4",
            "vocab_size": 262144,
            "text_config": {
                "hidden_size": 3072,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 4,
                "head_dim": 128,
                "global_head_dim": 256,
                "attention_k_eq_v": true,
                "sliding_window_pattern": 5,
                "num_hidden_layers": 5,
                "vocab_size": 262144
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "BF16", &[262144, 3072]),
            ("model.norm.weight", "BF16", &[3072]),
            ("lm_head.weight", "BF16", &[262144, 3072]),
            ("model.layers.4.input_layernorm.weight", "BF16", &[3072]),
            (
                "model.layers.4.self_attn.q_proj.weight",
                "BF16",
                &[8192, 3072],
            ),
            (
                "model.layers.4.self_attn.k_proj.weight",
                "BF16",
                &[1024, 3072],
            ),
            (
                "model.layers.4.self_attn.o_proj.weight",
                "BF16",
                &[3072, 8192],
            ),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("gemma4 conversion should succeed");

    assert_eq!(
        manifest.layer_types,
        vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ]
    );
    assert_eq!(manifest.attention_value_from_key_layers, vec![4]);

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn parses_root_gemma4_layer_types_before_text_config() {
    let config = serde_json::json!({
        "model_type": "gemma4",
        "layer_types": ["full_attention"],
        "text_config": {
            "layer_types": ["sliding_attention"]
        }
    });

    let layer_types = parse_layer_types(&config, "gemma4", 1);

    assert_eq!(layer_types, vec!["full_attention".to_string()]);
}

#[test]
fn parses_gemma4_sliding_rope_theta_from_flat_fallback() {
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "rope_theta": 1000000
        }
    });

    let (full_theta, sliding_theta, partial_rotary) = parse_rope_params(&config, "gemma4");

    assert_eq!(full_theta, Some(1000000));
    assert_eq!(sliding_theta, Some(1000000));
    assert_eq!(partial_rotary, None);
}

#[test]
fn parses_gemma4_assistant_nested_rope_like_gemma4() {
    // The assistant drafter carries the identical nested rope_parameters
    // layout and must take the gemma4 branch — otherwise its Q rotation
    // stops matching the target's cached K and the draft accept rate
    // collapses (~20%). Mirrors the real assistant config.json shape.
    let config = serde_json::json!({
        "model_type": "gemma4_assistant",
        "text_config": {
            "rope_parameters": {
                "full_attention": {
                    "rope_theta": 1000000,
                    "partial_rotary_factor": 0.25,
                },
                "sliding_attention": {
                    "rope_theta": 10000,
                },
            }
        }
    });

    let (full_theta, sliding_theta, partial_rotary) =
        parse_rope_params(&config, "gemma4_assistant");

    assert_eq!(full_theta, Some(1000000));
    assert_eq!(sliding_theta, Some(10000));
    assert_eq!(partial_rotary, Some(0.25));
}

#[test]
fn rejects_gemma4_layer_types_length_mismatch_at_conversion() {
    let dir = unique_test_dir("gemma4_bad_layer_types");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "gemma4",
            "vocab_size": 262144,
            "text_config": {
                "hidden_size": 3072,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "num_hidden_layers": 2,
                "vocab_size": 262144,
                "layer_types": ["sliding_attention"]
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "BF16", &[262144, 3072]),
            ("model.norm.weight", "BF16", &[3072]),
            ("lm_head.weight", "BF16", &[262144, 3072]),
        ],
    );

    let error = convert_hf_model_dir(&dir).expect_err("bad layer_types should fail closed");
    let ConvertError::InvalidModelContract {
        model_type,
        message,
    } = error
    else {
        panic!("expected invalid model contract");
    };
    assert_eq!(model_type, "gemma4");
    assert!(message.contains("layer_types"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn rejects_gemma4_per_layer_input_missing_weights_at_conversion() {
    let dir = unique_test_dir("gemma4_missing_per_layer_input");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "gemma4",
            "vocab_size": 262144,
            "text_config": {
                "hidden_size": 3072,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "num_hidden_layers": 2,
                "vocab_size": 262144,
                "hidden_size_per_layer_input": 64,
                "layer_types": ["sliding_attention", "sliding_attention"]
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "BF16", &[262144, 3072]),
            ("model.norm.weight", "BF16", &[3072]),
            ("lm_head.weight", "BF16", &[262144, 3072]),
        ],
    );

    let error =
        convert_hf_model_dir(&dir).expect_err("missing per-layer inputs should fail closed");
    let ConvertError::InvalidModelContract {
        model_type,
        message,
    } = error
    else {
        panic!("expected invalid model contract");
    };
    assert_eq!(model_type, "gemma4");
    assert!(message.contains("per-layer input"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn converts_gemma4_default_kv_shared_layers() {
    let dir = unique_test_dir("gemma4_default_kv_shared");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "gemma4",
            "vocab_size": 262144,
            "text_config": {
                "hidden_size": 3072,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "global_head_dim": 256,
                "sliding_window_pattern": 5,
                "num_hidden_layers": 35,
                "vocab_size": 262144
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "BF16", &[262144, 3072]),
            ("model.norm.weight", "BF16", &[3072]),
            ("lm_head.weight", "BF16", &[262144, 3072]),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("gemma4 conversion should succeed");

    assert_eq!(manifest.layer_types.len(), 35);
    assert_eq!(manifest.layer_types[4], "full_attention");
    assert_eq!(manifest.layer_types[34], "full_attention");
    assert_eq!(manifest.kv_shared_source_layers.len(), 20);
    assert_eq!(manifest.kv_shared_source_layers.get(&15), Some(&13));
    assert_eq!(manifest.kv_shared_source_layers.get(&19), Some(&14));
    assert!(!manifest.kv_shared_source_layers.contains_key(&14));
    assert!(manifest.attention_v_norm_no_scale_layers.contains(&14));
    assert!(!manifest.attention_v_norm_no_scale_layers.contains(&15));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn converts_gemma4_unified_text_without_tower_tensors() {
    let dir = unique_test_dir("gemma4_unified_text");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "vocab_size": 262144,
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 3072,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 4,
                "head_dim": 128,
                "global_head_dim": 256,
                "sliding_window": 1024,
                "attention_k_eq_v": true,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "vocab_size_per_layer_input": 262144,
                "final_logit_softcapping": 30.0,
                "num_hidden_layers": 2,
                "vocab_size": 262144,
                "layer_types": ["sliding_attention", "full_attention"],
                "rope_parameters": {
                    "full_attention": {
                        "rope_theta": 1000000,
                        "partial_rotary_factor": 0.25
                    },
                    "sliding_attention": {
                        "rope_theta": 10000
                    }
                }
            },
            "vision_config": {
                "model_type": "gemma4_unified_vision"
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            (
                "language_model.model.embed_tokens.weight",
                "BF16",
                &[262144, 3072],
            ),
            ("language_model.model.norm.weight", "BF16", &[3072]),
            ("language_model.lm_head.weight", "BF16", &[262144, 3072]),
            (
                "language_model.model.layers.0.input_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "BF16",
                &[4096, 3072],
            ),
            (
                "language_model.model.layers.0.self_attn.k_proj.weight",
                "BF16",
                &[1024, 3072],
            ),
            (
                "language_model.model.layers.0.self_attn.v_proj.weight",
                "BF16",
                &[1024, 3072],
            ),
            (
                "language_model.model.layers.0.self_attn.o_proj.weight",
                "BF16",
                &[3072, 4096],
            ),
            (
                "language_model.model.layers.0.self_attn.q_norm.weight",
                "BF16",
                &[128],
            ),
            (
                "language_model.model.layers.0.self_attn.k_norm.weight",
                "BF16",
                &[128],
            ),
            (
                "language_model.model.layers.0.post_attention_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "language_model.model.layers.0.pre_feedforward_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "language_model.model.layers.0.post_feedforward_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "language_model.model.layers.0.mlp.gate_proj.weight",
                "BF16",
                &[12288, 3072],
            ),
            (
                "language_model.model.layers.0.mlp.up_proj.weight",
                "BF16",
                &[12288, 3072],
            ),
            (
                "language_model.model.layers.0.mlp.down_proj.weight",
                "BF16",
                &[3072, 12288],
            ),
            (
                "language_model.model.layers.1.input_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "language_model.model.layers.1.self_attn.q_proj.weight",
                "BF16",
                &[8192, 3072],
            ),
            (
                "language_model.model.layers.1.self_attn.k_proj.weight",
                "BF16",
                &[1024, 3072],
            ),
            (
                "language_model.model.layers.1.self_attn.o_proj.weight",
                "BF16",
                &[3072, 8192],
            ),
            (
                "language_model.model.layers.1.self_attn.q_norm.weight",
                "BF16",
                &[256],
            ),
            (
                "language_model.model.layers.1.self_attn.k_norm.weight",
                "BF16",
                &[256],
            ),
            (
                "language_model.model.layers.1.post_attention_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "language_model.model.layers.1.pre_feedforward_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "language_model.model.layers.1.post_feedforward_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "language_model.model.layers.1.mlp.gate_proj.weight",
                "BF16",
                &[12288, 3072],
            ),
            (
                "language_model.model.layers.1.mlp.up_proj.weight",
                "BF16",
                &[12288, 3072],
            ),
            (
                "language_model.model.layers.1.mlp.down_proj.weight",
                "BF16",
                &[3072, 12288],
            ),
            // Unified multimodal tensors are global projector roles, not towers.
            ("vision_embedder.pos_embedding", "BF16", &[1120, 2, 3072]),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("unified text conversion should succeed");

    assert_eq!(manifest.model_family, "gemma4");
    assert_eq!(manifest.hidden_size, 3072);
    assert_eq!(manifest.rope_theta, Some(1000000));
    assert_eq!(manifest.rope_theta_swa, Some(10000));
    assert_eq!(manifest.partial_rotary_factor, Some(0.25));
    assert_eq!(manifest.global_head_dim, Some(256));
    assert_eq!(manifest.sliding_window_size, Some(1024));
    assert_eq!(manifest.final_logit_softcapping, Some(30.0));
    assert_eq!(manifest.hidden_states_scale, Some((3072_f32).sqrt()));
    assert_eq!(manifest.hidden_size_per_layer_input, 0);
    assert_eq!(manifest.vocab_size_per_layer_input, None);
    assert_eq!(
        manifest.layer_types,
        vec![
            "sliding_attention".to_string(),
            "full_attention".to_string()
        ]
    );
    assert!(manifest.kv_shared_source_layers.is_empty());
    assert_eq!(manifest.attention_value_from_key_layers, vec![1]);
    assert_eq!(manifest.attention_v_norm_no_scale_layers, vec![0, 1]);
    assert!(
        manifest.tensors.iter().any(|tensor| {
            tensor.role == NativeTensorRole::Gemma4UnifiedVisionPositionEmbedding
                && tensor.name == "vision_embedder.pos_embedding"
        }),
        "unified projector tensors should be mapped for multimodal runtime support"
    );
    write_manifest(&dir, &manifest).expect("write should succeed");
    crate::model::NativeModelArtifacts::from_dir(&dir)
        .expect("Gemma4 unified text manifest should validate");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn attention_k_eq_v_default_is_model_type_aware_when_field_absent() {
    let layer_types = vec![
        "sliding_attention".to_string(),
        "full_attention".to_string(),
    ];
    let no_shared = BTreeMap::new();
    // Field omitted from config: the dataclass defaults differ by type, so
    // gemma4_unified must default True (full-attention layer 1 uses V-from-K)
    // while standard gemma4 must default False (no V-from-K layers).
    let empty = serde_json::json!({});
    assert_eq!(
        compute_attention_value_from_key_layers(
            &empty,
            "gemma4_unified_text",
            &layer_types,
            &no_shared,
            2,
        ),
        vec![1],
    );
    assert_eq!(
        compute_attention_value_from_key_layers(
            &empty,
            "gemma4_unified",
            &layer_types,
            &no_shared,
            2,
        ),
        vec![1],
    );
    assert_eq!(
        compute_attention_value_from_key_layers(
            &empty,
            "diffusion_gemma",
            &layer_types,
            &no_shared,
            2,
        ),
        vec![1],
    );
    assert!(
        compute_attention_value_from_key_layers(&empty, "gemma4", &layer_types, &no_shared, 2,)
            .is_empty(),
    );
    // An explicit value still overrides the per-type default.
    let disabled = serde_json::json!({ "attention_k_eq_v": false });
    assert!(
        compute_attention_value_from_key_layers(
            &disabled,
            "gemma4_unified_text",
            &layer_types,
            &no_shared,
            2,
        )
        .is_empty(),
    );
}

#[test]
fn converts_gemma4_k_eq_v_full_attention_layers() {
    let dir = unique_test_dir("gemma4_k_eq_v");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "gemma4",
            "vocab_size": 262144,
            "text_config": {
                "hidden_size": 3072,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 4,
                "head_dim": 128,
                "global_head_dim": 256,
                "attention_k_eq_v": true,
                "num_hidden_layers": 1,
                "vocab_size": 262144,
                "layer_types": ["full_attention"]
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "BF16", &[262144, 3072]),
            ("model.norm.weight", "BF16", &[3072]),
            ("lm_head.weight", "BF16", &[262144, 3072]),
            ("model.layers.0.input_layernorm.weight", "BF16", &[3072]),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "BF16",
                &[8192, 3072],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "BF16",
                &[1024, 3072],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "BF16",
                &[3072, 8192],
            ),
            ("model.layers.0.self_attn.q_norm.weight", "BF16", &[256]),
            ("model.layers.0.self_attn.k_norm.weight", "BF16", &[256]),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "model.layers.0.pre_feedforward_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "model.layers.0.post_feedforward_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "BF16",
                &[12288, 3072],
            ),
            ("model.layers.0.mlp.up_proj.weight", "BF16", &[12288, 3072]),
            (
                "model.layers.0.mlp.down_proj.weight",
                "BF16",
                &[3072, 12288],
            ),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("gemma4 conversion should succeed");

    assert_eq!(manifest.attention_value_from_key_layers, vec![0]);
    assert!(
        !manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::AttentionV),
        "K=V Gemma4 full-attention layer should not require v_proj"
    );
    write_manifest(&dir, &manifest).expect("write should succeed");
    crate::model::NativeModelArtifacts::from_dir(&dir)
        .expect("K=V Gemma4 manifest should validate");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn converts_qwen3_f16_model_directory() {
    let dir = unique_test_dir("qwen3");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 2,
            "vocab_size": 151936,
            "tie_word_embeddings": true,
            "rope_theta": 1000000,
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "F16", &[151936, 4096]),
            ("model.norm.weight", "F16", &[4096]),
            ("model.layers.0.input_layernorm.weight", "F16", &[4096]),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "F16",
                &[4096, 4096],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "F16",
                &[1024, 4096],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                "F16",
                &[1024, 4096],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "F16",
                &[4096, 4096],
            ),
            ("model.layers.0.self_attn.q_norm.weight", "F16", &[128]),
            ("model.layers.0.self_attn.k_norm.weight", "F16", &[128]),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "F16",
                &[4096],
            ),
            ("model.layers.0.mlp.gate_proj.weight", "F16", &[12288, 4096]),
            ("model.layers.0.mlp.up_proj.weight", "F16", &[12288, 4096]),
            ("model.layers.0.mlp.down_proj.weight", "F16", &[4096, 12288]),
            ("model.layers.1.input_layernorm.weight", "F16", &[4096]),
            (
                "model.layers.1.self_attn.q_proj.weight",
                "F16",
                &[4096, 4096],
            ),
            (
                "model.layers.1.self_attn.k_proj.weight",
                "F16",
                &[1024, 4096],
            ),
            (
                "model.layers.1.self_attn.v_proj.weight",
                "F16",
                &[1024, 4096],
            ),
            (
                "model.layers.1.self_attn.o_proj.weight",
                "F16",
                &[4096, 4096],
            ),
            ("model.layers.1.self_attn.q_norm.weight", "F16", &[128]),
            ("model.layers.1.self_attn.k_norm.weight", "F16", &[128]),
            (
                "model.layers.1.post_attention_layernorm.weight",
                "F16",
                &[4096],
            ),
            ("model.layers.1.mlp.gate_proj.weight", "F16", &[12288, 4096]),
            ("model.layers.1.mlp.up_proj.weight", "F16", &[12288, 4096]),
            ("model.layers.1.mlp.down_proj.weight", "F16", &[4096, 12288]),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("qwen3 conversion should succeed");

    assert_eq!(manifest.model_family, "qwen3");
    assert_eq!(manifest.layer_count, 2);
    assert_eq!(manifest.hidden_size, 4096);
    assert_eq!(manifest.attention_head_count, 32);
    assert_eq!(manifest.attention_head_dim, 128);
    assert_eq!(manifest.kv_head_count, 8);
    assert_eq!(manifest.vocab_size, 151936);
    assert!(manifest.tie_word_embeddings);
    assert_eq!(manifest.rope_theta, Some(1000000));

    let roles: Vec<_> = manifest.tensors.iter().map(|t| t.role).collect();
    assert!(roles.contains(&NativeTensorRole::TokenEmbedding));
    assert!(roles.contains(&NativeTensorRole::FinalNorm));
    assert!(roles.contains(&NativeTensorRole::AttentionQ));
    assert!(roles.contains(&NativeTensorRole::AttentionQNorm));
    assert!(roles.contains(&NativeTensorRole::AttentionKNorm));
    assert!(roles.contains(&NativeTensorRole::FfnGate));
    assert!(roles.contains(&NativeTensorRole::FfnUp));
    assert!(roles.contains(&NativeTensorRole::FfnDown));

    let layer0_q = manifest
        .tensors
        .iter()
        .find(|t| t.role == NativeTensorRole::AttentionQ && t.layer_index == Some(0))
        .expect("layer 0 q_proj should exist");
    assert_eq!(layer0_q.dtype, NativeTensorDataType::F16);
    assert_eq!(layer0_q.shape, vec![4096, 4096]);

    // Verify no lm_head when tie_word_embeddings is true
    // (the model doesn't have lm_head.weight in the safetensors)
    assert!(!roles.contains(&NativeTensorRole::LmHead));

    // Write and re-read
    write_manifest(&dir, &manifest).expect("write should succeed");
    let reloaded = crate::model::NativeModelArtifacts::from_dir(&dir)
        .expect("reloaded manifest should validate");
    assert_eq!(reloaded.manifest().layer_count, 2);

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn converts_gemma4_with_text_config_nesting() {
    let dir = unique_test_dir("gemma4");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "gemma4",
            "vocab_size": 262144,
            "text_config": {
                "hidden_size": 3072,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "global_head_dim": 256,
                "sliding_window": 1024,
                "num_hidden_layers": 2,
                "vocab_size": 262144,
                "rope_theta": 1000000,
                "final_logit_softcapping": 30.0,
                "layer_types": ["sliding_attention", "sliding_attention"],
                "num_kv_shared_layers": 1,
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "BF16", &[262144, 3072]),
            ("model.norm.weight", "BF16", &[3072]),
            ("lm_head.weight", "BF16", &[262144, 3072]),
            ("model.layers.0.input_layernorm.weight", "BF16", &[3072]),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "BF16",
                &[4096, 3072],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "BF16",
                &[1024, 3072],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                "BF16",
                &[1024, 3072],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "BF16",
                &[3072, 4096],
            ),
            ("model.layers.0.self_attn.q_norm.weight", "BF16", &[128]),
            ("model.layers.0.self_attn.k_norm.weight", "BF16", &[128]),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "model.layers.0.pre_feedforward_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "model.layers.0.post_feedforward_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "BF16",
                &[12288, 3072],
            ),
            ("model.layers.0.mlp.up_proj.weight", "BF16", &[12288, 3072]),
            (
                "model.layers.0.mlp.down_proj.weight",
                "BF16",
                &[3072, 12288],
            ),
            ("model.layers.1.input_layernorm.weight", "BF16", &[3072]),
            (
                "model.layers.1.self_attn.q_proj.weight",
                "BF16",
                &[4096, 3072],
            ),
            (
                "model.layers.1.self_attn.o_proj.weight",
                "BF16",
                &[3072, 4096],
            ),
            ("model.layers.1.self_attn.q_norm.weight", "BF16", &[128]),
            (
                "model.layers.1.post_attention_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "model.layers.1.pre_feedforward_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "model.layers.1.post_feedforward_layernorm.weight",
                "BF16",
                &[3072],
            ),
            (
                "model.layers.1.mlp.gate_proj.weight",
                "BF16",
                &[12288, 3072],
            ),
            ("model.layers.1.mlp.up_proj.weight", "BF16", &[12288, 3072]),
            (
                "model.layers.1.mlp.down_proj.weight",
                "BF16",
                &[3072, 12288],
            ),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("gemma4 conversion should succeed");

    assert_eq!(manifest.model_family, "gemma4");
    assert_eq!(manifest.hidden_size, 3072);
    assert_eq!(manifest.attention_head_count, 32);
    assert_eq!(manifest.attention_head_dim, 128);
    assert_eq!(manifest.kv_head_count, 8);
    assert_eq!(manifest.vocab_size, 262144);
    assert_eq!(manifest.rope_theta, Some(1000000));
    assert_eq!(manifest.global_head_dim, Some(256));
    assert_eq!(manifest.sliding_window_size, Some(1024));
    assert_eq!(manifest.final_logit_softcapping, Some(30.0));
    assert_eq!(manifest.hidden_states_scale, Some((3072_f32).sqrt()));
    assert!(!manifest.moe_norm_topk_prob);
    assert_eq!(manifest.attention_v_norm_no_scale_layers, vec![0]);
    assert_eq!(
        manifest.layer_types,
        vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string()
        ]
    );
    assert_eq!(manifest.kv_shared_source_layers.get(&1), Some(&0));
    assert!(
        !manifest.tensors.iter().any(|tensor| {
            tensor.layer_index == Some(1)
                && matches!(
                    tensor.role,
                    NativeTensorRole::AttentionK | NativeTensorRole::AttentionV
                )
        }),
        "KV-shared Gemma4 layers should reuse source K/V instead of mapping their own"
    );

    let has_lm_head = manifest
        .tensors
        .iter()
        .any(|t| t.role == NativeTensorRole::LmHead);
    assert!(has_lm_head);

    write_manifest(&dir, &manifest).expect("write should succeed");
    crate::model::NativeModelArtifacts::from_dir(&dir)
        .expect("Gemma4 KV-shared manifest should validate");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn converts_qwen3_5_linear_attention_model_directory() {
    let dir = unique_test_dir("qwen3_5_linear");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "qwen3_5",
            "vocab_size": 32,
            "text_config": {
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "num_hidden_layers": 1,
                "vocab_size": 32,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 2,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("language_model.model.embed_tokens.weight", "BF16", &[32, 8]),
            ("language_model.model.norm.weight", "BF16", &[8]),
            ("language_model.lm_head.weight", "BF16", &[32, 8]),
            (
                "language_model.model.layers.0.input_layernorm.weight",
                "BF16",
                &[8],
            ),
            (
                "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
                "BF16",
                &[12, 8],
            ),
            (
                "language_model.model.layers.0.linear_attn.in_proj_z.weight",
                "BF16",
                &[4, 8],
            ),
            (
                "language_model.model.layers.0.linear_attn.in_proj_a.weight",
                "BF16",
                &[2, 8],
            ),
            (
                "language_model.model.layers.0.linear_attn.in_proj_b.weight",
                "BF16",
                &[2, 8],
            ),
            (
                "language_model.model.layers.0.linear_attn.conv1d.weight",
                "BF16",
                &[12, 1, 4],
            ),
            (
                "language_model.model.layers.0.linear_attn.dt_bias",
                "F32",
                &[2],
            ),
            (
                "language_model.model.layers.0.linear_attn.A_log",
                "F32",
                &[2],
            ),
            (
                "language_model.model.layers.0.linear_attn.norm.weight",
                "BF16",
                &[2],
            ),
            (
                "language_model.model.layers.0.linear_attn.out_proj.weight",
                "BF16",
                &[8, 4],
            ),
            (
                "language_model.model.layers.0.post_attention_layernorm.weight",
                "BF16",
                &[8],
            ),
            (
                "language_model.model.layers.0.pre_feedforward_layernorm.weight",
                "BF16",
                &[8],
            ),
            (
                "language_model.model.layers.0.mlp.gate_proj.weight",
                "BF16",
                &[16, 8],
            ),
            (
                "language_model.model.layers.0.mlp.up_proj.weight",
                "BF16",
                &[16, 8],
            ),
            (
                "language_model.model.layers.0.mlp.down_proj.weight",
                "BF16",
                &[8, 16],
            ),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("qwen3.5 linear conversion should succeed");

    assert_eq!(manifest.model_family, "qwen3_5");
    assert_eq!(manifest.linear_attention.num_value_heads, Some(2));
    assert_eq!(manifest.linear_attention.num_key_heads, Some(1));
    assert_eq!(manifest.linear_attention.key_head_dim, Some(4));
    assert_eq!(manifest.linear_attention.value_head_dim, Some(2));
    assert_eq!(manifest.linear_attention.conv_kernel_dim, Some(4));
    assert_eq!(manifest.linear_attention.full_attention_interval, Some(4));
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::LinearAttentionInProjQkv)
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::LinearAttentionConv1d)
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::LinearAttentionOutProj)
    );

    write_manifest(&dir, &manifest).expect("write should succeed");
    crate::model::NativeModelArtifacts::from_dir(&dir)
        .expect("linear-attention manifest should validate");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn converts_qwen3_5_moe_language_model_switch_mlp_directory() {
    let dir = unique_test_dir("qwen3_5_moe_language_model");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "qwen3_5_moe",
            "vocab_size": 32,
            "text_config": {
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "num_hidden_layers": 1,
                "vocab_size": 32,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 2,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8
            },
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "language_model.model.layers.0.mlp.gate": {
                    "group_size": 64,
                    "bits": 8
                }
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("language_model.model.embed_tokens.weight", "BF16", &[32, 8]),
            ("language_model.model.norm.weight", "BF16", &[8]),
            ("language_model.lm_head.weight", "BF16", &[32, 8]),
            (
                "language_model.model.layers.0.input_layernorm.weight",
                "BF16",
                &[8],
            ),
            (
                "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
                "BF16",
                &[12, 8],
            ),
            (
                "language_model.model.layers.0.linear_attn.in_proj_z.weight",
                "BF16",
                &[4, 8],
            ),
            (
                "language_model.model.layers.0.linear_attn.in_proj_a.weight",
                "BF16",
                &[2, 8],
            ),
            (
                "language_model.model.layers.0.linear_attn.in_proj_b.weight",
                "BF16",
                &[2, 8],
            ),
            (
                "language_model.model.layers.0.linear_attn.conv1d.weight",
                "BF16",
                &[12, 1, 4],
            ),
            (
                "language_model.model.layers.0.linear_attn.dt_bias",
                "F32",
                &[2],
            ),
            (
                "language_model.model.layers.0.linear_attn.A_log",
                "F32",
                &[2],
            ),
            (
                "language_model.model.layers.0.linear_attn.norm.weight",
                "BF16",
                &[2],
            ),
            (
                "language_model.model.layers.0.linear_attn.out_proj.weight",
                "BF16",
                &[8, 4],
            ),
            (
                "language_model.model.layers.0.post_attention_layernorm.weight",
                "BF16",
                &[8],
            ),
            (
                "language_model.model.layers.0.mlp.gate.weight",
                "U32",
                &[4, 2],
            ),
            (
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                "BF16",
                &[4, 8, 8],
            ),
            (
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight",
                "BF16",
                &[4, 8, 8],
            ),
            (
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
                "BF16",
                &[4, 8, 8],
            ),
            (
                "language_model.model.layers.0.mlp.shared_expert_gate.weight",
                "BF16",
                &[1, 8],
            ),
            (
                "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
                "BF16",
                &[8, 8],
            ),
            (
                "language_model.model.layers.0.mlp.shared_expert.up_proj.weight",
                "BF16",
                &[8, 8],
            ),
            (
                "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
                "BF16",
                &[8, 8],
            ),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("qwen3.5 MoE conversion should succeed");

    assert_eq!(manifest.model_family, "qwen3_5");
    assert_eq!(manifest.moe.expert_count, Some(4));
    assert_eq!(manifest.moe.experts_per_token, Some(2));
    assert_eq!(manifest.moe.expert_intermediate_size, Some(8));
    assert!(
        manifest.moe_norm_topk_prob,
        "Qwen3.5 MoE defaults norm_topk_prob=true in mlx_lm"
    );
    assert!(
        manifest.attn_output_gate,
        "Qwen3.5 MoE full-attention layers must default to the reference output gate"
    );
    for role in [
        NativeTensorRole::FfnGateInp,
        NativeTensorRole::FfnGateExps,
        NativeTensorRole::FfnUpExps,
        NativeTensorRole::FfnDownExps,
        NativeTensorRole::FfnSharedExpertDown,
    ] {
        assert!(
            manifest.tensors.iter().any(|tensor| tensor.role == role),
            "missing role {role:?}"
        );
    }
    let gate = manifest
        .tensors
        .iter()
        .find(|tensor| tensor.role == NativeTensorRole::FfnGateInp)
        .expect("router should map");
    assert_eq!(gate.quantization.as_ref().map(|q| q.bits), Some(8));

    write_manifest(&dir, &manifest).expect("write should succeed");
    crate::model::NativeModelArtifacts::from_dir(&dir)
        .expect("qwen3.5 MoE manifest should validate");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn maps_qwen3_moe_switch_mlp_tensors() {
    let empty_config = serde_json::json!({});
    let family =
        model_family_for_type("qwen3_moe", &empty_config).expect("qwen3_moe should be supported");

    assert_eq!(
        match_tensor("model.layers.2.mlp.gate.weight", &family),
        Some((NativeTensorRole::FfnGateInp, Some(2)))
    );
    assert_eq!(
        match_tensor("model.layers.2.mlp.switch_mlp.gate_proj.weight", &family),
        Some((NativeTensorRole::FfnGateExps, Some(2)))
    );
    assert_eq!(
        match_tensor("model.layers.2.mlp.switch_mlp.up_proj.weight", &family),
        Some((NativeTensorRole::FfnUpExps, Some(2)))
    );
    assert_eq!(
        match_tensor("model.layers.2.mlp.switch_mlp.down_proj.weight", &family),
        Some((NativeTensorRole::FfnDownExps, Some(2)))
    );
}

#[test]
fn qwen3_5_model_type_activates_moe_tensor_map_when_config_has_experts() {
    let moe_config = serde_json::json!({
        "text_config": {
            "num_experts": 128,
            "num_experts_per_tok": 8
        }
    });
    let family =
        model_family_for_type("qwen3_5", &moe_config).expect("qwen3_5 should be supported");
    assert_eq!(family.family_name, "qwen3_5");
    assert_eq!(
        match_tensor("model.layers.2.mlp.gate.weight", &family),
        Some((NativeTensorRole::FfnGateInp, Some(2)))
    );
    assert_eq!(
        match_tensor("model.layers.2.mlp.switch_mlp.gate_proj.weight", &family),
        Some((NativeTensorRole::FfnGateExps, Some(2)))
    );
    assert_eq!(
        match_tensor("model.layers.2.mlp.switch_mlp.up_proj.weight", &family),
        Some((NativeTensorRole::FfnUpExps, Some(2)))
    );
    assert_eq!(
        match_tensor("model.layers.2.mlp.switch_mlp.down_proj.weight", &family),
        Some((NativeTensorRole::FfnDownExps, Some(2)))
    );
}

#[test]
fn qwen3_5_model_type_without_moe_config_has_no_moe_tensors() {
    let dense_config = serde_json::json!({
        "text_config": {
            "hidden_size": 64
        }
    });
    let family =
        model_family_for_type("qwen3_5", &dense_config).expect("qwen3_5 should be supported");
    assert_eq!(family.family_name, "qwen3_5");
    assert_eq!(
        match_tensor("model.layers.2.mlp.gate.weight", &family),
        None
    );
    assert_eq!(
        match_tensor("model.layers.2.mlp.switch_mlp.gate_proj.weight", &family),
        None
    );
}

#[test]
fn qwen3_5_moe_config_detected_from_num_experts_in_text_config() {
    let config = serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": {
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 256
        }
    });
    let moe = moe_config(&config, "qwen3_5");
    assert_eq!(moe.expert_count, Some(128));
    assert_eq!(moe.experts_per_token, Some(8));
    assert_eq!(moe.expert_intermediate_size, Some(256));
}

#[test]
fn converts_qwen3_next_linear_moe_shared_expert_model_directory() {
    let dir = unique_test_dir("qwen3_next_linear_moe");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "qwen3_next",
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "num_hidden_layers": 1,
            "vocab_size": 128,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "norm_topk_prob": true,
            "quantization_config": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "model.layers.0.mlp.gate": {
                    "group_size": 64,
                    "bits": 8
                },
                "model.layers.0.mlp.shared_expert_gate": {
                    "group_size": 64,
                    "bits": 8
                }
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "BF16", &[128, 64]),
            ("model.norm.weight", "BF16", &[64]),
            ("lm_head.weight", "BF16", &[128, 64]),
            ("model.layers.0.input_layernorm.weight", "BF16", &[64]),
            (
                "model.layers.0.linear_attn.in_proj_qkvz.weight",
                "BF16",
                &[128, 64],
            ),
            (
                "model.layers.0.linear_attn.in_proj_ba.weight",
                "BF16",
                &[4, 64],
            ),
            (
                "model.layers.0.linear_attn.conv1d.weight",
                "BF16",
                &[96, 1, 4],
            ),
            ("model.layers.0.linear_attn.dt_bias", "F32", &[2]),
            ("model.layers.0.linear_attn.A_log", "F32", &[2]),
            ("model.layers.0.linear_attn.norm.weight", "BF16", &[16]),
            (
                "model.layers.0.linear_attn.out_proj.weight",
                "BF16",
                &[64, 32],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "BF16",
                &[64],
            ),
            ("model.layers.0.mlp.gate.weight", "U32", &[4, 16]),
            (
                "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                "BF16",
                &[4, 8, 64],
            ),
            (
                "model.layers.0.mlp.switch_mlp.up_proj.weight",
                "BF16",
                &[4, 8, 64],
            ),
            (
                "model.layers.0.mlp.switch_mlp.down_proj.weight",
                "BF16",
                &[4, 64, 8],
            ),
            (
                "model.layers.0.mlp.shared_expert_gate.weight",
                "U32",
                &[1, 16],
            ),
            (
                "model.layers.0.mlp.shared_expert.gate_proj.weight",
                "BF16",
                &[8, 64],
            ),
            (
                "model.layers.0.mlp.shared_expert.up_proj.weight",
                "BF16",
                &[8, 64],
            ),
            (
                "model.layers.0.mlp.shared_expert.down_proj.weight",
                "BF16",
                &[64, 8],
            ),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("qwen3_next conversion should succeed");

    assert_eq!(manifest.model_family, "qwen3_next");
    assert_eq!(manifest.linear_attention.full_attention_interval, Some(4));
    assert_eq!(manifest.moe.expert_count, Some(4));
    assert_eq!(manifest.moe.experts_per_token, Some(2));
    assert_eq!(manifest.moe.expert_intermediate_size, Some(8));
    assert!(manifest.moe_norm_topk_prob);
    // attn_output_gate must default to true for qwen3_next even when absent from config.json.
    // All full-attention layers in the qwen3_next architecture use the sigmoid output gate.
    assert!(
        manifest.attn_output_gate,
        "qwen3_next attn_output_gate must default to true"
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::FfnSharedExpertDown)
    );
    let gate = manifest
        .tensors
        .iter()
        .find(|tensor| tensor.role == NativeTensorRole::FfnGateInp)
        .expect("Qwen3Next router should map");
    assert_eq!(gate.quantization.as_ref().map(|q| q.bits), Some(8));
    let shared_gate = manifest
        .tensors
        .iter()
        .find(|tensor| tensor.role == NativeTensorRole::FfnSharedExpertGateInp)
        .expect("Qwen3Next shared expert gate should map");
    assert_eq!(shared_gate.quantization.as_ref().map(|q| q.bits), Some(8));

    write_manifest(&dir, &manifest).expect("write should succeed");
    crate::model::NativeModelArtifacts::from_dir(&dir)
        .expect("qwen3_next manifest should validate");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn qwen3_6_alias_produces_moe_config() {
    // Regression test for B1: moe_config() previously checked only
    // `model_type == "qwen3_next"` and missed the "qwen3.6" / "qwen3_6"
    // aliases. A checkpoint using either alias must still get MoE config.
    for alias in ["qwen3.6", "qwen3_6"] {
        let dir = unique_test_dir(&format!("qwen3_6_alias_{}", alias.replace('.', "_")));
        write_config(
            &dir,
            serde_json::json!({
                "model_type": alias,
                "hidden_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "num_hidden_layers": 1,
                "vocab_size": 128,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
            }),
        );
        write_fake_safetensors(
            &dir,
            "model.safetensors",
            &[
                ("model.embed_tokens.weight", "BF16", &[128, 64]),
                ("model.norm.weight", "BF16", &[64]),
                ("lm_head.weight", "BF16", &[128, 64]),
                ("model.layers.0.input_layernorm.weight", "BF16", &[64]),
                (
                    "model.layers.0.linear_attn.in_proj_qkvz.weight",
                    "BF16",
                    &[128, 64],
                ),
                (
                    "model.layers.0.linear_attn.in_proj_ba.weight",
                    "BF16",
                    &[4, 64],
                ),
                (
                    "model.layers.0.linear_attn.conv1d.weight",
                    "BF16",
                    &[96, 1, 4],
                ),
                ("model.layers.0.linear_attn.dt_bias", "F32", &[2]),
                ("model.layers.0.linear_attn.A_log", "F32", &[2]),
                ("model.layers.0.linear_attn.norm.weight", "BF16", &[16]),
                (
                    "model.layers.0.linear_attn.out_proj.weight",
                    "BF16",
                    &[64, 32],
                ),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    "BF16",
                    &[64],
                ),
                ("model.layers.0.mlp.gate.weight", "BF16", &[8, 64]),
                (
                    "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                    "BF16",
                    &[8, 8, 64],
                ),
                (
                    "model.layers.0.mlp.switch_mlp.up_proj.weight",
                    "BF16",
                    &[8, 8, 64],
                ),
                (
                    "model.layers.0.mlp.switch_mlp.down_proj.weight",
                    "BF16",
                    &[8, 64, 8],
                ),
            ],
        );
        let manifest = convert_hf_model_dir(&dir)
            .unwrap_or_else(|e| panic!("convert with alias '{alias}' failed: {e}"));
        assert_eq!(
            manifest.moe.expert_count,
            Some(8),
            "alias '{alias}' must produce MoE config with expert_count=8"
        );
        assert_eq!(manifest.model_family, "qwen3_next");
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
fn converts_gemma4_moe_model_directory() {
    let dir = unique_test_dir("gemma4_moe");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "gemma4",
            "vocab_size": 262144,
            "tie_word_embeddings": true,
            "text_config": {
                "hidden_size": 2816,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "num_hidden_layers": 1,
                "vocab_size": 262144,
                "enable_moe_block": true,
                "num_experts": 128,
                "top_k_experts": 8,
                "moe_intermediate_size": 704
            },
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine"
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            (
                "language_model.model.embed_tokens.weight",
                "BF16",
                &[262144, 2816],
            ),
            ("language_model.model.norm.weight", "BF16", &[2816]),
            (
                "language_model.model.layers.0.input_layernorm.weight",
                "BF16",
                &[2816],
            ),
            (
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "BF16",
                &[2048, 2816],
            ),
            (
                "language_model.model.layers.0.self_attn.k_proj.weight",
                "BF16",
                &[512, 2816],
            ),
            (
                "language_model.model.layers.0.self_attn.v_proj.weight",
                "BF16",
                &[512, 2816],
            ),
            (
                "language_model.model.layers.0.self_attn.o_proj.weight",
                "BF16",
                &[2816, 2048],
            ),
            (
                "language_model.model.layers.0.self_attn.q_norm.weight",
                "BF16",
                &[256],
            ),
            (
                "language_model.model.layers.0.self_attn.k_norm.weight",
                "BF16",
                &[256],
            ),
            (
                "language_model.model.layers.0.post_attention_layernorm.weight",
                "BF16",
                &[2816],
            ),
            (
                "language_model.model.layers.0.pre_feedforward_layernorm.weight",
                "BF16",
                &[2816],
            ),
            (
                "language_model.model.layers.0.post_feedforward_layernorm.weight",
                "BF16",
                &[2816],
            ),
            (
                "language_model.model.layers.0.pre_feedforward_layernorm_2.weight",
                "BF16",
                &[2816],
            ),
            (
                "language_model.model.layers.0.post_feedforward_layernorm_1.weight",
                "BF16",
                &[2816],
            ),
            (
                "language_model.model.layers.0.post_feedforward_layernorm_2.weight",
                "BF16",
                &[2816],
            ),
            (
                "language_model.model.layers.0.mlp.gate_proj.weight",
                "BF16",
                &[2112, 2816],
            ),
            (
                "language_model.model.layers.0.mlp.up_proj.weight",
                "BF16",
                &[2112, 2816],
            ),
            (
                "language_model.model.layers.0.mlp.down_proj.weight",
                "BF16",
                &[2816, 2112],
            ),
            (
                "language_model.model.layers.0.router.proj.weight",
                "U32",
                &[128, 704],
            ),
            (
                "language_model.model.layers.0.router.scale",
                "BF16",
                &[2816],
            ),
            (
                "language_model.model.layers.0.experts.switch_glu.gate_proj.weight",
                "BF16",
                &[128, 704, 2816],
            ),
            (
                "language_model.model.layers.0.experts.switch_glu.up_proj.weight",
                "BF16",
                &[128, 704, 2816],
            ),
            (
                "language_model.model.layers.0.experts.switch_glu.down_proj.weight",
                "BF16",
                &[128, 2816, 704],
            ),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("gemma4 moe conversion should succeed");

    assert_eq!(manifest.model_family, "gemma4");
    assert_eq!(manifest.moe.expert_count, Some(128));
    assert_eq!(manifest.moe.experts_per_token, Some(8));
    assert_eq!(manifest.moe.expert_intermediate_size, Some(704));
    assert_eq!(manifest.hidden_states_scale, Some((2816_f32).sqrt()));
    assert!(!manifest.moe_norm_topk_prob);
    assert_eq!(manifest.attention_v_norm_no_scale_layers, vec![0]);
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::FfnGateInp)
    );
    let router = manifest
        .tensors
        .iter()
        .find(|tensor| tensor.role == NativeTensorRole::FfnGateInp)
        .expect("router should map");
    assert!(router.source_quantized);
    assert_eq!(
        router.quantization.as_ref().map(|q| q.bits),
        Some(8),
        "Gemma4 router.proj should keep mlx-lm's 8-bit quantization contract"
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::AttentionPostNorm)
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::FfnNorm2)
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::FfnPostNorm)
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::FfnPostNorm1)
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::FfnPostNorm2)
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::FfnGateExps)
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::FfnDownExps)
    );

    write_manifest(&dir, &manifest).expect("write should succeed");
    crate::model::NativeModelArtifacts::from_dir(&dir).expect("moe manifest should validate");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn converts_diffusion_gemma_with_decoder_prefix_and_moe() {
    let dir = unique_test_dir("diffusion_gemma_decoder_prefix");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "diffusion_gemma",
            "vocab_size": 262144,
            "canvas_length": 256,
            "generation_config": {
                "max_denoising_steps": 48,
                "t_max": 0.8,
                "t_min": 0.4,
                "stability_threshold": 2,
                "sampler_config": {
                    "confidence_threshold": 0.005
                }
            },
            "sampler_config": {
                "entropy_bound": 0.1
            },
            "text_config": {
                "hidden_size": 3584,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "num_hidden_layers": 1,
                "vocab_size": 262144,
                "num_experts": 128,
                "top_k_experts": 8,
                "moe_intermediate_size": 704
            },
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine"
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.decoder.embed_tokens.weight", "BF16", &[262144, 3584]),
            ("model.decoder.norm.weight", "BF16", &[3584]),
            ("lm_head.weight", "BF16", &[262144, 3584]),
            (
                "model.decoder.layers.0.input_layernorm.weight",
                "BF16",
                &[3584],
            ),
            (
                "model.decoder.layers.0.self_attn.q_proj.weight",
                "BF16",
                &[2048, 3584],
            ),
            (
                "model.decoder.layers.0.self_attn.k_proj.weight",
                "BF16",
                &[512, 3584],
            ),
            (
                "model.decoder.layers.0.self_attn.v_proj.weight",
                "BF16",
                &[512, 3584],
            ),
            (
                "model.decoder.layers.0.self_attn.o_proj.weight",
                "BF16",
                &[3584, 2048],
            ),
            (
                "model.decoder.layers.0.post_attention_layernorm.weight",
                "BF16",
                &[3584],
            ),
            (
                "model.decoder.layers.0.pre_feedforward_layernorm.weight",
                "BF16",
                &[3584],
            ),
            (
                "model.decoder.layers.0.post_feedforward_layernorm.weight",
                "BF16",
                &[3584],
            ),
            (
                "model.decoder.layers.0.pre_feedforward_layernorm_2.weight",
                "BF16",
                &[3584],
            ),
            (
                "model.decoder.layers.0.post_feedforward_layernorm_1.weight",
                "BF16",
                &[3584],
            ),
            (
                "model.decoder.layers.0.post_feedforward_layernorm_2.weight",
                "BF16",
                &[3584],
            ),
            (
                "model.decoder.layers.0.router.proj.weight",
                "BF16",
                &[128, 3584],
            ),
            (
                "model.decoder.layers.0.experts.switch_glu.gate_proj.weight",
                "BF16",
                &[128, 704, 3584],
            ),
            (
                "model.decoder.layers.0.experts.switch_glu.up_proj.weight",
                "BF16",
                &[128, 704, 3584],
            ),
            (
                "model.decoder.layers.0.experts.switch_glu.down_proj.weight",
                "BF16",
                &[128, 3584, 704],
            ),
            (
                "model.decoder.self_conditioning.pre_norm.weight",
                "BF16",
                &[3584],
            ),
            (
                "model.decoder.self_conditioning.gate_proj.weight",
                "U32",
                &[2816, 56],
            ),
            (
                "model.decoder.self_conditioning.gate_proj.scales",
                "BF16",
                &[2816, 56],
            ),
            (
                "model.decoder.self_conditioning.gate_proj.biases",
                "BF16",
                &[2816, 56],
            ),
            (
                "model.decoder.self_conditioning.up_proj.weight",
                "U32",
                &[2816, 56],
            ),
            (
                "model.decoder.self_conditioning.up_proj.scales",
                "BF16",
                &[2816, 56],
            ),
            (
                "model.decoder.self_conditioning.up_proj.biases",
                "BF16",
                &[2816, 56],
            ),
            (
                "model.decoder.self_conditioning.down_proj.weight",
                "U32",
                &[3584, 44],
            ),
            (
                "model.decoder.self_conditioning.down_proj.scales",
                "BF16",
                &[3584, 44],
            ),
            (
                "model.decoder.self_conditioning.down_proj.biases",
                "BF16",
                &[3584, 44],
            ),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("diffusion_gemma conversion should succeed");

    // Model family.
    assert_eq!(manifest.model_family, "diffusion_gemma");

    // Diffusion config: canvas_length -> canvas_size, generation_config.max_denoising_steps.
    assert_eq!(manifest.diffusion.canvas_size, Some(256));
    assert_eq!(manifest.diffusion.max_denoise_steps, Some(48));
    assert_eq!(manifest.diffusion.entropy_bound, Some(0.1));
    assert_eq!(manifest.diffusion.entropy_threshold, Some(0.005));
    assert_eq!(manifest.diffusion.convergence_steps, Some(2));
    assert_eq!(manifest.diffusion.temperature_start, Some(0.8));
    assert_eq!(manifest.diffusion.temperature_end, Some(0.4));

    // MoE config detected via text_config.num_experts.
    assert_eq!(manifest.moe.expert_count, Some(128));
    assert_eq!(manifest.moe.experts_per_token, Some(8));
    assert_eq!(manifest.moe.expert_intermediate_size, Some(704));

    // Tensors mapped from model.decoder.* prefix.
    assert!(
        !manifest.tensors.is_empty(),
        "tensors must not be empty after mapping"
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|t| t.role == NativeTensorRole::TokenEmbedding),
        "embed_tokens must map via model.decoder.* prefix"
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|t| t.role == NativeTensorRole::FinalNorm),
        "norm must map via model.decoder.* prefix"
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|t| t.role == NativeTensorRole::AttentionQ),
        "q_proj must map via model.decoder.layers.* prefix"
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|t| t.role == NativeTensorRole::FfnGateExps),
        "expert gate must map via model.decoder.layers.* prefix"
    );
    assert!(
        manifest
            .tensors
            .iter()
            .any(|t| t.role == NativeTensorRole::DiffusionSelfConditionPreNorm),
        "self-conditioning pre-norm must map via model.decoder.* prefix"
    );
    assert!(
        manifest.tensors.iter().any(|t| {
            t.role == NativeTensorRole::DiffusionSelfConditionGate && t.source_quantized
        }),
        "self-conditioning gate projection must map as quantized weight"
    );
    assert!(
        manifest.tensors.iter().any(|t| {
            t.role == NativeTensorRole::DiffusionSelfConditionDown && t.source_quantized
        }),
        "self-conditioning down projection must map as quantized weight"
    );

    write_manifest(&dir, &manifest).expect("write should succeed");
    crate::model::NativeModelArtifacts::from_dir(&dir)
        .expect("diffusion_gemma manifest should validate");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn converts_glm4_moe_lite_to_draft_manifest_with_mla_roles() {
    let dir = unique_test_dir("glm4_moe_lite");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "glm4_moe_lite",
            "hidden_size": 2048,
            "intermediate_size": 10240,
            "num_attention_heads": 20,
            "num_key_value_heads": 20,
            "num_hidden_layers": 2,
            "vocab_size": 154880,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "q_lora_rank": 768,
            "kv_lora_rank": 512,
            "first_k_dense_replace": 1,
            "n_routed_experts": 64,
            "n_shared_experts": 1,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1536,
            "routed_scaling_factor": 1.8,
            "norm_topk_prob": true,
            "rope_theta": 1000000,
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine"
            }
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "BF16", &[154880, 2048]),
            ("model.norm.weight", "BF16", &[2048]),
            ("lm_head.weight", "BF16", &[154880, 2048]),
            ("model.layers.0.input_layernorm.weight", "BF16", &[2048]),
            (
                "model.layers.0.self_attn.q_a_proj.weight",
                "U32",
                &[768, 256],
            ),
            (
                "model.layers.0.self_attn.q_a_layernorm.weight",
                "BF16",
                &[768],
            ),
            (
                "model.layers.0.self_attn.q_b_proj.weight",
                "U32",
                &[5120, 96],
            ),
            (
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                "U32",
                &[576, 256],
            ),
            (
                "model.layers.0.self_attn.kv_a_layernorm.weight",
                "BF16",
                &[512],
            ),
            (
                "model.layers.0.self_attn.embed_q.weight",
                "U32",
                &[20, 512, 24],
            ),
            (
                "model.layers.0.self_attn.unembed_out.weight",
                "U32",
                &[20, 256, 64],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "U32",
                &[2048, 640],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "BF16",
                &[2048],
            ),
            ("model.layers.0.mlp.gate_proj.weight", "U32", &[10240, 256]),
            ("model.layers.0.mlp.up_proj.weight", "U32", &[10240, 256]),
            ("model.layers.0.mlp.down_proj.weight", "U32", &[2048, 1280]),
            ("model.layers.1.input_layernorm.weight", "BF16", &[2048]),
            (
                "model.layers.1.self_attn.q_a_proj.weight",
                "U32",
                &[768, 256],
            ),
            (
                "model.layers.1.self_attn.q_a_layernorm.weight",
                "BF16",
                &[768],
            ),
            (
                "model.layers.1.self_attn.q_b_proj.weight",
                "U32",
                &[5120, 96],
            ),
            (
                "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
                "U32",
                &[576, 256],
            ),
            (
                "model.layers.1.self_attn.kv_a_layernorm.weight",
                "BF16",
                &[512],
            ),
            (
                "model.layers.1.self_attn.embed_q.weight",
                "U32",
                &[20, 512, 24],
            ),
            (
                "model.layers.1.self_attn.unembed_out.weight",
                "U32",
                &[20, 256, 64],
            ),
            (
                "model.layers.1.self_attn.o_proj.weight",
                "U32",
                &[2048, 640],
            ),
            (
                "model.layers.1.post_attention_layernorm.weight",
                "BF16",
                &[2048],
            ),
            ("model.layers.1.mlp.gate.weight", "BF16", &[64, 2048]),
            (
                "model.layers.1.mlp.gate.e_score_correction_bias",
                "BF16",
                &[64],
            ),
            (
                "model.layers.1.mlp.switch_mlp.gate_proj.weight",
                "U32",
                &[64, 1536, 256],
            ),
            (
                "model.layers.1.mlp.switch_mlp.up_proj.weight",
                "U32",
                &[64, 1536, 256],
            ),
            (
                "model.layers.1.mlp.switch_mlp.down_proj.weight",
                "U32",
                &[64, 2048, 192],
            ),
            (
                "model.layers.1.mlp.shared_experts.gate_proj.weight",
                "U32",
                &[1536, 256],
            ),
            (
                "model.layers.1.mlp.shared_experts.up_proj.weight",
                "U32",
                &[1536, 256],
            ),
            (
                "model.layers.1.mlp.shared_experts.down_proj.weight",
                "U32",
                &[2048, 192],
            ),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("GLM conversion should succeed");

    assert_eq!(manifest.model_family, "glm4_moe_lite");
    assert_eq!(manifest.attention_head_dim, 256);
    assert_eq!(manifest.mla_attention.q_lora_rank, Some(768));
    assert_eq!(manifest.mla_attention.kv_lora_rank, Some(512));
    assert_eq!(manifest.mla_attention.qk_nope_head_dim, Some(192));
    assert_eq!(manifest.mla_attention.qk_rope_head_dim, Some(64));
    assert_eq!(manifest.mla_attention.value_head_dim, Some(256));
    assert_eq!(manifest.moe.expert_count, Some(64));
    assert_eq!(manifest.moe.experts_per_token, Some(4));
    assert_eq!(manifest.moe.expert_intermediate_size, Some(1536));
    assert_eq!(manifest.glm_router.first_dense_layer_count, Some(1));
    assert_eq!(manifest.glm_router.routed_scaling_factor, Some(1.8));
    assert_eq!(manifest.glm_router.n_group, Some(1));
    assert_eq!(manifest.glm_router.topk_group, Some(1));
    assert!(manifest.glm_router.has_shared_experts);
    assert!(manifest.moe_norm_topk_prob);
    assert!(manifest.runtime_status.ready);
    assert!(manifest.runtime_status.blockers.is_empty());

    for role in [
        NativeTensorRole::AttentionQa,
        NativeTensorRole::AttentionQaNorm,
        NativeTensorRole::AttentionQb,
        NativeTensorRole::AttentionKvA,
        NativeTensorRole::AttentionKvANorm,
        NativeTensorRole::AttentionEmbedQ,
        NativeTensorRole::AttentionUnembedOut,
        NativeTensorRole::FfnGateInpCorrectionBias,
        NativeTensorRole::FfnSharedExpertGate,
    ] {
        assert!(
            manifest.tensors.iter().any(|tensor| tensor.role == role),
            "GLM manifest should map {role:?}"
        );
    }

    write_manifest(&dir, &manifest).expect("write should succeed");
    crate::model::NativeModelArtifacts::from_dir(&dir)
        .expect("runtime-ready GLM manifest should validate");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn converts_deepseek_v3_raw_kv_b_projection_to_mla_manifest() {
    let dir = unique_test_dir("deepseek_v3");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "deepseek_v3",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "vocab_size": 64,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 2,
            "v_head_dim": 3,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "n_routed_experts": 3,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 5,
            "routed_scaling_factor": 2.5,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "rope_theta": 1000000
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "BF16", &[64, 16]),
            ("model.norm.weight", "BF16", &[16]),
            ("lm_head.weight", "BF16", &[64, 16]),
            ("model.layers.0.input_layernorm.weight", "BF16", &[16]),
            ("model.layers.0.self_attn.q_a_proj.weight", "BF16", &[4, 16]),
            (
                "model.layers.0.self_attn.q_a_layernorm.weight",
                "BF16",
                &[4],
            ),
            ("model.layers.0.self_attn.q_b_proj.weight", "BF16", &[12, 4]),
            (
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                "BF16",
                &[6, 16],
            ),
            (
                "model.layers.0.self_attn.kv_a_layernorm.weight",
                "BF16",
                &[4],
            ),
            (
                "model.layers.0.self_attn.kv_b_proj.weight",
                "BF16",
                &[14, 4],
            ),
            ("model.layers.0.self_attn.o_proj.weight", "BF16", &[16, 6]),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "BF16",
                &[16],
            ),
            ("model.layers.0.mlp.gate_proj.weight", "BF16", &[32, 16]),
            ("model.layers.0.mlp.up_proj.weight", "BF16", &[32, 16]),
            ("model.layers.0.mlp.down_proj.weight", "BF16", &[16, 32]),
            ("model.layers.1.input_layernorm.weight", "BF16", &[16]),
            ("model.layers.1.self_attn.q_a_proj.weight", "BF16", &[4, 16]),
            (
                "model.layers.1.self_attn.q_a_layernorm.weight",
                "BF16",
                &[4],
            ),
            ("model.layers.1.self_attn.q_b_proj.weight", "BF16", &[12, 4]),
            (
                "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
                "BF16",
                &[6, 16],
            ),
            (
                "model.layers.1.self_attn.kv_a_layernorm.weight",
                "BF16",
                &[4],
            ),
            (
                "model.layers.1.self_attn.kv_b_proj.weight",
                "BF16",
                &[14, 4],
            ),
            ("model.layers.1.self_attn.o_proj.weight", "BF16", &[16, 6]),
            (
                "model.layers.1.post_attention_layernorm.weight",
                "BF16",
                &[16],
            ),
            ("model.layers.1.mlp.gate.weight", "BF16", &[3, 16]),
            (
                "model.layers.1.mlp.gate.e_score_correction_bias",
                "BF16",
                &[3],
            ),
            (
                "model.layers.1.mlp.switch_mlp.gate_proj.weight",
                "BF16",
                &[3, 5, 16],
            ),
            (
                "model.layers.1.mlp.switch_mlp.up_proj.weight",
                "BF16",
                &[3, 5, 16],
            ),
            (
                "model.layers.1.mlp.switch_mlp.down_proj.weight",
                "BF16",
                &[3, 16, 5],
            ),
            (
                "model.layers.1.mlp.shared_experts.gate_proj.weight",
                "BF16",
                &[5, 16],
            ),
            (
                "model.layers.1.mlp.shared_experts.up_proj.weight",
                "BF16",
                &[5, 16],
            ),
            (
                "model.layers.1.mlp.shared_experts.down_proj.weight",
                "BF16",
                &[16, 5],
            ),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("DeepSeek V3 conversion should succeed");

    assert_eq!(manifest.model_family, "deepseek_v3");
    assert_eq!(manifest.attention_head_dim, 6);
    assert!(
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::AttentionKvB),
        "raw DeepSeek kv_b_proj should be preserved in the manifest"
    );
    assert!(
        !manifest
            .tensors
            .iter()
            .any(|tensor| tensor.role == NativeTensorRole::AttentionEmbedQ),
        "raw DeepSeek kv_b_proj should not be misreported as embed_q"
    );

    write_manifest(&dir, &manifest).expect("write should succeed");
    crate::model::NativeModelArtifacts::from_dir(&dir)
        .expect("runtime-ready DeepSeek manifest should validate");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn rejects_glm4_moe_lite_rope_scaling_until_scale_contract_is_manifested() {
    validate_glm4_moe_lite_rope_scaling(&serde_json::json!({}))
        .expect("missing rope_scaling should use current GLM scale contract");
    validate_glm4_moe_lite_rope_scaling(&serde_json::json!({ "rope_scaling": null }))
        .expect("null rope_scaling should use current GLM scale contract");

    let error = validate_glm4_moe_lite_rope_scaling(&serde_json::json!({
        "rope_scaling": {
            "factor": 2.0,
            "mscale_all_dim": 1.0
        }
    }))
    .expect_err("GLM rope scaling should fail closed until represented in the manifest");
    let ConvertError::InvalidModelContract {
        model_type,
        message,
    } = error
    else {
        panic!("expected invalid model contract");
    };
    assert_eq!(model_type, "glm4_moe_lite");
    assert!(message.contains("rope_scaling"));
    assert!(message.contains("mscale_all_dim"));
}

#[test]
fn rejects_qwen_rope_scaling_until_runtime_contract_is_manifested() {
    validate_qwen_rope_scaling(&serde_json::json!({}), "qwen3_next")
        .expect("missing rope_scaling should use current Qwen RoPE contract");
    validate_qwen_rope_scaling(
        &serde_json::json!({ "text_config": { "rope_scaling": null } }),
        "qwen3_next",
    )
    .expect("null rope_scaling should use current Qwen RoPE contract");

    let error = validate_qwen_rope_scaling(
        &serde_json::json!({
            "text_config": {
                "rope_scaling": {
                    "type": "longrope",
                    "factor": 4.0
                }
            }
        }),
        "qwen3_next",
    )
    .expect_err("Qwen rope scaling should fail closed until represented in the manifest");
    let ConvertError::InvalidModelContract {
        model_type,
        message,
    } = error
    else {
        panic!("expected invalid model contract");
    };
    assert_eq!(model_type, "qwen3_next");
    assert!(message.contains("rope_scaling"));
    assert!(message.contains("Qwen MLX runtime"));
}

#[test]
fn rejects_glm4_moe_lite_draft_manifest_when_router_bias_is_missing() {
    let dir = unique_test_dir("glm4_missing_router_bias");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "glm4_moe_lite",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "vocab_size": 64,
            "qk_nope_head_dim": 6,
            "qk_rope_head_dim": 2,
            "v_head_dim": 8,
            "q_lora_rank": 8,
            "kv_lora_rank": 8,
            "first_k_dense_replace": 0,
            "n_routed_experts": 4,
            "n_shared_experts": 0,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "routed_scaling_factor": 1.8
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "BF16", &[64, 16]),
            ("model.norm.weight", "BF16", &[16]),
            ("lm_head.weight", "BF16", &[64, 16]),
            ("model.layers.0.input_layernorm.weight", "BF16", &[16]),
            ("model.layers.0.self_attn.q_a_proj.weight", "BF16", &[8, 16]),
            (
                "model.layers.0.self_attn.q_a_layernorm.weight",
                "BF16",
                &[8],
            ),
            ("model.layers.0.self_attn.q_b_proj.weight", "BF16", &[16, 8]),
            (
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                "BF16",
                &[10, 16],
            ),
            (
                "model.layers.0.self_attn.kv_a_layernorm.weight",
                "BF16",
                &[8],
            ),
            ("model.layers.0.self_attn.embed_q.weight", "BF16", &[12, 8]),
            (
                "model.layers.0.self_attn.unembed_out.weight",
                "BF16",
                &[16, 8],
            ),
            ("model.layers.0.self_attn.o_proj.weight", "BF16", &[16, 16]),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "BF16",
                &[16],
            ),
            ("model.layers.0.mlp.gate.weight", "BF16", &[4, 16]),
            (
                "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                "BF16",
                &[4, 8, 16],
            ),
            (
                "model.layers.0.mlp.switch_mlp.up_proj.weight",
                "BF16",
                &[4, 8, 16],
            ),
            (
                "model.layers.0.mlp.switch_mlp.down_proj.weight",
                "BF16",
                &[4, 16, 8],
            ),
        ],
    );

    let error = convert_hf_model_dir(&dir)
        .expect_err("GLM MoE layer without correction bias should fail conversion");

    assert!(
        error.to_string().contains("FfnGateInpCorrectionBias"),
        "{error}"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn rejects_unsupported_model_type() {
    let dir = unique_test_dir("unsupported");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "gpt2",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 2,
            "vocab_size": 32000,
        }),
    );
    write_fake_safetensors(&dir, "model.safetensors", &[]);

    let error = convert_hf_model_dir(&dir).expect_err("gpt2 should be unsupported");
    assert!(matches!(error, ConvertError::UnsupportedModelType { .. }));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn diffusion_gemma_canvas_size_zero_is_rejected() {
    // A malformed manifest with canvas_size=0 must not pass through
    // to NativeDiffusionConfig.canvas_size as Some(0). Regression
    // test for https://github.com/defai-digital/ax-engine/issues/44.
    //
    // We only test parse_diffusion_config directly because the full
    // convert_hf_model_dir path requires many other fields to succeed.
    let config_with_zero = serde_json::json!({
        "model_type": "diffusion_gemma",
        "canvas_size": 0,
        "generation_config": {
            "max_denoising_steps": 48,
            "t_max": 0.8,
            "t_min": 0.4
        }
    });
    let parsed = super::parse_diffusion_config(&config_with_zero, "diffusion_gemma");
    assert_ne!(
        parsed.canvas_size,
        Some(0),
        "canvas_size=0 must be filtered to None by parse_diffusion_config"
    );

    // Also test canvas_length=0 (the alternate key used by real configs).
    let config_with_length_zero = serde_json::json!({
        "model_type": "diffusion_gemma",
        "canvas_length": 0,
        "generation_config": {
            "max_denoising_steps": 48,
            "t_max": 0.8,
            "t_min": 0.4
        }
    });
    let parsed2 = super::parse_diffusion_config(&config_with_length_zero, "diffusion_gemma");
    assert_ne!(
        parsed2.canvas_size,
        Some(0),
        "canvas_length=0 must be filtered to None by parse_diffusion_config"
    );

    // Verify that a valid canvas_size is preserved.
    let config_with_valid = serde_json::json!({
        "model_type": "diffusion_gemma",
        "canvas_length": 256,
        "generation_config": {
            "max_denoising_steps": 48,
            "t_max": 0.8,
            "t_min": 0.4
        }
    });
    let parsed3 = super::parse_diffusion_config(&config_with_valid, "diffusion_gemma");
    assert_eq!(
        parsed3.canvas_size,
        Some(256),
        "valid canvas_size=256 must be preserved"
    );

    // max_denoise_steps=0 must be filtered to None.
    let config_steps_zero = serde_json::json!({
        "model_type": "diffusion_gemma",
        "canvas_length": 256,
        "generation_config": {
            "max_denoising_steps": 0,
            "t_max": 0.8,
            "t_min": 0.4
        }
    });
    let parsed4 = super::parse_diffusion_config(&config_steps_zero, "diffusion_gemma");
    assert_ne!(
        parsed4.max_denoise_steps,
        Some(0),
        "max_denoise_steps=0 must be filtered to None"
    );

    // convergence_steps=0 (via stability_threshold) must be filtered to None.
    let config_conv_zero = serde_json::json!({
        "model_type": "diffusion_gemma",
        "canvas_length": 256,
        "generation_config": {
            "max_denoising_steps": 48,
            "t_max": 0.8,
            "t_min": 0.4,
            "stability_threshold": 0
        }
    });
    let parsed5 = super::parse_diffusion_config(&config_conv_zero, "diffusion_gemma");
    assert_ne!(
        parsed5.convergence_steps,
        Some(0),
        "convergence_steps=0 must be filtered to None"
    );

    // Valid non-zero values must be preserved.
    let config_valid_all = serde_json::json!({
        "model_type": "diffusion_gemma",
        "canvas_length": 256,
        "convergence_steps": 3,
        "generation_config": {
            "max_denoising_steps": 64,
            "t_max": 0.8,
            "t_min": 0.4
        }
    });
    let parsed6 = super::parse_diffusion_config(&config_valid_all, "diffusion_gemma");
    assert_eq!(parsed6.canvas_size, Some(256));
    assert_eq!(parsed6.max_denoise_steps, Some(64));
    assert_eq!(parsed6.convergence_steps, Some(3));
}

#[test]
fn rejects_quantized_dtypes() {
    let dir = unique_test_dir("quantized");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 1,
            "vocab_size": 151936,
        }),
    );
    // "I32" is not a supported safetensors dtype in our converter
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[("model.embed_tokens.weight", "I32", &[151936, 4096])],
    );

    let error = convert_hf_model_dir(&dir).expect_err("I32 dtype should fail");
    assert!(matches!(error, ConvertError::UnsupportedDtype { .. }));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn skips_unrecognised_tensors() {
    let dir = unique_test_dir("extra-tensors");
    write_config(
        &dir,
        serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "vocab_size": 32,
        }),
    );
    write_fake_safetensors(
        &dir,
        "model.safetensors",
        &[
            ("model.embed_tokens.weight", "F32", &[32, 8]),
            ("model.norm.weight", "F32", &[8]),
            ("model.layers.0.input_layernorm.weight", "F32", &[8]),
            ("model.layers.0.self_attn.q_proj.weight", "F32", &[8, 8]),
            ("model.layers.0.self_attn.k_proj.weight", "F32", &[8, 8]),
            ("model.layers.0.self_attn.v_proj.weight", "F32", &[8, 8]),
            ("model.layers.0.self_attn.o_proj.weight", "F32", &[8, 8]),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "F32",
                &[8],
            ),
            (
                "model.layers.0.pre_feedforward_layernorm.weight",
                "F32",
                &[8],
            ),
            ("model.layers.0.mlp.gate_proj.weight", "F32", &[16, 8]),
            ("model.layers.0.mlp.up_proj.weight", "F32", &[16, 8]),
            ("model.layers.0.mlp.down_proj.weight", "F32", &[8, 16]),
            // These should be silently skipped:
            ("model.layers.0.self_attn.rotary_emb.inv_freq", "F32", &[64]),
            ("some.unknown.tensor", "F32", &[100]),
        ],
    );

    let manifest = convert_hf_model_dir(&dir).expect("conversion should succeed");
    assert_eq!(manifest.tensors.len(), 12);

    let names: Vec<_> = manifest.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(!names.contains(&"model.layers.0.self_attn.rotary_emb.inv_freq"));
    assert!(!names.contains(&"some.unknown.tensor"));

    let _ = fs::remove_dir_all(dir);
}

/// Real model integration test — uses `.internal/models/Qwen3.5-2B-bf16` when available.
/// If the model is absent locally, the test exits early without failing.
#[test]
fn converts_real_qwen3_5_bf16_model() {
    let model_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.internal/models/Qwen3.5-2B-bf16");
    if !model_dir.join("config.json").exists() {
        eprintln!("skipping: model not downloaded at {}", model_dir.display());
        return;
    }

    with_real_model_manifest_lock(|| {
        let manifest = convert_hf_model_dir(&model_dir)
            .expect("real Qwen3.5-2B-bf16 conversion should succeed");

        assert_eq!(manifest.model_family, "qwen3_5");
        assert_eq!(manifest.layer_count, 24);
        assert_eq!(manifest.hidden_size, 2048);
        assert_eq!(manifest.attention_head_count, 8);
        assert_eq!(manifest.attention_head_dim, 256);
        assert_eq!(manifest.kv_head_count, 2);
        assert_eq!(manifest.vocab_size, 248320);
        assert!(manifest.tie_word_embeddings);
        assert_eq!(manifest.rope_theta, Some(10000000));

        // Qwen3.5 has mixed layers: only full_attention layers (3,7,11,15,19,23)
        // have self_attn tensors. All 24 layers have FFN + norms.
        let attn_q_layers: Vec<u32> = manifest
            .tensors
            .iter()
            .filter(|t| t.role == NativeTensorRole::AttentionQ)
            .filter_map(|t| t.layer_index)
            .collect();
        assert_eq!(attn_q_layers, vec![3, 7, 11, 15, 19, 23]);

        let ffn_gate_count = manifest
            .tensors
            .iter()
            .filter(|t| t.role == NativeTensorRole::FfnGate)
            .count();
        assert_eq!(ffn_gate_count, 24, "all 24 layers should have FFN gate");

        let norm_count = manifest
            .tensors
            .iter()
            .filter(|t| t.role == NativeTensorRole::AttentionNorm)
            .count();
        assert_eq!(norm_count, 24, "all 24 layers should have attention norm");

        // All tensors should be BF16
        assert!(
            manifest
                .tensors
                .iter()
                .all(|t| t.dtype == NativeTensorDataType::Bf16
                    || t.dtype == NativeTensorDataType::F32)
        );

        // Write manifest, then validate the full NativeModelArtifacts pipeline.
        // Hold the shared lock for the whole duration so parallel metal tests
        // never observe the temporary cleanup window.
        write_manifest(&model_dir, &manifest).expect("write manifest should succeed");
        let artifacts = crate::model::NativeModelArtifacts::from_dir(&model_dir)
            .expect("NativeModelArtifacts should validate the real Qwen3.5 model");
        assert_eq!(artifacts.manifest().layer_count, 24);
        assert_eq!(
            artifacts.summary().tensor_count,
            manifest.tensors.len() as u32
        );

        eprintln!(
            "✓ converted {} tensors, {} layers, family={}",
            manifest.tensors.len(),
            manifest.layer_count,
            manifest.model_family
        );

        // Clean up the generated manifest before releasing the shared lock.
        let _ = fs::remove_file(model_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE));
    });
}
