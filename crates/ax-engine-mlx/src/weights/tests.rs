
use super::*;
use ax_engine_core::NativeTensorDataType;
use mlx_sys::{MlxDtype, zeros};
use std::path::Path;

#[test]
fn flash_next_mtp_unavailable_schedule_never_opens_sidecar() {
    use qwen4_exp::Qwen4ExpTargetSchedule;
    let mut opened = 0;
    let result = load_flash_next_mtp_for_schedule(
        &Qwen4ExpTargetSchedule::Unavailable("unclassified MXFP4".into()),
        || {
            opened += 1;
            Ok(())
        },
    );
    assert!(matches!(result, Err(WeightLoadError::InvalidLayer(_))));
    assert_eq!(opened, 0);
    for schedule in [
        Qwen4ExpTargetSchedule::CanonicalSingleton,
        Qwen4ExpTargetSchedule::LegacyBatched,
        Qwen4ExpTargetSchedule::LegacyBlock,
    ] {
        let result = load_flash_next_mtp_for_schedule(&schedule, || {
            opened += 1;
            Err::<(), _>(WeightLoadError::FileMissing(
                "retained sidecar failure".into(),
            ))
        });
        assert!(matches!(result, Err(WeightLoadError::FileMissing(_))));
    }
    assert_eq!(opened, 3);
}

#[test]
fn flash_next_mtp_sidecar_present_requires_mtp_safetensors() {
    let dir = std::env::temp_dir().join(format!(
        "ax-flash-next-mtp-sidecar-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    assert!(!flash_next_mtp_sidecar_present(&dir));
    std::fs::write(dir.join(FLASH_NEXT_MTP_SIDECAR_FILE), b"stub").expect("sidecar");
    assert!(flash_next_mtp_sidecar_present(&dir));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mmap_weights_env_requires_a_nonzero_nonempty_value() {
    assert!(!mmap_weights_env_value_enabled(None));
    assert!(!mmap_weights_env_value_enabled(Some("")));
    assert!(!mmap_weights_env_value_enabled(Some("0")));
    assert!(mmap_weights_env_value_enabled(Some("1")));
    assert!(mmap_weights_env_value_enabled(Some("true")));
}

#[test]
fn skip_vision_sidecar_from_env_is_opt_in() {
    assert!(!skip_vision_sidecar_from_env(None));
    assert!(!skip_vision_sidecar_from_env(Some("")));
    assert!(!skip_vision_sidecar_from_env(Some("0")));
    assert!(skip_vision_sidecar_from_env(Some("1")));
    assert!(skip_vision_sidecar_from_env(Some("true")));
}

#[test]
fn skip_mtp_sidecar_from_env_is_opt_in() {
    assert!(!skip_mtp_sidecar_from_env(None));
    assert!(!skip_mtp_sidecar_from_env(Some("0")));
    assert!(skip_mtp_sidecar_from_env(Some("1")));
    assert!(skip_mtp_sidecar_from_env(Some("TRUE")));
}

#[test]
fn draft_lm_head_spec_matches_mlx_affine_contract() {
    for bits in [2, 3, 4, 5, 6, 8] {
        for group_size in [32, 64, 128] {
            assert_eq!(
                valid_draft_lm_head_spec(bits, group_size),
                Some(DraftLmHeadSpec { bits, group_size })
            );
        }
    }

    for bits in [-1, 0, 1, 7, 9] {
        assert_eq!(valid_draft_lm_head_spec(bits, 64), None);
    }
    for group_size in [-1, 0, 1, 16, 33, 256] {
        assert_eq!(valid_draft_lm_head_spec(4, group_size), None);
    }
}

#[test]
fn runtime_draft_lm_head_spec_rejects_out_of_range_integers() {
    let valid = serde_json::json!({
        "recommended_draft_lm_head": {"mode": "affine", "bits": 4, "group_size": 64}
    });
    assert_eq!(
        draft_lm_head_spec_from_runtime(&valid),
        Some(DraftLmHeadSpec {
            bits: 4,
            group_size: 64,
        })
    );

    // Both oversized values wrapped to supported 4/64 settings with the
    // former `as i32` conversion.
    for invalid in [
        serde_json::json!({
            "recommended_draft_lm_head": {
                "mode": "affine",
                "bits": i64::from(u32::MAX) + 5,
                "group_size": 64,
            }
        }),
        serde_json::json!({
            "recommended_draft_lm_head": {
                "mode": "affine",
                "bits": 4,
                "group_size": i64::from(u32::MAX) + 65,
            }
        }),
    ] {
        assert_eq!(draft_lm_head_spec_from_runtime(&invalid), None);
    }
}

#[test]
fn draft_lm_head_requantization_rejects_unaligned_dense_shape() {
    let lm_head = QuantizedWeight::new(zeros(&[8, 65], MlxDtype::Bfloat16, None), None, None);

    assert!(
        build_draft_lm_head(
            &lm_head,
            DraftLmHeadSpec {
                bits: 4,
                group_size: 64,
            },
        )
        .is_none()
    );
}

#[test]
fn draft_lm_head_dequantization_preserves_source_mode_contract() {
    let mut mxfp4 = QuantizedWeight::new(
        zeros(&[8, 8], MlxDtype::Uint32, None),
        Some(zeros(&[8, 2], MlxDtype::Bfloat16, None)),
        None,
    );
    mxfp4.group_size = 32;
    mxfp4.bits = 4;
    mxfp4.mode = "mxfp4".to_string();

    let (mode, biases) = draft_lm_head_dequantization_contract(&mxfp4);
    assert_eq!(mode, MlxQuantizationMode::Mxfp4);
    assert!(biases.is_none());

    let mut affine = mxfp4;
    affine.mode = "affine".to_string();
    affine.biases = Some(zeros(&[8, 2], MlxDtype::Bfloat16, None));
    let (mode, biases) = draft_lm_head_dequantization_contract(&affine);
    assert_eq!(mode, MlxQuantizationMode::Affine);
    assert!(biases.is_some());
}

#[test]
fn draft_lm_head_requantizes_mxfp_sources_without_affine_biases() {
    let dense = astype(
        &zeros(&[8, 64], MlxDtype::Float32, None),
        MlxDtype::Bfloat16,
        None,
    );
    for (source_mode, source_bits, mode_name) in [
        (MlxQuantizationMode::Mxfp4, 4, "mxfp4"),
        (MlxQuantizationMode::Mxfp8, 8, "mxfp8"),
    ] {
        let mut parts = quantize(&dense, Some(32), Some(source_bits), source_mode, None, None);
        assert_eq!(parts.len(), 2);
        let weight = parts.remove(0);
        let scales = parts.remove(0);
        let mut source = QuantizedWeight::new(weight, Some(scales), None);
        source.group_size = 32;
        source.bits = source_bits;
        source.mode = mode_name.to_string();

        let draft = build_draft_lm_head(
            &source,
            DraftLmHeadSpec {
                bits: 4,
                group_size: 64,
            },
        )
        .expect("MXFP source should dequantize with its own mode before affine requantization");
        assert_eq!(draft.mode, "affine");
        assert_eq!(draft.bits, 4);
        assert_eq!(draft.group_size, 64);
        assert!(draft.biases.is_some());
    }
}

#[test]
fn auto_buffer_caps_exclude_unlimited_ocr_aliases() {
    for family in ["unlimited_ocr", "unlimited-ocr", "deepseekocr"] {
        assert!(
            !auto_buffer_caps_supported_for_family(family),
            "{family} must retain MLX 0.32 defaults on the first Metal initialization"
        );
    }
}

#[test]
fn auto_buffer_caps_keep_proven_moe_families_enabled() {
    for family in ["qwen3_next", "qwen3"] {
        assert!(
            auto_buffer_caps_supported_for_family(family),
            "{family} must retain the measured gather-QMM overlap optimization"
        );
    }
}

#[test]
fn auto_buffer_caps_exclude_qwen3_5_family() {
    // Server-path A/B on Qwen3.6-35B-A3B measures the raise as a one-way
    // prefill degradation with no decode win; see the doc comment on
    // auto_buffer_caps_supported_for_family.
    assert!(!auto_buffer_caps_supported_for_family("qwen3_5"));
}

#[test]
fn auto_buffer_caps_na_host_raises_only_qwen3_next() {
    assert!(should_auto_raise_metal_buffer_caps_for(
        true,
        true,
        true,
        "qwen3_next"
    ));
    assert!(!should_auto_raise_metal_buffer_caps_for(
        true, true, true, "llama3"
    ));
    assert!(!should_auto_raise_metal_buffer_caps_for(
        true,
        true,
        true,
        "glm4_moe_lite"
    ));
    let _m5 = crate::hardware::override_hardware(crate::hardware::HardwareCapabilities::m5_na());
    assert!(should_auto_raise_metal_buffer_caps("qwen3_next"));
    assert!(!should_auto_raise_metal_buffer_caps("llama3"));
    assert!(!should_auto_raise_metal_buffer_caps("qwen3_5"));
}

#[test]
fn auto_buffer_caps_pre_m5_still_raises_eligible_families() {
    let _m4 = crate::hardware::override_hardware(crate::hardware::HardwareCapabilities::m4());
    assert!(should_auto_raise_metal_buffer_caps("llama3"));
    assert!(should_auto_raise_metal_buffer_caps("qwen3_next"));
    assert!(!should_auto_raise_metal_buffer_caps("qwen3_5"));
}

#[test]
fn auto_buffer_caps_m5_without_macos_26_2_keeps_pre_m5_policy() {
    let _hw =
        crate::hardware::override_hardware(crate::hardware::HardwareCapabilities::m5_old_macos());
    assert!(should_auto_raise_metal_buffer_caps("llama3"));
    assert!(should_auto_raise_metal_buffer_caps("qwen3_next"));
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn live_auto_buffer_caps_follow_detected_hardware() {
    if !crate::fastpath::auto_buffer_caps_enabled() {
        return;
    }
    let active = crate::hardware::neural_accelerator_active();
    eprintln!(
        "live auto_buffer_caps: na_active={} qwen3_next={} llama3={}",
        active,
        should_auto_raise_metal_buffer_caps("qwen3_next"),
        should_auto_raise_metal_buffer_caps("llama3"),
    );
    assert!(should_auto_raise_metal_buffer_caps("qwen3_next"));
    if active {
        assert!(
            !should_auto_raise_metal_buffer_caps("llama3"),
            "M5+ NA hosts raise command-buffer caps for qwen3_next only"
        );
    } else {
        assert!(
            should_auto_raise_metal_buffer_caps("llama3"),
            "pre-M5 hosts still raise eligible families"
        );
    }
}

fn spec(role: NativeTensorRole) -> NativeTensorSpec {
    NativeTensorSpec {
        name: format!("{role:?}"),
        role,
        layer_index: Some(0),
        dtype: NativeTensorDataType::Bf16,
        source_tensor_type: None,
        source_quantized: false,
        quantization: None,
        quantized_source: None,
        shape: vec![1],
        file: PathBuf::from("model.safetensors"),
        offset_bytes: 0,
        length_bytes: 2,
    }
}

#[test]
fn pipeline_file_selection_excludes_other_layers_and_endpoint_tensors() {
    let mut embedding = spec(NativeTensorRole::TokenEmbedding);
    embedding.layer_index = None;
    embedding.file = PathBuf::from("embedding.safetensors");
    let mut layer0 = spec(NativeTensorRole::AttentionQ);
    layer0.layer_index = Some(0);
    layer0.file = PathBuf::from("layer-0.safetensors");
    let mut layer1 = spec(NativeTensorRole::AttentionQ);
    layer1.layer_index = Some(1);
    layer1.file = PathBuf::from("layer-1.safetensors");
    let mut final_norm = spec(NativeTensorRole::FinalNorm);
    final_norm.layer_index = None;
    final_norm.file = PathBuf::from("head.safetensors");
    let mut lm_head = spec(NativeTensorRole::LmHead);
    lm_head.layer_index = None;
    lm_head.file = PathBuf::from("head.safetensors");
    let specs = vec![embedding, layer0, layer1, final_norm, lm_head];

    let first = PipelineRankAssignment {
        rank: 0,
        node_identity_digest: "node-a".into(),
        layers: ax_engine_core::PipelineLayerRange { start: 0, end: 1 },
        owns_embeddings: true,
        owns_output_head: false,
    };
    assert_eq!(
        pipeline_stage_files(&specs, &first, false),
        [
            PathBuf::from("embedding.safetensors"),
            PathBuf::from("layer-0.safetensors")
        ]
        .into_iter()
        .collect()
    );

    let last = PipelineRankAssignment {
        rank: 1,
        node_identity_digest: "node-b".into(),
        layers: ax_engine_core::PipelineLayerRange { start: 1, end: 2 },
        owns_embeddings: false,
        owns_output_head: true,
    };
    assert_eq!(
        pipeline_stage_files(&specs, &last, false),
        [
            PathBuf::from("head.safetensors"),
            PathBuf::from("layer-1.safetensors")
        ]
        .into_iter()
        .collect()
    );
}

#[test]
fn attention_layout_detects_linear_attention_without_full_attention_roles() {
    let specs = vec![spec(NativeTensorRole::LinearAttentionInProjQkv)];

    let layout = attention_layout_for_layer(&specs, Some(0)).expect("layout should resolve");

    assert_eq!(layout, AttentionLayout::Linear);
}

#[test]
fn attention_layout_defaults_to_full_attention() {
    let specs = vec![spec(NativeTensorRole::AttentionO)];

    let layout = attention_layout_for_layer(&specs, Some(0)).expect("layout should resolve");

    assert_eq!(layout, AttentionLayout::Full);
}

#[test]
fn attention_layout_rejects_mixed_attention_families() {
    let specs = vec![
        spec(NativeTensorRole::AttentionO),
        spec(NativeTensorRole::LinearAttentionInProjQkv),
    ];

    let error = attention_layout_for_layer(&specs, Some(0))
        .expect_err("mixed attention families should fail");

    assert!(matches!(error, WeightLoadError::InvalidLayer(_)));
}

#[test]
fn gemma4_unified_vision_keeps_patch_dense_bias_outside_quantized_weight() {
    let roles = [
        (
            "patch_ln1_weight",
            NativeTensorRole::Gemma4UnifiedVisionPatchNorm1,
        ),
        (
            "patch_ln1_bias",
            NativeTensorRole::Gemma4UnifiedVisionPatchNorm1Bias,
        ),
        (
            "patch_dense.weight",
            NativeTensorRole::Gemma4UnifiedVisionPatchDense,
        ),
        (
            "patch_dense.bias",
            NativeTensorRole::Gemma4UnifiedVisionPatchDenseBias,
        ),
        (
            "patch_ln2_weight",
            NativeTensorRole::Gemma4UnifiedVisionPatchNorm2,
        ),
        (
            "patch_ln2_bias",
            NativeTensorRole::Gemma4UnifiedVisionPatchNorm2Bias,
        ),
        (
            "pos_embedding",
            NativeTensorRole::Gemma4UnifiedVisionPositionEmbedding,
        ),
        (
            "pos_norm_weight",
            NativeTensorRole::Gemma4UnifiedVisionPositionNorm,
        ),
        (
            "pos_norm_bias",
            NativeTensorRole::Gemma4UnifiedVisionPositionNormBias,
        ),
        (
            "projection",
            NativeTensorRole::Gemma4UnifiedVisionProjection,
        ),
    ];
    let specs = roles
        .iter()
        .map(|(name, role)| NativeTensorSpec {
            name: (*name).to_string(),
            role: *role,
            layer_index: None,
            dtype: NativeTensorDataType::Bf16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: vec![1],
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 2,
        })
        .collect::<Vec<_>>();
    let mut name_map = roles
        .iter()
        .map(|(name, _)| (name.to_string(), zeros(&[1], MlxDtype::Bfloat16, None)))
        .collect::<HashMap<_, _>>();

    let weights = load_gemma4_unified_vision_weights(&specs, &mut name_map)
        .expect("vision weights should load")
        .expect("vision roles should enable the unified vision path");

    assert!(weights.patch_dense.linear_bias.is_none());
    assert_eq!(weights.patch_dense_bias.shape(), vec![1]);
    assert!(name_map.is_empty());
}

#[test]
fn small_linear_attention_gated_norm_is_allowed() {
    let norm = zeros(&[8], MlxDtype::Float32, None);

    assert_eq!(
        norm_mean_abs(&norm),
        Some(0.0),
        "Qwen3-Next gated linear-attention norms may be trained near zero"
    );
}

#[test]
fn hf_layout_conv1d_is_rejected() {
    // HuggingFace stores conv1d as [conv_dim, in=1, kernel]. A manifest
    // that mis-declares weight_sanitize would skip the axis swap, producing
    // silently wrong conv outputs. The check must catch this at load time.
    let conv1d = zeros(&[64, 1, 4], MlxDtype::Float32, None);

    let error = ensure_conv1d_mlx_layout(3, &conv1d)
        .expect_err("HF-layout conv1d [conv_dim, 1, kernel] must be rejected");

    let WeightLoadError::UnsanitizedWeights(message) = error else {
        panic!("expected unsanitized weights error");
    };
    assert!(message.contains("layer 3"));
    assert!(message.contains("[64, 1, 4]"));
    assert!(message.contains("mlx_lm.convert"));
}

#[test]
fn mlx_layout_conv1d_is_accepted() {
    // MLX layout: [conv_dim, kernel, in=1]. Both the HfToMlx and HfNormOnly
    // sanitization paths produce this shape; the check must allow it.
    let conv1d = zeros(&[64, 4, 1], MlxDtype::Float32, None);

    ensure_conv1d_mlx_layout(0, &conv1d)
        .expect("MLX-layout conv1d [conv_dim, kernel, 1] should load");
}

#[test]
fn normalize_f32_pack_casts_only_f32_for_qwen_class() {
    let mut map = HashMap::new();
    map.insert(
        "layers.0.input_layernorm.weight".to_string(),
        zeros(&[4], MlxDtype::Float32, None),
    );
    map.insert(
        "layers.0.self_attn.q_proj.scales".to_string(),
        zeros(&[4, 2], MlxDtype::Float32, None),
    );
    map.insert(
        "layers.0.self_attn.q_proj.weight".to_string(),
        zeros(&[4, 2], MlxDtype::Uint32, None),
    );
    map.insert(
        "layers.0.mlp.gate_proj.scales".to_string(),
        zeros(&[4, 2], MlxDtype::Bfloat16, None),
    );

    let mut quantized = spec(NativeTensorRole::AttentionQ);
    quantized.quantization = Some(NativeTensorQuantization::default());
    let specs = [quantized];

    // Non-qwen family: untouched even when it is quantized.
    let mut other = map.clone();
    normalize_f32_pack_to_bf16("gemma4", &specs, &mut other);
    assert_eq!(
        other["layers.0.input_layernorm.weight"].dtype(),
        MlxDtype::Float32
    );

    normalize_f32_pack_to_bf16("qwen3_5", &specs, &mut map);
    assert_eq!(
        map["layers.0.input_layernorm.weight"].dtype(),
        MlxDtype::Bfloat16,
        "f32 norms must normalize to bf16"
    );
    assert_eq!(
        map["layers.0.self_attn.q_proj.scales"].dtype(),
        MlxDtype::Bfloat16,
        "f32 scales must normalize to bf16"
    );
    assert_eq!(
        map["layers.0.self_attn.q_proj.weight"].dtype(),
        MlxDtype::Uint32,
        "quantized integer payloads stay untouched"
    );
    assert_eq!(
        map["layers.0.mlp.gate_proj.scales"].dtype(),
        MlxDtype::Bfloat16
    );
}

#[test]
fn normalize_f32_pack_preserves_dense_f32_qwen_checkpoints() {
    let dense_specs = [spec(NativeTensorRole::AttentionQ)];
    let mut map = HashMap::from([
        (
            "layers.0.self_attn.q_proj.weight".to_string(),
            zeros(&[4, 4], MlxDtype::Float32, None),
        ),
        (
            "layers.0.input_layernorm.weight".to_string(),
            zeros(&[4], MlxDtype::Float32, None),
        ),
    ]);

    normalize_f32_pack_to_bf16("qwen3_5", &dense_specs, &mut map);

    assert!(
        map.values()
            .all(|tensor| tensor.dtype() == MlxDtype::Float32),
        "an unquantized F32 checkpoint must not be silently downcast"
    );
}

#[test]
fn apply_hf_sanitize_transforms_lifts_norm_deltas_and_swaps_conv1d_axes() {
    // A raw HuggingFace checkpoint stores norm weights as zero-centered
    // deltas (so the "weight = 1.0 + delta" multiplier is materialised
    // by the runtime forward path). The sanitizer must restore the +1.0
    // baseline before loading.
    let delta = [-0.1_f32, 0.2, 0.05, 0.0];
    let norm_delta = MlxArray::from_raw_data(
        delta.as_ptr().cast(),
        std::mem::size_of_val(&delta),
        &[delta.len() as i32],
        MlxDtype::Float32,
    );

    // Conv1d weight in HF axis order (out, in, kernel) = (2, 3, 4).
    // Encode coordinates into values: data[o, i, k] = 100*o + 10*i + k.
    // After moveaxis(2, 1) MLX expects (out, kernel, in) = (2, 4, 3),
    // and the value at new[o, k, i] must equal 100*o + 10*i + k.
    const OUT_DIM: usize = 2;
    const IN_DIM: usize = 3;
    const KERNEL_DIM: usize = 4;
    let mut conv = [0.0_f32; OUT_DIM * IN_DIM * KERNEL_DIM];
    for o in 0..OUT_DIM {
        for i in 0..IN_DIM {
            for k in 0..KERNEL_DIM {
                conv[o * IN_DIM * KERNEL_DIM + i * KERNEL_DIM + k] = (100 * o + 10 * i + k) as f32;
            }
        }
    }
    let conv1d_hf = MlxArray::from_raw_data(
        conv.as_ptr().cast(),
        std::mem::size_of_val(&conv),
        &[OUT_DIM as i32, IN_DIM as i32, KERNEL_DIM as i32],
        MlxDtype::Float32,
    );

    let mut name_map: HashMap<String, MlxArray> = HashMap::new();
    name_map.insert("layers.0.attn_norm".to_string(), norm_delta);
    name_map.insert("layers.0.conv1d".to_string(), conv1d_hf);
    name_map.insert(
        "layers.0.linear_attn.norm".to_string(),
        MlxArray::from_raw_data(
            delta.as_ptr().cast(),
            std::mem::size_of_val(&delta),
            &[delta.len() as i32],
            MlxDtype::Float32,
        ),
    );

    fn make_spec(name: &str, role: NativeTensorRole) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index: Some(0),
            dtype: NativeTensorDataType::F32,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: vec![1],
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 4,
        }
    }
    let specs = vec![
        make_spec("layers.0.attn_norm", NativeTensorRole::AttentionNorm),
        make_spec(
            "layers.0.linear_attn.norm",
            NativeTensorRole::LinearAttentionNorm,
        ),
        make_spec("layers.0.conv1d", NativeTensorRole::LinearAttentionConv1d),
    ];

    apply_hf_sanitize_transforms(&specs, &mut name_map, true, true);

    let sanitized_norm = name_map
        .get("layers.0.attn_norm")
        .expect("norm tensor must still be present");
    let norm_values = sanitized_norm.data_f32();
    for (got, want) in norm_values.iter().zip([0.9_f32, 1.2, 1.05, 1.0].iter()) {
        assert!(
            (got - want).abs() < 1e-6,
            "norm sanitize: got {got}, want {want}"
        );
    }
    let linear_norm = name_map
        .get("layers.0.linear_attn.norm")
        .expect("linear-attention norm tensor must still be present");
    for (got, want) in linear_norm.data_f32().iter().zip(delta.iter()) {
        assert!(
            (got - want).abs() < 1e-6,
            "linear-attention gated norm must not be lifted: got {got}, want {want}"
        );
    }

    let sanitized_conv = name_map
        .get("layers.0.conv1d")
        .expect("conv1d tensor must still be present");
    assert_eq!(
        sanitized_conv.shape(),
        vec![OUT_DIM as i32, KERNEL_DIM as i32, IN_DIM as i32],
        "conv1d axes should swap from (out, in, kernel) to (out, kernel, in)"
    );
    // Verify every coordinate: new[o, k, i] must equal the encoded
    // coordinate 100*o + 10*i + k (note: encoding uses original axis
    // assignments, so the value identifies the source element).
    let conv_values = sanitized_conv.data_f32();
    for o in 0..OUT_DIM {
        for k in 0..KERNEL_DIM {
            for i in 0..IN_DIM {
                let flat = o * KERNEL_DIM * IN_DIM + k * IN_DIM + i;
                let want = (100 * o + 10 * i + k) as f32;
                let got = conv_values[flat];
                assert!(
                    (got - want).abs() < 1e-6,
                    "transposed conv[o={o}, k={k}, i={i}] at flat[{flat}]: got {got}, want {want}"
                );
            }
        }
    }
}

#[test]
fn apply_hf_sanitize_transforms_hf_norm_only_lifts_norm_but_preserves_conv1d_axes() {
    // Qwen3-Coder-Next ships with conv1d already in MLX layout (out, kernel, in)
    // but RMSNorm weights are still HF-style zero-centred deltas. The
    // HfNormOnly path must add +1.0 to norms without swapping conv1d axes.
    const OUT: usize = 2;
    const KERNEL: usize = 4;
    const IN: usize = 1;

    let norm_data = [-0.1_f32, 0.2, 0.05, 0.0];
    let norm = MlxArray::from_raw_data(
        norm_data.as_ptr().cast(),
        std::mem::size_of_val(&norm_data),
        &[norm_data.len() as i32],
        MlxDtype::Float32,
    );

    // Conv1d already in MLX layout (out, kernel, in) = (2, 4, 1)
    let conv_mlx_data: Vec<f32> = (0..OUT * KERNEL * IN).map(|i| i as f32).collect();
    let conv_mlx = MlxArray::from_raw_data(
        conv_mlx_data.as_ptr().cast(),
        std::mem::size_of_val(conv_mlx_data.as_slice()),
        &[OUT as i32, KERNEL as i32, IN as i32],
        MlxDtype::Float32,
    );

    let mut name_map: HashMap<String, MlxArray> = HashMap::new();
    name_map.insert("attn_norm".to_string(), norm);
    name_map.insert("conv1d".to_string(), conv_mlx);
    name_map.insert(
        "linear_attn_norm".to_string(),
        MlxArray::from_raw_data(
            norm_data.as_ptr().cast(),
            std::mem::size_of_val(&norm_data),
            &[norm_data.len() as i32],
            MlxDtype::Float32,
        ),
    );

    fn make_spec(name: &str, role: NativeTensorRole) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index: Some(0),
            dtype: NativeTensorDataType::F32,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: vec![1],
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 4,
        }
    }
    let specs = vec![
        make_spec("attn_norm", NativeTensorRole::AttentionNorm),
        make_spec("linear_attn_norm", NativeTensorRole::LinearAttentionNorm),
        make_spec("conv1d", NativeTensorRole::LinearAttentionConv1d),
    ];

    apply_hf_sanitize_transforms(&specs, &mut name_map, false, true);

    let norm_out = name_map.get("attn_norm").expect("attn_norm present");
    for (got, want) in norm_out
        .data_f32()
        .iter()
        .zip([0.9_f32, 1.2, 1.05, 1.0].iter())
    {
        assert!((got - want).abs() < 1e-5, "norm: got {got}, want {want}");
    }

    let conv_out = name_map.get("conv1d").expect("conv1d present");
    assert_eq!(
        conv_out.shape(),
        vec![OUT as i32, KERNEL as i32, IN as i32],
        "HfNormOnly must NOT swap conv1d axes — they are already in MLX layout"
    );
    for (i, (got, want)) in conv_out
        .data_f32()
        .iter()
        .zip(conv_mlx_data.iter())
        .enumerate()
    {
        assert!(
            (got - want).abs() < 1e-5,
            "conv1d[{i}]: got {got}, want {want}"
        );
    }
    let linear_norm_out = name_map
        .get("linear_attn_norm")
        .expect("linear_attn_norm present");
    for (got, want) in linear_norm_out.data_f32().iter().zip(norm_data.iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "linear_attn_norm must remain raw: got {got}, want {want}"
        );
    }
}

/// Build a minimal `name_map` + spec list mimicking the layer-0 slice of
/// a hybrid (linear-attention) checkpoint, for auto-detection tests.
fn fixture_layer0_linear_attention(
    norm_data: &[f32],
    linear_attn_norm_data: &[f32],
    conv1d_shape: &[i32],
) -> (Vec<NativeTensorSpec>, HashMap<String, MlxArray>) {
    let norm = MlxArray::from_raw_data(
        norm_data.as_ptr().cast(),
        std::mem::size_of_val(norm_data),
        &[norm_data.len() as i32],
        MlxDtype::Float32,
    );
    let linear_attn_norm = MlxArray::from_raw_data(
        linear_attn_norm_data.as_ptr().cast(),
        std::mem::size_of_val(linear_attn_norm_data),
        &[linear_attn_norm_data.len() as i32],
        MlxDtype::Float32,
    );
    let conv_elements: i32 = conv1d_shape.iter().product();
    let conv_data = vec![0.0_f32; conv_elements as usize];
    let conv1d = MlxArray::from_raw_data(
        conv_data.as_ptr().cast(),
        std::mem::size_of_val(conv_data.as_slice()),
        conv1d_shape,
        MlxDtype::Float32,
    );
    let mut name_map = HashMap::new();
    name_map.insert("layers.0.attn_norm".to_string(), norm);
    name_map.insert(
        "layers.0.linear_attn.gated_norm".to_string(),
        linear_attn_norm,
    );
    name_map.insert("layers.0.linear_attn.conv1d".to_string(), conv1d);

    let make_spec = |name: &str, role: NativeTensorRole| NativeTensorSpec {
        name: name.to_string(),
        role,
        layer_index: Some(0),
        dtype: NativeTensorDataType::F32,
        source_tensor_type: None,
        source_quantized: false,
        quantization: None,
        quantized_source: None,
        shape: vec![1],
        file: PathBuf::from("model.safetensors"),
        offset_bytes: 0,
        length_bytes: 4,
    };
    let specs = vec![
        make_spec("layers.0.attn_norm", NativeTensorRole::AttentionNorm),
        make_spec(
            "layers.0.linear_attn.gated_norm",
            NativeTensorRole::LinearAttentionNorm,
        ),
        make_spec(
            "layers.0.linear_attn.conv1d",
            NativeTensorRole::LinearAttentionConv1d,
        ),
    ];
    (specs, name_map)
}

#[test]
fn auto_detect_picks_hf_norm_only_for_unsanitized_norm_with_mlx_conv1d() {
    // Raw ordinary RMSNorm weights are zero-centred deltas, while
    // linear_attn.norm is a trained gated scale that should not drive the
    // sanitize decision.
    let norm_data: Vec<f32> = (0..256).map(|i| 0.01 * ((i as f32).sin())).collect();
    let gated_norm_data: Vec<f32> = vec![0.011; 256];
    let (specs, name_map) =
        fixture_layer0_linear_attention(&norm_data, &gated_norm_data, &[64, 4, 1]);

    let chosen = auto_detect_weight_sanitize("qwen3_next", &specs, &name_map);

    assert_eq!(
        chosen,
        WeightSanitize::HfNormOnly,
        "unsanitized norm + MLX-layout conv1d ⇒ HfNormOnly"
    );
}

#[test]
fn auto_detect_picks_hf_to_mlx_when_both_norm_and_conv1d_are_raw_hf() {
    // Raw HF safetensors path: norm is zero-centred deltas AND conv1d
    // is in HF layout `[conv_dim, in=1, kernel]`.
    let norm_data: Vec<f32> = (0..256).map(|i| 0.01 * ((i as f32).cos())).collect();
    let gated_norm_data: Vec<f32> = vec![0.011; 256];
    let (specs, name_map) =
        fixture_layer0_linear_attention(&norm_data, &gated_norm_data, &[64, 1, 4]);

    let chosen = auto_detect_weight_sanitize("qwen3_next", &specs, &name_map);

    assert_eq!(
        chosen,
        WeightSanitize::HfToMlx,
        "raw HF norm + HF conv1d ⇒ HfToMlx"
    );
}

#[test]
fn auto_detect_returns_none_when_weights_already_sanitized() {
    // Pre-sanitized norm clusters near 1.0; conv1d in MLX layout.
    let norm_data = vec![1.0_f32; 256];
    let gated_norm_data = vec![0.011_f32; 256];
    let (specs, name_map) =
        fixture_layer0_linear_attention(&norm_data, &gated_norm_data, &[64, 4, 1]);

    let chosen = auto_detect_weight_sanitize("qwen3_next", &specs, &name_map);

    assert_eq!(chosen, WeightSanitize::None);
}

#[test]
fn auto_detect_returns_none_for_non_hybrid_models() {
    // GLM MLA has small q/kv adapter RMSNorms whose trained values
    // legitimately cluster near zero. Adapter norms are excluded from the
    // detection sample, so an MLA-only spec set yields no signal and no
    // sanitize transform.
    let norm_data = vec![0.017_f32; 128];
    let norm = MlxArray::from_raw_data(
        norm_data.as_ptr().cast(),
        std::mem::size_of_val(norm_data.as_slice()),
        &[norm_data.len() as i32],
        MlxDtype::Float32,
    );
    let mut name_map: HashMap<String, MlxArray> = HashMap::new();
    name_map.insert("layers.0.self_attn.kv_a_layernorm".to_string(), norm);
    let specs = vec![NativeTensorSpec {
        name: "layers.0.self_attn.kv_a_layernorm".to_string(),
        role: NativeTensorRole::AttentionKvANorm,
        layer_index: Some(0),
        dtype: NativeTensorDataType::F32,
        source_tensor_type: None,
        source_quantized: false,
        quantization: None,
        quantized_source: None,
        shape: vec![128],
        file: PathBuf::from("model.safetensors"),
        offset_bytes: 0,
        length_bytes: 512,
    }];

    let chosen = auto_detect_weight_sanitize("glm4_moe_lite", &specs, &name_map);

    assert_eq!(chosen, WeightSanitize::None);
}

#[test]
fn auto_detect_returns_none_for_sanitized_norm_with_hf_conv1d() {
    // Partially-transformed checkpoint (norm OK, conv1d not). Don't
    // silently re-sanitize — let `ensure_conv1d_mlx_layout` fire with
    // its specific diagnostic so the user sees the actual inconsistency.
    let norm_data = vec![1.0_f32; 256];
    let gated_norm_data = vec![0.011_f32; 256];
    let (specs, name_map) =
        fixture_layer0_linear_attention(&norm_data, &gated_norm_data, &[64, 1, 4]);

    let chosen = auto_detect_weight_sanitize("qwen3_next", &specs, &name_map);

    assert_eq!(chosen, WeightSanitize::None);
}

/// Dense (non-hybrid) fixture: one layer of ordinary block norms plus the
/// final norm, mirroring a dense checkpoint's block-level RMSNorm tensors
/// (used for both Gemma-family and non-Gemma family cases).
fn fixture_dense_norms(norm_data: &[f32]) -> (Vec<NativeTensorSpec>, HashMap<String, MlxArray>) {
    let make_norm = || {
        MlxArray::from_raw_data(
            norm_data.as_ptr().cast(),
            std::mem::size_of_val(norm_data),
            &[norm_data.len() as i32],
            MlxDtype::Float32,
        )
    };
    let make_spec =
        |name: &str, role: NativeTensorRole, layer_index: Option<u32>| NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index,
            dtype: NativeTensorDataType::F32,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: vec![norm_data.len() as u64],
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: (norm_data.len() * 4) as u64,
        };
    let mut name_map = HashMap::new();
    for name in ["layers.0.attn_norm", "layers.0.ffn_norm", "final_norm"] {
        name_map.insert(name.to_string(), make_norm());
    }
    let specs = vec![
        make_spec(
            "layers.0.attn_norm",
            NativeTensorRole::AttentionNorm,
            Some(0),
        ),
        make_spec("layers.0.ffn_norm", NativeTensorRole::FfnNorm, Some(0)),
        make_spec("final_norm", NativeTensorRole::FinalNorm, None),
    ];
    (specs, name_map)
}

#[test]
fn auto_detect_picks_hf_norm_only_for_raw_hf_dense_model() {
    // Raw HF dense Gemma checkpoint: HF stores zero-centered gamma deltas
    // and there are no conv1d tensors at all, so the transform is the norm
    // lift only. (Dense Qwen/Llama families do not use zero-centered norms
    // and are gated out before the probe — see the regression test below.)
    let norm_data: Vec<f32> = (0..256).map(|i| 0.01 * ((i as f32).sin())).collect();
    let (specs, name_map) = fixture_dense_norms(&norm_data);

    let chosen = auto_detect_weight_sanitize("gemma4", &specs, &name_map);

    assert_eq!(
        chosen,
        WeightSanitize::HfNormOnly,
        "unsanitized dense norms + no conv1d ⇒ HfNormOnly"
    );
}

#[test]
fn auto_detect_returns_none_for_dense_qwen_with_small_trained_norms() {
    // Regression: real mlx-community Qwen3-4B block norms are fully
    // sanitized yet average |w| ≈ 0.02 — trained RMSNorm weights carry no
    // zero-centered signal for non-Gemma dense families. The probe must be
    // skipped entirely; lifting these norms again corrupts every layer.
    let norm_data = vec![0.024_f32; 256];
    let (specs, name_map) = fixture_dense_norms(&norm_data);

    let chosen = auto_detect_weight_sanitize("qwen3", &specs, &name_map);

    assert_eq!(chosen, WeightSanitize::None);
}

#[test]
fn auto_detect_returns_none_for_sanitized_dense_model() {
    // mlx-community dense checkpoints ship pre-sanitized norms near 1.0;
    // auto-detection must leave them untouched.
    let norm_data = vec![1.0_f32; 256];
    let (specs, name_map) = fixture_dense_norms(&norm_data);

    let chosen = auto_detect_weight_sanitize("gemma4", &specs, &name_map);

    assert_eq!(chosen, WeightSanitize::None);
}

#[test]
fn effective_weight_sanitize_honors_explicit_manifest_mode() {
    // The manifest wins even when the on-disk probe disagrees: a manifest
    // declaring `HfToMlx` on weights that look fully raw is applied as
    // declared, and a manifest declaring `None` semantics on sanitized
    // weights runs no transform.
    let raw_norm_data: Vec<f32> = (0..256).map(|i| 0.01 * ((i as f32).cos())).collect();
    let (specs, name_map) = fixture_dense_norms(&raw_norm_data);

    let chosen = effective_weight_sanitize("gemma4", WeightSanitize::HfToMlx, &specs, &name_map);
    assert_eq!(chosen, WeightSanitize::HfToMlx);

    let sanitized_norm_data = vec![1.0_f32; 256];
    let (specs, name_map) = fixture_dense_norms(&sanitized_norm_data);
    let chosen = effective_weight_sanitize("gemma4", WeightSanitize::HfNormOnly, &specs, &name_map);
    assert_eq!(chosen, WeightSanitize::HfNormOnly);
}

#[test]
fn apply_hf_sanitize_transforms_skips_non_norm_non_conv1d_roles() {
    // The sanitizer must leave projection weights, embeddings, and
    // other non-norm tensors untouched. Otherwise it would corrupt
    // the layout of every weight matrix in the model.
    let data = [3.0_f32, 4.0, 5.0, 6.0];
    let proj = MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(&data),
        &[data.len() as i32],
        MlxDtype::Float32,
    );
    let mut name_map: HashMap<String, MlxArray> = HashMap::new();
    name_map.insert("q_proj".to_string(), proj);

    let specs = vec![NativeTensorSpec {
        name: "q_proj".to_string(),
        role: NativeTensorRole::AttentionQ,
        layer_index: Some(0),
        dtype: NativeTensorDataType::F32,
        source_tensor_type: None,
        source_quantized: false,
        quantization: None,
        quantized_source: None,
        shape: vec![1],
        file: PathBuf::from("model.safetensors"),
        offset_bytes: 0,
        length_bytes: 4,
    }];

    apply_hf_sanitize_transforms(&specs, &mut name_map, true, true);

    let preserved = name_map.get("q_proj").expect("q_proj tensor still present");
    let values = preserved.data_f32();
    for (got, want) in values.iter().zip([3.0_f32, 4.0, 5.0, 6.0].iter()) {
        assert!(
            (got - want).abs() < 1e-6,
            "q_proj must be untouched: got {got}, want {want}"
        );
    }
}

#[test]
fn apply_hf_sanitize_transforms_preserves_norm_dtype() {
    // Raw HF norm weights are typically bf16. MLX's `add(bf16, f32)` would
    // promote the result to f32 without preservation, silently doubling
    // the stored norm-weight footprint. The sanitizer must cast back to
    // the original dtype so callers see a bf16 weight, matching what
    // mlx-community pre-sanitized weights look like.
    let delta_f32 = [-0.1_f32, 0.2, 0.05, 0.0];
    let mut delta_bf16_bytes = Vec::with_capacity(delta_f32.len() * 2);
    for v in &delta_f32 {
        // Round-to-nearest cast f32 -> bf16 by chopping the low 16 bits
        // of the f32 representation (sufficient for this small test).
        let bits = v.to_bits();
        delta_bf16_bytes.extend_from_slice(&(bits >> 16).to_le_bytes()[..2]);
    }
    let norm_bf16 = MlxArray::from_raw_data(
        delta_bf16_bytes.as_ptr(),
        delta_bf16_bytes.len(),
        &[delta_f32.len() as i32],
        MlxDtype::Bfloat16,
    );
    assert_eq!(norm_bf16.dtype(), MlxDtype::Bfloat16);

    let mut name_map: HashMap<String, MlxArray> = HashMap::new();
    name_map.insert("layers.0.attn_norm".to_string(), norm_bf16);

    let specs = vec![NativeTensorSpec {
        name: "layers.0.attn_norm".to_string(),
        role: NativeTensorRole::AttentionNorm,
        layer_index: Some(0),
        dtype: NativeTensorDataType::Bf16,
        source_tensor_type: None,
        source_quantized: false,
        quantization: None,
        quantized_source: None,
        shape: vec![1],
        file: PathBuf::from("model.safetensors"),
        offset_bytes: 0,
        length_bytes: 2,
    }];

    apply_hf_sanitize_transforms(&specs, &mut name_map, true, true);

    let sanitized = name_map
        .get("layers.0.attn_norm")
        .expect("norm tensor present");
    assert_eq!(
        sanitized.dtype(),
        MlxDtype::Bfloat16,
        "sanitize must preserve bf16 dtype, not silently upcast to f32"
    );
}

#[test]
fn full_attention_projection_layout_uses_q_only_for_kv_shared_layers() {
    let specs = vec![
        spec(NativeTensorRole::AttentionQ),
        spec(NativeTensorRole::AttentionO),
    ];

    let layout = full_attention_projection_layout(&specs, Some(0), true, false)
        .expect("KV-shared layout should resolve");

    assert_eq!(layout, FullAttentionProjectionLayout::QOnly);
}

#[test]
fn full_attention_projection_layout_uses_qk_for_value_from_key_layers() {
    let specs = vec![
        spec(NativeTensorRole::AttentionQ),
        spec(NativeTensorRole::AttentionK),
        spec(NativeTensorRole::AttentionO),
    ];

    let layout = full_attention_projection_layout(&specs, Some(0), false, true)
        .expect("K=V layout should resolve");

    assert_eq!(layout, FullAttentionProjectionLayout::SplitQkValueFromKey);
}

#[test]
fn full_attention_projection_layout_uses_glm_mla_roles() {
    let specs = vec![
        spec(NativeTensorRole::AttentionQa),
        spec(NativeTensorRole::AttentionQaNorm),
        spec(NativeTensorRole::AttentionQb),
        spec(NativeTensorRole::AttentionKvA),
        spec(NativeTensorRole::AttentionKvANorm),
        spec(NativeTensorRole::AttentionEmbedQ),
        spec(NativeTensorRole::AttentionUnembedOut),
        spec(NativeTensorRole::AttentionO),
    ];

    let layout = full_attention_projection_layout(&specs, Some(0), false, false)
        .expect("GLM MLA layout should resolve");

    assert_eq!(layout, FullAttentionProjectionLayout::GlmMla);
}

#[test]
fn full_attention_projection_layout_rejects_glm_mla_mixed_with_standard_qkv() {
    let specs = vec![
        spec(NativeTensorRole::AttentionQa),
        spec(NativeTensorRole::AttentionQ),
        spec(NativeTensorRole::AttentionO),
    ];

    let error = full_attention_projection_layout(&specs, Some(0), false, false)
        .expect_err("GLM MLA cannot mix with standard QKV projections");

    assert!(matches!(error, WeightLoadError::InvalidLayer(_)));
}

#[test]
fn full_attention_projection_layout_rejects_packed_qkv_for_kv_shared_layers() {
    let specs = vec![spec(NativeTensorRole::AttentionQkvPacked)];

    let error = full_attention_projection_layout(&specs, Some(0), true, false)
        .expect_err("packed QKV cannot represent Q-only KV sharing");

    assert!(matches!(error, WeightLoadError::InvalidLayer(_)));
}

#[test]
fn load_glm_mla_attention_weights_takes_all_reference_roles() {
    let roles = [
        NativeTensorRole::AttentionQa,
        NativeTensorRole::AttentionQaNorm,
        NativeTensorRole::AttentionQb,
        NativeTensorRole::AttentionKvA,
        NativeTensorRole::AttentionKvANorm,
        NativeTensorRole::AttentionEmbedQ,
        NativeTensorRole::AttentionUnembedOut,
    ];
    let specs = roles.iter().copied().map(spec).collect::<Vec<_>>();
    let mut name_map = roles
        .iter()
        .map(|role| (format!("{role:?}"), zeros(&[1, 1], MlxDtype::Float32, None)))
        .collect::<HashMap<_, _>>();

    let mla_attention = NativeMlaAttentionConfig {
        q_lora_rank: Some(1),
        kv_lora_rank: Some(1),
        qk_nope_head_dim: Some(1),
        qk_rope_head_dim: Some(1),
        value_head_dim: Some(1),
    };
    let weights = load_glm_mla_attention_weights(&specs, &mut name_map, Some(0), &mla_attention, 1)
        .expect("GLM MLA weights should load");

    assert_eq!(weights.q_a_norm.shape(), vec![1, 1]);
    assert_eq!(weights.kv_a_norm.shape(), vec![1, 1]);
    assert!(weights.qa_kva_fused.scales.is_none());
    assert!(weights.q_b_proj.scales.is_none());
    assert!(weights.embed_q.scales.is_none());
    assert!(weights.unembed_out.scales.is_none());
    assert!(name_map.is_empty());
}

#[test]
fn glm_mla_post_attention_layernorm_is_pre_ffn_only() {
    let specs = vec![
        spec(NativeTensorRole::AttentionKvA),
        spec(NativeTensorRole::AttentionPostNorm),
    ];
    let mut name_map = HashMap::from([(
        format!("{:?}", NativeTensorRole::AttentionPostNorm),
        zeros(&[4], MlxDtype::Float32, None),
    )]);

    let (attn_post_norm, ffn_norm) =
        take_layer_norms(&specs, &mut name_map, Some(0)).expect("GLM norm should load");

    assert!(
        attn_post_norm.is_none(),
        "GLM4MoELite follows mlx_lm: post_attention_layernorm is applied after the residual as the pre-FFN norm"
    );
    assert_eq!(ffn_norm.shape(), vec![4]);
    assert!(name_map.is_empty());
}

#[test]
fn load_glm_mla_attention_weights_splits_deepseek_kv_b_projection() {
    let roles = [
        NativeTensorRole::AttentionQa,
        NativeTensorRole::AttentionQaNorm,
        NativeTensorRole::AttentionQb,
        NativeTensorRole::AttentionKvA,
        NativeTensorRole::AttentionKvANorm,
        NativeTensorRole::AttentionKvB,
    ];
    let specs = roles.iter().copied().map(spec).collect::<Vec<_>>();
    let kv_b_values = (0..18).map(|value| value as f32).collect::<Vec<_>>();
    let mut name_map = roles
        .iter()
        .map(|role| {
            let value = if *role == NativeTensorRole::AttentionKvB {
                reshape(&MlxArray::from_f32_slice(&kv_b_values), &[6, 3], None)
            } else {
                zeros(&[1, 1], MlxDtype::Float32, None)
            };
            (format!("{role:?}"), value)
        })
        .collect::<HashMap<_, _>>();
    let mla_attention = NativeMlaAttentionConfig {
        q_lora_rank: Some(1),
        kv_lora_rank: Some(3),
        qk_nope_head_dim: Some(2),
        qk_rope_head_dim: Some(1),
        value_head_dim: Some(1),
    };

    let weights = load_glm_mla_attention_weights(&specs, &mut name_map, Some(0), &mla_attention, 2)
        .expect("DeepSeek KV-B weights should load");

    assert_eq!(weights.embed_q.weight.shape(), vec![2, 3, 2]);
    assert_eq!(
        weights.embed_q.weight.data_f32(),
        &[
            0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 9.0, 12.0, 10.0, 13.0, 11.0, 14.0
        ]
    );
    assert_eq!(weights.unembed_out.weight.shape(), vec![2, 1, 3]);
    assert_eq!(
        weights.unembed_out.weight.data_f32(),
        &[6.0, 7.0, 8.0, 15.0, 16.0, 17.0]
    );
    assert!(weights.embed_q.scales.is_none());
    assert!(weights.unembed_out.scales.is_none());
    assert!(name_map.is_empty());
}

fn glm_quantized_weight(group_size: i32, bits: i32, with_biases: bool) -> QuantizedWeight {
    QuantizedWeight {
        weight: zeros(&[2, 2], MlxDtype::Uint32, None),
        scales: Some(zeros(&[2, 1], MlxDtype::Bfloat16, None)),
        biases: with_biases.then(|| zeros(&[2, 1], MlxDtype::Bfloat16, None)),
        group_size,
        bits,

        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    }
}

fn invalid_layer_message(result: Result<QuantizedWeight, WeightLoadError>) -> String {
    match result {
        Err(WeightLoadError::InvalidLayer(message)) => message,
        Err(error) => panic!("expected invalid layer error, got {error}"),
        Ok(_) => panic!("expected fused GLM MLA weights to be rejected"),
    }
}

fn moe_expert_quantized_weight(
    experts: i32,
    out: i32,
    packed_in: i32,
    group_size: i32,
    bits: i32,
) -> QuantizedWeight {
    QuantizedWeight {
        weight: zeros(&[experts, out, packed_in], MlxDtype::Uint32, None),
        scales: Some(zeros(&[experts, out, 1], MlxDtype::Bfloat16, None)),
        biases: Some(zeros(&[experts, out, 1], MlxDtype::Bfloat16, None)),
        group_size,
        bits,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    }
}

#[test]
fn fuse_resident_split_moe_experts_concatenates_on_out_axis() {
    let gate = moe_expert_quantized_weight(4, 8, 2, 32, 2);
    let up = moe_expert_quantized_weight(4, 8, 2, 32, 2);
    let (packed, gate_out, up_out) = fuse_resident_split_moe_experts(None, Some(gate), Some(up));
    let packed = packed.expect("split V4 experts must fuse");
    assert!(gate_out.is_none());
    assert!(up_out.is_none());
    assert_eq!(packed.weight.shape(), vec![4, 16, 2]);
    assert_eq!(
        packed.scales.as_ref().expect("scales").shape(),
        vec![4, 16, 1]
    );
    assert_eq!(packed.bits, 2);
    assert_eq!(packed.group_size, 32);
}

#[test]
fn fuse_resident_split_moe_experts_keeps_existing_packed() {
    let packed_in = moe_expert_quantized_weight(4, 16, 2, 32, 2);
    let gate = moe_expert_quantized_weight(4, 8, 2, 32, 2);
    let (packed, gate_out, up_out) =
        fuse_resident_split_moe_experts(Some(packed_in), Some(gate), None);
    assert!(packed.is_some());
    assert!(gate_out.is_some());
    assert!(up_out.is_none());
    assert_eq!(packed.expect("kept").weight.shape(), vec![4, 16, 2]);
}

#[test]
fn fuse_resident_split_moe_experts_keeps_split_on_bit_mismatch() {
    let gate = moe_expert_quantized_weight(4, 8, 2, 32, 2);
    let up = moe_expert_quantized_weight(4, 8, 2, 32, 4);
    let (packed, gate_out, up_out) = fuse_resident_split_moe_experts(None, Some(gate), Some(up));
    assert!(packed.is_none());
    assert!(gate_out.is_some());
    assert!(up_out.is_some());
}

#[test]
fn fuse_resident_split_moe_experts_keeps_split_on_scale_shape_mismatch() {
    let gate = moe_expert_quantized_weight(4, 8, 2, 32, 2);
    let mut up = moe_expert_quantized_weight(4, 8, 2, 32, 2);
    up.scales = Some(zeros(&[5, 8, 1], MlxDtype::Bfloat16, None));
    let (packed, gate_out, up_out) = fuse_resident_split_moe_experts(None, Some(gate), Some(up));
    assert!(packed.is_none());
    assert!(gate_out.is_some());
    assert!(up_out.is_some());
}

#[test]
fn fuse_resident_split_moe_experts_keeps_split_on_rank_one_sidecars() {
    let mut gate = moe_expert_quantized_weight(4, 8, 2, 32, 2);
    let mut up = moe_expert_quantized_weight(4, 8, 2, 32, 2);
    gate.scales = Some(zeros(&[8], MlxDtype::Bfloat16, None));
    up.scales = Some(zeros(&[8], MlxDtype::Bfloat16, None));
    let (packed, gate_out, up_out) = fuse_resident_split_moe_experts(None, Some(gate), Some(up));
    assert!(packed.is_none());
    assert!(gate_out.is_some());
    assert!(up_out.is_some());
}

#[test]
fn concat_quantized_weight_rows_accepts_matching_quantized_metadata() {
    let a = glm_quantized_weight(64, 4, true);
    let b = glm_quantized_weight(64, 4, true);

    let fused = concat_quantized_weight_rows(&a, &b).expect("matching quantization can fuse");

    assert_eq!(fused.group_size, 64);
    assert_eq!(fused.bits, 4);
    assert!(fused.scales.is_some());
    assert!(fused.biases.is_some());
}

#[test]
fn pack_dense_ffn_gate_up_projection_concatenates_gate_then_up_rows() {
    let gate = glm_quantized_weight(64, 4, true);
    let up = glm_quantized_weight(64, 4, true);

    let packed =
        pack_dense_ffn_gate_up_projection(&gate, &up).expect("matching FFN projections pack");

    assert_eq!(packed.weight.shape(), vec![4, 2]);
    assert_eq!(
        packed.scales.as_ref().expect("scales should pack").shape(),
        vec![4, 1]
    );
    assert_eq!(
        packed.biases.as_ref().expect("biases should pack").shape(),
        vec![4, 1]
    );
    assert_eq!(packed.group_size, 64);
    assert_eq!(packed.bits, 4);
}

#[test]
fn split_packed_ffn_gate_up_recovers_gate_then_up_rows() {
    let gate = glm_quantized_weight(64, 4, true);
    let up = glm_quantized_weight(64, 4, true);
    let packed =
        pack_dense_ffn_gate_up_projection(&gate, &up).expect("matching FFN projections pack");
    let (gate_back, up_back) =
        split_packed_ffn_gate_up(&packed).expect("even packed FFN must split");
    assert_eq!(gate_back.weight.shape(), gate.weight.shape());
    assert_eq!(up_back.weight.shape(), up.weight.shape());
    assert_eq!(gate_back.bits, packed.bits);
    assert_eq!(up_back.group_size, packed.group_size);
    assert!(
        crate::fastpath::should_qwen_prefill_split_packed_for(true, "qwen3_5", 1024),
        "shipped split-packed gate must accept the p2048 chunk length"
    );
}

#[test]
fn dense_ffn_gate_up_packing_support_is_family_and_bit_specific() {
    let q4_gate = glm_quantized_weight(64, 4, true);
    let q4_up = glm_quantized_weight(64, 4, true);
    let q5_gate = glm_quantized_weight(64, 5, true);
    let q5_up = glm_quantized_weight(64, 5, true);
    let q8_up = glm_quantized_weight(64, 8, true);

    assert!(!dense_ffn_gate_up_packing_supported(
        "qwen3", &q4_gate, &q4_up
    ));
    assert!(!dense_ffn_gate_up_packing_supported(
        "qwen3_5", &q4_gate, &q4_up
    ));
    let q4_gs32_gate = glm_quantized_weight(32, 4, true);
    let q4_gs32_up = glm_quantized_weight(32, 4, true);
    assert!(
        !dense_ffn_gate_up_packing_supported("qwen3_5", &q4_gs32_gate, &q4_gs32_up),
        "4-bit gs32 Qwen packing regressed AXQ 27B prefill; keep split"
    );
    assert!(!dense_ffn_gate_up_packing_supported(
        "qwen3_next",
        &q4_gate,
        &q4_up,
    ));
    let q6_gate = glm_quantized_weight(64, 6, true);
    let q6_up = glm_quantized_weight(64, 6, true);
    assert!(dense_ffn_gate_up_packing_supported(
        "qwen3_next",
        &q6_gate,
        &q6_up,
    ));
    assert!(dense_ffn_gate_up_packing_supported(
        "qwen3_5", &q6_gate, &q6_up,
    ));
    assert!(!dense_ffn_gate_up_packing_supported(
        "glm4_moe_lite",
        &q4_gate,
        &q4_up,
    ));
    assert!(!dense_ffn_gate_up_packing_supported(
        "llama", &q5_gate, &q5_up
    ));
    assert!(dense_ffn_gate_up_packing_supported(
        "llama", &q4_gate, &q4_up
    ));
    assert!(!dense_ffn_gate_up_packing_supported(
        "gemma4", &q4_gate, &q8_up,
    ));
}

#[test]
fn linear_attention_projection_packing_skips_optiq_mixed_precision() {
    let qkv = glm_quantized_weight(64, 8, true);
    let z = glm_quantized_weight(64, 4, true);
    let a = glm_quantized_weight(64, 4, true);
    let b = glm_quantized_weight(64, 8, true);

    assert!(!linear_attention_projection_packing_supported(
        &qkv, &z, &a, &b,
    ));
}

#[test]
fn linear_attention_projection_packing_accepts_matching_precision() {
    let qkv = glm_quantized_weight(64, 4, true);
    let z = glm_quantized_weight(64, 4, true);
    let a = glm_quantized_weight(64, 4, true);
    let b = glm_quantized_weight(64, 4, true);

    assert!(linear_attention_projection_packing_supported(
        &qkv, &z, &a, &b,
    ));
}

/// OptiQ often keeps ba at 8-bit and qkvz at 4-bit as separate pairs —
/// each pack group is uniform, so packing remains valid.
#[test]
fn linear_attention_projection_packing_accepts_optiq_uniform_pairs() {
    let qkv = glm_quantized_weight(64, 4, true);
    let z = glm_quantized_weight(64, 4, true);
    let a = glm_quantized_weight(64, 8, true);
    let b = glm_quantized_weight(64, 8, true);

    assert!(
        linear_attention_projection_packing_supported(&qkv, &z, &a, &b),
        "uniform-within-pair OptiQ layouts should still pack"
    );
}

/// Dense FFN gate/up with OptiQ 8+4 bits must not pack (and must not error
/// on the supported check — load keeps split projections).
#[test]
fn dense_ffn_gate_up_packing_skips_optiq_mixed_bits_on_gemma() {
    let gate = glm_quantized_weight(64, 8, true);
    let up = glm_quantized_weight(64, 4, true);
    assert!(!dense_ffn_gate_up_packing_supported("gemma4", &gate, &up));
    assert!(!dense_ffn_gate_up_packing_supported(
        "gemma4_unified",
        &gate,
        &up
    ));
    // Qwen never packs non-6-bit dense FFN, including OptiQ 4/8.
    assert!(!dense_ffn_gate_up_packing_supported("qwen3_5", &gate, &up));
    assert!(!dense_ffn_gate_up_packing_supported(
        "qwen3_5",
        &glm_quantized_weight(64, 4, true),
        &glm_quantized_weight(64, 4, true),
    ));
}

#[test]
fn take_weight_loads_optiq_override_bits_on_gate_and_up() {
    let mut gate = spec(NativeTensorRole::FfnGate);
    gate.name = "language_model.model.layers.0.mlp.gate_proj.weight".into();
    gate.dtype = NativeTensorDataType::U32;
    gate.source_quantized = true;
    gate.quantization = Some(NativeTensorQuantization {
        mode: "affine".into(),
        group_size: 64,
        bits: 8,
    });
    let mut up = spec(NativeTensorRole::FfnUp);
    up.name = "language_model.model.layers.0.mlp.up_proj.weight".into();
    up.dtype = NativeTensorDataType::U32;
    up.source_quantized = true;
    up.quantization = Some(NativeTensorQuantization {
        mode: "affine".into(),
        group_size: 64,
        bits: 4,
    });
    let specs = vec![gate, up];
    let mut name_map = HashMap::from([
        (
            "language_model.model.layers.0.mlp.gate_proj.weight".into(),
            zeros(&[16, 2], MlxDtype::Uint32, None),
        ),
        (
            "language_model.model.layers.0.mlp.gate_proj.scales".into(),
            zeros(&[16, 1], MlxDtype::Bfloat16, None),
        ),
        (
            "language_model.model.layers.0.mlp.gate_proj.biases".into(),
            zeros(&[16, 1], MlxDtype::Bfloat16, None),
        ),
        (
            "language_model.model.layers.0.mlp.up_proj.weight".into(),
            zeros(&[16, 2], MlxDtype::Uint32, None),
        ),
        (
            "language_model.model.layers.0.mlp.up_proj.scales".into(),
            zeros(&[16, 1], MlxDtype::Bfloat16, None),
        ),
        (
            "language_model.model.layers.0.mlp.up_proj.biases".into(),
            zeros(&[16, 1], MlxDtype::Bfloat16, None),
        ),
    ]);

    let g = take_weight(
        &specs,
        &mut name_map,
        NativeTensorRole::FfnGate,
        Some(0),
        "gate",
    )
    .expect("gate");
    let u = take_weight(
        &specs,
        &mut name_map,
        NativeTensorRole::FfnUp,
        Some(0),
        "up",
    )
    .expect("up");
    assert_eq!(g.bits, 8);
    assert_eq!(u.bits, 4);
    assert!(!dense_ffn_gate_up_packing_supported("gemma4", &g, &u));
}

#[test]
fn pack_glm_mla_qa_kva_projection_concatenates_and_materializes_rows() {
    let q_a = glm_quantized_weight(64, 4, true);
    let kv_a = glm_quantized_weight(64, 4, true);

    let packed =
        pack_glm_mla_qa_kva_projection(&q_a, &kv_a).expect("matching MLA projections pack");

    assert_eq!(packed.weight.shape(), vec![4, 2]);
    assert_eq!(
        packed.scales.as_ref().expect("scales should pack").shape(),
        vec![4, 1]
    );
    assert_eq!(
        packed.biases.as_ref().expect("biases should pack").shape(),
        vec![4, 1]
    );
    assert_eq!(packed.group_size, 64);
    assert_eq!(packed.bits, 4);
}

#[test]
fn pack_dense_ffn_gate_up_projection_rejects_mixed_quantization() {
    let gate = QuantizedWeight::new(zeros(&[2, 2], MlxDtype::Float32, None), None, None);
    let up = glm_quantized_weight(64, 4, false);

    let message = invalid_layer_message(pack_dense_ffn_gate_up_projection(&gate, &up));

    assert!(message.contains("only one has quantization scales"));
}

#[test]
fn concat_quantized_weight_rows_rejects_mismatched_group_size() {
    let a = glm_quantized_weight(64, 4, true);
    let b = glm_quantized_weight(32, 4, true);

    let message = invalid_layer_message(concat_quantized_weight_rows(&a, &b));

    assert!(message.contains("different group sizes"));
}

#[test]
fn concat_quantized_weight_rows_rejects_mismatched_bits() {
    let a = glm_quantized_weight(64, 4, true);
    let b = glm_quantized_weight(64, 8, true);

    let message = invalid_layer_message(concat_quantized_weight_rows(&a, &b));

    assert!(message.contains("different bit widths"));
}

#[test]
fn concat_quantized_weight_rows_rejects_mismatched_bias_presence() {
    let a = glm_quantized_weight(64, 4, true);
    let b = glm_quantized_weight(64, 4, false);

    let message = invalid_layer_message(concat_quantized_weight_rows(&a, &b));

    assert!(message.contains("only one has quantization biases"));
}

#[test]
fn concat_quantized_weight_rows_rejects_mixed_dense_and_quantized_weights() {
    let a = QuantizedWeight::new(zeros(&[2, 2], MlxDtype::Float32, None), None, None);
    let b = glm_quantized_weight(64, 4, false);

    let message = invalid_layer_message(concat_quantized_weight_rows(&a, &b));

    assert!(message.contains("only one has quantization scales"));
}

#[test]
fn linear_attention_qkvz_pack_order_interleaves_by_key_head() {
    let rows = linear_attention_qkvz_row_sources(2, 2, 4, 3).expect("valid linear attention dims");

    assert_eq!(
        rows,
        vec![
            LinearAttentionProjectionRowSource::Qkv(0),
            LinearAttentionProjectionRowSource::Qkv(1),
            LinearAttentionProjectionRowSource::Qkv(4),
            LinearAttentionProjectionRowSource::Qkv(5),
            LinearAttentionProjectionRowSource::Qkv(8),
            LinearAttentionProjectionRowSource::Qkv(9),
            LinearAttentionProjectionRowSource::Qkv(10),
            LinearAttentionProjectionRowSource::Qkv(11),
            LinearAttentionProjectionRowSource::Qkv(12),
            LinearAttentionProjectionRowSource::Qkv(13),
            LinearAttentionProjectionRowSource::Z(0),
            LinearAttentionProjectionRowSource::Z(1),
            LinearAttentionProjectionRowSource::Z(2),
            LinearAttentionProjectionRowSource::Z(3),
            LinearAttentionProjectionRowSource::Z(4),
            LinearAttentionProjectionRowSource::Z(5),
            LinearAttentionProjectionRowSource::Qkv(2),
            LinearAttentionProjectionRowSource::Qkv(3),
            LinearAttentionProjectionRowSource::Qkv(6),
            LinearAttentionProjectionRowSource::Qkv(7),
            LinearAttentionProjectionRowSource::Qkv(14),
            LinearAttentionProjectionRowSource::Qkv(15),
            LinearAttentionProjectionRowSource::Qkv(16),
            LinearAttentionProjectionRowSource::Qkv(17),
            LinearAttentionProjectionRowSource::Qkv(18),
            LinearAttentionProjectionRowSource::Qkv(19),
            LinearAttentionProjectionRowSource::Z(6),
            LinearAttentionProjectionRowSource::Z(7),
            LinearAttentionProjectionRowSource::Z(8),
            LinearAttentionProjectionRowSource::Z(9),
            LinearAttentionProjectionRowSource::Z(10),
            LinearAttentionProjectionRowSource::Z(11),
        ]
    );
}

#[test]
fn linear_attention_ba_pack_order_is_b_then_a_per_key_head() {
    let rows = linear_attention_ba_row_sources(2, 4).expect("valid linear attention dims");

    assert_eq!(
        rows,
        vec![
            LinearAttentionProjectionRowSource::B(0),
            LinearAttentionProjectionRowSource::B(1),
            LinearAttentionProjectionRowSource::A(0),
            LinearAttentionProjectionRowSource::A(1),
            LinearAttentionProjectionRowSource::B(2),
            LinearAttentionProjectionRowSource::B(3),
            LinearAttentionProjectionRowSource::A(2),
            LinearAttentionProjectionRowSource::A(3),
        ]
    );
}

#[test]
fn linear_attention_pack_order_rejects_uneven_value_heads() {
    let qkvz_message = invalid_layer_message(
        linear_attention_qkvz_row_sources(3, 2, 4, 3)
            .map(|_| QuantizedWeight::new(zeros(&[1, 1], MlxDtype::Float32, None), None, None)),
    );
    let ba_message = invalid_layer_message(
        linear_attention_ba_row_sources(3, 4)
            .map(|_| QuantizedWeight::new(zeros(&[1, 1], MlxDtype::Float32, None), None, None)),
    );

    assert!(qkvz_message.contains("value heads divisible by key heads"));
    assert!(ba_message.contains("value heads divisible by key heads"));
}

#[test]
fn linear_attention_qkvz_pack_oracle_gathers_rows_in_packed_order() {
    let rows = linear_attention_qkvz_row_sources(2, 1, 4, 2).expect("valid linear attention dims");
    let qkv: Vec<i32> = (0..12).collect();
    let z: Vec<i32> = (100..108).collect();

    let packed =
        gather_linear_attention_projection_rows(&rows, &qkv, &z, &[], &[]).expect("pack rows");

    assert_eq!(
        packed,
        vec![
            0, 2, 4, 5, 6, 7, 100, 101, 102, 103, 1, 3, 8, 9, 10, 11, 104, 105, 106, 107,
        ]
    );
    assert_ne!(
        packed,
        [qkv.as_slice(), z.as_slice(),].concat(),
        "packed qkvz is not a simple qkv-then-z row concat"
    );
}

#[test]
fn linear_attention_ba_pack_oracle_gathers_b_before_a_per_key_head() {
    let rows = linear_attention_ba_row_sources(2, 4).expect("valid linear attention dims");
    let b: Vec<i32> = (200..204).collect();
    let a: Vec<i32> = (300..304).collect();

    let packed =
        gather_linear_attention_projection_rows(&rows, &[], &[], &b, &a).expect("pack rows");

    assert_eq!(packed, vec![200, 201, 300, 301, 202, 203, 302, 303]);
    assert_ne!(
        packed,
        [b.as_slice(), a.as_slice()].concat(),
        "packed ba is not a simple b-then-a row concat when multiple key heads exist"
    );
}

#[test]
fn linear_attention_pack_oracle_rejects_short_inputs() {
    let rows = linear_attention_ba_row_sources(2, 4).expect("valid linear attention dims");

    let message = invalid_layer_message(
        gather_linear_attention_projection_rows(&rows, &[], &[], &[1], &[2])
            .map(|_| QuantizedWeight::new(zeros(&[1, 1], MlxDtype::Float32, None), None, None)),
    );

    assert!(message.contains("row source exceeded input rows"));
}

#[test]
fn quantized_weight_uses_tensor_specific_quantization_metadata() {
    let quantization = NativeTensorQuantization {
        mode: "affine".to_string(),
        group_size: 32,
        bits: 8,
    };
    let weight = zeros(&[1, 1], MlxDtype::Uint32, None);
    let scales = Some(zeros(&[1, 1], MlxDtype::Bfloat16, None));

    let quantized = QuantizedWeight::with_quantization(weight, scales, None, Some(&quantization));

    assert_eq!(quantized.group_size, 32);
    assert_eq!(quantized.bits, 8);
}

#[test]
fn take_weight_preserves_tensor_specific_quantization_metadata() {
    let mut router = spec(NativeTensorRole::FfnGateInp);
    router.name = "model.layers.0.router.proj.weight".to_string();
    router.dtype = NativeTensorDataType::U32;
    router.source_quantized = true;
    router.quantization = Some(NativeTensorQuantization {
        mode: "affine".to_string(),
        group_size: 64,
        bits: 8,
    });
    let specs = vec![router];
    let mut name_map = HashMap::from([
        (
            "model.layers.0.router.proj.weight".to_string(),
            zeros(&[128, 704], MlxDtype::Uint32, None),
        ),
        (
            "model.layers.0.router.proj.scales".to_string(),
            zeros(&[128, 44], MlxDtype::Bfloat16, None),
        ),
        (
            "model.layers.0.router.proj.biases".to_string(),
            zeros(&[128, 44], MlxDtype::Bfloat16, None),
        ),
    ]);

    let weight = take_weight(
        &specs,
        &mut name_map,
        NativeTensorRole::FfnGateInp,
        Some(0),
        "router_proj",
    )
    .expect("quantized router should load");

    assert_eq!(weight.group_size, 64);
    assert_eq!(weight.bits, 8);
    assert!(weight.scales.is_some());
}

#[test]
fn take_weight_rejects_scales_only_affine_tensor() {
    // Affine dequant needs group biases; without them MLX panics on the
    // first matmul, so the loader must refuse the tensor up front.
    let mut router = spec(NativeTensorRole::FfnGateInp);
    router.name = "model.layers.0.router.proj.weight".to_string();
    router.dtype = NativeTensorDataType::U32;
    router.source_quantized = true;
    router.quantization = Some(NativeTensorQuantization {
        mode: "affine".to_string(),
        group_size: 64,
        bits: 8,
    });
    let specs = vec![router];
    let mut name_map = HashMap::from([
        (
            "model.layers.0.router.proj.weight".to_string(),
            zeros(&[128, 704], MlxDtype::Uint32, None),
        ),
        (
            "model.layers.0.router.proj.scales".to_string(),
            zeros(&[128, 44], MlxDtype::Bfloat16, None),
        ),
    ]);
    let Err(error) = take_weight(
        &specs,
        &mut name_map,
        NativeTensorRole::FfnGateInp,
        Some(0),
        "router_proj",
    ) else {
        panic!("scales-only affine tensor must be rejected");
    };
    assert!(
        matches!(error, WeightLoadError::QuantizationMissing(ref name) if name.ends_with(".biases")),
        "{error}"
    );

    // The 4/32 pack that labels MXFP4 as affine stays accepted.
    let mut mxfp4_compat = spec(NativeTensorRole::FfnGateInp);
    mxfp4_compat.name = "model.layers.0.router.proj.weight".to_string();
    mxfp4_compat.dtype = NativeTensorDataType::U32;
    mxfp4_compat.source_quantized = true;
    mxfp4_compat.quantization = Some(NativeTensorQuantization {
        mode: "affine".to_string(),
        group_size: 32,
        bits: 4,
    });
    let mut name_map = HashMap::from([
        (
            "model.layers.0.router.proj.weight".to_string(),
            zeros(&[128, 352], MlxDtype::Uint32, None),
        ),
        (
            "model.layers.0.router.proj.scales".to_string(),
            zeros(&[128, 88], MlxDtype::Bfloat16, None),
        ),
    ]);
    assert!(
        take_weight(
            &[mxfp4_compat],
            &mut name_map,
            NativeTensorRole::FfnGateInp,
            Some(0),
            "router_proj",
        )
        .is_ok()
    );
}

#[test]
fn plain_roles_reject_quantized_specs() {
    let mut norm = spec(NativeTensorRole::AttentionQNorm);
    norm.name = "model.layers.0.self_attn.q_norm.weight".to_string();
    norm.dtype = NativeTensorDataType::U32;
    norm.source_quantized = true;
    norm.quantization = Some(NativeTensorQuantization {
        mode: "affine".to_string(),
        group_size: 64,
        bits: 4,
    });
    let mut name_map = HashMap::from([(
        "model.layers.0.self_attn.q_norm.weight".to_string(),
        zeros(&[16], MlxDtype::Uint32, None),
    )]);
    let Err(error) = try_take_plain(
        &[norm],
        &mut name_map,
        NativeTensorRole::AttentionQNorm,
        Some(0),
    ) else {
        panic!("quantized tensor must not load as a plain norm");
    };
    assert!(matches!(error, WeightLoadError::InvalidLayer(_)), "{error}");
}

#[test]
fn mtp_take_weight_defaults_to_int4_shape_inference() {
    let mut name_map = HashMap::from([
        (
            "mtp.layers.0.mlp.up_proj.weight".to_string(),
            zeros(&[128, 352], MlxDtype::Uint32, None),
        ),
        (
            "mtp.layers.0.mlp.up_proj.scales".to_string(),
            zeros(&[128, 44], MlxDtype::Bfloat16, None),
        ),
    ]);

    let weight = mtp_take_weight(&mut name_map, "mtp.layers.0.mlp.up_proj", None)
        .expect("MTP INT4 weight should load");

    assert_eq!(weight.bits, 4);
    assert_eq!(weight.group_size, 64);
}

#[test]
fn mtp_take_weight_uses_int8_sidecar_hint_for_group_inference() {
    let mut name_map = HashMap::from([
        (
            "mtp.layers.0.mlp.up_proj.weight".to_string(),
            zeros(&[128, 704], MlxDtype::Uint32, None),
        ),
        (
            "mtp.layers.0.mlp.up_proj.scales".to_string(),
            zeros(&[128, 22], MlxDtype::Bfloat16, None),
        ),
    ]);

    let weight = mtp_take_weight(&mut name_map, "mtp.layers.0.mlp.up_proj", Some(8))
        .expect("MTP INT8 weight should load");

    assert_eq!(weight.bits, 8);
    assert_eq!(weight.group_size, 128);
}

#[test]
fn mtp_router_uses_pipeline_int8_hint_inside_int4_sidecar() {
    let mut name_map = HashMap::from([
        (
            "mtp.layers.0.mlp.gate.weight".to_string(),
            zeros(&[256, 512], MlxDtype::Uint32, None),
        ),
        (
            "mtp.layers.0.mlp.gate.scales".to_string(),
            zeros(&[256, 32], MlxDtype::Bfloat16, None),
        ),
    ]);

    let weight = mtp_take_weight(
        &mut name_map,
        "mtp.layers.0.mlp.gate",
        mtp_router_bits_hint(Some(4)),
    )
    .expect("pipeline MTP router should load");

    assert_eq!(weight.bits, 8);
    assert_eq!(weight.group_size, 64);
}

/// AXQ 35B-A3B MTP sidecars ship fused `mlp.experts.gate_up_proj` +
/// `mlp.experts.down_proj` rather than split `mlp.{gate,up,down}_proj`.
/// `load_mtp` must attach those tensors so MoE MTP is available for formal A/B.
#[test]
fn load_mtp_accepts_moe_fused_experts_gate_up_packing() {
    // Minimal shapes matching the Qwen3.5/3.6 MoE MTP layout (scaled down).
    let hidden = 32usize;
    let head_dim = 8usize;
    let n_heads = 2usize;
    let n_kv = 1usize;
    let n_experts = 4usize;
    let inter = 16usize;
    let q_rows = n_heads * head_dim * 2; // queries + gate
    let k_rows = n_kv * head_dim;

    let mut name_map = HashMap::new();
    let put = |map: &mut HashMap<String, MlxArray>, key: &str, shape: &[i32]| {
        map.insert(key.to_string(), zeros(shape, MlxDtype::Bfloat16, None));
    };
    put(
        &mut name_map,
        "mtp.pre_fc_norm_embedding.weight",
        &[hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.pre_fc_norm_hidden.weight",
        &[hidden as i32],
    );
    put(&mut name_map, "mtp.norm.weight", &[hidden as i32]);
    put(
        &mut name_map,
        "mtp.fc.weight",
        &[hidden as i32, (2 * hidden) as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.input_layernorm.weight",
        &[hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.post_attention_layernorm.weight",
        &[hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.self_attn.q_norm.weight",
        &[head_dim as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.self_attn.k_norm.weight",
        &[head_dim as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.self_attn.q_proj.weight",
        &[q_rows as i32, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.self_attn.k_proj.weight",
        &[k_rows as i32, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.self_attn.v_proj.weight",
        &[k_rows as i32, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.self_attn.o_proj.weight",
        &[hidden as i32, q_rows as i32],
    );
    // Router + shared expert (dense) + fused routed experts (MoE packing).
    put(
        &mut name_map,
        "mtp.layers.0.mlp.gate.weight",
        &[n_experts as i32, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.mlp.shared_expert_gate.weight",
        &[1, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        &[inter as i32, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        &[inter as i32, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        &[hidden as i32, inter as i32],
    );
    // No `.weight` suffix on fused expert keys (matches axquant 35B sidecars).
    put(
        &mut name_map,
        "mtp.layers.0.mlp.experts.gate_up_proj",
        &[n_experts as i32, (2 * inter) as i32, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.mlp.experts.down_proj",
        &[n_experts as i32, hidden as i32, inter as i32],
    );

    let lm_head = QuantizedWeight::new(
        zeros(&[64, hidden as i32], MlxDtype::Bfloat16, None),
        None,
        None,
    );
    let mtp = load_mtp(
        &mut name_map,
        &lm_head,
        1,
        MlxSamplingParams::new(0.0, 1.0, 0),
        None,
        None,
        MtpNormLayout::MlxMultiplier,
    )
    .expect("MoE fused-experts MTP sidecar must load");

    assert_eq!(mtp.max_depth, 1);
    assert_eq!(mtp.head_dim, head_dim);
    assert_eq!(mtp.n_heads, n_heads);
    assert_eq!(mtp.n_kv_heads, n_kv);
    assert!(
        mtp.ffn_layer.router_proj.is_some(),
        "router must attach for MoE MTP"
    );
    assert!(
        mtp.ffn_layer.gate_up_exps_packed.is_some(),
        "fused experts.gate_up_proj must populate gate_up_exps_packed"
    );
    assert!(
        mtp.ffn_layer.down_exps.is_some(),
        "experts.down_proj must populate down_exps"
    );
    assert!(
        mtp.ffn_layer.gate_exps.is_none() && mtp.ffn_layer.up_exps.is_none(),
        "split expert projs should stay empty when only fused packing is present"
    );
}

#[test]
fn load_mtp_rejects_incomplete_moe_without_expert_packs() {
    let hidden = 32usize;
    let head_dim = 8usize;
    let q_rows = 2 * head_dim * 2;
    let k_rows = head_dim;
    let mut name_map = HashMap::new();
    let put = |map: &mut HashMap<String, MlxArray>, key: &str, shape: &[i32]| {
        map.insert(key.to_string(), zeros(shape, MlxDtype::Bfloat16, None));
    };
    put(
        &mut name_map,
        "mtp.pre_fc_norm_embedding.weight",
        &[hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.pre_fc_norm_hidden.weight",
        &[hidden as i32],
    );
    put(&mut name_map, "mtp.norm.weight", &[hidden as i32]);
    put(
        &mut name_map,
        "mtp.fc.weight",
        &[hidden as i32, (2 * hidden) as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.input_layernorm.weight",
        &[hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.post_attention_layernorm.weight",
        &[hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.self_attn.q_norm.weight",
        &[head_dim as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.self_attn.k_norm.weight",
        &[head_dim as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.self_attn.q_proj.weight",
        &[q_rows as i32, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.self_attn.k_proj.weight",
        &[k_rows as i32, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.self_attn.v_proj.weight",
        &[k_rows as i32, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.self_attn.o_proj.weight",
        &[hidden as i32, q_rows as i32],
    );
    // Router present → MoE path, but no expert tensors → must fail closed.
    put(
        &mut name_map,
        "mtp.layers.0.mlp.gate.weight",
        &[4, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        &[16, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        &[16, hidden as i32],
    );
    put(
        &mut name_map,
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        &[hidden as i32, 16],
    );

    let lm_head = QuantizedWeight::new(
        zeros(&[64, hidden as i32], MlxDtype::Bfloat16, None),
        None,
        None,
    );
    assert!(
        load_mtp(
            &mut name_map,
            &lm_head,
            1,
            MlxSamplingParams::new(0.0, 1.0, 0),
            None,
            None,
            MtpNormLayout::MlxMultiplier,
        )
        .is_none(),
        "incomplete MoE MTP (router without expert packs) must not attach"
    );
}

fn put_mtp_moe_common(
    name_map: &mut HashMap<String, MlxArray>,
    hidden: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv: usize,
    n_experts: usize,
    inter: usize,
) {
    let put = |map: &mut HashMap<String, MlxArray>, key: &str, shape: &[i32]| {
        map.insert(key.to_string(), zeros(shape, MlxDtype::Bfloat16, None));
    };
    let q_rows = n_heads * head_dim * 2;
    let k_rows = n_kv * head_dim;
    put(
        name_map,
        "mtp.pre_fc_norm_embedding.weight",
        &[hidden as i32],
    );
    put(name_map, "mtp.pre_fc_norm_hidden.weight", &[hidden as i32]);
    put(name_map, "mtp.norm.weight", &[hidden as i32]);
    put(
        name_map,
        "mtp.fc.weight",
        &[hidden as i32, (2 * hidden) as i32],
    );
    put(
        name_map,
        "mtp.layers.0.input_layernorm.weight",
        &[hidden as i32],
    );
    put(
        name_map,
        "mtp.layers.0.post_attention_layernorm.weight",
        &[hidden as i32],
    );
    put(
        name_map,
        "mtp.layers.0.self_attn.q_norm.weight",
        &[head_dim as i32],
    );
    put(
        name_map,
        "mtp.layers.0.self_attn.k_norm.weight",
        &[head_dim as i32],
    );
    put(
        name_map,
        "mtp.layers.0.self_attn.q_proj.weight",
        &[q_rows as i32, hidden as i32],
    );
    put(
        name_map,
        "mtp.layers.0.self_attn.k_proj.weight",
        &[k_rows as i32, hidden as i32],
    );
    put(
        name_map,
        "mtp.layers.0.self_attn.v_proj.weight",
        &[k_rows as i32, hidden as i32],
    );
    put(
        name_map,
        "mtp.layers.0.self_attn.o_proj.weight",
        &[hidden as i32, q_rows as i32],
    );
    put(
        name_map,
        "mtp.layers.0.mlp.gate.weight",
        &[n_experts as i32, hidden as i32],
    );
    put(
        name_map,
        "mtp.layers.0.mlp.shared_expert_gate.weight",
        &[1, hidden as i32],
    );
    put(
        name_map,
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        &[inter as i32, hidden as i32],
    );
    put(
        name_map,
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        &[inter as i32, hidden as i32],
    );
    put(
        name_map,
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        &[hidden as i32, inter as i32],
    );
}

#[test]
fn load_mtp_accepts_hf_per_expert_qwen_moe_packing() {
    let hidden = 32usize;
    let head_dim = 8usize;
    let n_heads = 2usize;
    let n_kv = 1usize;
    let n_experts = 4usize;
    let inter = 16usize;
    let mut name_map = HashMap::new();
    put_mtp_moe_common(
        &mut name_map,
        hidden,
        head_dim,
        n_heads,
        n_kv,
        n_experts,
        inter,
    );
    for expert in 0..n_experts {
        name_map.insert(
            format!("mtp.layers.0.mlp.experts.{expert}.gate_proj.weight"),
            zeros(&[inter as i32, hidden as i32], MlxDtype::Bfloat16, None),
        );
        name_map.insert(
            format!("mtp.layers.0.mlp.experts.{expert}.up_proj.weight"),
            zeros(&[inter as i32, hidden as i32], MlxDtype::Bfloat16, None),
        );
        name_map.insert(
            format!("mtp.layers.0.mlp.experts.{expert}.down_proj.weight"),
            zeros(&[hidden as i32, inter as i32], MlxDtype::Bfloat16, None),
        );
    }

    let lm_head = QuantizedWeight::new(
        zeros(&[64, hidden as i32], MlxDtype::Bfloat16, None),
        None,
        None,
    );
    let mtp = load_mtp(
        &mut name_map,
        &lm_head,
        1,
        MlxSamplingParams::new(0.0, 1.0, 0),
        None,
        None,
        MtpNormLayout::MlxMultiplier,
    )
    .expect("HF per-expert MoE MTP sidecar must load");

    assert_eq!(mtp.max_depth, 1);
    assert_eq!(mtp.head_dim, head_dim);
    assert_eq!(mtp.n_heads, n_heads);
    assert_eq!(mtp.n_kv_heads, n_kv);
    assert!(mtp.ffn_layer.router_proj.is_some());
    let packed = mtp
        .ffn_layer
        .gate_up_exps_packed
        .as_ref()
        .expect("per-expert gate/up must stack into gate_up_exps_packed");
    assert_eq!(
        packed.weight.shape(),
        &[n_experts as i32, (2 * inter) as i32, hidden as i32]
    );
    let down = mtp
        .ffn_layer
        .down_exps
        .as_ref()
        .expect("per-expert down must stack into down_exps");
    assert_eq!(
        down.weight.shape(),
        &[n_experts as i32, hidden as i32, inter as i32]
    );
    assert!(mtp.ffn_layer.gate_exps.is_none() && mtp.ffn_layer.up_exps.is_none());
}

#[test]
fn load_mtp_rejects_hf_per_expert_when_an_expert_is_missing() {
    let hidden = 32usize;
    let mut name_map = HashMap::new();
    put_mtp_moe_common(&mut name_map, hidden, 8, 2, 1, 4, 16);
    for expert in [0usize, 1, 3] {
        name_map.insert(
            format!("mtp.layers.0.mlp.experts.{expert}.gate_proj.weight"),
            zeros(&[16, hidden as i32], MlxDtype::Bfloat16, None),
        );
        name_map.insert(
            format!("mtp.layers.0.mlp.experts.{expert}.up_proj.weight"),
            zeros(&[16, hidden as i32], MlxDtype::Bfloat16, None),
        );
        name_map.insert(
            format!("mtp.layers.0.mlp.experts.{expert}.down_proj.weight"),
            zeros(&[hidden as i32, 16], MlxDtype::Bfloat16, None),
        );
    }
    let lm_head = QuantizedWeight::new(
        zeros(&[64, hidden as i32], MlxDtype::Bfloat16, None),
        None,
        None,
    );
    assert!(
        load_mtp(
            &mut name_map,
            &lm_head,
            1,
            MlxSamplingParams::new(0.0, 1.0, 0),
            None,
            None,
            MtpNormLayout::MlxMultiplier,
        )
        .is_none(),
        "gapped per-expert MTP (missing expert 2 of 4) must not attach"
    );
}

#[test]
fn normalize_mtp_sidecar_namespace_accepts_tiel_language_model_prefix() {
    let tensors = HashMap::from([
        (
            "language_model.mtp.fc.weight".to_string(),
            zeros(&[2, 2], MlxDtype::Bfloat16, None),
        ),
        (
            "mtp.norm.weight".to_string(),
            zeros(&[2], MlxDtype::Bfloat16, None),
        ),
    ]);
    let normalized = normalize_mtp_sidecar_namespace(tensors)
        .expect("Tiel MTP namespace must normalize without ambiguity");

    assert!(normalized.contains_key("mtp.fc.weight"));
    assert!(normalized.contains_key("mtp.norm.weight"));
    assert!(!normalized.contains_key("language_model.mtp.fc.weight"));
}

#[test]
fn normalize_mtp_sidecar_namespace_rejects_duplicate_canonical_key() {
    let tensors = HashMap::from([
        (
            "language_model.mtp.fc.weight".to_string(),
            zeros(&[2, 2], MlxDtype::Bfloat16, None),
        ),
        (
            "mtp.fc.weight".to_string(),
            zeros(&[2, 2], MlxDtype::Bfloat16, None),
        ),
    ]);

    assert!(
        normalize_mtp_sidecar_namespace(tensors).is_none(),
        "ambiguous MTP sidecar namespaces must fail closed"
    );
}

#[test]
fn load_mtp_attaches_real_qwen35_per_expert_sidecar_when_configured() {
    if std::env::var("AX_ENGINE_MLX_LOAD_REAL_WEIGHTS").as_deref() != Ok("1") {
        return;
    }
    let Ok(model_dir) = std::env::var("AX_ENGINE_MLX_REAL_MODEL_DIR") else {
        return;
    };
    let manifest: ax_engine_core::NativeModelManifest = serde_json::from_value(serde_json::json!({
        "schema_version": "ax.native_model.v1",
        "model_family": "qwen3_5_moe",
        "tensor_format": "safetensors",
        "layer_count": 40,
        "hidden_size": 2048,
        "attention_head_count": 16,
        "attention_head_dim": 256,
        "kv_head_count": 2,
        "vocab_size": 248320,
        "tensors": []
    }))
    .expect("minimal Qwen3.5-MoE manifest fixture should deserialize");
    let mut name_map = HashMap::new();
    let (max_depth, draft_sampling, sidecar_bits, draft_lm_head, norm_layout) =
        load_mtp_sidecar(Path::new(&model_dir), &mut name_map, &manifest);
    assert!(max_depth >= 1);
    assert_eq!(norm_layout, MtpNormLayout::RawHfDelta);
    let hidden = 2048i32;
    let lm_head = QuantizedWeight::new(zeros(&[16, hidden], MlxDtype::Bfloat16, None), None, None);
    let mtp = load_mtp(
        &mut name_map,
        &lm_head,
        max_depth,
        draft_sampling,
        sidecar_bits,
        draft_lm_head,
        norm_layout,
    )
    .expect("Qwen3.5-MoE HF per-expert MTP sidecar must attach");
    assert!(mtp.ffn_layer.router_proj.is_some());
    let packed = mtp
        .ffn_layer
        .gate_up_exps_packed
        .as_ref()
        .expect("per-expert experts must pack gate_up");
    assert_eq!(packed.weight.shape().first().copied(), Some(256));
    assert!(mtp.ffn_layer.down_exps.is_some());
}

#[test]
fn parse_mtp_sidecar_bits_hint_detects_int8() {
    assert_eq!(
        parse_mtp_sidecar_bits_hint(
            &serde_json::json!({"mtp_sidecar": "INT8 quantized projections, bf16 norms/router"})
        ),
        Some(8)
    );
    assert_eq!(
        parse_mtp_sidecar_bits_hint(&serde_json::json!({"mtp_sidecar": "8bit"})),
        Some(8)
    );
}

#[test]
fn parse_mtp_sidecar_bits_hint_defaults_int4_for_other_text() {
    assert_eq!(
        parse_mtp_sidecar_bits_hint(&serde_json::json!({"mtp_sidecar": "INT4"})),
        Some(4)
    );
    assert_eq!(
        parse_mtp_sidecar_bits_hint(&serde_json::json!({"mtp_sidecar": "unquantized"})),
        Some(4)
    );
}

#[test]
fn parse_mtp_sidecar_bits_hint_none_when_field_absent() {
    assert_eq!(
        parse_mtp_sidecar_bits_hint(&serde_json::json!({"mtp_depth_max": 1})),
        None
    );
}

#[test]
fn parse_mtp_sidecar_bits_hint_prefers_structured_field_over_free_text() {
    for bits in [2, 4, 6, 8, 16] {
        assert_eq!(
            parse_mtp_sidecar_bits_hint(&serde_json::json!({"mtp_sidecar_bits": bits})),
            Some(bits)
        );
    }
    // The structured field wins even when the free text says otherwise.
    assert_eq!(
        parse_mtp_sidecar_bits_hint(&serde_json::json!({
            "mtp_sidecar": "INT8 quantized projections",
            "mtp_sidecar_bits": 4
        })),
        Some(4)
    );
}

#[test]
fn parse_mtp_sidecar_bits_hint_falls_back_when_structured_field_malformed() {
    // Out-of-set integer falls back to the free-text heuristic.
    assert_eq!(
        parse_mtp_sidecar_bits_hint(&serde_json::json!({
            "mtp_sidecar": "INT8",
            "mtp_sidecar_bits": 7
        })),
        Some(8)
    );
    // Wrong type falls back to the free-text heuristic.
    assert_eq!(
        parse_mtp_sidecar_bits_hint(&serde_json::json!({
            "mtp_sidecar": "INT4",
            "mtp_sidecar_bits": "8"
        })),
        Some(4)
    );
    // Malformed structured field with no free text yields no hint.
    assert_eq!(
        parse_mtp_sidecar_bits_hint(&serde_json::json!({"mtp_sidecar_bits": 3})),
        None
    );
}

#[test]
fn qwen_sidecar_depth_one_raises_to_throughput_depth_three() {
    assert_eq!(
        default_mtp_depth_without_env_with_throughput(1, None, true),
        crate::fastpath::QWEN_LINEAR_THROUGHPUT_MTP_DEPTH
    );
    assert_eq!(
        default_mtp_depth_without_env_with_throughput(1, None, false),
        1
    );
    assert_eq!(
        default_mtp_depth_without_env_with_throughput(0, None, true),
        0
    );
    assert!(default_mtp_depth_without_env_with_throughput(1, None, true) > 0);
}

#[test]
fn parse_mtp_max_depth_cap_accepts_zero_and_positive_values() {
    assert_eq!(parse_mtp_max_depth_cap("0"), Some(0));
    assert_eq!(parse_mtp_max_depth_cap("2"), Some(2));
    assert_eq!(parse_mtp_max_depth_cap(" 3 "), Some(3));
    assert_eq!(parse_mtp_max_depth_cap(""), None);
    assert_eq!(parse_mtp_max_depth_cap("abc"), None);
}

#[test]
fn parse_mtp_norm_layout_recognizes_declared_values() {
    assert_eq!(
        parse_mtp_norm_layout(&serde_json::json!({"mtp_norm_layout": "raw_hf_delta"})),
        MtpNormLayout::RawHfDelta
    );
    assert_eq!(
        parse_mtp_norm_layout(&serde_json::json!({"mtp_norm_layout": "mlx_multiplier"})),
        MtpNormLayout::MlxMultiplier
    );
    assert_eq!(
        parse_mtp_norm_layout(&serde_json::json!({"mtp_norm_layout": "surprising"})),
        MtpNormLayout::Auto
    );
    assert_eq!(
        parse_mtp_norm_layout(&serde_json::json!({"mtp_depth_max": 1})),
        MtpNormLayout::Auto
    );
}

fn tiel_mtp_export_fixture(cyber: bool) -> serde_json::Value {
    let (source, revision, digest) = if cyber {
        (
            "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP",
            "a443d4e30fd5228942cb7695f916f4b14a88fae9",
            "590e87c9c3fbbaa370c8dddbcc22611f00bbc78019d28e3192f180841c030527",
        )
    } else {
        (
            "peculiar-ragdoll/Tiel-Coder-35B-A3B-MLX-oQ6e-MTP",
            "88625754ac91b542280a5602239ce6b2166366f0",
            "0ced87b0462269a98bd393a808e1c9c02ace7054801c0e2745dd9dd9a4077915",
        )
    };
    serde_json::json!({
        "schema_version": "axquant.protected-tensor-sidecar.v1", "role": "mtp", "tensor_count": 785,
        "source_model": {"format": "mlx", "architecture": "Qwen3_5MoeForConditionalGeneration", "model_id": source, "revision": revision},
        "output": {"path": "mtp.safetensors", "size_bytes": 1_689_387_928_u64, "sha256": digest}
    })
}

#[test]
fn tiel_mtp_norm_compatibility_requires_exact_export_identity() {
    for cyber in [false, true] {
        let fixture = tiel_mtp_export_fixture(cyber);
        assert!(known_tiel_mtp_export_digest(&fixture).is_some());
        for (pointer, value) in [
            ("/schema_version", serde_json::json!("unknown")),
            ("/role", serde_json::json!("vision")),
            ("/tensor_count", serde_json::json!(15)),
            ("/source_model/format", serde_json::json!("hf")),
            (
                "/source_model/architecture",
                serde_json::json!("Qwen3_5ForConditionalGeneration"),
            ),
            (
                "/source_model/model_id",
                serde_json::json!("other/Tiel-Coder-35B-A3B-MLX-oQ6e-MTP"),
            ),
            ("/source_model/revision", serde_json::json!("main")),
            ("/output/path", serde_json::json!("other.safetensors")),
            ("/output/size_bytes", serde_json::json!(1)),
            ("/output/sha256", serde_json::json!("0".repeat(64))),
        ] {
            let mut changed = fixture.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            assert!(
                known_tiel_mtp_export_digest(&changed).is_none(),
                "{pointer}"
            );
        }
    }
    assert!(known_tiel_mtp_export_digest(&serde_json::json!({})).is_none());
}

#[test]
fn tiel_mtp_norm_compatibility_rejects_file_replacement() {
    let dir = vision_sidecar_test_dir("tiel-mtp-replace");
    let path = dir.join("mtp.safetensors");
    std::fs::write(&path, b"first").unwrap();
    let before = std::fs::metadata(&path).unwrap();
    assert!(mtp_sidecar_file_unchanged(Some(&before), Some(&before)));
    assert!(!mtp_sidecar_file_unchanged(None, Some(&before)));
    let replacement = dir.join("replacement");
    std::fs::write(&replacement, b"other").unwrap();
    std::fs::rename(replacement, &path).unwrap();
    let after = std::fs::metadata(&path).ok();
    assert!(!mtp_sidecar_file_unchanged(Some(&before), after.as_ref()));
    // A replaced file invalidates the digest check for the declared
    // raw-HF layout even when the override was not applied; other
    // declared layouts never consult the digest and are unaffected.
    assert!(mtp_sidecar_verification_is_stale(
        MtpNormLayout::RawHfDelta,
        Some(&before),
        after.as_ref()
    ));
    assert!(!mtp_sidecar_verification_is_stale(
        MtpNormLayout::RawHfDelta,
        Some(&before),
        Some(&before)
    ));
    assert!(!mtp_sidecar_verification_is_stale(
        MtpNormLayout::MlxMultiplier,
        Some(&before),
        after.as_ref()
    ));
    assert!(!mtp_sidecar_verification_is_stale(
        MtpNormLayout::Auto,
        None,
        after.as_ref()
    ));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tiel_mtp_norm_compatibility_does_not_trust_manifest_hash_alone() {
    let dir = vision_sidecar_test_dir("tiel-mtp-hash");
    std::fs::write(
        dir.join("axquant_mtp_sidecar_manifest.json"),
        serde_json::to_vec(&tiel_mtp_export_fixture(false)).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("mtp.safetensors"), b"different sidecar").unwrap();
    assert_eq!(
        resolve_mtp_norm_layout(
            &dir,
            &serde_json::json!({"mtp_norm_layout": "raw_hf_delta"})
        ),
        MtpNormLayout::RawHfDelta
    );
    assert_eq!(
        resolve_mtp_norm_layout(
            &dir,
            &serde_json::json!({"mtp_norm_layout": "mlx_multiplier"})
        ),
        MtpNormLayout::MlxMultiplier
    );
    assert_eq!(
        resolve_mtp_norm_layout(&dir, &serde_json::json!({})),
        MtpNormLayout::Auto
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn mtp_norm_shift_decision_is_per_sidecar_not_per_tensor() {
    // The measured raw Qwen 3.6 sidecar: only the input layernorm falls
    // below the 0.15 threshold, but all seven norms are raw deltas. The
    // old per-tensor decision shifted exactly one of them, producing a
    // silently mixed sidecar with zero draft acceptance.
    let raw_qwen36 = [
        Some(0.0827),
        Some(0.2110),
        Some(0.7438),
        Some(0.7610),
        Some(1.2741),
        Some(0.4400),
        Some(0.1792),
    ];
    assert!(mtp_norms_need_shift(MtpNormLayout::Auto, &raw_qwen36));

    // A sanitized sidecar clusters near 1.0 and must not be re-shifted.
    let sanitized = [Some(1.08); 7];
    assert!(!mtp_norms_need_shift(MtpNormLayout::Auto, &sanitized));

    // Tensors too small for a mean_abs verdict do not force a shift.
    let inconclusive = [None; 7];
    assert!(!mtp_norms_need_shift(MtpNormLayout::Auto, &inconclusive));
}

#[test]
fn mtp_norm_shift_declared_layout_overrides_statistics() {
    // A declared layout bypasses auto-detection entirely: raw_hf_delta
    // shifts even when every norm looks sanitized, and mlx_multiplier
    // never shifts even when the statistics look raw.
    let looks_sanitized = [Some(1.0); 7];
    assert!(mtp_norms_need_shift(
        MtpNormLayout::RawHfDelta,
        &looks_sanitized
    ));
    let looks_raw = [Some(0.05); 7];
    assert!(!mtp_norms_need_shift(
        MtpNormLayout::MlxMultiplier,
        &looks_raw
    ));
}

#[test]
fn default_mtp_depth_passes_through_configured_depth() {
    assert_eq!(
        default_mtp_depth_without_env_with_throughput(3, Some(8), false),
        3
    );
    assert_eq!(
        default_mtp_depth_without_env_with_throughput(1, Some(8), false),
        1
    );
    assert_eq!(
        default_mtp_depth_without_env_with_throughput(0, Some(8), false),
        0
    );
    assert_eq!(
        default_mtp_depth_without_env_with_throughput(3, Some(4), false),
        3
    );
    assert_eq!(
        default_mtp_depth_without_env_with_throughput(3, None, false),
        3
    );
}

#[test]
fn take_weight_rejects_quantized_tensor_without_scales() {
    let mut router = spec(NativeTensorRole::FfnGateInp);
    router.name = "model.layers.0.router.proj.weight".to_string();
    router.dtype = NativeTensorDataType::U32;
    router.source_quantized = true;
    router.quantization = Some(NativeTensorQuantization {
        mode: "affine".to_string(),
        group_size: 64,
        bits: 8,
    });
    let specs = vec![router];
    let mut name_map = HashMap::from([(
        "model.layers.0.router.proj.weight".to_string(),
        zeros(&[128, 704], MlxDtype::Uint32, None),
    )]);

    let error = match take_weight(
        &specs,
        &mut name_map,
        NativeTensorRole::FfnGateInp,
        Some(0),
        "router_proj",
    ) {
        Ok(_) => panic!("quantized MLX tensors require co-located scales"),
        Err(error) => error,
    };

    assert!(matches!(error, WeightLoadError::QuantizationMissing(_)));
}

#[test]
fn take_weight_rejects_quantization_sidecars_when_manifest_is_dense() {
    let mut router = spec(NativeTensorRole::FfnGateInp);
    router.name = "model.layers.0.router.proj.weight".to_string();
    router.dtype = NativeTensorDataType::Bf16;
    router.source_quantized = false;
    let specs = vec![router];
    let mut name_map = HashMap::from([
        (
            "model.layers.0.router.proj.weight".to_string(),
            zeros(&[128, 2816], MlxDtype::Bfloat16, None),
        ),
        (
            "model.layers.0.router.proj.scales".to_string(),
            zeros(&[128, 44], MlxDtype::Bfloat16, None),
        ),
    ]);

    let error = match take_weight(
        &specs,
        &mut name_map,
        NativeTensorRole::FfnGateInp,
        Some(0),
        "router_proj",
    ) {
        Ok(_) => panic!("dense manifest tensors must not consume quantization sidecars"),
        Err(error) => error,
    };

    assert!(matches!(error, WeightLoadError::InvalidLayer(_)));
}

#[test]
fn real_mlx_weights_load_qwen35_linear_attention_when_configured() {
    if std::env::var("AX_ENGINE_MLX_LOAD_REAL_WEIGHTS").as_deref() != Ok("1") {
        return;
    }
    let Ok(model_dir) = std::env::var("AX_ENGINE_MLX_REAL_MODEL_DIR") else {
        return;
    };
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .expect("real MLX manifest should load");

    let weights = load_weights(&artifacts).expect("real MLX weights should load");

    assert_eq!(
        weights.layers.len(),
        artifacts.manifest().layer_count as usize
    );
    assert!(
        weights
            .layers
            .first()
            .and_then(|layer| layer.linear_attn.as_ref())
            .is_some(),
        "Qwen3.5 layer 0 should load linear-attention weights"
    );
}

#[test]
fn load_glm_mtp_sidecar_returns_none_when_no_sidecar_file() {
    // When glm_mtp.safetensors is absent the loader must return None without
    // panicking.  We use a temp dir that contains no glm_mtp.* files.
    let tmp = std::env::temp_dir().join(format!(
        "ax-weights-test-glm-mtp-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();
    let manifest: ax_engine_core::NativeModelManifest = serde_json::from_value(serde_json::json!({
        "schema_version": "ax.native_model.v1",
        "model_family": "glm4_moe_lite",
        "tensor_format": "safetensors",
        "layer_count": 1,
        "hidden_size": 1,
        "attention_head_count": 1,
        "attention_head_dim": 1,
        "kv_head_count": 1,
        "vocab_size": 1,
        "tensors": []
    }))
    .expect("minimal manifest fixture should deserialize");
    let mut name_map = HashMap::new();
    let result = load_glm_mtp_sidecar(&tmp, &mut name_map, &manifest);
    assert!(
        result.is_none(),
        "expected None when glm_mtp.safetensors is absent"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn load_deepseek_v4_mtp_sidecar_gates_and_missing_file() {
    let tmp = std::env::temp_dir().join(format!(
        "ax-weights-test-dsv4-mtp-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();
    let v4_manifest: ax_engine_core::NativeModelManifest =
        serde_json::from_value(serde_json::json!({
            "schema_version": "ax.native_model.v1",
            "model_family": "deepseek_v4",
            "tensor_format": "safetensors",
            "layer_count": 1,
            "hidden_size": 1,
            "attention_head_count": 1,
            "attention_head_dim": 1,
            "kv_head_count": 1,
            "vocab_size": 1,
            "deepseek_v4": { "num_nextn_predict_layers": 1 },
            "tensors": []
        }))
        .expect("minimal V4 manifest fixture should deserialize");
    let qwen_manifest: ax_engine_core::NativeModelManifest =
        serde_json::from_value(serde_json::json!({
            "schema_version": "ax.native_model.v1",
            "model_family": "qwen3",
            "tensor_format": "safetensors",
            "layer_count": 1,
            "hidden_size": 1,
            "attention_head_count": 1,
            "attention_head_dim": 1,
            "kv_head_count": 1,
            "vocab_size": 1,
            "tensors": []
        }))
        .expect("minimal manifest fixture should deserialize");

    // Family gate: non-V4 manifests never touch `mtp.safetensors` here.
    let mut name_map = HashMap::new();
    assert!(load_deepseek_v4_mtp_sidecar(&tmp, &mut name_map, &qwen_manifest).is_none());
    // Missing sidecar file: graceful None, no panic.
    assert!(load_deepseek_v4_mtp_sidecar(&tmp, &mut name_map, &v4_manifest).is_none());
    std::fs::remove_dir_all(&tmp).ok();
}

fn array_u8(data: &[u8], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(data.as_ptr(), data.len(), shape, MlxDtype::Uint8)
}

#[test]
fn mtp_e8m0_lut_spot_checks() {
    let lut = mtp_e8m0_lut();
    assert_eq!(lut.len(), 256);
    assert_eq!(lut[127], 1.0);
    assert_eq!(lut[126], 0.5);
    assert_eq!(lut[128], 2.0);
    assert_eq!(lut[0], 2f32.powi(-127));
    assert!(lut[255].is_nan());
}

#[test]
fn mtp_take_fp8_blockscaled_dequantizes_tiny_block() {
    // e4m3fn 0x38 == 1.0 and 0x40 == 2.0 (sign 0, exp 0111/1000, mantissa
    // 000); e8m0 byte 128 == 2^1. A 1×1 scale over a 2×2 weight derives a
    // 2×2 block, so every element is scaled by 2.0.
    let mut name_map = HashMap::from([
        (
            "w.weight".to_string(),
            array_u8(&[0x38, 0x40, 0x40, 0x38], &[2, 2]),
        ),
        ("w.scale".to_string(), array_u8(&[128], &[1, 1])),
    ]);
    let qw = mtp_take_fp8_blockscaled(&mut name_map, "w")
        .expect("fp8 block-scaled pair should dequantize");
    assert!(qw.scales.is_none());
    assert!(qw.biases.is_none());
    assert_eq!(qw.weight.shape(), vec![2, 2]);
    assert_eq!(qw.weight.dtype(), MlxDtype::Bfloat16);
    assert!(name_map.is_empty(), "both tensors must be consumed");
    let as_f32 = astype(&qw.weight, MlxDtype::Float32, None);
    eval(&[&as_f32]);
    assert_eq!(as_f32.data_f32(), &[2.0, 4.0, 4.0, 2.0]);
}

#[test]
fn mtp_take_fp8_blockscaled_none_without_consuming_on_bad_input() {
    // Missing scale: weight must stay in the map for the BF16 fallback.
    let mut name_map = HashMap::from([("w.weight".to_string(), array_u8(&[0x38; 4], &[2, 2]))]);
    assert!(mtp_take_fp8_blockscaled(&mut name_map, "w").is_none());
    assert!(name_map.contains_key("w.weight"));
    // Non-byte-container weight (dense fallback territory): reject
    // without consuming even when a `.scale` tensor shares the prefix.
    let mut name_map = HashMap::from([
        (
            "w.weight".to_string(),
            reshape(&MlxArray::from_f32_slice(&[1.0; 12]), &[3, 4], None),
        ),
        ("w.scale".to_string(), array_u8(&[127; 2], &[2, 1])),
    ]);
    assert!(mtp_take_fp8_blockscaled(&mut name_map, "w").is_none());
    assert_eq!(name_map.len(), 2);
    // FP8 bytes with a scale grid that does not divide the weight dims:
    // fail closed by consuming both so the dense fallback cannot read
    // the raw E4M3 bytes as a dense weight.
    let mut name_map = HashMap::from([
        ("w.weight".to_string(), array_u8(&[0x38; 12], &[3, 4])),
        ("w.scale".to_string(), array_u8(&[127; 2], &[2, 1])),
    ]);
    assert!(mtp_take_fp8_blockscaled(&mut name_map, "w").is_none());
    assert!(name_map.is_empty(), "malformed fp8 pair must be consumed");
}

#[test]
fn mtp_take_mxfp4_experts_stacks_fused_gate_up_and_down() {
    // 2 experts, out = in = 32 real values: packed byte rows hold in/2 =
    // 16 nibbles-packed bytes → 4 u32; one e8m0 scale column (group 32).
    let mut name_map = HashMap::new();
    for expert in 0..2 {
        let prefix = format!("mtp.0.ffn.experts.{expert}");
        name_map.insert(
            format!("{prefix}.w1.weight"),
            array_u8(&[0x12; 32 * 16], &[32, 16]),
        );
        name_map.insert(format!("{prefix}.w1.scale"), array_u8(&[127; 32], &[32, 1]));
        name_map.insert(
            format!("{prefix}.w2.weight"),
            array_u8(&[0x34; 32 * 16], &[32, 16]),
        );
        name_map.insert(format!("{prefix}.w2.scale"), array_u8(&[127; 32], &[32, 1]));
        name_map.insert(
            format!("{prefix}.w3.weight"),
            array_u8(&[0x56; 32 * 16], &[32, 16]),
        );
        name_map.insert(format!("{prefix}.w3.scale"), array_u8(&[127; 32], &[32, 1]));
    }
    let (gate_up, down) = mtp_take_mxfp4_experts(&mut name_map, "mtp.0", 2)
        .expect("per-expert mxfp4 experts should stack");
    // Fused gate+up: [E, 2*32, 32*4/32] u32 with matching e8m0 scales.
    assert_eq!(gate_up.weight.shape(), vec![2, 64, 4]);
    assert_eq!(gate_up.weight.dtype(), MlxDtype::Uint32);
    assert_eq!(
        gate_up.scales.as_ref().map(MlxArray::shape),
        Some(vec![2, 64, 1])
    );
    assert!(gate_up.biases.is_none());
    assert_eq!(gate_up.group_size, 32);
    assert_eq!(gate_up.bits, 4);
    assert_eq!(gate_up.mode, "mxfp4");
    assert_eq!(down.weight.shape(), vec![2, 32, 4]);
    assert_eq!(down.weight.dtype(), MlxDtype::Uint32);
    assert_eq!(
        down.scales.as_ref().map(MlxArray::shape),
        Some(vec![2, 32, 1])
    );
    assert_eq!(down.mode, "mxfp4");
    assert!(name_map.is_empty(), "all expert tensors must be consumed");
}

#[test]
fn mtp_take_mxfp4_experts_none_when_any_expert_tensor_missing() {
    let mut name_map = HashMap::new();
    assert!(mtp_take_mxfp4_experts(&mut name_map, "mtp.0", 2).is_none());
    // Expert 0 complete, expert 1 missing w3 → incomplete.
    for (proj, byte) in [("w1", 0x12u8), ("w2", 0x34), ("w3", 0x56)] {
        let prefix = format!("mtp.0.ffn.experts.0.{proj}");
        name_map.insert(
            format!("{prefix}.weight"),
            array_u8(&[byte; 32 * 16], &[32, 16]),
        );
        name_map.insert(format!("{prefix}.scale"), array_u8(&[127; 32], &[32, 1]));
    }
    let prefix = "mtp.0.ffn.experts.1";
    name_map.insert(
        format!("{prefix}.w1.weight"),
        array_u8(&[0x12; 32 * 16], &[32, 16]),
    );
    name_map.insert(format!("{prefix}.w1.scale"), array_u8(&[127; 32], &[32, 1]));
    name_map.insert(
        format!("{prefix}.w2.weight"),
        array_u8(&[0x34; 32 * 16], &[32, 16]),
    );
    name_map.insert(format!("{prefix}.w2.scale"), array_u8(&[127; 32], &[32, 1]));
    assert!(mtp_take_mxfp4_experts(&mut name_map, "mtp.0", 2).is_none());
    // A partially-present set must leave every tensor in the map so the
    // stacked fallback and leftover diagnostics still see them.
    assert_eq!(name_map.len(), 10);
}

/// Zero-filled safetensors writer mirroring `write_vision_sidecar_fixture`
/// but with per-tensor dtype strings, so FP8/I8 AXQuant layouts can be
/// reproduced. `tensors` is `(name, dtype, shape)`.
fn write_dsv4_mtp_sidecar_fixture(dir: &Path, tensors: &[(&str, &str, &[usize])]) {
    let elem_size = |dtype: &str| match dtype {
        "F32" => 4,
        "F16" | "BF16" => 2,
        "F8_E4M3" | "F8_E8M0" | "I8" | "U8" => 1,
        other => panic!("unsupported fixture dtype {other}"),
    };
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, dtype, shape) in tensors {
        let numel: usize = shape.iter().product();
        let start = data.len();
        data.resize(start + numel * elem_size(dtype), 0);
        let end = data.len();
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file_bytes.extend_from_slice(&header_bytes);
    file_bytes.extend_from_slice(&data);
    std::fs::write(dir.join("mtp.safetensors"), &file_bytes).unwrap();
}

fn dsv4_mtp_test_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ax-weights-test-dsv4-mtp-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dsv4_mtp_test_manifest(expert_count: u32) -> ax_engine_core::NativeModelManifest {
    serde_json::from_value(serde_json::json!({
        "schema_version": "ax.native_model.v1",
        "model_family": "deepseek_v4",
        "tensor_format": "safetensors",
        "layer_count": 1,
        "hidden_size": 8,
        "attention_head_count": 1,
        "attention_head_dim": 1,
        "kv_head_count": 1,
        "vocab_size": 1,
        "deepseek_v4": { "num_nextn_predict_layers": 1 },
        "moe": { "expert_count": expert_count },
        "tensors": []
    }))
    .expect("minimal V4 manifest fixture should deserialize")
}

/// Tensor names every `mtp.0`-prefixed sidecar layout shares: input
/// norms/projections, norms, hyper-connection parameters, router.
fn dsv4_mtp_common_tensors(dtype: &str) -> Vec<(String, String, Vec<usize>)> {
    let mut tensors: Vec<(String, String, Vec<usize>)> = vec![
        ("mtp.0.enorm.weight", dtype, &[32][..]),
        ("mtp.0.hnorm.weight", dtype, &[32][..]),
        ("mtp.0.norm.weight", dtype, &[32][..]),
        ("mtp.0.attn_norm.weight", dtype, &[32][..]),
        ("mtp.0.ffn_norm.weight", dtype, &[32][..]),
        ("mtp.0.attn.q_norm.weight", dtype, &[32][..]),
        ("mtp.0.attn.kv_norm.weight", dtype, &[32][..]),
        ("mtp.0.ffn.gate.weight", dtype, &[2, 32][..]),
    ]
    .into_iter()
    .map(|(name, dtype, shape)| (name.to_string(), dtype.to_string(), shape.to_vec()))
    .collect();
    for hc in [
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
    ] {
        tensors.push((format!("mtp.0.{hc}"), "F32".to_string(), vec![1]));
    }
    tensors.push((
        "mtp.0.ffn.gate.bias".to_string(),
        "F32".to_string(),
        vec![2],
    ));
    tensors.push((
        "mtp.0.attn.attn_sink".to_string(),
        "F32".to_string(),
        vec![2],
    ));
    tensors
}

fn dsv4_mtp_tensor_refs(tensors: &[(String, String, Vec<usize>)]) -> Vec<(&str, &str, &[usize])> {
    tensors
        .iter()
        .map(|(name, dtype, shape)| (name.as_str(), dtype.as_str(), shape.as_slice()))
        .collect()
}

#[test]
fn load_deepseek_v4_mtp_sidecar_loads_bf16_stacked_fallback() {
    // Raw-HF style sidecar: dense BF16 tensors and the stacked
    // `ffn.experts.{gate,up,down}` triple must still load through
    // `mtp_take_weight` when no FP8 pairs / per-expert tensors exist.
    let dir = dsv4_mtp_test_dir("bf16-fallback");
    let mut tensors = dsv4_mtp_common_tensors("BF16");
    for (name, shape) in [
        ("mtp.0.e_proj.weight", vec![32, 32]),
        ("mtp.0.h_proj.weight", vec![32, 32]),
        ("mtp.0.attn.wq_a.weight", vec![32, 32]),
        ("mtp.0.attn.wq_b.weight", vec![32, 32]),
        ("mtp.0.attn.wkv.weight", vec![32, 32]),
        ("mtp.0.attn.wo_a.weight", vec![32, 32]),
        ("mtp.0.attn.wo_b.weight", vec![32, 32]),
        ("mtp.0.ffn.experts.gate.weight", vec![2, 32, 32]),
        ("mtp.0.ffn.experts.up.weight", vec![2, 32, 32]),
        ("mtp.0.ffn.experts.down.weight", vec![2, 32, 32]),
        ("mtp.0.ffn.shared_experts.w1.weight", vec![32, 32]),
        ("mtp.0.ffn.shared_experts.w2.weight", vec![32, 32]),
        ("mtp.0.ffn.shared_experts.w3.weight", vec![32, 32]),
    ] {
        tensors.push((name.to_string(), "BF16".to_string(), shape));
    }
    write_dsv4_mtp_sidecar_fixture(&dir, &dsv4_mtp_tensor_refs(&tensors));
    let manifest = dsv4_mtp_test_manifest(2);
    let mut name_map = HashMap::new();
    let nextn = load_deepseek_v4_mtp_sidecar(&dir, &mut name_map, &manifest)
        .expect("BF16 stacked sidecar should load");
    let layer = nextn.layer.as_ref().expect("nextn layer should be present");
    assert!(layer.gate_up_exps_packed.is_none());
    assert!(layer.gate_exps.is_some());
    assert!(layer.up_exps.is_some());
    assert!(layer.down_exps.is_some());
    assert!(nextn.e_proj.is_some());
    assert!(nextn.h_proj.is_some());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_deepseek_v4_mtp_sidecar_loads_axquant_fp8_mxfp4_layout() {
    // The published AXQuant artifact layout: FP8 blockwise projections
    // (E4M3 weight + E8M0 scale) and per-expert MXFP4 routed experts
    // ("I8" payloads + E8M0 scales). Zero-filled payloads dequantize to
    // zeros; this test checks dispatch and shapes, not values.
    let dir = dsv4_mtp_test_dir("axquant");
    let mut tensors = dsv4_mtp_common_tensors("BF16");
    for base in [
        "mtp.0.e_proj",
        "mtp.0.h_proj",
        "mtp.0.attn.wq_a",
        "mtp.0.attn.wq_b",
        "mtp.0.attn.wkv",
        "mtp.0.attn.wo_a",
        "mtp.0.attn.wo_b",
        "mtp.0.ffn.shared_experts.w1",
        "mtp.0.ffn.shared_experts.w2",
        "mtp.0.ffn.shared_experts.w3",
    ] {
        tensors.push((
            format!("{base}.weight"),
            "F8_E4M3".to_string(),
            vec![32, 32],
        ));
        tensors.push((format!("{base}.scale"), "F8_E8M0".to_string(), vec![1, 1]));
    }
    for expert in 0..2 {
        for proj in ["w1", "w2", "w3"] {
            let prefix = format!("mtp.0.ffn.experts.{expert}.{proj}");
            tensors.push((format!("{prefix}.weight"), "I8".to_string(), vec![32, 16]));
            tensors.push((
                format!("{prefix}.scale"),
                "F8_E8M0".to_string(),
                vec![32, 1],
            ));
        }
    }
    write_dsv4_mtp_sidecar_fixture(&dir, &dsv4_mtp_tensor_refs(&tensors));
    let manifest = dsv4_mtp_test_manifest(2);
    let mut name_map = HashMap::new();
    let nextn = load_deepseek_v4_mtp_sidecar(&dir, &mut name_map, &manifest)
        .expect("AXQuant FP8/MXFP4 sidecar should load");
    let layer = nextn.layer.as_ref().expect("nextn layer should be present");
    // Routed experts dispatch to fused per-expert MXFP4 packing.
    let gate_up = layer
        .gate_up_exps_packed
        .as_ref()
        .expect("AXQuant sidecar should pack gate_up experts");
    assert_eq!(gate_up.weight.shape(), vec![2, 64, 4]);
    assert_eq!(gate_up.weight.dtype(), MlxDtype::Uint32);
    assert_eq!(gate_up.mode, "mxfp4");
    assert_eq!(gate_up.group_size, 32);
    assert_eq!(gate_up.bits, 4);
    assert!(layer.gate_exps.is_none());
    assert!(layer.up_exps.is_none());
    let down = layer.down_exps.as_ref().expect("down experts should load");
    assert_eq!(down.weight.shape(), vec![2, 32, 4]);
    assert_eq!(down.mode, "mxfp4");
    // FP8 projections dequantize to dense BF16.
    let v4 = layer.deepseek_v4.as_ref().expect("v4 attention weights");
    assert!(!v4.wq_a.is_quantized());
    assert_eq!(v4.wq_a.weight.dtype(), MlxDtype::Bfloat16);
    assert_eq!(v4.wq_a.weight.shape(), vec![32, 32]);
    let e_proj = nextn.e_proj.as_ref().expect("e_proj should load");
    assert!(!e_proj.is_quantized());
    assert_eq!(e_proj.weight.dtype(), MlxDtype::Bfloat16);
    std::fs::remove_dir_all(&dir).ok();
}

fn vision_sidecar_test_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ax-weights-test-vision-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write a minimal F32 safetensors file as `vision.safetensors` in `dir`
/// and return the exact bytes written (for manifest hashing).
fn write_vision_sidecar_fixture(dir: &Path, tensors: &[(&str, &[f32], &[usize])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, values, shape) in tensors {
        let start = data.len();
        for value in *values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let end = data.len();
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file_bytes.extend_from_slice(&header_bytes);
    file_bytes.extend_from_slice(&data);
    std::fs::write(dir.join(VISION_SIDECAR_FILE), &file_bytes).unwrap();
    file_bytes
}

fn vision_sidecar_manifest(
    file_bytes: &[u8],
    role: &str,
    tensor_count: usize,
    parameters: u64,
) -> serde_json::Value {
    serde_json::json!({
        "schema_version": VISION_SIDECAR_SCHEMA,
        "source_model": {"model_id": "test/vision-model", "revision": "abc123"},
        "role": role,
        "tensor_count": tensor_count,
        "parameters": parameters,
        "dtypes": ["F32"],
        "tensor_names_sha256": "0".repeat(64),
        "source_files": [],
        "output": {
            "path": VISION_SIDECAR_FILE,
            "size_bytes": file_bytes.len(),
            "sha256": ax_engine_core::sha256_hex(file_bytes),
        }
    })
}

fn write_vision_sidecar_manifest_fixture(dir: &Path, manifest: &serde_json::Value) {
    std::fs::write(
        dir.join(VISION_SIDECAR_MANIFEST_FILE),
        serde_json::to_vec_pretty(manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn vision_sidecar_merges_tensors_without_overwriting_main_file_entries() {
    let dir = vision_sidecar_test_dir("happy");
    let file_bytes = write_vision_sidecar_fixture(
        &dir,
        &[
            (
                "vision_tower.patch_embed.weight",
                &[1.0, 2.0, 3.0, 4.0],
                &[2, 2],
            ),
            ("shared.weight", &[9.0, 9.0, 9.0, 9.0], &[2, 2]),
        ],
    );
    write_vision_sidecar_manifest_fixture(
        &dir,
        &vision_sidecar_manifest(&file_bytes, "vision", 2, 8),
    );

    // Simulate a tensor already loaded from the main safetensors files.
    let mut name_map = HashMap::from([(
        "shared.weight".to_string(),
        zeros(&[1, 1], MlxDtype::Float32, None),
    )]);

    let info = load_vision_sidecar(&dir, &mut name_map)
        .expect("vision sidecar should load")
        .expect("sidecar is present");

    assert_eq!(info.tensor_count, 2);
    assert_eq!(info.parameters, 8);
    assert_eq!(info.source_model_id, "test/vision-model");
    assert_eq!(name_map.len(), 2);
    let patch_embed = name_map
        .get("vision_tower.patch_embed.weight")
        .expect("sidecar tensor should be merged");
    eval(&[patch_embed]);
    assert_eq!(patch_embed.shape(), vec![2, 2]);
    assert_eq!(patch_embed.data_f32(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        name_map.get("shared.weight").map(|array| array.shape()),
        Some(vec![1, 1]),
        "main-file tensor must win over a sidecar duplicate"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn vision_sidecar_rejects_tampered_file_sha_mismatch() {
    let dir = vision_sidecar_test_dir("tampered");
    let file_bytes = write_vision_sidecar_fixture(&dir, &[("vision_tower.weight", &[1.0], &[1])]);
    write_vision_sidecar_manifest_fixture(
        &dir,
        &vision_sidecar_manifest(&file_bytes, "vision", 1, 1),
    );
    // Flip a data byte in place so the size still matches but the hash does not.
    let mut tampered = file_bytes.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0xFF;
    std::fs::write(dir.join(VISION_SIDECAR_FILE), &tampered).unwrap();

    let mut name_map = HashMap::new();
    let error = match load_vision_sidecar(&dir, &mut name_map) {
        Ok(_) => panic!("tampered sidecar must fail provenance verification"),
        Err(error) => error,
    };

    assert!(matches!(error, WeightLoadError::VisionSidecarInvalid(_)));
    assert!(error.to_string().contains("sha256"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn vision_sidecar_rejects_missing_manifest() {
    let dir = vision_sidecar_test_dir("no-manifest");
    write_vision_sidecar_fixture(&dir, &[("vision_tower.weight", &[1.0], &[1])]);

    let mut name_map = HashMap::new();
    let error = match load_vision_sidecar(&dir, &mut name_map) {
        Ok(_) => panic!("sidecar without a manifest must fail closed"),
        Err(error) => error,
    };

    assert!(matches!(error, WeightLoadError::VisionSidecarInvalid(_)));
    assert!(error.to_string().contains("manifest"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn vision_sidecar_rejects_wrong_schema_version_and_role() {
    for (tag, schema, role) in [
        ("schema", "axquant.protected-tensor-sidecar.v2", "vision"),
        ("role", VISION_SIDECAR_SCHEMA, "mtp"),
    ] {
        let dir = vision_sidecar_test_dir(tag);
        let file_bytes =
            write_vision_sidecar_fixture(&dir, &[("vision_tower.weight", &[1.0], &[1])]);
        let mut manifest = vision_sidecar_manifest(&file_bytes, role, 1, 1);
        manifest["schema_version"] = serde_json::json!(schema);
        write_vision_sidecar_manifest_fixture(&dir, &manifest);

        let mut name_map = HashMap::new();
        let result = load_vision_sidecar(&dir, &mut name_map);

        assert!(
            matches!(result, Err(WeightLoadError::VisionSidecarInvalid(_))),
            "{tag}: expected VisionSidecarInvalid, got {result:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[test]
fn vision_sidecar_rejects_manifest_without_sidecar_file() {
    let dir = vision_sidecar_test_dir("no-file");
    write_vision_sidecar_manifest_fixture(&dir, &vision_sidecar_manifest(b"", "vision", 0, 0));

    let mut name_map = HashMap::new();
    let error = match load_vision_sidecar(&dir, &mut name_map) {
        Ok(_) => panic!("manifest without the sidecar file must fail closed"),
        Err(error) => error,
    };

    assert!(matches!(error, WeightLoadError::VisionSidecarInvalid(_)));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn vision_sidecar_returns_none_when_no_sidecar_or_manifest() {
    let dir = vision_sidecar_test_dir("absent");

    let mut name_map = HashMap::new();
    let result = load_vision_sidecar(&dir, &mut name_map).expect("absent sidecar is not an error");

    assert!(result.is_none());
    assert!(name_map.is_empty());
    std::fs::remove_dir_all(&dir).ok();
}
