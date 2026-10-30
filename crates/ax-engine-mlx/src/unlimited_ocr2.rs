//! Independent diagnostic SR-SWA execution for the admitted OCR-2 graph.
//! Full K/V and routing keys are retained; this is not a bounded-memory path.

use std::collections::HashMap;

use ax_engine_core::unlimited_ocr2::{Ocr2PageState, UnlimitedOcr2Config, route_blocks};
use ax_engine_core::{NativeModelManifest, NativeTensorSpec};
use mlx_sys::{
    MlxArray, MlxDtype, ScaledDotProductAttentionMask, add, astype, concatenate, eval, matmul,
    multiply, reshape, rms_norm, scaled_dot_product_attention_with_mask, slice, softmax_precise,
    sum_axis, transpose,
};

use crate::kv_cache::MlxKVCache;
use crate::model::{
    ModelConfig,
    shared::{moe_experts_forward, moe_router_qwen3, qw, shared_expert_forward},
};
use crate::weights::{
    LayerWeights, ModelWeights, QuantizedWeight, WeightLoadError, take_named_weight,
};

pub struct Ocr2LayerWeights {
    pub config: UnlimitedOcr2Config,
    q: QuantizedWeight,
    k: QuantizedWeight,
    q_gain: MlxArray,
    k_gain: MlxArray,
    summary: MlxArray,
}

#[derive(Clone)]
pub(crate) struct Ocr2Cache {
    pub pages: Ocr2PageState,
    tokens: Vec<u32>,
    prompt_len: usize,
    config: UnlimitedOcr2Config,
    keys: Vec<Option<MlxArray>>,
}

impl Ocr2Cache {
    pub(crate) fn trim(&mut self, len: usize) -> bool {
        if len < self.prompt_len || len > self.tokens.len() {
            return false;
        }
        self.tokens.truncate(len);
        self.pages
            .trim(len, &self.tokens[self.prompt_len..], &self.config)
    }
}

#[allow(clippy::expect_used)]
pub(crate) fn prepare(weights: &ModelWeights, cache: &mut MlxKVCache, ids: &[u32], offset: usize) {
    let Some(w) = weights
        .layers
        .first()
        .and_then(|w| w.unlimited_ocr2.as_ref())
    else {
        return;
    };
    if offset == 0 {
        cache.unlimited_ocr2 = Some(Ocr2Cache {
            pages: Ocr2PageState::prefill(ids, &w.config).expect("validated OCR-2 page prompt"),
            tokens: ids.to_vec(),
            prompt_len: ids.len(),
            config: w.config.clone(),
            keys: (0..weights.layers.len()).map(|_| None).collect(),
        });
    } else {
        let state = cache
            .unlimited_ocr2
            .as_mut()
            .expect("OCR-2 requires its image prefill state");
        assert!(
            state.trim(offset),
            "OCR-2 cannot restore an incomplete prefill"
        );
        state.pages.append(ids, &w.config);
        state.tokens.extend_from_slice(ids);
    }
}

pub(crate) fn normalize_vision_layout(map: &mut HashMap<String, MlxArray>) {
    let names: Vec<_> = map
        .keys()
        .filter(|name| {
            name.starts_with("model.sam_model.") || name.starts_with("model.vision_model.")
        })
        .cloned()
        .collect();
    for name in names {
        if let Some(mut value) = map.remove(&name) {
            if value.shape().len() == 4 && name.ends_with(".weight") {
                // Source PyTorch OIHW -> native MLX OHWI.
                value = transpose(&value, &[0, 2, 3, 1], None);
            }
            map.insert(name.trim_start_matches("model.").to_string(), value);
        }
    }
}

pub(crate) fn load_layer(
    m: &NativeModelManifest,
    specs: &[NativeTensorSpec],
    map: &mut HashMap<String, MlxArray>,
    layer: usize,
) -> Result<Option<Ocr2LayerWeights>, WeightLoadError> {
    let Some(cfg) = m.unlimited_ocr2.as_ref() else {
        return Ok(None);
    };
    let prefix = format!("model.layers.{layer}.self_attn.dash_router");
    let q = take_named_weight(specs, map, &format!("{prefix}.q_proj.weight"))?;
    let k = take_named_weight(specs, map, &format!("{prefix}.k_proj.weight"))?;
    let mut plain = |suffix: &str| {
        map.remove(&format!("{prefix}.{suffix}"))
            .ok_or_else(|| WeightLoadError::TensorMissing(format!("{prefix}.{suffix}")))
    };
    let one = MlxArray::from_f32(1.0);
    let q_gain = add(
        &astype(&plain("q_norm.weight")?, MlxDtype::Float32, None),
        &one,
        None,
    );
    let k_gain = add(
        &astype(&plain("k_norm.weight")?, MlxDtype::Float32, None),
        &one,
        None,
    );
    let summary = astype(&plain("head_cls")?, MlxDtype::Float32, None);
    // The checkpoint also carries training/warmup indexer state. Its exact
    // presence/shapes are validated, but DASH inference uses its own router.
    for suffix in [
        "q_proj.weight",
        "k_proj.weight",
        "q_norm.weight",
        "k_norm.weight",
    ] {
        map.remove(&format!("model.layers.{layer}.self_attn.indexer.{suffix}"));
    }
    Ok(Some(Ocr2LayerWeights {
        config: cfg.clone(),
        q,
        k,
        q_gain,
        k_gain,
        summary,
    }))
}

pub(crate) fn load_experts(
    specs: &[NativeTensorSpec],
    map: &mut HashMap<String, MlxArray>,
    layer: usize,
    projection: &str,
) -> Result<QuantizedWeight, WeightLoadError> {
    let mut weights = Vec::with_capacity(64);
    let mut scales = Vec::with_capacity(64);
    let mut quantization = None;
    let mut quantized = None;
    for expert in 0..64 {
        let name = format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight");
        let spec = specs
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| WeightLoadError::TensorMissing(name.clone()))?;
        if expert == 0 {
            quantization = spec.quantization.clone();
            quantized = Some(spec.source_quantized);
        }
        if spec.quantization != quantization || Some(spec.source_quantized) != quantized {
            return Err(WeightLoadError::InvalidLayer(
                "OCR-2 expert quantization must be uniform within a stack".into(),
            ));
        }
        let weight = take_named_weight(specs, map, &name)?;
        weights.push(weight.weight);
        if let Some(scale) = weight.scales {
            scales.push(scale);
        }
    }
    let weight_refs: Vec<_> = weights.iter().collect();
    let scale_refs: Vec<_> = scales.iter().collect();
    let weight = mlx_sys::stack(&weight_refs, 0, None);
    let scales = (!scales.is_empty()).then(|| mlx_sys::stack(&scale_refs, 0, None));
    Ok(QuantizedWeight::with_quantization(
        weight,
        scales,
        None,
        quantization.as_ref(),
    ))
}

fn head_view(x: &MlxArray, heads: usize) -> MlxArray {
    transpose(
        &reshape(x, &[1, x.shape()[1], heads as i32, 128], None),
        &[0, 2, 1, 3],
        None,
    )
}

fn page_rope(x: &MlxArray, coordinates: &[(usize, usize)]) -> MlxArray {
    let mut cos = Vec::with_capacity(coordinates.len() * 128);
    let mut sin = Vec::with_capacity(coordinates.len() * 128);
    for &(tok, page) in coordinates {
        let angles: Vec<_> = (0..64)
            .map(|pair| {
                if pair == 63 {
                    (page % 2) as f32
                } else {
                    tok as f32 / 10000_f32.powf(pair as f32 / 63.0)
                }
            })
            .collect();
        for _ in 0..2 {
            for angle in &angles {
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
    }
    let shape = [1, 1, coordinates.len() as i32, 128];
    let cos = astype(
        &reshape(&MlxArray::from_f32_slice(&cos), &shape, None),
        x.dtype(),
        None,
    );
    let sin = astype(
        &reshape(&MlxArray::from_f32_slice(&sin), &shape, None),
        x.dtype(),
        None,
    );
    let dims = x.shape();
    let left = slice(
        x,
        &[0, 0, 0, 0],
        &[1, dims[1], dims[2], 64],
        &[1, 1, 1, 1],
        None,
    );
    let right = slice(
        x,
        &[0, 0, 0, 64],
        &[1, dims[1], dims[2], 128],
        &[1, 1, 1, 1],
        None,
    );
    let neg = mlx_sys::negative(&right, None);
    let rotated = concatenate(&[&neg, &left], -1, None);
    add(
        &multiply(x, &cos, None),
        &multiply(&rotated, &sin, None),
        None,
    )
}

fn router_project(
    x: &MlxArray,
    w: &QuantizedWeight,
    gain: &MlxArray,
    eps: f32,
    coordinates: &[(usize, usize)],
) -> MlxArray {
    let projected = qw(x, w);
    let norm = rms_norm(
        &astype(&projected, MlxDtype::Float32, None),
        Some(gain),
        eps,
        None,
    );
    let norm = astype(&norm, projected.dtype(), None);
    let rotated = page_rope(&head_view(&norm, 1), coordinates);
    reshape(
        &astype(&rotated, MlxDtype::Float32, None),
        &[coordinates.len() as i32, 128],
        None,
    )
}

#[allow(clippy::expect_used)]
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer: usize,
    offset: usize,
) -> MlxArray {
    let ocr = w.unlimited_ocr2.as_ref().expect("OCR-2 layer weights");
    let seq = hidden.shape()[1] as usize;
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
    let coordinates = cache
        .unlimited_ocr2
        .as_ref()
        .expect("OCR-2 page state")
        .pages
        .coordinates[offset..offset + seq]
        .to_vec();
    let q = page_rope(
        &head_view(
            &qw(&normed, w.q_proj.as_ref().expect("OCR-2 Q")),
            cfg.n_heads,
        ),
        &coordinates,
    );
    let k = page_rope(
        &head_view(
            &qw(&normed, w.k_proj.as_ref().expect("OCR-2 K")),
            cfg.n_kv_heads,
        ),
        &coordinates,
    );
    let v = head_view(
        &qw(&normed, w.v_proj.as_ref().expect("OCR-2 V")),
        cfg.n_kv_heads,
    );
    let (keys, values) = cache.append(layer, k, v);
    let query_index = router_project(&normed, &ocr.q, &ocr.q_gain, cfg.rms_norm_eps, &coordinates);
    let new_index = router_project(&normed, &ocr.k, &ocr.k_gain, cfg.rms_norm_eps, &coordinates);
    let state = cache.unlimited_ocr2.as_mut().expect("OCR-2 page state");
    let all_index = if let Some(previous) = state.keys[layer].as_ref() {
        let prefix = slice(previous, &[0, 0], &[offset as i32, 128], &[1, 1], None);
        concatenate(&[&prefix, &new_index], 0, None)
    } else {
        new_index
    };
    state.keys[layer] = Some(all_index.clone());
    let len = offset + seq;
    let blocks = len.div_ceil(ocr.config.block_size);
    let mut summaries = Vec::with_capacity(blocks);
    for block in 0..blocks {
        let begin = block * ocr.config.block_size;
        let end = (begin + ocr.config.block_size).min(len);
        let rows = slice(
            &all_index,
            &[begin as i32, 0],
            &[end as i32, 128],
            &[1, 1],
            None,
        );
        let scores = multiply(
            &sum_axis(&multiply(&rows, &ocr.summary, None), -1, true, None),
            &MlxArray::from_f32(1.0 / 128_f32.sqrt()),
            None,
        );
        let weights = softmax_precise(&scores, 0, None);
        summaries.push(sum_axis(&multiply(&rows, &weights, None), 0, false, None));
    }
    let refs: Vec<_> = summaries.iter().collect();
    let centroids = mlx_sys::stack(&refs, 0, None);
    let scores = multiply(
        &matmul(&query_index, &transpose(&centroids, &[1, 0], None), None),
        &MlxArray::from_f32(1.0 / 128_f32.sqrt()),
        None,
    );
    eval(&[&scores]);
    let scores = scores.data_f32();
    assert!(
        scores.iter().all(|score| score.is_finite()),
        "OCR-2 router produced non-finite scores"
    );
    let mut outputs = Vec::new();
    // Query chunks bound temporary attention scores and masks. The retained
    // state itself is intentionally unbounded in this diagnostic version.
    for start in (0..seq).step_by(64) {
        let end = (start + 64).min(seq);
        let mut mask = vec![f32::NEG_INFINITY; (end - start) * len];
        for row in start..end {
            let position = offset + row;
            let allowed = state
                .pages
                .candidate_blocks(position, ocr.config.block_size);
            let selected = route_blocks(
                &scores[row * blocks..(row + 1) * blocks],
                &allowed,
                position,
                &ocr.config,
                layer,
            );
            for (block, bias) in selected {
                let first = block * ocr.config.block_size;
                let last = (first + ocr.config.block_size).min(position + 1);
                for key in first..last {
                    mask[(row - start) * len + key] = bias;
                }
            }
            for key in (position + 1).saturating_sub(128)..=position {
                mask[(row - start) * len + key] = 0.0;
            }
        }
        let mask = reshape(
            &MlxArray::from_f32_slice(&mask),
            &[1, 1, (end - start) as i32, len as i32],
            None,
        );
        let queries = slice(
            &q,
            &[0, 0, start as i32, 0],
            &[1, cfg.n_heads as i32, end as i32, 128],
            &[1, 1, 1, 1],
            None,
        );
        outputs.push(scaled_dot_product_attention_with_mask(
            &astype(&queries, MlxDtype::Float32, None),
            &astype(&keys, MlxDtype::Float32, None),
            &astype(&values, MlxDtype::Float32, None),
            1.0 / 128_f32.sqrt(),
            ScaledDotProductAttentionMask::Array(&mask),
            None,
        ));
    }
    let refs: Vec<_> = outputs.iter().collect();
    let out = concatenate(&refs, 2, None);
    let flat = reshape(
        &transpose(&out, &[0, 2, 1, 3], None),
        &[1, seq as i32, cfg.hidden_size as i32],
        None,
    );
    let attention = qw(
        &astype(&flat, hidden.dtype(), None),
        w.o_proj.as_ref().expect("OCR-2 O"),
    );
    let residual = add(hidden, &attention, None);
    let ffn_input = rms_norm(&residual, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
    let ffn = if w.router_proj.is_some() {
        let (ids, weights) = moe_router_qwen3(cfg, w, &astype(&ffn_input, MlxDtype::Float32, None));
        add(
            &moe_experts_forward(cfg, w, &ffn_input, &ids, &weights),
            &shared_expert_forward(cfg, w, &ffn_input),
            None,
        )
    } else {
        let gate = qw(&ffn_input, w.gate_proj.as_ref().expect("OCR-2 dense gate"));
        let up = qw(&ffn_input, w.up_proj.as_ref().expect("OCR-2 dense up"));
        qw(
            &multiply(&mlx_sys::ops::silu(&gate, None), &up, None),
            w.down_proj.as_ref().expect("OCR-2 dense down"),
        )
    };
    add(&residual, &ffn, None)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn page_rope_preserves_dtype_and_rotation_pairs() {
        let mut data = vec![0.0; 128];
        data[0] = 1.0;
        data[63] = 1.0;
        let x = astype(
            &reshape(&MlxArray::from_f32_slice(&data), &[1, 1, 1, 128], None),
            MlxDtype::Bfloat16,
            None,
        );
        let y = page_rope(&x, &[(2, 3)]);
        assert_eq!(y.dtype(), MlxDtype::Bfloat16);
        let f32 = astype(&y, MlxDtype::Float32, None);
        eval(&[&f32]);
        let values = f32.data_f32();
        assert!((values[0] - 2_f32.cos()).abs() < 0.005);
        assert!((values[64] - 2_f32.sin()).abs() < 0.005);
        assert!((values[63] - 1_f32.cos()).abs() < 0.005);
        assert!((values[127] - 1_f32.sin()).abs() < 0.005);
        let even = page_rope(&x, &[(2, 4)]);
        let even = astype(&even, MlxDtype::Float32, None);
        eval(&[&even]);
        assert_eq!(even.data_f32()[63], 1.0);
        assert_eq!(even.data_f32()[127], 0.0);
    }

    #[test]
    fn source_experts_stack_by_id_and_missing_members_fail() {
        use ax_engine_core::{NativeTensorDataType, NativeTensorRole};
        let mut specs = Vec::new();
        let mut map = HashMap::new();
        // Reverse insertion makes both hash order and lexical order unsuitable.
        for expert in (0..64).rev() {
            let name = format!("model.layers.1.mlp.experts.{expert}.up_proj.weight");
            map.insert(
                name.clone(),
                reshape(&MlxArray::from_f32(expert as f32), &[1, 1], None),
            );
            specs.push(NativeTensorSpec {
                name,
                role: NativeTensorRole::Other,
                layer_index: None,
                dtype: NativeTensorDataType::F32,
                source_tensor_type: None,
                source_quantized: false,
                quantization: None,
                quantized_source: None,
                shape: vec![1, 1],
                file: "model.safetensors".into(),
                offset_bytes: 0,
                length_bytes: 4,
            });
        }
        let stacked = load_experts(&specs, &mut map, 1, "up_proj").expect("complete stack");
        eval(&[&stacked.weight]);
        assert_eq!(stacked.weight.shape(), [64, 1, 1]);
        assert_eq!(
            stacked.weight.data_f32(),
            &(0..64).map(|id| id as f32).collect::<Vec<_>>()
        );
        assert!(load_experts(&specs, &mut map, 1, "up_proj").is_err());
    }
}
