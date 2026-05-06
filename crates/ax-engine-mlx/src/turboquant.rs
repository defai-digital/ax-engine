use std::collections::VecDeque;

use ax_engine_core::MlxTurboQuantPreset;
use thiserror::Error;

use crate::model::ModelConfig;

pub type FullPrecisionKvTokenVectors = (Vec<f32>, Vec<f32>);

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TurboQuantCodecError {
    #[error("TurboQuant reference codec requires a non-empty vector")]
    EmptyVector,
    #[error("Hadamard rotation requires power-of-two dimensions, got {0}")]
    NonPowerOfTwoDimension(usize),
    #[error("unsupported key bit width {0}")]
    UnsupportedKeyBits(u32),
    #[error("unsupported packed bit width {0}")]
    UnsupportedPackedBits(u32),
    #[error("{bit_width}-bit TurboQuant centroids require {expected} entries, got {actual}")]
    InvalidCentroidCount {
        bit_width: u32,
        expected: usize,
        actual: usize,
    },
    #[error("packed index {index} value {value} exceeds {bit_width}-bit range")]
    PackedIndexOutOfRange {
        bit_width: u32,
        index: usize,
        value: u8,
    },
    #[error("packed buffer is too short for {index_count} indices at {bit_width} bits")]
    PackedBufferTooShort { bit_width: u32, index_count: usize },
    #[error("KV prototype requires matching K/V vector lengths, got K={key_len}, V={value_len}")]
    MismatchedKvVectorLengths { key_len: usize, value_len: usize },
    #[error("KV prototype expected vector dimension {expected}, got {actual}")]
    MismatchedVectorDimension { expected: usize, actual: usize },
    #[error("TurboQuant reference attention requires at least one KV token")]
    EmptyKvHistory,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedKeyVector {
    pub dim: usize,
    pub bit_width: u32,
    pub l2_norm: f32,
    pub packed_indices: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedValueGroups {
    pub element_count: usize,
    pub group_size: usize,
    pub mins: Vec<f32>,
    pub scales: Vec<f32>,
    pub packed_values: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurboQuantLayerSupportReason {
    Eligible,
    LinearAttention,
    SlidingWindow,
    KvShared,
    UnsupportedHeadDim,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantLayerSupport {
    pub layer_index: usize,
    pub head_dim: usize,
    pub reason: TurboQuantLayerSupportReason,
}

impl TurboQuantLayerSupport {
    pub fn is_eligible(self) -> bool {
        self.reason == TurboQuantLayerSupportReason::Eligible
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TurboQuantSupportReport {
    pub layers: Vec<TurboQuantLayerSupport>,
    pub eligible_layers: usize,
    pub linear_attention_layers: usize,
    pub sliding_window_layers: usize,
    pub kv_shared_layers: usize,
    pub unsupported_head_dim_layers: usize,
}

impl TurboQuantSupportReport {
    pub fn eligible_layer_mask(&self) -> Vec<bool> {
        self.layers
            .iter()
            .map(|layer| layer.is_eligible())
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TurboQuantKvPrototypeConfig {
    pub preset: MlxTurboQuantPreset,
    pub vector_dim: usize,
    pub hot_window_tokens: usize,
    pub value_group_size: usize,
}

impl TurboQuantKvPrototypeConfig {
    pub fn new(
        preset: MlxTurboQuantPreset,
        vector_dim: usize,
        hot_window_tokens: usize,
        value_group_size: usize,
    ) -> Result<Self, TurboQuantCodecError> {
        if vector_dim == 0 {
            return Err(TurboQuantCodecError::EmptyVector);
        }
        if !vector_dim.is_power_of_two() {
            return Err(TurboQuantCodecError::NonPowerOfTwoDimension(vector_dim));
        }
        key_centroids(preset.key_bits())?;
        packed_index_bytes(vector_dim, preset.value_bits())?;
        Ok(Self {
            preset,
            vector_dim,
            hot_window_tokens,
            value_group_size: value_group_size.max(1),
        })
    }
}

pub fn turboquant_support_report(
    cfg: &ModelConfig,
    preset: MlxTurboQuantPreset,
) -> Result<TurboQuantSupportReport, TurboQuantCodecError> {
    key_centroids(preset.key_bits())?;
    let mut report = TurboQuantSupportReport {
        layers: Vec::with_capacity(cfg.layer_count),
        eligible_layers: 0,
        linear_attention_layers: 0,
        sliding_window_layers: 0,
        kv_shared_layers: 0,
        unsupported_head_dim_layers: 0,
    };

    for layer_index in 0..cfg.layer_count {
        let layer_cfg = cfg.layer_configs.get(layer_index);
        let head_dim = layer_cfg
            .map(|layer| layer.head_dim)
            .unwrap_or(cfg.head_dim);
        let reason = if cfg.is_linear_attention_layer(layer_index) {
            TurboQuantLayerSupportReason::LinearAttention
        } else if layer_cfg.and_then(|layer| layer.sliding_window).is_some() {
            TurboQuantLayerSupportReason::SlidingWindow
        } else if layer_cfg.and_then(|layer| layer.kv_source_layer).is_some() {
            TurboQuantLayerSupportReason::KvShared
        } else if !head_dim.is_power_of_two() {
            TurboQuantLayerSupportReason::UnsupportedHeadDim
        } else {
            TurboQuantLayerSupportReason::Eligible
        };

        match reason {
            TurboQuantLayerSupportReason::Eligible => {
                report.eligible_layers = report.eligible_layers.saturating_add(1);
            }
            TurboQuantLayerSupportReason::LinearAttention => {
                report.linear_attention_layers = report.linear_attention_layers.saturating_add(1);
            }
            TurboQuantLayerSupportReason::SlidingWindow => {
                report.sliding_window_layers = report.sliding_window_layers.saturating_add(1);
            }
            TurboQuantLayerSupportReason::KvShared => {
                report.kv_shared_layers = report.kv_shared_layers.saturating_add(1);
            }
            TurboQuantLayerSupportReason::UnsupportedHeadDim => {
                report.unsupported_head_dim_layers =
                    report.unsupported_head_dim_layers.saturating_add(1);
            }
        }

        report.layers.push(TurboQuantLayerSupport {
            layer_index,
            head_dim,
            reason,
        });
    }

    Ok(report)
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TurboQuantKvPrototypeStats {
    pub cold_tokens: usize,
    pub hot_tokens: usize,
    pub full_precision_cold_bytes: usize,
    pub compressed_cold_bytes: usize,
    pub estimated_saved_bytes: usize,
}

impl TurboQuantKvPrototypeStats {
    pub fn compression_ratio_milli(&self) -> u32 {
        if self.full_precision_cold_bytes == 0 {
            0
        } else {
            self.compressed_cold_bytes
                .saturating_mul(1000)
                .saturating_div(self.full_precision_cold_bytes)
                .min(u32::MAX as usize) as u32
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantKvPrototypeStore {
    config: TurboQuantKvPrototypeConfig,
    cold: Vec<CompressedKvToken>,
    hot: VecDeque<FullPrecisionKvToken>,
}

#[derive(Clone, Debug, PartialEq)]
struct CompressedKvToken {
    key: QuantizedKeyVector,
    value: QuantizedValueGroups,
}

#[derive(Clone, Debug, PartialEq)]
struct FullPrecisionKvToken {
    key: Vec<f32>,
    value: Vec<f32>,
}

impl TurboQuantKvPrototypeStore {
    pub fn new(config: TurboQuantKvPrototypeConfig) -> Self {
        Self {
            config,
            cold: Vec::new(),
            hot: VecDeque::new(),
        }
    }

    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), TurboQuantCodecError> {
        self.validate_kv_token(key, value)?;
        self.hot.push_back(FullPrecisionKvToken {
            key: key.to_vec(),
            value: value.to_vec(),
        });
        self.evict_cold_until_hot_window()
    }

    pub fn append_many(
        &mut self,
        tokens: impl IntoIterator<Item = (Vec<f32>, Vec<f32>)>,
    ) -> Result<(), TurboQuantCodecError> {
        for (key, value) in tokens {
            self.append(&key, &value)?;
        }
        Ok(())
    }

    pub fn debug_reconstruct(
        &self,
    ) -> Result<Vec<FullPrecisionKvTokenVectors>, TurboQuantCodecError> {
        let mut tokens = Vec::with_capacity(self.token_count());
        for token in &self.cold {
            tokens.push((
                decode_key_vector(&token.key)?,
                decode_value_groups_4bit(&token.value)?,
            ));
        }
        tokens.extend(
            self.hot
                .iter()
                .map(|token| (token.key.clone(), token.value.clone())),
        );
        Ok(tokens)
    }

    pub fn debug_decode_attention(&self, query: &[f32]) -> Result<Vec<f32>, TurboQuantCodecError> {
        let tokens = self.debug_reconstruct()?;
        reference_decode_attention(query, &tokens)
    }

    pub fn stats(&self) -> TurboQuantKvPrototypeStats {
        let cold_tokens = self.cold.len();
        let full_precision_cold_bytes = cold_tokens
            .saturating_mul(self.config.vector_dim)
            .saturating_mul(std::mem::size_of::<f32>())
            .saturating_mul(2);
        let compressed_cold_bytes = self
            .cold
            .iter()
            .map(|token| {
                token.key.packed_indices.len()
                    + std::mem::size_of::<f32>()
                    + token.value.packed_values.len()
                    + token
                        .value
                        .mins
                        .len()
                        .saturating_mul(std::mem::size_of::<f32>())
                    + token
                        .value
                        .scales
                        .len()
                        .saturating_mul(std::mem::size_of::<f32>())
            })
            .sum::<usize>();

        TurboQuantKvPrototypeStats {
            cold_tokens,
            hot_tokens: self.hot.len(),
            full_precision_cold_bytes,
            compressed_cold_bytes,
            estimated_saved_bytes: full_precision_cold_bytes.saturating_sub(compressed_cold_bytes),
        }
    }

    pub fn token_count(&self) -> usize {
        self.cold.len().saturating_add(self.hot.len())
    }

    pub fn cold_token_count(&self) -> usize {
        self.cold.len()
    }

    pub fn hot_token_count(&self) -> usize {
        self.hot.len()
    }

    fn validate_kv_token(&self, key: &[f32], value: &[f32]) -> Result<(), TurboQuantCodecError> {
        if key.len() != value.len() {
            return Err(TurboQuantCodecError::MismatchedKvVectorLengths {
                key_len: key.len(),
                value_len: value.len(),
            });
        }
        if key.len() != self.config.vector_dim {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: self.config.vector_dim,
                actual: key.len(),
            });
        }
        Ok(())
    }

    fn evict_cold_until_hot_window(&mut self) -> Result<(), TurboQuantCodecError> {
        while self.hot.len() > self.config.hot_window_tokens {
            let token = self
                .hot
                .pop_front()
                .expect("hot len is above window so a token must exist");
            self.cold.push(CompressedKvToken {
                key: encode_key_vector(&token.key, self.config.preset)?,
                value: encode_value_groups_4bit(&token.value, self.config.value_group_size)?,
            });
        }
        Ok(())
    }
}

pub fn reference_decode_attention(
    query: &[f32],
    kv_tokens: &[FullPrecisionKvTokenVectors],
) -> Result<Vec<f32>, TurboQuantCodecError> {
    validate_key_shape(query)?;
    if kv_tokens.is_empty() {
        return Err(TurboQuantCodecError::EmptyKvHistory);
    }

    let value_dim = kv_tokens[0].1.len();
    let mut scores = Vec::with_capacity(kv_tokens.len());
    let scale = (query.len() as f32).sqrt().recip();

    for (key, value) in kv_tokens {
        validate_attention_token(query.len(), value_dim, key, value)?;
        let score = query
            .iter()
            .zip(key)
            .map(|(left, right)| left * right)
            .sum::<f32>()
            * scale;
        scores.push(score);
    }

    let max_score = scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, score| acc.max(score));
    let weights = scores
        .iter()
        .map(|score| (*score - max_score).exp())
        .collect::<Vec<_>>();
    let weight_sum = weights.iter().sum::<f32>().max(f32::EPSILON);
    let mut output = vec![0.0f32; value_dim];

    for (weight, (_, value)) in weights.iter().zip(kv_tokens) {
        let normalized_weight = *weight / weight_sum;
        for (out, value) in output.iter_mut().zip(value) {
            *out += normalized_weight * value;
        }
    }

    Ok(output)
}

fn validate_attention_token(
    query_dim: usize,
    value_dim: usize,
    key: &[f32],
    value: &[f32],
) -> Result<(), TurboQuantCodecError> {
    if key.len() != query_dim {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: query_dim,
            actual: key.len(),
        });
    }
    if value.len() != value_dim {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: value_dim,
            actual: value.len(),
        });
    }
    Ok(())
}

pub fn encode_key_vector(
    vector: &[f32],
    preset: MlxTurboQuantPreset,
) -> Result<QuantizedKeyVector, TurboQuantCodecError> {
    let bit_width = preset.key_bits();
    let centroids = key_centroids(bit_width)?;
    encode_key_vector_with_centroids(vector, bit_width, &centroids)
}

pub fn decode_key_vector(encoded: &QuantizedKeyVector) -> Result<Vec<f32>, TurboQuantCodecError> {
    let centroids = key_centroids(encoded.bit_width)?;
    decode_key_vector_with_centroids(encoded, &centroids)
}

pub fn encode_key_vector_with_centroids(
    vector: &[f32],
    bit_width: u32,
    centroids: &[f32],
) -> Result<QuantizedKeyVector, TurboQuantCodecError> {
    validate_key_shape(vector)?;
    validate_centroid_count(bit_width, centroids)?;

    let l2_norm = l2_norm(vector);
    let norm = l2_norm.max(f32::EPSILON);
    let mut rotated: Vec<f32> = vector.iter().map(|value| value / norm).collect();
    hadamard_in_place(&mut rotated)?;

    let indices = rotated
        .iter()
        .map(|value| nearest_centroid_index(*value, centroids))
        .collect::<Vec<_>>();
    let packed_indices = pack_indices(&indices, bit_width)?;

    Ok(QuantizedKeyVector {
        dim: vector.len(),
        bit_width,
        l2_norm,
        packed_indices,
    })
}

pub fn decode_key_vector_with_centroids(
    encoded: &QuantizedKeyVector,
    centroids: &[f32],
) -> Result<Vec<f32>, TurboQuantCodecError> {
    if encoded.dim == 0 {
        return Err(TurboQuantCodecError::EmptyVector);
    }
    if !encoded.dim.is_power_of_two() {
        return Err(TurboQuantCodecError::NonPowerOfTwoDimension(encoded.dim));
    }
    validate_centroid_count(encoded.bit_width, centroids)?;

    let indices = unpack_indices(&encoded.packed_indices, encoded.dim, encoded.bit_width)?;
    let mut rotated = indices
        .into_iter()
        .map(|index| centroids[index as usize])
        .collect::<Vec<_>>();
    hadamard_in_place(&mut rotated)?;

    for value in &mut rotated {
        *value *= encoded.l2_norm;
    }

    Ok(rotated)
}

pub fn nearest_centroid_index(value: f32, centroids: &[f32]) -> u8 {
    let mut best_index = 0usize;
    let mut best_distance = f32::INFINITY;
    for (index, centroid) in centroids.iter().enumerate() {
        let distance = (value - centroid).abs();
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }
    best_index.min(u8::MAX as usize) as u8
}

pub fn key_centroids(bit_width: u32) -> Result<Vec<f32>, TurboQuantCodecError> {
    match bit_width {
        3 | 4 | 8 => {
            let levels = 1usize << bit_width;
            let max_level = levels.saturating_sub(1).max(1) as f32;
            Ok((0..levels)
                .map(|idx| -1.0 + 2.0 * idx as f32 / max_level)
                .collect())
        }
        other => Err(TurboQuantCodecError::UnsupportedKeyBits(other)),
    }
}

pub fn packed_index_bytes(
    index_count: usize,
    bit_width: u32,
) -> Result<usize, TurboQuantCodecError> {
    if !matches!(bit_width, 3 | 4 | 8) {
        return Err(TurboQuantCodecError::UnsupportedPackedBits(bit_width));
    }
    Ok(index_count.saturating_mul(bit_width as usize).div_ceil(8))
}

pub fn packed_kv_bytes_per_token(
    elements_per_token: usize,
    key_bits: u32,
    value_bits: u32,
) -> Result<usize, TurboQuantCodecError> {
    let key_bytes = packed_index_bytes(elements_per_token, key_bits)?;
    let value_bytes = packed_index_bytes(elements_per_token, value_bits)?;
    Ok(key_bytes.saturating_add(value_bytes))
}

pub fn pack_indices(indices: &[u8], bit_width: u32) -> Result<Vec<u8>, TurboQuantCodecError> {
    if !matches!(bit_width, 3 | 4 | 8) {
        return Err(TurboQuantCodecError::UnsupportedPackedBits(bit_width));
    }

    let max_value = (1u16 << bit_width) - 1;
    let mut packed = vec![0u8; packed_index_bytes(indices.len(), bit_width)?];

    for (index, value) in indices.iter().copied().enumerate() {
        if value as u16 > max_value {
            return Err(TurboQuantCodecError::PackedIndexOutOfRange {
                bit_width,
                index,
                value,
            });
        }

        let bit_offset = index * bit_width as usize;
        for bit in 0..bit_width as usize {
            if ((value >> bit) & 1) == 1 {
                let target_bit = bit_offset + bit;
                packed[target_bit / 8] |= 1 << (target_bit % 8);
            }
        }
    }

    Ok(packed)
}

pub fn unpack_indices(
    packed: &[u8],
    index_count: usize,
    bit_width: u32,
) -> Result<Vec<u8>, TurboQuantCodecError> {
    if !matches!(bit_width, 3 | 4 | 8) {
        return Err(TurboQuantCodecError::UnsupportedPackedBits(bit_width));
    }

    let required_bytes = packed_index_bytes(index_count, bit_width)?;
    if packed.len() < required_bytes {
        return Err(TurboQuantCodecError::PackedBufferTooShort {
            bit_width,
            index_count,
        });
    }

    let mut indices = Vec::with_capacity(index_count);
    for index in 0..index_count {
        let bit_offset = index * bit_width as usize;
        let mut value = 0u8;
        for bit in 0..bit_width as usize {
            let source_bit = bit_offset + bit;
            let bit_value = (packed[source_bit / 8] >> (source_bit % 8)) & 1;
            value |= bit_value << bit;
        }
        indices.push(value);
    }

    Ok(indices)
}

pub fn encode_value_groups_4bit(
    values: &[f32],
    group_size: usize,
) -> Result<QuantizedValueGroups, TurboQuantCodecError> {
    if values.is_empty() {
        return Err(TurboQuantCodecError::EmptyVector);
    }
    let group_size = group_size.max(1);
    let group_count = values.len().div_ceil(group_size);
    let mut mins = Vec::with_capacity(group_count);
    let mut scales = Vec::with_capacity(group_count);
    let mut quantized = Vec::with_capacity(values.len());

    for group in values.chunks(group_size) {
        let min = group
            .iter()
            .fold(f32::INFINITY, |acc, value| acc.min(*value));
        let max = group
            .iter()
            .fold(f32::NEG_INFINITY, |acc, value| acc.max(*value));
        let scale = if max > min { (max - min) / 15.0 } else { 1.0 };
        mins.push(min);
        scales.push(scale);
        quantized.extend(group.iter().map(|value| {
            if max > min {
                ((*value - min) / scale).round().clamp(0.0, 15.0) as u8
            } else {
                0
            }
        }));
    }

    Ok(QuantizedValueGroups {
        element_count: values.len(),
        group_size,
        mins,
        scales,
        packed_values: pack_indices(&quantized, 4)?,
    })
}

pub fn decode_value_groups_4bit(
    encoded: &QuantizedValueGroups,
) -> Result<Vec<f32>, TurboQuantCodecError> {
    let quantized = unpack_indices(&encoded.packed_values, encoded.element_count, 4)?;
    let mut values = Vec::with_capacity(encoded.element_count);

    for (index, quantized_value) in quantized.into_iter().enumerate() {
        let group_index = index / encoded.group_size.max(1);
        let min = encoded.mins[group_index];
        let scale = encoded.scales[group_index];
        values.push(min + scale * quantized_value as f32);
    }

    Ok(values)
}

pub fn hadamard_in_place(values: &mut [f32]) -> Result<(), TurboQuantCodecError> {
    validate_key_shape(values)?;
    let n = values.len();
    let mut stride = 1usize;
    while stride < n {
        let step = stride * 2;
        for base in (0..n).step_by(step) {
            for offset in 0..stride {
                let left = values[base + offset];
                let right = values[base + offset + stride];
                values[base + offset] = left + right;
                values[base + offset + stride] = left - right;
            }
        }
        stride = step;
    }

    let scale = (n as f32).sqrt().recip();
    for value in values {
        *value *= scale;
    }

    Ok(())
}

fn validate_key_shape(values: &[f32]) -> Result<(), TurboQuantCodecError> {
    if values.is_empty() {
        return Err(TurboQuantCodecError::EmptyVector);
    }
    if !values.len().is_power_of_two() {
        return Err(TurboQuantCodecError::NonPowerOfTwoDimension(values.len()));
    }
    Ok(())
}

fn validate_centroid_count(bit_width: u32, centroids: &[f32]) -> Result<(), TurboQuantCodecError> {
    let expected = match bit_width {
        3 | 4 | 8 => 1usize << bit_width,
        other => return Err(TurboQuantCodecError::UnsupportedKeyBits(other)),
    };
    if centroids.len() != expected {
        return Err(TurboQuantCodecError::InvalidCentroidCount {
            bit_width,
            expected,
            actual: centroids.len(),
        });
    }
    Ok(())
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::model::{LayerConfig, LinearAttentionConfig};

    fn support_test_model(layer_count: usize, head_dim: usize) -> ModelConfig {
        ModelConfig {
            layer_count,
            hidden_size: 0,
            intermediate_size: 0,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim,
            vocab_size: 0,
            rope_theta: 10_000.0,
            rope_dims: head_dim,
            attn_output_gate: false,
            query_scale: 1.0,
            final_logit_softcapping: None,
            moe_expert_count: 0,
            moe_experts_per_token: 0,
            moe_expert_intermediate_size: 0,
            layer_configs: Vec::new(),
            gemma4_moe_router: false,
            uses_geglu: false,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            linear_attention: None,
        }
    }

    fn test_layer(head_dim: usize) -> LayerConfig {
        LayerConfig {
            head_dim,
            rope_theta: 10_000.0,
            rope_dims: head_dim,
            sliding_window: None,
            kv_source_layer: None,
            v_norm_no_scale: false,
        }
    }

    fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
        let dot = left
            .iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        let left_norm = l2_norm(left).max(f32::EPSILON);
        let right_norm = l2_norm(right).max(f32::EPSILON);
        dot / (left_norm * right_norm)
    }

    fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, |acc, diff| acc.max(diff))
    }

    #[test]
    fn support_report_marks_dense_full_attention_layers_eligible() {
        let cfg = support_test_model(3, 128);
        let report =
            turboquant_support_report(&cfg, MlxTurboQuantPreset::K8V4).expect("support report");

        assert_eq!(report.eligible_layers, 3);
        assert_eq!(report.linear_attention_layers, 0);
        assert_eq!(report.sliding_window_layers, 0);
        assert_eq!(report.kv_shared_layers, 0);
        assert_eq!(report.unsupported_head_dim_layers, 0);
        assert_eq!(report.eligible_layer_mask(), vec![true, true, true]);
        assert!(report.layers.iter().all(|layer| {
            layer.head_dim == 128 && layer.reason == TurboQuantLayerSupportReason::Eligible
        }));
    }

    #[test]
    fn support_report_fails_closed_for_mixed_attention_layouts() {
        let mut cfg = support_test_model(4, 128);
        let mut sliding = test_layer(128);
        sliding.sliding_window = Some(1024);
        let full = test_layer(128);
        let mut shared = test_layer(128);
        shared.kv_source_layer = Some(1);
        let unsupported = test_layer(96);
        cfg.layer_configs = vec![sliding, full, shared, unsupported];

        let report =
            turboquant_support_report(&cfg, MlxTurboQuantPreset::K8V4).expect("support report");

        assert_eq!(report.eligible_layers, 1);
        assert_eq!(report.sliding_window_layers, 1);
        assert_eq!(report.kv_shared_layers, 1);
        assert_eq!(report.unsupported_head_dim_layers, 1);
        assert_eq!(
            report.eligible_layer_mask(),
            vec![false, true, false, false]
        );
        assert_eq!(
            report
                .layers
                .iter()
                .map(|layer| layer.reason)
                .collect::<Vec<_>>(),
            vec![
                TurboQuantLayerSupportReason::SlidingWindow,
                TurboQuantLayerSupportReason::Eligible,
                TurboQuantLayerSupportReason::KvShared,
                TurboQuantLayerSupportReason::UnsupportedHeadDim,
            ]
        );
    }

    #[test]
    fn support_report_excludes_linear_attention_layers() {
        let mut cfg = support_test_model(4, 128);
        cfg.linear_attention = Some(LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 2,
            num_key_heads: 2,
            key_head_dim: 64,
            value_head_dim: 64,
            conv_kernel_dim: 4,
        });

        let report =
            turboquant_support_report(&cfg, MlxTurboQuantPreset::K4V4).expect("support report");

        assert_eq!(report.eligible_layers, 1);
        assert_eq!(report.linear_attention_layers, 3);
        assert_eq!(
            report.eligible_layer_mask(),
            vec![false, false, false, true]
        );
        assert_eq!(
            report
                .layers
                .iter()
                .map(|layer| layer.reason)
                .collect::<Vec<_>>(),
            vec![
                TurboQuantLayerSupportReason::LinearAttention,
                TurboQuantLayerSupportReason::LinearAttention,
                TurboQuantLayerSupportReason::LinearAttention,
                TurboQuantLayerSupportReason::Eligible,
            ]
        );
    }

    #[test]
    fn pack_unpack_round_trips_three_and_four_bit_indices() {
        let three_bit = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 7, 3];
        let packed = pack_indices(&three_bit, 3).expect("3-bit pack should work");
        assert_eq!(
            unpack_indices(&packed, three_bit.len(), 3).expect("3-bit unpack should work"),
            three_bit
        );

        let four_bit = vec![0, 15, 1, 14, 2, 13, 3, 12, 4];
        let packed = pack_indices(&four_bit, 4).expect("4-bit pack should work");
        assert_eq!(
            unpack_indices(&packed, four_bit.len(), 4).expect("4-bit unpack should work"),
            four_bit
        );
    }

    #[test]
    fn pack_rejects_values_that_do_not_fit_width() {
        let error = pack_indices(&[0, 8], 3).expect_err("8 does not fit in 3 bits");
        assert_eq!(
            error,
            TurboQuantCodecError::PackedIndexOutOfRange {
                bit_width: 3,
                index: 1,
                value: 8,
            }
        );
    }

    #[test]
    fn packed_kv_bytes_per_token_matches_bit_width_contract() {
        assert_eq!(packed_index_bytes(8, 3).expect("3-bit packed bytes"), 3);
        assert_eq!(packed_index_bytes(8, 4).expect("4-bit packed bytes"), 4);
        assert_eq!(packed_kv_bytes_per_token(8, 8, 4).expect("K8V4 bytes"), 12);
        assert_eq!(packed_kv_bytes_per_token(8, 3, 4).expect("K3V4 bytes"), 7);
    }

    #[test]
    fn centroid_lookup_prefers_nearest_boundary() {
        let centroids = vec![-1.0, -0.25, 0.25, 1.0];
        assert_eq!(nearest_centroid_index(-0.80, &centroids), 0);
        assert_eq!(nearest_centroid_index(-0.10, &centroids), 1);
        assert_eq!(nearest_centroid_index(0.40, &centroids), 2);
        assert_eq!(nearest_centroid_index(0.90, &centroids), 3);
    }

    #[test]
    fn key_codec_rejects_wrong_centroid_count() {
        let error = encode_key_vector_with_centroids(&[1.0, -1.0, 0.5, -0.5], 4, &[0.0, 1.0])
            .expect_err("wrong centroid count should fail closed");
        assert_eq!(
            error,
            TurboQuantCodecError::InvalidCentroidCount {
                bit_width: 4,
                expected: 16,
                actual: 2,
            }
        );
    }

    #[test]
    fn hadamard_is_self_inverse_for_power_of_two_vectors() {
        let original = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 0.0];
        let mut transformed = original.clone();
        hadamard_in_place(&mut transformed).expect("first transform");
        hadamard_in_place(&mut transformed).expect("second transform");

        for (actual, expected) in transformed.iter().zip(original) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn key_codec_preserves_direction_for_fixed_vector() {
        let vector = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 0.0];
        let encoded =
            encode_key_vector(&vector, MlxTurboQuantPreset::K8V4).expect("encode key vector");
        let decoded = decode_key_vector(&encoded).expect("decode key vector");

        assert_eq!(encoded.dim, vector.len());
        assert_eq!(encoded.bit_width, 8);
        assert!(cosine_similarity(&vector, &decoded) > 0.999);
        let mse = vector
            .iter()
            .zip(&decoded)
            .map(|(expected, actual)| (expected - actual).powi(2))
            .sum::<f32>()
            / vector.len() as f32;
        assert!(mse < 0.0002, "mse was {mse}");
    }

    #[test]
    fn value_group_codec_round_trips_with_small_error() {
        let values = vec![
            -1.0, -0.6, -0.2, 0.0, 0.2, 0.5, 0.9, 1.0, 2.0, 2.1, 2.4, 2.9,
        ];
        let encoded = encode_value_groups_4bit(&values, 4).expect("encode value groups");
        let decoded = decode_value_groups_4bit(&encoded).expect("decode value groups");

        assert_eq!(encoded.element_count, values.len());
        assert_eq!(encoded.group_size, 4);
        assert_eq!(encoded.mins.len(), 3);
        assert_eq!(decoded.len(), values.len());
        for (expected, actual) in values.iter().zip(decoded) {
            assert!(
                (expected - actual).abs() <= 0.08,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn key_codec_rejects_non_power_of_two_dimensions() {
        let error = encode_key_vector(&[1.0, 2.0, 3.0], MlxTurboQuantPreset::K8V4)
            .expect_err("non power-of-two dimensions should fail");
        assert_eq!(error, TurboQuantCodecError::NonPowerOfTwoDimension(3));
    }

    #[test]
    fn reference_decode_attention_matches_softmax_contract() {
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let tokens = vec![
            (vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]),
            (vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]),
        ];

        let output = reference_decode_attention(&query, &tokens).expect("decode attention");
        let first_weight = 0.5f32.exp() / (0.5f32.exp() + 1.0);
        let second_weight = 1.0 / (0.5f32.exp() + 1.0);

        assert!((output[0] - first_weight).abs() < 1e-6);
        assert!((output[1] - second_weight).abs() < 1e-6);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 0.0);
    }

    #[test]
    fn prototype_decode_attention_matches_full_precision_when_all_tokens_are_hot() {
        let config = TurboQuantKvPrototypeConfig::new(MlxTurboQuantPreset::K8V4, 4, 4, 2)
            .expect("prototype config should build");
        let mut store = TurboQuantKvPrototypeStore::new(config);
        let tokens = vec![
            (vec![0.5, -0.5, 1.0, 0.0], vec![0.2, 0.0, -0.2, 0.4]),
            (vec![0.0, 1.0, -0.5, 0.5], vec![0.1, 0.3, 0.5, -0.1]),
            (vec![-0.5, 0.25, 0.0, 1.0], vec![0.6, -0.2, 0.0, 0.2]),
        ];
        let query = vec![0.25, 0.5, -0.25, 1.0];

        store
            .append_many(tokens.clone())
            .expect("prototype append should work");

        let expected = reference_decode_attention(&query, &tokens).expect("full attention");
        let actual = store
            .debug_decode_attention(&query)
            .expect("prototype debug attention");

        assert_eq!(actual, expected);
    }

    #[test]
    fn prototype_decode_attention_stays_close_to_full_precision_oracle() {
        let config = TurboQuantKvPrototypeConfig::new(MlxTurboQuantPreset::K8V4, 8, 2, 4)
            .expect("prototype config should build");
        let mut store = TurboQuantKvPrototypeStore::new(config);
        let tokens = (0..6)
            .map(|token| {
                let key = (0..8)
                    .map(|idx| (token as f32 * 0.17) + (idx as f32 * 0.09) - 0.4)
                    .collect::<Vec<_>>();
                let value = (0..8)
                    .map(|idx| (token as f32 * 0.05) - (idx as f32 * 0.03) + 0.1)
                    .collect::<Vec<_>>();
                (key, value)
            })
            .collect::<Vec<_>>();
        let query = vec![0.3, -0.1, 0.2, 0.6, -0.4, 0.1, 0.5, -0.2];

        store
            .append_many(tokens.clone())
            .expect("prototype append should work");

        let expected = reference_decode_attention(&query, &tokens).expect("full attention");
        let actual = store
            .debug_decode_attention(&query)
            .expect("prototype debug attention");

        assert!(cosine_similarity(&expected, &actual) > 0.999);
        assert!(
            max_abs_diff(&expected, &actual) < 0.02,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn reference_decode_attention_rejects_empty_or_mismatched_inputs() {
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let error = reference_decode_attention(&query, &[])
            .expect_err("empty KV history should fail closed");
        assert_eq!(error, TurboQuantCodecError::EmptyKvHistory);

        let error =
            reference_decode_attention(&query, &[(vec![1.0, 0.0], vec![1.0, 0.0, 0.0, 0.0])])
                .expect_err("mismatched key dimension should fail closed");
        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedVectorDimension {
                expected: 4,
                actual: 2,
            }
        );

        let error = reference_decode_attention(
            &[1.0, 0.0, 0.0],
            &[(vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0])],
        )
        .expect_err("non power-of-two query dimension should fail closed");
        assert_eq!(error, TurboQuantCodecError::NonPowerOfTwoDimension(3));
    }

    #[test]
    fn prototype_store_keeps_hot_window_and_compresses_cold_tokens() {
        let config = TurboQuantKvPrototypeConfig::new(MlxTurboQuantPreset::K8V4, 8, 2, 4)
            .expect("prototype config should build");
        let mut store = TurboQuantKvPrototypeStore::new(config);
        let tokens = (0..5)
            .map(|token| {
                let key = (0..8)
                    .map(|idx| token as f32 * 0.25 + idx as f32 * 0.125 - 0.5)
                    .collect::<Vec<_>>();
                let value = (0..8)
                    .map(|idx| token as f32 * 0.1 - idx as f32 * 0.05)
                    .collect::<Vec<_>>();
                (key, value)
            })
            .collect::<Vec<_>>();

        store
            .append_many(tokens.clone())
            .expect("prototype append should work");

        assert_eq!(store.token_count(), 5);
        assert_eq!(store.cold_token_count(), 3);
        assert_eq!(store.hot_token_count(), 2);

        let stats = store.stats();
        assert_eq!(stats.cold_tokens, 3);
        assert_eq!(stats.hot_tokens, 2);
        assert_eq!(stats.full_precision_cold_bytes, 192);
        assert!(stats.compressed_cold_bytes < stats.full_precision_cold_bytes);
        assert!(stats.estimated_saved_bytes > 0);
        assert!(stats.compression_ratio_milli() < 1000);

        let reconstructed = store
            .debug_reconstruct()
            .expect("debug reconstruct should work");
        assert_eq!(reconstructed.len(), tokens.len());

        for (token_idx, ((expected_key, expected_value), (actual_key, actual_value))) in
            tokens.iter().zip(reconstructed.iter()).enumerate()
        {
            assert_eq!(actual_key.len(), expected_key.len());
            assert_eq!(actual_value.len(), expected_value.len());
            if token_idx >= 3 {
                assert_eq!(actual_key, expected_key);
                assert_eq!(actual_value, expected_value);
            } else {
                assert!(cosine_similarity(expected_key, actual_key) > 0.999);
                for (expected, actual) in expected_value.iter().zip(actual_value) {
                    assert!(
                        (expected - actual).abs() <= 0.04,
                        "token {token_idx} expected value {expected}, got {actual}"
                    );
                }
            }
        }
    }

    #[test]
    fn prototype_store_with_zero_hot_window_compresses_every_token() {
        let config = TurboQuantKvPrototypeConfig::new(MlxTurboQuantPreset::K4V4, 4, 0, 2)
            .expect("prototype config should build");
        let mut store = TurboQuantKvPrototypeStore::new(config);

        store
            .append(&[0.0, 0.5, -0.5, 1.0], &[1.0, 0.5, -0.5, -1.0])
            .expect("append should work");

        assert_eq!(store.cold_token_count(), 1);
        assert_eq!(store.hot_token_count(), 0);
        assert_eq!(store.debug_reconstruct().expect("reconstruct").len(), 1);
    }

    #[test]
    fn prototype_store_rejects_shape_drift() {
        let config = TurboQuantKvPrototypeConfig::new(MlxTurboQuantPreset::K8V4, 4, 2, 2)
            .expect("prototype config should build");
        let mut store = TurboQuantKvPrototypeStore::new(config);

        let error = store
            .append(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0])
            .expect_err("mismatched K/V vector lengths should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedKvVectorLengths {
                key_len: 4,
                value_len: 2,
            }
        );

        let error = store
            .append(&[1.0, 2.0], &[1.0, 2.0])
            .expect_err("dimension drift should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedVectorDimension {
                expected: 4,
                actual: 2,
            }
        );
    }
}
