use std::collections::VecDeque;

use ax_engine_core::MlxTurboQuantPreset;
use thiserror::Error;

use crate::model::ModelConfig;

pub type FullPrecisionKvTokenVectors = (Vec<f32>, Vec<f32>);

pub const TURBOQUANT_SLOT_ALIGNMENT_BYTES: usize = 16;

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
    #[error("TurboQuant expected {expected} KV heads, got {actual}")]
    MismatchedKvHeadCount { expected: usize, actual: usize },
    #[error("TurboQuant reference attention requires at least one KV token")]
    EmptyKvHistory,
    #[error("invalid TurboQuant layout parameter {name}={value}")]
    InvalidLayoutParameter { name: &'static str, value: usize },
    #[error("TurboQuant layout address {name} index {index} is out of range for limit {limit}")]
    LayoutAddressOutOfRange {
        name: &'static str,
        index: usize,
        limit: usize,
    },
    #[error("TurboQuant layout byte-size calculation overflowed")]
    LayoutSizeOverflow,
    #[error(
        "TurboQuant compressed slot token={token_index}, head={head_index} has not been written"
    )]
    CompressedSlotUnwritten {
        token_index: usize,
        head_index: usize,
    },
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

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantDecodeHeadComparison {
    pub head_index: usize,
    pub vector_dim: usize,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub cosine_similarity: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantDecodeComparisonReport {
    pub heads: Vec<TurboQuantDecodeHeadComparison>,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub min_cosine_similarity: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurboQuantDecodeQualityProfile {
    StrictDebug,
    ReferenceK8V4,
    ResearchLoose,
}

impl TurboQuantDecodeQualityProfile {
    pub const fn gate(self) -> TurboQuantDecodeQualityGate {
        match self {
            Self::StrictDebug => TurboQuantDecodeQualityGate::STRICT_DEBUG,
            Self::ReferenceK8V4 => TurboQuantDecodeQualityGate::REFERENCE_K8V4,
            Self::ResearchLoose => TurboQuantDecodeQualityGate::RESEARCH_LOOSE,
        }
    }

    pub const fn for_quantization_preset(preset: MlxTurboQuantPreset) -> Self {
        match preset {
            MlxTurboQuantPreset::K8V4 => Self::ReferenceK8V4,
            MlxTurboQuantPreset::K4V4 | MlxTurboQuantPreset::K3V4Research => Self::ResearchLoose,
        }
    }

    pub fn evaluate(
        self,
        report: &TurboQuantDecodeComparisonReport,
    ) -> TurboQuantDecodeQualityDecision {
        self.gate().evaluate(report)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurboQuantDecodeQualityGate {
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub min_cosine_similarity: f32,
}

impl TurboQuantDecodeQualityGate {
    pub const STRICT_DEBUG: Self = Self::new(0.02, 0.01, 0.999);
    pub const REFERENCE_K8V4: Self = Self::new(0.04, 0.02, 0.998);
    pub const RESEARCH_LOOSE: Self = Self::new(0.08, 0.04, 0.995);

    pub const fn new(max_abs_diff: f32, mean_abs_diff: f32, min_cosine_similarity: f32) -> Self {
        Self {
            max_abs_diff,
            mean_abs_diff,
            min_cosine_similarity,
        }
    }

    pub fn evaluate(
        self,
        report: &TurboQuantDecodeComparisonReport,
    ) -> TurboQuantDecodeQualityDecision {
        let max_abs_diff_passed = report.max_abs_diff <= self.max_abs_diff;
        let mean_abs_diff_passed = report.mean_abs_diff <= self.mean_abs_diff;
        let min_cosine_similarity_passed =
            report.min_cosine_similarity >= self.min_cosine_similarity;

        TurboQuantDecodeQualityDecision {
            passed: max_abs_diff_passed && mean_abs_diff_passed && min_cosine_similarity_passed,
            max_abs_diff_passed,
            mean_abs_diff_passed,
            min_cosine_similarity_passed,
            max_abs_diff: report.max_abs_diff,
            max_abs_diff_limit: self.max_abs_diff,
            mean_abs_diff: report.mean_abs_diff,
            mean_abs_diff_limit: self.mean_abs_diff,
            min_cosine_similarity: report.min_cosine_similarity,
            min_cosine_similarity_limit: self.min_cosine_similarity,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurboQuantDecodeQualityDecision {
    pub passed: bool,
    pub max_abs_diff_passed: bool,
    pub mean_abs_diff_passed: bool,
    pub min_cosine_similarity_passed: bool,
    pub max_abs_diff: f32,
    pub max_abs_diff_limit: f32,
    pub mean_abs_diff: f32,
    pub mean_abs_diff_limit: f32,
    pub min_cosine_similarity: f32,
    pub min_cosine_similarity_limit: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurboQuantDecodeQualityEvaluation {
    pub preset: MlxTurboQuantPreset,
    pub profile: TurboQuantDecodeQualityProfile,
    pub gate: TurboQuantDecodeQualityGate,
    pub decision: TurboQuantDecodeQualityDecision,
}

pub fn evaluate_decode_quality_for_preset(
    preset: MlxTurboQuantPreset,
    report: &TurboQuantDecodeComparisonReport,
) -> TurboQuantDecodeQualityEvaluation {
    let profile = TurboQuantDecodeQualityProfile::for_quantization_preset(preset);
    let gate = profile.gate();
    let decision = gate.evaluate(report);
    TurboQuantDecodeQualityEvaluation {
        preset,
        profile,
        gate,
        decision,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantDecodeQualityCheck {
    pub comparison: TurboQuantDecodeComparisonReport,
    pub evaluation: TurboQuantDecodeQualityEvaluation,
}

impl TurboQuantDecodeQualityCheck {
    pub fn for_preset(
        preset: MlxTurboQuantPreset,
        comparison: TurboQuantDecodeComparisonReport,
    ) -> Self {
        let evaluation = evaluate_decode_quality_for_preset(preset, &comparison);
        Self {
            comparison,
            evaluation,
        }
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantBlockLayoutConfig {
    pub preset: MlxTurboQuantPreset,
    pub block_tokens: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub value_group_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantBlockLayout {
    pub config: TurboQuantBlockLayoutConfig,
    pub value_group_count: usize,
    pub key_payload_bytes_per_head: usize,
    pub key_norm_bytes_per_head: usize,
    pub value_payload_bytes_per_head: usize,
    pub value_metadata_bytes_per_head: usize,
    pub raw_slot_bytes_per_head: usize,
    pub slot_bytes_per_head: usize,
    pub token_stride_bytes: usize,
    pub block_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantSlotAddress {
    pub block_index: usize,
    pub token_offset: usize,
    pub head_index: usize,
    pub slot_offset_bytes: usize,
    pub key_payload_offset_bytes: usize,
    pub key_norm_offset_bytes: usize,
    pub value_payload_offset_bytes: usize,
    pub value_mins_offset_bytes: usize,
    pub value_scales_offset_bytes: usize,
    pub slot_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurboQuantCompressedDecodePath {
    FullPrecisionOnly,
    CompressedColdWithHotWindow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantCompressedDecodePlan {
    pub layout: TurboQuantBlockLayout,
    pub total_tokens: usize,
    pub cold_tokens: usize,
    pub hot_tokens: usize,
    pub compressed_blocks: usize,
    pub compressed_buffer_bytes: usize,
    pub required_compressed_slots: usize,
    pub decode_path: TurboQuantCompressedDecodePath,
    pub quality_profile: TurboQuantDecodeQualityProfile,
}

impl TurboQuantCompressedDecodePlan {
    pub fn new(
        layout: TurboQuantBlockLayout,
        total_tokens: usize,
        hot_window_tokens: usize,
    ) -> Result<Self, TurboQuantCodecError> {
        validate_layout_nonzero("total_tokens", total_tokens)?;
        let cold_tokens = total_tokens.saturating_sub(hot_window_tokens);
        let hot_tokens = total_tokens.saturating_sub(cold_tokens);
        let compressed_blocks = layout.block_count_for_tokens(cold_tokens);
        let compressed_buffer_bytes = layout.buffer_bytes_for_tokens(cold_tokens)?;
        let required_compressed_slots = checked_mul(cold_tokens, layout.config.n_kv_heads)?;
        let decode_path = if cold_tokens == 0 {
            TurboQuantCompressedDecodePath::FullPrecisionOnly
        } else {
            TurboQuantCompressedDecodePath::CompressedColdWithHotWindow
        };
        let quality_profile =
            TurboQuantDecodeQualityProfile::for_quantization_preset(layout.config.preset);

        Ok(Self {
            layout,
            total_tokens,
            cold_tokens,
            hot_tokens,
            compressed_blocks,
            compressed_buffer_bytes,
            required_compressed_slots,
            decode_path,
            quality_profile,
        })
    }

    pub fn needs_compressed_decode(self) -> bool {
        matches!(
            self.decode_path,
            TurboQuantCompressedDecodePath::CompressedColdWithHotWindow
        )
    }
}

impl TurboQuantBlockLayout {
    pub fn new(config: TurboQuantBlockLayoutConfig) -> Result<Self, TurboQuantCodecError> {
        validate_layout_nonzero("block_tokens", config.block_tokens)?;
        validate_layout_nonzero("n_kv_heads", config.n_kv_heads)?;
        validate_layout_nonzero("head_dim", config.head_dim)?;
        validate_layout_nonzero("value_group_size", config.value_group_size)?;
        validate_key_shape_len(config.head_dim)?;
        key_centroids(config.preset.key_bits())?;

        let value_group_count = config.head_dim.div_ceil(config.value_group_size);
        let key_payload_bytes_per_head =
            packed_index_bytes(config.head_dim, config.preset.key_bits())?;
        let key_norm_bytes_per_head = std::mem::size_of::<f32>();
        let value_payload_bytes_per_head =
            packed_index_bytes(config.head_dim, config.preset.value_bits())?;
        let value_metadata_bytes_per_head = checked_mul(
            checked_mul(value_group_count, std::mem::size_of::<f32>())?,
            2,
        )?;
        let raw_slot_bytes_per_head = checked_add_many(&[
            key_payload_bytes_per_head,
            key_norm_bytes_per_head,
            value_payload_bytes_per_head,
            value_metadata_bytes_per_head,
        ])?;
        let slot_bytes_per_head =
            align_up(raw_slot_bytes_per_head, TURBOQUANT_SLOT_ALIGNMENT_BYTES)?;
        let token_stride_bytes = checked_mul(config.n_kv_heads, slot_bytes_per_head)?;
        let block_bytes = checked_mul(config.block_tokens, token_stride_bytes)?;

        Ok(Self {
            config,
            value_group_count,
            key_payload_bytes_per_head,
            key_norm_bytes_per_head,
            value_payload_bytes_per_head,
            value_metadata_bytes_per_head,
            raw_slot_bytes_per_head,
            slot_bytes_per_head,
            token_stride_bytes,
            block_bytes,
        })
    }

    pub fn block_count_for_tokens(&self, token_count: usize) -> usize {
        token_count.div_ceil(self.config.block_tokens)
    }

    pub fn buffer_bytes_for_tokens(
        &self,
        token_count: usize,
    ) -> Result<usize, TurboQuantCodecError> {
        checked_mul(self.block_count_for_tokens(token_count), self.block_bytes)
    }

    pub fn address(
        &self,
        block_index: usize,
        token_offset: usize,
        head_index: usize,
    ) -> Result<TurboQuantSlotAddress, TurboQuantCodecError> {
        if token_offset >= self.config.block_tokens {
            return Err(TurboQuantCodecError::LayoutAddressOutOfRange {
                name: "token_offset",
                index: token_offset,
                limit: self.config.block_tokens,
            });
        }
        if head_index >= self.config.n_kv_heads {
            return Err(TurboQuantCodecError::LayoutAddressOutOfRange {
                name: "head_index",
                index: head_index,
                limit: self.config.n_kv_heads,
            });
        }

        let block_offset = checked_mul(block_index, self.block_bytes)?;
        let token_offset_bytes = checked_mul(token_offset, self.token_stride_bytes)?;
        let head_offset_bytes = checked_mul(head_index, self.slot_bytes_per_head)?;
        let slot_offset_bytes =
            checked_add_many(&[block_offset, token_offset_bytes, head_offset_bytes])?;

        self.slot_address_at(block_index, token_offset, head_index, slot_offset_bytes)
    }

    pub fn address_for_token(
        &self,
        token_index: usize,
        head_index: usize,
    ) -> Result<TurboQuantSlotAddress, TurboQuantCodecError> {
        self.address(
            token_index / self.config.block_tokens,
            token_index % self.config.block_tokens,
            head_index,
        )
    }

    fn slot_address_at(
        &self,
        block_index: usize,
        token_offset: usize,
        head_index: usize,
        slot_offset_bytes: usize,
    ) -> Result<TurboQuantSlotAddress, TurboQuantCodecError> {
        let key_payload_offset_bytes = slot_offset_bytes;
        let key_norm_offset_bytes =
            checked_add_many(&[key_payload_offset_bytes, self.key_payload_bytes_per_head])?;
        let value_payload_offset_bytes =
            checked_add_many(&[key_norm_offset_bytes, self.key_norm_bytes_per_head])?;
        let value_mins_offset_bytes = checked_add_many(&[
            value_payload_offset_bytes,
            self.value_payload_bytes_per_head,
        ])?;
        let value_scales_offset_bytes = checked_add_many(&[
            value_mins_offset_bytes,
            self.value_metadata_bytes_per_head / 2,
        ])?;

        Ok(TurboQuantSlotAddress {
            block_index,
            token_offset,
            head_index,
            slot_offset_bytes,
            key_payload_offset_bytes,
            key_norm_offset_bytes,
            value_payload_offset_bytes,
            value_mins_offset_bytes,
            value_scales_offset_bytes,
            slot_bytes: self.slot_bytes_per_head,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantCompressedHeadSlot {
    pub key: QuantizedKeyVector,
    pub value: QuantizedValueGroups,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantCompressedBlockBuffer {
    layout: TurboQuantBlockLayout,
    bytes: Vec<u8>,
    written_slots: Vec<bool>,
    token_count: usize,
}

impl TurboQuantCompressedBlockBuffer {
    pub fn new(layout: TurboQuantBlockLayout) -> Self {
        Self {
            layout,
            bytes: Vec::new(),
            written_slots: Vec::new(),
            token_count: 0,
        }
    }

    pub fn layout(&self) -> &TurboQuantBlockLayout {
        &self.layout
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn block_count(&self) -> usize {
        self.layout.block_count_for_tokens(self.token_count)
    }

    pub fn write_slot(
        &mut self,
        token_index: usize,
        head_index: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), TurboQuantCodecError> {
        self.validate_slot_vectors(key, value)?;
        let address = self.layout.address_for_token(token_index, head_index)?;
        self.ensure_capacity_for_tokens(token_index.saturating_add(1))?;

        let key = encode_key_vector(key, self.layout.config.preset)?;
        let value = encode_value_groups_4bit(value, self.layout.config.value_group_size)?;

        self.zero_slot(&address)?;
        self.write_compressed_slot_at(&address, &key, &value)?;

        let slot_index = self.slot_index(token_index, head_index)?;
        self.written_slots[slot_index] = true;
        self.token_count = self.token_count.max(token_index.saturating_add(1));
        Ok(())
    }

    pub fn write_token(
        &mut self,
        token_index: usize,
        heads: &[FullPrecisionKvTokenVectors],
    ) -> Result<(), TurboQuantCodecError> {
        if heads.len() != self.layout.config.n_kv_heads {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: self.layout.config.n_kv_heads,
                actual: heads.len(),
            });
        }

        let compressed_heads = heads
            .iter()
            .map(|(key, value)| {
                self.validate_slot_vectors(key, value)?;
                Ok(TurboQuantCompressedHeadSlot {
                    key: encode_key_vector(key, self.layout.config.preset)?,
                    value: encode_value_groups_4bit(value, self.layout.config.value_group_size)?,
                })
            })
            .collect::<Result<Vec<_>, TurboQuantCodecError>>()?;

        self.ensure_capacity_for_tokens(token_index.saturating_add(1))?;
        for (head_index, slot) in compressed_heads.iter().enumerate() {
            let address = self.layout.address_for_token(token_index, head_index)?;
            self.zero_slot(&address)?;
            self.write_compressed_slot_at(&address, &slot.key, &slot.value)?;
            let slot_index = self.slot_index(token_index, head_index)?;
            self.written_slots[slot_index] = true;
        }
        self.token_count = self.token_count.max(token_index.saturating_add(1));
        Ok(())
    }

    pub fn read_compressed_slot(
        &self,
        token_index: usize,
        head_index: usize,
    ) -> Result<TurboQuantCompressedHeadSlot, TurboQuantCodecError> {
        self.validate_written_slot(token_index, head_index)?;
        let address = self.layout.address_for_token(token_index, head_index)?;

        Ok(TurboQuantCompressedHeadSlot {
            key: self.read_key_at(&address),
            value: self.read_value_at(&address),
        })
    }

    pub fn read_compressed_token(
        &self,
        token_index: usize,
    ) -> Result<Vec<TurboQuantCompressedHeadSlot>, TurboQuantCodecError> {
        (0..self.layout.config.n_kv_heads)
            .map(|head_index| self.read_compressed_slot(token_index, head_index))
            .collect()
    }

    pub fn debug_reconstruct_slot(
        &self,
        token_index: usize,
        head_index: usize,
    ) -> Result<FullPrecisionKvTokenVectors, TurboQuantCodecError> {
        let slot = self.read_compressed_slot(token_index, head_index)?;
        Ok((
            decode_key_vector(&slot.key)?,
            decode_value_groups_4bit(&slot.value)?,
        ))
    }

    pub fn debug_reconstruct_token(
        &self,
        token_index: usize,
    ) -> Result<Vec<FullPrecisionKvTokenVectors>, TurboQuantCodecError> {
        (0..self.layout.config.n_kv_heads)
            .map(|head_index| self.debug_reconstruct_slot(token_index, head_index))
            .collect()
    }

    pub fn debug_reconstruct_head_history(
        &self,
        head_index: usize,
        token_count: usize,
    ) -> Result<Vec<FullPrecisionKvTokenVectors>, TurboQuantCodecError> {
        self.validate_head_index(head_index)?;
        (0..token_count)
            .map(|token_index| self.debug_reconstruct_slot(token_index, head_index))
            .collect()
    }

    pub fn debug_decode_attention_for_head(
        &self,
        query: &[f32],
        head_index: usize,
        token_count: usize,
    ) -> Result<Vec<f32>, TurboQuantCodecError> {
        let tokens = self.debug_reconstruct_head_history(head_index, token_count)?;
        reference_decode_attention(query, &tokens)
    }

    pub fn debug_decode_attention_for_all_heads(
        &self,
        queries: &[Vec<f32>],
        token_count: usize,
    ) -> Result<Vec<Vec<f32>>, TurboQuantCodecError> {
        if queries.len() != self.layout.config.n_kv_heads {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: self.layout.config.n_kv_heads,
                actual: queries.len(),
            });
        }

        queries
            .iter()
            .enumerate()
            .map(|(head_index, query)| {
                self.debug_decode_attention_for_head(query, head_index, token_count)
            })
            .collect()
    }

    pub fn debug_compare_attention_for_all_heads(
        &self,
        queries: &[Vec<f32>],
        expected_outputs: &[Vec<f32>],
        token_count: usize,
    ) -> Result<TurboQuantDecodeComparisonReport, TurboQuantCodecError> {
        let actual_outputs = self.debug_decode_attention_for_all_heads(queries, token_count)?;
        compare_decode_outputs(expected_outputs, &actual_outputs)
    }

    pub fn debug_evaluate_attention_quality_for_all_heads(
        &self,
        queries: &[Vec<f32>],
        expected_outputs: &[Vec<f32>],
        token_count: usize,
    ) -> Result<TurboQuantDecodeQualityCheck, TurboQuantCodecError> {
        let comparison =
            self.debug_compare_attention_for_all_heads(queries, expected_outputs, token_count)?;
        Ok(TurboQuantDecodeQualityCheck::for_preset(
            self.layout.config.preset,
            comparison,
        ))
    }

    fn validate_slot_vectors(
        &self,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), TurboQuantCodecError> {
        let expected = self.layout.config.head_dim;
        if key.len() != expected {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected,
                actual: key.len(),
            });
        }
        if value.len() != expected {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected,
                actual: value.len(),
            });
        }
        Ok(())
    }

    fn ensure_capacity_for_tokens(
        &mut self,
        token_count: usize,
    ) -> Result<(), TurboQuantCodecError> {
        let buffer_bytes = self.layout.buffer_bytes_for_tokens(token_count)?;
        self.bytes.resize(buffer_bytes, 0);

        let slot_count = checked_mul(
            checked_mul(
                self.layout.block_count_for_tokens(token_count),
                self.layout.config.block_tokens,
            )?,
            self.layout.config.n_kv_heads,
        )?;
        self.written_slots.resize(slot_count, false);
        Ok(())
    }

    fn zero_slot(&mut self, address: &TurboQuantSlotAddress) -> Result<(), TurboQuantCodecError> {
        let end = checked_add_many(&[address.slot_offset_bytes, address.slot_bytes])?;
        self.bytes[address.slot_offset_bytes..end].fill(0);
        Ok(())
    }

    fn write_compressed_slot_at(
        &mut self,
        address: &TurboQuantSlotAddress,
        key: &QuantizedKeyVector,
        value: &QuantizedValueGroups,
    ) -> Result<(), TurboQuantCodecError> {
        self.write_bytes(
            address.key_payload_offset_bytes,
            self.layout.key_payload_bytes_per_head,
            &key.packed_indices,
        );
        self.write_f32(address.key_norm_offset_bytes, key.l2_norm);
        self.write_bytes(
            address.value_payload_offset_bytes,
            self.layout.value_payload_bytes_per_head,
            &value.packed_values,
        );
        self.write_f32_slice(address.value_mins_offset_bytes, &value.mins);
        self.write_f32_slice(address.value_scales_offset_bytes, &value.scales);
        Ok(())
    }

    fn write_bytes(&mut self, offset: usize, len: usize, source: &[u8]) {
        let end = offset + len;
        let target = &mut self.bytes[offset..end];
        target.fill(0);
        target[..source.len()].copy_from_slice(source);
    }

    fn write_f32(&mut self, offset: usize, value: f32) {
        self.bytes[offset..offset + std::mem::size_of::<f32>()]
            .copy_from_slice(&value.to_le_bytes());
    }

    fn write_f32_slice(&mut self, offset: usize, values: &[f32]) {
        for (idx, value) in values.iter().enumerate() {
            self.write_f32(offset + idx * std::mem::size_of::<f32>(), *value);
        }
    }

    fn read_key_at(&self, address: &TurboQuantSlotAddress) -> QuantizedKeyVector {
        QuantizedKeyVector {
            dim: self.layout.config.head_dim,
            bit_width: self.layout.config.preset.key_bits(),
            l2_norm: self.read_f32(address.key_norm_offset_bytes),
            packed_indices: self.bytes[address.key_payload_offset_bytes
                ..address.key_payload_offset_bytes + self.layout.key_payload_bytes_per_head]
                .to_vec(),
        }
    }

    fn read_value_at(&self, address: &TurboQuantSlotAddress) -> QuantizedValueGroups {
        QuantizedValueGroups {
            element_count: self.layout.config.head_dim,
            group_size: self.layout.config.value_group_size,
            mins: self.read_f32_vec(
                address.value_mins_offset_bytes,
                self.layout.value_group_count,
            ),
            scales: self.read_f32_vec(
                address.value_scales_offset_bytes,
                self.layout.value_group_count,
            ),
            packed_values: self.bytes[address.value_payload_offset_bytes
                ..address.value_payload_offset_bytes + self.layout.value_payload_bytes_per_head]
                .to_vec(),
        }
    }

    fn read_f32(&self, offset: usize) -> f32 {
        let mut bytes = [0u8; std::mem::size_of::<f32>()];
        bytes.copy_from_slice(&self.bytes[offset..offset + std::mem::size_of::<f32>()]);
        f32::from_le_bytes(bytes)
    }

    fn read_f32_vec(&self, offset: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|idx| self.read_f32(offset + idx * std::mem::size_of::<f32>()))
            .collect()
    }

    fn validate_written_slot(
        &self,
        token_index: usize,
        head_index: usize,
    ) -> Result<(), TurboQuantCodecError> {
        self.validate_head_index(head_index)?;
        let slot_index = self.slot_index(token_index, head_index)?;
        if token_index >= self.token_count
            || !self.written_slots.get(slot_index).copied().unwrap_or(false)
        {
            return Err(TurboQuantCodecError::CompressedSlotUnwritten {
                token_index,
                head_index,
            });
        }
        Ok(())
    }

    fn validate_head_index(&self, head_index: usize) -> Result<(), TurboQuantCodecError> {
        if head_index >= self.layout.config.n_kv_heads {
            return Err(TurboQuantCodecError::LayoutAddressOutOfRange {
                name: "head_index",
                index: head_index,
                limit: self.layout.config.n_kv_heads,
            });
        }
        Ok(())
    }

    fn slot_index(
        &self,
        token_index: usize,
        head_index: usize,
    ) -> Result<usize, TurboQuantCodecError> {
        checked_add_many(&[
            checked_mul(token_index, self.layout.config.n_kv_heads)?,
            head_index,
        ])
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

pub fn compare_decode_outputs(
    expected_outputs: &[Vec<f32>],
    actual_outputs: &[Vec<f32>],
) -> Result<TurboQuantDecodeComparisonReport, TurboQuantCodecError> {
    if expected_outputs.len() != actual_outputs.len() {
        return Err(TurboQuantCodecError::MismatchedKvHeadCount {
            expected: expected_outputs.len(),
            actual: actual_outputs.len(),
        });
    }

    let mut heads = Vec::with_capacity(expected_outputs.len());
    let mut total_abs_diff = 0.0f32;
    let mut total_elements = 0usize;
    let mut max_abs_diff = 0.0f32;
    let mut min_cosine_similarity = 1.0f32;

    for (head_index, (expected, actual)) in expected_outputs.iter().zip(actual_outputs).enumerate()
    {
        if expected.len() != actual.len() {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: expected.len(),
                actual: actual.len(),
            });
        }

        let mut head_total_abs_diff = 0.0f32;
        let mut head_max_abs_diff = 0.0f32;
        for (expected, actual) in expected.iter().zip(actual) {
            let diff = (expected - actual).abs();
            head_total_abs_diff += diff;
            head_max_abs_diff = head_max_abs_diff.max(diff);
        }

        let vector_dim = expected.len();
        let head_mean_abs_diff = if vector_dim == 0 {
            0.0
        } else {
            head_total_abs_diff / vector_dim as f32
        };
        let cosine_similarity = vector_cosine_similarity(expected, actual);

        max_abs_diff = max_abs_diff.max(head_max_abs_diff);
        total_abs_diff += head_total_abs_diff;
        total_elements = total_elements.saturating_add(vector_dim);
        min_cosine_similarity = min_cosine_similarity.min(cosine_similarity);
        heads.push(TurboQuantDecodeHeadComparison {
            head_index,
            vector_dim,
            max_abs_diff: head_max_abs_diff,
            mean_abs_diff: head_mean_abs_diff,
            cosine_similarity,
        });
    }

    Ok(TurboQuantDecodeComparisonReport {
        heads,
        max_abs_diff,
        mean_abs_diff: if total_elements == 0 {
            0.0
        } else {
            total_abs_diff / total_elements as f32
        },
        min_cosine_similarity,
    })
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
    validate_key_shape_len(values.len())
}

fn validate_key_shape_len(len: usize) -> Result<(), TurboQuantCodecError> {
    if len == 0 {
        return Err(TurboQuantCodecError::EmptyVector);
    }
    if !len.is_power_of_two() {
        return Err(TurboQuantCodecError::NonPowerOfTwoDimension(len));
    }
    Ok(())
}

fn validate_layout_nonzero(name: &'static str, value: usize) -> Result<(), TurboQuantCodecError> {
    if value == 0 {
        return Err(TurboQuantCodecError::InvalidLayoutParameter { name, value });
    }
    Ok(())
}

fn checked_add_many(values: &[usize]) -> Result<usize, TurboQuantCodecError> {
    values.iter().try_fold(0usize, |acc, value| {
        acc.checked_add(*value)
            .ok_or(TurboQuantCodecError::LayoutSizeOverflow)
    })
}

fn checked_mul(left: usize, right: usize) -> Result<usize, TurboQuantCodecError> {
    left.checked_mul(right)
        .ok_or(TurboQuantCodecError::LayoutSizeOverflow)
}

fn align_up(value: usize, alignment: usize) -> Result<usize, TurboQuantCodecError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(TurboQuantCodecError::InvalidLayoutParameter {
            name: "alignment",
            value: alignment,
        });
    }
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or(TurboQuantCodecError::LayoutSizeOverflow)
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

fn vector_cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = l2_norm(left).max(f32::EPSILON);
    let right_norm = l2_norm(right).max(f32::EPSILON);
    dot / (left_norm * right_norm)
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
            glm_mla_attention: None,
            glm_router: None,
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
    fn block_layout_computes_aligned_slot_and_block_sizes() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 16,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");

        assert_eq!(layout.value_group_count, 2);
        assert_eq!(layout.key_payload_bytes_per_head, 8);
        assert_eq!(layout.key_norm_bytes_per_head, 4);
        assert_eq!(layout.value_payload_bytes_per_head, 4);
        assert_eq!(layout.value_metadata_bytes_per_head, 16);
        assert_eq!(layout.raw_slot_bytes_per_head, 32);
        assert_eq!(layout.slot_bytes_per_head, 32);
        assert_eq!(layout.token_stride_bytes, 64);
        assert_eq!(layout.block_bytes, 1024);
        assert_eq!(layout.block_count_for_tokens(0), 0);
        assert_eq!(layout.block_count_for_tokens(33), 3);
        assert_eq!(
            layout
                .buffer_bytes_for_tokens(33)
                .expect("buffer size should fit"),
            3072
        );
    }

    #[test]
    fn block_layout_aligns_low_bit_research_slots() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K3V4Research,
            block_tokens: 4,
            n_kv_heads: 1,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");

        assert_eq!(layout.key_payload_bytes_per_head, 3);
        assert_eq!(layout.raw_slot_bytes_per_head, 27);
        assert_eq!(
            layout.slot_bytes_per_head,
            TURBOQUANT_SLOT_ALIGNMENT_BYTES * 2
        );
        assert_eq!(layout.block_bytes, 128);
    }

    #[test]
    fn block_layout_maps_token_and_head_to_slot_offsets() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 16,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");

        let address = layout
            .address(1, 3, 1)
            .expect("slot address should be in range");
        assert_eq!(
            address,
            TurboQuantSlotAddress {
                block_index: 1,
                token_offset: 3,
                head_index: 1,
                slot_offset_bytes: 1248,
                key_payload_offset_bytes: 1248,
                key_norm_offset_bytes: 1256,
                value_payload_offset_bytes: 1260,
                value_mins_offset_bytes: 1264,
                value_scales_offset_bytes: 1272,
                slot_bytes: 32,
            }
        );
        assert_eq!(
            layout
                .address_for_token(19, 1)
                .expect("absolute address should map to same slot"),
            address
        );
    }

    #[test]
    fn block_layout_rejects_invalid_config_and_out_of_range_addresses() {
        let error = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 0,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect_err("zero block size should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::InvalidLayoutParameter {
                name: "block_tokens",
                value: 0,
            }
        );

        let error = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 16,
            n_kv_heads: 2,
            head_dim: 12,
            value_group_size: 4,
        })
        .expect_err("non power-of-two head dim should fail");
        assert_eq!(error, TurboQuantCodecError::NonPowerOfTwoDimension(12));

        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 16,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let error = layout
            .address(0, 16, 0)
            .expect_err("token offset outside the block should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::LayoutAddressOutOfRange {
                name: "token_offset",
                index: 16,
                limit: 16,
            }
        );

        let error = layout
            .address(0, 0, 2)
            .expect_err("head index outside the token should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::LayoutAddressOutOfRange {
                name: "head_index",
                index: 2,
                limit: 2,
            }
        );
    }

    #[test]
    fn compressed_decode_plan_splits_cold_and_hot_history() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 16,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");

        let plan = TurboQuantCompressedDecodePlan::new(layout, 40, 8).expect("plan should build");

        assert_eq!(plan.total_tokens, 40);
        assert_eq!(plan.cold_tokens, 32);
        assert_eq!(plan.hot_tokens, 8);
        assert_eq!(plan.compressed_blocks, 2);
        assert_eq!(plan.compressed_buffer_bytes, 2048);
        assert_eq!(plan.required_compressed_slots, 64);
        assert_eq!(
            plan.decode_path,
            TurboQuantCompressedDecodePath::CompressedColdWithHotWindow
        );
        assert!(plan.needs_compressed_decode());
        assert_eq!(
            plan.quality_profile,
            TurboQuantDecodeQualityProfile::ReferenceK8V4
        );
    }

    #[test]
    fn compressed_decode_plan_falls_back_when_history_is_inside_hot_window() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K4V4,
            block_tokens: 16,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");

        let plan = TurboQuantCompressedDecodePlan::new(layout, 8, 16).expect("plan should build");

        assert_eq!(plan.cold_tokens, 0);
        assert_eq!(plan.hot_tokens, 8);
        assert_eq!(plan.compressed_blocks, 0);
        assert_eq!(plan.compressed_buffer_bytes, 0);
        assert_eq!(plan.required_compressed_slots, 0);
        assert_eq!(
            plan.decode_path,
            TurboQuantCompressedDecodePath::FullPrecisionOnly
        );
        assert!(!plan.needs_compressed_decode());
        assert_eq!(
            plan.quality_profile,
            TurboQuantDecodeQualityProfile::ResearchLoose
        );
    }

    #[test]
    fn compressed_decode_plan_fails_closed_for_empty_history() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 16,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");

        let error = TurboQuantCompressedDecodePlan::new(layout, 0, 16)
            .expect_err("empty decode history should fail");

        assert_eq!(
            error,
            TurboQuantCodecError::InvalidLayoutParameter {
                name: "total_tokens",
                value: 0,
            }
        );
    }

    #[test]
    fn compressed_block_buffer_writes_and_reads_slot_layout() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 4,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let key = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 0.0];
        let value = vec![-1.0, -0.6, -0.2, 0.0, 0.2, 0.5, 0.9, 1.0];

        buffer
            .write_slot(5, 1, &key, &value)
            .expect("slot write should work");

        assert_eq!(buffer.token_count(), 6);
        assert_eq!(buffer.block_count(), 2);
        assert_eq!(buffer.as_bytes().len(), layout.block_bytes * 2);

        let address = layout
            .address_for_token(5, 1)
            .expect("address should be in range");
        assert!(
            buffer.as_bytes()[address.key_payload_offset_bytes
                ..address.key_payload_offset_bytes + layout.key_payload_bytes_per_head]
                .iter()
                .any(|byte| *byte != 0)
        );

        let compressed = buffer
            .read_compressed_slot(5, 1)
            .expect("slot read should work");
        assert_eq!(compressed.key.dim, 8);
        assert_eq!(compressed.key.bit_width, 8);
        assert_eq!(compressed.value.element_count, 8);
        assert_eq!(compressed.value.group_size, 4);
        assert_eq!(compressed.value.mins.len(), 2);
        assert_eq!(compressed.value.scales.len(), 2);

        let (decoded_key, decoded_value) = buffer
            .debug_reconstruct_slot(5, 1)
            .expect("slot reconstruct should work");
        assert!(cosine_similarity(&key, &decoded_key) > 0.999);
        assert!(
            max_abs_diff(&value, &decoded_value) <= 0.08,
            "expected {value:?}, got {decoded_value:?}"
        );
    }

    #[test]
    fn compressed_block_buffer_keeps_token_heads_independent() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 4,
            value_group_size: 2,
        })
        .expect("layout should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let key0 = vec![1.0, 0.0, 0.0, 0.0];
        let value0 = vec![0.0, 1.0, 2.0, 3.0];
        let key1 = vec![0.0, 1.0, 0.0, 0.0];
        let value1 = vec![3.0, 2.0, 1.0, 0.0];

        buffer
            .write_slot(0, 0, &key0, &value0)
            .expect("head 0 write should work");
        buffer
            .write_slot(0, 1, &key1, &value1)
            .expect("head 1 write should work");

        let (decoded_key0, decoded_value0) = buffer
            .debug_reconstruct_slot(0, 0)
            .expect("head 0 read should work");
        let (decoded_key1, decoded_value1) = buffer
            .debug_reconstruct_slot(0, 1)
            .expect("head 1 read should work");

        assert!(cosine_similarity(&key0, &decoded_key0) > 0.999);
        assert!(cosine_similarity(&key1, &decoded_key1) > 0.999);
        assert!(max_abs_diff(&value0, &decoded_value0) <= 0.08);
        assert!(max_abs_diff(&value1, &decoded_value1) <= 0.08);
        assert_ne!(
            buffer
                .read_compressed_slot(0, 0)
                .expect("head 0 compressed slot"),
            buffer
                .read_compressed_slot(0, 1)
                .expect("head 1 compressed slot")
        );
    }

    #[test]
    fn compressed_block_buffer_fails_closed_for_bad_writes_and_unwritten_reads() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 4,
            value_group_size: 2,
        })
        .expect("layout should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);

        let error = buffer
            .read_compressed_slot(0, 0)
            .expect_err("unwritten slot should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::CompressedSlotUnwritten {
                token_index: 0,
                head_index: 0,
            }
        );

        let error = buffer
            .write_slot(0, 0, &[1.0, 0.0], &[0.0, 1.0, 2.0, 3.0])
            .expect_err("wrong key dimension should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedVectorDimension {
                expected: 4,
                actual: 2,
            }
        );

        let error = buffer
            .write_slot(0, 2, &[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 2.0, 3.0])
            .expect_err("head index outside layout should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::LayoutAddressOutOfRange {
                name: "head_index",
                index: 2,
                limit: 2,
            }
        );
    }

    #[test]
    fn compressed_block_buffer_writes_and_reads_full_token_heads() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let heads = vec![
            (
                vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 0.0],
                vec![-1.0, -0.6, -0.2, 0.0, 0.2, 0.5, 0.9, 1.0],
            ),
            (
                vec![0.25, 0.75, -1.5, 2.0, 0.0, -0.5, 1.25, -2.25],
                vec![1.0, 0.7, 0.4, 0.1, -0.2, -0.4, -0.8, -1.0],
            ),
        ];

        buffer
            .write_token(3, &heads)
            .expect("full token write should work");

        assert_eq!(buffer.token_count(), 4);
        assert_eq!(buffer.block_count(), 2);
        assert_eq!(
            buffer
                .read_compressed_token(3)
                .expect("compressed token should read")
                .len(),
            2
        );

        let reconstructed = buffer
            .debug_reconstruct_token(3)
            .expect("token reconstruct should work");
        assert_eq!(reconstructed.len(), heads.len());
        for ((expected_key, expected_value), (actual_key, actual_value)) in
            heads.iter().zip(reconstructed)
        {
            assert!(cosine_similarity(expected_key, &actual_key) > 0.999);
            assert!(max_abs_diff(expected_value, &actual_value) <= 0.08);
        }
    }

    #[test]
    fn compressed_block_buffer_token_write_fails_closed_for_head_count() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 4,
            value_group_size: 2,
        })
        .expect("layout should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let error = buffer
            .write_token(0, &[(vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 2.0, 3.0])])
            .expect_err("head count mismatch should fail");

        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedKvHeadCount {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(buffer.token_count(), 0);
    }

    #[test]
    fn compressed_block_buffer_token_write_does_not_leave_partial_slots() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 4,
            value_group_size: 2,
        })
        .expect("layout should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let error = buffer
            .write_token(
                0,
                &[
                    (vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 2.0, 3.0]),
                    (vec![0.0, 1.0], vec![3.0, 2.0, 1.0, 0.0]),
                ],
            )
            .expect_err("second head dimension mismatch should fail");

        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedVectorDimension {
                expected: 4,
                actual: 2,
            }
        );
        assert_eq!(buffer.token_count(), 0);
        let error = buffer
            .read_compressed_slot(0, 0)
            .expect_err("first head should not be partially written");
        assert_eq!(
            error,
            TurboQuantCodecError::CompressedSlotUnwritten {
                token_index: 0,
                head_index: 0,
            }
        );
    }

    #[test]
    fn compressed_block_buffer_decodes_attention_for_one_head_history() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let tokens = (0..3)
            .map(|token| {
                vec![
                    (
                        (0..8)
                            .map(|idx| token as f32 * 0.11 + idx as f32 * 0.07 - 0.2)
                            .collect::<Vec<_>>(),
                        (0..8)
                            .map(|idx| token as f32 * 0.03 - idx as f32 * 0.02)
                            .collect::<Vec<_>>(),
                    ),
                    (
                        (0..8)
                            .map(|idx| token as f32 * -0.09 + idx as f32 * 0.05 + 0.4)
                            .collect::<Vec<_>>(),
                        (0..8)
                            .map(|idx| token as f32 * 0.04 + idx as f32 * 0.01 - 0.3)
                            .collect::<Vec<_>>(),
                    ),
                ]
            })
            .collect::<Vec<_>>();
        for (token_index, heads) in tokens.iter().enumerate() {
            buffer
                .write_token(token_index, heads)
                .expect("token write should work");
        }
        let query = vec![0.3, -0.1, 0.2, 0.6, -0.4, 0.1, 0.5, -0.2];

        let reconstructed_history = buffer
            .debug_reconstruct_head_history(1, 3)
            .expect("head history should reconstruct");
        let expected =
            reference_decode_attention(&query, &reconstructed_history).expect("reference decode");
        let actual = buffer
            .debug_decode_attention_for_head(&query, 1, 3)
            .expect("buffer decode should work");

        assert_eq!(actual, expected);

        let original_history = tokens
            .iter()
            .map(|heads| heads[1].clone())
            .collect::<Vec<_>>();
        let full_precision =
            reference_decode_attention(&query, &original_history).expect("full precision decode");
        assert!(cosine_similarity(&full_precision, &actual) > 0.999);
        assert!(max_abs_diff(&full_precision, &actual) < 0.02);
    }

    #[test]
    fn compressed_block_buffer_head_decode_fails_closed_for_invalid_history() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 4,
            value_group_size: 2,
        })
        .expect("layout should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        buffer
            .write_token(
                0,
                &[
                    (vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 2.0, 3.0]),
                    (vec![0.0, 1.0, 0.0, 0.0], vec![3.0, 2.0, 1.0, 0.0]),
                ],
            )
            .expect("token write should work");

        let error = buffer
            .debug_decode_attention_for_head(&[1.0, 0.0, 0.0, 0.0], 2, 1)
            .expect_err("invalid head should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::LayoutAddressOutOfRange {
                name: "head_index",
                index: 2,
                limit: 2,
            }
        );

        let error = buffer
            .debug_decode_attention_for_head(&[1.0, 0.0, 0.0, 0.0], 0, 2)
            .expect_err("unwritten token in history should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::CompressedSlotUnwritten {
                token_index: 1,
                head_index: 0,
            }
        );

        let error = buffer
            .debug_decode_attention_for_head(&[1.0, 0.0, 0.0, 0.0], 0, 0)
            .expect_err("empty history should fail");
        assert_eq!(error, TurboQuantCodecError::EmptyKvHistory);
    }

    #[test]
    fn compressed_block_buffer_decodes_attention_for_all_heads() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        for token_index in 0..3 {
            let heads = vec![
                (
                    (0..8)
                        .map(|idx| token_index as f32 * 0.08 + idx as f32 * 0.04 - 0.3)
                        .collect::<Vec<_>>(),
                    (0..8)
                        .map(|idx| token_index as f32 * 0.02 - idx as f32 * 0.03)
                        .collect::<Vec<_>>(),
                ),
                (
                    (0..8)
                        .map(|idx| token_index as f32 * -0.06 + idx as f32 * 0.03 + 0.2)
                        .collect::<Vec<_>>(),
                    (0..8)
                        .map(|idx| token_index as f32 * 0.05 + idx as f32 * 0.02 - 0.4)
                        .collect::<Vec<_>>(),
                ),
            ];
            buffer
                .write_token(token_index, &heads)
                .expect("token write should work");
        }
        let queries = vec![
            vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.5, -0.3],
            vec![0.3, 0.1, -0.2, 0.6, -0.4, 0.0, 0.2, 0.5],
        ];

        let actual = buffer
            .debug_decode_attention_for_all_heads(&queries, 3)
            .expect("all-head decode should work");
        let expected = queries
            .iter()
            .enumerate()
            .map(|(head_index, query)| {
                buffer
                    .debug_decode_attention_for_head(query, head_index, 3)
                    .expect("single-head decode should work")
            })
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 2);
        assert!(actual.iter().all(|head| head.len() == 8));
    }

    #[test]
    fn compressed_block_buffer_all_heads_decode_fails_closed_for_query_count() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 4,
            value_group_size: 2,
        })
        .expect("layout should build");
        let buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let error = buffer
            .debug_decode_attention_for_all_heads(&[vec![1.0, 0.0, 0.0, 0.0]], 1)
            .expect_err("query head count mismatch should fail");

        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedKvHeadCount {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn compare_decode_outputs_reports_per_head_error_metrics() {
        let expected = vec![vec![1.0, 2.0, 3.0], vec![0.5, -0.5, 1.0]];
        let actual = vec![vec![1.0, 2.25, 2.5], vec![0.4, -0.4, 1.2]];

        let report = compare_decode_outputs(&expected, &actual).expect("compare outputs");

        assert_eq!(report.heads.len(), 2);
        assert_eq!(report.heads[0].head_index, 0);
        assert_eq!(report.heads[0].vector_dim, 3);
        assert!((report.heads[0].max_abs_diff - 0.5).abs() < 1e-6);
        assert!((report.heads[0].mean_abs_diff - 0.25).abs() < 1e-6);
        assert_eq!(report.heads[1].head_index, 1);
        assert!((report.heads[1].max_abs_diff - 0.2).abs() < 1e-6);
        assert!((report.max_abs_diff - 0.5).abs() < 1e-6);
        assert!((report.mean_abs_diff - 1.15 / 6.0).abs() < 1e-6);
        assert!(report.min_cosine_similarity > 0.98);
    }

    #[test]
    fn decode_quality_gate_passes_when_all_metrics_are_inside_limits() {
        let report = compare_decode_outputs(
            &[vec![1.0, 2.0, 3.0], vec![0.5, -0.5, 1.0]],
            &[vec![1.0, 2.01, 2.99], vec![0.51, -0.49, 0.99]],
        )
        .expect("compare outputs");
        let gate = TurboQuantDecodeQualityGate::new(0.02, 0.02, 0.999);

        let decision = gate.evaluate(&report);

        assert!(decision.passed);
        assert!(decision.max_abs_diff_passed);
        assert!(decision.mean_abs_diff_passed);
        assert!(decision.min_cosine_similarity_passed);
        assert_eq!(decision.max_abs_diff_limit, 0.02);
        assert_eq!(decision.mean_abs_diff_limit, 0.02);
        assert_eq!(decision.min_cosine_similarity_limit, 0.999);
    }

    #[test]
    fn decode_quality_gate_reports_individual_failed_conditions() {
        let report = compare_decode_outputs(
            &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            &[vec![0.0, 1.0, 0.0], vec![0.2, 0.9, 0.0]],
        )
        .expect("compare outputs");
        let gate = TurboQuantDecodeQualityGate::new(0.5, 0.1, 0.99);

        let decision = gate.evaluate(&report);

        assert!(!decision.passed);
        assert!(!decision.max_abs_diff_passed);
        assert!(!decision.mean_abs_diff_passed);
        assert!(!decision.min_cosine_similarity_passed);
        assert!(decision.max_abs_diff > decision.max_abs_diff_limit);
        assert!(decision.mean_abs_diff > decision.mean_abs_diff_limit);
        assert!(decision.min_cosine_similarity < decision.min_cosine_similarity_limit);
    }

    #[test]
    fn decode_quality_gate_presets_are_ordered_by_promotion_confidence() {
        const {
            assert!(
                TurboQuantDecodeQualityGate::STRICT_DEBUG.max_abs_diff
                    <= TurboQuantDecodeQualityGate::REFERENCE_K8V4.max_abs_diff
            );
            assert!(
                TurboQuantDecodeQualityGate::REFERENCE_K8V4.max_abs_diff
                    <= TurboQuantDecodeQualityGate::RESEARCH_LOOSE.max_abs_diff
            );
            assert!(
                TurboQuantDecodeQualityGate::STRICT_DEBUG.mean_abs_diff
                    <= TurboQuantDecodeQualityGate::REFERENCE_K8V4.mean_abs_diff
            );
            assert!(
                TurboQuantDecodeQualityGate::REFERENCE_K8V4.mean_abs_diff
                    <= TurboQuantDecodeQualityGate::RESEARCH_LOOSE.mean_abs_diff
            );
            assert!(
                TurboQuantDecodeQualityGate::STRICT_DEBUG.min_cosine_similarity
                    >= TurboQuantDecodeQualityGate::REFERENCE_K8V4.min_cosine_similarity
            );
            assert!(
                TurboQuantDecodeQualityGate::REFERENCE_K8V4.min_cosine_similarity
                    >= TurboQuantDecodeQualityGate::RESEARCH_LOOSE.min_cosine_similarity
            );
        }
    }

    #[test]
    fn decode_quality_profiles_map_quantization_presets_to_gates() {
        assert_eq!(
            TurboQuantDecodeQualityProfile::StrictDebug.gate(),
            TurboQuantDecodeQualityGate::STRICT_DEBUG
        );
        assert_eq!(
            TurboQuantDecodeQualityProfile::ReferenceK8V4.gate(),
            TurboQuantDecodeQualityGate::REFERENCE_K8V4
        );
        assert_eq!(
            TurboQuantDecodeQualityProfile::ResearchLoose.gate(),
            TurboQuantDecodeQualityGate::RESEARCH_LOOSE
        );
        assert_eq!(
            TurboQuantDecodeQualityProfile::for_quantization_preset(MlxTurboQuantPreset::K8V4),
            TurboQuantDecodeQualityProfile::ReferenceK8V4
        );
        assert_eq!(
            TurboQuantDecodeQualityProfile::for_quantization_preset(MlxTurboQuantPreset::K4V4),
            TurboQuantDecodeQualityProfile::ResearchLoose
        );
        assert_eq!(
            TurboQuantDecodeQualityProfile::for_quantization_preset(
                MlxTurboQuantPreset::K3V4Research
            ),
            TurboQuantDecodeQualityProfile::ResearchLoose
        );
    }

    #[test]
    fn decode_quality_evaluation_records_preset_profile_gate_and_decision() {
        let report =
            compare_decode_outputs(&[vec![1.0, 0.0, 0.0, 0.0]], &[vec![0.97, 0.0, 0.0, 0.0]])
                .expect("compare outputs");

        let evaluation = evaluate_decode_quality_for_preset(MlxTurboQuantPreset::K8V4, &report);

        assert_eq!(evaluation.preset, MlxTurboQuantPreset::K8V4);
        assert_eq!(
            evaluation.profile,
            TurboQuantDecodeQualityProfile::ReferenceK8V4
        );
        assert_eq!(evaluation.gate, TurboQuantDecodeQualityGate::REFERENCE_K8V4);
        assert!(evaluation.decision.passed, "evaluation was {evaluation:?}");
    }

    #[test]
    fn decode_quality_evaluation_keeps_aggressive_presets_on_research_gate() {
        let report =
            compare_decode_outputs(&[vec![1.0, 1.0, 1.0, 1.0]], &[vec![1.06, 1.0, 1.0, 1.0]])
                .expect("compare outputs");

        let reference = evaluate_decode_quality_for_preset(MlxTurboQuantPreset::K8V4, &report);
        let research = evaluate_decode_quality_for_preset(MlxTurboQuantPreset::K4V4, &report);

        assert_eq!(
            research.profile,
            TurboQuantDecodeQualityProfile::ResearchLoose
        );
        assert!(!reference.decision.passed, "reference was {reference:?}");
        assert!(research.decision.passed, "research was {research:?}");
    }

    #[test]
    fn decode_quality_check_wraps_comparison_and_preset_evaluation() {
        let comparison =
            compare_decode_outputs(&[vec![1.0, 0.0, 0.0, 0.0]], &[vec![0.99, 0.0, 0.0, 0.0]])
                .expect("compare outputs");

        let check = TurboQuantDecodeQualityCheck::for_preset(MlxTurboQuantPreset::K8V4, comparison);

        assert_eq!(check.evaluation.preset, MlxTurboQuantPreset::K8V4);
        assert_eq!(
            check.evaluation.profile,
            TurboQuantDecodeQualityProfile::ReferenceK8V4
        );
        assert!(check.evaluation.decision.passed, "check was {check:?}");
        assert_eq!(check.comparison.heads.len(), 1);
    }

    #[test]
    fn compressed_block_buffer_compares_all_head_decode_against_full_precision_oracle() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let tokens = (0..4)
            .map(|token_index| {
                vec![
                    (
                        (0..8)
                            .map(|idx| token_index as f32 * 0.06 + idx as f32 * 0.05 - 0.25)
                            .collect::<Vec<_>>(),
                        (0..8)
                            .map(|idx| token_index as f32 * 0.04 - idx as f32 * 0.015)
                            .collect::<Vec<_>>(),
                    ),
                    (
                        (0..8)
                            .map(|idx| token_index as f32 * -0.04 + idx as f32 * 0.06 + 0.15)
                            .collect::<Vec<_>>(),
                        (0..8)
                            .map(|idx| token_index as f32 * 0.03 + idx as f32 * 0.025 - 0.2)
                            .collect::<Vec<_>>(),
                    ),
                ]
            })
            .collect::<Vec<_>>();
        for (token_index, heads) in tokens.iter().enumerate() {
            buffer
                .write_token(token_index, heads)
                .expect("token write should work");
        }
        let queries = vec![
            vec![0.2, -0.1, 0.4, 0.1, -0.3, 0.5, 0.0, -0.2],
            vec![0.0, 0.3, -0.2, 0.6, -0.1, 0.2, 0.4, -0.5],
        ];
        let expected_outputs = queries
            .iter()
            .enumerate()
            .map(|(head_index, query)| {
                let history = tokens
                    .iter()
                    .map(|heads| heads[head_index].clone())
                    .collect::<Vec<_>>();
                reference_decode_attention(query, &history).expect("full precision decode")
            })
            .collect::<Vec<_>>();

        let report = buffer
            .debug_compare_attention_for_all_heads(&queries, &expected_outputs, tokens.len())
            .expect("decode comparison should work");
        let check = buffer
            .debug_evaluate_attention_quality_for_all_heads(
                &queries,
                &expected_outputs,
                tokens.len(),
            )
            .expect("decode quality check should work");

        assert_eq!(report.heads.len(), 2);
        assert!(report.max_abs_diff < 0.02, "report was {report:?}");
        assert!(report.mean_abs_diff < 0.01, "report was {report:?}");
        assert!(
            report.min_cosine_similarity > 0.999,
            "report was {report:?}"
        );

        assert!(
            TurboQuantDecodeQualityGate::REFERENCE_K8V4
                .evaluate(&report)
                .passed
        );
        assert_eq!(check.comparison, report);
        assert_eq!(check.evaluation.preset, MlxTurboQuantPreset::K8V4);
        assert_eq!(
            check.evaluation.profile,
            TurboQuantDecodeQualityProfile::ReferenceK8V4
        );
        assert!(
            check.evaluation.decision.passed,
            "quality check was {check:?}"
        );
    }

    #[test]
    fn compare_decode_outputs_fails_closed_for_shape_mismatch() {
        let error = compare_decode_outputs(&[vec![1.0, 2.0]], &[vec![1.0], vec![2.0]])
            .expect_err("head count mismatch should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedKvHeadCount {
                expected: 1,
                actual: 2,
            }
        );

        let error = compare_decode_outputs(&[vec![1.0, 2.0]], &[vec![1.0]])
            .expect_err("vector dimension mismatch should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedVectorDimension {
                expected: 2,
                actual: 1,
            }
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
