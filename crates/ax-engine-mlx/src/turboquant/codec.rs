use super::*;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TurboQuantCodecError {
    #[error("TurboQuant Metal kernel failed: {message}")]
    MetalKernelFailed { message: String },
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
    #[error("TurboQuant compressed decode plan layout does not match compressed buffer layout")]
    CompressedDecodePlanLayoutMismatch,
    #[error(
        "TurboQuant compressed decode plan expected {required_slots} written cold slots for {cold_tokens} cold tokens, got {written_slots}"
    )]
    CompressedDecodePlanIncomplete {
        cold_tokens: usize,
        required_slots: usize,
        written_slots: usize,
    },
    #[error("TurboQuant fused decode launch rejected: {status:?}")]
    FusedDecodeLaunchRejected {
        status: TurboQuantFusedDecodeCandidateStatus,
    },
    #[error("TurboQuant compressed buffer state is inconsistent: {message}")]
    InvalidCompressedBufferState { message: String },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TurboQuantKeyCentroidKind {
    #[default]
    Uniform,
    LloydMaxGaussian,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TurboQuantVectorRole {
    #[default]
    Key,
    Value,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantKeyCodecConfig {
    pub version: u32,
    pub centroid_kind: TurboQuantKeyCentroidKind,
}

impl Default for TurboQuantKeyCodecConfig {
    fn default() -> Self {
        Self::rht_lloyd_max()
    }
}

impl TurboQuantKeyCodecConfig {
    pub const fn legacy_uniform_hadamard() -> Self {
        Self {
            version: TURBOQUANT_CODEC_VERSION_UNIFORM_HADAMARD,
            centroid_kind: TurboQuantKeyCentroidKind::Uniform,
        }
    }

    pub const fn rht_lloyd_max() -> Self {
        Self {
            version: TURBOQUANT_CODEC_VERSION_RHT_LLOYD_MAX,
            centroid_kind: TurboQuantKeyCentroidKind::LloydMaxGaussian,
        }
    }

    pub const fn uses_random_signs(self) -> bool {
        self.version >= TURBOQUANT_CODEC_VERSION_RHT_LLOYD_MAX
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TurboQuantRotationSigns {
    pub dim: usize,
    pub seed: u64,
    signs: Vec<i8>,
}

impl TurboQuantRotationSigns {
    pub fn new(dim: usize, seed: u64) -> Result<Self, TurboQuantCodecError> {
        validate_key_shape_len(dim)?;
        let signs = (0..dim)
            .map(|idx| {
                if splitmix64(seed ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) & 1 == 0 {
                    1
                } else {
                    -1
                }
            })
            .collect();
        Ok(Self { dim, seed, signs })
    }

    pub fn sign_at(&self, index: usize) -> f32 {
        if self.signs[index] >= 0 { 1.0 } else { -1.0 }
    }

    pub fn apply(&self, values: &mut [f32]) -> Result<(), TurboQuantCodecError> {
        if values.len() != self.dim {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: self.dim,
                actual: values.len(),
            });
        }
        for (value, sign) in values.iter_mut().zip(&self.signs) {
            if *sign < 0 {
                *value = -*value;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedKeyVector {
    pub dim: usize,
    pub bit_width: u32,
    pub l2_norm: f32,
    pub packed_indices: Vec<u8>,
    pub codec_version: u32,
    pub centroid_kind: TurboQuantKeyCentroidKind,
    pub rotation_seed: u64,
    pub head_index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedValueGroups {
    pub element_count: usize,
    pub group_size: usize,
    pub value_bits_x2: u32,
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
pub struct TurboQuantBlockLayoutConfig {
    pub preset: TurboQuantPreset,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantCompressedDecodeReadiness {
    pub preset: TurboQuantPreset,
    pub key_bits: u32,
    pub value_bits: u32,
    pub decode_path: TurboQuantCompressedDecodePath,
    pub total_tokens: usize,
    pub cold_tokens: usize,
    pub hot_tokens: usize,
    pub query_heads: usize,
    pub query_head_dim: usize,
    pub compressed_blocks: usize,
    pub compressed_buffer_bytes: usize,
    pub required_compressed_slots: usize,
    pub written_compressed_slots: usize,
    pub quality_profile: TurboQuantDecodeQualityProfile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurboQuantFusedDecodeCandidateStatus {
    Candidate,
    FullPrecisionOnly,
    UnsupportedHeadDim,
    UnsupportedPreset,
}

pub fn turboquant_fused_decode_head_dim_supported(head_dim: usize) -> bool {
    matches!(
        head_dim,
        TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM
            | TURBOQUANT_EXTENDED_FUSED_DECODE_HEAD_DIM
            | TURBOQUANT_GEMMA4_FULL_ATTENTION_FUSED_DECODE_HEAD_DIM
    )
}

pub fn turboquant_query_head_to_kv_head(
    query_head_index: usize,
    query_heads: usize,
    kv_heads: usize,
) -> Result<usize, TurboQuantCodecError> {
    if query_heads == 0 || kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
        return Err(TurboQuantCodecError::MismatchedKvHeadCount {
            expected: kv_heads,
            actual: query_heads,
        });
    }
    if query_head_index >= query_heads {
        return Err(TurboQuantCodecError::MismatchedKvHeadCount {
            expected: query_heads,
            actual: query_head_index.saturating_add(1),
        });
    }
    Ok(query_head_index / (query_heads / kv_heads))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantFusedDecodeCandidate {
    pub status: TurboQuantFusedDecodeCandidateStatus,
    pub preset: TurboQuantPreset,
    pub head_dim: usize,
    pub compressed_blocks: usize,
    pub compressed_buffer_bytes: usize,
    pub required_compressed_slots: usize,
}

impl TurboQuantFusedDecodeCandidate {
    pub fn is_candidate(self) -> bool {
        self.status == TurboQuantFusedDecodeCandidateStatus::Candidate
    }
}

impl TurboQuantCompressedDecodeReadiness {
    pub fn fused_decode_candidate(self) -> TurboQuantFusedDecodeCandidate {
        let status = if self.decode_path == TurboQuantCompressedDecodePath::FullPrecisionOnly {
            TurboQuantFusedDecodeCandidateStatus::FullPrecisionOnly
        } else if !turboquant_fused_decode_head_dim_supported(self.query_head_dim) {
            TurboQuantFusedDecodeCandidateStatus::UnsupportedHeadDim
        } else if self.preset != TurboQuantPreset::K8V4 {
            TurboQuantFusedDecodeCandidateStatus::UnsupportedPreset
        } else {
            TurboQuantFusedDecodeCandidateStatus::Candidate
        };

        TurboQuantFusedDecodeCandidate {
            status,
            preset: self.preset,
            head_dim: self.query_head_dim,
            compressed_blocks: self.compressed_blocks,
            compressed_buffer_bytes: self.compressed_buffer_bytes,
            required_compressed_slots: self.required_compressed_slots,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantFusedDecodeLaunchDescriptor {
    pub preset: TurboQuantPreset,
    pub key_bits: u32,
    pub value_bits: u32,
    pub total_tokens: usize,
    pub cold_tokens: usize,
    pub hot_tokens: usize,
    pub n_query_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub block_tokens: usize,
    pub compressed_blocks: usize,
    pub compressed_buffer_bytes: usize,
    pub required_compressed_slots: usize,
    pub value_group_size: usize,
    pub value_group_count: usize,
    pub key_payload_bytes_per_head: usize,
    pub key_norm_bytes_per_head: usize,
    pub value_payload_bytes_per_head: usize,
    pub value_metadata_bytes_per_head: usize,
    pub raw_slot_bytes_per_head: usize,
    pub slot_bytes_per_head: usize,
    pub token_stride_bytes: usize,
    pub block_bytes: usize,
    pub key_payload_offset_in_slot: usize,
    pub key_norm_offset_in_slot: usize,
    pub value_payload_offset_in_slot: usize,
    pub value_mins_offset_in_slot: usize,
    pub value_scales_offset_in_slot: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantFusedDecodeLaunchWorkload {
    pub cold_score_elements: usize,
    pub hot_score_elements: usize,
    pub output_elements: usize,
    pub full_precision_cold_kv_bytes: usize,
    pub full_precision_total_kv_bytes: usize,
    pub compressed_key_payload_bytes: usize,
    pub compressed_key_norm_bytes: usize,
    pub compressed_value_payload_bytes: usize,
    pub compressed_value_metadata_bytes: usize,
    pub compressed_raw_slot_bytes: usize,
    pub compressed_aligned_slot_bytes: usize,
    pub hot_full_precision_kv_bytes: usize,
    pub estimated_total_read_bytes: usize,
    pub estimated_cold_saved_bytes: usize,
    pub estimated_total_saved_read_bytes: usize,
    pub cold_compression_ratio_milli: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantFusedDecodeBenchmarkEstimate {
    pub preset: TurboQuantPreset,
    pub key_bits: u32,
    pub value_bits: u32,
    pub total_tokens: usize,
    pub cold_tokens: usize,
    pub hot_tokens: usize,
    pub n_query_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub compressed_blocks: usize,
    pub cold_score_elements: usize,
    pub hot_score_elements: usize,
    pub output_elements: usize,
    pub compressed_buffer_kib: usize,
    pub full_precision_cold_kv_kib: usize,
    pub full_precision_total_kv_kib: usize,
    pub estimated_compressed_cold_kv_kib: usize,
    pub hot_full_precision_kv_kib: usize,
    pub estimated_total_read_kib: usize,
    pub estimated_cold_saved_kib: usize,
    pub estimated_total_saved_read_kib: usize,
    pub cold_compression_ratio_milli: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurboQuantFusedDecodePromotionStatus {
    Ready,
    QualityPresetMismatch,
    QualityGateFailed,
    NoColdSavings,
}

impl TurboQuantFusedDecodePromotionStatus {
    pub const fn code(self) -> u32 {
        match self {
            Self::Ready => 1,
            Self::QualityPresetMismatch => 2,
            Self::QualityGateFailed => 3,
            Self::NoColdSavings => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantFusedDecodePromotionReadiness {
    pub status: TurboQuantFusedDecodePromotionStatus,
    pub preset: TurboQuantPreset,
    pub quality_profile: TurboQuantDecodeQualityProfile,
    pub benchmark_estimate: TurboQuantFusedDecodeBenchmarkEstimate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantFusedDecodePromotionEvidence {
    pub readiness: TurboQuantFusedDecodePromotionReadiness,
    pub quality_passed: bool,
    pub max_abs_diff: f32,
    pub max_abs_diff_limit: f32,
    pub mean_abs_diff: f32,
    pub mean_abs_diff_limit: f32,
    pub min_cosine_similarity: f32,
    pub min_cosine_similarity_limit: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantFusedDecodePromotionEvidenceSummary {
    pub status_code: u32,
    pub ready: bool,
    pub preset_code: u32,
    pub quality_profile_code: u32,
    pub quality_passed: bool,
    pub max_abs_diff_microunits: u32,
    pub max_abs_diff_limit_microunits: u32,
    pub mean_abs_diff_microunits: u32,
    pub mean_abs_diff_limit_microunits: u32,
    pub min_cosine_similarity_microunits: u32,
    pub min_cosine_similarity_limit_microunits: u32,
    pub estimated_cold_saved_kib: usize,
    pub estimated_total_saved_read_kib: usize,
    pub cold_compression_ratio_milli: u32,
}

impl TurboQuantFusedDecodePromotionEvidence {
    pub fn summary(&self) -> TurboQuantFusedDecodePromotionEvidenceSummary {
        TurboQuantFusedDecodePromotionEvidenceSummary {
            status_code: self.readiness.status.code(),
            ready: self.readiness.is_ready(),
            preset_code: self.readiness.preset.route_code(),
            quality_profile_code: self.readiness.quality_profile.code(),
            quality_passed: self.quality_passed,
            max_abs_diff_microunits: f32_microunits(self.max_abs_diff),
            max_abs_diff_limit_microunits: f32_microunits(self.max_abs_diff_limit),
            mean_abs_diff_microunits: f32_microunits(self.mean_abs_diff),
            mean_abs_diff_limit_microunits: f32_microunits(self.mean_abs_diff_limit),
            min_cosine_similarity_microunits: f32_microunits(self.min_cosine_similarity),
            min_cosine_similarity_limit_microunits: f32_microunits(
                self.min_cosine_similarity_limit,
            ),
            estimated_cold_saved_kib: self.readiness.benchmark_estimate.estimated_cold_saved_kib,
            estimated_total_saved_read_kib: self
                .readiness
                .benchmark_estimate
                .estimated_total_saved_read_kib,
            cold_compression_ratio_milli: self
                .readiness
                .benchmark_estimate
                .cold_compression_ratio_milli,
        }
    }
}

impl TurboQuantFusedDecodePromotionReadiness {
    pub fn is_ready(self) -> bool {
        self.status == TurboQuantFusedDecodePromotionStatus::Ready
    }

    pub fn fallback_preset(self) -> Option<TurboQuantPreset> {
        if self.status == TurboQuantFusedDecodePromotionStatus::QualityGateFailed
            && self.preset == TurboQuantPreset::K8V4
        {
            Some(TurboQuantPreset::K16V4)
        } else {
            None
        }
    }

    pub fn effective_preset(self) -> TurboQuantPreset {
        self.fallback_preset().unwrap_or(self.preset)
    }
}

impl TurboQuantFusedDecodeLaunchDescriptor {
    pub fn workload(self) -> TurboQuantFusedDecodeLaunchWorkload {
        let compressed_key_payload_bytes =
            self.required_compressed_slots * self.key_payload_bytes_per_head;
        let compressed_key_norm_bytes =
            self.required_compressed_slots * self.key_norm_bytes_per_head;
        let compressed_value_payload_bytes =
            self.required_compressed_slots * self.value_payload_bytes_per_head;
        let compressed_value_metadata_bytes =
            self.required_compressed_slots * self.value_metadata_bytes_per_head;
        let compressed_raw_slot_bytes =
            self.required_compressed_slots * self.raw_slot_bytes_per_head;
        let compressed_aligned_slot_bytes =
            self.required_compressed_slots * self.slot_bytes_per_head;
        let full_precision_cold_kv_bytes =
            full_precision_kv_bytes(self.cold_tokens, self.n_kv_heads, self.head_dim);
        let hot_full_precision_kv_bytes =
            full_precision_kv_bytes(self.hot_tokens, self.n_kv_heads, self.head_dim);
        let full_precision_total_kv_bytes =
            full_precision_cold_kv_bytes.saturating_add(hot_full_precision_kv_bytes);
        let estimated_total_read_bytes =
            compressed_raw_slot_bytes.saturating_add(hot_full_precision_kv_bytes);

        TurboQuantFusedDecodeLaunchWorkload {
            cold_score_elements: self.cold_tokens * self.n_query_heads,
            hot_score_elements: self.hot_tokens * self.n_query_heads,
            output_elements: self.n_query_heads * self.head_dim,
            full_precision_cold_kv_bytes,
            full_precision_total_kv_bytes,
            compressed_key_payload_bytes,
            compressed_key_norm_bytes,
            compressed_value_payload_bytes,
            compressed_value_metadata_bytes,
            compressed_raw_slot_bytes,
            compressed_aligned_slot_bytes,
            hot_full_precision_kv_bytes,
            estimated_total_read_bytes,
            estimated_cold_saved_bytes: full_precision_cold_kv_bytes
                .saturating_sub(compressed_raw_slot_bytes),
            estimated_total_saved_read_bytes: full_precision_total_kv_bytes
                .saturating_sub(estimated_total_read_bytes),
            cold_compression_ratio_milli: compression_ratio_milli(
                compressed_raw_slot_bytes,
                full_precision_cold_kv_bytes,
            ),
        }
    }

    pub fn benchmark_estimate(self) -> TurboQuantFusedDecodeBenchmarkEstimate {
        let workload = self.workload();
        TurboQuantFusedDecodeBenchmarkEstimate {
            preset: self.preset,
            key_bits: self.key_bits,
            value_bits: self.value_bits,
            total_tokens: self.total_tokens,
            cold_tokens: self.cold_tokens,
            hot_tokens: self.hot_tokens,
            n_query_heads: self.n_query_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            compressed_blocks: self.compressed_blocks,
            cold_score_elements: workload.cold_score_elements,
            hot_score_elements: workload.hot_score_elements,
            output_elements: workload.output_elements,
            compressed_buffer_kib: kib_ceil_usize(self.compressed_buffer_bytes),
            full_precision_cold_kv_kib: kib_ceil_usize(workload.full_precision_cold_kv_bytes),
            full_precision_total_kv_kib: kib_ceil_usize(workload.full_precision_total_kv_bytes),
            estimated_compressed_cold_kv_kib: kib_ceil_usize(workload.compressed_raw_slot_bytes),
            hot_full_precision_kv_kib: kib_ceil_usize(workload.hot_full_precision_kv_bytes),
            estimated_total_read_kib: kib_ceil_usize(workload.estimated_total_read_bytes),
            estimated_cold_saved_kib: kib_ceil_usize(workload.estimated_cold_saved_bytes),
            estimated_total_saved_read_kib: kib_ceil_usize(
                workload.estimated_total_saved_read_bytes,
            ),
            cold_compression_ratio_milli: workload.cold_compression_ratio_milli,
        }
    }

    pub fn promotion_readiness(
        self,
        quality_check: &TurboQuantDecodeQualityCheck,
    ) -> TurboQuantFusedDecodePromotionReadiness {
        let workload = self.workload();
        let benchmark_estimate = self.benchmark_estimate();
        let status = if quality_check.evaluation.preset != self.preset {
            TurboQuantFusedDecodePromotionStatus::QualityPresetMismatch
        } else if !quality_check.evaluation.decision.passed {
            TurboQuantFusedDecodePromotionStatus::QualityGateFailed
        } else if workload.estimated_cold_saved_bytes == 0 {
            TurboQuantFusedDecodePromotionStatus::NoColdSavings
        } else {
            TurboQuantFusedDecodePromotionStatus::Ready
        };

        TurboQuantFusedDecodePromotionReadiness {
            status,
            preset: self.preset,
            quality_profile: quality_check.evaluation.profile,
            benchmark_estimate,
        }
    }

    pub fn promotion_evidence(
        self,
        quality_check: &TurboQuantDecodeQualityCheck,
    ) -> TurboQuantFusedDecodePromotionEvidence {
        let readiness = self.promotion_readiness(quality_check);
        let decision = quality_check.evaluation.decision;
        TurboQuantFusedDecodePromotionEvidence {
            readiness,
            quality_passed: decision.passed,
            max_abs_diff: decision.max_abs_diff,
            max_abs_diff_limit: decision.max_abs_diff_limit,
            mean_abs_diff: decision.mean_abs_diff,
            mean_abs_diff_limit: decision.mean_abs_diff_limit,
            min_cosine_similarity: decision.min_cosine_similarity,
            min_cosine_similarity_limit: decision.min_cosine_similarity_limit,
        }
    }
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

    pub fn validate_compressed_buffer(
        self,
        buffer: &TurboQuantCompressedBlockBuffer,
    ) -> Result<(), TurboQuantCodecError> {
        let written_slots = self.written_compressed_slots(buffer)?;

        if written_slots != self.required_compressed_slots {
            return Err(TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens: self.cold_tokens,
                required_slots: self.required_compressed_slots,
                written_slots,
            });
        }

        Ok(())
    }

    pub fn validate_queries(self, queries: &[Vec<f32>]) -> Result<(), TurboQuantCodecError> {
        self.validate_query_count(queries.len())?;

        for query in queries {
            if query.len() != self.layout.config.head_dim {
                return Err(TurboQuantCodecError::MismatchedVectorDimension {
                    expected: self.layout.config.head_dim,
                    actual: query.len(),
                });
            }
        }

        Ok(())
    }

    pub fn validate_query_count(self, n_query_heads: usize) -> Result<(), TurboQuantCodecError> {
        if n_query_heads == 0 || !n_query_heads.is_multiple_of(self.layout.config.n_kv_heads) {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: self.layout.config.n_kv_heads,
                actual: n_query_heads,
            });
        }

        Ok(())
    }

    pub fn validate_decode_inputs(
        self,
        buffer: &TurboQuantCompressedBlockBuffer,
        queries: &[Vec<f32>],
    ) -> Result<(), TurboQuantCodecError> {
        self.decode_readiness(buffer, queries).map(|_| ())
    }

    pub fn decode_readiness(
        self,
        buffer: &TurboQuantCompressedBlockBuffer,
        queries: &[Vec<f32>],
    ) -> Result<TurboQuantCompressedDecodeReadiness, TurboQuantCodecError> {
        self.validate_queries(queries)?;
        self.decode_readiness_for_query_count(buffer, queries.len())
    }

    pub fn decode_readiness_for_query_count(
        self,
        buffer: &TurboQuantCompressedBlockBuffer,
        n_query_heads: usize,
    ) -> Result<TurboQuantCompressedDecodeReadiness, TurboQuantCodecError> {
        self.validate_query_count(n_query_heads)?;
        let written_compressed_slots = self.written_compressed_slots(buffer)?;

        if written_compressed_slots != self.required_compressed_slots {
            return Err(TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens: self.cold_tokens,
                required_slots: self.required_compressed_slots,
                written_slots: written_compressed_slots,
            });
        }

        Ok(TurboQuantCompressedDecodeReadiness {
            preset: self.layout.config.preset,
            key_bits: self.layout.config.preset.key_bits(),
            value_bits: self.layout.config.preset.value_bits(),
            decode_path: self.decode_path,
            total_tokens: self.total_tokens,
            cold_tokens: self.cold_tokens,
            hot_tokens: self.hot_tokens,
            query_heads: n_query_heads,
            query_head_dim: self.layout.config.head_dim,
            compressed_blocks: self.compressed_blocks,
            compressed_buffer_bytes: self.compressed_buffer_bytes,
            required_compressed_slots: self.required_compressed_slots,
            written_compressed_slots,
            quality_profile: self.quality_profile,
        })
    }

    pub fn fused_decode_launch_descriptor(
        self,
        buffer: &TurboQuantCompressedBlockBuffer,
        queries: &[Vec<f32>],
    ) -> Result<TurboQuantFusedDecodeLaunchDescriptor, TurboQuantCodecError> {
        let readiness = self.decode_readiness(buffer, queries)?;
        self.fused_decode_launch_descriptor_from_readiness(readiness)
    }

    pub fn fused_decode_launch_descriptor_for_query_count(
        self,
        buffer: &TurboQuantCompressedBlockBuffer,
        n_query_heads: usize,
    ) -> Result<TurboQuantFusedDecodeLaunchDescriptor, TurboQuantCodecError> {
        let readiness = self.decode_readiness_for_query_count(buffer, n_query_heads)?;
        self.fused_decode_launch_descriptor_from_readiness(readiness)
    }

    fn fused_decode_launch_descriptor_from_readiness(
        self,
        readiness: TurboQuantCompressedDecodeReadiness,
    ) -> Result<TurboQuantFusedDecodeLaunchDescriptor, TurboQuantCodecError> {
        let candidate = readiness.fused_decode_candidate();
        if !candidate.is_candidate() {
            return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
                status: candidate.status,
            });
        }

        let slot_zero = self.layout.address(0, 0, 0)?;
        Ok(TurboQuantFusedDecodeLaunchDescriptor {
            preset: readiness.preset,
            key_bits: readiness.key_bits,
            value_bits: readiness.value_bits,
            total_tokens: readiness.total_tokens,
            cold_tokens: readiness.cold_tokens,
            hot_tokens: readiness.hot_tokens,
            n_query_heads: readiness.query_heads,
            n_kv_heads: self.layout.config.n_kv_heads,
            head_dim: readiness.query_head_dim,
            block_tokens: self.layout.config.block_tokens,
            compressed_blocks: readiness.compressed_blocks,
            compressed_buffer_bytes: readiness.compressed_buffer_bytes,
            required_compressed_slots: readiness.required_compressed_slots,
            value_group_size: self.layout.config.value_group_size,
            value_group_count: self.layout.value_group_count,
            key_payload_bytes_per_head: self.layout.key_payload_bytes_per_head,
            key_norm_bytes_per_head: self.layout.key_norm_bytes_per_head,
            value_payload_bytes_per_head: self.layout.value_payload_bytes_per_head,
            value_metadata_bytes_per_head: self.layout.value_metadata_bytes_per_head,
            raw_slot_bytes_per_head: self.layout.raw_slot_bytes_per_head,
            slot_bytes_per_head: self.layout.slot_bytes_per_head,
            token_stride_bytes: self.layout.token_stride_bytes,
            block_bytes: self.layout.block_bytes,
            key_payload_offset_in_slot: slot_zero.key_payload_offset_bytes,
            key_norm_offset_in_slot: slot_zero.key_norm_offset_bytes,
            value_payload_offset_in_slot: slot_zero.value_payload_offset_bytes,
            value_mins_offset_in_slot: slot_zero.value_mins_offset_bytes,
            value_scales_offset_in_slot: slot_zero.value_scales_offset_bytes,
        })
    }

    fn written_compressed_slots(
        self,
        buffer: &TurboQuantCompressedBlockBuffer,
    ) -> Result<usize, TurboQuantCodecError> {
        if !self.needs_compressed_decode() {
            return Ok(0);
        }
        if buffer.layout != self.layout {
            return Err(TurboQuantCodecError::CompressedDecodePlanLayoutMismatch);
        }

        // Sequential shadow sync writes every slot for tokens [0, token_count),
        // so full coverage makes the per-slot scan redundant; this runs every
        // decode step and the scan is O(cold_tokens * n_kv_heads).
        let n_kv_heads = self.layout.config.n_kv_heads;
        if buffer
            .token_count
            .checked_mul(n_kv_heads)
            .is_some_and(|full| full == buffer.written_slot_count)
        {
            return Ok(self
                .cold_tokens
                .min(buffer.token_count)
                .saturating_mul(n_kv_heads));
        }

        let mut written_slots = 0usize;
        for token_index in 0..self.cold_tokens {
            for head_index in 0..n_kv_heads {
                let slot_index = buffer.slot_index(token_index, head_index)?;
                if buffer
                    .written_slots
                    .get(slot_index)
                    .copied()
                    .unwrap_or(false)
                {
                    written_slots = written_slots.saturating_add(1);
                }
            }
        }

        Ok(written_slots)
    }
}

impl TurboQuantBlockLayout {
    pub fn new(config: TurboQuantBlockLayoutConfig) -> Result<Self, TurboQuantCodecError> {
        validate_layout_nonzero("block_tokens", config.block_tokens)?;
        validate_layout_nonzero("n_kv_heads", config.n_kv_heads)?;
        validate_layout_nonzero("head_dim", config.head_dim)?;
        validate_layout_nonzero("value_group_size", config.value_group_size)?;
        validate_key_shape_len(config.head_dim)?;
        validate_supported_key_bits(config.preset.key_bits())?;

        let value_group_count = config.head_dim.div_ceil(config.value_group_size);
        let key_payload_bytes_per_head =
            packed_index_bytes(config.head_dim, config.preset.key_bits())?;
        let key_norm_bytes_per_head = std::mem::size_of::<f32>();
        let value_payload_bytes_per_head =
            packed_value_bytes_for_preset(config.head_dim, config.preset)?;
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
    key_codec: TurboQuantKeyCodecConfig,
    bytes: Vec<u8>,
    written_slots: Vec<bool>,
    token_count: usize,
    written_slot_count: usize,
    k_deq_buf: Vec<Vec<u16>>,
    v_deq_buf: Vec<Vec<u16>>,
    deq_offset: usize,
    deq_alloc: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantCompressedBlockBufferState {
    pub bytes: Vec<u8>,
    pub written_slots: Vec<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TurboQuantCompressedBlockBufferMetaState {
    pub layout: TurboQuantBlockLayout,
    pub key_codec: TurboQuantKeyCodecConfig,
    pub token_count: usize,
    pub written_slot_count: usize,
}

impl TurboQuantCompressedBlockBuffer {
    pub fn new(layout: TurboQuantBlockLayout) -> Self {
        Self::new_with_key_codec(layout, TurboQuantKeyCodecConfig::default())
    }

    pub fn new_legacy_uniform_hadamard(layout: TurboQuantBlockLayout) -> Self {
        Self::new_with_key_codec(layout, TurboQuantKeyCodecConfig::legacy_uniform_hadamard())
    }

    pub fn new_with_key_codec(
        layout: TurboQuantBlockLayout,
        key_codec: TurboQuantKeyCodecConfig,
    ) -> Self {
        Self {
            layout,
            key_codec,
            bytes: Vec::new(),
            written_slots: Vec::new(),
            token_count: 0,
            written_slot_count: 0,
            k_deq_buf: Vec::new(),
            v_deq_buf: Vec::new(),
            deq_offset: 0,
            deq_alloc: 0,
        }
    }

    pub fn layout(&self) -> &TurboQuantBlockLayout {
        &self.layout
    }

    pub fn key_codec(&self) -> TurboQuantKeyCodecConfig {
        self.key_codec
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn state(&self) -> TurboQuantCompressedBlockBufferState {
        TurboQuantCompressedBlockBufferState {
            bytes: self.bytes.clone(),
            written_slots: self.written_slots.clone(),
        }
    }

    pub fn meta_state(&self) -> TurboQuantCompressedBlockBufferMetaState {
        TurboQuantCompressedBlockBufferMetaState {
            layout: self.layout,
            key_codec: self.key_codec,
            token_count: self.token_count,
            written_slot_count: self.written_slot_count,
        }
    }

    pub fn from_state(
        meta_state: TurboQuantCompressedBlockBufferMetaState,
        state: TurboQuantCompressedBlockBufferState,
    ) -> Result<Self, TurboQuantCodecError> {
        let expected_bytes = meta_state
            .layout
            .buffer_bytes_for_tokens(meta_state.token_count)?;
        if state.bytes.len() != expected_bytes {
            return Err(TurboQuantCodecError::InvalidCompressedBufferState {
                message: format!(
                    "expected {expected_bytes} bytes for {} tokens, got {}",
                    meta_state.token_count,
                    state.bytes.len()
                ),
            });
        }

        let expected_slots = checked_mul(
            checked_mul(
                meta_state
                    .layout
                    .block_count_for_tokens(meta_state.token_count),
                meta_state.layout.config.block_tokens,
            )?,
            meta_state.layout.config.n_kv_heads,
        )?;
        if state.written_slots.len() != expected_slots {
            return Err(TurboQuantCodecError::InvalidCompressedBufferState {
                message: format!(
                    "expected {expected_slots} written-slot flags, got {}",
                    state.written_slots.len()
                ),
            });
        }

        let actual_written_slot_count = state
            .written_slots
            .iter()
            .filter(|written| **written)
            .count();
        if actual_written_slot_count != meta_state.written_slot_count {
            return Err(TurboQuantCodecError::InvalidCompressedBufferState {
                message: format!(
                    "meta written_slot_count={} but state contains {actual_written_slot_count} flags",
                    meta_state.written_slot_count
                ),
            });
        }

        Ok(Self {
            layout: meta_state.layout,
            key_codec: meta_state.key_codec,
            bytes: state.bytes,
            written_slots: state.written_slots,
            token_count: meta_state.token_count,
            written_slot_count: meta_state.written_slot_count,
            k_deq_buf: Vec::new(),
            v_deq_buf: Vec::new(),
            deq_offset: 0,
            deq_alloc: 0,
        })
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn written_slot_count(&self) -> usize {
        self.written_slot_count
    }

    pub fn dequant_buffer_token_count(&self) -> usize {
        self.deq_offset
    }

    pub fn dequant_buffer_alloc_tokens(&self) -> usize {
        self.deq_alloc
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

        let key = encode_key_vector_for_head_with_codec(
            key,
            self.layout.config.preset,
            head_index,
            self.key_codec,
        )?;
        let value = encode_value_groups_for_preset(
            value,
            self.layout.config.value_group_size,
            self.layout.config.preset,
        )?;

        self.zero_slot(&address)?;
        self.write_compressed_slot_at(&address, &key, &value)?;

        let slot_index = self.slot_index(token_index, head_index)?;
        if !self.written_slots[slot_index] {
            self.written_slot_count = self.written_slot_count.saturating_add(1);
        }
        self.written_slots[slot_index] = true;
        self.token_count = self.token_count.max(token_index.saturating_add(1));
        self.invalidate_dequant_buffers();
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
            .enumerate()
            .map(|(head_index, (key, value))| {
                self.validate_slot_vectors(key, value)?;
                Ok(TurboQuantCompressedHeadSlot {
                    key: encode_key_vector_for_head_with_codec(
                        key,
                        self.layout.config.preset,
                        head_index,
                        self.key_codec,
                    )?,
                    value: encode_value_groups_for_preset(
                        value,
                        self.layout.config.value_group_size,
                        self.layout.config.preset,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, TurboQuantCodecError>>()?;

        self.ensure_capacity_for_tokens(token_index.saturating_add(1))?;
        for (head_index, slot) in compressed_heads.iter().enumerate() {
            let address = self.layout.address_for_token(token_index, head_index)?;
            self.zero_slot(&address)?;
            self.write_compressed_slot_at(&address, &slot.key, &slot.value)?;
            let slot_index = self.slot_index(token_index, head_index)?;
            if !self.written_slots[slot_index] {
                self.written_slot_count = self.written_slot_count.saturating_add(1);
            }
            self.written_slots[slot_index] = true;
        }
        self.token_count = self.token_count.max(token_index.saturating_add(1));
        self.update_dequant_buffers_after_token_write(token_index, heads)?;
        Ok(())
    }

    pub fn write_token_with_encoded_k8_keys(
        &mut self,
        token_index: usize,
        heads: &[FullPrecisionKvTokenVectors],
        packed_key_bytes: &[u8],
        key_norms: &[f32],
    ) -> Result<(), TurboQuantCodecError> {
        if self.layout.config.preset != TurboQuantPreset::K8V4
            || self.key_codec != TurboQuantKeyCodecConfig::rht_lloyd_max()
        {
            return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
                status: TurboQuantFusedDecodeCandidateStatus::UnsupportedPreset,
            });
        }
        if heads.len() != self.layout.config.n_kv_heads {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: self.layout.config.n_kv_heads,
                actual: heads.len(),
            });
        }
        if key_norms.len() != self.layout.config.n_kv_heads {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: self.layout.config.n_kv_heads,
                actual: key_norms.len(),
            });
        }
        let expected_key_bytes = checked_mul(
            self.layout.config.n_kv_heads,
            self.layout.key_payload_bytes_per_head,
        )?;
        if packed_key_bytes.len() != expected_key_bytes {
            return Err(TurboQuantCodecError::InvalidCompressedBufferState {
                message: format!(
                    "expected {expected_key_bytes} encoded K8 key bytes, got {}",
                    packed_key_bytes.len()
                ),
            });
        }

        let compressed_heads = heads
            .iter()
            .enumerate()
            .map(|(head_index, (key, value))| {
                self.validate_slot_vectors(key, value)?;
                let key_start = checked_mul(head_index, self.layout.key_payload_bytes_per_head)?;
                let key_end =
                    checked_add_many(&[key_start, self.layout.key_payload_bytes_per_head])?;
                Ok(TurboQuantCompressedHeadSlot {
                    key: QuantizedKeyVector {
                        dim: self.layout.config.head_dim,
                        bit_width: 8,
                        l2_norm: key_norms[head_index],
                        packed_indices: packed_key_bytes[key_start..key_end].to_vec(),
                        codec_version: self.key_codec.version,
                        centroid_kind: self.key_codec.centroid_kind,
                        rotation_seed: turboquant_rotation_seed(
                            self.layout.config.head_dim,
                            head_index,
                            TurboQuantVectorRole::Key,
                        ),
                        head_index,
                    },
                    value: encode_value_groups_for_preset(
                        value,
                        self.layout.config.value_group_size,
                        self.layout.config.preset,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, TurboQuantCodecError>>()?;

        self.ensure_capacity_for_tokens(token_index.saturating_add(1))?;
        for (head_index, slot) in compressed_heads.iter().enumerate() {
            let address = self.layout.address_for_token(token_index, head_index)?;
            self.zero_slot(&address)?;
            self.write_compressed_slot_at(&address, &slot.key, &slot.value)?;
            let slot_index = self.slot_index(token_index, head_index)?;
            if !self.written_slots[slot_index] {
                self.written_slot_count = self.written_slot_count.saturating_add(1);
            }
            self.written_slots[slot_index] = true;
        }
        self.token_count = self.token_count.max(token_index.saturating_add(1));
        self.update_dequant_buffers_after_token_write(token_index, heads)?;
        Ok(())
    }

    pub fn truncate_token_count(&mut self, token_count: usize) -> Result<(), TurboQuantCodecError> {
        if token_count >= self.token_count {
            return Ok(());
        }

        let old_token_count = self.token_count;
        for token_index in token_count..old_token_count {
            for head_index in 0..self.layout.config.n_kv_heads {
                let address = self.layout.address_for_token(token_index, head_index)?;
                self.zero_slot(&address)?;
                let slot_index = self.slot_index(token_index, head_index)?;
                if self.written_slots.get(slot_index).copied().unwrap_or(false) {
                    self.written_slot_count = self.written_slot_count.saturating_sub(1);
                    self.written_slots[slot_index] = false;
                }
            }
        }

        self.token_count = token_count;
        let buffer_bytes = self.layout.buffer_bytes_for_tokens(token_count)?;
        self.bytes.truncate(buffer_bytes);
        let slot_count = checked_mul(
            checked_mul(
                self.layout.block_count_for_tokens(token_count),
                self.layout.config.block_tokens,
            )?,
            self.layout.config.n_kv_heads,
        )?;
        self.written_slots.truncate(slot_count);
        self.invalidate_dequant_buffers();
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
            key: self.read_key_at(&address, head_index),
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

    pub fn debug_reconstruct_head_history_cached(
        &mut self,
        head_index: usize,
        token_count: usize,
    ) -> Result<Vec<FullPrecisionKvTokenVectors>, TurboQuantCodecError> {
        self.validate_head_index(head_index)?;
        self.ensure_dequant_buffers_for_tokens(token_count)?;
        let head_dim = self.layout.config.head_dim;
        let key_head =
            self.k_deq_buf
                .get(head_index)
                .ok_or(TurboQuantCodecError::MismatchedKvHeadCount {
                    expected: self.layout.config.n_kv_heads,
                    actual: head_index.saturating_add(1),
                })?;
        let value_head =
            self.v_deq_buf
                .get(head_index)
                .ok_or(TurboQuantCodecError::MismatchedKvHeadCount {
                    expected: self.layout.config.n_kv_heads,
                    actual: head_index.saturating_add(1),
                })?;

        (0..token_count)
            .map(|token_index| {
                let start = token_index.saturating_mul(head_dim);
                let end = start.saturating_add(head_dim);
                Ok((
                    key_head[start..end]
                        .iter()
                        .map(|bits| f16_bits_to_f32(*bits))
                        .collect(),
                    value_head[start..end]
                        .iter()
                        .map(|bits| f16_bits_to_f32(*bits))
                        .collect(),
                ))
            })
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

    pub fn debug_decode_partition_stats_for_head(
        &self,
        query: &[f32],
        head_index: usize,
        token_count: usize,
    ) -> Result<TurboQuantAttentionPartitionStats, TurboQuantCodecError> {
        let tokens = self.debug_reconstruct_head_history(head_index, token_count)?;
        reference_decode_attention_partition_stats(query, &tokens)
    }

    pub fn debug_decode_attention_for_all_heads(
        &self,
        queries: &[Vec<f32>],
        token_count: usize,
    ) -> Result<Vec<Vec<f32>>, TurboQuantCodecError> {
        if queries.is_empty() || !queries.len().is_multiple_of(self.layout.config.n_kv_heads) {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: self.layout.config.n_kv_heads,
                actual: queries.len(),
            });
        }

        queries
            .iter()
            .enumerate()
            .map(|(query_head_index, query)| {
                let kv_head_index = turboquant_query_head_to_kv_head(
                    query_head_index,
                    queries.len(),
                    self.layout.config.n_kv_heads,
                )?;
                self.debug_decode_attention_for_head(query, kv_head_index, token_count)
            })
            .collect()
    }

    pub fn debug_decode_partition_stats_for_all_heads(
        &self,
        queries: &[Vec<f32>],
        token_count: usize,
    ) -> Result<Vec<TurboQuantAttentionPartitionStats>, TurboQuantCodecError> {
        if queries.is_empty() || !queries.len().is_multiple_of(self.layout.config.n_kv_heads) {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: self.layout.config.n_kv_heads,
                actual: queries.len(),
            });
        }

        queries
            .iter()
            .enumerate()
            .map(|(query_head_index, query)| {
                let kv_head_index = turboquant_query_head_to_kv_head(
                    query_head_index,
                    queries.len(),
                    self.layout.config.n_kv_heads,
                )?;
                self.debug_decode_partition_stats_for_head(query, kv_head_index, token_count)
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

    pub(crate) fn invalidate_dequant_buffers(&mut self) {
        self.k_deq_buf.clear();
        self.v_deq_buf.clear();
        self.deq_offset = 0;
        self.deq_alloc = 0;
    }

    fn update_dequant_buffers_after_token_write(
        &mut self,
        token_index: usize,
        heads: &[FullPrecisionKvTokenVectors],
    ) -> Result<(), TurboQuantCodecError> {
        if token_index != self.deq_offset || heads.len() != self.layout.config.n_kv_heads {
            self.invalidate_dequant_buffers();
            return Ok(());
        }

        self.ensure_dequant_buffer_capacity(token_index.saturating_add(1))?;
        // Cache the quantize->dequantize reconstruction of the slot that was
        // just written, not the original full-precision inputs: the lazy fill
        // in `ensure_dequant_buffers_for_tokens` reconstructs from the
        // compressed buffer, and both paths must observe identical values.
        let token = self.debug_reconstruct_token(token_index)?;
        let head_dim = self.layout.config.head_dim;
        for (head_index, (key, value)) in token.iter().enumerate() {
            let start = token_index.saturating_mul(head_dim);
            let end = start.saturating_add(head_dim);
            for (target, source) in self.k_deq_buf[head_index][start..end].iter_mut().zip(key) {
                *target = f32_to_f16_bits(*source);
            }
            for (target, source) in self.v_deq_buf[head_index][start..end].iter_mut().zip(value) {
                *target = f32_to_f16_bits(*source);
            }
        }
        self.deq_offset = token_index.saturating_add(1);
        Ok(())
    }

    fn ensure_dequant_buffers_for_tokens(
        &mut self,
        token_count: usize,
    ) -> Result<(), TurboQuantCodecError> {
        if token_count <= self.deq_offset {
            return Ok(());
        }
        self.ensure_dequant_buffer_capacity(token_count)?;
        for token_index in self.deq_offset..token_count {
            let token = self.debug_reconstruct_token(token_index)?;
            let head_dim = self.layout.config.head_dim;
            for (head_index, (key, value)) in token.iter().enumerate() {
                let start = token_index.saturating_mul(head_dim);
                let end = start.saturating_add(head_dim);
                for (target, source) in self.k_deq_buf[head_index][start..end].iter_mut().zip(key) {
                    *target = f32_to_f16_bits(*source);
                }
                for (target, source) in self.v_deq_buf[head_index][start..end].iter_mut().zip(value)
                {
                    *target = f32_to_f16_bits(*source);
                }
            }
        }
        self.deq_offset = token_count;
        Ok(())
    }

    fn ensure_dequant_buffer_capacity(
        &mut self,
        token_count: usize,
    ) -> Result<(), TurboQuantCodecError> {
        let head_dim = self.layout.config.head_dim;
        let n_kv_heads = self.layout.config.n_kv_heads;
        if self.k_deq_buf.len() != n_kv_heads || self.v_deq_buf.len() != n_kv_heads {
            self.k_deq_buf = (0..n_kv_heads).map(|_| Vec::new()).collect();
            self.v_deq_buf = (0..n_kv_heads).map(|_| Vec::new()).collect();
            self.deq_alloc = 0;
            self.deq_offset = 0;
        }

        if token_count <= self.deq_alloc {
            return Ok(());
        }
        let new_alloc = token_count.next_multiple_of(4);
        let new_len = checked_mul(new_alloc, head_dim)?;
        for head in &mut self.k_deq_buf {
            head.resize(new_len, 0);
        }
        for head in &mut self.v_deq_buf {
            head.resize(new_len, 0);
        }
        self.deq_alloc = new_alloc;
        Ok(())
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

    fn read_key_at(
        &self,
        address: &TurboQuantSlotAddress,
        head_index: usize,
    ) -> QuantizedKeyVector {
        QuantizedKeyVector {
            dim: self.layout.config.head_dim,
            bit_width: self.layout.config.preset.key_bits(),
            l2_norm: self.read_f32(address.key_norm_offset_bytes),
            packed_indices: self.bytes[address.key_payload_offset_bytes
                ..address.key_payload_offset_bytes + self.layout.key_payload_bytes_per_head]
                .to_vec(),
            codec_version: self.key_codec.version,
            centroid_kind: self.key_codec.centroid_kind,
            rotation_seed: turboquant_rotation_seed(
                self.layout.config.head_dim,
                head_index,
                TurboQuantVectorRole::Key,
            ),
            head_index,
        }
    }

    fn read_value_at(&self, address: &TurboQuantSlotAddress) -> QuantizedValueGroups {
        QuantizedValueGroups {
            element_count: self.layout.config.head_dim,
            group_size: self.layout.config.value_group_size,
            value_bits_x2: self.layout.config.preset.value_bits_x2(),
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
    pub preset: TurboQuantPreset,
    pub vector_dim: usize,
    pub hot_window_tokens: usize,
    pub value_group_size: usize,
}

impl TurboQuantKvPrototypeConfig {
    pub fn new(
        preset: TurboQuantPreset,
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
        validate_supported_key_bits(preset.key_bits())?;
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
    preset: TurboQuantPreset,
) -> Result<TurboQuantSupportReport, TurboQuantCodecError> {
    validate_supported_key_bits(preset.key_bits())?;
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
pub(crate) struct CompressedKvToken {
    key: QuantizedKeyVector,
    value: QuantizedValueGroups,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FullPrecisionKvToken {
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
        let mut cold_tokens = Vec::with_capacity(self.cold.len());
        for token in &self.cold {
            cold_tokens.push((
                decode_key_vector(&token.key)?,
                decode_value_groups_4bit(&token.value)?,
            ));
        }
        let hot_tokens = self
            .hot
            .iter()
            .map(|token| (token.key.clone(), token.value.clone()))
            .collect::<Vec<_>>();
        reference_decode_attention_split(query, &cold_tokens, &hot_tokens)
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
                key: encode_key_vector_for_head(&token.key, self.config.preset, 0)?,
                value: encode_value_groups_for_preset(
                    &token.value,
                    self.config.value_group_size,
                    self.config.preset,
                )?,
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
    let weight_sum = weights.iter().sum::<f32>().max(f32::MIN_POSITIVE);
    let mut output = vec![0.0f32; value_dim];

    for (weight, (_, value)) in weights.iter().zip(kv_tokens) {
        let normalized_weight = *weight / weight_sum;
        for (out, value) in output.iter_mut().zip(value) {
            *out += normalized_weight * value;
        }
    }

    Ok(output)
}

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantAttentionPartitionStats {
    pub token_count: usize,
    pub value_dim: usize,
    pub max_score: f32,
    pub exp_sum: f32,
    pub weighted_value_sum: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantAttentionPartitionStatsBatch {
    pub token_count: usize,
    pub value_dim: usize,
    pub max_scores: Vec<f32>,
    pub exp_sums: Vec<f32>,
    pub weighted_value_sums: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct ValidatedTurboQuantAttentionPartitionStatsBatch<'a> {
    inner: &'a TurboQuantAttentionPartitionStatsBatch,
}

impl TurboQuantAttentionPartitionStatsBatch {
    pub fn query_heads(&self) -> usize {
        self.max_scores.len()
    }

    pub fn validate(&self) -> Result<(), TurboQuantCodecError> {
        if self.token_count == 0 {
            return Err(TurboQuantCodecError::EmptyKvHistory);
        }
        if self.exp_sums.len() != self.max_scores.len() {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: self.max_scores.len(),
                actual: self.exp_sums.len(),
            });
        }
        let expected_values = self.max_scores.len().saturating_mul(self.value_dim);
        if self.weighted_value_sums.len() != expected_values {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: expected_values,
                actual: self.weighted_value_sums.len(),
            });
        }
        Ok(())
    }

    pub fn validated(
        &self,
    ) -> Result<ValidatedTurboQuantAttentionPartitionStatsBatch<'_>, TurboQuantCodecError> {
        self.validate()?;
        Ok(ValidatedTurboQuantAttentionPartitionStatsBatch { inner: self })
    }

    pub fn weighted_value_sum_for_head(
        &self,
        head_index: usize,
    ) -> Result<&[f32], TurboQuantCodecError> {
        self.validate()?;
        self.weighted_value_sum_for_validated_head(head_index)
    }

    fn weighted_value_sum_for_validated_head(
        &self,
        head_index: usize,
    ) -> Result<&[f32], TurboQuantCodecError> {
        if head_index >= self.max_scores.len() {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: self.max_scores.len(),
                actual: head_index.saturating_add(1),
            });
        }
        let start = head_index.saturating_mul(self.value_dim);
        self.weighted_value_sums
            .get(start..start.saturating_add(self.value_dim))
            .ok_or(TurboQuantCodecError::MismatchedVectorDimension {
                expected: start.saturating_add(self.value_dim),
                actual: self.weighted_value_sums.len(),
            })
    }

    pub fn partition_stats(
        &self,
        head_index: usize,
    ) -> Result<TurboQuantAttentionPartitionStats, TurboQuantCodecError> {
        let weighted_value_sum = self.weighted_value_sum_for_head(head_index)?;
        Ok(TurboQuantAttentionPartitionStats {
            token_count: self.token_count,
            value_dim: self.value_dim,
            max_score: self.max_scores[head_index],
            exp_sum: self.exp_sums[head_index],
            weighted_value_sum: weighted_value_sum.to_vec(),
        })
    }
}

impl ValidatedTurboQuantAttentionPartitionStatsBatch<'_> {
    pub fn query_heads(&self) -> usize {
        self.inner.query_heads()
    }

    pub fn token_count(&self) -> usize {
        self.inner.token_count
    }
}

pub fn reference_decode_attention_partition_stats(
    query: &[f32],
    kv_tokens: &[FullPrecisionKvTokenVectors],
) -> Result<TurboQuantAttentionPartitionStats, TurboQuantCodecError> {
    validate_key_shape(query)?;
    if kv_tokens.is_empty() {
        return Err(TurboQuantCodecError::EmptyKvHistory);
    }

    let value_dim = kv_tokens[0].1.len();
    let scale = (query.len() as f32).sqrt().recip();
    let mut scores = Vec::with_capacity(kv_tokens.len());
    for (key, value) in kv_tokens {
        validate_attention_token(query.len(), value_dim, key, value)?;
        scores.push(
            query
                .iter()
                .zip(key)
                .map(|(left, right)| left * right)
                .sum::<f32>()
                * scale,
        );
    }

    let max_score = scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, score| acc.max(score));
    let mut exp_sum = 0.0f32;
    let mut weighted_value_sum = vec![0.0f32; value_dim];
    for (score, (_, value)) in scores.iter().zip(kv_tokens) {
        let weight = (*score - max_score).exp();
        exp_sum += weight;
        for (acc, value) in weighted_value_sum.iter_mut().zip(value) {
            *acc += weight * value;
        }
    }

    Ok(TurboQuantAttentionPartitionStats {
        token_count: kv_tokens.len(),
        value_dim,
        max_score,
        exp_sum,
        weighted_value_sum,
    })
}

pub fn merge_attention_partition_stats(
    partitions: &[TurboQuantAttentionPartitionStats],
) -> Result<Vec<f32>, TurboQuantCodecError> {
    if partitions.is_empty() {
        return Err(TurboQuantCodecError::EmptyKvHistory);
    }

    let value_dim = partitions[0].value_dim;
    for partition in partitions {
        if partition.value_dim != value_dim {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: value_dim,
                actual: partition.value_dim,
            });
        }
        if partition.weighted_value_sum.len() != value_dim {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: value_dim,
                actual: partition.weighted_value_sum.len(),
            });
        }
    }

    let global_max = partitions
        .iter()
        .map(|partition| partition.max_score)
        .fold(f32::NEG_INFINITY, |acc, score| acc.max(score));
    let mut denom = 0.0f32;
    let mut output = vec![0.0f32; value_dim];

    for partition in partitions {
        let partition_scale = (partition.max_score - global_max).exp();
        denom += partition.exp_sum * partition_scale;
        for (out, value_sum) in output.iter_mut().zip(&partition.weighted_value_sum) {
            *out += partition_scale * value_sum;
        }
    }

    let denom = denom.max(f32::MIN_POSITIVE);
    for value in &mut output {
        *value /= denom;
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn append_attention_partition_stats_batch_head_with_hot_tail_from_f32_slices(
    output: &mut Vec<f32>,
    cold_stats: ValidatedTurboQuantAttentionPartitionStatsBatch<'_>,
    query_head_index: usize,
    query: &[f32],
    k_values: &[f32],
    v_values: &[f32],
    n_kv_heads: usize,
    head_dim: usize,
    hot_token_count: usize,
    kv_head_index: usize,
) -> Result<(), TurboQuantCodecError> {
    let batch = cold_stats.inner;
    let weighted_value_sum = batch.weighted_value_sum_for_validated_head(query_head_index)?;
    let cold_max_score = batch.max_scores.get(query_head_index).copied().ok_or(
        TurboQuantCodecError::MismatchedKvHeadCount {
            expected: batch.max_scores.len(),
            actual: query_head_index.saturating_add(1),
        },
    )?;
    let cold_exp_sum = batch.exp_sums.get(query_head_index).copied().ok_or(
        TurboQuantCodecError::MismatchedKvHeadCount {
            expected: batch.max_scores.len(),
            actual: query_head_index.saturating_add(1),
        },
    )?;
    append_attention_partition_with_hot_tail_from_f32_slices(
        output,
        batch.token_count,
        batch.value_dim,
        cold_max_score,
        cold_exp_sum,
        weighted_value_sum,
        query,
        k_values,
        v_values,
        n_kv_heads,
        head_dim,
        hot_token_count,
        kv_head_index,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn append_attention_partition_with_hot_tail_from_f32_slices(
    output: &mut Vec<f32>,
    cold_token_count: usize,
    cold_value_dim: usize,
    cold_max_score: f32,
    cold_exp_sum: f32,
    cold_weighted_value_sum: &[f32],
    query: &[f32],
    k_values: &[f32],
    v_values: &[f32],
    n_kv_heads: usize,
    head_dim: usize,
    hot_token_count: usize,
    kv_head_index: usize,
) -> Result<(), TurboQuantCodecError> {
    if cold_token_count == 0 {
        return Err(TurboQuantCodecError::EmptyKvHistory);
    }
    if cold_value_dim != head_dim {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: cold_value_dim,
            actual: head_dim,
        });
    }
    if cold_weighted_value_sum.len() != head_dim {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: head_dim,
            actual: cold_weighted_value_sum.len(),
        });
    }
    if hot_token_count == 0 {
        let denom = cold_exp_sum.max(f32::MIN_POSITIVE);
        output.extend(cold_weighted_value_sum.iter().map(|value| *value / denom));
        return Ok(());
    }
    if query.len() != head_dim {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: head_dim,
            actual: query.len(),
        });
    }
    if kv_head_index >= n_kv_heads {
        return Err(TurboQuantCodecError::MismatchedKvHeadCount {
            expected: n_kv_heads,
            actual: kv_head_index.saturating_add(1),
        });
    }
    let expected_len = n_kv_heads
        .saturating_mul(hot_token_count)
        .saturating_mul(head_dim);
    if k_values.len() != expected_len {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: expected_len,
            actual: k_values.len(),
        });
    }
    if v_values.len() != expected_len {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: expected_len,
            actual: v_values.len(),
        });
    }

    let scale = (head_dim as f32).sqrt().recip();
    let head_offset = kv_head_index
        .saturating_mul(hot_token_count)
        .saturating_mul(head_dim);
    let token_range = |token_index: usize| {
        let start = head_offset.saturating_add(token_index.saturating_mul(head_dim));
        start..start.saturating_add(head_dim)
    };

    let mut hot_max_score = f32::NEG_INFINITY;
    for token_index in 0..hot_token_count {
        let key = &k_values[token_range(token_index)];
        let score = query
            .iter()
            .zip(key)
            .map(|(left, right)| left * right)
            .sum::<f32>()
            * scale;
        hot_max_score = hot_max_score.max(score);
    }

    let mut hot_exp_sum = 0.0f32;
    let output_start = output.len();
    output.resize(output_start.saturating_add(head_dim), 0.0);
    let hot_output = &mut output[output_start..];
    for token_index in 0..hot_token_count {
        let range = token_range(token_index);
        let key = &k_values[range.clone()];
        let value = &v_values[range];
        let score = query
            .iter()
            .zip(key)
            .map(|(left, right)| left * right)
            .sum::<f32>()
            * scale;
        let weight = (score - hot_max_score).exp();
        hot_exp_sum += weight;
        for (acc, value) in hot_output.iter_mut().zip(value) {
            *acc += weight * value;
        }
    }

    let global_max = cold_max_score.max(hot_max_score);
    let cold_scale = (cold_max_score - global_max).exp();
    let hot_scale = (hot_max_score - global_max).exp();
    let denom = (cold_exp_sum * cold_scale + hot_exp_sum * hot_scale).max(f32::MIN_POSITIVE);

    for (out, cold_value_sum) in hot_output.iter_mut().zip(cold_weighted_value_sum) {
        *out = (*out * hot_scale + cold_value_sum * cold_scale) / denom;
    }

    Ok(())
}

pub fn reference_decode_attention_split(
    query: &[f32],
    cold_tokens: &[FullPrecisionKvTokenVectors],
    hot_tokens: &[FullPrecisionKvTokenVectors],
) -> Result<Vec<f32>, TurboQuantCodecError> {
    match (cold_tokens.is_empty(), hot_tokens.is_empty()) {
        (true, true) => Err(TurboQuantCodecError::EmptyKvHistory),
        (true, false) => reference_decode_attention(query, hot_tokens),
        (false, true) => reference_decode_attention(query, cold_tokens),
        (false, false) => {
            let cold = reference_decode_attention_partition_stats(query, cold_tokens)?;
            let hot = reference_decode_attention_partition_stats(query, hot_tokens)?;
            merge_attention_partition_stats(&[cold, hot])
        }
    }
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

pub(crate) fn validate_attention_token(
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
    preset: TurboQuantPreset,
) -> Result<QuantizedKeyVector, TurboQuantCodecError> {
    encode_key_vector_for_head(vector, preset, 0)
}

pub fn encode_key_vector_for_head(
    vector: &[f32],
    preset: TurboQuantPreset,
    head_index: usize,
) -> Result<QuantizedKeyVector, TurboQuantCodecError> {
    encode_key_vector_for_head_with_codec(
        vector,
        preset,
        head_index,
        TurboQuantKeyCodecConfig::default(),
    )
}

pub fn encode_key_vector_for_head_with_codec(
    vector: &[f32],
    preset: TurboQuantPreset,
    head_index: usize,
    codec: TurboQuantKeyCodecConfig,
) -> Result<QuantizedKeyVector, TurboQuantCodecError> {
    let bit_width = preset.key_bits();
    validate_key_shape(vector)?;
    validate_supported_key_bits(bit_width)?;

    if preset.has_full_precision_keys() {
        return encode_full_precision_key_vector_f16(vector, preset, head_index, codec);
    }

    let l2_norm_val = l2_norm(vector);
    let norm = l2_norm_val.max(f32::EPSILON);
    let mut rotated: Vec<f32> = vector.iter().map(|v| v / norm).collect();
    let rotation_seed =
        turboquant_rotation_seed(vector.len(), head_index, TurboQuantVectorRole::Key);
    if codec.uses_random_signs() {
        randomized_hadamard_in_place(&mut rotated, rotation_seed)?;
    } else {
        hadamard_in_place(&mut rotated)?;
    }

    let centroids = key_centroids_for_dim_with_kind(bit_width, vector.len(), codec.centroid_kind)?;
    let boundaries = centroid_boundaries(&centroids);
    let indices: Vec<u8> = rotated
        .iter()
        .map(|&v| centroid_index_by_boundaries(v, &boundaries))
        .collect();
    let packed_indices = pack_indices(&indices, bit_width)?;

    Ok(QuantizedKeyVector {
        dim: vector.len(),
        bit_width,
        l2_norm: l2_norm_val,
        packed_indices,
        codec_version: codec.version,
        centroid_kind: codec.centroid_kind,
        rotation_seed,
        head_index,
    })
}

pub fn decode_key_vector(encoded: &QuantizedKeyVector) -> Result<Vec<f32>, TurboQuantCodecError> {
    if encoded.dim == 0 {
        return Err(TurboQuantCodecError::EmptyVector);
    }
    if !encoded.dim.is_power_of_two() {
        return Err(TurboQuantCodecError::NonPowerOfTwoDimension(encoded.dim));
    }
    validate_supported_key_bits(encoded.bit_width)?;

    if encoded.bit_width == 16 {
        return decode_full_precision_key_vector_f16(encoded);
    }

    let indices = unpack_indices(&encoded.packed_indices, encoded.dim, encoded.bit_width)?;
    let centroids =
        key_centroids_for_dim_with_kind(encoded.bit_width, encoded.dim, encoded.centroid_kind)?;
    let mut rotated: Vec<f32> = indices
        .into_iter()
        .map(|index| centroids[index as usize])
        .collect();
    if encoded.codec_version >= TURBOQUANT_CODEC_VERSION_RHT_LLOYD_MAX {
        inverse_randomized_hadamard_in_place(&mut rotated, encoded.rotation_seed)?;
    } else {
        hadamard_in_place(&mut rotated)?;
    }

    for v in &mut rotated {
        *v *= encoded.l2_norm;
    }

    Ok(rotated)
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
        codec_version: TURBOQUANT_CODEC_VERSION_UNIFORM_HADAMARD,
        centroid_kind: TurboQuantKeyCentroidKind::Uniform,
        rotation_seed: turboquant_rotation_seed(vector.len(), 0, TurboQuantVectorRole::Key),
        head_index: 0,
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
    legacy_uniform_key_centroids(bit_width)
}

pub fn key_centroids_for_dim(bit_width: u32, dim: usize) -> Result<Vec<f32>, TurboQuantCodecError> {
    key_centroids_for_dim_with_kind(bit_width, dim, TurboQuantKeyCentroidKind::LloydMaxGaussian)
}

pub fn key_centroids_for_dim_with_kind(
    bit_width: u32,
    dim: usize,
    kind: TurboQuantKeyCentroidKind,
) -> Result<Vec<f32>, TurboQuantCodecError> {
    validate_key_shape_len(dim)?;
    match (kind, bit_width) {
        (_, 16) => Ok(Vec::new()),
        (TurboQuantKeyCentroidKind::Uniform, _) => legacy_uniform_key_centroids(bit_width),
        (TurboQuantKeyCentroidKind::LloydMaxGaussian, 3 | 4) => {
            let scale = (dim as f32).sqrt().recip();
            Ok(raw_lloyd_max_gaussian_centroids(bit_width)?
                .into_iter()
                .map(|centroid| centroid * scale)
                .collect())
        }
        (TurboQuantKeyCentroidKind::LloydMaxGaussian, 7 | 8) => {
            legacy_uniform_key_centroids(bit_width)
        }
        (_, other) => Err(TurboQuantCodecError::UnsupportedKeyBits(other)),
    }
}

pub(crate) fn legacy_uniform_key_centroids(
    bit_width: u32,
) -> Result<Vec<f32>, TurboQuantCodecError> {
    match bit_width {
        3 | 4 | 7 | 8 => {
            let levels = 1usize << bit_width;
            let max_level = levels.saturating_sub(1).max(1) as f32;
            Ok((0..levels)
                .map(|idx| -1.0 + 2.0 * idx as f32 / max_level)
                .collect())
        }
        other => Err(TurboQuantCodecError::UnsupportedKeyBits(other)),
    }
}

pub(crate) fn raw_lloyd_max_gaussian_centroids(
    bit_width: u32,
) -> Result<Vec<f32>, TurboQuantCodecError> {
    match bit_width {
        3 => Ok(vec![
            -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
        ]),
        4 => Ok(vec![
            -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284, 0.1284, 0.3881,
            0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
        ]),
        other => Err(TurboQuantCodecError::UnsupportedKeyBits(other)),
    }
}

pub fn centroid_boundaries(centroids: &[f32]) -> Vec<f32> {
    centroids
        .windows(2)
        .map(|window| (window[0] + window[1]) * 0.5)
        .collect()
}

pub fn centroid_index_by_boundaries(value: f32, boundaries: &[f32]) -> u8 {
    boundaries
        .iter()
        .take_while(|boundary| value > **boundary)
        .count()
        .min(u8::MAX as usize) as u8
}

pub fn packed_index_bytes(
    index_count: usize,
    bit_width: u32,
) -> Result<usize, TurboQuantCodecError> {
    if !matches!(bit_width, 3 | 4 | 7 | 8 | 16) {
        return Err(TurboQuantCodecError::UnsupportedPackedBits(bit_width));
    }
    if bit_width == 16 {
        return Ok(index_count.saturating_mul(std::mem::size_of::<u16>()));
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

pub fn packed_value_bytes_for_preset(
    element_count: usize,
    preset: TurboQuantPreset,
) -> Result<usize, TurboQuantCodecError> {
    if preset.has_fractional_values() {
        let split = fractional_value_split(element_count);
        return Ok(packed_index_bytes(split, 4)?
            .saturating_add(packed_index_bytes(element_count.saturating_sub(split), 3)?));
    }
    packed_index_bytes(element_count, preset.value_bits())
}

pub fn pack_indices(indices: &[u8], bit_width: u32) -> Result<Vec<u8>, TurboQuantCodecError> {
    if !matches!(bit_width, 3 | 4 | 7 | 8) {
        return Err(TurboQuantCodecError::UnsupportedPackedBits(bit_width));
    }

    let max_value = (1u16 << bit_width) - 1;
    let mut packed = vec![0u8; packed_index_bytes(indices.len(), bit_width)?];

    match bit_width {
        8 => {
            for (i, &value) in indices.iter().enumerate() {
                if value as u16 > max_value {
                    return Err(TurboQuantCodecError::PackedIndexOutOfRange {
                        bit_width,
                        index: i,
                        value,
                    });
                }
                packed[i] = value;
            }
        }
        4 => {
            for (i, &value) in indices.iter().enumerate() {
                if value as u16 > max_value {
                    return Err(TurboQuantCodecError::PackedIndexOutOfRange {
                        bit_width,
                        index: i,
                        value,
                    });
                }
                packed[i >> 1] |= (value & 0x0f) << ((i & 1) << 2);
            }
        }
        _ => {
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
        }
    }

    Ok(packed)
}

pub fn unpack_indices(
    packed: &[u8],
    index_count: usize,
    bit_width: u32,
) -> Result<Vec<u8>, TurboQuantCodecError> {
    if !matches!(bit_width, 3 | 4 | 7 | 8) {
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
    match bit_width {
        8 => {
            indices.extend_from_slice(&packed[..index_count]);
        }
        4 => {
            for i in 0..index_count {
                let byte = packed[i >> 1];
                indices.push(if i & 1 == 0 {
                    byte & 0x0f
                } else {
                    (byte >> 4) & 0x0f
                });
            }
        }
        _ => {
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
        }
    }

    Ok(indices)
}

pub(crate) fn fractional_value_split(element_count: usize) -> usize {
    element_count / 2
}

pub(crate) fn value_bit_width_for_index(
    value_bits_x2: u32,
    element_count: usize,
    index: usize,
) -> u32 {
    if value_bits_x2 == 7 {
        if index < fractional_value_split(element_count) {
            4
        } else {
            3
        }
    } else {
        value_bits_x2 / 2
    }
}

pub(crate) fn pack_value_indices(
    indices: &[u8],
    value_bits_x2: u32,
) -> Result<Vec<u8>, TurboQuantCodecError> {
    if value_bits_x2 == 7 {
        let split = fractional_value_split(indices.len());
        let mut packed = pack_indices(&indices[..split], 4)?;
        packed.extend(pack_indices(&indices[split..], 3)?);
        return Ok(packed);
    }
    pack_indices(indices, value_bits_x2 / 2)
}

pub(crate) fn unpack_value_indices(
    packed: &[u8],
    element_count: usize,
    value_bits_x2: u32,
) -> Result<Vec<u8>, TurboQuantCodecError> {
    if value_bits_x2 == 7 {
        let split = fractional_value_split(element_count);
        let first_bytes = packed_index_bytes(split, 4)?;
        if packed.len() < first_bytes {
            return Err(TurboQuantCodecError::PackedBufferTooShort {
                bit_width: 4,
                index_count: split,
            });
        }
        let mut indices = unpack_indices(&packed[..first_bytes], split, 4)?;
        indices.extend(unpack_indices(
            &packed[first_bytes..],
            element_count.saturating_sub(split),
            3,
        )?);
        return Ok(indices);
    }
    unpack_indices(packed, element_count, value_bits_x2 / 2)
}

pub fn encode_value_groups_4bit(
    values: &[f32],
    group_size: usize,
) -> Result<QuantizedValueGroups, TurboQuantCodecError> {
    encode_value_groups_with_bits_x2(values, group_size, 8)
}

pub fn encode_value_groups_for_preset(
    values: &[f32],
    group_size: usize,
    preset: TurboQuantPreset,
) -> Result<QuantizedValueGroups, TurboQuantCodecError> {
    encode_value_groups_with_bits_x2(values, group_size, preset.value_bits_x2())
}

pub fn encode_value_groups_with_bits_x2(
    values: &[f32],
    group_size: usize,
    value_bits_x2: u32,
) -> Result<QuantizedValueGroups, TurboQuantCodecError> {
    if values.is_empty() {
        return Err(TurboQuantCodecError::EmptyVector);
    }
    let group_size = group_size.max(1);
    let group_count = values.len().div_ceil(group_size);
    let mut mins = Vec::with_capacity(group_count);
    let mut scales = Vec::with_capacity(group_count);
    let mut quantized = Vec::with_capacity(values.len());

    for (group_index, group) in values.chunks(group_size).enumerate() {
        let min = group
            .iter()
            .fold(f32::INFINITY, |acc, value| acc.min(*value));
        let max = group
            .iter()
            .fold(f32::NEG_INFINITY, |acc, value| acc.max(*value));
        let group_start = group_index.saturating_mul(group_size);
        let group_bits = (0..group.len())
            .map(|offset| {
                value_bit_width_for_index(value_bits_x2, values.len(), group_start + offset)
            })
            .min()
            .unwrap_or(4);
        let max_quantized = ((1u32 << group_bits) - 1) as f32;
        let scale = if max > min {
            (max - min) / max_quantized
        } else {
            1.0
        };
        mins.push(min);
        scales.push(scale);
        quantized.extend(group.iter().enumerate().map(|(offset, value)| {
            let bit_width =
                value_bit_width_for_index(value_bits_x2, values.len(), group_start + offset);
            let max_quantized = ((1u32 << bit_width) - 1) as f32;
            if max > min {
                ((*value - min) / scale).round().clamp(0.0, max_quantized) as u8
            } else {
                0
            }
        }));
    }

    Ok(QuantizedValueGroups {
        element_count: values.len(),
        group_size,
        value_bits_x2,
        mins,
        scales,
        packed_values: pack_value_indices(&quantized, value_bits_x2)?,
    })
}

pub fn decode_value_groups_4bit(
    encoded: &QuantizedValueGroups,
) -> Result<Vec<f32>, TurboQuantCodecError> {
    let quantized = unpack_value_indices(
        &encoded.packed_values,
        encoded.element_count,
        encoded.value_bits_x2,
    )?;
    let mut values = Vec::with_capacity(encoded.element_count);

    for (index, quantized_value) in quantized.into_iter().enumerate() {
        let group_index = index / encoded.group_size.max(1);
        let min = encoded.mins[group_index];
        let scale = encoded.scales[group_index];
        values.push(min + scale * quantized_value as f32);
    }

    Ok(values)
}

pub(crate) fn encode_full_precision_key_vector_f16(
    vector: &[f32],
    preset: TurboQuantPreset,
    head_index: usize,
    codec: TurboQuantKeyCodecConfig,
) -> Result<QuantizedKeyVector, TurboQuantCodecError> {
    let mut packed_indices = Vec::with_capacity(vector.len() * std::mem::size_of::<u16>());
    for value in vector {
        packed_indices.extend_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
    }
    Ok(QuantizedKeyVector {
        dim: vector.len(),
        bit_width: preset.key_bits(),
        l2_norm: 1.0,
        packed_indices,
        codec_version: codec.version,
        centroid_kind: codec.centroid_kind,
        rotation_seed: turboquant_rotation_seed(
            vector.len(),
            head_index,
            TurboQuantVectorRole::Key,
        ),
        head_index,
    })
}

pub(crate) fn decode_full_precision_key_vector_f16(
    encoded: &QuantizedKeyVector,
) -> Result<Vec<f32>, TurboQuantCodecError> {
    let required_bytes = encoded.dim.saturating_mul(std::mem::size_of::<u16>());
    if encoded.packed_indices.len() < required_bytes {
        return Err(TurboQuantCodecError::PackedBufferTooShort {
            bit_width: encoded.bit_width,
            index_count: encoded.dim,
        });
    }

    Ok(encoded
        .packed_indices
        .chunks_exact(std::mem::size_of::<u16>())
        .take(encoded.dim)
        .map(|bytes| f16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect())
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

pub fn randomized_hadamard_in_place(
    values: &mut [f32],
    seed: u64,
) -> Result<(), TurboQuantCodecError> {
    let signs = TurboQuantRotationSigns::new(values.len(), seed)?;
    signs.apply(values)?;
    hadamard_in_place(values)
}

pub fn inverse_randomized_hadamard_in_place(
    values: &mut [f32],
    seed: u64,
) -> Result<(), TurboQuantCodecError> {
    let signs = TurboQuantRotationSigns::new(values.len(), seed)?;
    hadamard_in_place(values)?;
    signs.apply(values)
}

pub fn turboquant_rotation_seed(dim: usize, head_index: usize, role: TurboQuantVectorRole) -> u64 {
    let role_offset = match role {
        TurboQuantVectorRole::Key => 1000u64,
        TurboQuantVectorRole::Value => 2000u64,
    };
    (dim as u64)
        .wrapping_mul(role_offset)
        .wrapping_add(head_index as u64)
}

pub(crate) fn validate_key_shape(values: &[f32]) -> Result<(), TurboQuantCodecError> {
    validate_key_shape_len(values.len())
}

pub(crate) fn validate_key_shape_len(len: usize) -> Result<(), TurboQuantCodecError> {
    if len == 0 {
        return Err(TurboQuantCodecError::EmptyVector);
    }
    if !len.is_power_of_two() {
        return Err(TurboQuantCodecError::NonPowerOfTwoDimension(len));
    }
    Ok(())
}

pub(crate) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00;
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let mut half_mantissa = (mantissa >> shift) as u16;
        if ((mantissa >> (shift - 1)) & 1) != 0 {
            half_mantissa = half_mantissa.saturating_add(1);
        }
        return sign | half_mantissa;
    }

    let mut half = sign | ((half_exponent as u16) << 10) | ((mantissa >> 13) as u16);
    if (mantissa & 0x0000_1000) != 0 {
        half = half.saturating_add(1);
    }
    half
}

pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = (bits & 0x03ff) as u32;

    let f32_bits = match exponent {
        0 => {
            if fraction == 0 {
                sign
            } else {
                let leading = fraction.leading_zeros() - 22;
                let mantissa = (fraction << (leading + 1)) & 0x03ff;
                let exponent = 127 - 15 - leading;
                sign | (exponent << 23) | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => {
            let exponent = (exponent as u32) + (127 - 15);
            sign | (exponent << 23) | (fraction << 13)
        }
    };

    f32::from_bits(f32_bits)
}

pub(crate) fn validate_layout_nonzero(
    name: &'static str,
    value: usize,
) -> Result<(), TurboQuantCodecError> {
    if value == 0 {
        return Err(TurboQuantCodecError::InvalidLayoutParameter { name, value });
    }
    Ok(())
}

/// Bytes per element of the full-precision serving KV cache used as the
/// savings baseline. Serving caches hold f16/bf16 K/V, so the honest baseline
/// is 2 bytes, not f32 — an f32 baseline overstates compression ~2x and makes
/// the `NoColdSavings` promotion gate unable to fire.
pub(crate) const FULL_PRECISION_KV_ELEMENT_BYTES: usize = 2;

pub(crate) fn full_precision_kv_bytes(
    token_count: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> usize {
    token_count
        .saturating_mul(n_kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(FULL_PRECISION_KV_ELEMENT_BYTES)
        .saturating_mul(2)
}

pub(crate) fn compression_ratio_milli(compressed_bytes: usize, full_precision_bytes: usize) -> u32 {
    if full_precision_bytes == 0 {
        0
    } else {
        compressed_bytes
            .saturating_mul(1000)
            .saturating_div(full_precision_bytes)
            .min(u32::MAX as usize) as u32
    }
}

pub(crate) fn f32_microunits(value: f32) -> u32 {
    if value <= 0.0 {
        0
    } else {
        (value * 1_000_000.0).round().min(u32::MAX as f32) as u32
    }
}

pub(crate) fn kib_ceil_usize(bytes: usize) -> usize {
    bytes / 1024 + usize::from(!bytes.is_multiple_of(1024))
}

pub(crate) fn checked_add_many(values: &[usize]) -> Result<usize, TurboQuantCodecError> {
    values.iter().try_fold(0usize, |acc, value| {
        acc.checked_add(*value)
            .ok_or(TurboQuantCodecError::LayoutSizeOverflow)
    })
}

pub(crate) fn checked_mul(left: usize, right: usize) -> Result<usize, TurboQuantCodecError> {
    left.checked_mul(right)
        .ok_or(TurboQuantCodecError::LayoutSizeOverflow)
}

pub(crate) fn align_up(value: usize, alignment: usize) -> Result<usize, TurboQuantCodecError> {
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

pub(crate) fn validate_centroid_count(
    bit_width: u32,
    centroids: &[f32],
) -> Result<(), TurboQuantCodecError> {
    let expected = match bit_width {
        3 | 4 | 7 | 8 => 1usize << bit_width,
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

pub(crate) fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

pub(crate) fn validate_supported_key_bits(bit_width: u32) -> Result<(), TurboQuantCodecError> {
    match bit_width {
        3 | 4 | 7 | 8 | 16 => Ok(()),
        other => Err(TurboQuantCodecError::UnsupportedKeyBits(other)),
    }
}

#[cfg(test)]
pub(crate) fn uniform_centroid_index(value: f32, levels: usize) -> u8 {
    let max_index = (levels - 1) as f32;
    ((value + 1.0) * 0.5 * max_index)
        .round()
        .clamp(0.0, max_index) as u8
}

pub(crate) fn vector_cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = l2_norm(left).max(f32::EPSILON);
    let right_norm = l2_norm(right).max(f32::EPSILON);
    dot / (left_norm * right_norm)
}
