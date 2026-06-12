use std::collections::VecDeque;

use ax_engine_core::TurboQuantPreset;
use thiserror::Error;

use crate::model::ModelConfig;

pub type FullPrecisionKvTokenVectors = (Vec<f32>, Vec<f32>);

pub const TURBOQUANT_SLOT_ALIGNMENT_BYTES: usize = 16;
pub const TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM: usize = 128;
pub const TURBOQUANT_EXTENDED_FUSED_DECODE_HEAD_DIM: usize = 256;
pub const TURBOQUANT_GEMMA4_FULL_ATTENTION_FUSED_DECODE_HEAD_DIM: usize = 512;
pub const TURBOQUANT_ROUTE_METADATA_SCHEMA_VERSION: u32 = 3;
pub const TURBOQUANT_CODEC_VERSION_UNIFORM_HADAMARD: u32 = 1;
pub const TURBOQUANT_CODEC_VERSION_RHT_LLOYD_MAX: u32 = 2;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TurboQuantProductionRequirements {
    pub fused_decode_kernel: bool,
    pub runtime_kv_storage: bool,
    pub runner_route_metadata: bool,
    pub long_context_benchmark_artifact: bool,
    pub public_switch_and_docs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurboQuantProductionBlocker {
    FusedDecodeKernel,
    RuntimeKvStorage,
    RunnerRouteMetadata,
    LongContextBenchmarkArtifact,
    PublicSwitchAndDocs,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TurboQuantProductionReadiness {
    pub ready: bool,
    pub blockers: Vec<TurboQuantProductionBlocker>,
}

impl TurboQuantProductionRequirements {
    pub const fn mlx_shadow_fused_kernel() -> Self {
        Self {
            fused_decode_kernel: true,
            runtime_kv_storage: true,
            runner_route_metadata: true,
            long_context_benchmark_artifact: false,
            public_switch_and_docs: true,
        }
    }

    pub fn evaluate(self) -> TurboQuantProductionReadiness {
        let mut blockers = Vec::new();
        if !self.fused_decode_kernel {
            blockers.push(TurboQuantProductionBlocker::FusedDecodeKernel);
        }
        if !self.runtime_kv_storage {
            blockers.push(TurboQuantProductionBlocker::RuntimeKvStorage);
        }
        if !self.runner_route_metadata {
            blockers.push(TurboQuantProductionBlocker::RunnerRouteMetadata);
        }
        if !self.long_context_benchmark_artifact {
            blockers.push(TurboQuantProductionBlocker::LongContextBenchmarkArtifact);
        }
        if !self.public_switch_and_docs {
            blockers.push(TurboQuantProductionBlocker::PublicSwitchAndDocs);
        }

        TurboQuantProductionReadiness {
            ready: blockers.is_empty(),
            blockers,
        }
    }
}

impl TurboQuantProductionReadiness {
    pub fn is_ready(&self) -> bool {
        self.ready
    }
}

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
pub enum TurboQuantDecodeQualityProfile {
    StrictDebug,
    ReferenceK8V4,
    ResearchLoose,
}

impl TurboQuantDecodeQualityProfile {
    pub const fn code(self) -> u32 {
        match self {
            Self::StrictDebug => 1,
            Self::ReferenceK8V4 => 2,
            Self::ResearchLoose => 3,
        }
    }

    pub const fn gate(self) -> TurboQuantDecodeQualityGate {
        match self {
            Self::StrictDebug => TurboQuantDecodeQualityGate::STRICT_DEBUG,
            Self::ReferenceK8V4 => TurboQuantDecodeQualityGate::REFERENCE_K8V4,
            Self::ResearchLoose => TurboQuantDecodeQualityGate::RESEARCH_LOOSE,
        }
    }

    pub const fn for_quantization_preset(preset: TurboQuantPreset) -> Self {
        match preset {
            TurboQuantPreset::K8V4 | TurboQuantPreset::K16V4 => Self::ReferenceK8V4,
            TurboQuantPreset::K4V4
            | TurboQuantPreset::K3V4Research
            | TurboQuantPreset::K8V3_5
            | TurboQuantPreset::K7V4 => Self::ResearchLoose,
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
    pub preset: TurboQuantPreset,
    pub profile: TurboQuantDecodeQualityProfile,
    pub gate: TurboQuantDecodeQualityGate,
    pub decision: TurboQuantDecodeQualityDecision,
}

pub fn evaluate_decode_quality_for_preset(
    preset: TurboQuantPreset,
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
        preset: TurboQuantPreset,
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

    fn invalidate_dequant_buffers(&mut self) {
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
fn append_attention_partition_with_hot_tail_from_f32_slices(
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

fn legacy_uniform_key_centroids(bit_width: u32) -> Result<Vec<f32>, TurboQuantCodecError> {
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

fn raw_lloyd_max_gaussian_centroids(bit_width: u32) -> Result<Vec<f32>, TurboQuantCodecError> {
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

fn fractional_value_split(element_count: usize) -> usize {
    element_count / 2
}

fn value_bit_width_for_index(value_bits_x2: u32, element_count: usize, index: usize) -> u32 {
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

fn pack_value_indices(indices: &[u8], value_bits_x2: u32) -> Result<Vec<u8>, TurboQuantCodecError> {
    if value_bits_x2 == 7 {
        let split = fractional_value_split(indices.len());
        let mut packed = pack_indices(&indices[..split], 4)?;
        packed.extend(pack_indices(&indices[split..], 3)?);
        return Ok(packed);
    }
    pack_indices(indices, value_bits_x2 / 2)
}

fn unpack_value_indices(
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

fn encode_full_precision_key_vector_f16(
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

fn decode_full_precision_key_vector_f16(
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

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

fn f32_to_f16_bits(value: f32) -> u16 {
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

fn f16_bits_to_f32(bits: u16) -> f32 {
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

fn validate_layout_nonzero(name: &'static str, value: usize) -> Result<(), TurboQuantCodecError> {
    if value == 0 {
        return Err(TurboQuantCodecError::InvalidLayoutParameter { name, value });
    }
    Ok(())
}

/// Bytes per element of the full-precision serving KV cache used as the
/// savings baseline. Serving caches hold f16/bf16 K/V, so the honest baseline
/// is 2 bytes, not f32 — an f32 baseline overstates compression ~2x and makes
/// the `NoColdSavings` promotion gate unable to fire.
const FULL_PRECISION_KV_ELEMENT_BYTES: usize = 2;

fn full_precision_kv_bytes(token_count: usize, n_kv_heads: usize, head_dim: usize) -> usize {
    token_count
        .saturating_mul(n_kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(FULL_PRECISION_KV_ELEMENT_BYTES)
        .saturating_mul(2)
}

fn compression_ratio_milli(compressed_bytes: usize, full_precision_bytes: usize) -> u32 {
    if full_precision_bytes == 0 {
        0
    } else {
        compressed_bytes
            .saturating_mul(1000)
            .saturating_div(full_precision_bytes)
            .min(u32::MAX as usize) as u32
    }
}

fn f32_microunits(value: f32) -> u32 {
    if value <= 0.0 {
        0
    } else {
        (value * 1_000_000.0).round().min(u32::MAX as f32) as u32
    }
}

fn kib_ceil_usize(bytes: usize) -> usize {
    bytes / 1024 + usize::from(!bytes.is_multiple_of(1024))
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

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn validate_supported_key_bits(bit_width: u32) -> Result<(), TurboQuantCodecError> {
    match bit_width {
        3 | 4 | 7 | 8 | 16 => Ok(()),
        other => Err(TurboQuantCodecError::UnsupportedKeyBits(other)),
    }
}

#[cfg(test)]
fn uniform_centroid_index(value: f32, levels: usize) -> u8 {
    let max_index = (levels - 1) as f32;
    ((value + 1.0) * 0.5 * max_index)
        .round()
        .clamp(0.0, max_index) as u8
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
            model_family: "qwen3".to_string(),
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
            global_sliding_window: None,
            gemma4_moe_router: false,
            uses_geglu: false,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            linear_attention: None,
            mla_attention: None,
            glm_router: None,
            rms_norm_eps: 1e-6,
            rope_freqs: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: 8192.0,
            attn_temperature_scale: 0.1,
            intermediate_size_mlp: 0,
            moe_layer_freq: 1,
            moe_first_dense_layers: 0,
            moe_shared_expert_count: 0,
            moe_sigmoid_routing: false,
            moe_routed_scaling_factor: 1.0,
            moe_n_group: 1,
            moe_topk_group: 1,
            think_start_token_id: None,
            think_end_token_id: None,
        }
    }

    fn test_layer(head_dim: usize) -> LayerConfig {
        LayerConfig {
            head_dim,
            rope_theta: 10_000.0,
            rope_dims: head_dim,
            rope_freqs: None,
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
            turboquant_support_report(&cfg, TurboQuantPreset::K8V4).expect("support report");

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
    fn production_readiness_fails_closed_until_every_runtime_gate_is_present() {
        let readiness = TurboQuantProductionRequirements::default().evaluate();

        assert!(!readiness.is_ready());
        assert_eq!(
            readiness.blockers,
            vec![
                TurboQuantProductionBlocker::FusedDecodeKernel,
                TurboQuantProductionBlocker::RuntimeKvStorage,
                TurboQuantProductionBlocker::RunnerRouteMetadata,
                TurboQuantProductionBlocker::LongContextBenchmarkArtifact,
                TurboQuantProductionBlocker::PublicSwitchAndDocs,
            ]
        );

        let readiness = TurboQuantProductionRequirements {
            fused_decode_kernel: true,
            runtime_kv_storage: true,
            runner_route_metadata: true,
            long_context_benchmark_artifact: true,
            public_switch_and_docs: false,
        }
        .evaluate();

        assert!(!readiness.is_ready());
        assert_eq!(
            readiness.blockers,
            vec![TurboQuantProductionBlocker::PublicSwitchAndDocs]
        );
    }

    #[test]
    fn production_readiness_marks_runtime_docs_and_metadata_gates_present() {
        let readiness = TurboQuantProductionRequirements::mlx_shadow_fused_kernel().evaluate();

        assert!(!readiness.is_ready());
        assert_eq!(
            readiness.blockers,
            vec![TurboQuantProductionBlocker::LongContextBenchmarkArtifact]
        );
    }

    #[test]
    fn production_readiness_passes_only_when_all_runtime_gates_are_present() {
        let readiness = TurboQuantProductionRequirements {
            fused_decode_kernel: true,
            runtime_kv_storage: true,
            runner_route_metadata: true,
            long_context_benchmark_artifact: true,
            public_switch_and_docs: true,
        }
        .evaluate();

        assert!(readiness.is_ready());
        assert!(readiness.blockers.is_empty());
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
            turboquant_support_report(&cfg, TurboQuantPreset::K8V4).expect("support report");

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
        cfg.linear_attention = Some({
            let (q_scale, k_scale) = crate::linear_attention_ops::linear_attention_qk_scale(64);
            LinearAttentionConfig {
                full_attention_interval: 4,
                num_value_heads: 2,
                num_key_heads: 2,
                key_head_dim: 64,
                value_head_dim: 64,
                conv_kernel_dim: 4,
                q_scale,
                k_scale,
            }
        });

        let report =
            turboquant_support_report(&cfg, TurboQuantPreset::K4V4).expect("support report");

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
            preset: TurboQuantPreset::K8V4,
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
            preset: TurboQuantPreset::K3V4Research,
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
            preset: TurboQuantPreset::K8V4,
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
            preset: TurboQuantPreset::K8V4,
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
            preset: TurboQuantPreset::K8V4,
            block_tokens: 16,
            n_kv_heads: 2,
            head_dim: 12,
            value_group_size: 4,
        })
        .expect_err("non power-of-two head dim should fail");
        assert_eq!(error, TurboQuantCodecError::NonPowerOfTwoDimension(12));

        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
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
            preset: TurboQuantPreset::K8V4,
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
            preset: TurboQuantPreset::K4V4,
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
            preset: TurboQuantPreset::K8V4,
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
    fn compressed_decode_plan_validates_required_cold_slots() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 3, 1).expect("plan should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let token = vec![(vec![0.1; 8], vec![0.2; 8]), (vec![0.3; 8], vec![0.4; 8])];

        let error = plan
            .validate_compressed_buffer(&buffer)
            .expect_err("empty compressed buffer must fail coverage");
        assert_eq!(
            error,
            TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens: 2,
                required_slots: 4,
                written_slots: 0,
            }
        );

        buffer.write_token(0, &token).expect("first cold token");
        let error = plan
            .validate_compressed_buffer(&buffer)
            .expect_err("partial compressed buffer must fail coverage");
        assert_eq!(
            error,
            TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens: 2,
                required_slots: 4,
                written_slots: 2,
            }
        );

        buffer.write_token(1, &token).expect("second cold token");
        plan.validate_compressed_buffer(&buffer)
            .expect("complete cold coverage should pass");
    }

    #[test]
    fn compressed_decode_plan_rejects_mismatched_buffer_layout() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let other_layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K4V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("other layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 3, 1).expect("plan should build");
        let buffer = TurboQuantCompressedBlockBuffer::new(other_layout);

        let error = plan
            .validate_compressed_buffer(&buffer)
            .expect_err("layout mismatch must fail closed");

        assert_eq!(
            error,
            TurboQuantCodecError::CompressedDecodePlanLayoutMismatch
        );
    }

    #[test]
    fn compressed_decode_plan_skips_buffer_validation_for_hot_window_only_path() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let other_layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K4V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("other layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 1, 2).expect("plan should build");
        let buffer = TurboQuantCompressedBlockBuffer::new(other_layout);

        assert!(!plan.needs_compressed_decode());
        plan.validate_compressed_buffer(&buffer)
            .expect("hot-window-only plan does not require compressed coverage");
    }

    #[test]
    fn compressed_decode_plan_validates_query_shapes() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 1, 2).expect("plan should build");

        plan.validate_queries(&[vec![0.1; 8], vec![0.2; 8]])
            .expect("matching per-head queries should pass");
        plan.validate_queries(&[vec![0.1; 8], vec![0.2; 8], vec![0.3; 8], vec![0.4; 8]])
            .expect("GQA query heads should pass when divisible by KV heads");

        let error = plan
            .validate_queries(&[vec![0.1; 8]])
            .expect_err("query head count mismatch should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedKvHeadCount {
                expected: 2,
                actual: 1,
            }
        );

        let error = plan
            .validate_queries(&[vec![0.1; 8], vec![0.2; 7]])
            .expect_err("query dimension mismatch should fail");
        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedVectorDimension {
                expected: 8,
                actual: 7,
            }
        );
    }

    #[test]
    fn compressed_decode_plan_validates_buffer_and_queries_together() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 3, 1).expect("plan should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let token = vec![(vec![0.1; 8], vec![0.2; 8]), (vec![0.3; 8], vec![0.4; 8])];
        buffer.write_token(0, &token).expect("first cold token");
        buffer.write_token(1, &token).expect("second cold token");

        plan.validate_decode_inputs(&buffer, &[vec![0.1; 8], vec![0.2; 8]])
            .expect("complete compressed coverage and matching queries should pass");

        let error = plan
            .validate_decode_inputs(&buffer, &[vec![0.1; 8]])
            .expect_err("query validation should happen before coverage");
        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedKvHeadCount {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn compressed_decode_plan_reports_decode_readiness() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 3, 1).expect("plan should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let token = vec![(vec![0.1; 8], vec![0.2; 8]), (vec![0.3; 8], vec![0.4; 8])];
        buffer.write_token(0, &token).expect("first cold token");
        buffer.write_token(1, &token).expect("second cold token");

        let readiness = plan
            .decode_readiness(&buffer, &[vec![0.1; 8], vec![0.2; 8]])
            .expect("complete compressed inputs should report readiness");

        assert_eq!(
            readiness,
            TurboQuantCompressedDecodeReadiness {
                preset: TurboQuantPreset::K8V4,
                key_bits: 8,
                value_bits: 4,
                decode_path: TurboQuantCompressedDecodePath::CompressedColdWithHotWindow,
                total_tokens: 3,
                cold_tokens: 2,
                hot_tokens: 1,
                query_heads: 2,
                query_head_dim: 8,
                compressed_blocks: 1,
                compressed_buffer_bytes: layout.block_bytes,
                required_compressed_slots: 4,
                written_compressed_slots: 4,
                quality_profile: TurboQuantDecodeQualityProfile::ReferenceK8V4,
            }
        );
    }

    #[test]
    fn compressed_decode_readiness_marks_initial_fused_decode_candidate() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 1,
            head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
            value_group_size: 32,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 2, 1).expect("plan should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let token = vec![(
            vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
            vec![0.2; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
        )];
        buffer.write_token(0, &token).expect("cold token");

        let readiness = plan
            .decode_readiness(
                &buffer,
                &[vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
            )
            .expect("complete compressed inputs should report readiness");
        let candidate = readiness.fused_decode_candidate();

        assert!(candidate.is_candidate());
        assert_eq!(
            candidate,
            TurboQuantFusedDecodeCandidate {
                status: TurboQuantFusedDecodeCandidateStatus::Candidate,
                preset: TurboQuantPreset::K8V4,
                head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
                compressed_blocks: 1,
                compressed_buffer_bytes: layout.block_bytes,
                required_compressed_slots: 1,
            }
        );
    }

    #[test]
    fn compressed_decode_plan_builds_fused_decode_launch_descriptor() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 1,
            head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
            value_group_size: 32,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 2, 1).expect("plan should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        buffer
            .write_token(
                0,
                &[(
                    vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
                    vec![0.2; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
                )],
            )
            .expect("cold token");

        let descriptor = plan
            .fused_decode_launch_descriptor(
                &buffer,
                &[vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
            )
            .expect("candidate launch should produce descriptor");

        assert_eq!(
            descriptor,
            TurboQuantFusedDecodeLaunchDescriptor {
                preset: TurboQuantPreset::K8V4,
                key_bits: 8,
                value_bits: 4,
                total_tokens: 2,
                cold_tokens: 1,
                hot_tokens: 1,
                n_query_heads: 1,
                n_kv_heads: 1,
                head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
                block_tokens: 2,
                compressed_blocks: 1,
                compressed_buffer_bytes: layout.block_bytes,
                required_compressed_slots: 1,
                value_group_size: 32,
                value_group_count: 4,
                key_payload_bytes_per_head: 128,
                key_norm_bytes_per_head: 4,
                value_payload_bytes_per_head: 64,
                value_metadata_bytes_per_head: 32,
                raw_slot_bytes_per_head: 228,
                slot_bytes_per_head: 240,
                token_stride_bytes: 240,
                block_bytes: 480,
                key_payload_offset_in_slot: 0,
                key_norm_offset_in_slot: 128,
                value_payload_offset_in_slot: 132,
                value_mins_offset_in_slot: 196,
                value_scales_offset_in_slot: 212,
            }
        );
    }

    #[test]
    fn fused_decode_launch_descriptor_reports_workload_estimate() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 1,
            head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
            value_group_size: 32,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 2, 1).expect("plan should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        buffer
            .write_token(
                0,
                &[(
                    vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
                    vec![0.2; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
                )],
            )
            .expect("cold token");

        let workload = plan
            .fused_decode_launch_descriptor(
                &buffer,
                &[vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
            )
            .expect("candidate launch should produce descriptor")
            .workload();

        assert_eq!(
            workload,
            TurboQuantFusedDecodeLaunchWorkload {
                cold_score_elements: 1,
                hot_score_elements: 1,
                output_elements: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
                full_precision_cold_kv_bytes: 512,
                full_precision_total_kv_bytes: 1024,
                compressed_key_payload_bytes: 128,
                compressed_key_norm_bytes: 4,
                compressed_value_payload_bytes: 64,
                compressed_value_metadata_bytes: 32,
                compressed_raw_slot_bytes: 228,
                compressed_aligned_slot_bytes: 240,
                hot_full_precision_kv_bytes: 512,
                estimated_total_read_bytes: 740,
                estimated_cold_saved_bytes: 284,
                estimated_total_saved_read_bytes: 284,
                cold_compression_ratio_milli: 445,
            }
        );
    }

    #[test]
    fn fused_decode_launch_descriptor_reports_benchmark_estimate() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 1,
            head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
            value_group_size: 32,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 2, 1).expect("plan should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        buffer
            .write_token(
                0,
                &[(
                    vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
                    vec![0.2; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
                )],
            )
            .expect("cold token");

        let estimate = plan
            .fused_decode_launch_descriptor(
                &buffer,
                &[vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
            )
            .expect("candidate launch should produce descriptor")
            .benchmark_estimate();

        assert_eq!(
            estimate,
            TurboQuantFusedDecodeBenchmarkEstimate {
                preset: TurboQuantPreset::K8V4,
                key_bits: 8,
                value_bits: 4,
                total_tokens: 2,
                cold_tokens: 1,
                hot_tokens: 1,
                n_query_heads: 1,
                n_kv_heads: 1,
                head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
                compressed_blocks: 1,
                cold_score_elements: 1,
                hot_score_elements: 1,
                output_elements: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
                compressed_buffer_kib: 1,
                full_precision_cold_kv_kib: 1,
                full_precision_total_kv_kib: 1,
                estimated_compressed_cold_kv_kib: 1,
                hot_full_precision_kv_kib: 1,
                estimated_total_read_kib: 1,
                estimated_cold_saved_kib: 1,
                estimated_total_saved_read_kib: 1,
                cold_compression_ratio_milli: 445,
            }
        );
    }

    #[test]
    fn fused_decode_launch_descriptor_reports_promotion_readiness() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 1,
            head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
            value_group_size: 32,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 2, 1).expect("plan should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        buffer
            .write_token(
                0,
                &[(
                    vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
                    vec![0.2; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
                )],
            )
            .expect("cold token");
        let descriptor = plan
            .fused_decode_launch_descriptor(
                &buffer,
                &[vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
            )
            .expect("candidate launch should produce descriptor");
        let comparison = compare_decode_outputs(
            &[vec![1.0; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
            &[vec![1.0; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
        )
        .expect("comparison should pass");
        let quality_check =
            TurboQuantDecodeQualityCheck::for_preset(TurboQuantPreset::K8V4, comparison);

        let readiness = descriptor.promotion_readiness(&quality_check);

        assert!(readiness.is_ready());
        assert_eq!(
            readiness.status,
            TurboQuantFusedDecodePromotionStatus::Ready
        );
        assert_eq!(readiness.preset, TurboQuantPreset::K8V4);
        assert_eq!(
            readiness.quality_profile,
            TurboQuantDecodeQualityProfile::ReferenceK8V4
        );
        assert_eq!(readiness.benchmark_estimate.estimated_cold_saved_kib, 1);
        assert_eq!(
            readiness.benchmark_estimate.cold_compression_ratio_milli,
            445
        );
    }

    #[test]
    fn fused_decode_launch_descriptor_reports_promotion_evidence() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 1,
            head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
            value_group_size: 32,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 2, 1).expect("plan should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        buffer
            .write_token(
                0,
                &[(
                    vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
                    vec![0.2; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
                )],
            )
            .expect("cold token");
        let descriptor = plan
            .fused_decode_launch_descriptor(
                &buffer,
                &[vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
            )
            .expect("candidate launch should produce descriptor");
        let comparison = compare_decode_outputs(
            &[vec![1.0; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
            &[vec![1.0; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
        )
        .expect("comparison should pass");
        let quality_check =
            TurboQuantDecodeQualityCheck::for_preset(TurboQuantPreset::K8V4, comparison);

        let evidence = descriptor.promotion_evidence(&quality_check);

        assert!(evidence.readiness.is_ready());
        assert!(evidence.quality_passed);
        assert_eq!(evidence.max_abs_diff, 0.0);
        assert_eq!(evidence.max_abs_diff_limit, 0.04);
        assert_eq!(evidence.mean_abs_diff, 0.0);
        assert_eq!(evidence.mean_abs_diff_limit, 0.02);
        assert_eq!(evidence.min_cosine_similarity, 1.0);
        assert_eq!(evidence.min_cosine_similarity_limit, 0.998);
        assert_eq!(
            evidence
                .readiness
                .benchmark_estimate
                .estimated_cold_saved_kib,
            1
        );
    }

    #[test]
    fn fused_decode_promotion_evidence_reports_artifact_summary() {
        let descriptor = TurboQuantFusedDecodeLaunchDescriptor {
            preset: TurboQuantPreset::K8V4,
            key_bits: 8,
            value_bits: 4,
            total_tokens: 2,
            cold_tokens: 1,
            hot_tokens: 1,
            n_query_heads: 1,
            n_kv_heads: 1,
            head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
            block_tokens: 2,
            compressed_blocks: 1,
            compressed_buffer_bytes: 480,
            required_compressed_slots: 1,
            value_group_size: 32,
            value_group_count: 4,
            key_payload_bytes_per_head: 128,
            key_norm_bytes_per_head: 4,
            value_payload_bytes_per_head: 64,
            value_metadata_bytes_per_head: 32,
            raw_slot_bytes_per_head: 228,
            slot_bytes_per_head: 240,
            token_stride_bytes: 240,
            block_bytes: 480,
            key_payload_offset_in_slot: 0,
            key_norm_offset_in_slot: 128,
            value_payload_offset_in_slot: 132,
            value_mins_offset_in_slot: 196,
            value_scales_offset_in_slot: 212,
        };
        let comparison = compare_decode_outputs(
            &[vec![1.0; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
            &[vec![1.0; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
        )
        .expect("comparison should pass");
        let quality_check =
            TurboQuantDecodeQualityCheck::for_preset(TurboQuantPreset::K8V4, comparison);

        let summary = descriptor.promotion_evidence(&quality_check).summary();

        assert_eq!(
            summary,
            TurboQuantFusedDecodePromotionEvidenceSummary {
                status_code: 1,
                ready: true,
                preset_code: 1,
                quality_profile_code: 2,
                quality_passed: true,
                max_abs_diff_microunits: 0,
                max_abs_diff_limit_microunits: 40_000,
                mean_abs_diff_microunits: 0,
                mean_abs_diff_limit_microunits: 20_000,
                min_cosine_similarity_microunits: 1_000_000,
                min_cosine_similarity_limit_microunits: 998_000,
                estimated_cold_saved_kib: 1,
                estimated_total_saved_read_kib: 1,
                cold_compression_ratio_milli: 445,
            }
        );
    }

    #[test]
    fn fused_decode_promotion_readiness_fails_closed() {
        let descriptor = TurboQuantFusedDecodeLaunchDescriptor {
            preset: TurboQuantPreset::K8V4,
            key_bits: 8,
            value_bits: 4,
            total_tokens: 2,
            cold_tokens: 1,
            hot_tokens: 1,
            n_query_heads: 1,
            n_kv_heads: 1,
            head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
            block_tokens: 2,
            compressed_blocks: 1,
            compressed_buffer_bytes: 480,
            required_compressed_slots: 1,
            value_group_size: 32,
            value_group_count: 4,
            key_payload_bytes_per_head: 128,
            key_norm_bytes_per_head: 4,
            value_payload_bytes_per_head: 64,
            value_metadata_bytes_per_head: 32,
            raw_slot_bytes_per_head: 228,
            slot_bytes_per_head: 240,
            token_stride_bytes: 240,
            block_bytes: 480,
            key_payload_offset_in_slot: 0,
            key_norm_offset_in_slot: 128,
            value_payload_offset_in_slot: 132,
            value_mins_offset_in_slot: 196,
            value_scales_offset_in_slot: 212,
        };
        let passing_comparison = compare_decode_outputs(
            &[vec![1.0; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
            &[vec![1.0; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
        )
        .expect("comparison should pass");
        let mismatched_quality_check = TurboQuantDecodeQualityCheck::for_preset(
            TurboQuantPreset::K4V4,
            passing_comparison.clone(),
        );
        let failing_comparison = compare_decode_outputs(
            &[vec![1.0; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
            &[vec![0.0; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
        )
        .expect("comparison should report failure metrics");
        let failing_quality_check =
            TurboQuantDecodeQualityCheck::for_preset(TurboQuantPreset::K8V4, failing_comparison);
        let passing_quality_check =
            TurboQuantDecodeQualityCheck::for_preset(TurboQuantPreset::K8V4, passing_comparison);

        assert_eq!(
            descriptor
                .promotion_readiness(&mismatched_quality_check)
                .status,
            TurboQuantFusedDecodePromotionStatus::QualityPresetMismatch
        );
        let failing_readiness = descriptor.promotion_readiness(&failing_quality_check);
        assert_eq!(
            failing_readiness.status,
            TurboQuantFusedDecodePromotionStatus::QualityGateFailed
        );
        assert_eq!(
            failing_readiness.fallback_preset(),
            Some(TurboQuantPreset::K16V4)
        );
        assert_eq!(
            failing_readiness.effective_preset(),
            TurboQuantPreset::K16V4
        );

        let no_savings_descriptor = TurboQuantFusedDecodeLaunchDescriptor {
            raw_slot_bytes_per_head: 2048,
            ..descriptor
        };
        assert_eq!(
            no_savings_descriptor
                .promotion_readiness(&passing_quality_check)
                .status,
            TurboQuantFusedDecodePromotionStatus::NoColdSavings
        );
    }

    #[test]
    fn compressed_decode_launch_descriptor_rejects_non_candidates() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 1,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 2, 1).expect("plan should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        buffer
            .write_token(0, &[(vec![0.1; 8], vec![0.2; 8])])
            .expect("cold token");

        let error = plan
            .fused_decode_launch_descriptor(&buffer, &[vec![0.1; 8]])
            .expect_err("unsupported head dim should reject fused launch");

        assert_eq!(
            error,
            TurboQuantCodecError::FusedDecodeLaunchRejected {
                status: TurboQuantFusedDecodeCandidateStatus::UnsupportedHeadDim,
            }
        );
    }

    #[test]
    fn compressed_decode_readiness_fails_closed_for_non_candidate_launches() {
        let unsupported_head_layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 1,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("unsupported head layout should build for reference coverage");
        let unsupported_head_plan =
            TurboQuantCompressedDecodePlan::new(unsupported_head_layout, 2, 1)
                .expect("plan should build");
        let mut unsupported_head_buffer =
            TurboQuantCompressedBlockBuffer::new(unsupported_head_layout);
        unsupported_head_buffer
            .write_token(0, &[(vec![0.1; 8], vec![0.2; 8])])
            .expect("cold token");
        let unsupported_head = unsupported_head_plan
            .decode_readiness(&unsupported_head_buffer, &[vec![0.1; 8]])
            .expect("readiness should still report")
            .fused_decode_candidate();
        assert_eq!(
            unsupported_head.status,
            TurboQuantFusedDecodeCandidateStatus::UnsupportedHeadDim
        );
        assert!(!unsupported_head.is_candidate());

        let unsupported_preset_layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K4V4,
            block_tokens: 2,
            n_kv_heads: 1,
            head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
            value_group_size: 32,
        })
        .expect("unsupported preset layout should build");
        let unsupported_preset_plan =
            TurboQuantCompressedDecodePlan::new(unsupported_preset_layout, 2, 1)
                .expect("plan should build");
        let mut unsupported_preset_buffer =
            TurboQuantCompressedBlockBuffer::new(unsupported_preset_layout);
        unsupported_preset_buffer
            .write_token(
                0,
                &[(
                    vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
                    vec![0.2; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM],
                )],
            )
            .expect("cold token");
        let unsupported_preset = unsupported_preset_plan
            .decode_readiness(
                &unsupported_preset_buffer,
                &[vec![0.1; TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM]],
            )
            .expect("readiness should still report")
            .fused_decode_candidate();
        assert_eq!(
            unsupported_preset.status,
            TurboQuantFusedDecodeCandidateStatus::UnsupportedPreset
        );
        assert!(!unsupported_preset.is_candidate());
    }

    #[test]
    fn compressed_decode_readiness_skips_buffer_layout_for_hot_window_only_path() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K4V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let other_layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("other layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 1, 4).expect("plan should build");
        let buffer = TurboQuantCompressedBlockBuffer::new(other_layout);

        let readiness = plan
            .decode_readiness(&buffer, &[vec![0.1; 8], vec![0.2; 8]])
            .expect("hot-window-only readiness should not require compressed storage");

        assert_eq!(
            readiness.decode_path,
            TurboQuantCompressedDecodePath::FullPrecisionOnly
        );
        assert_eq!(readiness.required_compressed_slots, 0);
        assert_eq!(readiness.written_compressed_slots, 0);
        assert_eq!(
            readiness.quality_profile,
            TurboQuantDecodeQualityProfile::ResearchLoose
        );
        assert_eq!(
            readiness.fused_decode_candidate().status,
            TurboQuantFusedDecodeCandidateStatus::FullPrecisionOnly
        );
    }

    #[test]
    fn compressed_block_buffer_writes_and_reads_slot_layout() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
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
            preset: TurboQuantPreset::K8V4,
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
            preset: TurboQuantPreset::K8V4,
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
            preset: TurboQuantPreset::K8V4,
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
            preset: TurboQuantPreset::K8V4,
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
            preset: TurboQuantPreset::K8V4,
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
            preset: TurboQuantPreset::K8V4,
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
    fn compressed_block_buffer_partition_stats_merge_with_hot_tail() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let cold_tokens = (0..3)
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
        for (token_index, heads) in cold_tokens.iter().enumerate() {
            buffer
                .write_token(token_index, heads)
                .expect("token write should work");
        }

        let hot_tokens = [
            (
                vec![0.3, -0.2, 0.1, 0.4, -0.1, 0.2, 0.7, -0.5],
                vec![0.1, -0.2, 0.4, 0.0, 0.2, -0.1, 0.3, 0.5],
            ),
            (
                vec![-0.1, 0.2, 0.4, -0.3, 0.6, 0.1, -0.2, 0.5],
                vec![0.0, 0.1, -0.1, 0.3, 0.5, -0.2, 0.2, 0.4],
            ),
        ];
        let query = vec![0.3, -0.1, 0.2, 0.6, -0.4, 0.1, 0.5, -0.2];
        let cold_head = buffer
            .debug_reconstruct_head_history(1, cold_tokens.len())
            .expect("cold head should reconstruct");
        let hot_head = hot_tokens.to_vec();
        let mut full_history = cold_head.clone();
        full_history.extend(hot_head.clone());

        let expected =
            reference_decode_attention(&query, &full_history).expect("full split attention");
        let cold_stats = buffer
            .debug_decode_partition_stats_for_head(&query, 1, cold_tokens.len())
            .expect("cold partition stats");
        let hot_stats = reference_decode_attention_partition_stats(&query, &hot_head)
            .expect("hot partition stats");
        let actual = merge_attention_partition_stats(&[cold_stats.clone(), hot_stats])
            .expect("merged attention");

        assert!(
            max_abs_diff(&expected, &actual) < 1e-6,
            "expected {expected:?}, got {actual:?}"
        );

        let cold_batch = TurboQuantAttentionPartitionStatsBatch {
            token_count: cold_stats.token_count,
            value_dim: cold_stats.value_dim,
            max_scores: vec![cold_stats.max_score],
            exp_sums: vec![cold_stats.exp_sum],
            weighted_value_sums: cold_stats.weighted_value_sum.clone(),
        };
        let mut hot_k_values = Vec::with_capacity(hot_head.len() * 8);
        let mut hot_v_values = Vec::with_capacity(hot_head.len() * 8);
        for (key, value) in &hot_head {
            hot_k_values.extend(key);
            hot_v_values.extend(value);
        }
        let mut flat_actual = Vec::new();
        append_attention_partition_stats_batch_head_with_hot_tail_from_f32_slices(
            &mut flat_actual,
            cold_batch.validated().expect("batch should validate"),
            0,
            &query,
            &hot_k_values,
            &hot_v_values,
            1,
            8,
            hot_head.len(),
            0,
        )
        .expect("flat hot-tail merge should work");
        assert!(
            max_abs_diff(&expected, &flat_actual) < 1e-6,
            "expected {expected:?}, got {flat_actual:?}"
        );
    }

    #[test]
    fn partition_stats_batch_validated_rejects_malformed_weighted_sum() {
        let batch = TurboQuantAttentionPartitionStatsBatch {
            token_count: 1,
            value_dim: 4,
            max_scores: vec![0.0],
            exp_sums: vec![1.0],
            weighted_value_sums: vec![1.0, 2.0, 3.0],
        };

        let error = batch
            .validated()
            .expect_err("malformed batch should fail validation");
        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedVectorDimension {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn compressed_block_buffer_partition_stats_validate_query_heads() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let buffer = TurboQuantCompressedBlockBuffer::new(layout);

        let error = buffer
            .debug_decode_partition_stats_for_all_heads(&[vec![0.0; 8]], 1)
            .expect_err("query heads should fail closed");
        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedKvHeadCount {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn compressed_block_buffer_head_decode_fails_closed_for_invalid_history() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
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
            preset: TurboQuantPreset::K8V4,
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
    fn compressed_block_buffer_decodes_attention_for_grouped_query_heads() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 2,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        for token_index in 0..2 {
            let heads = vec![
                (
                    (0..8)
                        .map(|idx| token_index as f32 * 0.03 + idx as f32 * 0.02 - 0.2)
                        .collect::<Vec<_>>(),
                    (0..8)
                        .map(|idx| token_index as f32 * 0.04 - idx as f32 * 0.01)
                        .collect::<Vec<_>>(),
                ),
                (
                    (0..8)
                        .map(|idx| token_index as f32 * -0.02 + idx as f32 * 0.03 + 0.1)
                        .collect::<Vec<_>>(),
                    (0..8)
                        .map(|idx| token_index as f32 * 0.01 + idx as f32 * 0.02 - 0.1)
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
            vec![0.0, 0.2, 0.4, -0.1, 0.1, -0.3, 0.6, 0.2],
            vec![-0.2, 0.3, 0.1, 0.5, 0.0, 0.4, -0.1, 0.2],
        ];

        let actual = buffer
            .debug_decode_attention_for_all_heads(&queries, 2)
            .expect("GQA decode should work");
        let expected = queries
            .iter()
            .enumerate()
            .map(|(query_head_index, query)| {
                let kv_head_index = query_head_index / 2;
                buffer
                    .debug_decode_attention_for_head(query, kv_head_index, 2)
                    .expect("single-head decode should work")
            })
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 4);
    }

    #[test]
    fn compressed_block_buffer_all_heads_decode_fails_closed_for_query_count() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
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
            TurboQuantDecodeQualityProfile::for_quantization_preset(TurboQuantPreset::K8V4),
            TurboQuantDecodeQualityProfile::ReferenceK8V4
        );
        assert_eq!(
            TurboQuantDecodeQualityProfile::for_quantization_preset(TurboQuantPreset::K4V4),
            TurboQuantDecodeQualityProfile::ResearchLoose
        );
        assert_eq!(
            TurboQuantDecodeQualityProfile::for_quantization_preset(TurboQuantPreset::K3V4Research),
            TurboQuantDecodeQualityProfile::ResearchLoose
        );
    }

    #[test]
    fn decode_quality_evaluation_records_preset_profile_gate_and_decision() {
        let report =
            compare_decode_outputs(&[vec![1.0, 0.0, 0.0, 0.0]], &[vec![0.97, 0.0, 0.0, 0.0]])
                .expect("compare outputs");

        let evaluation = evaluate_decode_quality_for_preset(TurboQuantPreset::K8V4, &report);

        assert_eq!(evaluation.preset, TurboQuantPreset::K8V4);
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

        let reference = evaluate_decode_quality_for_preset(TurboQuantPreset::K8V4, &report);
        let research = evaluate_decode_quality_for_preset(TurboQuantPreset::K4V4, &report);

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

        let check = TurboQuantDecodeQualityCheck::for_preset(TurboQuantPreset::K8V4, comparison);

        assert_eq!(check.evaluation.preset, TurboQuantPreset::K8V4);
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
            preset: TurboQuantPreset::K8V4,
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
        assert_eq!(check.evaluation.preset, TurboQuantPreset::K8V4);
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
        assert_eq!(
            packed_value_bytes_for_preset(8, TurboQuantPreset::K8V3_5).expect("K8V3.5 value bytes"),
            4
        );
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
    fn randomized_hadamard_is_self_inverse_for_stable_seed() {
        let original = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 0.0];
        let seed = turboquant_rotation_seed(original.len(), 3, TurboQuantVectorRole::Key);
        let mut transformed = original.clone();
        randomized_hadamard_in_place(&mut transformed, seed).expect("forward RHT");
        inverse_randomized_hadamard_in_place(&mut transformed, seed).expect("inverse RHT");

        for (actual, expected) in transformed.iter().zip(original) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn lloyd_max_key_centroids_are_scaled_to_head_dim() {
        let centroids = key_centroids_for_dim(4, 128).expect("4-bit Lloyd-Max centroids");
        let scale = (128.0f32).sqrt().recip();
        assert!((centroids[0] - (-2.7326 * scale)).abs() < 1e-6);
        assert!((centroids[15] - (2.7326 * scale)).abs() < 1e-6);

        let k8 = key_centroids_for_dim(8, 128).expect("8-bit centroids");
        assert_eq!(k8[0], -1.0);
        assert_eq!(k8[255], 1.0);
    }

    #[test]
    fn key_codec_preserves_direction_for_fixed_vector() {
        let vector = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 0.0];
        let encoded =
            encode_key_vector(&vector, TurboQuantPreset::K8V4).expect("encode key vector");
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
    fn k16v4_key_codec_uses_fp16_payload_without_rotation_error() {
        let vector = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 0.0];
        let encoded =
            encode_key_vector(&vector, TurboQuantPreset::K16V4).expect("encode K16 key vector");
        let decoded = decode_key_vector(&encoded).expect("decode K16 key vector");

        assert_eq!(encoded.bit_width, 16);
        assert_eq!(
            encoded.packed_indices.len(),
            vector.len() * std::mem::size_of::<u16>()
        );
        for (expected, actual) in vector.iter().zip(decoded) {
            assert!(
                (expected - actual).abs() < 0.001,
                "expected {expected}, got {actual}"
            );
        }
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
    fn fractional_value_group_codec_uses_mixed_three_and_four_bit_payloads() {
        let values = vec![
            -1.0, -0.6, -0.2, 0.0, 0.2, 0.5, 0.9, 1.0, 2.0, 2.1, 2.4, 2.9,
        ];
        let encoded = encode_value_groups_for_preset(&values, 3, TurboQuantPreset::K8V3_5)
            .expect("encode fractional values");
        let decoded = decode_value_groups_4bit(&encoded).expect("decode fractional values");

        assert_eq!(encoded.value_bits_x2, 7);
        assert_eq!(
            encoded.packed_values.len(),
            packed_value_bytes_for_preset(values.len(), TurboQuantPreset::K8V3_5)
                .expect("fractional payload bytes")
        );
        assert_eq!(decoded.len(), values.len());
        assert!(cosine_similarity(&values, &decoded) > 0.995);
    }

    #[test]
    fn key_codec_rejects_non_power_of_two_dimensions() {
        let error = encode_key_vector(&[1.0, 2.0, 3.0], TurboQuantPreset::K8V4)
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
    fn reference_decode_attention_split_matches_full_softmax_contract() {
        let query = vec![0.25, 0.5, -0.25, 1.0];
        let cold_tokens = vec![
            (vec![0.5, -0.5, 1.0, 0.0], vec![0.2, 0.0, -0.2, 0.4]),
            (vec![0.0, 1.0, -0.5, 0.5], vec![0.1, 0.3, 0.5, -0.1]),
            (vec![-0.5, 0.25, 0.0, 1.0], vec![0.6, -0.2, 0.0, 0.2]),
        ];
        let hot_tokens = vec![
            (vec![0.75, 0.0, -0.25, 0.5], vec![0.0, 0.2, 0.4, 0.6]),
            (vec![-0.5, 0.5, 0.75, -0.25], vec![0.3, -0.1, 0.2, 0.1]),
        ];
        let all_tokens = cold_tokens
            .iter()
            .chain(&hot_tokens)
            .cloned()
            .collect::<Vec<_>>();

        let expected = reference_decode_attention(&query, &all_tokens).expect("full attention");
        let actual = reference_decode_attention_split(&query, &cold_tokens, &hot_tokens)
            .expect("split attention");

        assert!(
            max_abs_diff(&expected, &actual) < 1e-6,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn reference_decode_attention_split_handles_single_partition_fallbacks() {
        let query = vec![0.25, 0.5, -0.25, 1.0];
        let tokens = vec![
            (vec![0.5, -0.5, 1.0, 0.0], vec![0.2, 0.0, -0.2, 0.4]),
            (vec![0.0, 1.0, -0.5, 0.5], vec![0.1, 0.3, 0.5, -0.1]),
        ];

        let expected = reference_decode_attention(&query, &tokens).expect("full attention");
        assert_eq!(
            reference_decode_attention_split(&query, &tokens, &[]).expect("cold-only attention"),
            expected
        );
        assert_eq!(
            reference_decode_attention_split(&query, &[], &tokens).expect("hot-only attention"),
            expected
        );
        assert_eq!(
            reference_decode_attention_split(&query, &[], &[]).expect_err("empty split"),
            TurboQuantCodecError::EmptyKvHistory
        );
    }

    #[test]
    fn merge_attention_partition_stats_rejects_value_dim_mismatch() {
        let error = merge_attention_partition_stats(&[
            TurboQuantAttentionPartitionStats {
                token_count: 1,
                value_dim: 2,
                max_score: 0.0,
                exp_sum: 1.0,
                weighted_value_sum: vec![1.0, 0.0],
            },
            TurboQuantAttentionPartitionStats {
                token_count: 1,
                value_dim: 3,
                max_score: 0.0,
                exp_sum: 1.0,
                weighted_value_sum: vec![1.0, 0.0, 0.0],
            },
        ])
        .expect_err("partition merge should fail closed");

        assert_eq!(
            error,
            TurboQuantCodecError::MismatchedVectorDimension {
                expected: 2,
                actual: 3
            }
        );
    }

    #[test]
    fn prototype_decode_attention_matches_full_precision_when_all_tokens_are_hot() {
        let config = TurboQuantKvPrototypeConfig::new(TurboQuantPreset::K8V4, 4, 4, 2)
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
        let config = TurboQuantKvPrototypeConfig::new(TurboQuantPreset::K8V4, 8, 2, 4)
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
        let config = TurboQuantKvPrototypeConfig::new(TurboQuantPreset::K8V4, 8, 2, 4)
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
        let config = TurboQuantKvPrototypeConfig::new(TurboQuantPreset::K4V4, 4, 0, 2)
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
        let config = TurboQuantKvPrototypeConfig::new(TurboQuantPreset::K8V4, 4, 2, 2)
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

    #[test]
    fn uniform_centroid_index_matches_linear_scan_for_all_bit_widths() {
        for &bit_width in &[3u32, 4, 8] {
            let centroids = key_centroids(bit_width).expect("centroids");
            let levels = 1usize << bit_width;
            let steps = 200;
            for step in 0..=steps {
                let value = -1.0 + 2.0 * step as f32 / steps as f32;
                let fast_index = uniform_centroid_index(value, levels);
                let scan_index = nearest_centroid_index(value, &centroids);
                let fast_dist = (value - centroids[fast_index as usize]).abs();
                let scan_dist = (value - centroids[scan_index as usize]).abs();
                assert!(
                    (fast_dist - scan_dist).abs() < 1e-6,
                    "bit_width={bit_width} value={value:.4}: \
                     fast_index={fast_index} (dist={fast_dist:.6}) vs \
                     scan_index={scan_index} (dist={scan_dist:.6})"
                );
            }
        }
    }

    #[test]
    fn written_slot_count_is_maintained_incrementally() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 4,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        assert_eq!(buffer.written_slot_count(), 0);

        let heads = vec![
            (
                vec![0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.75, -0.75],
                vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
            ),
            (
                vec![-0.5, 0.5, -0.25, 0.25, -0.1, 0.1, -0.75, 0.75],
                vec![-0.1, -0.2, -0.3, -0.4, 0.1, 0.2, 0.3, 0.4],
            ),
        ];
        buffer.write_token(0, &heads).expect("write token 0");
        assert_eq!(buffer.written_slot_count(), 2);

        buffer.write_token(1, &heads).expect("write token 1");
        assert_eq!(buffer.written_slot_count(), 4);

        buffer.write_token(0, &heads).expect("overwrite token 0");
        assert_eq!(
            buffer.written_slot_count(),
            4,
            "overwrite must not double-count"
        );
    }

    #[test]
    fn write_token_with_encoded_k8_keys_matches_cpu_token_write() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 4,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout");
        let heads = vec![
            (
                vec![0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.75, -0.75],
                vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
            ),
            (
                vec![-0.5, 0.5, -0.25, 0.25, -0.1, 0.1, -0.75, 0.75],
                vec![-0.1, -0.2, -0.3, -0.4, 0.1, 0.2, 0.3, 0.4],
            ),
        ];
        let encoded_keys = heads
            .iter()
            .enumerate()
            .map(|(head_index, (key, _))| {
                encode_key_vector_for_head(key, TurboQuantPreset::K8V4, head_index)
                    .expect("CPU key encode")
            })
            .collect::<Vec<_>>();
        let packed_key_bytes = encoded_keys
            .iter()
            .flat_map(|key| key.packed_indices.iter().copied())
            .collect::<Vec<_>>();
        let key_norms = encoded_keys
            .iter()
            .map(|key| key.l2_norm)
            .collect::<Vec<_>>();

        let mut cpu_buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let mut preencoded_buffer = TurboQuantCompressedBlockBuffer::new(layout);
        cpu_buffer.write_token(0, &heads).expect("CPU write");
        preencoded_buffer
            .write_token_with_encoded_k8_keys(0, &heads, &packed_key_bytes, &key_norms)
            .expect("preencoded write");

        assert_eq!(preencoded_buffer.as_bytes(), cpu_buffer.as_bytes());
        assert_eq!(
            preencoded_buffer
                .read_compressed_token(0)
                .expect("preencoded token"),
            cpu_buffer.read_compressed_token(0).expect("CPU token")
        );
    }

    #[test]
    fn compressed_block_buffer_state_round_trips_codec_metadata() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 4,
            n_kv_heads: 1,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let token = (
            vec![0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.75, -0.75],
            vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
        );
        buffer
            .write_token(0, std::slice::from_ref(&token))
            .expect("write token");

        let state = buffer.state();
        let meta_state = buffer.meta_state();
        assert_eq!(
            meta_state.key_codec,
            TurboQuantKeyCodecConfig::rht_lloyd_max()
        );
        let restored = TurboQuantCompressedBlockBuffer::from_state(meta_state, state)
            .expect("buffer state should restore");

        assert_eq!(
            restored.key_codec(),
            TurboQuantKeyCodecConfig::rht_lloyd_max()
        );
        assert_eq!(restored.written_slot_count(), 1);
        let expected = buffer
            .debug_reconstruct_slot(0, 0)
            .expect("original reconstruct");
        let actual = restored
            .debug_reconstruct_slot(0, 0)
            .expect("restored reconstruct");
        assert_eq!(actual, expected);
    }

    #[test]
    fn compressed_block_buffer_maintains_incremental_dequant_cache_for_sequential_tokens() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 4,
            n_kv_heads: 1,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        for token_index in 0..3 {
            let token = (
                (0..8)
                    .map(|idx| token_index as f32 * 0.1 + idx as f32 * 0.01)
                    .collect::<Vec<_>>(),
                (0..8)
                    .map(|idx| token_index as f32 * -0.1 + idx as f32 * 0.02)
                    .collect::<Vec<_>>(),
            );
            buffer
                .write_token(token_index, std::slice::from_ref(&token))
                .expect("write sequential token");
            assert_eq!(buffer.dequant_buffer_token_count(), token_index + 1);
        }

        assert_eq!(buffer.dequant_buffer_alloc_tokens(), 4);
        let cached = buffer
            .debug_reconstruct_head_history_cached(0, 3)
            .expect("cached reconstruct");
        assert_eq!(cached.len(), 3);

        buffer
            .write_slot(0, 0, &[0.0; 8], &[0.0; 8])
            .expect("overwrite slot");
        assert_eq!(buffer.dequant_buffer_token_count(), 0);
        assert_eq!(buffer.dequant_buffer_alloc_tokens(), 0);
    }

    #[test]
    fn incremental_dequant_cache_matches_lazy_reconstruction() {
        // Regression: the sequential-write path used to cache f16(original
        // input) while a cache miss reconstructed from the compressed buffer,
        // so the same token returned different values depending on access
        // order — and a warm cache reported zero quantization error.
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 4,
            n_kv_heads: 1,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout");
        let tokens = (0..3)
            .map(|token_index| {
                (
                    (0..8)
                        .map(|idx| ((idx * 7 + token_index * 3) % 11) as f32 / 9.0 - 0.5)
                        .collect::<Vec<_>>(),
                    (0..8)
                        .map(|idx| ((idx * 5 + token_index * 2) % 13) as f32 / 7.0 - 0.8)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let mut warm = TurboQuantCompressedBlockBuffer::new(layout);
        let mut cold = TurboQuantCompressedBlockBuffer::new(layout);
        for (token_index, token) in tokens.iter().enumerate() {
            warm.write_token(token_index, std::slice::from_ref(token))
                .expect("warm write");
            cold.write_token(token_index, std::slice::from_ref(token))
                .expect("cold write");
        }
        assert_eq!(warm.dequant_buffer_token_count(), 3);
        cold.invalidate_dequant_buffers();
        assert_eq!(cold.dequant_buffer_token_count(), 0);

        let warm_history = warm
            .debug_reconstruct_head_history_cached(0, 3)
            .expect("warm cached reconstruct");
        let cold_history = cold
            .debug_reconstruct_head_history_cached(0, 3)
            .expect("cold lazy reconstruct");
        assert_eq!(warm_history, cold_history);

        // The cached history must reflect the codec roundtrip, not the
        // original inputs: K8V4 keys are lossy on this data.
        let (original_key, _) = &tokens[0];
        let (cached_key, _) = &warm_history[0];
        assert!(
            original_key
                .iter()
                .zip(cached_key)
                .any(|(original, cached)| (original - cached).abs() > 1e-4),
            "cached history unexpectedly matches the pre-quantization inputs"
        );
    }

    #[test]
    fn compressed_block_buffer_truncate_invalidates_tail_and_dequant_cache() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 4,
            n_kv_heads: 1,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        for token_index in 0..3 {
            let token = (
                (0..8)
                    .map(|idx| token_index as f32 * 0.1 + idx as f32 * 0.01)
                    .collect::<Vec<_>>(),
                (0..8)
                    .map(|idx| token_index as f32 * -0.1 + idx as f32 * 0.02)
                    .collect::<Vec<_>>(),
            );
            buffer
                .write_token(token_index, std::slice::from_ref(&token))
                .expect("write sequential token");
        }
        assert_eq!(buffer.token_count(), 3);
        assert_eq!(buffer.written_slot_count(), 3);
        assert_eq!(buffer.dequant_buffer_token_count(), 3);

        buffer.truncate_token_count(1).expect("truncate buffer");

        assert_eq!(buffer.token_count(), 1);
        assert_eq!(buffer.written_slot_count(), 1);
        assert_eq!(buffer.dequant_buffer_token_count(), 0);
        assert_eq!(buffer.dequant_buffer_alloc_tokens(), 0);
        assert!(matches!(
            buffer.read_compressed_slot(1, 0),
            Err(TurboQuantCodecError::CompressedSlotUnwritten { .. })
        ));
    }

    #[test]
    fn seven_bit_pack_round_trips_at_tail_lengths() {
        for index_count in [1usize, 7, 8, 13, 64] {
            let indices = (0..index_count)
                .map(|index| ((index * 37 + 11) % 128) as u8)
                .collect::<Vec<_>>();
            let packed = pack_indices(&indices, 7).expect("7-bit pack");
            assert_eq!(packed.len(), packed_index_bytes(index_count, 7).unwrap());
            let unpacked = unpack_indices(&packed, index_count, 7).expect("7-bit unpack");
            assert_eq!(
                unpacked, indices,
                "round trip failed at {index_count} indices"
            );
        }
        assert!(matches!(
            pack_indices(&[128], 7),
            Err(TurboQuantCodecError::PackedIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn k7v4_key_codec_round_trips_with_high_cosine() {
        let dim = 64usize;
        let vector = (0..dim)
            .map(|index| ((index * 13 + 5) % 17) as f32 / 8.0 - 1.0)
            .collect::<Vec<_>>();
        let encoded =
            encode_key_vector_for_head(&vector, TurboQuantPreset::K7V4, 0).expect("K7V4 encode");
        assert_eq!(encoded.bit_width, 7);
        let decoded = decode_key_vector(&encoded).expect("K7V4 decode");

        let dot = vector.iter().zip(&decoded).map(|(a, b)| a * b).sum::<f32>();
        let norm_a = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b = decoded.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cosine = dot / (norm_a * norm_b).max(f32::EPSILON);
        assert!(cosine > 0.995, "K7V4 cosine too low: {cosine}");
    }

    #[test]
    fn ragged_value_groups_round_trip_within_group_scale() {
        // 10 elements with group size 4 leaves a ragged 2-element tail group.
        let values = (0..10)
            .map(|index| ((index * 19 + 3) % 23) as f32 / 7.0 - 1.5)
            .collect::<Vec<_>>();
        for preset in [TurboQuantPreset::K8V4, TurboQuantPreset::K8V3_5] {
            let encoded =
                encode_value_groups_for_preset(&values, 4, preset).expect("ragged encode");
            assert_eq!(encoded.element_count, 10);
            assert_eq!(encoded.mins.len(), 3);
            let decoded = decode_value_groups_4bit(&encoded).expect("ragged decode");
            assert_eq!(decoded.len(), values.len());
            for (index, (original, decoded)) in values.iter().zip(&decoded).enumerate() {
                let scale = encoded.scales[index / 4];
                assert!(
                    (original - decoded).abs() <= scale * 0.5 + 1e-6,
                    "{preset:?} element {index}: original={original} decoded={decoded} scale={scale}"
                );
            }
        }
    }

    #[test]
    fn f16_conversion_edges() {
        // Exactly representable values round-trip bit-perfectly.
        for value in [0.0f32, 1.0, -2.5, 0.25, 65504.0, -65504.0] {
            assert_eq!(f16_bits_to_f32(f32_to_f16_bits(value)), value);
        }
        // Values beyond f16 range saturate to infinity (current contract:
        // callers must bound inputs; K vectors never approach 65504).
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(70000.0)), f32::INFINITY);
        assert_eq!(
            f16_bits_to_f32(f32_to_f16_bits(-70000.0)),
            f32::NEG_INFINITY
        );
        // NaN stays NaN.
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
        // Subnormals round-trip within one subnormal step (~5.96e-8).
        let tiny = 1.0e-7f32;
        let round_tripped = f16_bits_to_f32(f32_to_f16_bits(tiny));
        assert!(
            (round_tripped - tiny).abs() <= 6.0e-8,
            "subnormal round trip drifted: {round_tripped}"
        );
        // Values straddling a representable step round to a neighbour within
        // half a ULP (f16 ULP at 1.0 is ~9.77e-4).
        let nearly_one = 1.0003f32;
        let round_tripped = f16_bits_to_f32(f32_to_f16_bits(nearly_one));
        assert!(
            (round_tripped - nearly_one).abs() <= 4.9e-4,
            "rounding drifted: {round_tripped}"
        );
    }

    // ── SSD memory optimisation: TurboQuant memory reduction tests ──

    #[test]
    fn preset_key_bits_and_value_bits_match_documentation() {
        use ax_engine_core::TurboQuantPreset;
        // Verify every preset reports the documented bit widths.
        assert_eq!(TurboQuantPreset::K8V4.key_bits(), 8);
        assert_eq!(TurboQuantPreset::K8V4.value_bits(), 4);

        assert_eq!(TurboQuantPreset::K4V4.key_bits(), 4);
        assert_eq!(TurboQuantPreset::K4V4.value_bits(), 4);

        assert_eq!(TurboQuantPreset::K3V4Research.key_bits(), 3);
        assert_eq!(TurboQuantPreset::K3V4Research.value_bits(), 4);

        assert_eq!(TurboQuantPreset::K16V4.key_bits(), 16);
        assert_eq!(TurboQuantPreset::K16V4.value_bits(), 4);

        assert_eq!(TurboQuantPreset::K8V3_5.key_bits(), 8);
        // K8V3_5 has fractional values (3.5 bits): value_bits_x2 = 7
        assert_eq!(TurboQuantPreset::K8V3_5.value_bits_x2(), 7);
        assert_eq!(TurboQuantPreset::K8V3_5.value_bits(), 4); // 7.div_ceil(2) = 4

        assert_eq!(TurboQuantPreset::K7V4.key_bits(), 7);
        assert_eq!(TurboQuantPreset::K7V4.value_bits(), 4);
    }

    #[test]
    fn k8v4_memory_reduction_vs_fp16_is_approximately_62_percent() {
        use ax_engine_core::TurboQuantPreset;
        let preset = TurboQuantPreset::K8V4;
        let compressed_bits = preset.key_bits() + preset.value_bits(); // 12
        let fp16_bits: u32 = 32; // 16-bit key + 16-bit value
        let ratio = compressed_bits as f64 / fp16_bits as f64;
        // 12/32 = 0.375 → 62.5% reduction
        assert!(
            (ratio - 0.375).abs() < 0.01,
            "K8V4 ratio must be ~37.5% of fp16 (62.5% reduction), got {ratio}"
        );
    }

    #[test]
    fn k4v4_memory_reduction_vs_fp16_is_approximately_75_percent() {
        use ax_engine_core::TurboQuantPreset;
        let preset = TurboQuantPreset::K4V4;
        let compressed_bits = preset.key_bits() + preset.value_bits(); // 8
        let fp16_bits: u32 = 32;
        let ratio = compressed_bits as f64 / fp16_bits as f64;
        // 8/32 = 0.25 → 75% reduction
        assert!(
            (ratio - 0.25).abs() < 0.01,
            "K4V4 ratio must be ~25% of fp16 (75% reduction), got {ratio}"
        );
    }

    #[test]
    fn k3v4_research_memory_reduction_vs_fp16_is_approximately_78_percent() {
        use ax_engine_core::TurboQuantPreset;
        let preset = TurboQuantPreset::K3V4Research;
        let compressed_bits = preset.key_bits() + preset.value_bits(); // 7
        let fp16_bits: u32 = 32;
        let ratio = compressed_bits as f64 / fp16_bits as f64;
        // 7/32 = ~0.219 → ~78.1% reduction
        assert!(
            (ratio - 0.21875).abs() < 0.01,
            "K3V4 ratio must be ~21.9% of fp16 (~78% reduction), got {ratio}"
        );
    }

    #[test]
    fn k16v4_fallback_preset_has_full_precision_keys() {
        use ax_engine_core::TurboQuantPreset;
        assert!(
            TurboQuantPreset::K16V4.has_full_precision_keys(),
            "K16V4 must report full-precision keys (fallback mode)"
        );
        // Non-K16V4 presets must not report full precision keys.
        assert!(!TurboQuantPreset::K8V4.has_full_precision_keys());
        assert!(!TurboQuantPreset::K4V4.has_full_precision_keys());
    }

    #[test]
    fn k8v3_5_is_the_only_fractional_value_preset() {
        use ax_engine_core::TurboQuantPreset;
        assert!(
            TurboQuantPreset::K8V3_5.has_fractional_values(),
            "K8V3_5 must report fractional values"
        );
        // All other presets must not have fractional values.
        assert!(!TurboQuantPreset::K8V4.has_fractional_values());
        assert!(!TurboQuantPreset::K4V4.has_fractional_values());
        assert!(!TurboQuantPreset::K3V4Research.has_fractional_values());
        assert!(!TurboQuantPreset::K16V4.has_fractional_values());
        assert!(!TurboQuantPreset::K7V4.has_fractional_values());
    }

    #[test]
    fn all_presets_have_unique_route_codes() {
        use ax_engine_core::TurboQuantPreset;
        use std::collections::HashSet;
        let codes: HashSet<u32> = [
            TurboQuantPreset::K8V4,
            TurboQuantPreset::K4V4,
            TurboQuantPreset::K3V4Research,
            TurboQuantPreset::K16V4,
            TurboQuantPreset::K8V3_5,
            TurboQuantPreset::K7V4,
        ]
        .iter()
        .map(|p| p.route_code())
        .collect();
        assert_eq!(codes.len(), 6, "all 6 presets must have unique route codes");
    }

    #[test]
    fn turboquant_layout_estimates_real_compressed_memory_savings() {
        fn estimate_for_context(total_tokens: usize) -> TurboQuantFusedDecodeBenchmarkEstimate {
            let hot_tokens = 128;
            let cold_tokens = total_tokens - hot_tokens;
            let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
                preset: TurboQuantPreset::K8V4,
                block_tokens: 16,
                n_kv_heads: 8,
                head_dim: TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM,
                value_group_size: 32,
            })
            .expect("layout should build");
            let compressed_buffer_bytes = layout
                .buffer_bytes_for_tokens(cold_tokens)
                .expect("buffer size should fit");
            let key_payload_offset_in_slot = 0;
            let key_norm_offset_in_slot =
                key_payload_offset_in_slot + layout.key_payload_bytes_per_head;
            let value_payload_offset_in_slot =
                key_norm_offset_in_slot + layout.key_norm_bytes_per_head;
            let value_mins_offset_in_slot =
                value_payload_offset_in_slot + layout.value_payload_bytes_per_head;
            let value_scales_offset_in_slot =
                value_mins_offset_in_slot + layout.value_group_count * std::mem::size_of::<f32>();

            TurboQuantFusedDecodeLaunchDescriptor {
                preset: TurboQuantPreset::K8V4,
                key_bits: TurboQuantPreset::K8V4.key_bits(),
                value_bits: TurboQuantPreset::K8V4.value_bits(),
                total_tokens,
                cold_tokens,
                hot_tokens,
                n_query_heads: 32,
                n_kv_heads: layout.config.n_kv_heads,
                head_dim: layout.config.head_dim,
                block_tokens: layout.config.block_tokens,
                compressed_blocks: layout.block_count_for_tokens(cold_tokens),
                compressed_buffer_bytes,
                required_compressed_slots: cold_tokens * layout.config.n_kv_heads,
                value_group_size: layout.config.value_group_size,
                value_group_count: layout.value_group_count,
                key_payload_bytes_per_head: layout.key_payload_bytes_per_head,
                key_norm_bytes_per_head: layout.key_norm_bytes_per_head,
                value_payload_bytes_per_head: layout.value_payload_bytes_per_head,
                value_metadata_bytes_per_head: layout.value_metadata_bytes_per_head,
                raw_slot_bytes_per_head: layout.raw_slot_bytes_per_head,
                slot_bytes_per_head: layout.slot_bytes_per_head,
                token_stride_bytes: layout.token_stride_bytes,
                block_bytes: layout.block_bytes,
                key_payload_offset_in_slot,
                key_norm_offset_in_slot,
                value_payload_offset_in_slot,
                value_mins_offset_in_slot,
                value_scales_offset_in_slot,
            }
            .benchmark_estimate()
        }

        let ctx_1k = estimate_for_context(1024);
        let ctx_8k = estimate_for_context(8192);

        assert_eq!(ctx_1k.full_precision_total_kv_kib, 4096);
        assert_eq!(ctx_8k.full_precision_total_kv_kib, 32768);
        assert!(
            ctx_1k.compressed_buffer_kib < ctx_1k.full_precision_cold_kv_kib,
            "compressed K8V4 buffer should be smaller than cold fp16 KV"
        );
        assert!(
            ctx_1k.estimated_total_saved_read_kib > 0,
            "real layout estimate should report positive read savings"
        );
        assert!(
            ctx_1k.cold_compression_ratio_milli < 1000,
            "real layout cold read ratio should stay below fp16"
        );
        assert!(
            ctx_8k.estimated_total_saved_read_kib > ctx_1k.estimated_total_saved_read_kib * 7,
            "savings should scale with context length after fixed hot window"
        );
    }

    #[test]
    fn turboquant_memory_savings_scale_with_context_length() {
        // Verify the math: for N tokens with head_dim D and n_kv_heads H,
        // the memory savings from TurboQuant are proportional to context length.
        let head_dim: u64 = 128;
        let n_kv_heads: u64 = 8;
        let fp16_bytes_per_token = 2 * head_dim * n_kv_heads * 2; // 2 (K+V) × D × H × 2 bytes

        // 1K context
        let ctx_1k: u64 = 1024;
        let fp16_1k = fp16_bytes_per_token * ctx_1k;

        // K8V4 compressed: (8+4)/32 of fp16
        let compressed_1k = (fp16_1k * 12) / 32;
        let savings_1k = fp16_1k - compressed_1k;

        // 8K context — savings should scale 8x
        let ctx_8k: u64 = 8192;
        let fp16_8k = fp16_bytes_per_token * ctx_8k;
        let compressed_8k = (fp16_8k * 12) / 32;
        let savings_8k = fp16_8k - compressed_8k;

        assert_eq!(
            savings_8k / savings_1k,
            8,
            "TurboQuant memory savings must scale linearly with context length"
        );

        // Verify the absolute savings for 1K context
        // fp16_1k = 2 × 128 × 8 × 2 × 1024 = 4,194,304 bytes = 4 MiB
        assert_eq!(fp16_1k, 4_194_304, "fp16 KV for 1K ctx must be 4 MiB");
        // compressed_1k = 4,194,304 × 12 / 32 = 1,572,864 bytes ≈ 1.5 MiB
        assert_eq!(compressed_1k, 1_572_864, "K8V4 compressed must be 1.5 MiB");
        // savings = 4 MiB - 1.5 MiB = 2.5 MiB = 62.5%
        assert_eq!(savings_1k, 2_621_440, "savings must be 2.5 MiB (62.5%)");
    }

    #[test]
    fn fallback_preset_returns_k16v4_when_quality_gate_fails_for_k8v4() {
        use ax_engine_core::TurboQuantPreset;
        // When K8V4 fails the quality gate, the fallback must be K16V4.
        let readiness = TurboQuantFusedDecodePromotionReadiness {
            status: TurboQuantFusedDecodePromotionStatus::QualityGateFailed,
            preset: TurboQuantPreset::K8V4,
            quality_profile: TurboQuantDecodeQualityProfile::for_quantization_preset(
                TurboQuantPreset::K8V4,
            ),
            benchmark_estimate: zero_benchmark_estimate(TurboQuantPreset::K8V4),
        };
        assert_eq!(
            readiness.fallback_preset(),
            Some(TurboQuantPreset::K16V4),
            "K8V4 with failed quality gate must fallback to K16V4"
        );
        assert_eq!(
            readiness.effective_preset(),
            TurboQuantPreset::K16V4,
            "effective_preset must return the fallback"
        );
    }

    #[test]
    fn fallback_preset_returns_none_when_quality_gate_passes() {
        use ax_engine_core::TurboQuantPreset;
        let readiness = TurboQuantFusedDecodePromotionReadiness {
            status: TurboQuantFusedDecodePromotionStatus::Ready,
            preset: TurboQuantPreset::K8V4,
            quality_profile: TurboQuantDecodeQualityProfile::for_quantization_preset(
                TurboQuantPreset::K8V4,
            ),
            benchmark_estimate: zero_benchmark_estimate(TurboQuantPreset::K8V4),
        };
        assert_eq!(
            readiness.fallback_preset(),
            None,
            "no fallback when quality gate passes"
        );
        assert_eq!(
            readiness.effective_preset(),
            TurboQuantPreset::K8V4,
            "effective_preset must return the original preset"
        );
    }

    fn zero_benchmark_estimate(
        preset: ax_engine_core::TurboQuantPreset,
    ) -> TurboQuantFusedDecodeBenchmarkEstimate {
        TurboQuantFusedDecodeBenchmarkEstimate {
            preset,
            key_bits: preset.key_bits(),
            value_bits: preset.value_bits(),
            total_tokens: 0,
            cold_tokens: 0,
            hot_tokens: 0,
            n_query_heads: 0,
            n_kv_heads: 0,
            head_dim: 0,
            compressed_blocks: 0,
            cold_score_elements: 0,
            hot_score_elements: 0,
            output_elements: 0,
            compressed_buffer_kib: 0,
            full_precision_cold_kv_kib: 0,
            full_precision_total_kv_kib: 0,
            estimated_compressed_cold_kv_kib: 0,
            hot_full_precision_kv_kib: 0,
            estimated_total_read_kib: 0,
            estimated_cold_saved_kib: 0,
            estimated_total_saved_read_kib: 0,
            cold_compression_ratio_milli: 0,
        }
    }
}
