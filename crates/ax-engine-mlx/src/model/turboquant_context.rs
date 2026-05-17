use ax_engine_core::{KvCompressionConfig, TurboQuantPreset};

use crate::fastpath;
use crate::kv_cache::{MlxKVCache, MlxKvCompressionDecodeCandidate};
use crate::turboquant::turboquant_fused_decode_head_dim_supported;

use super::ModelConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurboQuantModelDecodeCandidateStatus {
    Disabled,
    PrefillOnly,
    LinearAttentionLayer,
    GlmMlaLayer,
    SlidingWindowLayer,
    KvSharedLayer,
    IneligibleLayer,
    UnsupportedPreset,
    UnsupportedHeadDim,
    GroupedQueryAttention,
    MissingRuntimeStorage,
    Ready,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantModelDecodeCandidate {
    pub status: TurboQuantModelDecodeCandidateStatus,
    pub cold_tokens: usize,
    pub hot_tokens: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct TurboQuantModelDecodeContext<'a> {
    pub config: KvCompressionConfig,
    pub layer_eligible: &'a [bool],
}

impl TurboQuantModelDecodeCandidate {
    pub const fn disabled() -> Self {
        Self {
            status: TurboQuantModelDecodeCandidateStatus::Disabled,
            cold_tokens: 0,
            hot_tokens: 0,
        }
    }

    pub const fn telemetry_status(self) -> MlxKvCompressionDecodeCandidate {
        match self.status {
            TurboQuantModelDecodeCandidateStatus::Disabled => {
                MlxKvCompressionDecodeCandidate::Disabled
            }
            TurboQuantModelDecodeCandidateStatus::PrefillOnly => {
                MlxKvCompressionDecodeCandidate::PrefillOnly
            }
            TurboQuantModelDecodeCandidateStatus::LinearAttentionLayer => {
                MlxKvCompressionDecodeCandidate::LinearAttention
            }
            TurboQuantModelDecodeCandidateStatus::GlmMlaLayer => {
                MlxKvCompressionDecodeCandidate::GlmMla
            }
            TurboQuantModelDecodeCandidateStatus::SlidingWindowLayer => {
                MlxKvCompressionDecodeCandidate::SlidingWindow
            }
            TurboQuantModelDecodeCandidateStatus::KvSharedLayer => {
                MlxKvCompressionDecodeCandidate::KvShared
            }
            TurboQuantModelDecodeCandidateStatus::IneligibleLayer => {
                MlxKvCompressionDecodeCandidate::IneligibleLayer
            }
            TurboQuantModelDecodeCandidateStatus::UnsupportedPreset => {
                MlxKvCompressionDecodeCandidate::UnsupportedPreset
            }
            TurboQuantModelDecodeCandidateStatus::UnsupportedHeadDim => {
                MlxKvCompressionDecodeCandidate::UnsupportedHeadDim
            }
            TurboQuantModelDecodeCandidateStatus::GroupedQueryAttention => {
                MlxKvCompressionDecodeCandidate::GroupedQueryAttention
            }
            TurboQuantModelDecodeCandidateStatus::MissingRuntimeStorage => {
                MlxKvCompressionDecodeCandidate::MissingRuntimeStorage
            }
            TurboQuantModelDecodeCandidateStatus::Ready => MlxKvCompressionDecodeCandidate::Ready,
        }
    }
}

impl<'a> TurboQuantModelDecodeContext<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn decode_candidate(
        self,
        cfg: &ModelConfig,
        cache: &MlxKVCache,
        layer_idx: usize,
        seq: usize,
        head_dim: usize,
        kv_heads: usize,
        sliding_window: Option<usize>,
        kv_source: Option<usize>,
        has_glm_mla_attention: bool,
    ) -> TurboQuantModelDecodeCandidate {
        let status = if !self.config.requests_fused_decode()
            || fastpath::turboquant_fused_decode_disabled()
        {
            TurboQuantModelDecodeCandidateStatus::Disabled
        } else if seq != 1 {
            TurboQuantModelDecodeCandidateStatus::PrefillOnly
        } else if cfg.is_linear_attention_layer(layer_idx) {
            TurboQuantModelDecodeCandidateStatus::LinearAttentionLayer
        } else if has_glm_mla_attention {
            TurboQuantModelDecodeCandidateStatus::GlmMlaLayer
        } else if sliding_window.is_some() {
            TurboQuantModelDecodeCandidateStatus::SlidingWindowLayer
        } else if kv_source.is_some() {
            TurboQuantModelDecodeCandidateStatus::KvSharedLayer
        } else if !self.layer_eligible.get(layer_idx).copied().unwrap_or(false) {
            TurboQuantModelDecodeCandidateStatus::IneligibleLayer
        } else if self.config.preset != TurboQuantPreset::K8V4 {
            TurboQuantModelDecodeCandidateStatus::UnsupportedPreset
        } else if !turboquant_fused_decode_head_dim_supported(head_dim) {
            TurboQuantModelDecodeCandidateStatus::UnsupportedHeadDim
        } else if kv_heads == 0 || !cfg.n_heads.is_multiple_of(kv_heads) {
            TurboQuantModelDecodeCandidateStatus::GroupedQueryAttention
        } else if cache
            .turboquant_shadow_storage_cold_tokens(layer_idx)
            .unwrap_or(0)
            == 0
        {
            TurboQuantModelDecodeCandidateStatus::MissingRuntimeStorage
        } else {
            TurboQuantModelDecodeCandidateStatus::Ready
        };

        let cold_tokens = if status == TurboQuantModelDecodeCandidateStatus::Ready {
            cache
                .turboquant_shadow_storage_cold_tokens(layer_idx)
                .unwrap_or(0)
        } else {
            0
        };
        let total_tokens = cache.seq_len.saturating_add(seq);
        TurboQuantModelDecodeCandidate {
            status,
            cold_tokens,
            hot_tokens: total_tokens.saturating_sub(cold_tokens),
        }
    }
}
