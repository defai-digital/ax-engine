use ax_engine_core::TurboQuantPreset;

use crate::model::ModelConfig;
use crate::turboquant::{
    TurboQuantBlockLayout, TurboQuantBlockLayoutConfig, TurboQuantCodecError,
    TurboQuantLayerSupportReason, turboquant_support_report,
};

pub const OPEN_TQ_METAL_PRESET: TurboQuantPreset = TurboQuantPreset::K4V4;
pub const OPEN_TQ_METAL_VALUE_GROUP_SIZE: usize = 32;
pub const OPEN_TQ_METAL_FULL_PRECISION_KV_BYTES_PER_ELEMENT: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenTqMetalModelFit {
    DenseFullAttention,
    PartialHybridAttention,
    NoEligibleLayers,
    InvalidGroupedQueryAttention,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenTqMetalSupportReport {
    pub preset: TurboQuantPreset,
    pub fit: OpenTqMetalModelFit,
    pub total_layers: usize,
    pub eligible_layers: usize,
    pub linear_attention_layers: usize,
    pub sliding_window_layers: usize,
    pub kv_shared_layers: usize,
    pub unsupported_head_dim_layers: usize,
    pub estimated_full_precision_bytes_per_token: usize,
    pub estimated_compressed_bytes_per_token: usize,
}

impl OpenTqMetalSupportReport {
    pub fn has_candidate_layers(&self) -> bool {
        self.eligible_layers > 0
            && self.fit != OpenTqMetalModelFit::InvalidGroupedQueryAttention
    }

    pub fn eligible_layer_ratio_milli(&self) -> u32 {
        if self.total_layers == 0 {
            return 0;
        }
        (self.eligible_layers.saturating_mul(1000) / self.total_layers)
            .min(u32::MAX as usize) as u32
    }

    pub fn estimated_compression_ratio_milli(&self) -> u32 {
        if self.estimated_full_precision_bytes_per_token == 0 {
            return 0;
        }
        (self.estimated_compressed_bytes_per_token.saturating_mul(1000)
            / self.estimated_full_precision_bytes_per_token)
            .min(u32::MAX as usize) as u32
    }
}

pub fn open_tq_metal_support_report(
    cfg: &ModelConfig,
) -> Result<OpenTqMetalSupportReport, TurboQuantCodecError> {
    let support = turboquant_support_report(cfg, OPEN_TQ_METAL_PRESET)?;
    let valid_gqa = cfg.n_kv_heads > 0 && cfg.n_heads.is_multiple_of(cfg.n_kv_heads);
    let fit = if !valid_gqa {
        OpenTqMetalModelFit::InvalidGroupedQueryAttention
    } else if support.eligible_layers == 0 {
        OpenTqMetalModelFit::NoEligibleLayers
    } else if support.eligible_layers == cfg.layer_count {
        OpenTqMetalModelFit::DenseFullAttention
    } else {
        OpenTqMetalModelFit::PartialHybridAttention
    };

    let mut estimated_full_precision_bytes_per_token = 0usize;
    let mut estimated_compressed_bytes_per_token = 0usize;
    for layer in &support.layers {
        if layer.reason != TurboQuantLayerSupportReason::Eligible {
            continue;
        }
        let full_precision_elements = cfg
            .n_kv_heads
            .saturating_mul(layer.head_dim)
            .saturating_mul(2);
        estimated_full_precision_bytes_per_token = estimated_full_precision_bytes_per_token
            .saturating_add(
                full_precision_elements
                    .saturating_mul(OPEN_TQ_METAL_FULL_PRECISION_KV_BYTES_PER_ELEMENT),
            );

        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: OPEN_TQ_METAL_PRESET,
            block_tokens: 1,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: layer.head_dim,
            value_group_size: OPEN_TQ_METAL_VALUE_GROUP_SIZE
                .min(layer.head_dim)
                .max(1),
        })?;
        estimated_compressed_bytes_per_token =
            estimated_compressed_bytes_per_token.saturating_add(layout.token_stride_bytes);
    }

    Ok(OpenTqMetalSupportReport {
        preset: OPEN_TQ_METAL_PRESET,
        fit,
        total_layers: cfg.layer_count,
        eligible_layers: support.eligible_layers,
        linear_attention_layers: support.linear_attention_layers,
        sliding_window_layers: support.sliding_window_layers,
        kv_shared_layers: support.kv_shared_layers,
        unsupported_head_dim_layers: support.unsupported_head_dim_layers,
        estimated_full_precision_bytes_per_token,
        estimated_compressed_bytes_per_token,
    })
}
