use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION: &str = "ax.native_model.v1";
pub const AX_NATIVE_MODEL_MANIFEST_FILE: &str = "model-manifest.json";
pub const QWEN3_5_DEFAULT_FULL_ATTENTION_INTERVAL: u32 = 4;
pub const SUPPORTED_MLX_AFFINE_QUANTIZATION_BITS: &[u32] = &[4, 5, 6, 8];
/// Set to `"1"` to allow loading affine-quantized MLX artifacts at 3-bit.
/// Production validation rejects 3-bit by default; this gate is for
/// experimental benchmarking only and carries no quality or correctness guarantee.
pub const AX_ENGINE_3BIT_EXPERIMENTAL_ENV: &str = "AX_ENGINE_3BIT_EXPERIMENTAL";
pub const EXPERIMENTAL_MLX_AFFINE_QUANTIZATION_BITS: &[u32] = &[3];
/// Set to `"1"` to allow loading affine-quantized MLX artifacts at 2-bit.
/// Same contract as the 3-bit gate: experimental benchmarking only, no
/// quality or correctness guarantee. MLX affine kernels execute 2-bit
/// natively; production validation still rejects it by default.
pub const AX_ENGINE_2BIT_EXPERIMENTAL_ENV: &str = "AX_ENGINE_2BIT_EXPERIMENTAL";
pub const EXPERIMENTAL_2BIT_MLX_AFFINE_QUANTIZATION_BITS: &[u32] = &[2];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTensorFormat {
    Safetensors,
    /// GGUF file, loaded directly without conversion.
    Gguf,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTensorDataType {
    F16,
    Bf16,
    F32,
    I8,
    U8,
    /// Packed uint32 — used by MLX affine quantization for the weight tensor.
    /// Bit width and group size are carried by per-tensor quantization metadata.
    /// Scales and biases are stored as separate bf16/f32 tensors with the same base name.
    U32,
    /// Q4_K_M quantized: 256-element super-blocks, 144 bytes each (4.5 bits/weight).
    /// Raw block_q4_K bytes stored directly in the Metal buffer; dequant happens in kernel.
    Q4Km,
    /// Q5_K quantized: 256-element super-blocks, 176 bytes each. Dequantized to F16 at load.
    Q5Km,
    /// Q6_K quantized: 256-element super-blocks, 210 bytes each. Dequantized to F16 at load.
    Q6Km,
    /// Q8_0 quantized: 32-element blocks, 34 bytes each. Dequantized to F16 at load.
    Q8Zero,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTensorRole {
    TokenEmbedding,
    AttentionNorm,
    AttentionPostNorm,
    AttentionQNorm,
    AttentionKNorm,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionQkvPacked,
    AttentionQa,
    AttentionQaNorm,
    AttentionQb,
    AttentionKvA,
    AttentionKvB,
    AttentionKvANorm,
    AttentionEmbedQ,
    AttentionUnembedOut,
    /// Separate attention output gate projection: `sigmoid(x·W_gate)` is
    /// multiplied into the attention output before `o_proj` (Muse Glimmer
    /// `self_attn.gate_proj`; distinct from the Qwen3.5 interleaved-in-q gate).
    AttentionOutputGate,
    AttentionO,
    LinearAttentionInProjQkv,
    LinearAttentionInProjQkvz,
    LinearAttentionInProjZ,
    LinearAttentionInProjA,
    LinearAttentionInProjB,
    LinearAttentionInProjBa,
    LinearAttentionConv1d,
    LinearAttentionDtBias,
    LinearAttentionALog,
    LinearAttentionNorm,
    LinearAttentionOutProj,
    FfnNorm,
    FfnNorm2,
    FfnPostNorm,
    FfnPostNorm1,
    FfnPostNorm2,
    FfnGateInp,
    FfnGateInpScale,
    FfnGateInpExpertScale,
    FfnGateInpCorrectionBias,
    FfnGate,
    FfnUp,
    FfnGateUpPacked,
    FfnSharedExpertGateInp,
    FfnSharedExpertGate,
    FfnSharedExpertUp,
    FfnSharedExpertDown,
    FfnGateExps,
    FfnUpExps,
    FfnGateUpExpsPacked,
    FfnDown,
    FfnDownExps,
    FfnDownExpsScale,
    /// GPT-OSS per-head learned attention sink weight.
    AttnSink,
    /// GPT-OSS MXFP4 gate-up expert weight blocks (u8 packed).
    FfnGateUpExpsMxfp4Blocks,
    /// GPT-OSS MXFP4 gate-up expert weight scales (E8M0).
    FfnGateUpExpsMxfp4Scales,
    /// GPT-OSS MXFP4 down expert weight blocks (u8 packed).
    FfnDownExpsMxfp4Blocks,
    /// GPT-OSS MXFP4 down expert weight scales (E8M0).
    FfnDownExpsMxfp4Scales,
    LayerScalar,
    /// Global embedding table for per-layer token inputs (Gemma4 2B/4B).
    PerLayerEmbedding,
    /// Global projection from hidden state to stacked per-layer inputs (Gemma4 2B/4B).
    PerLayerModelProjection,
    /// Global RMSNorm weight over hidden_size_per_layer_input (Gemma4 2B/4B).
    PerLayerProjectionNorm,
    /// Per-layer gate projection: hidden → hidden_size_per_layer_input (Gemma4 2B/4B).
    PerLayerInputGate,
    /// Per-layer output projection: hidden_size_per_layer_input → hidden (Gemma4 2B/4B).
    PerLayerInputProjection,
    /// Per-layer post-gating RMSNorm weight (Gemma4 2B/4B).
    PerLayerInputPostNorm,
    /// Gemma4 Assistant projection from target embedding+hidden into assistant hidden space.
    AssistantPreProjection,
    /// Gemma4 Assistant projection from assistant hidden back into target hidden space.
    AssistantPostProjection,
    /// Gemma4 Unified raw-patch projection weight.
    Gemma4UnifiedVisionPatchDense,
    /// Gemma4 Unified raw-patch projection bias.
    Gemma4UnifiedVisionPatchDenseBias,
    /// Gemma4 Unified pre-patch-projection LayerNorm weight.
    Gemma4UnifiedVisionPatchNorm1,
    /// Gemma4 Unified pre-patch-projection LayerNorm bias.
    Gemma4UnifiedVisionPatchNorm1Bias,
    /// Gemma4 Unified post-patch-projection LayerNorm weight.
    Gemma4UnifiedVisionPatchNorm2,
    /// Gemma4 Unified post-patch-projection LayerNorm bias.
    Gemma4UnifiedVisionPatchNorm2Bias,
    /// Gemma4 Unified factorized 2D positional embedding table.
    Gemma4UnifiedVisionPositionEmbedding,
    /// Gemma4 Unified post-position LayerNorm weight.
    Gemma4UnifiedVisionPositionNorm,
    /// Gemma4 Unified post-position LayerNorm bias.
    Gemma4UnifiedVisionPositionNormBias,
    /// Gemma4 Unified vision multimodal projection into LM hidden space.
    Gemma4UnifiedVisionProjection,
    /// Gemma4 Unified audio multimodal projection into LM hidden space.
    Gemma4UnifiedAudioProjection,
    DiffusionSelfConditionPreNorm,
    DiffusionSelfConditionGate,
    DiffusionSelfConditionUp,
    DiffusionSelfConditionDown,
    /// EmbeddingGemma sentence-transformers Dense projection 1 (hidden → 4*hidden,
    /// no bias, identity activation). Applied after mean pooling.
    EmbeddingDense0,
    /// EmbeddingGemma sentence-transformers Dense projection 2 (4*hidden → hidden,
    /// no bias, identity activation). Applied after `EmbeddingDense0`, before L2 norm.
    EmbeddingDense1,
    /// Unlimited-OCR linear projector weight (2048 → hidden).
    UnlimitedOcrProjector,
    /// Unlimited-OCR image newline embedding.
    UnlimitedOcrImageNewline,
    /// Unlimited-OCR view separator embedding.
    UnlimitedOcrViewSeparator,
    /// DeepSeek V4 fused KV projection (`attn.wkv`), replacing the V3 split
    /// kv_a/kv_b MLA projections.
    AttentionKv,
    /// DeepSeek V4 RMSNorm over the per-head KV output of `attn.wkv`.
    AttentionKvNorm,
    /// DeepSeek V4 grouped output down-projection (`attn.wo_a`, per o_groups).
    AttentionOutA,
    /// DeepSeek V4 output LoRA up-projection (`attn.wo_b`, o_lora_rank → hidden).
    AttentionOutB,
    /// DeepSeek V4 hyper-connection attention-branch coefficients (`hc_attn_fn`).
    HcAttnFn,
    /// DeepSeek V4 hyper-connection attention-branch base stream (`hc_attn_base`).
    HcAttnBase,
    /// DeepSeek V4 hyper-connection attention-branch scale (`hc_attn_scale`).
    HcAttnScale,
    /// DeepSeek V4 hyper-connection FFN-branch coefficients (`hc_ffn_fn`).
    HcFfnFn,
    /// DeepSeek V4 hyper-connection FFN-branch base stream (`hc_ffn_base`).
    HcFfnBase,
    /// DeepSeek V4 hyper-connection FFN-branch scale (`hc_ffn_scale`).
    HcFfnScale,
    /// DeepSeek V4 root-level hyper-connection head coefficients (`hc_head_fn`).
    HcHeadFn,
    /// DeepSeek V4 root-level hyper-connection head base stream (`hc_head_base`).
    HcHeadBase,
    /// DeepSeek V4 root-level hyper-connection head scale (`hc_head_scale`).
    HcHeadScale,
    /// DeepSeek V4 sliding-window compressor KV projection (`attn.compressor.wkv`).
    CompressorKv,
    /// DeepSeek V4 sliding-window compressor gate (`attn.compressor.wgate`).
    CompressorGate,
    /// DeepSeek V4 compressor absolute positional embedding (`attn.compressor.ape`).
    CompressorApe,
    /// DeepSeek V4 compressor RMSNorm (`attn.compressor.norm`).
    CompressorNorm,
    /// DeepSeek V4 sparse-indexer per-token score projection (`attn.indexer.weights_proj`).
    IndexerProj,
    /// DeepSeek V4 sparse-indexer query up-projection (`attn.indexer.wq_b`).
    IndexerQb,
    /// DeepSeek V4 indexer compressor KV projection (`attn.indexer.compressor.wkv`).
    IndexerCompressorKv,
    /// DeepSeek V4 indexer compressor gate (`attn.indexer.compressor.wgate`).
    IndexerCompressorGate,
    /// DeepSeek V4 indexer compressor positional embedding (`attn.indexer.compressor.ape`).
    IndexerCompressorApe,
    /// DeepSeek V4 indexer compressor RMSNorm (`attn.indexer.compressor.norm`).
    IndexerCompressorNorm,
    /// DeepSeek V4 hash-routing token→expert table (`ffn.gate.tid2eid`, I32),
    /// present on the first `num_hash_layers` MoE layers.
    FfnGateTid2Eid,
    /// DeepSeek V4 MTP embedding projection (`nextn.e_proj` / `mtp.N.e_proj`).
    NextnEproj,
    /// DeepSeek V4 MTP hidden projection (`nextn.h_proj` / `mtp.N.h_proj`).
    NextnHproj,
    /// DeepSeek V4 MTP fused embedding+hidden projection (`nextn.eh_proj`).
    NextnEhProj,
    /// DeepSeek V4 MTP embedding RMSNorm (`nextn.enorm`).
    NextnEnorm,
    /// DeepSeek V4 MTP hidden RMSNorm (`nextn.hnorm`).
    NextnHnorm,
    /// DeepSeek V4 MTP shared-head RMSNorm (`nextn.shared_head_norm` / `mtp.N.norm`).
    NextnSharedHeadNorm,
    /// DeepSeek V4 MTP shared token embedding (`nextn.embed_tokens`).
    NextnEmbedTokens,
    /// DeepSeek V4 MTP shared LM head (`nextn.shared_head_head`).
    NextnSharedHeadHead,
    /// DeepSeek V4 MTP hyper-connection head coefficients (`mtp.N.hc_head_fn`).
    /// Distinct from the target root `HcHeadFn` so conversion cannot collide.
    NextnHcHeadFn,
    /// DeepSeek V4 MTP hyper-connection head base stream (`mtp.N.hc_head_base`).
    NextnHcHeadBase,
    /// DeepSeek V4 MTP hyper-connection head scale (`mtp.N.hc_head_scale`).
    NextnHcHeadScale,
    /// Qwen3-VL vision patch embed projection (visual.patch_embed.proj).
    Qwen3VlVisionPatchEmbed,
    /// Qwen3-VL vision spatial-merge projector (visual.merger).
    Qwen3VlVisionMerger,
    /// Qwen3-VL vision transformer block attention qkv (layer-indexed).
    Qwen3VlVisionLayerQkv,
    /// Qwen3-VL vision transformer block attention proj (layer-indexed).
    Qwen3VlVisionLayerProj,
    /// Qwen3-VL vision transformer block norm1 weight (layer-indexed).
    Qwen3VlVisionLayerNorm1,
    /// Qwen3-VL vision transformer block norm2 weight (layer-indexed).
    Qwen3VlVisionLayerNorm2,
    /// Qwen3-VL vision transformer block MLP fc1 (layer-indexed).
    Qwen3VlVisionLayerFc1,
    /// Qwen3-VL vision transformer block MLP fc2 (layer-indexed).
    Qwen3VlVisionLayerFc2,
    FinalNorm,
    LmHead,
    RopeFreqs,
    /// Catch-all for extension roles (e.g. MTP sidecar tensors) not yet enumerated here.
    #[serde(other)]
    Other,
}

impl NativeTensorRole {
    const fn requires_layer_index(self) -> bool {
        matches!(
            self,
            Self::AttentionNorm
                | Self::AttentionPostNorm
                | Self::AttentionQNorm
                | Self::AttentionKNorm
                | Self::AttentionQ
                | Self::AttentionK
                | Self::AttentionV
                | Self::AttentionQkvPacked
                | Self::AttentionQa
                | Self::AttentionQaNorm
                | Self::AttentionQb
                | Self::AttentionKvA
                | Self::AttentionKvB
                | Self::AttentionKvANorm
                | Self::AttentionEmbedQ
                | Self::AttentionUnembedOut
                | Self::AttentionOutputGate
                | Self::AttentionO
                | Self::LinearAttentionInProjQkv
                | Self::LinearAttentionInProjQkvz
                | Self::LinearAttentionInProjZ
                | Self::LinearAttentionInProjA
                | Self::LinearAttentionInProjB
                | Self::LinearAttentionInProjBa
                | Self::LinearAttentionConv1d
                | Self::LinearAttentionDtBias
                | Self::LinearAttentionALog
                | Self::LinearAttentionNorm
                | Self::LinearAttentionOutProj
                | Self::FfnNorm
                | Self::FfnNorm2
                | Self::FfnPostNorm
                | Self::FfnPostNorm1
                | Self::FfnPostNorm2
                | Self::FfnGateInp
                | Self::FfnGateInpScale
                | Self::FfnGateInpExpertScale
                | Self::FfnGateInpCorrectionBias
                | Self::FfnGate
                | Self::FfnUp
                | Self::FfnGateUpPacked
                | Self::FfnSharedExpertGateInp
                | Self::FfnSharedExpertGate
                | Self::FfnSharedExpertUp
                | Self::FfnSharedExpertDown
                | Self::FfnGateExps
                | Self::FfnUpExps
                | Self::FfnGateUpExpsPacked
                | Self::FfnDown
                | Self::FfnDownExps
                | Self::FfnDownExpsScale
                | Self::AttnSink
                | Self::AttentionKv
                | Self::AttentionKvNorm
                | Self::AttentionOutA
                | Self::AttentionOutB
                | Self::HcAttnFn
                | Self::HcAttnBase
                | Self::HcAttnScale
                | Self::HcFfnFn
                | Self::HcFfnBase
                | Self::HcFfnScale
                | Self::CompressorKv
                | Self::CompressorGate
                | Self::CompressorApe
                | Self::CompressorNorm
                | Self::IndexerProj
                | Self::IndexerQb
                | Self::IndexerCompressorKv
                | Self::IndexerCompressorGate
                | Self::IndexerCompressorApe
                | Self::IndexerCompressorNorm
                | Self::FfnGateTid2Eid
                | Self::FfnGateUpExpsMxfp4Blocks
                | Self::FfnGateUpExpsMxfp4Scales
                | Self::FfnDownExpsMxfp4Blocks
                | Self::FfnDownExpsMxfp4Scales
                | Self::LayerScalar
                | Self::PerLayerInputGate
                | Self::PerLayerInputProjection
                | Self::PerLayerInputPostNorm
                | Self::Qwen3VlVisionLayerQkv
                | Self::Qwen3VlVisionLayerProj
                | Self::Qwen3VlVisionLayerNorm1
                | Self::Qwen3VlVisionLayerNorm2
                | Self::Qwen3VlVisionLayerFc1
                | Self::Qwen3VlVisionLayerFc2
        )
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeLinearAttentionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_attention_interval: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_value_heads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_key_heads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_head_dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_head_dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conv_kernel_dim: Option<u32>,
}

impl NativeLinearAttentionConfig {
    pub fn is_enabled(&self) -> bool {
        self.full_attention_interval.is_some()
            || self.num_value_heads.is_some()
            || self.num_key_heads.is_some()
            || self.key_head_dim.is_some()
            || self.value_head_dim.is_some()
            || self.conv_kernel_dim.is_some()
    }

    pub fn is_disabled(&self) -> bool {
        !self.is_enabled()
    }

    pub fn resolved_full_attention_interval(&self, model_family: &str) -> Option<u32> {
        self.full_attention_interval.or_else(|| {
            let is_hybrid_family = matches!(model_family, "qwen3_5" | "qwen3_next" | "minicpmv4_6");
            (self.is_enabled() && is_hybrid_family)
                .then_some(QWEN3_5_DEFAULT_FULL_ATTENTION_INTERVAL)
        })
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct NativeMlaAttentionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q_lora_rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_lora_rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qk_nope_head_dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qk_rope_head_dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_head_dim: Option<u32>,
}

impl NativeMlaAttentionConfig {
    pub fn is_enabled(&self) -> bool {
        self.q_lora_rank.is_some()
            || self.kv_lora_rank.is_some()
            || self.qk_nope_head_dim.is_some()
            || self.qk_rope_head_dim.is_some()
            || self.value_head_dim.is_some()
    }

    pub fn is_disabled(&self) -> bool {
        !self.is_enabled()
    }
}

/// DeepSeek V4 (Flash) attention geometry.
///
/// V4 drops the V3 MLA keys (`kv_lora_rank`, `qk_nope_head_dim`, `v_head_dim`):
/// a fused `attn.wkv` projection feeds per-head K/V directly (single KV head),
/// and the output projection is a grouped LoRA pair (`wo_a` per `o_groups`,
/// `wo_b` from `o_lora_rank`). Do not reuse [`NativeMlaAttentionConfig`].
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct NativeDeepseekV4AttentionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qk_rope_head_dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q_lora_rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub o_lora_rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub o_groups: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_topk: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_n_heads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_head_dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compress_rope_theta: Option<u32>,
    /// V4 attention layers carry a learned per-head attention sink
    /// (`attn.attn_sink`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub has_attn_sinks: bool,
}

impl NativeDeepseekV4AttentionConfig {
    pub fn is_enabled(&self) -> bool {
        self.head_dim.is_some()
            || self.qk_rope_head_dim.is_some()
            || self.q_lora_rank.is_some()
            || self.o_lora_rank.is_some()
            || self.o_groups.is_some()
            || self.index_topk.is_some()
            || self.index_n_heads.is_some()
            || self.index_head_dim.is_some()
            || self.compress_rope_theta.is_some()
            || self.has_attn_sinks
    }

    pub fn is_disabled(&self) -> bool {
        !self.is_enabled()
    }
}

/// DeepSeek V4 (Flash) architecture parameters that have no home in the
/// generic manifest fields: per-layer compressor ratios, hyper-connection
/// (HC) constants, hash routing, and the routing scoring function.
///
/// The MoE shape (expert counts, shared experts, scaling factor) stays in
/// [`NativeMoeConfig`]; note V4 routing is `scoring_func`-based (e.g.
/// "sqrtsoftplus"), **not** the V3 sigmoid routing, so
/// `NativeMoeConfig::sigmoid_routing` must remain false for V4 manifests.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct NativeDeepseekV4Config {
    #[serde(
        default,
        skip_serializing_if = "NativeDeepseekV4AttentionConfig::is_disabled"
    )]
    pub attention: NativeDeepseekV4AttentionConfig,
    /// Per-layer compressor ratios (values 0 / 4 / 128; 0 = uncompressed).
    /// Empty when not configured; otherwise one entry per layer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compress_ratios: Vec<u32>,
    /// Hyper-connection stream multiplier (`hc_mult`, e.g. 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hc_mult: Option<u32>,
    /// Sinkhorn iterations for the HC mixing coefficients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hc_sinkhorn_iters: Option<u32>,
    /// Epsilon for the HC Sinkhorn normalisation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hc_eps: Option<f32>,
    /// Number of leading MoE layers that route via the hash table
    /// (`ffn.gate.tid2eid`) instead of the learned gate + correction bias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_hash_layers: Option<u32>,
    /// Number of MTP (nextn) predictor layers stacked after the main layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_nextn_predict_layers: Option<u32>,
    /// Routing scoring function (e.g. "sqrtsoftplus").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scoring_func: Option<String>,
    /// SwiGLU clamp limit applied in the expert/shared-expert FFNs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swiglu_limit: Option<f32>,
}

impl NativeDeepseekV4Config {
    pub fn is_enabled(&self) -> bool {
        self.attention.is_enabled()
            || !self.compress_ratios.is_empty()
            || self.hc_mult.is_some()
            || self.hc_sinkhorn_iters.is_some()
            || self.hc_eps.is_some()
            || self.num_hash_layers.is_some()
            || self.num_nextn_predict_layers.is_some()
            || self.scoring_func.is_some()
            || self.swiglu_limit.is_some()
    }

    pub fn is_disabled(&self) -> bool {
        !self.is_enabled()
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct NativeMoeConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experts_per_token: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_intermediate_size: Option<u32>,
    /// MoE every N layers (1 = every layer, 0 = use GlmRouter dispatch). Default 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_freq: Option<u32>,
    /// First K layers are dense (DeepSeek: first_k_dense_replace). Default 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_dense_layers: Option<u32>,
    /// Number of shared (always-active) experts. Default 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_expert_count: Option<u32>,
    /// Use sigmoid routing instead of softmax (DeepSeek V3). Default false.
    #[serde(default)]
    pub sigmoid_routing: bool,
    /// Scale factor applied to selected expert weights after routing (DeepSeek V3: 2.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routed_scaling_factor: Option<f32>,
    /// Number of expert groups for group-based top-k selection (DeepSeek V3: 8).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_group: Option<u32>,
    /// Number of groups to retain after group scoring (DeepSeek V3: 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topk_group: Option<u32>,
}

impl NativeMoeConfig {
    pub fn is_enabled(&self) -> bool {
        self.expert_count.is_some()
            || self.experts_per_token.is_some()
            || self.expert_intermediate_size.is_some()
    }

    pub fn is_disabled(&self) -> bool {
        !self.is_enabled()
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct NativeGlmRouterConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_dense_layer_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routed_scaling_factor: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_group: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topk_group: Option<u32>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub has_shared_experts: bool,
}

impl NativeGlmRouterConfig {
    pub fn is_enabled(&self) -> bool {
        self.first_dense_layer_count.is_some()
            || self.routed_scaling_factor.is_some()
            || self.n_group.is_some()
            || self.topk_group.is_some()
            || self.has_shared_experts
    }

    pub fn is_disabled(&self) -> bool {
        !self.is_enabled()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeTensorSpec {
    pub name: String,
    pub role: NativeTensorRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_index: Option<u32>,
    pub dtype: NativeTensorDataType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_tensor_type: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub source_quantized: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<NativeTensorQuantization>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantized_source: Option<NativeQuantizedTensorSource>,
    pub shape: Vec<u64>,
    pub file: PathBuf,
    pub offset_bytes: u64,
    pub length_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeTensorQuantization {
    pub mode: String,
    pub group_size: u32,
    pub bits: u32,
}

impl Default for NativeTensorQuantization {
    fn default() -> Self {
        Self {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 4,
        }
    }
}

/// Per-layer KV-cache quantization table lifted from a converted checkpoint's
/// `axquant_runtime.json` (`kv_cache` block, schema `axquant.runtime.v1`).
/// `layer_bits[i]` / `layer_group_sizes[i]` apply to layer `i`; bits 16 marks a
/// full-precision layer.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct KvCacheQuantizationManifest {
    pub layer_bits: Vec<u32>,
    pub layer_group_sizes: Vec<u32>,
    pub basis: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeQuantizedTensorSource {
    pub format: String,
    pub file: PathBuf,
    #[serde(default)]
    pub offset_bytes: u64,
    pub length_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeSourceQuantization {
    pub format: String,
    #[serde(default)]
    pub tensor_type_counts: BTreeMap<String, u32>,
    #[serde(default)]
    pub quantized_tensor_count: u32,
    #[serde(default)]
    pub contains_quantized_tensors: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeRuntimeStatus {
    #[serde(default = "default_runtime_ready")]
    pub ready: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

impl Default for NativeRuntimeStatus {
    fn default() -> Self {
        Self {
            ready: true,
            blockers: Vec::new(),
            notes: Vec::new(),
        }
    }
}

impl NativeRuntimeStatus {
    pub fn ready_without_details(&self) -> bool {
        self.ready && self.blockers.is_empty() && self.notes.is_empty()
    }
}

fn default_runtime_ready() -> bool {
    true
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Identifies the on-disk weight convention. `mlx-community` checkpoints come
/// pre-sanitized; raw HuggingFace checkpoints need two transforms (norm delta
/// +1.0, conv1d axis swap) that the weight loader applies at load time when
/// this field is set to `HfToMlx`.
///
/// Some checkpoints (e.g. Qwen3-Coder-Next) are partially sanitized: the
/// conv1d axes were already swapped by `mlx_lm.convert` to the MLX layout
/// `(out, kernel, in)` but the RMSNorm weights were NOT lifted from their
/// zero-centered delta representation. Use `HfNormOnly` for these.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightSanitize {
    /// Weights are already in MLX layout (the mlx-community convention).
    /// No transforms are applied at load time. This is the default to keep
    /// existing manifests backward-compatible.
    #[default]
    None,
    /// Weights are in the raw HuggingFace convention. The loader will:
    /// - add 1.0 to every RMSNorm-variant norm weight (these are stored as
    ///   zero-centered deltas in HF format)
    /// - swap axes (2, 1) on conv1d projection weights (HF stores them in
    ///   a different axis order than MLX expects)
    HfToMlx,
    /// Partially-sanitized HuggingFace convention: conv1d axes are already in
    /// MLX layout `(out, kernel, in)` but RMSNorm weights are still stored as
    /// zero-centered deltas. The loader adds 1.0 to norm weights only; it does
    /// NOT re-swap the conv1d axes.
    HfNormOnly,
    /// Like `HfNormOnly`, but only the per-layer sandwich norms carry
    /// zero-centered deltas; the final `model.norm` is already a plain gain
    /// and must NOT get `+1` (Muse Glimmer: `MuseRmsNorm::centered` for the
    /// four per-layer norms vs `standard` for the final norm).
    HfLayerNormsOnly,
}

impl WeightSanitize {
    pub fn is_none(&self) -> bool {
        matches!(self, WeightSanitize::None)
    }
}

/// Sampling strategy for DiffusionGemma denoising steps.
///
/// Controls how canvas positions are accepted or renoised during the
/// iterative denoise loop. The choice of sampler has a large impact on
/// throughput: confidence-threshold is 4–5× faster than entropy-bound
/// with equivalent output quality (per mlx-optiq benchmarks).
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeDiffusionSampler {
    /// Entropy-bound sampling: sort positions by entropy ascending, accept
    /// greedily until cumulative entropy exceeds the budget. Uses argsort +
    /// cumsum + inverse-sort per step (the DiffusionGemma paper default).
    EntropyBound,
    /// Confidence-threshold sampling: accept a position when its peak
    /// softmax probability exceeds a fixed threshold (default 0.9). Uses
    /// only argmax + take_along_axis + one comparison — no sorting.
    /// **Default: 4–5× faster with equivalent quality.**
    #[default]
    ConfidenceThreshold,
}

/// Diffusion-specific generation parameters (DiffusionGemma).
///
/// DiffusionGemma generates tokens via block-autoregressive discrete diffusion:
/// a 256-token canvas is initialized randomly, iteratively denoised with
/// bidirectional attention, then committed with a causal encoder pass.
/// All fields are optional; the runtime applies reference defaults when absent.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct NativeDiffusionConfig {
    /// Number of tokens generated per diffusion block (default 256).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canvas_size: Option<u32>,
    /// Maximum denoising steps per block before forced convergence (default 48).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_denoise_steps: Option<u32>,
    /// Enable self-conditioning feedback between denoising steps (default true).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub self_conditioning: Option<bool>,
    /// Entropy bound for position acceptance during denoising (default 0.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entropy_bound: Option<f32>,
    /// Mean entropy threshold for convergence detection (default 0.02).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entropy_threshold: Option<f32>,
    /// Consecutive stable argmax steps required for convergence (default 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub convergence_steps: Option<u32>,
    /// Temperature schedule start (high, for exploration; default 0.8).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_start: Option<f32>,
    /// Temperature schedule end (low, for locking final tokens; default 0.4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_end: Option<f32>,
    /// Steps between convergence checks (default 4). Non-check steps skip
    /// argmax stability and mean-entropy materialisation to reduce GPU→CPU syncs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub convergence_check_interval: Option<u32>,
    /// Update-rate threshold for adaptive convergence (default 0.075 = 7.5%).
    /// When the fraction of positions still changing drops below this, the model
    /// has converged regardless of absolute entropy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance_rate_threshold: Option<f32>,
    /// Sampling strategy for denoising steps (default: confidence_threshold).
    /// `confidence_threshold` is 4–5× faster with equivalent quality.
    /// Set to `entropy_bound` to use the original paper sampler.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler: Option<NativeDiffusionSampler>,
    /// Confidence threshold for `confidence_threshold` sampler (default 0.9).
    /// Positions with peak softmax probability >= this value are accepted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence_threshold: Option<f32>,
}

impl NativeDiffusionConfig {
    pub fn is_enabled(&self) -> bool {
        self.canvas_size.is_some()
            || self.max_denoise_steps.is_some()
            || self.self_conditioning.is_some()
            || self.entropy_bound.is_some()
            || self.entropy_threshold.is_some()
            || self.convergence_steps.is_some()
            || self.temperature_start.is_some()
            || self.temperature_end.is_some()
            || self.convergence_check_interval.is_some()
            || self.acceptance_rate_threshold.is_some()
            || self.sampler.is_some()
            || self.confidence_threshold.is_some()
    }

    pub fn is_disabled(&self) -> bool {
        !self.is_enabled()
    }
}

/// Provenance for tensors skipped during convert (WS-C1 / R-C1).
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
pub struct DroppedTensorsProvenance {
    #[serde(default)]
    pub count: u64,
    #[serde(default)]
    pub media_role_hits: u64,
    /// Sample of dropped tensor names (bounded).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub names_sample: Vec<String>,
}

impl DroppedTensorsProvenance {
    pub fn is_empty(&self) -> bool {
        self.count == 0 && self.media_role_hits == 0 && self.names_sample.is_empty()
    }

    pub fn has_media_role_drops(&self) -> bool {
        self.media_role_hits > 0
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NativeModelManifest {
    pub schema_version: String,
    pub model_family: String,
    pub tensor_format: NativeTensorFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_quantization: Option<NativeSourceQuantization>,
    #[serde(
        default,
        skip_serializing_if = "NativeRuntimeStatus::ready_without_details"
    )]
    pub runtime_status: NativeRuntimeStatus,
    pub layer_count: u32,
    pub hidden_size: u32,
    #[serde(default)]
    pub intermediate_size: u32,
    pub attention_head_count: u32,
    pub attention_head_dim: u32,
    pub kv_head_count: u32,
    pub vocab_size: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<u32>,
    /// Rope theta for sliding-window attention (SWA) layers in ISWA models (e.g. Gemma4).
    /// Corresponds to GGUF key `{arch}.rope.freq_base_swa`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta_swa: Option<u32>,
    /// RoPE scaling strategy. Supported values: "llama3", "linear", "dynamic".
    /// Absent for standard (unscaled) RoPE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling_type: Option<String>,
    /// RoPE scaling factor (divisor applied to low-frequency components).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling_factor: Option<f32>,
    /// LLaMA-3 low-frequency correction factor (default 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_low_freq_factor: Option<f32>,
    /// LLaMA-3 high-frequency correction factor (default 4.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_high_freq_factor: Option<f32>,
    /// LLaMA-3 original training context length for wavelen boundary computation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_original_context_len: Option<u32>,
    /// YaRN `beta_fast` (high-frequency correction boundary; default 32).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_beta_fast: Option<f32>,
    /// YaRN `beta_slow` (low-frequency correction boundary; default 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_beta_slow: Option<f32>,
    /// LLaMA-4 iRoPE: every N-th layer has no RoPE (N=4 for LLaMA4 Scout/Maverick).
    /// 0 means all layers use RoPE.
    #[serde(default)]
    pub no_rope_layer_interval: u32,
    /// LLaMA-4 attention temperature floor scale (default 8192).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attn_temperature_floor: Option<u32>,
    /// LLaMA-4 attention temperature scale factor (default 0.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attn_temperature_scale: Option<f32>,
    /// Dense FFN intermediate size for models where MoE and dense layers use different sizes (LLaMA4).
    /// 0 means use `intermediate_size` for both.
    #[serde(default)]
    pub intermediate_size_mlp: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_pre_attn_scalar: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_logit_softcap: Option<u32>,
    #[serde(default)]
    pub attn_output_gate: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f32>,
    /// Epsilon used by RMSNorm operations. When absent, runtimes may apply
    /// architecture-specific compatibility defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rms_norm_eps: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attention_value_from_key_layers: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attention_v_norm_no_scale_layers: Vec<u32>,
    /// Head dimension for full-attention layers in interleaved SWA models (e.g. Gemma4).
    /// Sliding-attention layers use `attention_head_dim`; full-attention layers use this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_head_dim: Option<u32>,
    /// KV head count for full-attention layers in interleaved SWA models (e.g. Gemma4).
    ///
    /// Older manifests omit this field and preserve the legacy constant-total-KV-width
    /// rule: `kv_head_count * attention_head_dim` is divided by `global_head_dim`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_kv_head_count: Option<u32>,
    /// Sliding-window size for SWA layers (None = global attention).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window_size: Option<u32>,
    /// Per-layer type annotations ("sliding_attention" / "full_attention").
    /// Empty for homogeneous models; populated for interleaved-SWA models (Gemma4).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layer_types: Vec<String>,
    /// Maps KV-shared layer index → source layer index that supplies K/V.
    /// For layers absent from this map, K/V is computed from the layer's own weights.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub kv_shared_source_layers: BTreeMap<u32, u32>,
    /// Final-logit softcapping: apply `tanh(x / cap) * cap` after lm_head (Gemma4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f32>,
    /// Scalar multiplied into logits after lm_head, BEFORE
    /// `final_logit_softcapping` (Muse Glimmer `output_multiplier`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logits_scale: Option<f32>,
    /// Multiplier folded into the SDPA query scale on top of
    /// `head_dim^-0.5` (Muse Glimmer `qk_scale_factor`: scale =
    /// `head_dim^-0.5 * qk_scale_factor` on every layer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_scale_multiplier: Option<f32>,
    /// RMSNorm eps for the post-attention and post-feedforward sandwich
    /// norms when it differs from `rms_norm_eps` (Muse Glimmer
    /// `post_norm_eps` = 1e-8; the input/pre-FFN norms keep `rms_norm_eps`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_norm_eps: Option<f32>,
    /// Scale applied to token embeddings before the first layer (Gemma4: sqrt(hidden_size)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_states_scale: Option<f32>,
    /// When true, normalise the selected top-k MoE weights to sum to 1 (Qwen3 MoE).
    #[serde(default)]
    pub moe_norm_topk_prob: bool,
    /// Dimension of per-layer token embeddings (Gemma4 2B/4B). 0 = feature disabled.
    #[serde(default)]
    pub hidden_size_per_layer_input: u32,
    /// Vocab size for the per-layer embedding table (Gemma4 2B/4B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size_per_layer_input: Option<u32>,
    #[serde(
        default,
        skip_serializing_if = "NativeLinearAttentionConfig::is_disabled"
    )]
    pub linear_attention: NativeLinearAttentionConfig,
    #[serde(default, skip_serializing_if = "NativeMlaAttentionConfig::is_disabled")]
    pub mla_attention: NativeMlaAttentionConfig,
    #[serde(default, skip_serializing_if = "NativeMoeConfig::is_disabled")]
    pub moe: NativeMoeConfig,
    #[serde(default, skip_serializing_if = "NativeGlmRouterConfig::is_disabled")]
    pub glm_router: NativeGlmRouterConfig,
    /// DeepSeek V4 (Flash) architecture parameters. Disabled for all other
    /// model families.
    #[serde(default, skip_serializing_if = "NativeDeepseekV4Config::is_disabled")]
    pub deepseek_v4: NativeDeepseekV4Config,
    /// Weight on-disk convention. Defaults to `None` (mlx-community
    /// pre-sanitized layout) so existing manifests deserialize unchanged.
    /// Set to `hf_to_mlx` in raw HuggingFace checkpoints' manifests to
    /// have the loader apply the norm-delta and conv1d-axis transforms.
    #[serde(default, skip_serializing_if = "WeightSanitize::is_none")]
    pub weight_sanitize: WeightSanitize,
    /// Token ID for `<think>` (Qwen3 reasoning models: 151668).
    /// When present, n-gram acceleration gates drafting to only run inside
    /// `<think>...</think>` blocks, where repetition patterns are much denser.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub think_start_token_id: Option<u32>,
    /// Token ID for `</think>` (Qwen3 reasoning models: 151669).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub think_end_token_id: Option<u32>,
    /// Diffusion generation parameters (DiffusionGemma).
    /// Disabled for all non-diffusion model families.
    #[serde(default, skip_serializing_if = "NativeDiffusionConfig::is_disabled")]
    pub diffusion: NativeDiffusionConfig,
    /// Tensors skipped at convert time (WS-C1). Default empty for legacy manifests.
    #[serde(default, skip_serializing_if = "DroppedTensorsProvenance::is_empty")]
    pub dropped_tensors: DroppedTensorsProvenance,
    /// Per-layer KV-cache quantization table lifted from `axquant_runtime.json`
    /// at convert time. Absent for manifests without KV-cache quantization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_cache_quantization: Option<KvCacheQuantizationManifest>,
    pub tensors: Vec<NativeTensorSpec>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeModelArtifacts {
    root_dir: PathBuf,
    manifest: NativeModelManifest,
}

impl NativeModelManifest {
    /// Structural architecture view (ADR-038). Derived; not a second on-disk format.
    pub fn architecture_spec(&self) -> crate::architecture::ArchitectureSpec {
        crate::architecture::ArchitectureSpec::from_manifest(self)
    }

    /// Generation paradigm derived from this manifest (ADR-038).
    pub fn generation_kind(&self) -> crate::generation::GenerationKind {
        crate::generation::GenerationKind::from_manifest(self)
    }
}

impl NativeModelArtifacts {
    /// Build artifacts directly from a pre-parsed manifest and root directory.
    /// Used by the GGUF loader to bypass the JSON manifest file.
    pub fn from_manifest_and_root(
        root_dir: PathBuf,
        manifest: NativeModelManifest,
    ) -> Result<Self, NativeModelError> {
        validate_native_model_manifest(&root_dir, &manifest)?;
        Ok(Self { root_dir, manifest })
    }

    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, NativeModelError> {
        let root_dir = path.as_ref().to_path_buf();
        let manifest_path = root_dir.join(AX_NATIVE_MODEL_MANIFEST_FILE);
        let bytes = fs::read(&manifest_path).map_err(|source| NativeModelError::ReadManifest {
            path: manifest_path.clone(),
            source,
        })?;
        let manifest = serde_json::from_slice::<NativeModelManifest>(&bytes).map_err(|source| {
            NativeModelError::ParseManifest {
                path: manifest_path.clone(),
                source,
            }
        })?;

        validate_native_model_manifest(&root_dir, &manifest)?;

        Ok(Self { root_dir, manifest })
    }

    /// Load artifacts from `path`, auto-generating `model-manifest.json` from
    /// a raw HuggingFace / MLX snapshot (`config.json` + safetensors headers)
    /// when the manifest file is absent.
    ///
    /// `from_dir` semantics are unchanged everywhere else: an unparsable or
    /// invalid manifest still fails closed, and a directory that is not a
    /// convertible HF snapshot surfaces the convert failure via
    /// `NativeModelError::AutoConvert`.
    pub fn from_dir_or_convert(path: impl AsRef<Path>) -> Result<Self, NativeModelError> {
        let root_dir = path.as_ref().to_path_buf();
        match Self::from_dir(&root_dir) {
            Err(NativeModelError::ReadManifest { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                crate::convert::ensure_manifest_for_hf_model_dir(&root_dir).map_err(|source| {
                    NativeModelError::AutoConvert {
                        path: root_dir.clone(),
                        source,
                    }
                })?;
                Self::from_dir(&root_dir)
            }
            result => result,
        }
    }

    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    pub fn manifest(&self) -> &NativeModelManifest {
        &self.manifest
    }

    pub fn tensor_specs(&self) -> &[NativeTensorSpec] {
        &self.manifest.tensors
    }

    pub fn global_tensor(&self, role: NativeTensorRole) -> Option<&NativeTensorSpec> {
        self.manifest
            .tensors
            .iter()
            .find(|tensor| tensor.role == role && tensor.layer_index.is_none())
    }

    pub fn layer_tensor(
        &self,
        layer_index: u32,
        role: NativeTensorRole,
    ) -> Option<&NativeTensorSpec> {
        self.manifest
            .tensors
            .iter()
            .find(|tensor| tensor.role == role && tensor.layer_index == Some(layer_index))
    }

    pub fn resolve_tensor_path(&self, tensor: &NativeTensorSpec) -> PathBuf {
        self.root_dir.join(&tensor.file)
    }

    pub fn summary(&self) -> NativeModelArtifactsSummary {
        let is_hybrid_attention = self.manifest.linear_attention.is_enabled();
        NativeModelArtifactsSummary {
            model_family: self.manifest.model_family.clone(),
            tensor_format: self.manifest.tensor_format,
            source_quantization: self.manifest.source_quantization.clone(),
            runtime_status: self.manifest.runtime_status.clone(),
            layer_count: self.manifest.layer_count,
            tensor_count: self.manifest.tensors.len() as u32,
            tie_word_embeddings: self.manifest.tie_word_embeddings,
            is_moe: self.manifest.moe.is_enabled(),
            is_hybrid_attention,
            hybrid_full_attention_interval: is_hybrid_attention
                .then(|| {
                    self.manifest
                        .linear_attention
                        .resolved_full_attention_interval(&self.manifest.model_family)
                })
                .flatten(),
            mla_kv_latent_dim: self
                .manifest
                .mla_attention
                .is_enabled()
                .then_some(self.manifest.mla_attention.kv_lora_rank)
                .flatten(),
            moe_active_experts: self
                .manifest
                .moe
                .is_enabled()
                .then_some(self.manifest.moe.experts_per_token)
                .flatten(),
        }
    }

    pub fn layer_uses_attention_value_from_key(&self, layer_index: u32) -> bool {
        self.manifest
            .attention_value_from_key_layers
            .contains(&layer_index)
    }

    pub fn layer_uses_attention_v_norm_no_scale(&self, layer_index: u32) -> bool {
        self.manifest
            .attention_v_norm_no_scale_layers
            .contains(&layer_index)
    }

    pub fn linear_attention_config(&self) -> Option<&NativeLinearAttentionConfig> {
        self.manifest
            .linear_attention
            .is_enabled()
            .then_some(&self.manifest.linear_attention)
    }

    pub fn moe_config(&self) -> Option<&NativeMoeConfig> {
        self.manifest.moe.is_enabled().then_some(&self.manifest.moe)
    }

    /// Returns the number of head dimensions that receive rotary embedding.
    /// When `partial_rotary_factor` is set, only a fraction of head_dim is rotated.
    pub fn rotary_dim(&self) -> usize {
        let head_dim = self.manifest.attention_head_dim as usize;
        if let Some(factor) = self.manifest.partial_rotary_factor {
            let dim = (head_dim as f32 * factor) as usize;
            // Rotary dim must be even; round down to nearest even
            dim & !1
        } else {
            head_dim
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeModelArtifactsSummary {
    pub model_family: String,
    pub tensor_format: NativeTensorFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_quantization: Option<NativeSourceQuantization>,
    #[serde(
        default,
        skip_serializing_if = "NativeRuntimeStatus::ready_without_details"
    )]
    pub runtime_status: NativeRuntimeStatus,
    pub layer_count: u32,
    pub tensor_count: u32,
    pub tie_word_embeddings: bool,
    /// True when the model uses a mixture-of-experts FFN (e.g. Gemma 4, Qwen3-MoE).
    #[serde(default)]
    pub is_moe: bool,
    /// True when the model interleaves linear-attention layers with standard attention
    /// (e.g. Qwen3.5, Qwen3-Next).
    #[serde(default)]
    pub is_hybrid_attention: bool,
    /// For hybrid-attention models: how many layers apart the full-attention layers occur.
    /// None for pure-attention or pure-linear-attention models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hybrid_full_attention_interval: Option<u32>,
    /// For MLA models: latent KV dimension (`kv_lora_rank` in the manifest).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mla_kv_latent_dim: Option<u32>,
    /// For MoE models: active experts selected per token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_active_experts: Option<u32>,
}

#[derive(Debug, Error)]
pub enum NativeModelError {
    #[error("failed to read native model manifest {path}: {source}")]
    ReadManifest {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse native model manifest {path}: {source}")]
    ParseManifest {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid native model manifest: {message}")]
    InvalidManifest { message: String },
    #[error("failed to auto-generate native model manifest for {path}: {source}")]
    AutoConvert {
        path: PathBuf,
        #[source]
        source: crate::convert::ConvertError,
    },
}

pub(crate) fn validate_native_model_manifest(
    root_dir: &Path,
    manifest: &NativeModelManifest,
) -> Result<(), NativeModelError> {
    if manifest.schema_version != AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "schema_version must be {}, got {}",
                AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, manifest.schema_version
            ),
        });
    }
    if manifest.model_family.trim().is_empty() {
        return Err(NativeModelError::InvalidManifest {
            message: "model_family must not be empty".to_string(),
        });
    }
    if !manifest.runtime_status.ready || !manifest.runtime_status.blockers.is_empty() {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "native model manifest is not runtime ready: ready={} blockers={:?}",
                manifest.runtime_status.ready, manifest.runtime_status.blockers
            ),
        });
    }
    if manifest.layer_count == 0
        || manifest.hidden_size == 0
        || manifest.attention_head_count == 0
        || manifest.attention_head_dim == 0
        || manifest.kv_head_count == 0
        || manifest.vocab_size == 0
    {
        return Err(NativeModelError::InvalidManifest {
            message: "layer_count, hidden_size, attention_head_count, attention_head_dim, kv_head_count, and vocab_size must be greater than zero".to_string(),
        });
    }
    if manifest.tensors.is_empty() {
        return Err(NativeModelError::InvalidManifest {
            message: "tensors must not be empty".to_string(),
        });
    }
    validate_kv_cache_quantization(manifest)?;
    validate_manifest_layer_index_list(
        manifest,
        &manifest.attention_value_from_key_layers,
        "attention_value_from_key_layers",
    )?;
    validate_manifest_layer_index_list(
        manifest,
        &manifest.attention_v_norm_no_scale_layers,
        "attention_v_norm_no_scale_layers",
    )?;
    validate_interleaved_attention_metadata(manifest)?;
    if let Some(rope_theta) = manifest.rope_theta {
        if rope_theta == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!("rope_theta must be > 0, got {rope_theta}"),
            });
        }
    }
    if let Some(query_pre_attn_scalar) = manifest.query_pre_attn_scalar {
        if query_pre_attn_scalar == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!("query_pre_attn_scalar must be > 0, got {query_pre_attn_scalar}"),
            });
        }
    }
    if let Some(attention_logit_softcap) = manifest.attention_logit_softcap {
        if attention_logit_softcap == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "attention_logit_softcap must be > 0, got {attention_logit_softcap}"
                ),
            });
        }
    }
    if let Some(factor) = manifest.partial_rotary_factor {
        if factor <= 0.0 || factor > 1.0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!("partial_rotary_factor must be in (0.0, 1.0], got {factor}"),
            });
        }
        let rotary_dim = (manifest.attention_head_dim as f32 * factor) as u32;
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "partial_rotary_factor {factor} yields rotary_dim {rotary_dim} which must be even and > 0"
                ),
            });
        }
    }
    if let Some(eps) = manifest.rms_norm_eps {
        if !eps.is_finite() || eps <= 0.0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!("rms_norm_eps must be finite and > 0, got {eps}"),
            });
        }
    }
    for (value, field_name) in [
        (manifest.final_logit_softcapping, "final_logit_softcapping"),
        (manifest.final_logits_scale, "final_logits_scale"),
        (
            manifest.attention_scale_multiplier,
            "attention_scale_multiplier",
        ),
        (manifest.post_norm_eps, "post_norm_eps"),
    ] {
        if let Some(value) = value
            && (!value.is_finite() || value <= 0.0)
        {
            return Err(NativeModelError::InvalidManifest {
                message: format!("{field_name} must be finite and > 0, got {value}"),
            });
        }
    }
    if manifest.model_family == "muse_glimmer" {
        validate_muse_glimmer_manifest_contract(manifest)?;
    }
    if manifest.linear_attention.is_enabled() {
        require_positive_field(
            manifest
                .linear_attention
                .resolved_full_attention_interval(&manifest.model_family),
            "linear_attention.full_attention_interval",
        )?;
        require_positive_field(
            manifest.linear_attention.num_value_heads,
            "linear_attention.num_value_heads",
        )?;
        require_positive_field(
            manifest.linear_attention.num_key_heads,
            "linear_attention.num_key_heads",
        )?;
        require_positive_field(
            manifest.linear_attention.key_head_dim,
            "linear_attention.key_head_dim",
        )?;
        require_positive_field(
            manifest.linear_attention.value_head_dim,
            "linear_attention.value_head_dim",
        )?;
        require_positive_field(
            manifest.linear_attention.conv_kernel_dim,
            "linear_attention.conv_kernel_dim",
        )?;
        if let (Some(num_value_heads), Some(num_key_heads)) = (
            manifest.linear_attention.num_value_heads,
            manifest.linear_attention.num_key_heads,
        ) {
            if !num_value_heads.is_multiple_of(num_key_heads) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "linear_attention.num_value_heads {} must be divisible by linear_attention.num_key_heads {}",
                        num_value_heads, num_key_heads
                    ),
                });
            }
        }
    }
    if manifest.mla_attention.is_enabled() {
        require_positive_field(
            manifest.mla_attention.q_lora_rank,
            "mla_attention.q_lora_rank",
        )?;
        require_positive_field(
            manifest.mla_attention.kv_lora_rank,
            "mla_attention.kv_lora_rank",
        )?;
        require_positive_field(
            manifest.mla_attention.qk_nope_head_dim,
            "mla_attention.qk_nope_head_dim",
        )?;
        require_positive_field(
            manifest.mla_attention.qk_rope_head_dim,
            "mla_attention.qk_rope_head_dim",
        )?;
        require_positive_field(
            manifest.mla_attention.value_head_dim,
            "mla_attention.value_head_dim",
        )?;
        if let (Some(nope_dim), Some(rope_dim)) = (
            manifest.mla_attention.qk_nope_head_dim,
            manifest.mla_attention.qk_rope_head_dim,
        ) {
            let expected_head_dim = nope_dim.saturating_add(rope_dim);
            if expected_head_dim != manifest.attention_head_dim {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "mla_attention qk_nope_head_dim + qk_rope_head_dim must equal attention_head_dim {}, got {} + {}",
                        manifest.attention_head_dim, nope_dim, rope_dim
                    ),
                });
            }
        }
    }
    if manifest.moe.is_enabled() {
        require_positive_field(manifest.moe.expert_count, "moe.expert_count")?;
        require_positive_field(manifest.moe.experts_per_token, "moe.experts_per_token")?;
        require_positive_field(
            manifest.moe.expert_intermediate_size,
            "moe.expert_intermediate_size",
        )?;
        if let (Some(expert_count), Some(experts_per_token)) =
            (manifest.moe.expert_count, manifest.moe.experts_per_token)
        {
            if experts_per_token > expert_count {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "moe.experts_per_token {} must be <= moe.expert_count {}",
                        experts_per_token, expert_count
                    ),
                });
            }
        }
    }
    if manifest.glm_router.is_enabled() {
        if manifest.glm_router.first_dense_layer_count.is_none() {
            return Err(NativeModelError::InvalidManifest {
                message: "glm_router.first_dense_layer_count must be configured".to_string(),
            });
        }
        if manifest
            .glm_router
            .first_dense_layer_count
            .is_some_and(|count| count > manifest.layer_count)
        {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "glm_router.first_dense_layer_count must be <= layer_count {}, got {}",
                    manifest.layer_count,
                    manifest
                        .glm_router
                        .first_dense_layer_count
                        .unwrap_or_default()
                ),
            });
        }
        match manifest.glm_router.routed_scaling_factor {
            Some(value) if value.is_finite() && value > 0.0 => {}
            Some(value) => {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "glm_router.routed_scaling_factor must be finite and > 0, got {value}"
                    ),
                });
            }
            None => {
                return Err(NativeModelError::InvalidManifest {
                    message: "glm_router.routed_scaling_factor must be configured".to_string(),
                });
            }
        }
        require_positive_field(manifest.glm_router.n_group, "glm_router.n_group")?;
        require_positive_field(manifest.glm_router.topk_group, "glm_router.topk_group")?;
        if let (Some(n_group), Some(topk_group)) =
            (manifest.glm_router.n_group, manifest.glm_router.topk_group)
        {
            if topk_group > n_group {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "glm_router.topk_group {} must be <= glm_router.n_group {}",
                        topk_group, n_group
                    ),
                });
            }
            if manifest
                .moe
                .expert_count
                .is_some_and(|expert_count| expert_count % n_group != 0)
            {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "moe.expert_count must divide evenly across glm_router.n_group {n_group}"
                    ),
                });
            }
        }
    }

    let mut tensor_names = BTreeMap::new();
    let mut layer_roles = BTreeMap::<u32, Vec<NativeTensorRole>>::new();
    let mut global_roles = Vec::new();

    let allow_experimental_3bit =
        std::env::var(AX_ENGINE_3BIT_EXPERIMENTAL_ENV).as_deref() == Ok("1");
    let allow_experimental_2bit =
        std::env::var(AX_ENGINE_2BIT_EXPERIMENTAL_ENV).as_deref() == Ok("1");

    for tensor in &manifest.tensors {
        if tensor.name.trim().is_empty() {
            return Err(NativeModelError::InvalidManifest {
                message: "tensor name must not be empty".to_string(),
            });
        }
        if tensor_names.insert(tensor.name.clone(), ()).is_some() {
            return Err(NativeModelError::InvalidManifest {
                message: format!("duplicate tensor name {}", tensor.name),
            });
        }
        // Safetensors permits rank-0 scalars. Keep structural language roles
        // rank-positive, but allow extension tensors such as Gemma 4 ViT
        // clipping thresholds to preserve their native scalar shape.
        if (tensor.shape.is_empty() && tensor.role != NativeTensorRole::Other)
            || tensor.shape.contains(&0)
        {
            return Err(NativeModelError::InvalidManifest {
                message: format!("tensor {} must have only positive dimensions", tensor.name),
            });
        }
        if tensor.length_bytes == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!("tensor {} must have positive length_bytes", tensor.name),
            });
        }
        validate_tensor_path(root_dir, tensor)?;
        validate_quantized_source_path(root_dir, tensor)?;
        validate_tensor_quantization(
            tensor,
            manifest.tensor_format,
            allow_experimental_3bit,
            allow_experimental_2bit,
        )?;

        if tensor.role == NativeTensorRole::Other {
            // Extension/sidecar roles (e.g. MTP): skip layer_index validation entirely.
        } else if tensor.role.requires_layer_index() {
            let Some(layer_index) = tensor.layer_index else {
                return Err(NativeModelError::InvalidManifest {
                    message: format!("tensor {} requires layer_index", tensor.name),
                });
            };
            if layer_index >= manifest.layer_count {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "tensor {} layer_index {} exceeds layer_count {}",
                        tensor.name, layer_index, manifest.layer_count
                    ),
                });
            }
            layer_roles
                .entry(layer_index)
                .or_default()
                .push(tensor.role);
        } else {
            if tensor.layer_index.is_some() {
                return Err(NativeModelError::InvalidManifest {
                    message: format!("tensor {} must not declare layer_index", tensor.name),
                });
            }
            global_roles.push(tensor.role);
        }
    }

    if manifest.model_family == "whisper" {
        validate_whisper_manifest(root_dir, manifest)?;
        return Ok(());
    }

    require_global_role(
        &global_roles,
        NativeTensorRole::TokenEmbedding,
        "token_embedding",
    )?;
    require_global_role(&global_roles, NativeTensorRole::FinalNorm, "final_norm")?;
    // EmbeddingGemma is a bidirectional encoder: no LM head (never produces
    // logits), but it requires the two sentence-transformers Dense projections.
    // Nemotron 3 Embed is also encoder-only mean-pool with no Dense head and no
    // required lm_head (even when tie_word_embeddings is false on 8B packs).
    if manifest.model_family == "embeddinggemma" {
        require_global_role(
            &global_roles,
            NativeTensorRole::EmbeddingDense0,
            "embedding_dense0",
        )?;
        require_global_role(
            &global_roles,
            NativeTensorRole::EmbeddingDense1,
            "embedding_dense1",
        )?;
    } else if manifest.model_family == "nemotron_embed" {
        // Encoder-only: TokenEmbedding + FinalNorm are enough at the global
        // level; skip lm_head / Dense head requirements.
    } else if !manifest.tie_word_embeddings {
        require_global_role(&global_roles, NativeTensorRole::LmHead, "lm_head")?;
    }
    if manifest.model_family == "gemma4_assistant" {
        require_global_role(
            &global_roles,
            NativeTensorRole::AssistantPreProjection,
            "assistant_pre_projection",
        )?;
        require_global_role(
            &global_roles,
            NativeTensorRole::AssistantPostProjection,
            "assistant_post_projection",
        )?;
    }
    if manifest.model_family == "deepseek_v4" {
        for (role, label) in [
            (NativeTensorRole::HcHeadFn, "hc_head_fn"),
            (NativeTensorRole::HcHeadBase, "hc_head_base"),
            (NativeTensorRole::HcHeadScale, "hc_head_scale"),
        ] {
            require_global_role(&global_roles, role, label)?;
        }
    }

    let is_nemotron_h = manifest.model_family == "nemotron_h";
    let is_deepseek_v4 = manifest.model_family == "deepseek_v4";
    let is_muse_glimmer = manifest.model_family == "muse_glimmer";

    for layer_index in 0..manifest.layer_count {
        let roles =
            layer_roles
                .get(&layer_index)
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: format!("missing tensors for layer {}", layer_index),
                })?;
        require_layer_role(
            roles,
            NativeTensorRole::AttentionNorm,
            layer_index,
            "attention_norm",
        )?;

        // Nemotron-H layers are single residual mixers (Mamba / attention / MoE)
        // without a classic attn+FFN sandwich — skip FFN/post-norm requirements
        // and validate the mixer kind from layer_types / tensor roles below.
        if is_nemotron_h {
            validate_nemotron_h_layer(manifest, layer_index, roles)?;
            continue;
        }

        // DeepSeek V4 layers replace the classic attention-O projection with the
        // grouped wo_a/wo_b pair, add hyper-connection tensors, and gate
        // compressor/indexer/hash-routing tensors per layer — validate the V4
        // layout directly instead of the generic attn+FFN sandwich.
        if is_deepseek_v4 {
            validate_deepseek_v4_layer(manifest, layer_index, roles)?;
            continue;
        }

        // Muse Glimmer's dedicated route reads a fixed split-attention and
        // sandwich-norm layout. Require every tensor it dereferences so a
        // malformed/stale manifest fails at load instead of panicking during
        // generation or silently selecting a packed fallback.
        if is_muse_glimmer {
            for (role, label) in [
                (NativeTensorRole::AttentionQ, "attention_q"),
                (NativeTensorRole::AttentionK, "attention_k"),
                (NativeTensorRole::AttentionV, "attention_v"),
                (NativeTensorRole::AttentionO, "attention_o"),
                (NativeTensorRole::AttentionOutputGate, "attn_out_gate"),
                (NativeTensorRole::AttentionPostNorm, "attention_post_norm"),
                (NativeTensorRole::FfnNorm, "ffn_norm"),
                (NativeTensorRole::FfnPostNorm, "ffn_post_norm"),
                (NativeTensorRole::FfnGate, "ffn_gate"),
                (NativeTensorRole::FfnUp, "ffn_up"),
                (NativeTensorRole::FfnDown, "ffn_down"),
            ] {
                require_layer_role(roles, role, layer_index, label)?;
            }
        }

        // ffn_norm is optional when attention_post_norm serves as the FFN norm
        // (e.g. Qwen3.5 linear attention layers).
        if !roles.contains(&NativeTensorRole::FfnNorm)
            && !roles.contains(&NativeTensorRole::AttentionPostNorm)
        {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} is missing required tensor role ffn_norm or attention_post_norm",
                    layer_index
                ),
            });
        }
        let has_packed_gate_up = roles.contains(&NativeTensorRole::FfnGateUpPacked);
        let has_split_gate_up =
            roles.contains(&NativeTensorRole::FfnGate) && roles.contains(&NativeTensorRole::FfnUp);
        let has_dense_ffn =
            roles.contains(&NativeTensorRole::FfnDown) && (has_packed_gate_up || has_split_gate_up);
        let has_shared_expert_ffn = roles.contains(&NativeTensorRole::FfnSharedExpertGateInp)
            && roles.contains(&NativeTensorRole::FfnSharedExpertGate)
            && roles.contains(&NativeTensorRole::FfnSharedExpertUp)
            && roles.contains(&NativeTensorRole::FfnSharedExpertDown);
        let has_mla_shared_expert_ffn = matches!(
            manifest.model_family.as_str(),
            "glm4_moe_lite" | "deepseek_v3" | "deepseek_v32" | "unlimited_ocr"
        ) && roles.contains(&NativeTensorRole::FfnSharedExpertGate)
            && roles.contains(&NativeTensorRole::FfnSharedExpertUp)
            && roles.contains(&NativeTensorRole::FfnSharedExpertDown);
        let has_gpt_oss_mxfp4_moe = manifest.model_family == "gpt_oss"
            && roles.contains(&NativeTensorRole::FfnGateUpExpsMxfp4Blocks)
            && roles.contains(&NativeTensorRole::FfnGateUpExpsMxfp4Scales)
            && roles.contains(&NativeTensorRole::FfnDownExpsMxfp4Blocks)
            && roles.contains(&NativeTensorRole::FfnDownExpsMxfp4Scales);
        let has_moe_expert_ffn = roles.contains(&NativeTensorRole::FfnGateInp)
            && (has_gpt_oss_mxfp4_moe
                || (roles.contains(&NativeTensorRole::FfnDownExps)
                    && (roles.contains(&NativeTensorRole::FfnGateUpExpsPacked)
                        || roles.contains(&NativeTensorRole::FfnGateExps)
                        || roles.contains(&NativeTensorRole::FfnUpExps))));
        if !(has_dense_ffn
            || has_shared_expert_ffn
            || has_mla_shared_expert_ffn
            || has_moe_expert_ffn)
        {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} must provide dense FFN tensors or MoE expert tensors",
                    layer_index
                ),
            });
        }

        // Attention QKV/O are required for full-attention layers but optional
        // for mixed-architecture models (e.g. Qwen3.5 linear_attention layers).
        let has_any_attention = roles.contains(&NativeTensorRole::AttentionO)
            || roles.contains(&NativeTensorRole::AttentionQ)
            || roles.contains(&NativeTensorRole::AttentionK)
            || roles.contains(&NativeTensorRole::AttentionV)
            || roles.contains(&NativeTensorRole::AttentionQkvPacked)
            || has_any_glm_mla_attention_role(roles);
        let has_any_linear_attention = roles.contains(&NativeTensorRole::LinearAttentionInProjQkv)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjQkvz)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjZ)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjA)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjB)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjBa)
            || roles.contains(&NativeTensorRole::LinearAttentionConv1d)
            || roles.contains(&NativeTensorRole::LinearAttentionDtBias)
            || roles.contains(&NativeTensorRole::LinearAttentionALog)
            || roles.contains(&NativeTensorRole::LinearAttentionNorm)
            || roles.contains(&NativeTensorRole::LinearAttentionOutProj);
        let has_any_moe = roles.contains(&NativeTensorRole::FfnGateInp)
            || roles.contains(&NativeTensorRole::FfnGateInpScale)
            || roles.contains(&NativeTensorRole::FfnGateInpCorrectionBias)
            || roles.contains(&NativeTensorRole::FfnGateInpExpertScale)
            || roles.contains(&NativeTensorRole::FfnNorm2)
            || roles.contains(&NativeTensorRole::FfnPostNorm1)
            || roles.contains(&NativeTensorRole::FfnPostNorm2)
            || roles.contains(&NativeTensorRole::FfnSharedExpertGateInp)
            || roles.contains(&NativeTensorRole::FfnSharedExpertGate)
            || roles.contains(&NativeTensorRole::FfnSharedExpertUp)
            || roles.contains(&NativeTensorRole::FfnSharedExpertDown)
            || roles.contains(&NativeTensorRole::FfnGateExps)
            || roles.contains(&NativeTensorRole::FfnUpExps)
            || roles.contains(&NativeTensorRole::FfnGateUpExpsPacked)
            || roles.contains(&NativeTensorRole::FfnDownExps)
            || roles.contains(&NativeTensorRole::FfnDownExpsScale)
            || roles.contains(&NativeTensorRole::FfnGateUpExpsMxfp4Blocks)
            || roles.contains(&NativeTensorRole::FfnGateUpExpsMxfp4Scales)
            || roles.contains(&NativeTensorRole::FfnDownExpsMxfp4Blocks)
            || roles.contains(&NativeTensorRole::FfnDownExpsMxfp4Scales);
        // Gemma 4 MoE dual-norm stack (post-attn, dual FFN norms, router scale).
        // gemma4_vl shares the same text-tower MoE layout when packaged as VL.
        if matches!(manifest.model_family.as_str(), "gemma4" | "gemma4_vl") && has_moe_expert_ffn {
            if has_any_attention {
                require_layer_role(
                    roles,
                    NativeTensorRole::AttentionPostNorm,
                    layer_index,
                    "attention_post_norm",
                )?;
            }
            require_layer_role(
                roles,
                NativeTensorRole::FfnPostNorm,
                layer_index,
                "ffn_post_norm",
            )?;
            for (role, label) in [
                (NativeTensorRole::FfnGateInpScale, "ffn_gate_inp_scale"),
                (NativeTensorRole::FfnNorm2, "ffn_norm_2"),
                (NativeTensorRole::FfnPostNorm1, "ffn_post_norm_1"),
                (NativeTensorRole::FfnPostNorm2, "ffn_post_norm_2"),
            ] {
                require_layer_role(roles, role, layer_index, label)?;
            }
        }
        if has_any_attention {
            require_layer_role(
                roles,
                NativeTensorRole::AttentionO,
                layer_index,
                "attention_o",
            )?;
            if has_any_glm_mla_attention_role(roles) {
                if !matches!(
                    manifest.model_family.as_str(),
                    "glm4_moe_lite" | "deepseek_v3" | "deepseek_v32"
                ) {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} provides MLA attention tensors but model_family is {:?}",
                            layer_index, manifest.model_family
                        ),
                    });
                }
                for (role, label) in [
                    (NativeTensorRole::AttentionQa, "attention_qa"),
                    (NativeTensorRole::AttentionQaNorm, "attention_qa_norm"),
                    (NativeTensorRole::AttentionQb, "attention_qb"),
                    (NativeTensorRole::AttentionKvA, "attention_kv_a"),
                    (NativeTensorRole::AttentionKvANorm, "attention_kv_a_norm"),
                ] {
                    require_layer_role(roles, role, layer_index, label)?;
                }
                let has_kv_b = roles.contains(&NativeTensorRole::AttentionKvB);
                let has_embed_q = roles.contains(&NativeTensorRole::AttentionEmbedQ);
                let has_unembed_out = roles.contains(&NativeTensorRole::AttentionUnembedOut);
                if (has_kv_b && (has_embed_q || has_unembed_out))
                    || (!has_kv_b && (!has_embed_q || !has_unembed_out))
                {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} must provide exactly one MLA KV-B layout: attention_kv_b or attention_embed_q plus attention_unembed_out",
                            layer_index
                        ),
                    });
                }
                if roles.contains(&NativeTensorRole::AttentionQkvPacked)
                    || roles.contains(&NativeTensorRole::AttentionQ)
                    || roles.contains(&NativeTensorRole::AttentionK)
                    || roles.contains(&NativeTensorRole::AttentionV)
                {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} must not mix MLA attention with standard Q/K/V tensors",
                            layer_index
                        ),
                    });
                }
            } else {
                let uses_external_shared_kv = manifest.model_family == "gemma4_assistant";
                let uses_shared_kv = uses_external_shared_kv
                    || manifest.kv_shared_source_layers.contains_key(&layer_index);
                if uses_shared_kv {
                    require_layer_role(
                        roles,
                        NativeTensorRole::AttentionQ,
                        layer_index,
                        "attention_q",
                    )?;
                    if roles.contains(&NativeTensorRole::AttentionQkvPacked)
                        || roles.contains(&NativeTensorRole::AttentionK)
                        || roles.contains(&NativeTensorRole::AttentionV)
                    {
                        return Err(NativeModelError::InvalidManifest {
                            message: format!(
                                "KV-shared layer {} must provide attention_q/attention_o only and reuse source K/V",
                                layer_index
                            ),
                        });
                    }
                } else {
                    let uses_value_from_key = manifest
                        .attention_value_from_key_layers
                        .contains(&layer_index);
                    if uses_value_from_key
                        && (roles.contains(&NativeTensorRole::AttentionQkvPacked)
                            || roles.contains(&NativeTensorRole::AttentionV))
                    {
                        return Err(NativeModelError::InvalidManifest {
                            message: format!(
                                "value-from-key layer {} must provide split attention_q/attention_k without attention_v or attention_qkv_packed",
                                layer_index
                            ),
                        });
                    }
                    let has_packed_qkv = roles.contains(&NativeTensorRole::AttentionQkvPacked);
                    let has_split_qkv = roles.contains(&NativeTensorRole::AttentionQ)
                        && roles.contains(&NativeTensorRole::AttentionK)
                        && (roles.contains(&NativeTensorRole::AttentionV) || uses_value_from_key);
                    if !(has_packed_qkv || has_split_qkv) {
                        return Err(NativeModelError::InvalidManifest {
                            message: format!(
                                "layer {} must provide attention_qkv_packed or attention_q/attention_k plus attention_v (or mark the layer in attention_value_from_key_layers)",
                                layer_index
                            ),
                        });
                    }
                }
            }
        }
        if has_any_linear_attention {
            if !manifest.linear_attention.is_enabled() {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} provides linear attention tensors but manifest.linear_attention is not configured",
                        layer_index
                    ),
                });
            }
            let has_split_linear = roles.contains(&NativeTensorRole::LinearAttentionInProjQkv)
                && roles.contains(&NativeTensorRole::LinearAttentionInProjZ)
                && roles.contains(&NativeTensorRole::LinearAttentionInProjA)
                && roles.contains(&NativeTensorRole::LinearAttentionInProjB);
            let has_packed_linear = roles.contains(&NativeTensorRole::LinearAttentionInProjQkvz)
                && roles.contains(&NativeTensorRole::LinearAttentionInProjBa);
            if !(has_split_linear || has_packed_linear) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} must provide linear_attention split qkv/z/a/b or packed qkvz/ba projections",
                        layer_index
                    ),
                });
            }
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionConv1d,
                layer_index,
                "linear_attention_conv1d",
            )?;
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionDtBias,
                layer_index,
                "linear_attention_dt_bias",
            )?;
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionALog,
                layer_index,
                "linear_attention_a_log",
            )?;
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionNorm,
                layer_index,
                "linear_attention_norm",
            )?;
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionOutProj,
                layer_index,
                "linear_attention_out_proj",
            )?;
        }
        if has_any_moe {
            if !manifest.moe.is_enabled() {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} provides MoE tensors but manifest.moe is not configured",
                        layer_index
                    ),
                });
            }
            require_layer_role(
                roles,
                NativeTensorRole::FfnGateInp,
                layer_index,
                "ffn_gate_inp",
            )?;
            if manifest.model_family == "gpt_oss" {
                require_layer_role(roles, NativeTensorRole::AttnSink, layer_index, "attn_sink")?;
            }
            let has_any_shared_expert = roles.contains(&NativeTensorRole::FfnSharedExpertGateInp)
                || roles.contains(&NativeTensorRole::FfnSharedExpertGate)
                || roles.contains(&NativeTensorRole::FfnSharedExpertUp)
                || roles.contains(&NativeTensorRole::FfnSharedExpertDown);
            if has_any_shared_expert || moe_requires_shared_expert(manifest) {
                if !matches!(
                    manifest.model_family.as_str(),
                    "glm4_moe_lite" | "deepseek_v3" | "deepseek_v32" | "llama4" | "unlimited_ocr"
                ) {
                    require_layer_role(
                        roles,
                        NativeTensorRole::FfnSharedExpertGateInp,
                        layer_index,
                        "ffn_shared_expert_gate_inp",
                    )?;
                }
                require_layer_role(
                    roles,
                    NativeTensorRole::FfnSharedExpertGate,
                    layer_index,
                    "ffn_shared_expert_gate",
                )?;
                require_layer_role(
                    roles,
                    NativeTensorRole::FfnSharedExpertUp,
                    layer_index,
                    "ffn_shared_expert_up",
                )?;
                require_layer_role(
                    roles,
                    NativeTensorRole::FfnSharedExpertDown,
                    layer_index,
                    "ffn_shared_expert_down",
                )?;
            }
            let has_packed_moe = roles.contains(&NativeTensorRole::FfnGateUpExpsPacked);
            let has_gate_exps = roles.contains(&NativeTensorRole::FfnGateExps);
            let has_up_exps = roles.contains(&NativeTensorRole::FfnUpExps);
            let has_split_moe = has_gate_exps && has_up_exps;
            let has_any_mxfp4_moe = roles.contains(&NativeTensorRole::FfnGateUpExpsMxfp4Blocks)
                || roles.contains(&NativeTensorRole::FfnGateUpExpsMxfp4Scales)
                || roles.contains(&NativeTensorRole::FfnDownExpsMxfp4Blocks)
                || roles.contains(&NativeTensorRole::FfnDownExpsMxfp4Scales);
            if has_any_mxfp4_moe && !has_gpt_oss_mxfp4_moe {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {layer_index} must provide all four GPT-OSS MXFP4 block/scale tensors"
                    ),
                });
            }
            if has_gpt_oss_mxfp4_moe
                && (has_packed_moe
                    || has_gate_exps
                    || has_up_exps
                    || roles.contains(&NativeTensorRole::FfnDownExps))
            {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {layer_index} must not mix GPT-OSS MXFP4 blocks with sanitized expert tensors"
                    ),
                });
            }
            if has_packed_moe && (has_gate_exps || has_up_exps) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} must not mix ffn_gate_up_exps_packed with ffn_gate_exps/ffn_up_exps",
                        layer_index
                    ),
                });
            }
            if manifest.model_family == "gpt_oss" && has_packed_moe {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {layer_index} GPT-OSS experts must use split tensors or native MXFP4 blocks"
                    ),
                });
            }
            if !(has_gpt_oss_mxfp4_moe || has_packed_moe || has_split_moe) {
                let required_layout = if manifest.model_family == "gpt_oss" {
                    "ffn_gate_exps/ffn_up_exps or all four GPT-OSS MXFP4 block/scale tensors"
                } else {
                    "ffn_gate_up_exps_packed or ffn_gate_exps/ffn_up_exps"
                };
                return Err(NativeModelError::InvalidManifest {
                    message: format!("layer {layer_index} must provide {required_layout}"),
                });
            }
            if !has_gpt_oss_mxfp4_moe {
                require_layer_role(
                    roles,
                    NativeTensorRole::FfnDownExps,
                    layer_index,
                    "ffn_down_exps",
                )?;
            }
        }
    }

    validate_native_model_tensor_shapes(manifest)?;

    Ok(())
}

fn validate_whisper_manifest(
    root_dir: &Path,
    manifest: &NativeModelManifest,
) -> Result<(), NativeModelError> {
    if manifest
        .tensors
        .iter()
        .any(|tensor| tensor.role != NativeTensorRole::Other || tensor.layer_index.is_some())
    {
        return Err(NativeModelError::InvalidManifest {
            message: "Whisper tensors must preserve exact checkpoint names with role=other"
                .to_string(),
        });
    }
    if manifest
        .tensors
        .iter()
        .any(|tensor| tensor.source_quantized)
    {
        return Err(NativeModelError::InvalidManifest {
            message: "Whisper native runtime currently requires floating-point weights".to_string(),
        });
    }

    let config_path = root_dir.join("config.json");
    let config_bytes = fs::read(&config_path).map_err(|source| NativeModelError::ReadManifest {
        path: config_path.clone(),
        source,
    })?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes).map_err(|source| {
        NativeModelError::ParseManifest {
            path: config_path,
            source,
        }
    })?;
    let field = |name: &'static str| -> Result<u32, NativeModelError> {
        config
            .get(name)
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| NativeModelError::InvalidManifest {
                message: format!("Whisper config requires positive {name}"),
            })
    };
    if config.get("model_type").and_then(serde_json::Value::as_str) != Some("whisper") {
        return Err(NativeModelError::InvalidManifest {
            message: "Whisper manifest requires config.json model_type=whisper".to_string(),
        });
    }
    let n_mels = field("n_mels")?;
    let n_audio_ctx = field("n_audio_ctx")?;
    let n_audio_state = field("n_audio_state")?;
    let n_audio_head = field("n_audio_head")?;
    let n_audio_layer = field("n_audio_layer")?;
    let n_vocab = field("n_vocab")?;
    let n_text_ctx = field("n_text_ctx")?;
    let n_text_state = field("n_text_state")?;
    let n_text_head = field("n_text_head")?;
    let n_text_layer = field("n_text_layer")?;
    if n_audio_state != n_text_state {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "Whisper audio/text state widths must match, got {n_audio_state}/{n_text_state}"
            ),
        });
    }
    if !n_audio_state.is_multiple_of(n_audio_head) || !n_text_state.is_multiple_of(n_text_head) {
        return Err(NativeModelError::InvalidManifest {
            message: "Whisper state widths must divide evenly across attention heads".to_string(),
        });
    }
    if !matches!(n_vocab, 51_865 | 51_866) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "Whisper native tokenizer supports multilingual vocabularies 51865/51866, got {n_vocab}"
            ),
        });
    }
    if manifest.layer_count != n_audio_layer
        || manifest.hidden_size != n_audio_state
        || manifest.attention_head_count != n_audio_head
        || manifest.attention_head_dim != n_audio_state / n_audio_head
        || manifest.vocab_size != n_vocab
    {
        return Err(NativeModelError::InvalidManifest {
            message: "Whisper manifest architecture does not match config.json".to_string(),
        });
    }

    let require_shape = |name: &str, expected: &[u64]| -> Result<(), NativeModelError> {
        let tensor = manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| NativeModelError::InvalidManifest {
                message: format!("Whisper manifest is missing tensor {name}"),
            })?;
        if tensor.shape != expected {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "Whisper tensor {name} shape {:?}, expected {expected:?}",
                    tensor.shape
                ),
            });
        }
        Ok(())
    };
    require_shape(
        "encoder.conv1.weight",
        &[u64::from(n_audio_state), 3, u64::from(n_mels)],
    )?;
    require_shape(
        "encoder.conv2.weight",
        &[u64::from(n_audio_state), 3, u64::from(n_audio_state)],
    )?;
    require_shape("encoder.ln_post.weight", &[u64::from(n_audio_state)])?;
    require_shape(
        "decoder.token_embedding.weight",
        &[u64::from(n_vocab), u64::from(n_text_state)],
    )?;
    require_shape(
        "decoder.positional_embedding",
        &[u64::from(n_text_ctx), u64::from(n_text_state)],
    )?;
    require_shape("decoder.ln.weight", &[u64::from(n_text_state)])?;

    for layer in 0..n_audio_layer {
        require_shape(
            &format!("encoder.blocks.{layer}.attn.query.weight"),
            &[u64::from(n_audio_state), u64::from(n_audio_state)],
        )?;
        require_shape(
            &format!("encoder.blocks.{layer}.mlp1.weight"),
            &[
                u64::from(n_audio_state).saturating_mul(4),
                u64::from(n_audio_state),
            ],
        )?;
        require_shape(
            &format!("encoder.blocks.{layer}.mlp2.weight"),
            &[
                u64::from(n_audio_state),
                u64::from(n_audio_state).saturating_mul(4),
            ],
        )?;
    }
    for layer in 0..n_text_layer {
        for name in [
            "attn.query.weight",
            "cross_attn.query.weight",
            "cross_attn.key.weight",
            "cross_attn.value.weight",
        ] {
            require_shape(
                &format!("decoder.blocks.{layer}.{name}"),
                &[u64::from(n_text_state), u64::from(n_text_state)],
            )?;
        }
    }
    // These two dimensions are not represented in the decoder-only manifest
    // schema, so exercising them here keeps malformed configs from reaching
    // the MLX graph builder.
    if n_audio_ctx != 1_500 || n_text_ctx > 448 {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "Whisper runtime expects n_audio_ctx=1500 and n_text_ctx<=448, got {n_audio_ctx}/{n_text_ctx}"
            ),
        });
    }
    Ok(())
}

fn has_any_glm_mla_attention_role(roles: &[NativeTensorRole]) -> bool {
    roles.contains(&NativeTensorRole::AttentionQa)
        || roles.contains(&NativeTensorRole::AttentionQaNorm)
        || roles.contains(&NativeTensorRole::AttentionQb)
        || roles.contains(&NativeTensorRole::AttentionKvA)
        || roles.contains(&NativeTensorRole::AttentionKvB)
        || roles.contains(&NativeTensorRole::AttentionKvANorm)
        || roles.contains(&NativeTensorRole::AttentionEmbedQ)
        || roles.contains(&NativeTensorRole::AttentionUnembedOut)
}

/// Validate one Nemotron-H residual mixer layer (Mamba / attention / MoE).
fn validate_nemotron_h_layer(
    manifest: &NativeModelManifest,
    layer_index: u32,
    roles: &[NativeTensorRole],
) -> Result<(), NativeModelError> {
    let layer_kind = manifest
        .layer_types
        .get(layer_index as usize)
        .map(|s| s.as_str())
        .unwrap_or("");

    let has_attn = roles.contains(&NativeTensorRole::AttentionQ)
        && roles.contains(&NativeTensorRole::AttentionK)
        && roles.contains(&NativeTensorRole::AttentionV)
        && roles.contains(&NativeTensorRole::AttentionO);
    // Packed Mamba-2 in_proj is mapped to LinearAttentionInProjQkvz (no ba).
    let has_mamba = roles.contains(&NativeTensorRole::LinearAttentionInProjQkvz)
        && roles.contains(&NativeTensorRole::LinearAttentionConv1d)
        && roles.contains(&NativeTensorRole::LinearAttentionDtBias)
        && roles.contains(&NativeTensorRole::LinearAttentionALog)
        && roles.contains(&NativeTensorRole::LinearAttentionNorm)
        && roles.contains(&NativeTensorRole::LinearAttentionOutProj);
    let has_moe = roles.contains(&NativeTensorRole::FfnGateInp)
        && roles.contains(&NativeTensorRole::FfnUpExps)
        && roles.contains(&NativeTensorRole::FfnDownExps);

    match layer_kind {
        "mamba" | "M" | "m" => {
            if !has_mamba {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "nemotron_h layer {layer_index} is mamba but missing Mamba-2 mixer tensors"
                    ),
                });
            }
            if !manifest.linear_attention.is_enabled() {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "nemotron_h layer {layer_index} is mamba but linear_attention is not configured"
                    ),
                });
            }
        }
        "attention" | "*" | "full_attention" => {
            if !has_attn {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "nemotron_h layer {layer_index} is attention but missing Q/K/V/O projections"
                    ),
                });
            }
        }
        "moe" | "E" | "e" => {
            if !has_moe {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "nemotron_h layer {layer_index} is moe but missing router/expert tensors"
                    ),
                });
            }
            if !manifest.moe.is_enabled() {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "nemotron_h layer {layer_index} is moe but manifest.moe is not configured"
                    ),
                });
            }
            // Shared expert is ReLU² up+down only (no SwiGLU gate).
            if roles.contains(&NativeTensorRole::FfnSharedExpertUp)
                != roles.contains(&NativeTensorRole::FfnSharedExpertDown)
            {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "nemotron_h layer {layer_index} shared expert must provide both up and down"
                    ),
                });
            }
        }
        "mlp" | "-" => {
            let has_mlp = roles.contains(&NativeTensorRole::FfnUp)
                && roles.contains(&NativeTensorRole::FfnDown);
            if !has_mlp {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "nemotron_h layer {layer_index} is mlp but missing up/down projections"
                    ),
                });
            }
        }
        "" => {
            // Fall back to tensor-role inference when layer_types is empty.
            if !(has_mamba || has_attn || has_moe) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "nemotron_h layer {layer_index} has no mamba, attention, or moe mixer tensors"
                    ),
                });
            }
        }
        other => {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "nemotron_h layer {layer_index} has unknown layer_types entry {other:?}"
                ),
            });
        }
    }
    Ok(())
}

/// Validate one DeepSeek V4 layer.
///
/// Every V4 layer is an MoE layer with the same attention layout: q_a/q_norm/
/// q_b + fused wkv + kv_norm + grouped wo_a/wo_b output, an attention sink,
/// and both hyper-connection trios. Compressor tensors are required exactly
/// when the layer's `compress_ratios` entry is 4 or 128; indexer tensors only
/// when it is 4. The first `num_hash_layers` layers route via the
/// `ffn.gate.tid2eid` hash table, the rest via the learned gate correction
/// bias — exactly one of the two must be present per layer.
fn validate_deepseek_v4_layer(
    manifest: &NativeModelManifest,
    layer_index: u32,
    roles: &[NativeTensorRole],
) -> Result<(), NativeModelError> {
    let require = |role: NativeTensorRole, label: &str| -> Result<(), NativeModelError> {
        if roles.contains(&role) {
            return Ok(());
        }
        Err(NativeModelError::InvalidManifest {
            message: format!(
                "deepseek_v4 layer {layer_index} is missing required tensor role {label}"
            ),
        })
    };

    for (role, label) in [
        (NativeTensorRole::AttentionQa, "attention_qa"),
        (NativeTensorRole::AttentionQaNorm, "attention_qa_norm"),
        (NativeTensorRole::AttentionQb, "attention_qb"),
        (NativeTensorRole::AttentionKv, "attention_kv"),
        (NativeTensorRole::AttentionKvNorm, "attention_kv_norm"),
        (NativeTensorRole::AttentionOutA, "attention_out_a"),
        (NativeTensorRole::AttentionOutB, "attention_out_b"),
        (NativeTensorRole::HcAttnFn, "hc_attn_fn"),
        (NativeTensorRole::HcAttnBase, "hc_attn_base"),
        (NativeTensorRole::HcAttnScale, "hc_attn_scale"),
        (NativeTensorRole::HcFfnFn, "hc_ffn_fn"),
        (NativeTensorRole::HcFfnBase, "hc_ffn_base"),
        (NativeTensorRole::HcFfnScale, "hc_ffn_scale"),
        (NativeTensorRole::FfnNorm, "ffn_norm"),
        (NativeTensorRole::FfnGateInp, "ffn_gate_inp"),
        (NativeTensorRole::FfnDownExps, "ffn_down_exps"),
    ] {
        require(role, label)?;
    }
    // Routed experts ship exactly one layout: split gate/up stacks (raw HF /
    // sanitized, or AXQ `switch_mlp` with both `gate_proj` and `up_proj`)
    // or the fused AXQ gate+up tensor (`ffn_gate_up_exps_packed`).
    let has_packed_experts = roles.contains(&NativeTensorRole::FfnGateUpExpsPacked);
    let has_gate_exps = roles.contains(&NativeTensorRole::FfnGateExps);
    let has_up_exps = roles.contains(&NativeTensorRole::FfnUpExps);
    if has_packed_experts == (has_gate_exps || has_up_exps) || has_gate_exps != has_up_exps {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "deepseek_v4 layer {layer_index} must provide exactly one routed-expert layout: ffn_gate_up_exps_packed or ffn_gate_exps plus ffn_up_exps"
            ),
        });
    }
    if manifest.deepseek_v4.attention.has_attn_sinks {
        require(NativeTensorRole::AttnSink, "attn_sink")?;
    }
    if manifest.moe.shared_expert_count.unwrap_or(0) > 0 {
        for (role, label) in [
            (
                NativeTensorRole::FfnSharedExpertGate,
                "ffn_shared_expert_gate",
            ),
            (NativeTensorRole::FfnSharedExpertUp, "ffn_shared_expert_up"),
            (
                NativeTensorRole::FfnSharedExpertDown,
                "ffn_shared_expert_down",
            ),
        ] {
            require(role, label)?;
        }
    }

    let compress_ratio = manifest
        .deepseek_v4
        .compress_ratios
        .get(layer_index as usize)
        .copied()
        .unwrap_or(0);
    const COMPRESSOR_ROLES: [NativeTensorRole; 4] = [
        NativeTensorRole::CompressorKv,
        NativeTensorRole::CompressorGate,
        NativeTensorRole::CompressorApe,
        NativeTensorRole::CompressorNorm,
    ];
    const INDEXER_ROLES: [NativeTensorRole; 6] = [
        NativeTensorRole::IndexerProj,
        NativeTensorRole::IndexerQb,
        NativeTensorRole::IndexerCompressorKv,
        NativeTensorRole::IndexerCompressorGate,
        NativeTensorRole::IndexerCompressorApe,
        NativeTensorRole::IndexerCompressorNorm,
    ];
    if matches!(compress_ratio, 4 | 128) {
        for (role, label) in COMPRESSOR_ROLES.into_iter().zip([
            "compressor_kv",
            "compressor_gate",
            "compressor_ape",
            "compressor_norm",
        ]) {
            require(role, label)?;
        }
    } else {
        // The compressor exists iff the layer compresses (ratio 4/128).
        for role in COMPRESSOR_ROLES {
            if roles.contains(&role) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "deepseek_v4 layer {layer_index} must not provide compressor role {role:?} with compress_ratio {compress_ratio}"
                    ),
                });
            }
        }
    }
    if compress_ratio == 4 {
        for (role, label) in INDEXER_ROLES.into_iter().zip([
            "indexer_proj",
            "indexer_qb",
            "indexer_compressor_kv",
            "indexer_compressor_gate",
            "indexer_compressor_ape",
            "indexer_compressor_norm",
        ]) {
            require(role, label)?;
        }
    } else {
        // The sparse indexer exists iff compress_ratio == 4.
        for role in INDEXER_ROLES {
            if roles.contains(&role) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "deepseek_v4 layer {layer_index} must not provide indexer role {role:?} with compress_ratio {compress_ratio}"
                    ),
                });
            }
        }
    }

    let has_tid2eid = roles.contains(&NativeTensorRole::FfnGateTid2Eid);
    let has_correction_bias = roles.contains(&NativeTensorRole::FfnGateInpCorrectionBias);
    let is_hash_layer = layer_index < manifest.deepseek_v4.num_hash_layers.unwrap_or(0);
    if is_hash_layer != has_tid2eid || is_hash_layer == has_correction_bias {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "deepseek_v4 layer {layer_index} must provide ffn_gate_tid2eid on hash layers (index < num_hash_layers) or ffn_gate_inp_correction_bias otherwise, exactly one"
            ),
        });
    }

    Ok(())
}

fn moe_requires_shared_expert(manifest: &NativeModelManifest) -> bool {
    manifest.moe.is_enabled() && matches!(manifest.model_family.as_str(), "qwen3_5" | "qwen3_next")
}

fn validate_native_model_tensor_shapes(
    manifest: &NativeModelManifest,
) -> Result<(), NativeModelError> {
    if !manifest
        .attention_head_count
        .is_multiple_of(manifest.kv_head_count)
    {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "attention_head_count {} must be divisible by kv_head_count {}",
                manifest.attention_head_count, manifest.kv_head_count
            ),
        });
    }

    let hidden_size = u64::from(manifest.hidden_size);
    let vocab_size = u64::from(manifest.vocab_size);
    let token_embedding = required_global_tensor_spec(
        manifest,
        NativeTensorRole::TokenEmbedding,
        "token_embedding",
    )?;
    expect_matrix_shape(token_embedding, vocab_size, hidden_size, "token_embedding")?;

    let final_norm =
        required_global_tensor_spec(manifest, NativeTensorRole::FinalNorm, "final_norm")?;
    expect_vector_shape(final_norm, hidden_size, "final_norm")?;

    // EmbeddingGemma encoder: no LM head (see role-presence check above), so skip
    // the lm_head shape validation; validate the Dense projection head instead.
    // Nemotron Embed is encoder-only mean-pool without Dense head or lm_head.
    if manifest.model_family == "embeddinggemma" {
        let dense0 = required_global_tensor_spec(
            manifest,
            NativeTensorRole::EmbeddingDense0,
            "embedding_dense0",
        )?;
        expect_matrix_shape(dense0, hidden_size * 4, hidden_size, "embedding_dense0")?;
        let dense1 = required_global_tensor_spec(
            manifest,
            NativeTensorRole::EmbeddingDense1,
            "embedding_dense1",
        )?;
        expect_matrix_shape(dense1, hidden_size, hidden_size * 4, "embedding_dense1")?;
    } else if manifest.model_family == "nemotron_embed" {
        // no lm_head / Dense head shapes
    } else if !manifest.tie_word_embeddings {
        let lm_head = required_global_tensor_spec(manifest, NativeTensorRole::LmHead, "lm_head")?;
        expect_matrix_shape(lm_head, vocab_size, hidden_size, "lm_head")?;
    }
    if manifest.model_family == "gemma4_assistant" {
        let pre_projection = required_global_tensor_spec(
            manifest,
            NativeTensorRole::AssistantPreProjection,
            "assistant_pre_projection",
        )?;
        let (pre_rows, pre_cols) =
            matrix_shape(pre_projection).ok_or_else(|| NativeModelError::InvalidManifest {
                message: "assistant_pre_projection must be a rank-2 matrix".to_string(),
            })?;
        if pre_rows != hidden_size || pre_cols == 0 || pre_cols % 2 != 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "assistant_pre_projection must have shape [{hidden_size}, 2 * backbone_hidden_size], got {:?}",
                    pre_projection.shape
                ),
            });
        }
        let post_projection = required_global_tensor_spec(
            manifest,
            NativeTensorRole::AssistantPostProjection,
            "assistant_post_projection",
        )?;
        let (post_rows, _post_cols) =
            matrix_shape(post_projection).ok_or_else(|| NativeModelError::InvalidManifest {
                message: "assistant_post_projection must be a rank-2 matrix".to_string(),
            })?;
        if post_rows == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "assistant_post_projection must have shape [backbone_hidden_size, {hidden_size}], got {:?}",
                    post_projection.shape
                ),
            });
        }
        expect_matrix_shape(
            post_projection,
            post_rows,
            hidden_size,
            "assistant_post_projection",
        )?;
    }
    validate_per_layer_input_tensor_shapes(manifest, hidden_size, vocab_size)?;

    // Nemotron-H reuses linear_attention / MoE roles with Mamba-2 and ReLU²
    // shapes that differ from Qwen gated-delta / SwiGLU contracts. Role presence
    // is already enforced in validate_nemotron_h_layer; skip Qwen-shaped checks.
    if manifest.model_family == "nemotron_h" {
        for layer_index in 0..manifest.layer_count {
            let attention_norm = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::AttentionNorm,
                "attention_norm",
            )?;
            expect_vector_shape(attention_norm, hidden_size, "attention_norm")?;
            if let Some(attention_q) =
                manifest_tensor(manifest, NativeTensorRole::AttentionQ, Some(layer_index))
            {
                let (q_rows, kv_rows) = configured_attention_projection_dims(manifest, layer_index);
                let attention_k = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::AttentionK,
                    "attention_k",
                )?;
                let attention_v = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::AttentionV,
                    "attention_v",
                )?;
                let attention_o = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::AttentionO,
                    "attention_o",
                )?;
                expect_matrix_shape(attention_q, q_rows, hidden_size, "attention_q")?;
                expect_matrix_shape(attention_k, kv_rows, hidden_size, "attention_k")?;
                expect_matrix_shape(attention_v, kv_rows, hidden_size, "attention_v")?;
                expect_matrix_shape(attention_o, hidden_size, q_rows, "attention_o")?;
            }
        }
        return Ok(());
    }

    for layer_index in 0..manifest.layer_count {
        let attention_norm = required_layer_tensor_spec(
            manifest,
            layer_index,
            NativeTensorRole::AttentionNorm,
            "attention_norm",
        )?;
        expect_vector_shape(attention_norm, hidden_size, "attention_norm")?;
        if let Some(attention_post_norm) = manifest_tensor(
            manifest,
            NativeTensorRole::AttentionPostNorm,
            Some(layer_index),
        ) {
            expect_vector_shape(attention_post_norm, hidden_size, "attention_post_norm")?;
        }
        if let Some(attention_q_norm) = manifest_tensor(
            manifest,
            NativeTensorRole::AttentionQNorm,
            Some(layer_index),
        ) {
            let head_dim = configured_attention_head_dim(manifest, layer_index);
            expect_vector_shape(attention_q_norm, head_dim, "attention_q_norm")?;
        }
        if let Some(attention_k_norm) = manifest_tensor(
            manifest,
            NativeTensorRole::AttentionKNorm,
            Some(layer_index),
        ) {
            let head_dim = configured_attention_head_dim(manifest, layer_index);
            expect_vector_shape(attention_k_norm, head_dim, "attention_k_norm")?;
        }
        if let Some(attn_sink) =
            manifest_tensor(manifest, NativeTensorRole::AttnSink, Some(layer_index))
        {
            expect_vector_shape(
                attn_sink,
                u64::from(manifest.attention_head_count),
                "attn_sink",
            )?;
        }
        // Attention O shape validation — only for layers that have attention tensors.
        // The output projection maps from attention output dim back to hidden_size.
        // For standard attention: o_proj shape is [hidden_size, num_heads * head_dim].
        // For gated attention (Qwen3.5): q_proj has 2x rows (queries + gate), but
        // o_proj still maps from num_heads * head_dim, not from q_proj rows.
        if let Some(attention_o) =
            manifest_tensor(manifest, NativeTensorRole::AttentionO, Some(layer_index))
            && manifest_tensor(manifest, NativeTensorRole::AttentionQa, Some(layer_index)).is_none()
        {
            let attention_output_cols = u64::from(manifest.attention_head_count)
                * configured_attention_head_dim(manifest, layer_index);
            expect_matrix_shape(
                attention_o,
                hidden_size,
                attention_output_cols,
                "attention_o",
            )?;
        }

        let ffn_norm = manifest_tensor(manifest, NativeTensorRole::FfnNorm, Some(layer_index))
            .or_else(|| {
                manifest_tensor(
                    manifest,
                    NativeTensorRole::AttentionPostNorm,
                    Some(layer_index),
                )
            });
        if let Some(ffn_norm) = ffn_norm {
            expect_vector_shape(ffn_norm, hidden_size, "ffn_norm")?;
        }
        if let Some(ffn_norm_2) =
            manifest_tensor(manifest, NativeTensorRole::FfnNorm2, Some(layer_index))
        {
            expect_vector_shape(ffn_norm_2, hidden_size, "ffn_norm_2")?;
        }
        if let Some(ffn_post_norm) =
            manifest_tensor(manifest, NativeTensorRole::FfnPostNorm, Some(layer_index))
        {
            expect_vector_shape(ffn_post_norm, hidden_size, "ffn_post_norm")?;
        }
        if let Some(ffn_post_norm_1) =
            manifest_tensor(manifest, NativeTensorRole::FfnPostNorm1, Some(layer_index))
        {
            expect_vector_shape(ffn_post_norm_1, hidden_size, "ffn_post_norm_1")?;
        }
        if let Some(ffn_post_norm_2) =
            manifest_tensor(manifest, NativeTensorRole::FfnPostNorm2, Some(layer_index))
        {
            expect_vector_shape(ffn_post_norm_2, hidden_size, "ffn_post_norm_2")?;
        }

        let ffn_down = manifest_tensor(manifest, NativeTensorRole::FfnDown, Some(layer_index));
        let ffn_down_shape = ffn_down
            .map(|tensor| {
                matrix_shape(tensor).ok_or_else(|| NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_down must be a rank-2 matrix",
                        layer_index
                    ),
                })
            })
            .transpose()?;
        if let (Some(ffn_down), Some(ffn_down_shape)) = (ffn_down, ffn_down_shape) {
            if ffn_down_shape.0 != hidden_size {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_down must have shape [{}, intermediate_dim], got {:?}",
                        layer_index, hidden_size, ffn_down.shape
                    ),
                });
            }
        }

        if let Some(attention_qkv) = manifest_tensor(
            manifest,
            NativeTensorRole::AttentionQkvPacked,
            Some(layer_index),
        ) {
            let (q_rows, kv_rows) = configured_attention_projection_dims(manifest, layer_index);
            let packed_q_rows = if manifest.attn_output_gate {
                q_rows.saturating_mul(2)
            } else {
                q_rows
            };
            expect_matrix_shape(
                attention_qkv,
                packed_q_rows + kv_rows + kv_rows,
                hidden_size,
                "attention_qkv_packed",
            )?;
        } else if manifest_tensor(manifest, NativeTensorRole::AttentionQ, Some(layer_index))
            .is_some()
        {
            let attention_q = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::AttentionQ,
                "attention_q",
            )?;
            if manifest.model_family == "gemma4_assistant"
                || manifest.kv_shared_source_layers.contains_key(&layer_index)
            {
                validate_q_only_attention_tensor(manifest, layer_index, attention_q)?;
            } else {
                let attention_k = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::AttentionK,
                    "attention_k",
                )?;
                let split_dims = resolved_split_attention_dims(manifest, layer_index)?;
                expect_matrix_shape(attention_q, split_dims.q_rows, hidden_size, "attention_q")?;
                expect_matrix_shape(attention_k, split_dims.kv_rows, hidden_size, "attention_k")?;
                if let Some(attention_v) =
                    manifest_tensor(manifest, NativeTensorRole::AttentionV, Some(layer_index))
                {
                    expect_matrix_shape(
                        attention_v,
                        split_dims.kv_rows,
                        hidden_size,
                        "attention_v",
                    )?;
                }
            }
        } else if manifest_tensor(manifest, NativeTensorRole::AttentionQa, Some(layer_index))
            .is_some()
        {
            // DeepSeek V4 reuses the q_a/q_norm/q_b roles but replaces the V3
            // MLA KV/O projections with fused wkv + grouped wo_a/wo_b.
            if manifest.model_family == "deepseek_v4" {
                validate_deepseek_v4_attention_tensor_shapes(manifest, layer_index)?;
            } else {
                validate_glm_mla_attention_tensor_shapes(manifest, layer_index)?;
            }
        }
        // Layers without standard attention tensors (e.g. linear_attention) skip
        // QKV shape validation but still validate their projection contract below.
        let has_split_linear_projection = manifest_tensor(
            manifest,
            NativeTensorRole::LinearAttentionInProjQkv,
            Some(layer_index),
        )
        .is_some();
        let has_packed_linear_projection = manifest_tensor(
            manifest,
            NativeTensorRole::LinearAttentionInProjQkvz,
            Some(layer_index),
        )
        .is_some();
        if has_split_linear_projection || has_packed_linear_projection {
            let linear_dims = resolved_linear_attention_dims(manifest)?;
            if has_packed_linear_projection {
                let in_proj_qkvz = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::LinearAttentionInProjQkvz,
                    "linear_attention_in_proj_qkvz",
                )?;
                expect_matrix_shape(
                    in_proj_qkvz,
                    linear_dims.conv_dim + linear_dims.value_dim,
                    hidden_size,
                    "linear_attention_in_proj_qkvz",
                )?;
                let in_proj_ba = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::LinearAttentionInProjBa,
                    "linear_attention_in_proj_ba",
                )?;
                expect_matrix_shape(
                    in_proj_ba,
                    linear_dims.num_value_heads.saturating_mul(2),
                    hidden_size,
                    "linear_attention_in_proj_ba",
                )?;
            } else {
                let in_proj_qkv = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::LinearAttentionInProjQkv,
                    "linear_attention_in_proj_qkv",
                )?;
                expect_matrix_shape(
                    in_proj_qkv,
                    linear_dims.conv_dim,
                    hidden_size,
                    "linear_attention_in_proj_qkv",
                )?;
                let in_proj_z = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::LinearAttentionInProjZ,
                    "linear_attention_in_proj_z",
                )?;
                expect_matrix_shape(
                    in_proj_z,
                    linear_dims.value_dim,
                    hidden_size,
                    "linear_attention_in_proj_z",
                )?;
                let in_proj_a = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::LinearAttentionInProjA,
                    "linear_attention_in_proj_a",
                )?;
                expect_matrix_shape(
                    in_proj_a,
                    linear_dims.num_value_heads,
                    hidden_size,
                    "linear_attention_in_proj_a",
                )?;
                let in_proj_b = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::LinearAttentionInProjB,
                    "linear_attention_in_proj_b",
                )?;
                expect_matrix_shape(
                    in_proj_b,
                    linear_dims.num_value_heads,
                    hidden_size,
                    "linear_attention_in_proj_b",
                )?;
            }
            let dt_bias = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::LinearAttentionDtBias,
                "linear_attention_dt_bias",
            )?;
            expect_vector_shape(
                dt_bias,
                linear_dims.num_value_heads,
                "linear_attention_dt_bias",
            )?;
            let a_log = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::LinearAttentionALog,
                "linear_attention_a_log",
            )?;
            expect_vector_shape(a_log, linear_dims.num_value_heads, "linear_attention_a_log")?;
            let norm = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::LinearAttentionNorm,
                "linear_attention_norm",
            )?;
            expect_vector_shape(norm, linear_dims.value_head_dim, "linear_attention_norm")?;
            let out_proj = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::LinearAttentionOutProj,
                "linear_attention_out_proj",
            )?;
            expect_matrix_shape(
                out_proj,
                hidden_size,
                linear_dims.value_dim,
                "linear_attention_out_proj",
            )?;
            let conv1d = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::LinearAttentionConv1d,
                "linear_attention_conv1d",
            )?;
            validate_linear_attention_conv_tensor(
                conv1d,
                linear_dims.conv_dim,
                linear_dims.conv_kernel_dim,
            )?;
        }
        // Router sidecars are validated independently of `ffn_gate_inp` so a
        // layer cannot smuggle in an unchecked expert-indexed vector.
        for (role, label) in [
            (
                NativeTensorRole::FfnGateInpCorrectionBias,
                "ffn_gate_inp_correction_bias",
            ),
            (
                NativeTensorRole::FfnGateInpExpertScale,
                "ffn_gate_inp_expert_scale",
            ),
        ] {
            if let Some(sidecar) = manifest_tensor(manifest, role, Some(layer_index)) {
                let moe_dims = resolved_moe_dims(manifest)?;
                expect_vector_shape(sidecar, moe_dims.expert_count, label)?;
            }
        }
        if manifest_tensor(manifest, NativeTensorRole::FfnGateInp, Some(layer_index)).is_some() {
            let moe_dims = resolved_moe_dims(manifest)?;
            let gate_inp = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnGateInp,
                "ffn_gate_inp",
            )?;
            expect_matrix_shape(gate_inp, moe_dims.expert_count, hidden_size, "ffn_gate_inp")?;
            if let Some(gate_inp_scale) = manifest_tensor(
                manifest,
                NativeTensorRole::FfnGateInpScale,
                Some(layer_index),
            ) {
                expect_vector_shape(gate_inp_scale, hidden_size, "ffn_gate_inp_scale")?;
            }
            if manifest_tensor(
                manifest,
                NativeTensorRole::FfnGateUpExpsMxfp4Blocks,
                Some(layer_index),
            )
            .is_some()
            {
                validate_gpt_oss_mxfp4_tensor_shapes(manifest, layer_index)?;
            } else {
                if let Some(ffn_gate_up_exps_packed) = manifest_tensor(
                    manifest,
                    NativeTensorRole::FfnGateUpExpsPacked,
                    Some(layer_index),
                ) {
                    expect_tensor_shape(
                        ffn_gate_up_exps_packed,
                        &[
                            moe_dims.expert_count,
                            moe_dims.expert_intermediate_size.saturating_mul(2),
                            hidden_size,
                        ],
                        "ffn_gate_up_exps_packed",
                    )?;
                } else {
                    let ffn_gate_exps = required_layer_tensor_spec(
                        manifest,
                        layer_index,
                        NativeTensorRole::FfnGateExps,
                        "ffn_gate_exps",
                    )?;
                    let ffn_up_exps = required_layer_tensor_spec(
                        manifest,
                        layer_index,
                        NativeTensorRole::FfnUpExps,
                        "ffn_up_exps",
                    )?;
                    expect_tensor_shape(
                        ffn_gate_exps,
                        &[
                            moe_dims.expert_count,
                            moe_dims.expert_intermediate_size,
                            hidden_size,
                        ],
                        "ffn_gate_exps",
                    )?;
                    expect_tensor_shape(
                        ffn_up_exps,
                        &[
                            moe_dims.expert_count,
                            moe_dims.expert_intermediate_size,
                            hidden_size,
                        ],
                        "ffn_up_exps",
                    )?;
                }
                let ffn_down_exps = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::FfnDownExps,
                    "ffn_down_exps",
                )?;
                expect_tensor_shape(
                    ffn_down_exps,
                    &[
                        moe_dims.expert_count,
                        hidden_size,
                        moe_dims.expert_intermediate_size,
                    ],
                    "ffn_down_exps",
                )?;
                if let Some(ffn_down_exps_scale) = manifest_tensor(
                    manifest,
                    NativeTensorRole::FfnDownExpsScale,
                    Some(layer_index),
                ) {
                    expect_vector_shape(
                        ffn_down_exps_scale,
                        moe_dims.expert_count,
                        "ffn_down_exps_scale",
                    )?;
                }
            }
        }

        let dense_intermediate_dim = if let Some(ffn_gate_up_packed) = manifest_tensor(
            manifest,
            NativeTensorRole::FfnGateUpPacked,
            Some(layer_index),
        ) {
            let (rows, cols) = matrix_shape(ffn_gate_up_packed).ok_or_else(|| {
                NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_gate_up_packed must be a rank-2 matrix",
                        layer_index
                    ),
                }
            })?;
            if uses_packed_u32_storage(ffn_gate_up_packed) {
                let expected_cols = expected_packed_cols(hidden_size, ffn_gate_up_packed)?;
                if cols != expected_cols {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} tensor ffn_gate_up_packed must have packed quantized shape [rows, {}], got {:?}",
                            layer_index, expected_cols, ffn_gate_up_packed.shape
                        ),
                    });
                }
            } else if cols != hidden_size {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_gate_up_packed must have hidden_size {} columns, got {:?}",
                        layer_index, hidden_size, ffn_gate_up_packed.shape
                    ),
                });
            }
            if !rows.is_multiple_of(2) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_gate_up_packed row count must be even, got {}",
                        layer_index, rows
                    ),
                });
            }
            rows / 2
        } else if manifest_tensor(manifest, NativeTensorRole::FfnGate, Some(layer_index)).is_some()
        {
            let ffn_gate = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnGate,
                "ffn_gate",
            )?;
            let gate_shape =
                matrix_shape(ffn_gate).ok_or_else(|| NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_gate must be a rank-2 matrix",
                        layer_index
                    ),
                })?;
            let ffn_up = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnUp,
                "ffn_up",
            )?;
            let up_shape =
                matrix_shape(ffn_up).ok_or_else(|| NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_up must be a rank-2 matrix",
                        layer_index
                    ),
                })?;
            if uses_packed_u32_storage(ffn_gate) {
                let expected_cols = expected_packed_cols(hidden_size, ffn_gate)?;
                if gate_shape.1 != expected_cols {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} tensor ffn_gate must have packed quantized shape [rows, {}], got {:?}",
                            layer_index, expected_cols, ffn_gate.shape
                        ),
                    });
                }
            } else if gate_shape.1 != hidden_size {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_gate must have hidden_size {} columns, got {:?}",
                        layer_index, hidden_size, ffn_gate.shape
                    ),
                });
            }
            if uses_packed_u32_storage(ffn_up) {
                let expected_cols = expected_packed_cols(hidden_size, ffn_up)?;
                if up_shape.1 != expected_cols {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} tensor ffn_up must have packed quantized shape [rows, {}], got {:?}",
                            layer_index, expected_cols, ffn_up.shape
                        ),
                    });
                }
            } else if up_shape.1 != hidden_size {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_up must have hidden_size {} columns, got {:?}",
                        layer_index, hidden_size, ffn_up.shape
                    ),
                });
            }
            if gate_shape.0 != up_shape.0 {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensors ffn_gate and ffn_up must agree on intermediate rows, got {:?} and {:?}",
                        layer_index, ffn_gate.shape, ffn_up.shape
                    ),
                });
            }
            gate_shape.0
        } else {
            0
        };

        if let (Some(ffn_down), Some(ffn_down_shape)) = (
            manifest_tensor(manifest, NativeTensorRole::FfnDown, Some(layer_index)),
            ffn_down_shape,
        ) {
            if uses_packed_u32_storage(ffn_down) {
                let expected_cols = expected_packed_cols(dense_intermediate_dim, ffn_down)?;
                if ffn_down_shape.1 != expected_cols {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} tensor ffn_down must have packed quantized shape [rows, {}], got {:?}",
                            layer_index, expected_cols, ffn_down.shape
                        ),
                    });
                }
            } else if ffn_down_shape.1 != dense_intermediate_dim {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_down must have intermediate_dim {} columns, got {:?}",
                        layer_index, dense_intermediate_dim, ffn_down.shape
                    ),
                });
            }
        }

        if let Some(shared_gate_inp) = manifest_tensor(
            manifest,
            NativeTensorRole::FfnSharedExpertGateInp,
            Some(layer_index),
        ) {
            let moe_dims = resolved_moe_dims(manifest)?;
            expect_matrix_shape(
                shared_gate_inp,
                1,
                hidden_size,
                "ffn_shared_expert_gate_inp",
            )?;
            let shared_gate = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnSharedExpertGate,
                "ffn_shared_expert_gate",
            )?;
            expect_matrix_shape(
                shared_gate,
                moe_dims.expert_intermediate_size,
                hidden_size,
                "ffn_shared_expert_gate",
            )?;
            let shared_up = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnSharedExpertUp,
                "ffn_shared_expert_up",
            )?;
            expect_matrix_shape(
                shared_up,
                moe_dims.expert_intermediate_size,
                hidden_size,
                "ffn_shared_expert_up",
            )?;
            let shared_down = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnSharedExpertDown,
                "ffn_shared_expert_down",
            )?;
            expect_matrix_shape(
                shared_down,
                hidden_size,
                moe_dims.expert_intermediate_size,
                "ffn_shared_expert_down",
            )?;
        }
    }

    Ok(())
}

fn validate_gpt_oss_mxfp4_tensor_shapes(
    manifest: &NativeModelManifest,
    layer_index: u32,
) -> Result<(), NativeModelError> {
    let moe_dims = resolved_moe_dims(manifest)?;
    if !manifest.hidden_size.is_multiple_of(32)
        || !moe_dims.expert_intermediate_size.is_multiple_of(32)
    {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {layer_index} GPT-OSS MXFP4 dimensions must be divisible by group size 32"
            ),
        });
    }

    let expected_gate_up_scales = vec![
        moe_dims.expert_count,
        moe_dims.expert_intermediate_size.saturating_mul(2),
        u64::from(manifest.hidden_size / 32),
    ];
    let expected_down_scales = vec![
        moe_dims.expert_count,
        u64::from(manifest.hidden_size),
        moe_dims.expert_intermediate_size / 32,
    ];

    for (blocks_role, scales_role, label, expected_scales) in [
        (
            NativeTensorRole::FfnGateUpExpsMxfp4Blocks,
            NativeTensorRole::FfnGateUpExpsMxfp4Scales,
            "ffn_gate_up_exps_mxfp4",
            expected_gate_up_scales,
        ),
        (
            NativeTensorRole::FfnDownExpsMxfp4Blocks,
            NativeTensorRole::FfnDownExpsMxfp4Scales,
            "ffn_down_exps_mxfp4",
            expected_down_scales,
        ),
    ] {
        let blocks = required_layer_tensor_spec(
            manifest,
            layer_index,
            blocks_role,
            &format!("{label}_blocks"),
        )?;
        let scales = required_layer_tensor_spec(
            manifest,
            layer_index,
            scales_role,
            &format!("{label}_scales"),
        )?;
        if blocks.dtype != NativeTensorDataType::U8 || scales.dtype != NativeTensorDataType::U8 {
            return Err(NativeModelError::InvalidManifest {
                message: format!("layer {layer_index} {label} blocks and scales must use u8"),
            });
        }
        let mut expected_blocks = expected_scales.clone();
        expected_blocks.push(16);
        if scales.shape != expected_scales || blocks.shape != expected_blocks {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {layer_index} {label} must have block shape {expected_blocks:?} and scale shape {expected_scales:?}, got {:?} and {:?}",
                    blocks.shape, scales.shape
                ),
            });
        }
    }

    Ok(())
}

fn validate_glm_mla_attention_tensor_shapes(
    manifest: &NativeModelManifest,
    layer_index: u32,
) -> Result<(), NativeModelError> {
    let hidden_size = u64::from(manifest.hidden_size);
    let head_count = u64::from(manifest.attention_head_count);
    let q_lora_rank = u64::from(manifest.mla_attention.q_lora_rank.ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "mla_attention.q_lora_rank must be configured".to_string(),
        }
    })?);
    let kv_lora_rank = u64::from(manifest.mla_attention.kv_lora_rank.ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "mla_attention.kv_lora_rank must be configured".to_string(),
        }
    })?);
    let qk_nope_head_dim = u64::from(manifest.mla_attention.qk_nope_head_dim.ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "mla_attention.qk_nope_head_dim must be configured".to_string(),
        }
    })?);
    let qk_rope_head_dim = u64::from(manifest.mla_attention.qk_rope_head_dim.ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "mla_attention.qk_rope_head_dim must be configured".to_string(),
        }
    })?);
    let value_head_dim = u64::from(manifest.mla_attention.value_head_dim.ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "mla_attention.value_head_dim must be configured".to_string(),
        }
    })?);
    let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;

    let attention_qa = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionQa,
        "attention_qa",
    )?;
    expect_matrix_shape(attention_qa, q_lora_rank, hidden_size, "attention_qa")?;
    let attention_qa_norm = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionQaNorm,
        "attention_qa_norm",
    )?;
    expect_vector_shape(attention_qa_norm, q_lora_rank, "attention_qa_norm")?;
    let attention_qb = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionQb,
        "attention_qb",
    )?;
    expect_matrix_shape(
        attention_qb,
        head_count * q_head_dim,
        q_lora_rank,
        "attention_qb",
    )?;
    let attention_kv_a = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionKvA,
        "attention_kv_a",
    )?;
    expect_matrix_shape(
        attention_kv_a,
        kv_lora_rank + qk_rope_head_dim,
        hidden_size,
        "attention_kv_a",
    )?;
    let attention_kv_a_norm = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionKvANorm,
        "attention_kv_a_norm",
    )?;
    expect_vector_shape(attention_kv_a_norm, kv_lora_rank, "attention_kv_a_norm")?;
    if let Some(attention_kv_b) =
        manifest_tensor(manifest, NativeTensorRole::AttentionKvB, Some(layer_index))
    {
        expect_matrix_shape(
            attention_kv_b,
            head_count * (qk_nope_head_dim + value_head_dim),
            kv_lora_rank,
            "attention_kv_b",
        )?;
    } else {
        let attention_embed_q = required_layer_tensor_spec(
            manifest,
            layer_index,
            NativeTensorRole::AttentionEmbedQ,
            "attention_embed_q",
        )?;
        expect_tensor_shape(
            attention_embed_q,
            &[head_count, kv_lora_rank, qk_nope_head_dim],
            "attention_embed_q",
        )?;
        let attention_unembed_out = required_layer_tensor_spec(
            manifest,
            layer_index,
            NativeTensorRole::AttentionUnembedOut,
            "attention_unembed_out",
        )?;
        expect_tensor_shape(
            attention_unembed_out,
            &[head_count, value_head_dim, kv_lora_rank],
            "attention_unembed_out",
        )?;
    }
    let attention_o = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionO,
        "attention_o",
    )?;
    expect_matrix_shape(
        attention_o,
        hidden_size,
        head_count * value_head_dim,
        "attention_o",
    )
}

/// DeepSeek V4 attention tensor shapes.
///
/// Layout (vllm `DeepseekV4Attention` / llama.cpp `DeepseekV4Model`):
/// `wq_a` [q_lora_rank, hidden], `q_norm` [q_lora_rank],
/// `wq_b` [num_heads * head_dim, q_lora_rank], fused `wkv` [head_dim, hidden]
/// (single KV head), `kv_norm` [head_dim], grouped `wo_a`
/// [o_groups * o_lora_rank, num_heads * head_dim / o_groups], and
/// `wo_b` [hidden, o_groups * o_lora_rank].
fn validate_deepseek_v4_attention_tensor_shapes(
    manifest: &NativeModelManifest,
    layer_index: u32,
) -> Result<(), NativeModelError> {
    let hidden_size = u64::from(manifest.hidden_size);
    let head_count = u64::from(manifest.attention_head_count);
    let attention = &manifest.deepseek_v4.attention;
    let head_dim =
        u64::from(
            attention
                .head_dim
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "deepseek_v4.attention.head_dim must be configured".to_string(),
                })?,
        );
    let q_lora_rank =
        u64::from(
            attention
                .q_lora_rank
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "deepseek_v4.attention.q_lora_rank must be configured".to_string(),
                })?,
        );
    let o_lora_rank =
        u64::from(
            attention
                .o_lora_rank
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "deepseek_v4.attention.o_lora_rank must be configured".to_string(),
                })?,
        );
    let o_groups =
        u64::from(
            attention
                .o_groups
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "deepseek_v4.attention.o_groups must be configured".to_string(),
                })?,
        );
    let Some(grouped_o_rows) = o_groups.checked_mul(o_lora_rank) else {
        return Err(NativeModelError::InvalidManifest {
            message: "deepseek_v4 o_groups * o_lora_rank overflowed".to_string(),
        });
    };
    let attention_out_dim = head_count * head_dim;
    if o_groups == 0 || !attention_out_dim.is_multiple_of(o_groups) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "deepseek_v4 num_heads * head_dim ({attention_out_dim}) must be divisible by o_groups ({o_groups})"
            ),
        });
    }

    let attention_qa = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionQa,
        "attention_qa",
    )?;
    expect_matrix_shape(attention_qa, q_lora_rank, hidden_size, "attention_qa")?;
    let attention_qa_norm = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionQaNorm,
        "attention_qa_norm",
    )?;
    expect_vector_shape(attention_qa_norm, q_lora_rank, "attention_qa_norm")?;
    let attention_qb = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionQb,
        "attention_qb",
    )?;
    expect_matrix_shape(attention_qb, attention_out_dim, q_lora_rank, "attention_qb")?;
    let attention_kv = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionKv,
        "attention_kv",
    )?;
    expect_matrix_shape(attention_kv, head_dim, hidden_size, "attention_kv")?;
    let attention_kv_norm = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionKvNorm,
        "attention_kv_norm",
    )?;
    expect_vector_shape(attention_kv_norm, head_dim, "attention_kv_norm")?;
    let attention_out_a = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionOutA,
        "attention_out_a",
    )?;
    // AXQ/mlx-lm checkpoints store `attn.wo_a` in its native grouped 3-D
    // layout `[o_groups, o_lora_rank, H*D/o_groups]`; raw HF / GGUF-derived
    // checkpoints carry the flattened `[o_groups*o_lora_rank, H*D/o_groups]`
    // matrix. Both feed the runtime's reshape-to-3-D grouped projection.
    let group_width = attention_out_dim / o_groups;
    if attention_out_a.shape.len() == 3 && !uses_packed_u32_storage(attention_out_a) {
        let expected = [o_groups, o_lora_rank, group_width];
        if attention_out_a.shape != expected {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "tensor attention_out_a must have shape {expected:?}, got {:?}",
                    attention_out_a.shape
                ),
            });
        }
    } else {
        expect_matrix_shape(
            attention_out_a,
            grouped_o_rows,
            group_width,
            "attention_out_a",
        )?;
    }
    let attention_out_b = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionOutB,
        "attention_out_b",
    )?;
    expect_matrix_shape(
        attention_out_b,
        hidden_size,
        grouped_o_rows,
        "attention_out_b",
    )
}

fn validate_per_layer_input_tensor_shapes(
    manifest: &NativeModelManifest,
    hidden_size: u64,
    vocab_size: u64,
) -> Result<(), NativeModelError> {
    let per_layer_dim = u64::from(manifest.hidden_size_per_layer_input);
    if per_layer_dim == 0 {
        return Ok(());
    }
    let stacked_dim = u64::from(manifest.layer_count)
        .checked_mul(per_layer_dim)
        .ok_or_else(|| NativeModelError::InvalidManifest {
            message: "per-layer input stacked dimension overflowed".to_string(),
        })?;
    let per_layer_vocab_size = match manifest.vocab_size_per_layer_input {
        Some(0) => {
            return Err(NativeModelError::InvalidManifest {
                message: "vocab_size_per_layer_input must be > 0 when configured".to_string(),
            });
        }
        Some(value) => u64::from(value),
        None => vocab_size,
    };

    let per_layer_embed = required_global_tensor_spec(
        manifest,
        NativeTensorRole::PerLayerEmbedding,
        "per_layer_embed",
    )?;
    expect_matrix_shape(
        per_layer_embed,
        per_layer_vocab_size,
        stacked_dim,
        "per_layer_embed",
    )?;
    let per_layer_model_proj = required_global_tensor_spec(
        manifest,
        NativeTensorRole::PerLayerModelProjection,
        "per_layer_model_proj",
    )?;
    expect_matrix_shape(
        per_layer_model_proj,
        stacked_dim,
        hidden_size,
        "per_layer_model_proj",
    )?;
    let per_layer_proj_norm = required_global_tensor_spec(
        manifest,
        NativeTensorRole::PerLayerProjectionNorm,
        "per_layer_proj_norm",
    )?;
    expect_vector_shape(per_layer_proj_norm, per_layer_dim, "per_layer_proj_norm")?;

    for layer_index in 0..manifest.layer_count {
        let per_layer_gate = required_layer_tensor_spec(
            manifest,
            layer_index,
            NativeTensorRole::PerLayerInputGate,
            "per_layer_input_gate",
        )?;
        expect_matrix_shape(
            per_layer_gate,
            per_layer_dim,
            hidden_size,
            "per_layer_input_gate",
        )?;
        let per_layer_projection = required_layer_tensor_spec(
            manifest,
            layer_index,
            NativeTensorRole::PerLayerInputProjection,
            "per_layer_projection",
        )?;
        expect_matrix_shape(
            per_layer_projection,
            hidden_size,
            per_layer_dim,
            "per_layer_projection",
        )?;
        let per_layer_post_norm = required_layer_tensor_spec(
            manifest,
            layer_index,
            NativeTensorRole::PerLayerInputPostNorm,
            "post_per_layer_input_norm",
        )?;
        expect_vector_shape(
            per_layer_post_norm,
            hidden_size,
            "post_per_layer_input_norm",
        )?;
    }

    Ok(())
}

fn validate_manifest_layer_index_list(
    manifest: &NativeModelManifest,
    layer_indices: &[u32],
    field_name: &str,
) -> Result<(), NativeModelError> {
    for &layer_index in layer_indices {
        if layer_index >= manifest.layer_count {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "{} contains out-of-range layer index {} (layer_count={})",
                    field_name, layer_index, manifest.layer_count
                ),
            });
        }
    }

    Ok(())
}

fn validate_kv_cache_quantization(manifest: &NativeModelManifest) -> Result<(), NativeModelError> {
    let Some(kv_cache_quantization) = &manifest.kv_cache_quantization else {
        return Ok(());
    };
    let layer_count = manifest.layer_count as usize;
    if kv_cache_quantization.layer_bits.len() != layer_count {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "kv_cache_quantization.layer_bits length {} must equal layer_count {}",
                kv_cache_quantization.layer_bits.len(),
                manifest.layer_count
            ),
        });
    }
    if kv_cache_quantization.layer_group_sizes.len() != layer_count {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "kv_cache_quantization.layer_group_sizes length {} must equal layer_count {}",
                kv_cache_quantization.layer_group_sizes.len(),
                manifest.layer_count
            ),
        });
    }
    for (layer, (&bits, &group_size)) in kv_cache_quantization
        .layer_bits
        .iter()
        .zip(kv_cache_quantization.layer_group_sizes.iter())
        .enumerate()
    {
        if !matches!(bits, 4 | 6 | 8 | 16) {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "kv_cache_quantization.layer_bits[{layer}] must be one of 4, 6, 8, 16 (16 = full precision), got {bits}"
                ),
            });
        }
        if bits < 16 && !matches!(group_size, 32 | 64 | 128) {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "kv_cache_quantization.layer_group_sizes[{layer}] must be one of 32, 64, 128 when bits < 16, got {group_size}"
                ),
            });
        }
    }
    Ok(())
}

fn validate_interleaved_attention_metadata(
    manifest: &NativeModelManifest,
) -> Result<(), NativeModelError> {
    if let Some(rope_theta_swa) = manifest.rope_theta_swa {
        if rope_theta_swa == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!("rope_theta_swa must be > 0, got {rope_theta_swa}"),
            });
        }
    }
    if let Some(global_head_dim) = manifest.global_head_dim {
        if global_head_dim == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: "global_head_dim must be > 0".to_string(),
            });
        }
    }
    if let Some(global_kv_head_count) = manifest.global_kv_head_count {
        if global_kv_head_count == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: "global_kv_head_count must be > 0".to_string(),
            });
        }
        if manifest.global_head_dim.is_none() {
            return Err(NativeModelError::InvalidManifest {
                message: "global_kv_head_count requires global_head_dim".to_string(),
            });
        }
        if !manifest
            .attention_head_count
            .is_multiple_of(global_kv_head_count)
        {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "attention_head_count {} must be divisible by global_kv_head_count {}",
                    manifest.attention_head_count, global_kv_head_count
                ),
            });
        }
    }
    if let Some(sliding_window_size) = manifest.sliding_window_size {
        if sliding_window_size == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: "sliding_window_size must be > 0".to_string(),
            });
        }
    }

    if !manifest.layer_types.is_empty() {
        if manifest.layer_types.len() != manifest.layer_count as usize {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer_types must contain one entry per layer, got {} for layer_count {}",
                    manifest.layer_types.len(),
                    manifest.layer_count
                ),
            });
        }
        let allow_nemotron_kinds = manifest.model_family == "nemotron_h";
        for (idx, layer_type) in manifest.layer_types.iter().enumerate() {
            let ok = matches!(layer_type.as_str(), "sliding_attention" | "full_attention")
                || (allow_nemotron_kinds
                    && matches!(
                        layer_type.as_str(),
                        "mamba" | "attention" | "moe" | "mlp" | "M" | "E" | "*" | "-"
                    ));
            if !ok {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer_types[{idx}] must be sliding_attention or full_attention, got {layer_type:?}"
                    ),
                });
            }
        }
    }

    for (&layer_index, &source_layer) in &manifest.kv_shared_source_layers {
        if layer_index >= manifest.layer_count || source_layer >= manifest.layer_count {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "kv_shared_source_layers contains out-of-range mapping {} -> {} (layer_count={})",
                    layer_index, source_layer, manifest.layer_count
                ),
            });
        }
        if source_layer >= layer_index {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "kv_shared_source_layers layer {} must reference an earlier source layer, got {}",
                    layer_index, source_layer
                ),
            });
        }
        if !manifest.layer_types.is_empty() {
            let layer_type = &manifest.layer_types[layer_index as usize];
            let source_type = &manifest.layer_types[source_layer as usize];
            if layer_type != source_type {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "kv_shared_source_layers layer {} type {:?} cannot reuse source {} type {:?}",
                        layer_index, layer_type, source_layer, source_type
                    ),
                });
            }
        }
    }

    Ok(())
}

fn require_positive_field(value: Option<u32>, field_name: &str) -> Result<u32, NativeModelError> {
    match value {
        Some(0) => Err(NativeModelError::InvalidManifest {
            message: format!("{field_name} must be > 0"),
        }),
        None => Err(NativeModelError::InvalidManifest {
            message: format!("{field_name} is required when its feature is enabled"),
        }),
        Some(value) => Ok(value),
    }
}

/// Validate the non-tensor invariants consumed unconditionally by the
/// dedicated Muse Glimmer forward route.
///
/// Muse manifests created before the route learned its scalar and iRoPE
/// contract can still deserialize because these fields are optional for every
/// other family. Loading one with generic defaults would produce plausible but
/// numerically wrong text, so this family must fail closed instead.
fn validate_muse_glimmer_manifest_contract(
    manifest: &NativeModelManifest,
) -> Result<(), NativeModelError> {
    for (present, field_name) in [
        (manifest.rms_norm_eps.is_some(), "rms_norm_eps"),
        (manifest.post_norm_eps.is_some(), "post_norm_eps"),
        (
            manifest.attention_scale_multiplier.is_some(),
            "attention_scale_multiplier",
        ),
        (manifest.final_logits_scale.is_some(), "final_logits_scale"),
        (
            manifest.final_logit_softcapping.is_some(),
            "final_logit_softcapping",
        ),
        (manifest.rope_theta.is_some(), "rope_theta"),
        (manifest.rope_theta_swa.is_some(), "rope_theta_swa"),
        (
            manifest.sliding_window_size.is_some(),
            "sliding_window_size",
        ),
    ] {
        if !present {
            return Err(NativeModelError::InvalidManifest {
                message: format!("muse_glimmer requires {field_name}"),
            });
        }
    }
    if manifest.layer_types.len() != manifest.layer_count as usize
        || !manifest
            .layer_types
            .iter()
            .any(|kind| kind == "sliding_attention")
        || !manifest
            .layer_types
            .iter()
            .any(|kind| kind == "full_attention")
    {
        return Err(NativeModelError::InvalidManifest {
            message: "muse_glimmer requires one interleaved sliding_attention/full_attention layer_type per layer"
                .to_string(),
        });
    }
    if manifest.attn_output_gate {
        return Err(NativeModelError::InvalidManifest {
            message:
                "muse_glimmer uses AttentionOutputGate tensors and must not set attn_output_gate"
                    .to_string(),
        });
    }
    if manifest.hidden_states_scale.is_some() {
        return Err(NativeModelError::InvalidManifest {
            message: "muse_glimmer must not set hidden_states_scale".to_string(),
        });
    }
    if manifest.linear_attention.is_enabled()
        || manifest.mla_attention.is_enabled()
        || manifest.moe.is_enabled()
        || !manifest.kv_shared_source_layers.is_empty()
        || !manifest.attention_value_from_key_layers.is_empty()
    {
        return Err(NativeModelError::InvalidManifest {
            message: "muse_glimmer does not support linear attention, MLA, MoE, or shared/value-from-key KV layouts"
                .to_string(),
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NativeSplitAttentionDims {
    q_rows: u64,
    kv_rows: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NativeLinearAttentionDims {
    num_value_heads: u64,
    num_key_heads: u64,
    key_head_dim: u64,
    value_head_dim: u64,
    conv_kernel_dim: u64,
    key_dim: u64,
    value_dim: u64,
    conv_dim: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NativeMoeDims {
    expert_count: u64,
    experts_per_token: u64,
    expert_intermediate_size: u64,
}

fn resolved_linear_attention_dims(
    manifest: &NativeModelManifest,
) -> Result<NativeLinearAttentionDims, NativeModelError> {
    let config = &manifest.linear_attention;
    let num_value_heads =
        u64::from(
            config
                .num_value_heads
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "linear_attention.num_value_heads must be configured".to_string(),
                })?,
        );
    let num_key_heads =
        u64::from(
            config
                .num_key_heads
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "linear_attention.num_key_heads must be configured".to_string(),
                })?,
        );
    let key_head_dim =
        u64::from(
            config
                .key_head_dim
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "linear_attention.key_head_dim must be configured".to_string(),
                })?,
        );
    let value_head_dim =
        u64::from(
            config
                .value_head_dim
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "linear_attention.value_head_dim must be configured".to_string(),
                })?,
        );
    let conv_kernel_dim =
        u64::from(
            config
                .conv_kernel_dim
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "linear_attention.conv_kernel_dim must be configured".to_string(),
                })?,
        );
    let key_dim = num_key_heads.checked_mul(key_head_dim).ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "linear attention key_dim overflowed".to_string(),
        }
    })?;
    let value_dim = num_value_heads.checked_mul(value_head_dim).ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "linear attention value_dim overflowed".to_string(),
        }
    })?;
    let conv_dim = key_dim
        .checked_mul(2)
        .and_then(|twice_key_dim| twice_key_dim.checked_add(value_dim))
        .ok_or_else(|| NativeModelError::InvalidManifest {
            message: "linear attention conv_dim overflowed".to_string(),
        })?;

    Ok(NativeLinearAttentionDims {
        num_value_heads,
        num_key_heads,
        key_head_dim,
        value_head_dim,
        conv_kernel_dim,
        key_dim,
        value_dim,
        conv_dim,
    })
}

fn resolved_moe_dims(manifest: &NativeModelManifest) -> Result<NativeMoeDims, NativeModelError> {
    let config = &manifest.moe;
    Ok(NativeMoeDims {
        expert_count: u64::from(config.expert_count.ok_or_else(|| {
            NativeModelError::InvalidManifest {
                message: "moe.expert_count must be configured".to_string(),
            }
        })?),
        experts_per_token: u64::from(config.experts_per_token.ok_or_else(|| {
            NativeModelError::InvalidManifest {
                message: "moe.experts_per_token must be configured".to_string(),
            }
        })?),
        expert_intermediate_size: u64::from(config.expert_intermediate_size.ok_or_else(|| {
            NativeModelError::InvalidManifest {
                message: "moe.expert_intermediate_size must be configured".to_string(),
            }
        })?),
    })
}

/// Per-layer attention projection widths as (q_rows, kv_rows).
///
/// Q rows widen with the layer's configured head dim. Full-attention layers
/// use `global_kv_head_count` when it is explicit. Older manifests can omit
/// that field, so prefer the projection's recorded tensor geometry before
/// falling back to the legacy constant-total-KV-width rule.
fn configured_attention_projection_dims(
    manifest: &NativeModelManifest,
    layer_index: u32,
) -> (u64, u64) {
    let head_dim = configured_attention_head_dim(manifest, layer_index);
    let q_rows = u64::from(manifest.attention_head_count) * head_dim;
    let is_full_attention = manifest
        .layer_types
        .get(layer_index as usize)
        .is_some_and(|layer_type| layer_type == "full_attention");
    let kv_rows = if is_full_attention {
        manifest
            .global_kv_head_count
            .map(|count| u64::from(count) * head_dim)
            .or_else(|| inferred_attention_kv_rows(manifest, layer_index, q_rows))
            .unwrap_or_else(|| {
                u64::from(manifest.kv_head_count) * u64::from(manifest.attention_head_dim)
            })
    } else {
        u64::from(manifest.kv_head_count) * u64::from(manifest.attention_head_dim)
    };
    (q_rows, kv_rows)
}

/// Resolve the K/V projection width carried by a layer's tensor metadata.
///
/// This is the compatibility path for manifests generated before
/// `global_kv_head_count` existed. Gemma 4 full-attention layers do not all
/// preserve the base layer's total KV width (for example E2B uses one
/// 256-wide sliding KV head and one 512-wide global KV head), so the tensor
/// rows are the only authoritative fallback.
fn inferred_attention_kv_rows(
    manifest: &NativeModelManifest,
    layer_index: u32,
    q_rows: u64,
) -> Option<u64> {
    if let Some(attention_k) =
        manifest_tensor(manifest, NativeTensorRole::AttentionK, Some(layer_index))
    {
        return matrix_shape(attention_k).map(|(rows, _)| rows);
    }

    let packed = manifest_tensor(
        manifest,
        NativeTensorRole::AttentionQkvPacked,
        Some(layer_index),
    )?;
    let (packed_rows, _) = matrix_shape(packed)?;
    let packed_q_rows = if manifest.attn_output_gate {
        q_rows.checked_mul(2)?
    } else {
        q_rows
    };
    let remaining = packed_rows.checked_sub(packed_q_rows)?;
    remaining
        .is_multiple_of(2)
        .then_some(remaining / 2)
        .filter(|rows| *rows > 0)
}

fn configured_attention_head_dim(manifest: &NativeModelManifest, layer_index: u32) -> u64 {
    if manifest
        .layer_types
        .get(layer_index as usize)
        .is_some_and(|layer_type| layer_type == "full_attention")
    {
        u64::from(
            manifest
                .global_head_dim
                .unwrap_or(manifest.attention_head_dim),
        )
    } else {
        u64::from(manifest.attention_head_dim)
    }
}

fn resolved_split_attention_dims(
    manifest: &NativeModelManifest,
    layer_index: u32,
) -> Result<NativeSplitAttentionDims, NativeModelError> {
    let attention_q = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionQ,
        "attention_q",
    )?;
    let attention_k = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionK,
        "attention_k",
    )?;
    let (q_rows, q_cols) =
        matrix_shape(attention_q).ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!(
                "layer {} tensor attention_q must be a rank-2 matrix",
                layer_index
            ),
        })?;
    let (k_rows, k_cols) =
        matrix_shape(attention_k).ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!(
                "layer {} tensor attention_k must be a rank-2 matrix",
                layer_index
            ),
        })?;
    let hidden_size = u64::from(manifest.hidden_size);
    // Apply raw-column checks per tensor. Mixed-precision plans can preserve Q
    // while packing K/V/O, so one projection's storage must not determine the
    // layout validation applied to another projection.
    if !uses_packed_u32_storage(attention_q) && q_cols != hidden_size {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} tensor attention_q must have shape [q_rows, {}], got {:?}",
                layer_index, hidden_size, attention_q.shape
            ),
        });
    }
    if !uses_packed_u32_storage(attention_k) && k_cols != hidden_size {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} tensor attention_k must have shape [kv_rows, {}], got {:?}",
                layer_index, hidden_size, attention_k.shape
            ),
        });
    }

    let mut head_dim = None;
    if let Some(attention_q_norm) = manifest_tensor(
        manifest,
        NativeTensorRole::AttentionQNorm,
        Some(layer_index),
    ) {
        let q_norm_dim =
            vector_shape(attention_q_norm).ok_or_else(|| NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor attention_q_norm must be a rank-1 vector",
                    layer_index
                ),
            })?;
        head_dim = Some(q_norm_dim);
    }
    if let Some(attention_k_norm) = manifest_tensor(
        manifest,
        NativeTensorRole::AttentionKNorm,
        Some(layer_index),
    ) {
        let k_norm_dim =
            vector_shape(attention_k_norm).ok_or_else(|| NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor attention_k_norm must be a rank-1 vector",
                    layer_index
                ),
            })?;
        if let Some(existing) = head_dim {
            if existing != k_norm_dim {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} attention_q_norm and attention_k_norm must agree on head_dim, got {} vs {}",
                        layer_index, existing, k_norm_dim
                    ),
                });
            }
        } else {
            head_dim = Some(k_norm_dim);
        }
    }
    let head_dim = head_dim.unwrap_or_else(|| configured_attention_head_dim(manifest, layer_index));
    if head_dim == 0 {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} resolved attention head_dim must be > 0",
                layer_index
            ),
        });
    }
    // When attn_output_gate is enabled, q_proj encodes both queries and gate
    // values, so the effective row count for head derivation is halved.
    let effective_q_rows = if manifest.attn_output_gate {
        if !q_rows.is_multiple_of(2) {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} attention_q rows {} must be even when attn_output_gate is enabled",
                    layer_index, q_rows
                ),
            });
        }
        q_rows / 2
    } else {
        q_rows
    };
    if !effective_q_rows.is_multiple_of(head_dim) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} attention_q rows {} (effective {}) must be divisible by head_dim {}",
                layer_index, q_rows, effective_q_rows, head_dim
            ),
        });
    }
    if !k_rows.is_multiple_of(head_dim) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} attention_k rows {} must be divisible by head_dim {}",
                layer_index, k_rows, head_dim
            ),
        });
    }
    let q_heads = effective_q_rows / head_dim;
    let kv_heads = k_rows / head_dim;
    let expected_q_heads = u64::from(manifest.attention_head_count);
    // Resolve the per-layer KV head count from the explicit global geometry
    // when available; legacy manifests derive it from their fixed total width.
    let (_, configured_kv_rows) = configured_attention_projection_dims(manifest, layer_index);
    if !configured_kv_rows.is_multiple_of(head_dim) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} configured KV projection width {} must be divisible by head_dim {}",
                layer_index, configured_kv_rows, head_dim
            ),
        });
    }
    let expected_kv_heads = configured_kv_rows / head_dim;
    if q_heads != expected_q_heads || kv_heads != expected_kv_heads {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} split attention head counts must match manifest q_heads={} kv_heads={}, resolved q_heads={} kv_heads={}",
                layer_index, expected_q_heads, expected_kv_heads, q_heads, kv_heads
            ),
        });
    }
    // q_heads/kv_heads are non-zero here: both equal manifest-derived counts
    // already validated as positive.
    if q_heads < kv_heads || !q_heads.is_multiple_of(kv_heads) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} requires q_heads >= kv_heads and divisible; resolved q_heads={} kv_heads={}",
                layer_index, q_heads, kv_heads
            ),
        });
    }

    let kv_rows = if let Some(attention_v) =
        manifest_tensor(manifest, NativeTensorRole::AttentionV, Some(layer_index))
    {
        let (v_rows, v_cols) =
            matrix_shape(attention_v).ok_or_else(|| NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor attention_v must be a rank-2 matrix",
                    layer_index
                ),
            })?;
        if !uses_packed_u32_storage(attention_v) && v_cols != hidden_size {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor attention_v must have shape [kv_rows, {}], got {:?}",
                    layer_index, hidden_size, attention_v.shape
                ),
            });
        }
        if v_rows != k_rows {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} attention_k and attention_v must agree on row count, got {} vs {}",
                    layer_index, k_rows, v_rows
                ),
            });
        }
        v_rows
    } else if manifest
        .attention_value_from_key_layers
        .contains(&layer_index)
    {
        k_rows
    } else {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} must provide attention_v or be listed in attention_value_from_key_layers",
                layer_index
            ),
        });
    };

    Ok(NativeSplitAttentionDims { q_rows, kv_rows })
}

fn validate_q_only_attention_tensor(
    manifest: &NativeModelManifest,
    layer_index: u32,
    attention_q: &NativeTensorSpec,
) -> Result<(), NativeModelError> {
    let (q_rows, q_cols) =
        matrix_shape(attention_q).ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!(
                "layer {} tensor attention_q must be a rank-2 matrix",
                layer_index
            ),
        })?;
    let hidden_size = u64::from(manifest.hidden_size);
    if !uses_packed_u32_storage(attention_q) && q_cols != hidden_size {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} tensor attention_q must have shape [q_rows, {}], got {:?}",
                layer_index, hidden_size, attention_q.shape
            ),
        });
    }

    let head_dim = manifest_tensor(
        manifest,
        NativeTensorRole::AttentionQNorm,
        Some(layer_index),
    )
    .map(|q_norm| {
        vector_shape(q_norm).ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!(
                "layer {} tensor attention_q_norm must be a rank-1 vector",
                layer_index
            ),
        })
    })
    .transpose()?
    .unwrap_or_else(|| configured_attention_head_dim(manifest, layer_index));
    if head_dim == 0 {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} resolved attention head_dim must be > 0",
                layer_index
            ),
        });
    }
    let effective_q_rows = if manifest.attn_output_gate {
        if !q_rows.is_multiple_of(2) {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} attention_q rows {} must be even when attn_output_gate is enabled",
                    layer_index, q_rows
                ),
            });
        }
        q_rows / 2
    } else {
        q_rows
    };
    if effective_q_rows == 0 || !effective_q_rows.is_multiple_of(head_dim) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} attention_q rows {} (effective {}) must be divisible by head_dim {}",
                layer_index, q_rows, effective_q_rows, head_dim
            ),
        });
    }
    let q_heads = effective_q_rows / head_dim;
    let expected_q_heads = u64::from(manifest.attention_head_count);
    if q_heads != expected_q_heads {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} Q-only attention head count must match manifest q_heads={}, resolved q_heads={}",
                layer_index, expected_q_heads, q_heads
            ),
        });
    }
    expect_matrix_shape(attention_q, q_rows, hidden_size, "attention_q")
}

fn manifest_tensor(
    manifest: &NativeModelManifest,
    role: NativeTensorRole,
    layer_index: Option<u32>,
) -> Option<&NativeTensorSpec> {
    manifest
        .tensors
        .iter()
        .find(|tensor| tensor.role == role && tensor.layer_index == layer_index)
}

fn required_global_tensor_spec<'a>(
    manifest: &'a NativeModelManifest,
    role: NativeTensorRole,
    label: &str,
) -> Result<&'a NativeTensorSpec, NativeModelError> {
    manifest_tensor(manifest, role, None).ok_or_else(|| NativeModelError::InvalidManifest {
        message: format!("manifest is missing required global tensor role {}", label),
    })
}

fn required_layer_tensor_spec<'a>(
    manifest: &'a NativeModelManifest,
    layer_index: u32,
    role: NativeTensorRole,
    label: &str,
) -> Result<&'a NativeTensorSpec, NativeModelError> {
    manifest_tensor(manifest, role, Some(layer_index)).ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: format!(
                "layer {} is missing required tensor role {}",
                layer_index, label
            ),
        }
    })
}

fn matrix_shape(tensor: &NativeTensorSpec) -> Option<(u64, u64)> {
    (tensor.shape.len() == 2).then_some((*tensor.shape.first()?, *tensor.shape.get(1)?))
}

fn total_elements(tensor: &NativeTensorSpec) -> Option<u64> {
    tensor
        .shape
        .iter()
        .try_fold(1_u64, |acc, dim| acc.checked_mul(*dim))
}

fn vector_shape(tensor: &NativeTensorSpec) -> Option<u64> {
    (tensor.shape.len() == 1).then_some(*tensor.shape.first()?)
}

fn expect_vector_shape(
    tensor: &NativeTensorSpec,
    expected_len: u64,
    label: &str,
) -> Result<(), NativeModelError> {
    if tensor.shape == [expected_len] {
        Ok(())
    } else {
        Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} must have shape [{}], got {:?}",
                label, expected_len, tensor.shape
            ),
        })
    }
}

fn expect_matrix_shape(
    tensor: &NativeTensorSpec,
    expected_rows: u64,
    expected_cols: u64,
    label: &str,
) -> Result<(), NativeModelError> {
    // MLX affine weights pack logical columns into U32 words. GGUF block
    // dtypes retain their logical tensor shape even though their byte storage
    // is quantized.
    if uses_packed_u32_storage(tensor) {
        let Some((rows, cols)) = matrix_shape(tensor) else {
            return Err(NativeModelError::InvalidManifest {
                message: format!("tensor {} must be a rank-2 quantized matrix", label),
            });
        };
        let expected_packed_cols = expected_packed_cols(expected_cols, tensor)?;
        if rows == expected_rows && cols == expected_packed_cols {
            return Ok(());
        }
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} must have packed quantized shape [{}, {}], got {:?}",
                label, expected_rows, expected_packed_cols, tensor.shape
            ),
        });
    }
    if tensor.shape == [expected_rows, expected_cols] {
        Ok(())
    } else {
        Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} must have shape [{}, {}], got {:?}",
                label, expected_rows, expected_cols, tensor.shape
            ),
        })
    }
}

fn expect_tensor_shape(
    tensor: &NativeTensorSpec,
    expected_shape: &[u64],
    label: &str,
) -> Result<(), NativeModelError> {
    if uses_packed_u32_storage(tensor) {
        if tensor.shape.len() != expected_shape.len() {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "tensor {} must have rank {} for quantized shape, got {:?}",
                    label,
                    expected_shape.len(),
                    tensor.shape
                ),
            });
        }
        let mut expected = expected_shape.to_vec();
        let Some(expected_last) = expected.last_mut() else {
            return Ok(());
        };
        *expected_last = expected_packed_cols(*expected_last, tensor)?;
        if tensor.shape == expected {
            return Ok(());
        }
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} must have packed quantized shape {:?}, got {:?}",
                label, expected, tensor.shape
            ),
        });
    }
    if tensor.shape == expected_shape {
        Ok(())
    } else {
        Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} must have shape {:?}, got {:?}",
                label, expected_shape, tensor.shape
            ),
        })
    }
}

fn tensor_quantization_or_default(tensor: &NativeTensorSpec) -> NativeTensorQuantization {
    tensor.quantization.clone().unwrap_or_default()
}

fn uses_packed_u32_storage(tensor: &NativeTensorSpec) -> bool {
    tensor.source_quantized && tensor.dtype == NativeTensorDataType::U32
}

fn is_gguf_block_dtype(dtype: NativeTensorDataType) -> bool {
    matches!(
        dtype,
        NativeTensorDataType::Q4Km
            | NativeTensorDataType::Q5Km
            | NativeTensorDataType::Q6Km
            | NativeTensorDataType::Q8Zero
    )
}

fn validate_tensor_quantization(
    tensor: &NativeTensorSpec,
    tensor_format: NativeTensorFormat,
    allow_experimental_3bit: bool,
    allow_experimental_2bit: bool,
) -> Result<(), NativeModelError> {
    if is_gguf_block_dtype(tensor.dtype) {
        if tensor_format != NativeTensorFormat::Gguf {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "tensor {} uses GGUF block dtype {:?} but tensor_format is {:?}, expected gguf",
                    tensor.name, tensor.dtype, tensor_format
                ),
            });
        }
        if !tensor.source_quantized {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "tensor {} uses GGUF block dtype {:?} but source_quantized is false",
                    tensor.name, tensor.dtype
                ),
            });
        }
    }
    // DeepSeek V4's `ffn.gate.tid2eid` hash-routing table is a genuine I32
    // integer tensor riding the U32 container dtype (see convert_dtype), not
    // an MLX affine-quantized weight, so it is exempt from the packed-weight
    // invariant below.
    if tensor.dtype == NativeTensorDataType::U32
        && !tensor.source_quantized
        && tensor.role != NativeTensorRole::FfnGateTid2Eid
    {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} uses dtype u32 but source_quantized is false",
                tensor.name
            ),
        });
    }
    let Some(quantization) = &tensor.quantization else {
        // source_quantized is already proven true for GGUF block dtypes above.
        if !tensor.source_quantized
            || uses_packed_u32_storage(tensor)
            || is_gguf_block_dtype(tensor.dtype)
        {
            return Ok(());
        }
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} is source_quantized but dtype is {:?}, expected u32 or a supported GGUF block dtype",
                tensor.name, tensor.dtype
            ),
        });
    };
    if !tensor.source_quantized {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} declares quantization but source_quantized is false",
                tensor.name
            ),
        });
    }
    if tensor.dtype != NativeTensorDataType::U32 {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} declares quantization but dtype is {:?}, expected u32",
                tensor.name, tensor.dtype
            ),
        });
    }
    if quantization.group_size == 0 {
        return Err(NativeModelError::InvalidManifest {
            message: format!("tensor {} quantization group_size must be > 0", tensor.name),
        });
    }
    match quantization.mode.as_str() {
        "affine" => {
            if !SUPPORTED_MLX_AFFINE_QUANTIZATION_BITS.contains(&quantization.bits) {
                if EXPERIMENTAL_MLX_AFFINE_QUANTIZATION_BITS.contains(&quantization.bits) {
                    if !allow_experimental_3bit {
                        return Err(NativeModelError::InvalidManifest {
                            message: format!(
                                "tensor {} quantization bits {} requires experimental gate (set {}=1)",
                                tensor.name, quantization.bits, AX_ENGINE_3BIT_EXPERIMENTAL_ENV
                            ),
                        });
                    }
                } else if EXPERIMENTAL_2BIT_MLX_AFFINE_QUANTIZATION_BITS
                    .contains(&quantization.bits)
                {
                    if !allow_experimental_2bit {
                        return Err(NativeModelError::InvalidManifest {
                            message: format!(
                                "tensor {} quantization bits {} requires experimental gate (set {}=1)",
                                tensor.name, quantization.bits, AX_ENGINE_2BIT_EXPERIMENTAL_ENV
                            ),
                        });
                    }
                } else {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "tensor {} quantization bits must be one of {:?}, got {}",
                            tensor.name, SUPPORTED_MLX_AFFINE_QUANTIZATION_BITS, quantization.bits
                        ),
                    });
                }
            }
        }
        "mxfp4" if quantization.group_size == 32 && quantization.bits == 4 => {}
        "mxfp4" => {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "tensor {} MXFP4 quantization requires group_size 32 and bits 4",
                    tensor.name
                ),
            });
        }
        "mxfp8" if quantization.group_size == 32 && quantization.bits == 8 => {}
        "mxfp8" => {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "tensor {} MXFP8 quantization requires group_size 32 and bits 8",
                    tensor.name
                ),
            });
        }
        _ => {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "tensor {} quantization mode {} is unsupported",
                    tensor.name, quantization.mode
                ),
            });
        }
    }
    Ok(())
}

fn expected_packed_cols(
    expected_cols: u64,
    tensor: &NativeTensorSpec,
) -> Result<u64, NativeModelError> {
    let quantization = tensor_quantization_or_default(tensor);
    let packed_bits = expected_cols
        .checked_mul(u64::from(quantization.bits))
        .ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized column count overflowed for {} columns at {} bits",
                tensor.name, expected_cols, quantization.bits
            ),
        })?;
    Ok(packed_bits.div_ceil(32))
}

fn validate_linear_attention_conv_tensor(
    tensor: &NativeTensorSpec,
    expected_channels: u64,
    expected_kernel_dim: u64,
) -> Result<(), NativeModelError> {
    if tensor.shape.is_empty() || tensor.shape[0] != expected_channels {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor linear_attention_conv1d must start with channel dimension {}, got {:?}",
                expected_channels, tensor.shape
            ),
        });
    }
    let remaining_product = tensor.shape[1..]
        .iter()
        .try_fold(1_u64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| NativeModelError::InvalidManifest {
            message: "linear_attention_conv1d shape overflowed".to_string(),
        })?;
    if remaining_product != expected_kernel_dim {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor linear_attention_conv1d must encode kernel size {}, got {:?}",
                expected_kernel_dim, tensor.shape
            ),
        });
    }
    if total_elements(tensor) != expected_channels.checked_mul(expected_kernel_dim) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor linear_attention_conv1d total element count must be {}, got {:?}",
                expected_channels.saturating_mul(expected_kernel_dim),
                tensor.shape
            ),
        });
    }
    Ok(())
}

fn require_global_role(
    roles: &[NativeTensorRole],
    required: NativeTensorRole,
    label: &str,
) -> Result<(), NativeModelError> {
    if roles.contains(&required) {
        Ok(())
    } else {
        Err(NativeModelError::InvalidManifest {
            message: format!("manifest is missing required global tensor role {}", label),
        })
    }
}

fn require_layer_role(
    roles: &[NativeTensorRole],
    required: NativeTensorRole,
    layer_index: u32,
    label: &str,
) -> Result<(), NativeModelError> {
    if roles.contains(&required) {
        Ok(())
    } else {
        Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} is missing required tensor role {}",
                layer_index, label
            ),
        })
    }
}

fn validate_tensor_path(
    root_dir: &Path,
    tensor: &NativeTensorSpec,
) -> Result<(), NativeModelError> {
    if tensor.file.is_absolute() {
        return Err(NativeModelError::InvalidManifest {
            message: format!("tensor {} file path must be relative", tensor.name),
        });
    }
    if tensor
        .file
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(NativeModelError::InvalidManifest {
            message: format!("tensor {} file path must not escape root_dir", tensor.name),
        });
    }

    let path = root_dir.join(&tensor.file);
    let metadata = fs::metadata(&path).map_err(|source| NativeModelError::InvalidManifest {
        message: format!(
            "tensor {} references missing file {}: {}",
            tensor.name,
            path.display(),
            source
        ),
    })?;
    if !metadata.is_file() {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} path {} is not a file",
                tensor.name,
                path.display()
            ),
        });
    }
    let file_len = metadata.len();
    let end = tensor
        .offset_bytes
        .checked_add(tensor.length_bytes)
        .ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!("tensor {} byte range overflowed", tensor.name),
        })?;
    if end > file_len {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} byte range [{}, {}) exceeds file length {}",
                tensor.name, tensor.offset_bytes, end, file_len
            ),
        });
    }

    Ok(())
}

fn validate_quantized_source_path(
    root_dir: &Path,
    tensor: &NativeTensorSpec,
) -> Result<(), NativeModelError> {
    let Some(source) = &tensor.quantized_source else {
        return Ok(());
    };
    if !tensor.source_quantized {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} declares quantized_source but source_quantized is false",
                tensor.name
            ),
        });
    }
    if source.length_bytes == 0 {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized_source must have positive length_bytes",
                tensor.name
            ),
        });
    }
    if source.file.is_absolute() {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized_source file path must be relative",
                tensor.name
            ),
        });
    }
    if source
        .file
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized_source file path must not escape root_dir",
                tensor.name
            ),
        });
    }

    let path = root_dir.join(&source.file);
    let metadata =
        fs::metadata(&path).map_err(|source_error| NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} references missing quantized_source file {}: {}",
                tensor.name,
                path.display(),
                source_error
            ),
        })?;
    if !metadata.is_file() {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized_source path {} is not a file",
                tensor.name,
                path.display()
            ),
        });
    }
    let file_len = metadata.len();
    let end = source
        .offset_bytes
        .checked_add(source.length_bytes)
        .ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized_source byte range overflowed",
                tensor.name
            ),
        })?;
    if end > file_len {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized_source byte range [{}, {}) exceeds file length {}",
                tensor.name, source.offset_bytes, end, file_len
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_test_dir(label: &str) -> PathBuf {
        static NEXT_SUFFIX: AtomicU64 = AtomicU64::new(0);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let suffix = NEXT_SUFFIX.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "ax-native-model-{label}-{}-{unique}-{suffix}",
            std::process::id()
        ))
    }

    fn write_fixture(
        mut manifest: NativeModelManifest,
        file_names: &[&str],
    ) -> (PathBuf, NativeModelManifest) {
        let dir = unique_test_dir("fixture");
        fs::create_dir_all(&dir).expect("fixture directory should create");
        for file_name in file_names {
            fs::write(dir.join(file_name), vec![0_u8; 4096]).expect("tensor file should write");
        }
        for tensor in &mut manifest.tensors {
            tensor.length_bytes = 32;
        }
        fs::write(
            dir.join(AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should write");
        (dir, manifest)
    }

    #[test]
    fn native_diffusion_config_is_enabled_by_any_field() {
        assert!(NativeDiffusionConfig::default().is_disabled());

        assert!(
            NativeDiffusionConfig {
                entropy_threshold: Some(0.005),
                ..Default::default()
            }
            .is_enabled()
        );
        assert!(
            NativeDiffusionConfig {
                convergence_steps: Some(2),
                ..Default::default()
            }
            .is_enabled()
        );
        assert!(
            NativeDiffusionConfig {
                temperature_start: Some(0.8),
                ..Default::default()
            }
            .is_enabled()
        );
        assert!(
            NativeDiffusionConfig {
                temperature_end: Some(0.4),
                ..Default::default()
            }
            .is_enabled()
        );
    }

    fn packed_layer_manifest() -> NativeModelManifest {
        NativeModelManifest {
            schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus::default(),
            layer_count: 2,
            hidden_size: 2048,
            intermediate_size: 11008,
            attention_head_count: 16,
            attention_head_dim: 128,
            kv_head_count: 8,
            vocab_size: 151936,
            tie_word_embeddings: false,
            rope_theta: None,
            rope_theta_swa: None,
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_low_freq_factor: None,
            rope_high_freq_factor: None,
            rope_original_context_len: None,
            rope_beta_fast: None,
            rope_beta_slow: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: None,
            attn_temperature_scale: None,
            intermediate_size_mlp: 0,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            rms_norm_eps: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            global_kv_head_count: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: Default::default(),
            final_logit_softcapping: None,
            final_logits_scale: None,
            attention_scale_multiplier: None,
            post_norm_eps: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig::default(),
            mla_attention: Default::default(),
            moe: NativeMoeConfig::default(),
            glm_router: Default::default(),
            deepseek_v4: Default::default(),
            weight_sanitize: WeightSanitize::default(),
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: NativeDiffusionConfig::default(),
            dropped_tensors: Default::default(),
            kv_cache_quantization: None,
            tensors: vec![
                tensor(
                    "model.embed_tokens.weight",
                    NativeTensorRole::TokenEmbedding,
                    None,
                    vec![151936, 2048],
                ),
                tensor(
                    "model.norm.weight",
                    NativeTensorRole::FinalNorm,
                    None,
                    vec![2048],
                ),
                tensor(
                    "lm_head.weight",
                    NativeTensorRole::LmHead,
                    None,
                    vec![151936, 2048],
                ),
                tensor(
                    "model.layers.0.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(0),
                    vec![2048],
                ),
                tensor(
                    "model.layers.0.self_attn.qkv_proj.weight",
                    NativeTensorRole::AttentionQkvPacked,
                    Some(0),
                    vec![4096, 2048],
                ),
                tensor(
                    "model.layers.0.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(0),
                    vec![2048, 2048],
                ),
                tensor(
                    "model.layers.0.post_attention_layernorm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(0),
                    vec![2048],
                ),
                tensor(
                    "model.layers.0.mlp.gate_up_proj.weight",
                    NativeTensorRole::FfnGateUpPacked,
                    Some(0),
                    vec![8192, 2048],
                ),
                tensor(
                    "model.layers.0.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(0),
                    vec![2048, 4096],
                ),
                tensor(
                    "model.layers.1.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(1),
                    vec![2048],
                ),
                tensor(
                    "model.layers.1.self_attn.qkv_proj.weight",
                    NativeTensorRole::AttentionQkvPacked,
                    Some(1),
                    vec![4096, 2048],
                ),
                tensor(
                    "model.layers.1.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(1),
                    vec![2048, 2048],
                ),
                tensor(
                    "model.layers.1.post_attention_layernorm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(1),
                    vec![2048],
                ),
                tensor(
                    "model.layers.1.mlp.gate_up_proj.weight",
                    NativeTensorRole::FfnGateUpPacked,
                    Some(1),
                    vec![8192, 2048],
                ),
                tensor(
                    "model.layers.1.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(1),
                    vec![2048, 4096],
                ),
            ],
        }
    }

    fn mixed_split_projection_manifest() -> NativeModelManifest {
        let mut manifest = packed_layer_manifest();
        manifest.kv_head_count = 4;
        let mut found_layer_zero_qkv = false;
        for projection in &mut manifest.tensors {
            if projection.layer_index == Some(0)
                && projection.role == NativeTensorRole::AttentionQkvPacked
            {
                projection.shape = vec![3072, 2048];
                found_layer_zero_qkv = true;
                break;
            }
        }
        assert!(
            found_layer_zero_qkv,
            "fixture should include layer-0 packed QKV"
        );
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::AttentionQkvPacked)
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.1.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(1),
                vec![2048, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(1),
                vec![512, 256],
            ),
            tensor(
                "model.layers.1.self_attn.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(1),
                vec![512, 256],
            ),
        ]);
        for projection in manifest.tensors.iter_mut().filter(|tensor| {
            tensor.layer_index == Some(1)
                && matches!(
                    tensor.role,
                    NativeTensorRole::AttentionK
                        | NativeTensorRole::AttentionV
                        | NativeTensorRole::AttentionO
                )
        }) {
            projection.dtype = NativeTensorDataType::U32;
            projection.source_quantized = true;
            projection.quantization = Some(NativeTensorQuantization {
                mode: "affine".to_string(),
                group_size: 64,
                bits: 4,
            });
            if projection.role == NativeTensorRole::AttentionO {
                projection.shape = vec![2048, 256];
            }
        }
        manifest
    }

    fn split_layer_manifest_with_value_from_key() -> NativeModelManifest {
        let mut manifest = packed_layer_manifest();
        manifest.attention_value_from_key_layers = vec![1];
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::AttentionQkvPacked)
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.1.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(1),
                vec![2048, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(1),
                vec![1024, 2048],
            ),
        ]);
        manifest
    }

    fn q_only_kv_shared_manifest() -> NativeModelManifest {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(1, 0);
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1)
                && matches!(
                    tensor.role,
                    NativeTensorRole::AttentionQkvPacked
                        | NativeTensorRole::AttentionK
                        | NativeTensorRole::AttentionV
                ))
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.1.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(1),
                vec![2048, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.q_norm.weight",
                NativeTensorRole::AttentionQNorm,
                Some(1),
                vec![128],
            ),
        ]);
        manifest
    }

    fn nemotron_attention_manifest() -> NativeModelManifest {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = "nemotron_h".to_string();
        manifest.layer_count = 1;
        manifest.layer_types = vec!["attention".to_string()];
        manifest.tensors.retain(|tensor| {
            tensor.layer_index.is_none()
                || (tensor.layer_index == Some(0)
                    && matches!(
                        tensor.role,
                        NativeTensorRole::AttentionNorm | NativeTensorRole::AttentionO
                    ))
        });
        manifest.tensors.extend([
            tensor(
                "backbone.layers.0.mixer.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(0),
                vec![2048, 2048],
            ),
            tensor(
                "backbone.layers.0.mixer.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(0),
                vec![1024, 2048],
            ),
            tensor(
                "backbone.layers.0.mixer.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(0),
                vec![1024, 2048],
            ),
        ]);
        manifest
    }

    fn packed_linear_attention_manifest() -> NativeModelManifest {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = "qwen3_5".to_string();
        manifest.linear_attention = NativeLinearAttentionConfig {
            full_attention_interval: Some(4),
            num_value_heads: Some(32),
            num_key_heads: Some(16),
            key_head_dim: Some(128),
            value_head_dim: Some(128),
            conv_kernel_dim: Some(4),
        };
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1)
                && matches!(
                    tensor.role,
                    NativeTensorRole::AttentionQkvPacked | NativeTensorRole::AttentionO
                ))
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.1.linear_attn.in_proj_qkvz.weight",
                NativeTensorRole::LinearAttentionInProjQkvz,
                Some(1),
                vec![12288, 2048],
            ),
            tensor(
                "model.layers.1.linear_attn.in_proj_ba.weight",
                NativeTensorRole::LinearAttentionInProjBa,
                Some(1),
                vec![64, 2048],
            ),
            tensor(
                "model.layers.1.linear_attn.conv1d.weight",
                NativeTensorRole::LinearAttentionConv1d,
                Some(1),
                vec![8192, 4],
            ),
            tensor(
                "model.layers.1.linear_attn.dt_bias",
                NativeTensorRole::LinearAttentionDtBias,
                Some(1),
                vec![32],
            ),
            tensor(
                "model.layers.1.linear_attn.A_log",
                NativeTensorRole::LinearAttentionALog,
                Some(1),
                vec![32],
            ),
            tensor(
                "model.layers.1.linear_attn.norm.weight",
                NativeTensorRole::LinearAttentionNorm,
                Some(1),
                vec![128],
            ),
            tensor(
                "model.layers.1.linear_attn.out_proj.weight",
                NativeTensorRole::LinearAttentionOutProj,
                Some(1),
                vec![2048, 4096],
            ),
        ]);
        manifest
    }

    fn moe_layer_manifest() -> NativeModelManifest {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.hidden_size = 2816;
        manifest.attention_head_count = 8;
        manifest.attention_head_dim = 256;
        manifest.kv_head_count = 2;
        manifest.vocab_size = 262144;
        manifest.tie_word_embeddings = true;
        manifest
            .tensors
            .retain(|tensor| tensor.role != NativeTensorRole::LmHead);
        for tensor in &mut manifest.tensors {
            match tensor.role {
                NativeTensorRole::TokenEmbedding => tensor.shape = vec![262144, 2816],
                NativeTensorRole::FinalNorm => tensor.shape = vec![2816],
                NativeTensorRole::AttentionNorm | NativeTensorRole::FfnNorm => {
                    tensor.shape = vec![2816]
                }
                NativeTensorRole::AttentionQkvPacked => tensor.shape = vec![3072, 2816],
                NativeTensorRole::AttentionO => tensor.shape = vec![2816, 2048],
                NativeTensorRole::FfnGateUpPacked => tensor.shape = vec![4224, 2816],
                NativeTensorRole::FfnDown => tensor.shape = vec![2816, 2112],
                _ => {}
            }
            if tensor.role == NativeTensorRole::FfnNorm {
                let layer = tensor.layer_index.expect("fixture layer tensor");
                tensor.name = format!("model.layers.{layer}.pre_feedforward_layernorm.weight");
            }
        }
        manifest.moe = NativeMoeConfig {
            expert_count: Some(128),
            experts_per_token: Some(8),
            expert_intermediate_size: Some(704),
            layer_freq: None,
            first_dense_layers: None,
            shared_expert_count: None,
            sigmoid_routing: false,
            routed_scaling_factor: None,
            n_group: None,
            topk_group: None,
        };
        manifest.tensors.extend([
            tensor(
                "model.layers.0.post_attention_layernorm.weight",
                NativeTensorRole::AttentionPostNorm,
                Some(0),
                vec![2816],
            ),
            tensor(
                "model.layers.0.post_feedforward_layernorm.weight",
                NativeTensorRole::FfnPostNorm,
                Some(0),
                vec![2816],
            ),
            tensor(
                "model.layers.0.pre_feedforward_layernorm_2.weight",
                NativeTensorRole::FfnNorm2,
                Some(0),
                vec![2816],
            ),
            tensor(
                "model.layers.0.post_feedforward_layernorm_1.weight",
                NativeTensorRole::FfnPostNorm1,
                Some(0),
                vec![2816],
            ),
            tensor(
                "model.layers.0.post_feedforward_layernorm_2.weight",
                NativeTensorRole::FfnPostNorm2,
                Some(0),
                vec![2816],
            ),
            tensor(
                "model.layers.0.router.proj.weight",
                NativeTensorRole::FfnGateInp,
                Some(0),
                vec![128, 2816],
            ),
            tensor(
                "model.layers.0.router.scale",
                NativeTensorRole::FfnGateInpScale,
                Some(0),
                vec![2816],
            ),
            tensor(
                "model.layers.0.experts.gate_up_proj.weight",
                NativeTensorRole::FfnGateUpExpsPacked,
                Some(0),
                vec![128, 1408, 2816],
            ),
            tensor(
                "model.layers.0.experts.down_proj.weight",
                NativeTensorRole::FfnDownExps,
                Some(0),
                vec![128, 2816, 704],
            ),
            tensor(
                "model.layers.0.experts.down_proj.scale",
                NativeTensorRole::FfnDownExpsScale,
                Some(0),
                vec![128],
            ),
            tensor(
                "model.layers.1.post_attention_layernorm.weight",
                NativeTensorRole::AttentionPostNorm,
                Some(1),
                vec![2816],
            ),
            tensor(
                "model.layers.1.post_feedforward_layernorm.weight",
                NativeTensorRole::FfnPostNorm,
                Some(1),
                vec![2816],
            ),
            tensor(
                "model.layers.1.pre_feedforward_layernorm_2.weight",
                NativeTensorRole::FfnNorm2,
                Some(1),
                vec![2816],
            ),
            tensor(
                "model.layers.1.post_feedforward_layernorm_1.weight",
                NativeTensorRole::FfnPostNorm1,
                Some(1),
                vec![2816],
            ),
            tensor(
                "model.layers.1.post_feedforward_layernorm_2.weight",
                NativeTensorRole::FfnPostNorm2,
                Some(1),
                vec![2816],
            ),
            tensor(
                "model.layers.1.router.proj.weight",
                NativeTensorRole::FfnGateInp,
                Some(1),
                vec![128, 2816],
            ),
            tensor(
                "model.layers.1.router.scale",
                NativeTensorRole::FfnGateInpScale,
                Some(1),
                vec![2816],
            ),
            tensor(
                "model.layers.1.experts.gate_proj.weight",
                NativeTensorRole::FfnGateExps,
                Some(1),
                vec![128, 704, 2816],
            ),
            tensor(
                "model.layers.1.experts.up_proj.weight",
                NativeTensorRole::FfnUpExps,
                Some(1),
                vec![128, 704, 2816],
            ),
            tensor(
                "model.layers.1.experts.down_proj.weight",
                NativeTensorRole::FfnDownExps,
                Some(1),
                vec![128, 2816, 704],
            ),
        ]);
        manifest
    }

    fn switch_moe_manifest(model_family: &str, include_shared_expert: bool) -> NativeModelManifest {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = model_family.to_string();
        manifest.layer_count = 1;
        manifest.moe = NativeMoeConfig {
            expert_count: Some(4),
            experts_per_token: Some(2),
            expert_intermediate_size: Some(512),
            layer_freq: None,
            first_dense_layers: None,
            shared_expert_count: None,
            sigmoid_routing: false,
            routed_scaling_factor: None,
            n_group: None,
            topk_group: None,
        };
        manifest.tensors.retain(|tensor| {
            tensor.layer_index != Some(1)
                && !matches!(
                    tensor.role,
                    NativeTensorRole::FfnGateUpPacked | NativeTensorRole::FfnDown
                )
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.0.mlp.gate.weight",
                NativeTensorRole::FfnGateInp,
                Some(0),
                vec![4, 2048],
            ),
            tensor(
                "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                NativeTensorRole::FfnGateExps,
                Some(0),
                vec![4, 512, 2048],
            ),
            tensor(
                "model.layers.0.mlp.switch_mlp.up_proj.weight",
                NativeTensorRole::FfnUpExps,
                Some(0),
                vec![4, 512, 2048],
            ),
            tensor(
                "model.layers.0.mlp.switch_mlp.down_proj.weight",
                NativeTensorRole::FfnDownExps,
                Some(0),
                vec![4, 2048, 512],
            ),
        ]);
        if include_shared_expert {
            manifest.tensors.extend([
                tensor(
                    "model.layers.0.mlp.shared_expert_gate.weight",
                    NativeTensorRole::FfnSharedExpertGateInp,
                    Some(0),
                    vec![1, 2048],
                ),
                tensor(
                    "model.layers.0.mlp.shared_expert.gate_proj.weight",
                    NativeTensorRole::FfnSharedExpertGate,
                    Some(0),
                    vec![512, 2048],
                ),
                tensor(
                    "model.layers.0.mlp.shared_expert.up_proj.weight",
                    NativeTensorRole::FfnSharedExpertUp,
                    Some(0),
                    vec![512, 2048],
                ),
                tensor(
                    "model.layers.0.mlp.shared_expert.down_proj.weight",
                    NativeTensorRole::FfnSharedExpertDown,
                    Some(0),
                    vec![2048, 512],
                ),
            ]);
        }
        manifest
    }

    fn tensor(
        name: &str,
        role: NativeTensorRole,
        layer_index: Option<u32>,
        shape: Vec<u64>,
    ) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index,
            dtype: NativeTensorDataType::F16,
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

    #[test]
    fn unique_test_dir_produces_distinct_paths_in_a_burst() {
        let paths = (0..1024)
            .map(|_| unique_test_dir("burst"))
            .collect::<Vec<_>>();
        let unique = paths.iter().cloned().collect::<BTreeSet<_>>();

        assert_eq!(paths.len(), unique.len());
    }

    #[test]
    fn native_model_artifacts_load_valid_packed_manifest() {
        let manifest = packed_layer_manifest();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let artifacts =
            NativeModelArtifacts::from_dir(&dir).expect("packed manifest should validate");

        assert_eq!(artifacts.manifest().model_family, "qwen3");
        assert_eq!(
            artifacts.summary(),
            NativeModelArtifactsSummary {
                model_family: "qwen3".to_string(),
                tensor_format: NativeTensorFormat::Safetensors,
                source_quantization: None,
                runtime_status: NativeRuntimeStatus::default(),
                layer_count: 2,
                tensor_count: 15,
                tie_word_embeddings: false,
                is_moe: false,
                is_hybrid_attention: false,
                hybrid_full_attention_interval: None,
                mla_kv_latent_dim: None,
                moe_active_experts: None,
            }
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_scalar_other_tensors() {
        let mut manifest = packed_layer_manifest();
        manifest.tensors.push(tensor(
            "vision_tower.encoder.layers.0.self_attn.q_proj.input_min",
            NativeTensorRole::Other,
            None,
            Vec::new(),
        ));
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("rank-0 extension tensors should remain valid safetensors");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_load_valid_kv_cache_quantization_table() {
        let mut manifest = packed_layer_manifest();
        manifest.kv_cache_quantization = Some(KvCacheQuantizationManifest {
            layer_bits: vec![8, 16],
            layer_group_sizes: vec![64, 128],
            basis: "measured".to_string(),
        });
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let artifacts = NativeModelArtifacts::from_dir(&dir)
            .expect("manifest with valid kv_cache_quantization should validate");

        let table = artifacts
            .manifest()
            .kv_cache_quantization
            .as_ref()
            .expect("kv_cache_quantization should round-trip");
        assert_eq!(table.layer_bits, vec![8, 16]);
        assert_eq!(table.layer_group_sizes, vec![64, 128]);
        assert_eq!(table.basis, "measured");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_kv_cache_quantization_length_mismatch() {
        let mut manifest = packed_layer_manifest();
        manifest.kv_cache_quantization = Some(KvCacheQuantizationManifest {
            layer_bits: vec![8, 4, 4],
            layer_group_sizes: vec![64, 32],
            basis: "measured".to_string(),
        });
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("length-mismatched kv_cache_quantization should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };

        assert!(message.contains("kv_cache_quantization.layer_bits"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_kv_cache_quantization_bad_bits() {
        let mut manifest = packed_layer_manifest();
        manifest.kv_cache_quantization = Some(KvCacheQuantizationManifest {
            layer_bits: vec![8, 2],
            layer_group_sizes: vec![64, 32],
            basis: "measured".to_string(),
        });
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("bits=2 should fail validation");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };

        assert!(message.contains("kv_cache_quantization.layer_bits"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_kv_cache_quantization_bad_group_size() {
        let mut manifest = packed_layer_manifest();
        manifest.kv_cache_quantization = Some(KvCacheQuantizationManifest {
            layer_bits: vec![8, 4],
            layer_group_sizes: vec![64, 16],
            basis: "measured".to_string(),
        });
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("group_size=16 with bits<16 should fail validation");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };

        assert!(message.contains("kv_cache_quantization.layer_group_sizes"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_manifest_kv_cache_quantization_defaults_to_none_and_is_not_serialized() {
        let manifest = packed_layer_manifest();
        assert!(manifest.kv_cache_quantization.is_none());

        let value = serde_json::to_value(&manifest).expect("manifest should serialize");
        assert!(value.get("kv_cache_quantization").is_none());

        let round_tripped: NativeModelManifest =
            serde_json::from_value(value).expect("manifest should deserialize");
        assert!(round_tripped.kv_cache_quantization.is_none());
    }

    #[test]
    fn native_model_artifacts_load_valid_packed_linear_attention_manifest() {
        let manifest = packed_linear_attention_manifest();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let artifacts = NativeModelArtifacts::from_dir(&dir)
            .expect("packed linear attention manifest should validate");

        assert!(artifacts.summary().is_hybrid_attention);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_bad_packed_linear_attention_projection_shape() {
        let mut manifest = packed_linear_attention_manifest();
        let qkvz = manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.layer_index == Some(1)
                    && tensor.role == NativeTensorRole::LinearAttentionInProjQkvz
            })
            .expect("fixture has packed qkvz");
        qkvz.shape = vec![12287, 2048];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("bad packed linear attention qkvz shape should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };

        assert!(message.contains("linear_attention_in_proj_qkvz"));
        assert!(message.contains("[12288, 2048]"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_bad_packed_linear_attention_ba_shape() {
        let mut manifest = packed_linear_attention_manifest();
        let ba = manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.layer_index == Some(1)
                    && tensor.role == NativeTensorRole::LinearAttentionInProjBa
            })
            .expect("fixture has packed ba");
        ba.shape = vec![63, 2048];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("bad packed linear attention ba shape should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };

        assert!(message.contains("linear_attention_in_proj_ba"));
        assert!(message.contains("[64, 2048]"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_summary_reports_mla_and_moe_dimensions() {
        let mut manifest = packed_layer_manifest();
        manifest.mla_attention = NativeMlaAttentionConfig {
            q_lora_rank: Some(768),
            kv_lora_rank: Some(512),
            qk_nope_head_dim: Some(192),
            qk_rope_head_dim: Some(64),
            value_head_dim: Some(256),
        };
        manifest.moe = NativeMoeConfig {
            expert_count: Some(64),
            experts_per_token: Some(4),
            expert_intermediate_size: Some(1536),
            layer_freq: None,
            first_dense_layers: None,
            shared_expert_count: None,
            sigmoid_routing: false,
            routed_scaling_factor: None,
            n_group: None,
            topk_group: None,
        };
        let artifacts = NativeModelArtifacts {
            root_dir: PathBuf::new(),
            manifest,
        };

        let summary = artifacts.summary();

        assert_eq!(summary.mla_kv_latent_dim, Some(512));
        assert_eq!(summary.moe_active_experts, Some(4));
    }

    #[test]
    fn qwen35_linear_attention_defaults_missing_full_interval() {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = "qwen3_5".to_string();
        manifest.linear_attention = NativeLinearAttentionConfig {
            full_attention_interval: None,
            num_value_heads: Some(32),
            num_key_heads: Some(16),
            key_head_dim: Some(128),
            value_head_dim: Some(128),
            conv_kernel_dim: Some(4),
        };
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let artifacts =
            NativeModelArtifacts::from_dir(&dir).expect("Qwen3.5 should inherit interval 4");

        assert_eq!(
            artifacts
                .manifest()
                .linear_attention
                .resolved_full_attention_interval(&artifacts.manifest().model_family),
            Some(QWEN3_5_DEFAULT_FULL_ATTENTION_INTERVAL)
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn non_qwen35_linear_attention_requires_full_interval() {
        let mut manifest = packed_layer_manifest();
        manifest.linear_attention = NativeLinearAttentionConfig {
            full_attention_interval: None,
            num_value_heads: Some(32),
            num_key_heads: Some(16),
            key_head_dim: Some(128),
            value_head_dim: Some(128),
            conv_kernel_dim: Some(4),
        };
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("non-Qwen3.5 manifests must carry an explicit interval");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };

        assert!(message.contains("linear_attention.full_attention_interval"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_invalid_rms_norm_eps() {
        for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let (dir, mut manifest) =
                write_fixture(packed_layer_manifest(), &["model.safetensors"]);
            manifest.rms_norm_eps = Some(eps);

            let error = NativeModelArtifacts::from_manifest_and_root(dir.clone(), manifest)
                .expect_err("invalid rms_norm_eps should fail closed");
            let NativeModelError::InvalidManifest { message } = error else {
                panic!("expected invalid manifest error");
            };
            assert!(message.contains("rms_norm_eps must be finite and > 0"));

            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn native_model_artifacts_allow_positive_rms_norm_eps() {
        let mut manifest = packed_layer_manifest();
        manifest.rms_norm_eps = Some(1e-5);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let artifacts =
            NativeModelArtifacts::from_dir(&dir).expect("positive rms_norm_eps should validate");

        assert_eq!(artifacts.manifest().rms_norm_eps, Some(1e-5));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_runtime_not_ready_manifest() {
        let mut manifest = packed_layer_manifest();
        manifest.runtime_status = NativeRuntimeStatus {
            ready: false,
            blockers: vec!["qwen35_quantized_gguf_native_runtime_not_implemented".to_string()],
            notes: Vec::new(),
        };
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let err = NativeModelArtifacts::from_dir(&dir)
            .expect_err("runtime-not-ready manifest should fail closed");
        let message = err.to_string();
        assert!(message.contains("not runtime ready"));
        assert!(message.contains("qwen35_quantized_gguf_native_runtime_not_implemented"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_attn_output_gate_with_packed_qkv() {
        let mut manifest = packed_layer_manifest();
        manifest.attn_output_gate = true;
        for tensor in &mut manifest.tensors {
            if tensor.role == NativeTensorRole::AttentionQkvPacked {
                tensor.shape[0] = 6144;
            }
        }
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("packed attn_output_gate manifest should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_attn_output_gate_with_split_qkv() {
        let mut manifest = packed_layer_manifest();
        manifest.attn_output_gate = true;
        for tensor in &mut manifest.tensors {
            if tensor.role == NativeTensorRole::AttentionQkvPacked {
                tensor.shape[0] = 6144;
            }
        }
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::AttentionQkvPacked)
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.1.self_attn.q_norm.weight",
                NativeTensorRole::AttentionQNorm,
                Some(1),
                vec![128],
            ),
            tensor(
                "model.layers.1.self_attn.k_norm.weight",
                NativeTensorRole::AttentionKNorm,
                Some(1),
                vec![128],
            ),
            tensor(
                "model.layers.1.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(1),
                vec![4096, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(1),
                vec![1024, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(1),
                vec![1024, 2048],
            ),
        ]);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("split gated-attention manifest should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_expose_tensor_accessors_and_resolved_paths() {
        let manifest = packed_layer_manifest();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let artifacts =
            NativeModelArtifacts::from_dir(&dir).expect("packed manifest should validate");
        let embedding = artifacts
            .global_tensor(NativeTensorRole::TokenEmbedding)
            .expect("token embedding should resolve");
        let layer_qkv = artifacts
            .layer_tensor(1, NativeTensorRole::AttentionQkvPacked)
            .expect("layer qkv should resolve");

        assert_eq!(artifacts.tensor_specs().len(), 15);
        assert_eq!(embedding.name, "model.embed_tokens.weight");
        assert_eq!(layer_qkv.name, "model.layers.1.self_attn.qkv_proj.weight");
        assert_eq!(
            artifacts.resolve_tensor_path(layer_qkv),
            dir.join("model.safetensors")
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_tied_embeddings_without_lm_head_tensor() {
        let mut manifest = packed_layer_manifest();
        manifest.tie_word_embeddings = true;
        manifest
            .tensors
            .retain(|tensor| tensor.role != NativeTensorRole::LmHead);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("tied embeddings should allow lm_head omission");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_load_valid_moe_manifest() {
        let manifest = moe_layer_manifest();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let artifacts = NativeModelArtifacts::from_dir(&dir).expect("moe manifest should validate");

        assert_eq!(
            artifacts
                .moe_config()
                .and_then(|config| config.expert_count),
            Some(128)
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_bad_moe_router_sidecar_lengths() {
        for (role, name, label) in [
            (
                NativeTensorRole::FfnGateInpCorrectionBias,
                "model.layers.0.router.correction_bias",
                "ffn_gate_inp_correction_bias",
            ),
            (
                NativeTensorRole::FfnGateInpExpertScale,
                "model.layers.0.router.per_expert_scale",
                "ffn_gate_inp_expert_scale",
            ),
        ] {
            let mut manifest = moe_layer_manifest();
            manifest
                .tensors
                .push(tensor(name, role, Some(0), vec![127]));
            let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

            let error = NativeModelArtifacts::from_dir(&dir)
                .expect_err("MoE router sidecars must match expert_count");
            let NativeModelError::InvalidManifest { message } = error else {
                panic!("expected invalid manifest error");
            };
            assert!(message.contains(label), "unexpected error: {message}");
            assert!(message.contains("128"), "unexpected error: {message}");

            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn native_model_artifacts_allow_gemma4_dense_without_moe_only_norms() {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = "gemma4".to_string();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("Gemma4 dense manifests should not require MoE-only norm roles");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_gemma4_moe_missing_second_ffn_norm() {
        let mut manifest = moe_layer_manifest();
        manifest
            .tensors
            .retain(|tensor| tensor.role != NativeTensorRole::FfnNorm2);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("Gemma4 MoE manifests must carry pre_feedforward_layernorm_2");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("ffn_norm_2"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_gemma4_vl_moe_missing_second_ffn_norm() {
        // gemma4_vl MoE text towers use the same dual-norm contract as gemma4.
        let mut manifest = moe_layer_manifest();
        manifest.model_family = "gemma4_vl".to_string();
        manifest
            .tensors
            .retain(|tensor| tensor.role != NativeTensorRole::FfnNorm2);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("gemma4_vl MoE must require ffn_norm_2 like gemma4");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("ffn_norm_2"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_bad_quantized_packed_columns() {
        let mut manifest = packed_layer_manifest();
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::U32;
        gate.source_quantized = true;
        gate.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 4,
        });
        gate.shape = vec![8192, 1024];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("wrong packed column count should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("packed quantized shape"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_bad_quantized_ffn_down_output_rows() {
        let mut manifest = packed_layer_manifest();
        let down = manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.role == NativeTensorRole::FfnDown && tensor.layer_index == Some(0)
            })
            .expect("fixture should include ffn down");
        down.dtype = NativeTensorDataType::U32;
        down.source_quantized = true;
        down.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 4,
        });
        down.shape = vec![2047, 512];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("quantized ffn_down must still output hidden_size rows");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("ffn_down"), "unexpected error: {message}");
        assert!(message.contains("2048"), "unexpected error: {message}");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_5_and_6_bit_quantized_packed_columns() {
        for bits in [5, 6] {
            let mut manifest = packed_layer_manifest();
            let gate = manifest
                .tensors
                .iter_mut()
                .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
                .expect("fixture should include packed ffn gate/up");
            gate.dtype = NativeTensorDataType::U32;
            gate.source_quantized = true;
            gate.quantization = Some(NativeTensorQuantization {
                mode: "affine".to_string(),
                group_size: 64,
                bits,
            });
            gate.shape = vec![8192, (2048 * u64::from(bits)).div_ceil(32)];
            let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

            NativeModelArtifacts::from_dir(&dir).unwrap_or_else(|error| {
                panic!("{bits}-bit quantized packed columns should validate: {error}")
            });

            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn native_model_artifacts_allow_bf16_q_with_packed_k_v_o() {
        let manifest = mixed_split_projection_manifest();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let result = NativeModelArtifacts::from_dir(&dir);
        let _ = fs::remove_dir_all(dir);
        assert!(
            result.is_ok(),
            "BF16 Q must not force packed K/V/O to use raw column shapes"
        );
    }

    #[test]
    fn native_model_artifacts_infer_legacy_gemma4_global_kv_width_from_tensors() {
        let mut manifest = mixed_split_projection_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.kv_head_count = 8;
        manifest.global_head_dim = Some(256);
        manifest.global_kv_head_count = None;
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ];

        for projection in &mut manifest.tensors {
            match (projection.layer_index, projection.role) {
                (Some(0), NativeTensorRole::AttentionQkvPacked) => {
                    projection.shape = vec![4096, 2048];
                }
                (Some(1), NativeTensorRole::AttentionQ) => {
                    projection.shape = vec![4096, 2048];
                }
                (Some(1), NativeTensorRole::AttentionK | NativeTensorRole::AttentionV) => {
                    projection.shape = vec![512, 256];
                }
                (Some(1), NativeTensorRole::AttentionO) => {
                    projection.shape = vec![2048, 512];
                }
                _ => {}
            }
        }

        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);
        NativeModelArtifacts::from_dir(&dir).expect(
            "legacy Gemma4 manifests must use full-attention tensor rows when the global KV field is absent",
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_split_attention_head_count_mismatches() {
        for (projection_group, q_rows, kv_rows) in
            [("query", 1024_u64, 512_u64), ("key/value", 2048, 256)]
        {
            let mut manifest = mixed_split_projection_manifest();
            for projection in manifest.tensors.iter_mut().filter(|tensor| {
                tensor.layer_index == Some(1)
                    && matches!(
                        tensor.role,
                        NativeTensorRole::AttentionQ
                            | NativeTensorRole::AttentionK
                            | NativeTensorRole::AttentionV
                    )
            }) {
                match projection.role {
                    NativeTensorRole::AttentionQ => projection.shape[0] = q_rows,
                    NativeTensorRole::AttentionK | NativeTensorRole::AttentionV => {
                        projection.shape[0] = kv_rows;
                    }
                    _ => {}
                }
            }
            let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

            let error = NativeModelArtifacts::from_dir(&dir)
                .expect_err("split projection head counts must match manifest metadata");
            let NativeModelError::InvalidManifest { message } = error else {
                panic!("expected invalid manifest error");
            };
            assert!(
                message.contains("head counts must match manifest"),
                "{projection_group} mismatch produced unexpected error: {message}"
            );

            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn native_model_artifacts_allow_raw_k_with_mixed_packed_q_v_o() {
        let mut manifest = mixed_split_projection_manifest();
        for projection in manifest.tensors.iter_mut().filter(|tensor| {
            tensor.layer_index == Some(1)
                && matches!(
                    tensor.role,
                    NativeTensorRole::AttentionQ
                        | NativeTensorRole::AttentionK
                        | NativeTensorRole::AttentionV
                )
        }) {
            match projection.role {
                NativeTensorRole::AttentionQ => {
                    projection.dtype = NativeTensorDataType::U32;
                    projection.source_quantized = true;
                    projection.quantization = Some(NativeTensorQuantization {
                        mode: "affine".to_string(),
                        group_size: 64,
                        bits: 6,
                    });
                    projection.shape = vec![2048, 384];
                }
                NativeTensorRole::AttentionK => {
                    projection.dtype = NativeTensorDataType::Bf16;
                    projection.source_quantized = false;
                    projection.quantization = None;
                    projection.shape = vec![512, 2048];
                }
                NativeTensorRole::AttentionV => {
                    projection.quantization = Some(NativeTensorQuantization {
                        mode: "affine".to_string(),
                        group_size: 64,
                        bits: 8,
                    });
                    projection.shape = vec![512, 512];
                }
                _ => {}
            }
        }
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let result = NativeModelArtifacts::from_dir(&dir);
        let _ = fs::remove_dir_all(dir);
        assert!(
            result.is_ok(),
            "raw K must validate independently of mixed 6/8/4-bit Q/V/O storage"
        );
    }

    #[test]
    fn native_model_artifacts_reject_bad_packed_k_columns_with_bf16_q() {
        let mut manifest = mixed_split_projection_manifest();
        let mut found_attention_k = false;
        for projection in &mut manifest.tensors {
            if projection.layer_index == Some(1) && projection.role == NativeTensorRole::AttentionK
            {
                projection.shape = vec![512, 255];
                found_attention_k = true;
                break;
            }
        }
        assert!(
            found_attention_k,
            "fixture should include packed K projection"
        );
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let result = NativeModelArtifacts::from_dir(&dir);
        let rejected_bad_packed_k = matches!(
            &result,
            Err(NativeModelError::InvalidManifest { message })
                if message.contains("attention_k must have packed quantized shape [512, 256]")
        );
        let _ = fs::remove_dir_all(dir);
        assert!(
            rejected_bad_packed_k,
            "incorrect packed K columns must fail closed with the expected validation error"
        );
    }

    #[test]
    fn native_model_artifacts_reject_bad_raw_k_columns_with_packed_q() {
        let mut manifest = mixed_split_projection_manifest();
        let mut found_attention_q = false;
        let mut found_attention_k = false;
        for projection in &mut manifest.tensors {
            if projection.layer_index != Some(1) {
                continue;
            }
            match projection.role {
                NativeTensorRole::AttentionQ => {
                    projection.dtype = NativeTensorDataType::U32;
                    projection.source_quantized = true;
                    projection.quantization = Some(NativeTensorQuantization {
                        mode: "affine".to_string(),
                        group_size: 64,
                        bits: 4,
                    });
                    projection.shape = vec![2048, 256];
                    found_attention_q = true;
                }
                NativeTensorRole::AttentionK => {
                    projection.dtype = NativeTensorDataType::Bf16;
                    projection.source_quantized = false;
                    projection.quantization = None;
                    projection.shape = vec![512, 2047];
                    found_attention_k = true;
                }
                _ => {}
            }
        }
        assert!(found_attention_q, "fixture should include Q projection");
        assert!(found_attention_k, "fixture should include K projection");
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let result = NativeModelArtifacts::from_dir(&dir);
        let rejected_bad_raw_k = matches!(
            &result,
            Err(NativeModelError::InvalidManifest { message })
                if message.contains("attention_k must have shape [kv_rows, 2048]")
        );
        let _ = fs::remove_dir_all(dir);
        assert!(
            rejected_bad_raw_k,
            "incorrect raw K columns must fail with the expected validation error"
        );
    }

    #[test]
    fn native_model_artifacts_reject_unbenchmarked_affine_quantization_bits() {
        let mut manifest = packed_layer_manifest();
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::U32;
        gate.source_quantized = true;
        gate.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 7,
        });
        gate.shape = vec![8192, (2048 * 7_u64).div_ceil(32)];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("unbenchmarked affine bit widths should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("quantization bits must be one of"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_3bit_without_experimental_gate() {
        let mut manifest = packed_layer_manifest();
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::U32;
        gate.source_quantized = true;
        gate.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 3,
        });
        // packed_cols = ceil(2048 * 3 / 32) = 192
        gate.shape = vec![8192, (2048_u64 * 3).div_ceil(32)];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("3-bit without experimental gate should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("requires experimental gate"),
            "error should reference gate: {message}"
        );
        assert!(
            message.contains(AX_ENGINE_3BIT_EXPERIMENTAL_ENV),
            "error should name the env var: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_validate_3bit_tensor_quantization_with_experimental_gate() {
        // Test the internal validator directly with the gate flag to avoid env var mutation.
        let spec = NativeTensorSpec {
            name: "layers.0.ffn.gate_up".to_string(),
            role: NativeTensorRole::FfnGateUpPacked,
            layer_index: Some(0),
            dtype: NativeTensorDataType::U32,
            source_tensor_type: None,
            source_quantized: true,
            quantization: Some(NativeTensorQuantization {
                mode: "affine".to_string(),
                group_size: 64,
                bits: 3,
            }),
            quantized_source: None,
            shape: vec![8192, (2048_u64 * 3).div_ceil(32)],
            file: "model.safetensors".into(),
            offset_bytes: 0,
            length_bytes: 192 * 8192 * 4,
        };

        validate_tensor_quantization(&spec, NativeTensorFormat::Safetensors, true, false)
            .expect("3-bit should be accepted when experimental gate is enabled");
        validate_tensor_quantization(&spec, NativeTensorFormat::Safetensors, false, false)
            .expect_err("3-bit should be rejected when experimental gate is disabled");
    }

    #[test]
    fn native_model_artifacts_validate_mxfp4_tensor_quantization() {
        let spec = NativeTensorSpec {
            name: "layers.0.ffn.experts.down_proj".to_string(),
            role: NativeTensorRole::FfnDownExps,
            layer_index: Some(0),
            dtype: NativeTensorDataType::U32,
            source_tensor_type: None,
            source_quantized: true,
            quantization: Some(NativeTensorQuantization {
                mode: "mxfp4".to_string(),
                group_size: 32,
                bits: 4,
            }),
            quantized_source: None,
            shape: vec![128, 2880, 360],
            file: "model.safetensors".into(),
            offset_bytes: 0,
            length_bytes: 128 * 2880 * 360 * 4,
        };

        validate_tensor_quantization(&spec, NativeTensorFormat::Safetensors, false, false)
            .expect("MXFP4 weights should be accepted with their fixed layout");
    }

    #[test]
    fn native_model_artifacts_reject_2bit_as_unsupported() {
        let mut manifest = packed_layer_manifest();
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::U32;
        gate.source_quantized = true;
        gate.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 2,
        });
        gate.shape = vec![8192, (2048_u64 * 2).div_ceil(32)];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("2-bit without the experimental gate should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        // 2-bit now mirrors the 3-bit contract: rejected by default, admitted
        // only behind its own experimental env gate.
        assert!(
            message.contains("requires experimental gate"),
            "error should reference gate: {message}"
        );
        assert!(
            message.contains(AX_ENGINE_2BIT_EXPERIMENTAL_ENV),
            "error should name the env var: {message}"
        );
    }

    #[test]
    fn native_model_artifacts_validate_3bit_packed_column_shape() {
        // Verify the packed-column formula for 3-bit: ceil(cols * 3 / 32).
        // For cols=2048: ceil(6144/32) = 192.
        // For cols=4096: ceil(12288/32) = 384.
        for (cols, expected_packed) in [(2048_u64, 192_u64), (4096, 384), (1024, 96)] {
            let spec = NativeTensorSpec {
                name: format!("layers.0.ffn.gate_up_{cols}"),
                role: NativeTensorRole::FfnGateUpPacked,
                layer_index: Some(0),
                dtype: NativeTensorDataType::U32,
                source_tensor_type: None,
                source_quantized: true,
                quantization: Some(NativeTensorQuantization {
                    mode: "affine".to_string(),
                    group_size: 64,
                    bits: 3,
                }),
                quantized_source: None,
                shape: vec![8192, expected_packed],
                file: "model.safetensors".into(),
                offset_bytes: 0,
                length_bytes: expected_packed * 8192 * 4,
            };
            let packed = expected_packed_cols(cols, &spec)
                .unwrap_or_else(|e| panic!("packed cols for {cols} should compute: {e}"));
            assert_eq!(
                packed, expected_packed,
                "3-bit packed_cols for {cols} cols: expected {expected_packed}, got {packed}"
            );
        }
    }

    #[test]
    fn native_model_artifacts_validate_mixed_3bit_4bit_with_experimental_gate() {
        // A tensor quantized at 3-bit (low layer) passes with gate=true;
        // a tensor quantized at 4-bit (sensitive layer) always passes.
        let low_layer = NativeTensorSpec {
            name: "layers.0.ffn.gate_up".to_string(),
            role: NativeTensorRole::FfnGateUpPacked,
            layer_index: Some(0),
            dtype: NativeTensorDataType::U32,
            source_tensor_type: None,
            source_quantized: true,
            quantization: Some(NativeTensorQuantization {
                mode: "affine".to_string(),
                group_size: 64,
                bits: 3,
            }),
            quantized_source: None,
            shape: vec![8192, (2048_u64 * 3).div_ceil(32)],
            file: "model.safetensors".into(),
            offset_bytes: 0,
            length_bytes: 192 * 8192 * 4,
        };
        let sensitive_layer = NativeTensorSpec {
            name: "layers.0.attn.v_proj".to_string(),
            role: NativeTensorRole::AttentionQkvPacked,
            layer_index: Some(0),
            dtype: NativeTensorDataType::U32,
            source_tensor_type: None,
            source_quantized: true,
            quantization: Some(NativeTensorQuantization {
                mode: "affine".to_string(),
                group_size: 64,
                bits: 4,
            }),
            quantized_source: None,
            shape: vec![4096, 256],
            file: "model.safetensors".into(),
            offset_bytes: 0,
            length_bytes: 256 * 4096 * 4,
        };

        validate_tensor_quantization(&low_layer, NativeTensorFormat::Safetensors, true, false)
            .expect("3-bit low layer with gate should be accepted");
        validate_tensor_quantization(
            &sensitive_layer,
            NativeTensorFormat::Safetensors,
            true,
            false,
        )
        .expect("4-bit sensitive layer should always be accepted");
        validate_tensor_quantization(
            &sensitive_layer,
            NativeTensorFormat::Safetensors,
            false,
            false,
        )
        .expect("4-bit sensitive layer should always be accepted without gate");
    }

    #[test]
    fn native_model_artifacts_reject_u32_tensor_without_source_quantized_flag() {
        let mut manifest = packed_layer_manifest();
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::U32;
        gate.source_quantized = false;
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("u32 tensors should be declared source-quantized");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("dtype u32 but source_quantized is false"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_affine_quantization_on_non_u32_tensor() {
        let mut manifest = packed_layer_manifest();
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::Bf16;
        gate.source_quantized = true;
        gate.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 4,
        });
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("affine quantization metadata should belong to u32 tensors");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("declares quantization"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_source_quantized_non_u32_without_metadata() {
        let mut manifest = packed_layer_manifest();
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::Bf16;
        gate.source_quantized = true;
        gate.quantization = None;
        gate.shape = vec![8192, 256];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("source-quantized storage must use packed u32 even without metadata");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("source_quantized") && message.contains("expected u32"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_gguf_block_dtype_in_safetensors_manifest() {
        let mut manifest = packed_layer_manifest();
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::Q4Km;
        gate.source_quantized = true;
        gate.quantization = None;
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("GGUF block dtypes must not be admitted by a safetensors manifest");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("tensor_format") && message.contains("expected gguf"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_gguf_block_dtype_in_gguf_manifest() {
        let mut manifest = packed_layer_manifest();
        manifest.tensor_format = NativeTensorFormat::Gguf;
        for tensor in &mut manifest.tensors {
            tensor.file = PathBuf::from("model.gguf");
        }
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::Q4Km;
        gate.source_quantized = true;
        gate.quantization = None;
        let (dir, _) = write_fixture(manifest, &["model.gguf"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("GGUF block dtype with logical columns should validate in a GGUF manifest");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_shape_helpers_use_logical_columns_for_gguf_block_dtypes() {
        for dtype in [
            NativeTensorDataType::Q4Km,
            NativeTensorDataType::Q5Km,
            NativeTensorDataType::Q6Km,
            NativeTensorDataType::Q8Zero,
        ] {
            let mut matrix = tensor(
                "model.layers.0.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(0),
                vec![128, 2048],
            );
            matrix.dtype = dtype;
            matrix.source_quantized = true;

            validate_tensor_quantization(&matrix, NativeTensorFormat::Gguf, false, false)
                .unwrap_or_else(|error| panic!("{dtype:?} should be valid GGUF storage: {error}"));
            expect_matrix_shape(&matrix, 128, 2048, "attention_q").unwrap_or_else(|error| {
                panic!("{dtype:?} should retain logical matrix columns: {error}")
            });

            matrix.shape[1] = 256;
            let error = expect_matrix_shape(&matrix, 128, 2048, "attention_q")
                .expect_err("packed-U32 columns must not be accepted for GGUF block dtypes");
            assert!(
                error.to_string().contains("[128, 2048]"),
                "unexpected {dtype:?} matrix error: {error}"
            );

            let mut experts = matrix.clone();
            experts.role = NativeTensorRole::FfnGateExps;
            experts.shape = vec![4, 512, 2048];
            expect_tensor_shape(&experts, &[4, 512, 2048], "ffn_gate_exps").unwrap_or_else(
                |error| panic!("{dtype:?} should retain logical expert columns: {error}"),
            );

            experts.shape[2] = 256;
            let error = expect_tensor_shape(&experts, &[4, 512, 2048], "ffn_gate_exps")
                .expect_err("packed-U32 expert columns must not be accepted for GGUF block dtypes");
            assert!(
                error.to_string().contains("[4, 512, 2048]"),
                "unexpected {dtype:?} expert error: {error}"
            );
        }
    }

    #[test]
    fn native_model_tensor_shapes_use_logical_gguf_dense_ffn_columns() {
        let mut manifest = packed_layer_manifest();
        manifest.tensor_format = NativeTensorFormat::Gguf;
        for tensor in manifest.tensors.iter_mut().filter(|tensor| {
            tensor.layer_index == Some(0)
                && matches!(
                    tensor.role,
                    NativeTensorRole::FfnGateUpPacked | NativeTensorRole::FfnDown
                )
        }) {
            tensor.dtype = NativeTensorDataType::Q4Km;
            tensor.source_quantized = true;
            tensor.quantization = None;
        }

        validate_native_model_tensor_shapes(&manifest)
            .expect("GGUF FFN weights should validate with logical columns");

        manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.layer_index == Some(0) && tensor.role == NativeTensorRole::FfnGateUpPacked
            })
            .expect("fixture should include FFN gate/up")
            .shape[1] = 256;
        let error = validate_native_model_tensor_shapes(&manifest)
            .expect_err("packed-U32 columns must not be accepted for a GGUF FFN weight");
        assert!(
            error.to_string().contains("hidden_size 2048 columns"),
            "unexpected FFN error: {error}"
        );
    }

    #[test]
    fn native_model_tensor_shapes_use_logical_gguf_split_and_q_only_columns() {
        let mut split = mixed_split_projection_manifest();
        split.tensor_format = NativeTensorFormat::Gguf;
        for projection in split.tensors.iter_mut().filter(|tensor| {
            tensor.layer_index == Some(1)
                && matches!(
                    tensor.role,
                    NativeTensorRole::AttentionQ
                        | NativeTensorRole::AttentionK
                        | NativeTensorRole::AttentionV
                        | NativeTensorRole::AttentionO
                )
        }) {
            projection.dtype = NativeTensorDataType::Q4Km;
            projection.source_quantized = true;
            projection.quantization = None;
            projection.shape[1] = 2048;
        }
        validate_native_model_tensor_shapes(&split)
            .expect("GGUF split projections should validate with logical columns");

        split
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::AttentionK
            })
            .expect("fixture should include split K projection")
            .shape[1] = 256;
        let error = validate_native_model_tensor_shapes(&split)
            .expect_err("packed-U32 columns must not be accepted for a GGUF split projection");
        assert!(
            error.to_string().contains("attention_k"),
            "unexpected split projection error: {error}"
        );

        let mut q_only = q_only_kv_shared_manifest();
        q_only.tensor_format = NativeTensorFormat::Gguf;
        let attention_q = q_only
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::AttentionQ
            })
            .expect("fixture should include Q-only projection");
        attention_q.dtype = NativeTensorDataType::Q4Km;
        attention_q.source_quantized = true;
        attention_q.quantization = None;
        validate_native_model_tensor_shapes(&q_only)
            .expect("GGUF Q-only projection should validate with logical columns");

        q_only
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::AttentionQ
            })
            .expect("fixture should include Q-only projection")
            .shape[1] = 256;
        let error = validate_native_model_tensor_shapes(&q_only)
            .expect_err("packed-U32 columns must not be accepted for a GGUF Q-only projection");
        assert!(
            error.to_string().contains("attention_q"),
            "unexpected Q-only projection error: {error}"
        );
    }

    #[test]
    fn native_model_artifacts_allow_8_bit_quantized_moe_router_columns() {
        let mut manifest = moe_layer_manifest();
        let router = manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.layer_index == Some(0) && tensor.role == NativeTensorRole::FfnGateInp
            })
            .expect("fixture should include MoE router projection");
        router.dtype = NativeTensorDataType::U32;
        router.source_quantized = true;
        router.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 8,
        });
        router.shape = vec![128, 704];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("8-bit quantized MoE router should validate with 4 values per u32");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_moe_tensors_without_manifest_config() {
        let mut manifest = moe_layer_manifest();
        manifest.moe = NativeMoeConfig::default();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("missing moe config should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("manifest.moe"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_incomplete_split_moe_experts() {
        let mut manifest = moe_layer_manifest();
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::FfnUpExps)
        });
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("split MoE expert weights should require gate and up tensors");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("ffn_gate_up_exps_packed or ffn_gate_exps/ffn_up_exps"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_mixed_packed_and_split_moe_experts() {
        let mut manifest = moe_layer_manifest();
        manifest.tensors.extend([
            tensor(
                "model.layers.0.experts.gate_proj.weight",
                NativeTensorRole::FfnGateExps,
                Some(0),
                vec![128, 704, 2816],
            ),
            tensor(
                "model.layers.0.experts.up_proj.weight",
                NativeTensorRole::FfnUpExps,
                Some(0),
                vec![128, 704, 2816],
            ),
        ]);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("MoE expert format should be unambiguous per layer");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("must not mix ffn_gate_up_exps_packed"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_qwen3_moe_without_shared_expert() {
        let manifest = switch_moe_manifest("qwen3_moe", false);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("Qwen3 MoE switch experts do not require a shared expert block");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_qwen35_moe_without_shared_expert() {
        let manifest = switch_moe_manifest("qwen3_5", false);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("Qwen3.5 MoE requires the reference shared expert block");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("ffn_shared_expert_gate_inp"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_qwen35_moe_with_shared_expert() {
        let manifest = switch_moe_manifest("qwen3_5", true);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("Qwen3.5 MoE should validate with switch experts and shared expert");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_llama4_shared_expert_without_gate_input() {
        let mut manifest = switch_moe_manifest("llama4", true);
        manifest
            .tensors
            .retain(|tensor| tensor.role != NativeTensorRole::FfnSharedExpertGateInp);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("Llama 4 shared experts do not provide a separate gate input");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_missing_layer_qkv_roles() {
        let mut manifest = packed_layer_manifest();
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::AttentionQkvPacked)
        });
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir).expect_err("missing qkv role should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("attention_qkv_packed"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_orphan_attention_v_projection() {
        let mut manifest = packed_layer_manifest();
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1)
                && matches!(
                    tensor.role,
                    NativeTensorRole::AttentionQkvPacked | NativeTensorRole::AttentionO
                ))
        });
        manifest.tensors.push(tensor(
            "model.layers.1.self_attn.v_proj.weight",
            NativeTensorRole::AttentionV,
            Some(1),
            vec![1024, 2048],
        ));
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("an orphan V projection must trigger the full-attention contract");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("attention_o"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_q_only_kv_shared_layer() {
        let manifest = q_only_kv_shared_manifest();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir).expect("Q-only KV-shared layer should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_q_only_attention_head_count_mismatch() {
        let mut manifest = q_only_kv_shared_manifest();
        manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.role == NativeTensorRole::AttentionQ && tensor.layer_index == Some(1)
            })
            .expect("fixture should include Q-only projection")
            .shape[0] = 1024;
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("Q-only projection heads must match manifest metadata");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("Q-only attention head count must match manifest"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_bad_q_only_quantized_columns() {
        let mut manifest = q_only_kv_shared_manifest();
        let attention_q = manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.role == NativeTensorRole::AttentionQ && tensor.layer_index == Some(1)
            })
            .expect("fixture should include Q-only projection");
        attention_q.dtype = NativeTensorDataType::U32;
        attention_q.source_quantized = true;
        attention_q.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 4,
        });
        attention_q.shape = vec![2048, 255];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("Q-only quantized projections must validate packed columns");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("attention_q must have packed quantized shape [2048, 256]"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_kv_shared_layer_with_own_kv() {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(1, 0);
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::AttentionQkvPacked)
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.1.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(1),
                vec![2048, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(1),
                vec![1024, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(1),
                vec![1024, 2048],
            ),
        ]);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("KV-shared layer with packed QKV should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("KV-shared layer"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_missing_attention_v_when_value_comes_from_key() {
        let manifest = split_layer_manifest_with_value_from_key();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("attention_value_from_key_layers should allow missing attention_v");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_value_from_key_layers_with_attention_v() {
        let mut manifest = split_layer_manifest_with_value_from_key();
        manifest.tensors.push(tensor(
            "model.layers.1.self_attn.v_proj.weight",
            NativeTensorRole::AttentionV,
            Some(1),
            vec![1024, 2048],
        ));
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("value-from-key layer must not also provide attention_v");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("value-from-key layer 1"));
        assert!(message.contains("attention_v"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_value_from_key_layers_with_packed_qkv() {
        let mut manifest = packed_layer_manifest();
        manifest.attention_value_from_key_layers = vec![1];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("value-from-key layer must not provide packed QKV");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("value-from-key layer 1"));
        assert!(message.contains("attention_qkv_packed"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_out_of_range_attention_value_from_key_layers() {
        let mut manifest = packed_layer_manifest();
        manifest.attention_value_from_key_layers = vec![99];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("out-of-range attention_value_from_key_layers should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("attention_value_from_key_layers"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_mismatched_layer_types_length() {
        let mut manifest = packed_layer_manifest();
        manifest.layer_types = vec!["sliding_attention".to_string()];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("layer_types length mismatch should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("layer_types"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_missing_per_layer_input_weights() {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.hidden_size_per_layer_input = 64;
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("per-layer input contract should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("per_layer_embed"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_invalid_kv_shared_source_layer() {
        let mut manifest = packed_layer_manifest();
        manifest.kv_shared_source_layers.insert(1, 99);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("bad KV source should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("kv_shared_source_layers"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_cross_type_kv_shared_source_layer() {
        let mut manifest = packed_layer_manifest();
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(1, 0);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("bad KV source should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("cannot reuse source"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_zero_interleaved_attention_fields() {
        let mut manifest = packed_layer_manifest();
        manifest.rope_theta_swa = Some(0);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("zero rope_theta_swa should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("rope_theta_swa"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_parent_escaping_tensor_paths() {
        let mut manifest = packed_layer_manifest();
        manifest.tensors[0].file = PathBuf::from("../escape.safetensors");
        let (dir, manifest) = write_fixture(manifest, &["model.safetensors"]);
        fs::write(dir.join("..").join("escape.safetensors"), vec![0_u8; 16])
            .expect("escape file should write");
        fs::write(
            dir.join(AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should rewrite");

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("parent path traversal should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("must not escape root_dir"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_hidden_size_shape_mismatches() {
        let mut manifest = packed_layer_manifest();
        manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.role == NativeTensorRole::AttentionNorm && tensor.layer_index == Some(0)
            })
            .expect("attention norm should exist")
            .shape = vec![1024];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("hidden-size mismatch should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("attention_norm"));
        assert!(message.contains("2048"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_ffn_intermediate_shape_mismatches() {
        let mut manifest = packed_layer_manifest();
        manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.role == NativeTensorRole::FfnDown && tensor.layer_index == Some(1)
            })
            .expect("ffn down should exist")
            .shape = vec![2048, 2048];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("ffn intermediate mismatch should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("ffn_down"));
        assert!(message.contains("4096"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_attention_q_norm_shape_mismatches() {
        let mut manifest = packed_layer_manifest();
        manifest.tensors.push(tensor(
            "model.layers.0.self_attn.q_norm.weight",
            NativeTensorRole::AttentionQNorm,
            Some(0),
            vec![2048],
        ));
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("q norm shape mismatch should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("attention_q_norm"));
        assert!(message.contains("128"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_bad_nemotron_attention_projection_shapes() {
        for (role, bad_shape, label) in [
            (
                NativeTensorRole::AttentionQ,
                vec![1024, 2048],
                "attention_q",
            ),
            (NativeTensorRole::AttentionK, vec![512, 2048], "attention_k"),
            (NativeTensorRole::AttentionV, vec![512, 2048], "attention_v"),
            (
                NativeTensorRole::AttentionO,
                vec![2048, 1024],
                "attention_o",
            ),
        ] {
            let mut manifest = nemotron_attention_manifest();
            manifest
                .tensors
                .iter_mut()
                .find(|tensor| tensor.role == role && tensor.layer_index == Some(0))
                .expect("fixture should include Nemotron attention projection")
                .shape = bad_shape;
            let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

            let error = NativeModelArtifacts::from_dir(&dir)
                .expect_err("Nemotron attention projections must match runtime reshape dimensions");
            let NativeModelError::InvalidManifest { message } = error else {
                panic!("expected invalid manifest error");
            };
            assert!(message.contains(label), "unexpected error: {message}");

            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn native_model_artifacts_reject_attention_o_input_dim_mismatches() {
        let mut manifest = packed_layer_manifest();
        manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.role == NativeTensorRole::AttentionO && tensor.layer_index == Some(0)
            })
            .expect("attention o should exist")
            .shape = vec![2048, 1024];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("attention o mismatch should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("attention_o"));
        assert!(message.contains("2048"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_non_positive_rope_theta() {
        let mut manifest = packed_layer_manifest();
        manifest.rope_theta = Some(0);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("non-positive rope theta should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("rope_theta"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_non_positive_query_pre_attn_scalar() {
        let mut manifest = packed_layer_manifest();
        manifest.query_pre_attn_scalar = Some(0);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("non-positive query pre attention scalar should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("query_pre_attn_scalar"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_non_positive_attention_logit_softcap() {
        let mut manifest = packed_layer_manifest();
        manifest.attention_logit_softcap = Some(0);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("non-positive attention softcap should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("attention_logit_softcap"));

        let _ = fs::remove_dir_all(dir);
    }
}
