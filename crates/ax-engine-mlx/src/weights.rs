use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::PathBuf;
use std::sync::OnceLock;

use mlx_sys::{
    MlxArray, MlxDtype, MlxQuantizationMode, add, astype, broadcast_to, concatenate, contiguous,
    dequantize, dequantize_with_mode, eval, flatten, from_fp8, load_safetensors,
    load_safetensors_mmap, multiply, quantize, reshape, slice, stack, take, transpose, view,
};

use ax_engine_core::{
    NativeLinearAttentionConfig, NativeMlaAttentionConfig, NativeModelArtifacts,
    NativeTensorQuantization, NativeTensorRole, NativeTensorSpec, PipelineRankAssignment,
    WeightSanitize,
};

use crate::fastpath::{
    dense_attention_qkv_packing_enabled, dense_ffn_gate_up_packing_enabled,
    linear_attention_projection_packing_enabled,
};
use crate::gemma4_assistant_mtp::{Gemma4AssistantMtpStatus, load_gemma4_assistant_mtp_status};
use crate::model::MlaAttentionConfig;
use crate::sampling::MlxSamplingParams;

/// All weight arrays for one model.
pub struct ModelWeights {
    pub token_embedding: QuantizedWeight,
    pub final_norm: MlxArray,
    pub lm_head: QuantizedWeight,
    pub layers: Vec<LayerWeights>,
    /// Per-layer token embedding table (Gemma4 2B/4B, shape [vocab_per_layer, num_layers*per_layer_dim]).
    pub per_layer_embed: Option<QuantizedWeight>,
    /// Global projection hidden → num_layers*per_layer_dim (Gemma4 2B/4B).
    pub per_layer_model_proj: Option<QuantizedWeight>,
    /// RMSNorm weight over per_layer_dim applied after model projection (Gemma4 2B/4B).
    pub per_layer_proj_norm: Option<MlxArray>,
    /// MTP (Multi-Token Prediction) weights loaded from a `mtp.safetensors` sidecar.
    /// `None` when the checkpoint has no MTP sidecar.
    pub mtp: Option<MtpWeights>,
    /// Gemma 4 Assistant MTP contract/validation status. The assistant forward
    /// path is intentionally separate from the Qwen3-Next `MtpWeights` path.
    pub gemma4_assistant_mtp: Gemma4AssistantMtpStatus,
    /// Gemma4 Assistant pre-projection from target embedding+hidden to assistant hidden.
    pub assistant_pre_projection: Option<QuantizedWeight>,
    /// Gemma4 Assistant post-projection from assistant hidden back to target hidden.
    pub assistant_post_projection: Option<QuantizedWeight>,
    /// EmbeddingGemma sentence-transformers Dense head, projection 1 (hidden →
    /// 4*hidden, no bias, identity). Applied after mean pooling, before L2 norm.
    pub embedding_dense_0: Option<QuantizedWeight>,
    /// EmbeddingGemma sentence-transformers Dense head, projection 2 (4*hidden →
    /// hidden, no bias, identity). Applied after `embedding_dense_0`, before L2 norm.
    pub embedding_dense_1: Option<QuantizedWeight>,
    /// Gemma4 Unified encoder-free vision embedder + connector.
    pub gemma4_unified_vision: Option<Gemma4UnifiedVisionWeights>,
    /// Gemma4 Unified encoder-free audio connector.
    pub gemma4_unified_audio: Option<Gemma4UnifiedAudioWeights>,
    /// Standard Gemma 4 ViT tower + vision-to-language projection.
    pub gemma4_vl_vision: Option<crate::gemma4_vl::Gemma4VlVisionWeights>,
    pub diffusion_self_conditioning: Option<DiffusionSelfConditioningWeights>,
    /// MTP weights for GLM 4.7 Flash: separate sidecar with MLA-based head.
    pub glm_mtp: Option<GlmMtpWeights>,
    /// Root-level DeepSeek V4 hyper-connection head (`hc_head_*`).
    pub deepseek_v4_head: Option<DeepseekV4HeadWeights>,
    /// DeepSeek V4 MTP (nextn) predictor tensors, loaded for a deferred
    /// runtime-MTP phase; not consumed by the forward path yet.
    pub deepseek_v4_nextn: Option<DeepseekV4NextnWeights>,
    /// Unlimited-OCR dual vision (SAM-ViT-B + CLIP-L) + projector.
    pub unlimited_ocr_vision: Option<crate::unlimited_ocr::UnlimitedOcrVisionWeights>,
    /// Qwen3-VL portable ViT tower (WS-V2). `None` until HF vision weights are
    /// mapped for the checkpoint; image prefill fail-closes when media is present.
    pub qwen3_vl_vision: Option<crate::qwen3_vl::Qwen3VlVisionWeights>,
    /// MiniCPM-V 4.6 SigLIP + VitMerger + pixel-shuffle merger.
    pub minicpm_v46_vision: Option<crate::minicpm_v::MiniCpmV46VisionWeights>,
    /// Nemotron H Nano Omni RADIO vision and Parakeet media towers.
    pub nemotron_omni: Option<crate::nemotron_omni::NemotronOmniWeights>,
    /// SSD expert-streaming pager (`ax_expert_stream.json` layer-stack mode).
    /// `None` for the default fully-resident load; when set, MoE expert
    /// stacks are paged per layer instead of resident.
    pub expert_stream: Option<std::sync::Arc<crate::expert_stream::ExpertStackPager>>,
}

/// The exact weights owned by one static pipeline rank.
///
/// The initial distributed runtime deliberately supports the dense Llama 3
/// family only. Endpoint tensors are optional because only rank 0 embeds
/// tokens and only the final rank applies normalization and the LM head.
pub struct PipelineStageWeights {
    pub assignment: PipelineRankAssignment,
    pub token_embedding: Option<QuantizedWeight>,
    pub final_norm: Option<MlxArray>,
    pub lm_head: Option<QuantizedWeight>,
    /// Layer weights ordered by global layer index in `assignment.layers`.
    pub layers: Vec<LayerWeights>,
}

impl PipelineStageWeights {
    pub fn global_layer_index(&self, local_index: usize) -> Option<usize> {
        let local = u32::try_from(local_index).ok()?;
        let global = self.assignment.layers.start.checked_add(local)?;
        (global < self.assignment.layers.end).then_some(global as usize)
    }
}

/// Gemma4 Unified vision path, matching vLLM's
/// `Gemma4UnifiedVisionEmbedder` followed by `Gemma4MultimodalEmbedder`.
pub struct Gemma4UnifiedVisionWeights {
    pub patch_ln1_weight: MlxArray,
    pub patch_ln1_bias: MlxArray,
    pub patch_dense: QuantizedWeight,
    pub patch_dense_bias: MlxArray,
    pub patch_ln2_weight: MlxArray,
    pub patch_ln2_bias: MlxArray,
    pub pos_embedding: MlxArray,
    pub pos_norm_weight: MlxArray,
    pub pos_norm_bias: MlxArray,
    pub projection: QuantizedWeight,
}

/// Gemma4 Unified audio path, matching vLLM's direct
/// `Gemma4MultimodalEmbedder` projection.
pub struct Gemma4UnifiedAudioWeights {
    pub projection: QuantizedWeight,
}

pub struct DiffusionSelfConditioningWeights {
    pub pre_norm: MlxArray,
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
}

/// Weights for a recurrent MTP (Multi-Token Prediction) draft head.
///
/// The single transformer layer is applied up to `max_depth` times to produce
/// up to `max_depth` speculative draft tokens per decode step.
///
/// Input combination: `fc(cat([rms_norm(embed(prev_token), pre_fc_norm_embedding),
///                              rms_norm(main_hidden, pre_fc_norm_hidden)], dim=-1))`
/// Draft logits: `rms_norm(h, mtp_norm) @ main_model.lm_head`
pub struct MtpWeights {
    /// FC projection: concat(enorm, hnorm) [2*hidden] → hidden.  Plain BF16.
    pub fc: QuantizedWeight,
    /// Final RMSNorm before applying the shared lm_head (mtp.norm.weight).
    pub mtp_norm: MlxArray,
    /// Optional draft-only LM head, re-quantized from the main `lm_head` using
    /// the MTPLX runtime recommendation or an explicit AX override.
    pub draft_lm_head: Option<QuantizedWeight>,
    /// Pre-FC RMSNorm applied to the embedded token (mtp.pre_fc_norm_embedding.weight).
    pub pre_fc_norm_embedding: MlxArray,
    /// Pre-FC RMSNorm applied to the main model hidden state (mtp.pre_fc_norm_hidden.weight).
    pub pre_fc_norm_hidden: MlxArray,
    // Transformer layer norms.
    pub attn_norm: MlxArray,
    pub ffn_norm: MlxArray,
    /// Optional per-head QK norms (Qwen3-style).
    pub q_norm: Option<MlxArray>,
    pub k_norm: Option<MlxArray>,
    // Attention projections.
    pub q_proj: QuantizedWeight,
    pub k_proj: QuantizedWeight,
    pub v_proj: QuantizedWeight,
    pub o_proj: QuantizedWeight,
    // FFN projections.
    pub ffn_layer: LayerWeights,
    /// Number of query heads (inferred from q_proj shape).
    pub n_heads: usize,
    /// Number of KV heads (inferred from k_proj shape).
    pub n_kv_heads: usize,
    /// Per-head dimension (inferred from q_proj / n_heads).
    pub head_dim: usize,
    /// Maximum speculative depth: how many times to apply this head recurrently.
    pub max_depth: usize,
    /// Sampling parameters for draft token generation.
    /// From `mtplx_runtime.json` `recommended_draft_sampler` (default: temp=0.7, top_k=20, top_p=0.95).
    /// Temperature > 0 enables rejection-sampling acceptance instead of greedy argmax comparison.
    pub draft_sampling: MlxSamplingParams,
}

/// Weights for GLM 4.7 Flash MTP head.
///
/// Layout diverges from `MtpWeights` (Qwen3): uses GLM MLA attention (not
/// standard q/k/v/o), fuses tokens via `eh_proj` (not `mtp.fc`), and projects
/// draft logits through a private `shared_head` rather than the shared `lm_head`.
pub struct GlmMtpWeights {
    /// enorm: RMSNorm applied to embedded prev token before concat.
    pub enorm: MlxArray,
    /// hnorm: RMSNorm applied to main hidden state before concat.
    pub hnorm: MlxArray,
    /// eh_proj: [2*hidden → hidden] linear projection.
    pub eh_proj: QuantizedWeight,
    /// shared_head.norm: RMSNorm before draft logit projection.
    pub shared_head_norm: MlxArray,
    /// shared_head.head: [hidden → vocab] draft logit projection.
    pub shared_head: QuantizedWeight,
    /// Full GLM transformer layer (MLA attention + MoE FFN).
    pub layer: LayerWeights,
    /// MLA attention config cloned from main model config at load time.
    pub mla_config: MlaAttentionConfig,
    /// Maximum speculative depth (from `glm_mtp_runtime.json` or default 1).
    pub max_depth: usize,
    /// Draft sampling parameters.
    pub draft_sampling: MlxSamplingParams,
}

/// Weights (and optional quantization data) for one transformer layer.
pub struct LayerWeights {
    pub attn_norm: MlxArray,
    /// post_attention_layernorm for models that apply it to attention output
    /// before the residual add (for example Gemma4). Qwen3 and GLM4MoELite use
    /// post_attention_layernorm as the pre-FFN norm after the residual instead.
    pub attn_post_norm: Option<MlxArray>,
    pub q_norm: Option<MlxArray>,
    pub k_norm: Option<MlxArray>,
    // Split Q/K/V projections (None for KV-shared layers that reuse a source layer's KV).
    pub q_proj: Option<QuantizedWeight>,
    pub k_proj: Option<QuantizedWeight>,
    pub v_proj: Option<QuantizedWeight>,
    // Packed QKV projection (some architectures).
    pub qkv_packed: Option<QuantizedWeight>,
    pub o_proj: Option<QuantizedWeight>,
    // Linear attention (Qwen3.5 hybrid layers). Present instead of full-attention QKV/O.
    pub linear_attn: Option<LinearAttentionWeights>,
    // GLM4MoELite MLA attention. Present instead of standard full-attention Q/K/V.
    pub glm_mla_attn: Option<GlmMlaAttentionWeights>,
    // DeepSeek V4 (Flash) attention + hyper-connection tensors. Present instead
    // of standard full-attention Q/K/V/O for deepseek_v4 manifests.
    pub deepseek_v4: Option<DeepseekV4LayerWeights>,
    // Dense FFN norms and weights.
    pub ffn_norm: MlxArray,
    pub ffn_post_norm: Option<MlxArray>,
    pub gate_proj: Option<QuantizedWeight>,
    pub up_proj: Option<QuantizedWeight>,
    pub gate_up_packed: Option<QuantizedWeight>,
    pub down_proj: Option<QuantizedWeight>,
    // MoE: extra norms (present when this layer has a MoE block).
    pub ffn_norm2: Option<MlxArray>,
    pub ffn_post_norm1: Option<MlxArray>,
    pub ffn_post_norm2: Option<MlxArray>,
    // MoE: router weights.
    pub router_proj: Option<QuantizedWeight>,
    pub router_correction_bias: Option<MlxArray>,
    pub router_scale: Option<MlxArray>,
    /// Precomputed `router_scale * hidden_size^-0.5` for Gemma4 MoE router RMSNorm.
    pub router_combined_scale: Option<MlxArray>,
    /// Per-expert output scale (Gemma4 MoE): multiply top-k weights by this after softmax.
    pub router_expert_scale: Option<MlxArray>,
    /// Per-layer scalar applied to hidden states after the FFN residual (Gemma4).
    pub layer_scalar: Option<MlxArray>,
    /// Per-layer input gate projection: hidden → per_layer_dim (Gemma4 2B/4B).
    pub per_layer_gate: Option<QuantizedWeight>,
    /// Per-layer output projection: per_layer_dim → hidden (Gemma4 2B/4B).
    pub per_layer_proj_w: Option<QuantizedWeight>,
    /// Post-gating RMSNorm weight (Gemma4 2B/4B).
    pub per_layer_post_norm: Option<MlxArray>,
    // MoE: expert weights (shape [num_experts, expert_size, hidden] / packed).
    pub shared_expert_gate: Option<QuantizedWeight>,
    pub shared_gate_up_proj: Option<QuantizedWeight>,
    pub shared_gate_proj: Option<QuantizedWeight>,
    pub shared_up_proj: Option<QuantizedWeight>,
    pub shared_down_proj: Option<QuantizedWeight>,
    pub gate_up_exps_packed: Option<QuantizedWeight>,
    pub gate_exps: Option<QuantizedWeight>,
    pub up_exps: Option<QuantizedWeight>,
    pub down_exps: Option<QuantizedWeight>,
    /// GPT-OSS MXFP4 gate-up expert weights kept packed at load time.
    ///
    /// `weight` is the sanitized u32 view of the MXFP4 blocks tensor
    /// (`[num_experts, 2 * intermediate, packed_in]`); `scales` is the
    /// matching E8M0 scale tensor. Forward uses `gather_qmm` with
    /// `mode=mxfp4` so expert parameters stay 4-bit in unified memory.
    pub mxfp4_gate_up_exps: Option<Mxfp4ExpertWeight>,
    /// GPT-OSS MXFP4 down expert weights kept packed at load time.
    /// Shape convention matches gate-up: `[num_experts, hidden, packed_in]`
    /// for the packed weight and matching scales.
    pub mxfp4_down_exps: Option<Mxfp4ExpertWeight>,
    /// GPT-OSS per-head learned attention sink weight. Shape: `[num_attention_heads]`.
    pub attn_sink: Option<MlxArray>,
    /// Per-layer AWQ-lite smoothing reciprocal `1/s` of shape `[hidden_size]`.
    /// Populated by `apply_rotated_checkpoint` when the rotated checkpoint was
    /// generated with `--smoothing weight_mag`. The forward path multiplies
    /// the rotated activation by this vector before the gate/up matmul so
    /// the per-channel scaling baked into the rotated weights cancels.
    pub rotation_smoothing_inverse: Option<MlxArray>,
    /// SSD expert-streaming source for this layer. When set, the packed
    /// expert fields above stay `None` until the MoE forward pages the
    /// layer's fused expert stack in through this handle.
    pub expert_stream: Option<std::sync::Arc<crate::expert_stream::ExpertLayerSource>>,
}

/// Weights for a GLM4MoELite MLA attention layer.
pub struct GlmMlaAttentionWeights {
    /// q_a_proj and kv_a_proj fused into one `[q_lora_rank + kv_lora_rank + qk_rope_head_dim, hidden]`
    /// weight. Eliminates one matmul kernel launch per layer during prefill.
    pub qa_kva_fused: QuantizedWeight,
    pub q_a_norm: MlxArray,
    pub q_b_proj: QuantizedWeight,
    pub kv_a_norm: MlxArray,
    pub embed_q: QuantizedWeight,
    pub unembed_out: QuantizedWeight,
}

/// Weights for a DeepSeek V4 (Flash) sliding-window compressor
/// (`attn.compressor.*`). Present on layers whose compress ratio is 4 or 128.
pub struct DeepseekV4CompressorWeights {
    /// Fused KV projection (`compressor.wkv`).
    pub kv: QuantizedWeight,
    /// Compressor gate (`compressor.wgate`).
    pub gate: QuantizedWeight,
    /// Absolute positional embedding (`compressor.ape`,
    /// `[coff*head_dim, ratio]` with `coff = 1 + (ratio == 4)`).
    pub ape: MlxArray,
    /// Compressor RMSNorm (`compressor.norm`).
    pub norm: MlxArray,
}

/// Weights for a DeepSeek V4 (Flash) sparse indexer (`attn.indexer.*`).
/// Present only on ratio-4 layers.
pub struct DeepseekV4IndexerWeights {
    /// Per-token score projection (`indexer.weights_proj`).
    pub proj: QuantizedWeight,
    /// Query up-projection (`indexer.wq_b`).
    pub qb: QuantizedWeight,
    /// Indexer compressor KV projection (`indexer.compressor.wkv`).
    pub compressor_kv: QuantizedWeight,
    /// Indexer compressor gate (`indexer.compressor.wgate`).
    pub compressor_gate: QuantizedWeight,
    /// Indexer compressor positional embedding (`indexer.compressor.ape`).
    pub compressor_ape: MlxArray,
    /// Indexer compressor RMSNorm (`indexer.compressor.norm`).
    pub compressor_norm: MlxArray,
}

/// Weights for a DeepSeek V4 (Flash) layer's attention + hyper-connection
/// tensors. The MoE router gate, expert stacks, and shared experts use the
/// generic `LayerWeights` fields (`router_proj`, `gate_exps`, …).
pub struct DeepseekV4LayerWeights {
    /// Q LoRA down-projection (`attn.wq_a`).
    pub wq_a: QuantizedWeight,
    /// RMSNorm over the Q latent (`attn.q_a_norm`).
    pub q_a_norm: MlxArray,
    /// Q LoRA up-projection (`attn.wq_b`).
    pub wq_b: QuantizedWeight,
    /// Fused KV projection (`attn.wkv`) feeding the single latent KV head.
    pub wkv: QuantizedWeight,
    /// RMSNorm over the `wkv` output (`attn.kv_norm`).
    pub kv_norm: MlxArray,
    /// Grouped output down-projection (`attn.wo_a`, per `o_groups`;
    /// `[H*D/G, R_o, G]`).
    pub wo_a: QuantizedWeight,
    /// Output LoRA up-projection (`attn.wo_b`, `o_lora_rank → hidden`).
    pub wo_b: QuantizedWeight,
    /// Learned per-head attention sink (`attn.attn_sink`), `[n_heads]` f32.
    pub attn_sink: Option<MlxArray>,
    /// Hyper-connection attention-branch coefficients (`hc_attn_fn`).
    pub hc_attn_fn: MlxArray,
    /// Hyper-connection attention-branch base (`hc_attn_base`).
    pub hc_attn_base: MlxArray,
    /// Hyper-connection attention-branch scale (`hc_attn_scale`).
    pub hc_attn_scale: MlxArray,
    /// Hyper-connection FFN-branch coefficients (`hc_ffn_fn`).
    pub hc_ffn_fn: MlxArray,
    /// Hyper-connection FFN-branch base (`hc_ffn_base`).
    pub hc_ffn_base: MlxArray,
    /// Hyper-connection FFN-branch scale (`hc_ffn_scale`).
    pub hc_ffn_scale: MlxArray,
    /// Sliding-window compressor (ratio-4/128 layers only).
    pub compressor: Option<DeepseekV4CompressorWeights>,
    /// Sparse indexer (ratio-4 layers only).
    pub indexer: Option<DeepseekV4IndexerWeights>,
    /// Hash-routing token→expert table (`ffn.gate.tid2eid`, `[vocab, topk]`
    /// I32/U32) on the first `num_hash_layers` MoE layers; mutually exclusive
    /// with the generic `router_correction_bias`.
    pub tid2eid: Option<MlxArray>,
}

/// Root-level DeepSeek V4 hyper-connection head (`hc_head_*`), applied before
/// the final norm to collapse the packed residual stream.
pub struct DeepseekV4HeadWeights {
    pub hc_head_fn: MlxArray,
    pub hc_head_base: MlxArray,
    pub hc_head_scale: MlxArray,
}

/// DeepSeek V4 MTP (nextn) predictor tensors. Loaded from manifest-side
/// nextn roles (GGUF layout `layers.N.nextn.*` / raw HF `mtp.N.*`) and/or the
/// `mtp.safetensors` sidecar. The block itself (`layer`) is one full V4
/// transformer layer: GGUF-layout manifests carry it at layer index
/// `num_hidden_layers`, raw-HF packages ship it in the sidecar.
#[derive(Default)]
pub struct DeepseekV4NextnWeights {
    /// MTP embedding projection (`nextn.e_proj` / `mtp.N.e_proj`).
    pub e_proj: Option<QuantizedWeight>,
    /// MTP hidden projection (`nextn.h_proj` / `mtp.N.h_proj`).
    pub h_proj: Option<QuantizedWeight>,
    /// MTP fused embedding+hidden projection (`nextn.eh_proj`).
    pub eh_proj: Option<QuantizedWeight>,
    /// MTP embedding RMSNorm (`nextn.enorm`).
    pub enorm: Option<MlxArray>,
    /// MTP hidden RMSNorm (`nextn.hnorm`).
    pub hnorm: Option<MlxArray>,
    /// MTP shared-head RMSNorm (`nextn.shared_head_norm` / `mtp.N.norm`).
    pub shared_head_norm: Option<MlxArray>,
    /// MTP shared token embedding (`nextn.embed_tokens`).
    pub embed_tokens: Option<QuantizedWeight>,
    /// MTP shared LM head (`nextn.shared_head_head`).
    pub shared_head_head: Option<QuantizedWeight>,
    /// MTP-specific hyper-connection head (`mtp.N.hc_head_*`). When absent the
    /// draft path falls back to the target root head (legacy packs only).
    pub hc_head: Option<DeepseekV4HeadWeights>,
    /// The nextn transformer block (raw-path attention + learned-router MoE,
    /// never hash-routed). `None` when the artifact ships only the
    /// manifest-side sidecar roles (deferred runtime-MTP phase).
    pub layer: Option<Box<LayerWeights>>,
}

impl DeepseekV4NextnWeights {
    fn is_empty(&self) -> bool {
        self.e_proj.is_none()
            && self.h_proj.is_none()
            && self.eh_proj.is_none()
            && self.enorm.is_none()
            && self.hnorm.is_none()
            && self.shared_head_norm.is_none()
            && self.embed_tokens.is_none()
            && self.shared_head_head.is_none()
            && self.hc_head.is_none()
            && self.layer.is_none()
    }

    /// Fill every missing piece from `sidecar` (manifest-side tensors win).
    fn merged_with(mut self, sidecar: DeepseekV4NextnWeights) -> DeepseekV4NextnWeights {
        self.e_proj = self.e_proj.or(sidecar.e_proj);
        self.h_proj = self.h_proj.or(sidecar.h_proj);
        self.eh_proj = self.eh_proj.or(sidecar.eh_proj);
        self.enorm = self.enorm.or(sidecar.enorm);
        self.hnorm = self.hnorm.or(sidecar.hnorm);
        self.shared_head_norm = self.shared_head_norm.or(sidecar.shared_head_norm);
        self.embed_tokens = self.embed_tokens.or(sidecar.embed_tokens);
        self.shared_head_head = self.shared_head_head.or(sidecar.shared_head_head);
        self.hc_head = self.hc_head.or(sidecar.hc_head);
        self.layer = self.layer.or(sidecar.layer);
        self
    }
}

/// Weights for a Qwen3.5 GatedDelta linear-attention layer.
///
/// Also carries Nemotron-H Mamba-2 mixer tensors when `in_proj_qkvz` is the
/// packed Mamba `in_proj` (gate | conv_input | dt) and `d` / `conv1d_bias` are set.
pub struct LinearAttentionWeights {
    pub in_proj_qkv: Option<QuantizedWeight>,
    pub in_proj_z: Option<QuantizedWeight>,
    pub in_proj_a: Option<QuantizedWeight>,
    pub in_proj_b: Option<QuantizedWeight>,
    pub in_proj_qkvz: Option<QuantizedWeight>,
    pub in_proj_ba: Option<QuantizedWeight>,
    /// Load-time row-concat of matching-bit `in_proj_qkvz` + `in_proj_ba`.
    /// Prefill `qw`s this once instead of concatenating packed weights every
    /// layer/chunk. Mixed-bit AXQ layers leave this `None`.
    pub fused_qkvz_ba: Option<QuantizedWeight>,
    /// Prefill-only 2-bit gs32 overlay of `in_proj_qkvz`. Checkpoint files
    /// stay 4/6-bit; decode keeps the original pack. Not a Hub requant.
    pub prefill_q2_qkvz: Option<QuantizedWeight>,
    /// Prefill-only 2-bit gs32 overlay of `in_proj_ba`.
    pub prefill_q2_ba: Option<QuantizedWeight>,
    /// Conv1d kernel dequantized at load time so `linear_attention_forward` never
    /// re-dequantizes per step. Shape: `[conv_dim, conv_kernel_dim, 1]`.
    pub conv1d_dense: MlxArray,
    /// Optional conv1d bias (Nemotron-H Mamba-2 `use_conv_bias=true`).
    pub conv1d_bias: Option<MlxArray>,
    pub dt_bias: MlxArray,
    pub a_log: MlxArray,
    /// Mamba-2 skip residual `D` (per head). Qwen gated-delta leaves this `None`.
    pub d: Option<MlxArray>,
    pub norm: MlxArray,
    pub out_proj: QuantizedWeight,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LinearAttentionProjectionRowSource {
    Qkv(usize),
    Z(usize),
    B(usize),
    A(usize),
}

/// Packed MXFP4 expert matrix for GPT-OSS MoE.
///
/// Stays quantized in memory; runtime matmul uses `gather_qmm` with
/// `MlxQuantizationMode::Mxfp4` (group_size=32, bits=4, no biases).
#[derive(Clone)]
pub struct Mxfp4ExpertWeight {
    pub weight: MlxArray,
    pub scales: MlxArray,
}

impl Mxfp4ExpertWeight {
    pub const GROUP_SIZE: i32 = 32;
    pub const BITS: i32 = 4;
}

/// A weight matrix plus optional MLX affine quantization metadata.
///
/// When `scales` is `Some`, the weight tensor contains packed affine-quantized
/// integers and must be multiplied via `mlx_quantized_matmul` rather than
/// regular matmul.
///
/// `Clone` is implemented because `MlxArray` clones are cheap refcount bumps
/// (`mlx_array_set`), and per-layer compiled closures (e.g. the compiled
/// shared-expert forward) need to capture cloned `QuantizedWeight`s as graph
/// constants.
#[derive(Clone)]
pub struct QuantizedWeight {
    pub weight: MlxArray,
    pub scales: Option<MlxArray>,
    /// Affine (or mode-specific) **group** quant biases, shape tied to groups —
    /// **not** the dense per-expert Linear bias.
    pub biases: Option<MlxArray>,
    pub group_size: i32,
    pub bits: i32,
    /// Quantization mode string from the manifest (`affine`, `mxfp4`, …).
    /// Defaults to affine for legacy checkpoints.
    pub mode: String,
    /// Dense Linear bias for switch/expert layers, shape `[num_experts, out]`.
    /// Matches mlx-lm `SwitchLinear.bias` / `QuantizedSwitchLinear.bias`, applied
    /// as `y += bias[indices]` after `gather_qmm` (see switch_layers.py).
    pub linear_bias: Option<MlxArray>,
    /// Contiguous `[in, out]` view of an unquantized `[out, in]` weight,
    /// materialized once at load. Decode `qw` uses this so each token does a
    /// coalesced `x @ W_t` instead of a lazy strided transpose of a multi-GB
    /// `lm_head`. Quantized tensors leave this `None`.
    pub decode_weight_t: Option<MlxArray>,
    /// Load-time affine decode cache of an unquantized `lm_head`.
    /// Decode `qw` prefers this (2-bit gs=64) so each token streams ~0.4 GB
    /// instead of 2.54 GB BF16. Prefill keeps the BF16 `W_t` GEMM. Not a Hub
    /// requant; the checkpoint files are unchanged.
    pub decode_q4_weight: Option<MlxArray>,
    pub decode_q4_scales: Option<MlxArray>,
    pub decode_q4_biases: Option<MlxArray>,
}

/// Decode-only `lm_head` cache. 2-bit gs64 cuts ~2.14 GB/token vs BF16
/// (q4 left ~0.8 tok/s on the 1.20 bar). Prefill must not use this cache.
pub const DECODE_LM_HEAD_QUANT_BITS: i32 = 2;
pub const DECODE_LM_HEAD_QUANT_GROUP_SIZE: i32 = 64;

impl QuantizedWeight {
    pub fn new(weight: MlxArray, scales: Option<MlxArray>, biases: Option<MlxArray>) -> Self {
        Self::with_quantization(weight, scales, biases, None)
    }

    pub fn with_quantization(
        weight: MlxArray,
        scales: Option<MlxArray>,
        biases: Option<MlxArray>,
        quantization: Option<&NativeTensorQuantization>,
    ) -> Self {
        let quantization = quantization.cloned().unwrap_or_default();
        Self {
            weight,
            scales,
            biases,
            group_size: quantization.group_size as i32,
            bits: quantization.bits as i32,
            mode: if quantization.mode.is_empty() {
                "affine".to_string()
            } else {
                quantization.mode
            },
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        }
    }

    pub fn with_linear_bias(mut self, linear_bias: Option<MlxArray>) -> Self {
        self.linear_bias = linear_bias;
        self
    }

    /// Materialize a contiguous `[in, out]` copy of an unquantized weight.
    ///
    /// Intended for the decode `lm_head` only. After the copy is resident,
    /// `weight` is replaced with a lazy `[out, in]` transpose *of that copy*
    /// so decode does not keep a second 2.54 GB buffer in the Metal
    /// residency set. No-ops when the tensor is quantized, not rank-2, or
    /// already prepared.
    pub fn prepare_contiguous_decode_weight_t(&mut self) {
        if self.decode_weight_t.is_some() || self.scales.is_some() {
            return;
        }
        let shape = self.weight.shape();
        if shape.len() != 2 || shape[0] <= 0 || shape[1] <= 0 {
            return;
        }
        let transposed = contiguous(&transpose(&self.weight, &[1, 0], None), None);
        eval(&[&transposed]);
        // Drop the original [out, in] allocation. Readers that still want
        // that layout go through a lazy view of the single W_t buffer.
        self.weight = transpose(&transposed, &[1, 0], None);
        self.decode_weight_t = Some(transposed);
    }

    /// Build a 2-bit gs64 affine decode cache from an unquantized rank-2 weight.
    ///
    /// Same `mlx_sys::quantize` path as MTP `draft_lm_head`. No-ops when the
    /// tensor is already quantized, not rank-2, or last dim is not a
    /// multiple of 64.
    pub fn prepare_decode_q4_lm_head(&mut self) {
        if self.decode_q4_weight.is_some() || self.scales.is_some() {
            return;
        }
        let shape = self.weight.shape();
        if shape.len() != 2 || shape[0] <= 0 || shape[1] <= 0 {
            return;
        }
        if shape[1] % DECODE_LM_HEAD_QUANT_GROUP_SIZE != 0 {
            return;
        }
        let dense = astype(&self.weight, MlxDtype::Bfloat16, None);
        eval(&[&dense]);
        let quantized = quantize(
            &dense,
            Some(DECODE_LM_HEAD_QUANT_GROUP_SIZE),
            Some(DECODE_LM_HEAD_QUANT_BITS),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        if quantized.len() < 3 {
            return;
        }
        eval(&[&quantized[0], &quantized[1], &quantized[2]]);
        self.decode_q4_weight = Some(quantized[0].clone());
        self.decode_q4_scales = Some(quantized[1].clone());
        self.decode_q4_biases = Some(quantized[2].clone());
    }

    pub fn is_quantized(&self) -> bool {
        self.scales.is_some()
    }

    pub fn mlx_quantization_mode(&self) -> MlxQuantizationMode {
        match self.mode.as_str() {
            "mxfp4" => MlxQuantizationMode::Mxfp4,
            "mxfp8" => MlxQuantizationMode::Mxfp8,
            "nvfp4" => MlxQuantizationMode::Nvfp4,
            _ => MlxQuantizationMode::Affine,
        }
    }

    pub fn matching_affine_quant(&self, other: &Self) -> bool {
        self.scales.is_some()
            && other.scales.is_some()
            && self.bits == other.bits
            && self.group_size == other.group_size
            && self.mode == other.mode
            && self.bits > 0
            && self.group_size > 0
    }

    /// Concatenate two matching affine projections along the output axis.
    pub fn concat_output_rows(&self, other: &Self) -> Option<Self> {
        if !self.matching_affine_quant(other) {
            return None;
        }
        let a_shape = self.weight.shape();
        let b_shape = other.weight.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[1] != b_shape[1] {
            return None;
        }
        let weight = concatenate(&[&self.weight, &other.weight], 0, None);
        let scales = concatenate(&[self.scales.as_ref()?, other.scales.as_ref()?], 0, None);
        let biases = match (&self.biases, &other.biases) {
            (Some(ab), Some(bb)) => Some(concatenate(&[ab, bb], 0, None)),
            (None, None) => None,
            _ => return None,
        };
        Some(Self {
            weight,
            scales: Some(scales),
            biases,
            group_size: self.group_size,
            bits: self.bits,
            mode: self.mode.clone(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        })
    }
}

impl LinearAttentionWeights {
    /// Materialize matching-bit QKVZ+BA into one packed weight at load.
    pub fn prepare_fused_qkvz_ba_prefill(&mut self) {
        if self.fused_qkvz_ba.is_some() {
            return;
        }
        let (Some(qkvz), Some(ba)) = (&self.in_proj_qkvz, &self.in_proj_ba) else {
            return;
        };
        let Some(fused) = qkvz.concat_output_rows(ba) else {
            return;
        };
        eval_packed_projection(&fused);
        self.fused_qkvz_ba = Some(fused);
    }

    /// Build 2-bit gs32 prefill overlays of packed QKVZ/BA.
    ///
    /// Dequantizes the checkpoint affine pack and requants to 2-bit. No-ops
    /// when either projection is missing, not affine-quantized, or already
    /// 2-bit. Decode keeps `in_proj_qkvz` / `in_proj_ba`.
    pub fn prepare_prefill_q2_projections(&mut self) {
        if self.prefill_q2_qkvz.is_some() || self.prefill_q2_ba.is_some() {
            return;
        }
        self.prefill_q2_qkvz = self
            .in_proj_qkvz
            .as_ref()
            .and_then(requant_affine_to_prefill_q2);
        self.prefill_q2_ba = self
            .in_proj_ba
            .as_ref()
            .and_then(requant_affine_to_prefill_q2);
    }
}

/// Prefill-only LA projection overlay. Same `mlx_sys::quantize` path as the
/// decode 2-bit `lm_head`, but gs32 to match the packed QKVZ/BA group size.
pub const PREFILL_LA_Q2_BITS: i32 = 2;
pub const PREFILL_LA_Q2_GROUP_SIZE: i32 = 32;

pub(crate) fn requant_affine_to_prefill_q2(src: &QuantizedWeight) -> Option<QuantizedWeight> {
    let scales = src.scales.as_ref()?;
    if src.bits <= PREFILL_LA_Q2_BITS || src.group_size <= 0 || src.mode != "affine" {
        return None;
    }
    let dense = dequantize(
        &src.weight,
        scales,
        src.biases.as_ref(),
        Some(src.group_size),
        Some(src.bits),
        None,
    );
    let last = *dense.shape().last()?;
    if last <= 0 || last % PREFILL_LA_Q2_GROUP_SIZE != 0 {
        return None;
    }
    let quantized = quantize(
        &dense,
        Some(PREFILL_LA_Q2_GROUP_SIZE),
        Some(PREFILL_LA_Q2_BITS),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    if quantized.len() < 3 {
        return None;
    }
    eval(&[&quantized[0], &quantized[1], &quantized[2]]);
    Some(QuantizedWeight {
        weight: quantized[0].clone(),
        scales: Some(quantized[1].clone()),
        biases: Some(quantized[2].clone()),
        group_size: PREFILL_LA_Q2_GROUP_SIZE,
        bits: PREFILL_LA_Q2_BITS,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q4_weight: None,
        decode_q4_scales: None,
        decode_q4_biases: None,
    })
}

/// Tensors above this size bust MLX's default per-command-buffer byte cap
/// (40–50 MB depending on GPU architecture) on their own.
const BUFFER_CAP_BIG_TENSOR_BYTES: u64 = 48 * 1024 * 1024;
/// Minimum count of cap-busting tensors before the checkpoint is treated as
/// MoE-class for buffer-cap purposes. Dense checkpoints carry ~2 (embedding
/// + lm_head); MoE expert stacks push this to ~90–150.
pub const BUFFER_CAP_MIN_BIG_TENSORS: usize = 16;
const BUFFER_CAP_TARGET_MB: u32 = 1024;
const BUFFER_CAP_TARGET_OPS: u32 = 1000;

/// Whether a model family may use the optimistic MLX command-buffer caps.
///
/// Unlimited-OCR is a compact MXFP8 MoE whose vision-prefill/decode mix does
/// not benefit from the Qwen gather-QMM overlap tuning. On MLX 0.32 / M5 Max,
/// setting both 1024 MB and 1000 ops regresses its end-to-end generation by
/// roughly 26%, so retain MLX's defaults for this family. Explicitly supplied
/// `MLX_MAX_*_PER_BUFFER` variables remain untouched and still take precedence.
///
/// Gemma is excluded for the same reason with direct A/B evidence: the caps
/// were promoted on Qwen3-Coder-Next decode (70.19 vs 54.88 tok/s), but on
/// MLX 0.32.0 / M5 Max the adjacent-commit pair around 6cf02b11 (the commit
/// that made the raise effective on the server path) measures Gemma 4
/// 26B-A4B 4-bit at p2048 prefill 4164 -> 2829 tok/s (-32%) and decode
/// 141.8 -> 128.5 tok/s (-9%): giant command buffers stop `async_eval` from
/// overlapping host graph build with GPU execution on the dual-path
/// dense+MoE layer shape. A pure loss both ways, so Gemma keeps MLX's
/// defaults.
///
/// `qwen3_5` (Qwen3.5/Qwen3.6 hybrids) is excluded on the same mechanism with
/// server-path A/B evidence on Qwen3.6-35B-A3B-6bit-MTP (M3 Max, 2 warmups +
/// 5 reps, 273-token prompt, sampled decode): with the raise, prefill
/// degrades one-way across requests (816 -> 579 tok/s and still falling,
/// warmups 856-890); with MLX defaults it stays flat (937 -> 895, mean
/// +26%). The published 6-bit MTP matrix shows the same signature on M5 Max
/// (35B-A3B prefill 971 -> 517 tok/s from the pre-caps 6.9.0 build to the
/// capped 6.12.1 build) while MTP decode stayed flat (143-145 tok/s), so the
/// decode win the raise was promoted on does not materialize for this family
/// on the server path. `qwen3_next` (Coder-Next) keeps the raise despite a
/// measured ~5-6% prefill cost (interleaved A/B, sampled decode parity): its
/// promotion evidence is the greedy server decode path (+28%), which
/// dominates the coding workload. `glm4_moe_lite` measured parity.
fn auto_buffer_caps_supported_for_family(model_family: &str) -> bool {
    !matches!(
        model_family,
        "unlimited_ocr" | "unlimited-ocr" | "deepseekocr" | "qwen3_5"
    ) && !model_family.contains("gemma")
}

/// Auto-raise MLX's Metal command-buffer caps so `async_eval` keeps overlapping
/// host graph build with GPU execution on MoE-class checkpoints
/// (`docs/performance/gather-qmm-async-serialization.md`).
///
/// **Decides once per process** — MLX reads `MLX_MAX_*_PER_BUFFER` a single
/// time at Metal device init, so later loads cannot change the outcome.
///
/// **Optimistic raise when auto is ON:** for eligible families, caps are raised
/// on the first decision regardless of whether the first checkpoint is
/// MoE-class. Dense-first loads were previously a silent multi-model footgun
/// (Llama then Coder-Next never got the MoE win). Dense impact is measured
/// neutral (Gemma A/B ≈ 0.998); MoE impact is the ship reason (+11–25%).
/// Unlimited-OCR is excluded by measured MLX 0.32 evidence. Pre-set env vars
/// still win.
///
/// Must run before the process's first MLX Metal init.
/// `MlxRunner::from_artifacts_inner` calls this ahead of `set_wired_limit`;
/// `load_weights` covers direct consumers (decode-trace, probes, benches).
pub(crate) fn maybe_raise_metal_buffer_caps(artifacts: &NativeModelArtifacts) {
    static DECIDED: OnceLock<()> = OnceLock::new();
    DECIDED.get_or_init(|| {
        if !crate::fastpath::auto_buffer_caps_enabled() {
            return;
        }
        let model_family = artifacts.manifest().model_family.as_str();
        if !auto_buffer_caps_supported_for_family(model_family) {
            tracing::info!(
                target = "ax_engine_mlx",
                model_family,
                "retained MLX default command-buffer caps for an excluded model family; \
                 explicit MLX_MAX_*_PER_BUFFER values remain authoritative"
            );
            return;
        }
        let big_tensors = artifacts
            .tensor_specs()
            .iter()
            .filter(|spec| spec.length_bytes > BUFFER_CAP_BIG_TENSOR_BYTES)
            .count();
        let is_moe_class = big_tensors >= BUFFER_CAP_MIN_BIG_TENSORS;
        // Always raise for eligible families under auto-ON (see doc above).
        // Telemetry records whether the triggering checkpoint was MoE-class so
        // operators can diagnose multi-model order.
        let (mb_applied, ops_applied) =
            mlx_sys::set_metal_buffer_caps_env(BUFFER_CAP_TARGET_MB, BUFFER_CAP_TARGET_OPS);
        tracing::info!(
            target = "ax_engine_mlx",
            model_family,
            big_tensors,
            is_moe_class,
            mb_applied,
            ops_applied,
            target_mb = BUFFER_CAP_TARGET_MB,
            target_ops = BUFFER_CAP_TARGET_OPS,
            "auto-raised MLX Metal command-buffer caps (optimistic; applies to \
             all subsequent models in this process). AX_MLX_AUTO_BUFFER_CAPS=0 \
             to disable; pre-set MLX_MAX_*_PER_BUFFER wins. First-model-wins: \
             MLX freezes these at Metal device init."
        );
    });
}

fn mmap_weights_env_value_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| !value.is_empty() && value != "0")
}

/// Whether the opt-in memory-mapped safetensors loader is enabled for this process.
pub fn mmap_weights_enabled() -> bool {
    let value = std::env::var("AX_MMAP_WEIGHTS").ok();
    mmap_weights_env_value_enabled(value.as_deref())
}

pub fn load_weights(artifacts: &NativeModelArtifacts) -> Result<ModelWeights, WeightLoadError> {
    maybe_raise_metal_buffer_caps(artifacts);
    let root = artifacts.root_dir().to_path_buf();
    // SSD expert streaming admission (ax_expert_stream.json). A pack marked
    // `required=true` fails closed without --stream-experts /
    // AX_STREAM_EXPERTS=1; requesting streaming without a manifest also
    // fails closed. Default resident loads (certified Qwen 3.6 / GPT-OSS /
    // Gemma paths) are untouched: with no manifest this returns None and
    // streaming stays off.
    let stream_mode = crate::expert_stream::stream_experts_mode();
    let file_manifest = crate::expert_stream::ExpertStreamManifest::read_from_dir(&root)
        .map_err(WeightLoadError::ExpertStream)?;
    let experts_per_tok = artifacts
        .manifest()
        .moe
        .experts_per_token
        .unwrap_or(1)
        .max(1);
    let specs = artifacts.tensor_specs();
    let expert_stream_manifest = crate::expert_stream::resolve_expert_stream(
        stream_mode,
        file_manifest,
        || crate::expert_stream::infer_layer_stack_manifest(specs, experts_per_tok),
        crate::expert_stream::unified_memory_bytes(),
    )
    .map_err(WeightLoadError::ExpertStream)?;
    let expert_stream_skip: Option<std::collections::HashSet<String>> = expert_stream_manifest
        .as_ref()
        .map(crate::expert_stream::streamed_skip_names);
    // AX_MMAP_WEIGHTS=1 uses the memory-mapped safetensors path. No bytes
    // are read into a heap buffer up front; pages are pulled in by the
    // OS on first access (CPU touch or GPU dispatch). On warm page cache
    // this is roughly equivalent to the C loader; on cold disk it lets
    // the kernel decide when to read what, which can roughly halve cold
    // startup time for large models. The default remains the C loader
    // until the mmap path has wider integration test coverage. Note that
    // neither loader is a substitute for expert streaming: both still
    // materialize every tensor in a file, which is why streaming uses the
    // name-filtered loader below.
    let use_mmap = mmap_weights_enabled();
    let mut file_cache: HashMap<PathBuf, HashMap<String, MlxArray>> = HashMap::new();
    for spec in artifacts.tensor_specs() {
        // Streamed tensors never enter the resident map and are never eval'd
        // here; a safetensors file whose specs are all streamed is never
        // opened at init because nothing below requests it.
        if expert_stream_skip
            .as_ref()
            .is_some_and(|skip| skip.contains(&spec.name))
        {
            continue;
        }
        let full = root.join(&spec.file);
        if let Entry::Vacant(entry) = file_cache.entry(full) {
            let path = entry.key().clone();
            let tensors = if let Some(skip) = expert_stream_skip.as_ref() {
                mlx_sys::load_safetensors_filtered(
                    &path,
                    mlx_sys::SafetensorsNameFilter::Exclude(skip),
                )
                .map_err(WeightLoadError::FileMissing)?
            } else if use_mmap {
                mlx_sys::load_safetensors_mmap(&path).map_err(WeightLoadError::FileMissing)?
            } else {
                load_safetensors(&path, None).map_err(WeightLoadError::FileMissing)?
            };
            if tensors.is_empty() {
                return Err(WeightLoadError::FileMissing(path.display().to_string()));
            }
            // Both loaders need an explicit eval here. The C loader path
            // builds in-memory MLX arrays that haven't been routed
            // through the lazy graph yet; the mmap path needs the eval
            // to wire the page-backed buffers into MLX's working set so
            // GPU dispatches see initialised data. (Without it, GPU
            // reads return uninitialised memory → NaN outputs.)
            let refs: Vec<&MlxArray> = tensors.values().collect();
            mlx_sys::eval(&refs);
            entry.insert(tensors);
        }
    }

    // Merge all tensors from all files into one flat map.
    let mut name_map: HashMap<String, MlxArray> = HashMap::new();
    for tensors in file_cache.into_values() {
        name_map.extend(tensors);
    }

    // AXQuant protected vision sidecar: when the quantizer extracted the
    // vision tower into `vision.safetensors`, merge it here (provenance
    // verified against `axquant_vision_sidecar_manifest.json`) so the vision
    // loaders below can find the tensors. Main-file tensors always win over
    // sidecar duplicates.
    load_vision_sidecar(&root, &mut name_map)?;

    let specs = artifacts.tensor_specs();
    let layer_count = artifacts.manifest().layer_count as usize;
    // Family-specific towers need geometry that is intentionally kept in the
    // source config rather than duplicated into the language manifest.
    let source_config = std::fs::read(artifacts.root_dir().join("config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());

    // Raw HuggingFace checkpoints store RMSNorm weights as zero-centered deltas
    // and conv1d weights in a different axis order than MLX expects. The MLX
    // community fork pre-applies these transforms; raw HF checkpoints need them
    // applied here. The manifest's `weight_sanitize` field selects the path.
    //
    // When the manifest leaves `weight_sanitize=None` (the default emitted by
    // `convert_hf_model_dir`), auto-detection samples a block-level norm so raw
    // HF snapshots still load correctly: some mlx-community quantized hybrid
    // models (e.g. `Qwen3-Coder-Next-4bit`) ship unsanitized norm weights
    // because mlx_lm runs `sanitize()` at load time rather than persisting the
    // +1.0 baseline on disk, and raw HF snapshots of dense families (Qwen3,
    // Gemma 4) keep the zero-centered deltas as well. Re-running
    // `mlx_lm.convert` on an already quantized checkpoint dequantizes →
    // normalizes → re-quantizes and produces garbage weights, so
    // auto-detection here is the only viable recovery path.
    let effective_sanitize = effective_weight_sanitize(
        artifacts.manifest().model_family.as_str(),
        artifacts.manifest().weight_sanitize,
        specs,
        &name_map,
    );
    match effective_sanitize {
        WeightSanitize::HfToMlx => apply_hf_sanitize_transforms(specs, &mut name_map, true),
        WeightSanitize::HfNormOnly => apply_hf_sanitize_transforms(specs, &mut name_map, false),
        WeightSanitize::None => {}
    }

    let token_embedding = take_weight(
        specs,
        &mut name_map,
        NativeTensorRole::TokenEmbedding,
        None,
        "token_embedding",
    )?;
    let final_norm = take_weight(
        specs,
        &mut name_map,
        NativeTensorRole::FinalNorm,
        None,
        "final_norm",
    )?
    .weight;
    // Encoder-only families (EmbeddingGemma, Nemotron Embed) have no LM head;
    // reuse the token embedding as a placeholder `lm_head` (never consumed on
    // the embedding-only forward path) so the shared ModelWeights shape stays
    // non-optional.
    let lm_head = if artifacts.manifest().tie_word_embeddings
        || artifacts.manifest().model_family == "embeddinggemma"
        || artifacts.manifest().model_family == "nemotron_embed"
    {
        let mut tied = QuantizedWeight::new(
            token_embedding.weight.clone(),
            token_embedding.scales.clone(),
            token_embedding.biases.clone(),
        );
        tied.group_size = token_embedding.group_size;
        tied.bits = token_embedding.bits;
        tied
    } else {
        take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::LmHead,
            None,
            "lm_head",
        )?
    };

    // Global per-layer input gating weights (Gemma4 2B/4B, optional).
    let per_layer_embed = if has_role(specs, NativeTensorRole::PerLayerEmbedding, None) {
        Some(take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::PerLayerEmbedding,
            None,
            "per_layer_embed",
        )?)
    } else {
        None
    };
    let per_layer_model_proj = if has_role(specs, NativeTensorRole::PerLayerModelProjection, None) {
        Some(take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::PerLayerModelProjection,
            None,
            "per_layer_model_proj",
        )?)
    } else {
        None
    };
    let per_layer_proj_norm = if has_role(specs, NativeTensorRole::PerLayerProjectionNorm, None) {
        let w = take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::PerLayerProjectionNorm,
            None,
            "per_layer_proj_norm",
        )?;
        Some(w.weight)
    } else {
        None
    };
    let assistant_pre_projection =
        if has_role(specs, NativeTensorRole::AssistantPreProjection, None) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::AssistantPreProjection,
                None,
                "assistant_pre_projection",
            )?)
        } else {
            None
        };
    let assistant_post_projection =
        if has_role(specs, NativeTensorRole::AssistantPostProjection, None) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::AssistantPostProjection,
                None,
                "assistant_post_projection",
            )?)
        } else {
            None
        };
    let embedding_dense_0 = if has_role(specs, NativeTensorRole::EmbeddingDense0, None) {
        Some(take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::EmbeddingDense0,
            None,
            "embedding_dense_0",
        )?)
    } else {
        None
    };
    let embedding_dense_1 = if has_role(specs, NativeTensorRole::EmbeddingDense1, None) {
        Some(take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::EmbeddingDense1,
            None,
            "embedding_dense_1",
        )?)
    } else {
        None
    };
    let gemma4_unified_vision = load_gemma4_unified_vision_weights(specs, &mut name_map)?;
    let gemma4_unified_audio = load_gemma4_unified_audio_weights(specs, &mut name_map)?;
    let gemma4_vl_vision = crate::gemma4_vl::load_gemma4_vl_vision_weights(
        specs,
        &mut name_map,
        source_config.as_ref(),
    )?;
    let diffusion_self_conditioning =
        load_diffusion_self_conditioning_weights(specs, &mut name_map)?;
    // Unlimited-OCR: projector roles + leftover sam_model.*/vision_model.* keys.
    // Load before layer loop so language tensors are still present for layers,
    // but vision keys are independent and safe to consume early.
    let unlimited_ocr_vision =
        crate::unlimited_ocr::load_unlimited_ocr_vision_weights(specs, &mut name_map)?;
    // Qwen3-VL vision tower (WS-V2): roles + visual.* leftovers → Some when present.
    let qwen3_vl_vision = crate::qwen3_vl::load_qwen3_vl_vision_weights(
        specs,
        &mut name_map,
        source_config.as_ref(),
    )?;
    let minicpm_v46_vision =
        crate::minicpm_v::load_minicpm_v46_vision_weights(&mut name_map, source_config.as_ref())?;
    let nemotron_omni =
        crate::nemotron_omni::load_nemotron_omni_weights(&mut name_map, source_config.as_ref())?;

    let mut layers = Vec::with_capacity(layer_count);
    // DeepSeek V4 layers carry attention/hyper-connection tensors that must
    // not be routed through the standard/GLM-MLA projection loaders.
    let is_deepseek_v4 = artifacts.manifest().deepseek_v4.is_enabled();
    // GGUF-layout manifests place the nextn (MTP) block's per-layer tensors at
    // layer index `layer_count`; load them through the same per-layer path and
    // detach the extra layer into `DeepseekV4NextnWeights` below.
    let nextn_block_in_manifest =
        is_deepseek_v4 && has_role(specs, NativeTensorRole::HcAttnFn, Some(layer_count as u32));
    let load_layer_total = layer_count + usize::from(nextn_block_in_manifest);
    for li in 0..load_layer_total {
        let idx = Some(li as u32);
        let uses_shared_kv = artifacts.manifest().model_family == "gemma4_assistant"
            || artifacts
                .manifest()
                .kv_shared_source_layers
                .contains_key(&(li as u32));
        let uses_value_from_key = artifacts
            .manifest()
            .attention_value_from_key_layers
            .contains(&(li as u32));
        let attention_layout = attention_layout_for_layer(specs, idx)?;
        let is_nemotron_h = artifacts.manifest().model_family == "nemotron_h";

        let attn_norm = take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::AttentionNorm,
            idx,
            "attn_norm",
        )?
        .weight;
        let o_proj = match attention_layout {
            // V4 uses the grouped wo_a/wo_b output LoRA, loaded below.
            AttentionLayout::Full if !is_deepseek_v4 => Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::AttentionO,
                idx,
                "o_proj",
            )?),
            AttentionLayout::Full | AttentionLayout::Linear | AttentionLayout::None => None,
        };
        let linear_attn = match attention_layout {
            AttentionLayout::Full | AttentionLayout::None => None,
            AttentionLayout::Linear => Some(load_linear_attention_weights(
                specs,
                &mut name_map,
                idx,
                &artifacts.manifest().linear_attention,
            )?),
        };
        // Nemotron-H has a single pre-mixer norm; reuse it as ffn_norm placeholder.
        let (attn_post_norm, ffn_norm) = if is_nemotron_h {
            (None, attn_norm.clone())
        } else {
            take_layer_norms(specs, &mut name_map, idx)?
        };
        let down_proj = if has_role(specs, NativeTensorRole::FfnDown, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnDown,
                idx,
                "down_proj",
            )?)
        } else {
            None
        };

        let ffn_post_norm =
            try_take_plain(specs, &mut name_map, NativeTensorRole::FfnPostNorm, idx)?;
        let ffn_norm2 = try_take_plain(specs, &mut name_map, NativeTensorRole::FfnNorm2, idx)?;
        let ffn_post_norm1 =
            try_take_plain(specs, &mut name_map, NativeTensorRole::FfnPostNorm1, idx)?;
        let ffn_post_norm2 =
            try_take_plain(specs, &mut name_map, NativeTensorRole::FfnPostNorm2, idx)?;

        let router_proj = if has_role(specs, NativeTensorRole::FfnGateInp, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnGateInp,
                idx,
                "router_proj",
            )?)
        } else {
            None
        };
        let router_scale =
            try_take_plain(specs, &mut name_map, NativeTensorRole::FfnGateInpScale, idx)?;
        // Gemma4 MoE router scale applies to the Gemma4 backbone including
        // DiffusionGemma (same architecture, BlockDiffusion generation — ADR-038)
        // and gemma4_vl (encoder-VL packaging of the same text tower).
        let router_combined_scale = if matches!(
            artifacts.manifest().model_family.as_str(),
            "gemma4" | "gemma4_vl" | "gemma4_assistant" | "diffusion_gemma"
        ) || artifacts.manifest().generation_kind()
            == ax_engine_core::GenerationKind::BlockDiffusion
        {
            router_scale
                .as_ref()
                .map(|scale| gemma4_router_combined_scale(artifacts.manifest().hidden_size, scale))
        } else {
            None
        };
        let router_expert_scale = try_take_plain(
            specs,
            &mut name_map,
            NativeTensorRole::FfnGateInpExpertScale,
            idx,
        )?;
        let layer_scalar =
            try_take_plain(specs, &mut name_map, NativeTensorRole::LayerScalar, idx)?;
        let per_layer_gate = if has_role(specs, NativeTensorRole::PerLayerInputGate, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::PerLayerInputGate,
                idx,
                "per_layer_gate",
            )?)
        } else {
            None
        };
        let per_layer_proj_w = if has_role(specs, NativeTensorRole::PerLayerInputProjection, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::PerLayerInputProjection,
                idx,
                "per_layer_proj_w",
            )?)
        } else {
            None
        };
        let per_layer_post_norm = try_take_plain(
            specs,
            &mut name_map,
            NativeTensorRole::PerLayerInputPostNorm,
            idx,
        )?;

        let shared_expert_gate = if has_role(specs, NativeTensorRole::FfnSharedExpertGateInp, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnSharedExpertGateInp,
                idx,
                "shared_expert_gate",
            )?)
        } else {
            None
        };
        let shared_gate_proj = if has_role(specs, NativeTensorRole::FfnSharedExpertGate, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnSharedExpertGate,
                idx,
                "shared_gate_proj",
            )?)
        } else {
            None
        };
        let shared_up_proj = if has_role(specs, NativeTensorRole::FfnSharedExpertUp, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnSharedExpertUp,
                idx,
                "shared_up_proj",
            )?)
        } else {
            None
        };
        let shared_down_proj = if has_role(specs, NativeTensorRole::FfnSharedExpertDown, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnSharedExpertDown,
                idx,
                "shared_down_proj",
            )?)
        } else {
            None
        };
        let (shared_gate_up_proj, shared_gate_proj, shared_up_proj) =
            match (shared_gate_proj, shared_up_proj) {
                // Keep shared experts on split gate/up projections. Qwen3.6
                // A3B's shared-expert packed path diverges from mlx_lm output,
                // while the split path is token-exact against the reference.
                (Some(gate), Some(up)) => (None, Some(gate), Some(up)),
                (gate, up) => (None, gate, up),
            };

        // Expert streaming: streamed tensor names are absent from the
        // resident name map, so their expert slots stay None here until the
        // MoE forward pages the layer stack in via its ExpertLayerSource.
        let expert_streamed_role = |role: NativeTensorRole| {
            expert_stream_skip.as_ref().is_some_and(|skip| {
                specs
                    .iter()
                    .any(|s| s.role == role && s.layer_index == idx && skip.contains(&s.name))
            })
        };
        let gate_up_exps_packed = if has_role(specs, NativeTensorRole::FfnGateUpExpsPacked, idx)
            && !expert_streamed_role(NativeTensorRole::FfnGateUpExpsPacked)
        {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnGateUpExpsPacked,
                idx,
                "gate_up_exps",
            )?)
        } else {
            None
        };
        let gate_exps = if has_role(specs, NativeTensorRole::FfnGateExps, idx)
            && !expert_streamed_role(NativeTensorRole::FfnGateExps)
        {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnGateExps,
                idx,
                "gate_exps",
            )?)
        } else {
            None
        };
        let up_exps = if has_role(specs, NativeTensorRole::FfnUpExps, idx)
            && !expert_streamed_role(NativeTensorRole::FfnUpExps)
        {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnUpExps,
                idx,
                "up_exps",
            )?)
        } else {
            None
        };
        let down_exps = if has_role(specs, NativeTensorRole::FfnDownExps, idx)
            && !expert_streamed_role(NativeTensorRole::FfnDownExps)
        {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnDownExps,
                idx,
                "down_exps",
            )?)
        } else {
            None
        };

        // GPT-OSS openai native MXFP4: fused gate_up_proj_blocks → de-interleave
        // into split gate/up experts (matches mlx-lm gpt_oss.Model.sanitize).
        // mlx-community product checkpoints already ship split gate/up/down and
        // are loaded above via FfnGateExps / FfnUpExps / FfnDownExps.
        let (gate_exps, up_exps, down_exps, mxfp4_gate_up_exps, mxfp4_down_exps) = if gate_exps
            .is_none()
            && has_role(specs, NativeTensorRole::FfnGateUpExpsMxfp4Blocks, idx)
        {
            let (gate, up, down) =
                load_gpt_oss_openai_mxfp4_split_experts(specs, &mut name_map, idx)?;
            (Some(gate), Some(up), Some(down), None, None)
        } else {
            (gate_exps, up_exps, down_exps, None, None)
        };

        // GPT-OSS per-head attention sink. V4 owns its sink inside
        // `DeepseekV4LayerWeights` (loaded below), so skip the generic slot here.
        let attn_sink = if is_deepseek_v4 {
            None
        } else {
            try_take_plain(specs, &mut name_map, NativeTensorRole::AttnSink, idx)?
        };

        let q_norm = try_take_plain(specs, &mut name_map, NativeTensorRole::AttentionQNorm, idx)?;
        let k_norm = try_take_plain(specs, &mut name_map, NativeTensorRole::AttentionKNorm, idx)?;

        let (qkv_packed, q_proj, k_proj, v_proj, glm_mla_attn) = if matches!(
            attention_layout,
            AttentionLayout::Linear | AttentionLayout::None
        ) || is_deepseek_v4
        {
            // V4 projections load into `DeepseekV4LayerWeights` below; never
            // through the standard/GLM-MLA layout detection (its Qa/QaNorm/Qb
            // roles would otherwise misdetect as GLM MLA and hit the V3-only
            // `split_deepseek_kv_b_projection`).
            (None, None, None, None, None)
        } else {
            match full_attention_projection_layout(specs, idx, uses_shared_kv, uses_value_from_key)?
            {
                FullAttentionProjectionLayout::GlmMla => {
                    let glm = load_glm_mla_attention_weights(
                        specs,
                        &mut name_map,
                        idx,
                        &artifacts.manifest().mla_attention,
                        artifacts.manifest().attention_head_count,
                    )?;
                    (None, None, None, None, Some(glm))
                }
                FullAttentionProjectionLayout::QOnly => {
                    let q = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionQ,
                        idx,
                        "q_proj",
                    )?;
                    (None, Some(q), None, None, None)
                }
                FullAttentionProjectionLayout::PackedQkv => {
                    let p = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionQkvPacked,
                        idx,
                        "qkv",
                    )?;
                    (Some(p), None, None, None, None)
                }
                FullAttentionProjectionLayout::SplitQkValueFromKey => {
                    let q = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionQ,
                        idx,
                        "q_proj",
                    )?;
                    let k = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionK,
                        idx,
                        "k_proj",
                    )?;
                    (None, Some(q), Some(k), None, None)
                }
                FullAttentionProjectionLayout::SplitQkv => {
                    let q = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionQ,
                        idx,
                        "q_proj",
                    )?;
                    let k = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionK,
                        idx,
                        "k_proj",
                    )?;
                    let v = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionV,
                        idx,
                        "v_proj",
                    )?;
                    // W6 (mlx-lm-prefill-parity PRD §7): materialize a packed
                    // QKV at load time so single-request runtime paths dispatch
                    // one quantized matmul + last-dim slice instead of three
                    // separate matmuls. Keep the split weights too: batched
                    // embedding workloads can select the mlx-lm-shaped split
                    // projections, which are faster for B > 1 on Qwen3
                    // embedding models.
                    if dense_attention_qkv_packing_enabled() {
                        if let Ok(qk) = concat_quantized_weight_rows(&q, &k)
                            && let Ok(qkv) = concat_quantized_weight_rows(&qk, &v)
                        {
                            eval_packed_projection(&qkv);
                            (Some(qkv), Some(q), Some(k), Some(v), None)
                        } else {
                            (None, Some(q), Some(k), Some(v), None)
                        }
                    } else {
                        (None, Some(q), Some(k), Some(v), None)
                    }
                }
            }
        };

        let deepseek_v4 = if is_deepseek_v4 {
            Some(load_deepseek_v4_layer_weights(specs, &mut name_map, idx)?)
        } else {
            None
        };

        let (gate_up_packed, gate_proj, up_proj) =
            if has_role(specs, NativeTensorRole::FfnGateUpPacked, idx) {
                let p = take_weight(
                    specs,
                    &mut name_map,
                    NativeTensorRole::FfnGateUpPacked,
                    idx,
                    "gate_up",
                )?;
                (Some(p), None, None)
            } else if has_role(specs, NativeTensorRole::FfnGate, idx) {
                let g = take_weight(
                    specs,
                    &mut name_map,
                    NativeTensorRole::FfnGate,
                    idx,
                    "gate_proj",
                )?;
                let u = take_weight(
                    specs,
                    &mut name_map,
                    NativeTensorRole::FfnUp,
                    idx,
                    "up_proj",
                )?;
                if dense_ffn_gate_up_packing_enabled()
                    && dense_ffn_gate_up_packing_supported(
                        artifacts.manifest().model_family.as_str(),
                        &g,
                        &u,
                    )
                {
                    let packed = pack_dense_ffn_gate_up_projection(&g, &u)?;
                    (Some(packed), Some(g), Some(u))
                } else {
                    (None, Some(g), Some(u))
                }
            } else {
                (None, None, None)
            };

        layers.push(LayerWeights {
            attn_norm,
            attn_post_norm,
            q_norm,
            k_norm,
            q_proj,
            k_proj,
            v_proj,
            qkv_packed,
            o_proj,
            linear_attn,
            glm_mla_attn,
            deepseek_v4,
            ffn_norm,
            ffn_post_norm,
            gate_proj,
            up_proj,
            gate_up_packed,
            down_proj,
            ffn_norm2,
            ffn_post_norm1,
            ffn_post_norm2,
            router_proj,
            router_correction_bias: try_take_plain(
                specs,
                &mut name_map,
                NativeTensorRole::FfnGateInpCorrectionBias,
                idx,
            )?,
            router_scale,
            router_combined_scale,
            router_expert_scale,
            layer_scalar,
            per_layer_gate,
            per_layer_proj_w,
            per_layer_post_norm,
            shared_expert_gate,
            shared_gate_up_proj,
            shared_gate_proj,
            shared_up_proj,
            shared_down_proj,
            gate_up_exps_packed,
            gate_exps,
            up_exps,
            down_exps,
            mxfp4_gate_up_exps,
            mxfp4_down_exps,
            attn_sink,
            rotation_smoothing_inverse: None,
            expert_stream: None,
        });
    }

    // Expert streaming: attach the pager to layers whose fused expert stacks
    // are streamed. Their resident expert fields stay None; the MoE forward
    // resolves them through the handle, which pages the layer stack in.
    let expert_stream_pager = expert_stream_manifest.as_ref().map(|manifest| {
        let pager = std::sync::Arc::new(crate::expert_stream::ExpertStackPager::new(
            std::sync::Arc::new(manifest.clone()),
            root.clone(),
            crate::expert_stream::expert_layer_budget(),
        ));
        let streamed_layers: std::collections::HashSet<u32> =
            manifest.layer_indices().into_iter().collect();
        for (li, layer) in layers.iter_mut().enumerate() {
            if streamed_layers.contains(&(li as u32)) {
                layer.expert_stream = Some(std::sync::Arc::new(
                    crate::expert_stream::ExpertLayerSource::new(pager.clone(), li as u32),
                ));
            }
        }
        tracing::info!(
            target = "ax_engine_mlx",
            layers = streamed_layers.len(),
            budget = pager.budget_layers(),
            required = manifest.required,
            "expert streaming active: layer-stack paging replaces resident expert loads"
        );
        pager
    });

    // Conv1d remains the reliable fail-closed check for raw HF linear-attention
    // layout. LinearAttentionNorm is a gated-norm scale in Qwen3-Next-style
    // models and can legitimately be near zero, so do not require a +1.0
    // baseline for that tensor.
    for (idx, layer) in layers.iter().enumerate() {
        if let Some(la) = layer.linear_attn.as_ref() {
            ensure_conv1d_mlx_layout(idx, &la.conv1d_dense)?;
        }
    }

    crate::weight_rotation::shadow_log_rotation_candidates(specs);

    // Root-level DeepSeek V4 hyper-connection head and deferred MTP (nextn)
    // tensors; both are global (non-layer-indexed) roles. Detach the nextn
    // block layer first so `layers` keeps exactly `layer_count` entries.
    let nextn_block_layer = if nextn_block_in_manifest {
        Some(
            layers
                .pop()
                .expect("nextn block layer loaded with the extended layer loop"),
        )
    } else {
        None
    };
    let deepseek_v4_head = load_deepseek_v4_head_weights(specs, &mut name_map)?;
    let mut deepseek_v4_nextn =
        load_deepseek_v4_nextn_weights(specs, &mut name_map, nextn_block_layer)?;
    // Raw-HF packages ship the nextn block in an `mtp.safetensors` sidecar;
    // fill any missing piece from it (manifest-side tensors win).
    if artifacts.manifest().deepseek_v4.is_enabled() {
        let nextn_incomplete = deepseek_v4_nextn.as_ref().is_none_or(|n| {
            n.layer.is_none()
                || n.enorm.is_none()
                || n.hnorm.is_none()
                || (n.eh_proj.is_none() && (n.e_proj.is_none() || n.h_proj.is_none()))
        });
        if nextn_incomplete
            && let Some(sidecar) =
                load_deepseek_v4_mtp_sidecar(&root, &mut name_map, artifacts.manifest())
        {
            deepseek_v4_nextn = Some(match deepseek_v4_nextn {
                Some(base) => base.merged_with(sidecar),
                None => sidecar,
            });
        }
    }

    // Load MTP sidecar if present (e.g. `mtp.safetensors` alongside the main files).
    let (
        mtp_max_depth,
        mtp_draft_sampling,
        mtp_sidecar_bits,
        mtp_draft_lm_head_spec,
        mtp_norm_layout,
    ) = load_mtp_sidecar(&root, &mut name_map, artifacts.manifest());
    let mtp = load_mtp(
        &mut name_map,
        &lm_head,
        mtp_max_depth,
        mtp_draft_sampling,
        mtp_sidecar_bits,
        mtp_draft_lm_head_spec,
        mtp_norm_layout,
    );
    let gemma4_assistant_mtp = load_gemma4_assistant_mtp_status(&root, artifacts.manifest());
    let glm_mtp = load_glm_mtp_sidecar(&root, &mut name_map, artifacts.manifest());

    let mut lm_head = lm_head;
    lm_head.prepare_decode_q4_lm_head();
    lm_head.prepare_contiguous_decode_weight_t();
    let mut model = ModelWeights {
        token_embedding,
        final_norm,
        lm_head,
        layers,
        per_layer_embed,
        per_layer_model_proj,
        per_layer_proj_norm,
        mtp,
        gemma4_assistant_mtp,
        assistant_pre_projection,
        assistant_post_projection,
        embedding_dense_0,
        embedding_dense_1,
        gemma4_unified_vision,
        gemma4_unified_audio,
        gemma4_vl_vision,
        diffusion_self_conditioning,
        glm_mtp,
        deepseek_v4_head,
        deepseek_v4_nextn,
        unlimited_ocr_vision,
        qwen3_vl_vision,
        minicpm_v46_vision,
        nemotron_omni,
        expert_stream: expert_stream_pager,
    };

    apply_rotated_checkpoint(&mut model, artifacts)?;

    Ok(model)
}

/// Load only the safetensor files and tensors required by one dense Llama 3
/// pipeline stage.
///
/// Selection happens before `load_safetensors`, avoiding whole-model peak
/// memory. Safetensor files remain the smallest independently loadable unit:
/// if a checkpoint file contains tensors for adjacent stages, both ranks may
/// map that boundary file, but neither rank opens unrelated files.
pub fn load_pipeline_stage_weights(
    artifacts: &NativeModelArtifacts,
    assignment: PipelineRankAssignment,
) -> Result<PipelineStageWeights, WeightLoadError> {
    let manifest = artifacts.manifest();
    if manifest.model_family != "llama3" {
        return Err(WeightLoadError::UnsupportedPipelineFamily(
            manifest.model_family.clone(),
        ));
    }
    let range = assignment.layers;
    if range.start >= range.end || range.end > manifest.layer_count {
        return Err(WeightLoadError::InvalidPipelineAssignment(format!(
            "rank {} layer range [{}, {}) is outside model layer_count {}",
            assignment.rank, range.start, range.end, manifest.layer_count
        )));
    }
    if assignment.owns_embeddings != (range.start == 0)
        || assignment.owns_output_head != (range.end == manifest.layer_count)
    {
        return Err(WeightLoadError::InvalidPipelineAssignment(format!(
            "rank {} endpoint ownership does not match layer range [{}, {})",
            assignment.rank, range.start, range.end
        )));
    }
    if manifest.moe.is_enabled()
        || manifest.linear_attention.is_enabled()
        || manifest.mla_attention.is_enabled()
    {
        return Err(WeightLoadError::UnsupportedPipelineFamily(
            "llama3 pipeline requires dense full-attention weights".into(),
        ));
    }

    maybe_raise_metal_buffer_caps(artifacts);
    let specs = artifacts.tensor_specs();
    let tied_output_embedding = assignment.owns_output_head && manifest.tie_word_embeddings;
    let selected_files = pipeline_stage_required_files(artifacts, &assignment);
    if selected_files.is_empty() {
        return Err(WeightLoadError::InvalidPipelineAssignment(
            "stage selected no weight files".into(),
        ));
    }

    let root = artifacts.root_dir();
    let use_mmap = mmap_weights_enabled();
    let mut name_map = HashMap::<String, MlxArray>::new();
    for relative in selected_files {
        let path = root.join(&relative);
        let tensors = if use_mmap {
            mlx_sys::load_safetensors_mmap(&path).map_err(WeightLoadError::FileMissing)?
        } else {
            load_safetensors(&path, None).map_err(WeightLoadError::FileMissing)?
        };
        if tensors.is_empty() {
            return Err(WeightLoadError::FileMissing(path.display().to_string()));
        }
        let refs = tensors.values().collect::<Vec<_>>();
        eval(&refs);
        name_map.extend(tensors);
    }

    let effective_sanitize = effective_weight_sanitize(
        manifest.model_family.as_str(),
        manifest.weight_sanitize,
        specs,
        &name_map,
    );
    match effective_sanitize {
        WeightSanitize::HfToMlx => apply_hf_sanitize_transforms(specs, &mut name_map, true),
        WeightSanitize::HfNormOnly => apply_hf_sanitize_transforms(specs, &mut name_map, false),
        WeightSanitize::None => {}
    }

    let token_embedding = (assignment.owns_embeddings || tied_output_embedding)
        .then(|| {
            take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::TokenEmbedding,
                None,
                "token_embedding",
            )
        })
        .transpose()?;
    let final_norm = assignment
        .owns_output_head
        .then(|| {
            take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FinalNorm,
                None,
                "final_norm",
            )
            .map(|weight| weight.weight)
        })
        .transpose()?;
    let lm_head = if assignment.owns_output_head {
        if manifest.tie_word_embeddings {
            let embedding = token_embedding.as_ref().ok_or_else(|| {
                WeightLoadError::InvalidPipelineAssignment(
                    "tied output head requires token embedding".into(),
                )
            })?;
            let mut tied = QuantizedWeight::new(
                embedding.weight.clone(),
                embedding.scales.clone(),
                embedding.biases.clone(),
            );
            tied.group_size = embedding.group_size;
            tied.bits = embedding.bits;
            tied.mode.clone_from(&embedding.mode);
            tied.prepare_decode_q4_lm_head();
            tied.prepare_contiguous_decode_weight_t();
            Some(tied)
        } else {
            let mut head = take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::LmHead,
                None,
                "lm_head",
            )?;
            head.prepare_decode_q4_lm_head();
            head.prepare_contiguous_decode_weight_t();
            Some(head)
        }
    } else {
        None
    };

    let mut layers = Vec::with_capacity(range.len() as usize);
    for layer_index in range.start..range.end {
        layers.push(load_dense_llama3_layer(specs, &mut name_map, layer_index)?);
    }

    let stage_token_embedding = assignment
        .owns_embeddings
        .then_some(token_embedding)
        .flatten();
    Ok(PipelineStageWeights {
        assignment,
        token_embedding: stage_token_embedding,
        final_norm,
        lm_head,
        layers,
    })
}

/// Return the smallest set of checkpoint files that the assigned stage will
/// open. Callers can use this before loading to enforce an artifact allowlist.
pub fn pipeline_stage_required_files(
    artifacts: &NativeModelArtifacts,
    assignment: &PipelineRankAssignment,
) -> std::collections::BTreeSet<PathBuf> {
    let tied_output_embedding =
        assignment.owns_output_head && artifacts.manifest().tie_word_embeddings;
    pipeline_stage_files(artifacts.tensor_specs(), assignment, tied_output_embedding)
}

fn pipeline_stage_files(
    specs: &[NativeTensorSpec],
    assignment: &PipelineRankAssignment,
    tied_output_embedding: bool,
) -> std::collections::BTreeSet<PathBuf> {
    specs
        .iter()
        .filter(|spec| {
            spec.layer_index
                .is_some_and(|layer| assignment.layers.contains(layer))
                || (assignment.owns_embeddings && spec.role == NativeTensorRole::TokenEmbedding)
                || (assignment.owns_output_head
                    && matches!(
                        spec.role,
                        NativeTensorRole::FinalNorm | NativeTensorRole::LmHead
                    ))
                || (tied_output_embedding && spec.role == NativeTensorRole::TokenEmbedding)
        })
        .map(|spec| spec.file.clone())
        .collect()
}

fn load_dense_llama3_layer(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer_index: u32,
) -> Result<LayerWeights, WeightLoadError> {
    let idx = Some(layer_index);
    let attn_norm = take_weight(
        specs,
        name_map,
        NativeTensorRole::AttentionNorm,
        idx,
        "attn_norm",
    )?
    .weight;
    let (attn_post_norm, ffn_norm) = take_layer_norms(specs, name_map, idx)?;
    let q_norm = try_take_plain(specs, name_map, NativeTensorRole::AttentionQNorm, idx)?;
    let k_norm = try_take_plain(specs, name_map, NativeTensorRole::AttentionKNorm, idx)?;
    let q_proj = take_weight(specs, name_map, NativeTensorRole::AttentionQ, idx, "q_proj")?;
    let k_proj = take_weight(specs, name_map, NativeTensorRole::AttentionK, idx, "k_proj")?;
    let v_proj = take_weight(specs, name_map, NativeTensorRole::AttentionV, idx, "v_proj")?;
    let o_proj = take_weight(specs, name_map, NativeTensorRole::AttentionO, idx, "o_proj")?;
    let gate_proj = take_weight(specs, name_map, NativeTensorRole::FfnGate, idx, "gate_proj")?;
    let up_proj = take_weight(specs, name_map, NativeTensorRole::FfnUp, idx, "up_proj")?;
    let down_proj = take_weight(specs, name_map, NativeTensorRole::FfnDown, idx, "down_proj")?;

    Ok(LayerWeights {
        attn_norm,
        attn_post_norm,
        q_norm,
        k_norm,
        q_proj: Some(q_proj),
        k_proj: Some(k_proj),
        v_proj: Some(v_proj),
        qkv_packed: None,
        o_proj: Some(o_proj),
        linear_attn: None,
        glm_mla_attn: None,
        deepseek_v4: None,
        ffn_norm,
        ffn_post_norm: None,
        gate_proj: Some(gate_proj),
        up_proj: Some(up_proj),
        gate_up_packed: None,
        down_proj: Some(down_proj),
        ffn_norm2: None,
        ffn_post_norm1: None,
        ffn_post_norm2: None,
        router_proj: None,
        router_correction_bias: None,
        router_scale: None,
        router_combined_scale: None,
        router_expert_scale: None,
        layer_scalar: None,
        per_layer_gate: None,
        per_layer_proj_w: None,
        per_layer_post_norm: None,
        shared_expert_gate: None,
        shared_gate_up_proj: None,
        shared_gate_proj: None,
        shared_up_proj: None,
        shared_down_proj: None,
        gate_up_exps_packed: None,
        gate_exps: None,
        up_exps: None,
        down_exps: None,
        mxfp4_gate_up_exps: None,
        mxfp4_down_exps: None,
        attn_sink: None,
        rotation_smoothing_inverse: None,
        expert_stream: None,
    })
}

fn load_diffusion_self_conditioning_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
) -> Result<Option<DiffusionSelfConditioningWeights>, WeightLoadError> {
    if !has_role(specs, NativeTensorRole::DiffusionSelfConditionPreNorm, None) {
        return Ok(None);
    }

    Ok(Some(DiffusionSelfConditioningWeights {
        pre_norm: take_plain_required(
            specs,
            name_map,
            NativeTensorRole::DiffusionSelfConditionPreNorm,
            None,
            "diffusion_self_conditioning.pre_norm",
        )?,
        gate_proj: take_weight(
            specs,
            name_map,
            NativeTensorRole::DiffusionSelfConditionGate,
            None,
            "diffusion_self_conditioning.gate_proj",
        )?,
        up_proj: take_weight(
            specs,
            name_map,
            NativeTensorRole::DiffusionSelfConditionUp,
            None,
            "diffusion_self_conditioning.up_proj",
        )?,
        down_proj: take_weight(
            specs,
            name_map,
            NativeTensorRole::DiffusionSelfConditionDown,
            None,
            "diffusion_self_conditioning.down_proj",
        )?,
    }))
}

fn load_gemma4_unified_vision_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
) -> Result<Option<Gemma4UnifiedVisionWeights>, WeightLoadError> {
    if !has_role(specs, NativeTensorRole::Gemma4UnifiedVisionPatchDense, None)
        && !has_role(specs, NativeTensorRole::Gemma4UnifiedVisionProjection, None)
    {
        return Ok(None);
    }

    let mut patch_dense = take_weight(
        specs,
        name_map,
        NativeTensorRole::Gemma4UnifiedVisionPatchDense,
        None,
        "gemma4_unified.patch_dense",
    )?;
    // `take_weight` recognizes the conventional `.bias` sibling and consumes
    // it as a dense linear bias. The unified vision path applies that bias
    // explicitly after its patch projection, so move it back out here to
    // avoid both a missing-role error and adding it twice in `qw`.
    let patch_dense_bias = match patch_dense.linear_bias.take() {
        Some(bias) => bias,
        None => take_plain_required(
            specs,
            name_map,
            NativeTensorRole::Gemma4UnifiedVisionPatchDenseBias,
            None,
            "gemma4_unified.patch_dense.bias",
        )?,
    };

    Ok(Some(Gemma4UnifiedVisionWeights {
        patch_ln1_weight: take_plain_required(
            specs,
            name_map,
            NativeTensorRole::Gemma4UnifiedVisionPatchNorm1,
            None,
            "gemma4_unified.patch_ln1.weight",
        )?,
        patch_ln1_bias: take_plain_required(
            specs,
            name_map,
            NativeTensorRole::Gemma4UnifiedVisionPatchNorm1Bias,
            None,
            "gemma4_unified.patch_ln1.bias",
        )?,
        patch_dense,
        patch_dense_bias,
        patch_ln2_weight: take_plain_required(
            specs,
            name_map,
            NativeTensorRole::Gemma4UnifiedVisionPatchNorm2,
            None,
            "gemma4_unified.patch_ln2.weight",
        )?,
        patch_ln2_bias: take_plain_required(
            specs,
            name_map,
            NativeTensorRole::Gemma4UnifiedVisionPatchNorm2Bias,
            None,
            "gemma4_unified.patch_ln2.bias",
        )?,
        pos_embedding: take_plain_required(
            specs,
            name_map,
            NativeTensorRole::Gemma4UnifiedVisionPositionEmbedding,
            None,
            "gemma4_unified.pos_embedding",
        )?,
        pos_norm_weight: take_plain_required(
            specs,
            name_map,
            NativeTensorRole::Gemma4UnifiedVisionPositionNorm,
            None,
            "gemma4_unified.pos_norm.weight",
        )?,
        pos_norm_bias: take_plain_required(
            specs,
            name_map,
            NativeTensorRole::Gemma4UnifiedVisionPositionNormBias,
            None,
            "gemma4_unified.pos_norm.bias",
        )?,
        projection: take_weight(
            specs,
            name_map,
            NativeTensorRole::Gemma4UnifiedVisionProjection,
            None,
            "gemma4_unified.embed_vision.embedding_projection",
        )?,
    }))
}

fn load_gemma4_unified_audio_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
) -> Result<Option<Gemma4UnifiedAudioWeights>, WeightLoadError> {
    if !has_role(specs, NativeTensorRole::Gemma4UnifiedAudioProjection, None) {
        return Ok(None);
    }

    Ok(Some(Gemma4UnifiedAudioWeights {
        projection: take_weight(
            specs,
            name_map,
            NativeTensorRole::Gemma4UnifiedAudioProjection,
            None,
            "gemma4_unified.embed_audio.embedding_projection",
        )?,
    }))
}

fn take_plain_required(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    role: NativeTensorRole,
    layer_index: Option<u32>,
    label: &str,
) -> Result<MlxArray, WeightLoadError> {
    try_take_plain(specs, name_map, role, layer_index)?
        .ok_or_else(|| WeightLoadError::RoleMissing(format!("{label}[{layer_index:?}]")))
}

/// Take a plain (non-quantized) tensor from `name_map` by exact key.
/// Returns `None` if the key is absent (does not require the key to be present).
fn mtp_take_plain(name_map: &mut HashMap<String, MlxArray>, key: &str) -> Option<MlxArray> {
    name_map.remove(key)
}

/// Take a (possibly quantized) weight from `name_map` by base key.
/// Looks for `{key}.weight` (or `{key}` directly) plus optional `.scales` / `.biases`.
/// Returns `None` if the base weight is absent.
///
/// `bits_hint`: when `Some(b)`, use `b` as the quantization bits (overrides inference).
/// Pass `None` to infer; inferred bits default to 4 when ambiguous.
fn mtp_take_weight(
    name_map: &mut HashMap<String, MlxArray>,
    base: &str,
    bits_hint: Option<i32>,
) -> Option<QuantizedWeight> {
    let weight_key = format!("{base}.weight");
    let weight = name_map
        .remove(&weight_key)
        .or_else(|| name_map.remove(base))?;
    let scales = name_map.remove(&format!("{base}.scales"));
    let biases = name_map.remove(&format!("{base}.biases"));
    let (group_size, bits) = if let Some(ref s) = scales {
        let w_shape = weight.shape();
        let s_shape = s.shape();
        let bits = bits_hint.unwrap_or(4);
        let gs = if w_shape.len() == 2 && s_shape.len() == 2 && s_shape[1] > 0 {
            // packed_cols = ceil(cols * bits / 32), so cols = packed_cols * 32 / bits.
            // This holds for both power-of-two (4, 8) and non-power-of-two (3, 5, 6) bit widths.
            let real_cols = (w_shape[1] as usize) * 32 / bits as usize;
            let scale_cols = s_shape[1] as usize;
            let inferred = real_cols / scale_cols;
            if inferred == 0 { 64 } else { inferred as i32 }
        } else {
            64
        };
        (gs, bits)
    } else {
        (1, 32)
    };
    Some(QuantizedWeight {
        weight,
        scales,
        biases,
        group_size,
        bits,

        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q4_weight: None,
        decode_q4_scales: None,
        decode_q4_biases: None,
    })
}

/// 256-entry f32 lookup table for E8M0 scale bytes: `2^(b - 127)`.
///
/// `0xFF` is NaN by the OCP MX spec; the entry carries `f32::NAN` so an
/// out-of-range scale byte propagates NaN instead of silently mis-scaling.
fn mtp_e8m0_lut() -> Vec<f32> {
    (0..=255u32)
        .map(|b| {
            if b == 0xFF {
                f32::NAN
            } else {
                2f32.powi(b as i32 - 127)
            }
        })
        .collect()
}

/// Take an FP8 block-scaled weight pair (`{base}.weight` E4M3 bytes +
/// `{base}.scale` E8M0 bytes) and dequantize to a dense BF16 tensor.
///
/// This is the AXQuant DeepSeek V4 sidecar layout: standard DeepSeek
/// blockwise FP8 where the scale shape is the weight shape divided by the
/// block size per dim (128×128 in the published artifact). The block size
/// is derived per-dim from the shapes, not hardcoded. Returns `None` —
/// leaving both tensors in `name_map` — when the pair is absent or the
/// weight is not the FP8 byte container, so callers can fall back to
/// [`mtp_take_weight`] for raw-HF BF16 / MLX-packed sidecars. A malformed
/// FP8 pair (byte-container weight with an inconsistent scale grid) is
/// consumed before returning `None`, so the dense fallback cannot mistake
/// the raw E4M3 bytes for a dense weight; the sidecar is then reported
/// incomplete (fail closed).
fn mtp_take_fp8_blockscaled(
    name_map: &mut HashMap<String, MlxArray>,
    base: &str,
) -> Option<QuantizedWeight> {
    let weight_key = format!("{base}.weight");
    let scale_key = format!("{base}.scale");
    let (w_shape, s_shape) = {
        let weight = name_map.get(&weight_key)?;
        let scale = name_map.get(&scale_key)?;
        // Only the FP8 byte-container layout belongs to this helper; a dense
        // (e.g. BF16 raw-HF) weight falls through to `mtp_take_weight` even
        // when a `.scale` tensor happens to share the prefix.
        if weight.dtype() != MlxDtype::Uint8 {
            return None;
        }
        (weight.shape(), scale.shape())
    };
    if w_shape.len() != 2
        || s_shape.len() != 2
        || s_shape[0] == 0
        || s_shape[1] == 0
        || w_shape[0] % s_shape[0] != 0
        || w_shape[1] % s_shape[1] != 0
    {
        // Malformed FP8 pair: consume both so the dense fallback cannot read
        // the raw E4M3 bytes as a dense weight (fail closed).
        name_map.remove(&weight_key);
        name_map.remove(&scale_key);
        return None;
    }
    let block_rows = w_shape[0] / s_shape[0];
    let block_cols = w_shape[1] / s_shape[1];
    let weight = name_map.remove(&weight_key)?;
    let scale = name_map.remove(&scale_key)?;

    // E4M3 bytes → f32 via MLX's fp8 cast (the shim exposes fp8 payloads as
    // uint8 containers, the same contract as `to_fp8`'s output).
    let w_f32 = from_fp8(&weight, MlxDtype::Float32, None);
    // E8M0 bytes → f32 scales through the 256-entry LUT.
    let lut = MlxArray::from_f32_slice(&mtp_e8m0_lut());
    let scale_idx = astype(&scale, MlxDtype::Int32, None);
    let s_flat = take(&lut, &reshape(&scale_idx, &[-1], None), 0, None);
    let s_f32 = reshape(&s_flat, &s_shape, None);
    // Block broadcast: [so, si] → [so, 1, si, 1] → [so, bo, si, bi] → [out, in].
    let s4 = reshape(&s_f32, &[s_shape[0], 1, s_shape[1], 1], None);
    let s_block = broadcast_to(&s4, &[s_shape[0], block_rows, s_shape[1], block_cols], None);
    let s_full = reshape(&s_block, &w_shape, None);
    let dequantized = multiply(&w_f32, &s_full, None);
    Some(QuantizedWeight::new(
        astype(&dequantized, MlxDtype::Bfloat16, None),
        None,
        None,
    ))
}

/// Sanitize one expert's packed MXFP4 byte tensor (`[out, in/2]` u8/i8) into
/// MLX's packed-u32 quantized weight layout (`[out, in*4/32]` u32), mirroring
/// `load_mxfp4_blocks_scales` (u8→u32 view, then flatten the trailing dims
/// when the payload carries an explicit per-group axis; a 2-D payload is
/// already in the target layout after the view).
fn mxfp4_bytes_to_packed_u32(blocks: &MlxArray) -> Option<MlxArray> {
    let last = *blocks.shape().last()?;
    if last == 0 || last % 4 != 0 {
        return None;
    }
    let blocks_u32 = view(blocks, MlxDtype::Uint32, None);
    let ndim = blocks_u32.ndim();
    if ndim > 2 {
        Some(flatten(
            &blocks_u32,
            (ndim - 2) as i32,
            (ndim - 1) as i32,
            None,
        ))
    } else {
        Some(blocks_u32)
    }
}

/// Take per-expert MXFP4 routed experts from an AXQuant DeepSeek V4 sidecar
/// (`{bp}.ffn.experts.{N}.w{1,2,3}.{weight,scale}`) and stack them into the
/// packed SwitchGLU layout the V4 MoE forward consumes.
///
/// The checkpoint's "I8" expert tensors are MXFP4 payloads: E2M1 values
/// nibbled two-per-byte with E8M0 scales on group_size 32 — the exact byte
/// layout MLX's mxfp4 `gather_qmm` expects after the same u8→u32 view
/// sanitize as `load_mxfp4_blocks_scales`. Gate (`w1`) and up (`w3`) are
/// fused along the out dim into `gate_up_exps_packed`; down (`w2`) becomes
/// `down_exps`. Returns `None` when the per-expert naming is absent or any
/// expert tensor is missing; the caller then falls back to the stacked
/// `ffn.experts.{gate,up,down}` triple or treats the sidecar as incomplete.
fn mtp_take_mxfp4_experts(
    name_map: &mut HashMap<String, MlxArray>,
    bp: &str,
    expert_count: u32,
) -> Option<(QuantizedWeight, QuantizedWeight)> {
    if !name_map.contains_key(&format!("{bp}.ffn.experts.0.w1.weight")) {
        return None;
    }
    // Pre-flight: verify every expert tensor exists before consuming any, so
    // a partially-present per-expert set leaves `name_map` intact for the
    // stacked fallback and the leftover-tensor diagnostics.
    for expert in 0..expert_count {
        let prefix = format!("{bp}.ffn.experts.{expert}");
        for suffix in [
            "w1.weight",
            "w1.scale",
            "w2.weight",
            "w2.scale",
            "w3.weight",
            "w3.scale",
        ] {
            if !name_map.contains_key(&format!("{prefix}.{suffix}")) {
                return None;
            }
        }
    }
    let mut gate_up_weights = Vec::with_capacity(expert_count as usize);
    let mut gate_up_scales = Vec::with_capacity(expert_count as usize);
    let mut down_weights = Vec::with_capacity(expert_count as usize);
    let mut down_scales = Vec::with_capacity(expert_count as usize);
    for expert in 0..expert_count {
        let prefix = format!("{bp}.ffn.experts.{expert}");
        let w1 = name_map.remove(&format!("{prefix}.w1.weight"))?;
        let s1 = name_map.remove(&format!("{prefix}.w1.scale"))?;
        let w2 = name_map.remove(&format!("{prefix}.w2.weight"))?;
        let s2 = name_map.remove(&format!("{prefix}.w2.scale"))?;
        let w3 = name_map.remove(&format!("{prefix}.w3.weight"))?;
        let s3 = name_map.remove(&format!("{prefix}.w3.scale"))?;
        // Fuse gate (w1) + up (w3) along the out dim before packing so the
        // forward's last-dim split recovers the two halves.
        let gate_up = concatenate(&[&w1, &w3], 0, None);
        gate_up_weights.push(mxfp4_bytes_to_packed_u32(&gate_up)?);
        gate_up_scales.push(concatenate(&[&s1, &s3], 0, None));
        down_weights.push(mxfp4_bytes_to_packed_u32(&w2)?);
        down_scales.push(s2);
    }
    let gate_up_weight_refs: Vec<&MlxArray> = gate_up_weights.iter().collect();
    let gate_up_scale_refs: Vec<&MlxArray> = gate_up_scales.iter().collect();
    let down_weight_refs: Vec<&MlxArray> = down_weights.iter().collect();
    let down_scale_refs: Vec<&MlxArray> = down_scales.iter().collect();
    let gate_up = QuantizedWeight {
        weight: stack(&gate_up_weight_refs, 0, None),
        scales: Some(stack(&gate_up_scale_refs, 0, None)),
        biases: None,
        group_size: 32,
        bits: 4,
        mode: "mxfp4".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q4_weight: None,
        decode_q4_scales: None,
        decode_q4_biases: None,
    };
    let down = QuantizedWeight {
        weight: stack(&down_weight_refs, 0, None),
        scales: Some(stack(&down_scale_refs, 0, None)),
        biases: None,
        group_size: 32,
        bits: 4,
        mode: "mxfp4".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q4_weight: None,
        decode_q4_scales: None,
        decode_q4_biases: None,
    };
    Some((gate_up, down))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DraftLmHeadSpec {
    bits: i32,
    group_size: i32,
}

fn draft_lm_head_spec_from_runtime(v: &serde_json::Value) -> Option<DraftLmHeadSpec> {
    let spec = v.get("recommended_draft_lm_head")?;
    let mode = spec
        .get("mode")
        .and_then(|x| x.as_str())
        .unwrap_or("affine");
    if mode != "affine" {
        return None;
    }
    let bits = spec.get("bits").and_then(|x| x.as_i64())? as i32;
    let group_size = spec
        .get("group_size")
        .and_then(|x| x.as_i64())
        .unwrap_or(64) as i32;
    valid_draft_lm_head_spec(bits, group_size)
}

fn draft_lm_head_spec_from_env() -> Option<DraftLmHeadSpec> {
    let bits = std::env::var("AX_MLX_MTP_DRAFT_LM_HEAD_BITS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())?;
    let group_size = std::env::var("AX_MLX_MTP_DRAFT_LM_HEAD_GROUP_SIZE")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(64);
    valid_draft_lm_head_spec(bits, group_size)
}

fn runtime_draft_lm_head_spec_enabled() -> bool {
    std::env::var("AX_MLX_MTP_USE_RUNTIME_DRAFT_LM_HEAD")
        .map(|v| v != "0")
        .unwrap_or(false)
}

fn valid_draft_lm_head_spec(bits: i32, group_size: i32) -> Option<DraftLmHeadSpec> {
    if (2..=8).contains(&bits) && group_size > 0 {
        Some(DraftLmHeadSpec { bits, group_size })
    } else {
        None
    }
}

fn build_draft_lm_head(
    lm_head: &QuantizedWeight,
    spec: DraftLmHeadSpec,
) -> Option<QuantizedWeight> {
    if lm_head.is_quantized() && lm_head.bits == spec.bits && lm_head.group_size == spec.group_size
    {
        return Some(lm_head.clone());
    }
    let dense = if let Some(scales) = &lm_head.scales {
        dequantize_with_mode(
            &lm_head.weight,
            scales,
            lm_head.biases.as_ref(),
            Some(lm_head.group_size),
            Some(lm_head.bits),
            MlxQuantizationMode::Affine,
            None,
            Some(MlxDtype::Bfloat16),
            None,
        )
    } else {
        lm_head.weight.clone()
    };
    let dense = astype(&dense, MlxDtype::Bfloat16, None);
    eval(&[&dense]);
    let mut quantized = quantize(
        &dense,
        Some(spec.group_size),
        Some(spec.bits),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    if quantized.len() < 3 {
        return None;
    }
    let weight = quantized.remove(0);
    let scales = quantized.remove(0);
    let biases = quantized.remove(0);
    eval(&[&weight, &scales, &biases]);
    Some(QuantizedWeight {
        weight,
        scales: Some(scales),
        biases: Some(biases),
        group_size: spec.group_size,
        bits: spec.bits,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q4_weight: None,
        decode_q4_scales: None,
        decode_q4_biases: None,
    })
}

/// `AX_MLX_SKIP_VISION_SIDECAR=1` — do not merge `vision.safetensors`.
///
/// Text-only `--ax-direct` does not read the vision tower. Skipping the
/// sidecar avoids eval-ing ~0.9 GB of unused buffers into the Metal
/// residency set. Default off (load when present).
pub(crate) fn skip_vision_sidecar_from_env(raw: Option<&str>) -> bool {
    matches!(raw, Some(v) if v == "1" || v.eq_ignore_ascii_case("true"))
}

/// `AX_MLX_SKIP_MTP_SIDECAR=1` — do not merge `mtp.safetensors`.
///
/// `--ax-direct` does not run the MTP module. Skipping the sidecar avoids
/// eval-ing ~0.85 GB of unused buffers. Default off so MTP lanes still load.
pub(crate) fn skip_mtp_sidecar_from_env(raw: Option<&str>) -> bool {
    matches!(raw, Some(v) if v == "1" || v.eq_ignore_ascii_case("true"))
}

fn skip_vision_sidecar() -> bool {
    skip_vision_sidecar_from_env(std::env::var("AX_MLX_SKIP_VISION_SIDECAR").ok().as_deref())
}

fn skip_mtp_sidecar() -> bool {
    skip_mtp_sidecar_from_env(std::env::var("AX_MLX_SKIP_MTP_SIDECAR").ok().as_deref())
}

/// AXQuant protected vision sidecar file and provenance manifest names.
const VISION_SIDECAR_FILE: &str = "vision.safetensors";
const VISION_SIDECAR_MANIFEST_FILE: &str = "axquant_vision_sidecar_manifest.json";
const VISION_SIDECAR_SCHEMA: &str = "axquant.protected-tensor-sidecar.v1";

/// Provenance summary of a loaded AXQuant vision sidecar, surfaced via tracing.
#[derive(Debug)]
struct VisionSidecarInfo {
    tensor_count: usize,
    parameters: u64,
    source_model_id: String,
}

/// Streaming SHA-256 of a file as lowercase hex (sidecars can be large; avoid
/// reading them into memory just to hash).
fn file_sha256_hex(path: &std::path::Path) -> Result<String, std::io::Error> {
    use sha2::Digest as _;
    use std::fmt::Write as _;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        let _ = write!(output, "{byte:02x}");
    }
    Ok(output)
}

/// Load the AXQuant protected vision sidecar (`vision.safetensors`) if present.
///
/// AXQuant extracts vision towers into a sidecar plus a strict provenance
/// manifest (`axquant_vision_sidecar_manifest.json`). The sidecar file on its
/// own is not trusted: the manifest is required and its `output` binding
/// (path, size, SHA-256) must match the actual file, and `tensor_count` must
/// match the number of tensors the file yields. Sidecar tensors are merged
/// into `name_map` without overwriting tensors already loaded from the main
/// safetensors files (main file wins; duplicates are skipped with a debug
/// log).
///
/// Returns `Ok(None)` when neither the sidecar nor the manifest exists
/// (unchanged behavior for checkpoints without an extracted vision tower).
/// Every other inconsistency fails closed with
/// `WeightLoadError::VisionSidecarInvalid`.
fn load_vision_sidecar(
    root: &std::path::Path,
    name_map: &mut HashMap<String, MlxArray>,
) -> Result<Option<VisionSidecarInfo>, WeightLoadError> {
    if skip_vision_sidecar() {
        return Ok(None);
    }
    let sidecar = root.join(VISION_SIDECAR_FILE);
    let manifest_path = root.join(VISION_SIDECAR_MANIFEST_FILE);
    if !sidecar.exists() {
        return if manifest_path.exists() {
            Err(WeightLoadError::VisionSidecarInvalid(format!(
                "provenance manifest {} exists but {VISION_SIDECAR_FILE} is missing",
                manifest_path.display()
            )))
        } else {
            Ok(None)
        };
    }

    // The sidecar file requires its provenance manifest.
    let invalid = |message: String| WeightLoadError::VisionSidecarInvalid(message);
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|_| {
        invalid(format!(
            "{VISION_SIDECAR_FILE} requires provenance manifest {} (missing or unreadable)",
            manifest_path.display()
        ))
    })?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).map_err(|error| {
        invalid(format!(
            "vision sidecar manifest {} is not valid JSON: {error}",
            manifest_path.display()
        ))
    })?;
    let schema_version = manifest.get("schema_version").and_then(|v| v.as_str());
    if schema_version != Some(VISION_SIDECAR_SCHEMA) {
        return Err(invalid(format!(
            "vision sidecar manifest schema_version must be {VISION_SIDECAR_SCHEMA:?}, \
             found {schema_version:?}"
        )));
    }
    let role = manifest.get("role").and_then(|v| v.as_str());
    if role != Some("vision") {
        return Err(invalid(format!(
            "vision sidecar manifest role must be \"vision\", found {role:?}"
        )));
    }

    // Verify the manifest's output binding against the actual file.
    let output = manifest
        .get("output")
        .ok_or_else(|| invalid("vision sidecar manifest missing output binding".to_string()))?;
    let output_path = output.get("path").and_then(|v| v.as_str());
    if output_path
        .and_then(|p| std::path::Path::new(p).file_name())
        .and_then(|n| n.to_str())
        != Some(VISION_SIDECAR_FILE)
    {
        return Err(invalid(format!(
            "vision sidecar manifest output.path must name {VISION_SIDECAR_FILE}, \
             found {output_path:?}"
        )));
    }
    let actual_size = std::fs::metadata(&sidecar)
        .map_err(|error| invalid(format!("cannot stat {VISION_SIDECAR_FILE}: {error}")))?
        .len();
    let expected_size = output.get("size_bytes").and_then(|v| v.as_u64());
    if expected_size != Some(actual_size) {
        return Err(invalid(format!(
            "vision sidecar size mismatch: manifest output.size_bytes {expected_size:?}, \
             actual {actual_size}"
        )));
    }
    let expected_sha = output.get("sha256").and_then(|v| v.as_str());
    let actual_sha = file_sha256_hex(&sidecar)
        .map_err(|error| invalid(format!("cannot hash {VISION_SIDECAR_FILE}: {error}")))?;
    if expected_sha != Some(actual_sha.as_str()) {
        return Err(invalid(format!(
            "vision sidecar sha256 mismatch: manifest output.sha256 {expected_sha:?}, \
             actual {actual_sha:?}"
        )));
    }

    let tensors = load_safetensors(&sidecar, None)
        .map_err(|error| invalid(format!("cannot load {VISION_SIDECAR_FILE}: {error}")))?;
    let tensor_count = manifest
        .get("tensor_count")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| invalid("vision sidecar manifest missing tensor_count".to_string()))?;
    if tensor_count != tensors.len() as u64 {
        return Err(invalid(format!(
            "vision sidecar tensor_count mismatch: manifest {tensor_count}, \
             file yields {}",
            tensors.len()
        )));
    }
    let parameters = manifest
        .get("parameters")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| invalid("vision sidecar manifest missing parameters".to_string()))?;
    let source_model_id = manifest
        .get("source_model")
        .and_then(|v| v.get("model_id"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            invalid("vision sidecar manifest missing source_model.model_id".to_string())
        })?
        .to_string();

    // Same eval contract as the main-file loader: route the freshly built
    // arrays through the lazy graph before anything consumes them.
    if !tensors.is_empty() {
        let refs: Vec<&MlxArray> = tensors.values().collect();
        eval(&refs);
    }
    let mut skipped = 0usize;
    for (name, array) in tensors {
        match name_map.entry(name) {
            Entry::Occupied(_) => skipped += 1,
            Entry::Vacant(entry) => {
                entry.insert(array);
            }
        }
    }
    if skipped > 0 {
        tracing::debug!(
            target = "ax_engine_mlx",
            skipped,
            "vision sidecar tensors already present from main files; main file wins"
        );
    }

    let info = VisionSidecarInfo {
        tensor_count: tensor_count as usize,
        parameters,
        source_model_id,
    };
    tracing::info!(
        target = "ax_engine_mlx",
        tensor_count = info.tensor_count,
        parameters = info.parameters,
        source_model_id = info.source_model_id.as_str(),
        "loaded AXQuant vision sidecar (provenance verified)"
    );
    Ok(Some(info))
}

/// Load the MTP sidecar file (`mtp.safetensors`) if present alongside the main model.
///
/// Adds sidecar tensors into `name_map` and returns
/// `(max_depth, draft_sampling, sidecar_bits)`:
/// - `max_depth`: from `mtplx_runtime.json` `mtp_depth_max`, capped by local depth policy
///   (default 1, 0 when no sidecar).
/// - `draft_sampling`: from `mtplx_runtime.json` `recommended_draft_sampler`
///   (defaults: temperature=0.7, top_k=20, top_p=0.95).
/// - `sidecar_bits`: `Some(8)` for INT8 sidecars, `Some(4)` for INT4, or
///   `None` (default 4).
/// - `norm_layout`: from `mtplx_runtime.json` `mtp_norm_layout`
///   (`"raw_hf_delta"` / `"mlx_multiplier"`, default auto-detection).
fn load_mtp_sidecar(
    root: &std::path::Path,
    name_map: &mut HashMap<String, MlxArray>,
    manifest: &ax_engine_core::NativeModelManifest,
) -> (
    usize,
    MlxSamplingParams,
    Option<i32>,
    Option<DraftLmHeadSpec>,
    MtpNormLayout,
) {
    // MTPLX default draft sampler: temperature slightly above target (0.6) to
    // ensure rejection-sampling acceptance rates ≥97%.
    // AX_MLX_MTP_DRAFT_TEMPERATURE overrides the draft temperature from the
    // sidecar config or this default.  Lightning-MLX uses 0.5 for code/tool-call
    // workloads where tighter draft distributions lift acceptance.
    let default_draft = MlxSamplingParams::new(0.7, 0.95, 20);

    // DeepSeek V4 packages also ship their nextn block in `mtp.safetensors`;
    // the Qwen layout (`mtp.layers.N.*`) must never consume those tensors —
    // V4 nextn weights load via `load_deepseek_v4_mtp_sidecar`.
    if manifest.deepseek_v4.is_enabled() {
        return (0, default_draft, None, None, MtpNormLayout::Auto);
    }
    if skip_mtp_sidecar() {
        return (0, default_draft, None, None, MtpNormLayout::Auto);
    }

    let sidecar = root.join("mtp.safetensors");
    if !sidecar.exists() {
        return (0, default_draft, None, None, MtpNormLayout::Auto);
    }
    let tensors = match load_safetensors(&sidecar, None) {
        Ok(t) => t,
        Err(_) => return (0, default_draft, None, None, MtpNormLayout::Auto),
    };
    if !tensors.is_empty() {
        let refs: Vec<&MlxArray> = tensors.values().collect();
        eval(&refs);
    }
    name_map.extend(tensors);

    // Parse depth, draft sampling, and sidecar quantization bits from MTPLX runtime config.
    let runtime_path = root.join("mtplx_runtime.json");
    if let Ok(bytes) = std::fs::read(&runtime_path)
        && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes)
    {
        let raw_depth = v.get("mtp_depth_max").and_then(|x| x.as_u64()).unwrap_or(1) as usize;
        let draft_sampling = if let Some(ds) = v.get("recommended_draft_sampler") {
            let temp = ds
                .get("temperature")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.7) as f32;
            let top_k = ds.get("top_k").and_then(|x| x.as_u64()).unwrap_or(20) as u32;
            let top_p = ds.get("top_p").and_then(|x| x.as_f64()).unwrap_or(0.95) as f32;
            MlxSamplingParams::new(temp, top_p, top_k)
        } else {
            default_draft
        };
        let sidecar_bits = parse_mtp_sidecar_bits_hint(&v);
        let depth = apply_mtp_depth_policy(raw_depth, sidecar_bits);
        return (
            depth,
            apply_draft_temperature_override(draft_sampling),
            sidecar_bits,
            if runtime_draft_lm_head_spec_enabled() {
                draft_lm_head_spec_from_runtime(&v).or_else(draft_lm_head_spec_from_env)
            } else {
                draft_lm_head_spec_from_env()
            },
            parse_mtp_norm_layout(&v),
        );
    }
    (
        apply_mtp_max_depth_cap(1),
        apply_draft_temperature_override(default_draft),
        None,
        draft_lm_head_spec_from_env(),
        MtpNormLayout::Auto,
    )
}

/// Load the GLM MTP sidecar (`glm_mtp.safetensors`) if present alongside the main model.
///
/// Returns `Some(GlmMtpWeights)` when the sidecar is found and all required tensors are present.
/// Returns `None` gracefully (no MTP head active) when the sidecar is absent or incomplete.
fn load_glm_mtp_sidecar(
    root: &std::path::Path,
    name_map: &mut HashMap<String, MlxArray>,
    manifest: &ax_engine_core::NativeModelManifest,
) -> Option<GlmMtpWeights> {
    let default_draft = MlxSamplingParams::new(0.7, 0.95, 20);

    // DeepSeek V4 manifests carry nextn (MTP) tensors but no V3 MLA dims, so
    // the kv_b split below (which hard-requires V3 MLA dims) must never run
    // for them; V4 nextn weights load via `load_deepseek_v4_nextn_weights`.
    if manifest.deepseek_v4.is_enabled() {
        return None;
    }

    let sidecar = root.join("glm_mtp.safetensors");
    if !sidecar.exists() {
        return None;
    }
    let tensors = match load_safetensors(&sidecar, None) {
        Ok(t) => t,
        Err(_) => return None,
    };
    if !tensors.is_empty() {
        let refs: Vec<&MlxArray> = tensors.values().collect();
        eval(&refs);
    }
    name_map.extend(tensors);

    // Parse depth, draft sampling, and sidecar quantization bits from runtime config.
    let runtime_path = root.join("glm_mtp_runtime.json");
    let (max_depth, draft_sampling, bits) = if let Ok(bytes) = std::fs::read(&runtime_path)
        && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes)
    {
        let raw_depth = v.get("mtp_depth_max").and_then(|x| x.as_u64()).unwrap_or(1) as usize;
        let draft_sampling = if let Some(ds) = v.get("recommended_draft_sampler") {
            let temp = ds
                .get("temperature")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.7) as f32;
            let top_k = ds.get("top_k").and_then(|x| x.as_u64()).unwrap_or(20) as u32;
            let top_p = ds.get("top_p").and_then(|x| x.as_f64()).unwrap_or(0.95) as f32;
            MlxSamplingParams::new(temp, top_p, top_k)
        } else {
            default_draft
        };
        // Without this hint, every 2-D projection below fell back to
        // `bits=4` in `mtp_take_weight`, silently mis-inferring `group_size`
        // (~2x too large) for INT8 sidecars produced by
        // `scripts/prepare_glm_mtp_sidecar.py --quantize 8` — the packed
        // integers get unpacked as 8x4-bit values instead of 4x8-bit values,
        // producing wrong dequantized weights with no validation error.
        let sidecar_bits = parse_mtp_sidecar_bits_hint(&v);
        (
            apply_mtp_max_depth_cap(raw_depth),
            apply_draft_temperature_override(draft_sampling),
            sidecar_bits,
        )
    } else {
        (
            apply_mtp_max_depth_cap(1),
            apply_draft_temperature_override(default_draft),
            None,
        )
    };

    if max_depth == 0 {
        return None;
    }

    let p = "glm_mtp";

    // Resolve MLA attention config from the manifest.
    let mla_config = MlaAttentionConfig::from_manifest(manifest)?;

    // Scalar norms for the MTP head.
    let enorm = mtp_take_plain(name_map, &format!("{p}.enorm.weight"))?;
    let hnorm = mtp_take_plain(name_map, &format!("{p}.hnorm.weight"))?;
    let shared_head_norm = mtp_take_plain(name_map, &format!("{p}.shared_head.norm.weight"))?;

    // eh_proj: [2*hidden → hidden] linear.
    let eh_proj = mtp_take_weight(name_map, &format!("{p}.eh_proj"), bits)?;
    // shared_head.head: [hidden → vocab] draft logit projection.
    let shared_head = mtp_take_weight(name_map, &format!("{p}.shared_head.head"), bits)?;

    // Layer norms for the transformer block.
    let attn_norm = mtp_take_plain(name_map, &format!("{p}.layer.input_layernorm.weight"))?;
    let ffn_norm = mtp_take_plain(
        name_map,
        &format!("{p}.layer.post_attention_layernorm.weight"),
    )?;

    // MLA attention projections.
    let q_a_proj = mtp_take_weight(name_map, &format!("{p}.layer.self_attn.q_a_proj"), bits)?;
    let kv_a_proj = mtp_take_weight(name_map, &format!("{p}.layer.self_attn.kv_a_proj"), bits)
        .or_else(|| {
            mtp_take_weight(
                name_map,
                &format!("{p}.layer.self_attn.kv_a_proj_with_mqa"),
                bits,
            )
        })?;
    let q_a_norm = mtp_take_plain(
        name_map,
        &format!("{p}.layer.self_attn.q_a_layernorm.weight"),
    )?;
    let kv_a_norm = mtp_take_plain(
        name_map,
        &format!("{p}.layer.self_attn.kv_a_layernorm.weight"),
    )?;
    let q_b_proj = mtp_take_weight(name_map, &format!("{p}.layer.self_attn.q_b_proj"), bits)?;
    let (embed_q, unembed_out) = if let Some(kv_b) =
        mtp_take_weight(name_map, &format!("{p}.layer.self_attn.kv_b_proj"), bits)
    {
        split_deepseek_kv_b_projection(kv_b, &manifest.mla_attention, manifest.attention_head_count)
            .ok()?
    } else {
        (
            mtp_take_weight(name_map, &format!("{p}.layer.self_attn.embed_q"), bits)?,
            mtp_take_weight(name_map, &format!("{p}.layer.self_attn.unembed_out"), bits)?,
        )
    };
    let o_proj = mtp_take_weight(name_map, &format!("{p}.layer.self_attn.o_proj"), bits)?;

    // Fuse q_a_proj + kv_a_proj into a single matmul weight.
    let qa_kva_fused = pack_glm_mla_qa_kva_projection(&q_a_proj, &kv_a_proj).ok()?;

    let glm_mla_attn = Some(GlmMlaAttentionWeights {
        qa_kva_fused,
        q_a_norm,
        q_b_proj,
        kv_a_norm,
        embed_q,
        unembed_out,
    });

    // MoE FFN: router + expert stacks.
    let router_proj = mtp_take_weight(name_map, &format!("{p}.layer.mlp.gate"), bits);
    let router_correction_bias = mtp_take_plain(
        name_map,
        &format!("{p}.layer.mlp.gate.e_score_correction_bias"),
    );
    let shared_gate_proj = mtp_take_weight(
        name_map,
        &format!("{p}.layer.mlp.shared_expert.gate_proj"),
        bits,
    )
    .or_else(|| {
        mtp_take_weight(
            name_map,
            &format!("{p}.layer.mlp.shared_experts.gate_proj"),
            bits,
        )
    });
    let shared_up_proj = mtp_take_weight(
        name_map,
        &format!("{p}.layer.mlp.shared_expert.up_proj"),
        bits,
    )
    .or_else(|| {
        mtp_take_weight(
            name_map,
            &format!("{p}.layer.mlp.shared_experts.up_proj"),
            bits,
        )
    });
    let shared_down_proj = mtp_take_weight(
        name_map,
        &format!("{p}.layer.mlp.shared_expert.down_proj"),
        bits,
    )
    .or_else(|| {
        mtp_take_weight(
            name_map,
            &format!("{p}.layer.mlp.shared_experts.down_proj"),
            bits,
        )
    });
    let gate_exps = mtp_take_weight(name_map, &format!("{p}.layer.mlp.gate_proj"), bits);
    let up_exps = mtp_take_weight(name_map, &format!("{p}.layer.mlp.up_proj"), bits);
    let down_exps = mtp_take_weight(name_map, &format!("{p}.layer.mlp.down_proj"), bits);

    let has_moe_ffn = router_proj.is_some();
    if has_moe_ffn
        && (gate_exps.is_none()
            || up_exps.is_none()
            || down_exps.is_none()
            || shared_gate_proj.is_none()
            || shared_up_proj.is_none()
            || shared_down_proj.is_none())
    {
        tracing::warn!(
            target: "ax_mlx::weights",
            "GLM MTP sidecar: found router but missing MoE expert tensors — skipping MTP"
        );
        return None;
    }

    let layer = LayerWeights {
        attn_norm,
        attn_post_norm: None,
        q_norm: None,
        k_norm: None,
        q_proj: None,
        k_proj: None,
        v_proj: None,
        qkv_packed: None,
        o_proj: Some(o_proj),
        linear_attn: None,
        glm_mla_attn,
        deepseek_v4: None,
        ffn_norm,
        ffn_post_norm: None,
        gate_proj: None,
        up_proj: None,
        gate_up_packed: None,
        down_proj: None,
        ffn_norm2: None,
        ffn_post_norm1: None,
        ffn_post_norm2: None,
        router_proj,
        router_correction_bias,
        router_scale: None,
        router_combined_scale: None,
        router_expert_scale: None,
        layer_scalar: None,
        per_layer_gate: None,
        per_layer_proj_w: None,
        per_layer_post_norm: None,
        shared_expert_gate: None,
        shared_gate_up_proj: None,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        gate_up_exps_packed: None,
        gate_exps,
        up_exps,
        down_exps,
        mxfp4_gate_up_exps: None,
        mxfp4_down_exps: None,
        attn_sink: None,
        rotation_smoothing_inverse: None,
        expert_stream: None,
    };

    Some(GlmMtpWeights {
        enorm,
        hnorm,
        eh_proj,
        shared_head_norm,
        shared_head,
        layer,
        mla_config,
        max_depth,
        draft_sampling,
    })
}

/// Override the MTP draft sampling temperature from `AX_MLX_MTP_DRAFT_TEMPERATURE`.
///
/// Lightning-MLX defaults to draft temperature 0.5 for code/tool-call workloads
/// where tighter draft distributions lift MTP acceptance.  Our sidecar default
/// is 0.7 (from `mtplx_runtime.json` or the hardcoded fallback).  This env var
/// lets benchmark runs tune the draft temperature without rebuilding the sidecar.
fn apply_draft_temperature_override(params: MlxSamplingParams) -> MlxSamplingParams {
    static CACHED: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    let override_temp = CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_DRAFT_TEMPERATURE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|&t| (0.0..=2.0).contains(&t))
    });
    if let Some(temp) = override_temp {
        MlxSamplingParams::new(*temp, params.top_p, params.top_k)
    } else {
        params
    }
}

/// Detect sidecar quantization bits from an MTP runtime JSON config
/// (`mtplx_runtime.json` for Qwen, `glm_mtp_runtime.json` for GLM).
///
/// The structured `mtp_sidecar_bits` field wins when present: integer values
/// in {2, 4, 6, 8, 16} are used directly; a present-but-malformed value (wrong
/// type or outside the set) logs a warning and falls through to the free-text
/// heuristic. The heuristic substring-matches the `mtp_sidecar` description:
/// `"INT8"`/`"8BIT"` → 8-bit; anything else present → 4-bit; the field absent
/// entirely → `None` (caller infers from tensor shapes, defaulting to 4-bit
/// in `mtp_take_weight`).
/// Sidecar quantization bit widths the MTP loaders accept. Shared with the
/// `ax-engine mtp-capability` CLI contract so capability reporting can never
/// drift from what `mtp_take_weight` actually executes.
pub const MTP_SIDECAR_SUPPORTED_BITS: [i32; 5] = [2, 4, 6, 8, 16];

/// Layout id AXQuant stamps in `mtplx_runtime.json` for Qwen 3.6 MTP sidecars
/// consumed by this loader (byte-preserved or MLX-packed quantized).
pub const MTP_SIDECAR_QWEN36_LAYOUT: &str = "ax-engine-qwen36-v1";

fn parse_mtp_sidecar_bits_hint(runtime_config: &serde_json::Value) -> Option<i32> {
    if let Some(structured) = runtime_config.get("mtp_sidecar_bits") {
        match structured
            .as_i64()
            .and_then(|bits| i32::try_from(bits).ok())
        {
            Some(bits) if MTP_SIDECAR_SUPPORTED_BITS.contains(&bits) => return Some(bits),
            _ => tracing::warn!(
                target: "ax_mlx::weights",
                value = %structured,
                "malformed mtp_sidecar_bits in MTP runtime config; \
                 falling back to the mtp_sidecar free-text heuristic"
            ),
        }
    }
    runtime_config
        .get("mtp_sidecar")
        .and_then(|s| s.as_str())
        .map(|s| {
            let upper = s.to_ascii_uppercase();
            let bits = if upper.contains("INT8") || upper.contains("8BIT") {
                8
            } else {
                4
            };
            tracing::debug!(
                target: "ax_mlx::weights",
                bits,
                "guessed MTP sidecar bits from mtp_sidecar free text; \
                 declare mtp_sidecar_bits to make this explicit"
            );
            bits
        })
}

fn mtp_router_bits_hint(sidecar_bits: Option<i32>) -> Option<i32> {
    // prepare_mtp_sidecar.py keeps the MoE router at INT8 even when the other
    // eligible 2-D projections use INT4. Without this tensor-specific hint,
    // the packed router columns are interpreted as INT4 and expand to twice
    // the model hidden size.
    sidecar_bits.map(|_| 8)
}

fn apply_mtp_depth_policy(depth: usize, sidecar_bits: Option<i32>) -> usize {
    if std::env::var("AX_MLX_MTP_MAX_DEPTH").is_ok() {
        return apply_mtp_max_depth_cap(depth);
    }

    default_mtp_depth_without_env(depth, sidecar_bits)
}

fn default_mtp_depth_without_env(depth: usize, _sidecar_bits: Option<i32>) -> usize {
    depth
}

fn apply_mtp_max_depth_cap(depth: usize) -> usize {
    let Ok(raw) = std::env::var("AX_MLX_MTP_MAX_DEPTH") else {
        return depth;
    };
    match parse_mtp_max_depth_cap(&raw) {
        Some(cap) => depth.min(cap),
        None => depth,
    }
}

fn parse_mtp_max_depth_cap(raw: &str) -> Option<usize> {
    raw.trim().parse::<usize>().ok()
}

/// Declared representation of the MTP sidecar's 1-D RMSNorm tensors, from the
/// optional `mtp_norm_layout` field of `mtplx_runtime.json`.
///
/// Third-party converters (e.g. AXQuant byte-preserved sidecars) can declare
/// the layout explicitly so the loader never has to guess from tensor
/// statistics. Absent or unknown values fall back to auto-detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MtpNormLayout {
    /// Norms are raw HF zero-centred deltas; the loader applies `+1.0` to every norm.
    RawHfDelta,
    /// Norms are already shifted MLX multipliers; the loader leaves them unchanged.
    MlxMultiplier,
    /// No declaration; decide from whole-sidecar `mean_abs` statistics.
    Auto,
}

fn parse_mtp_norm_layout(v: &serde_json::Value) -> MtpNormLayout {
    match v.get("mtp_norm_layout").and_then(|x| x.as_str()) {
        Some("raw_hf_delta") => MtpNormLayout::RawHfDelta,
        Some("mlx_multiplier") => MtpNormLayout::MlxMultiplier,
        Some(other) => {
            tracing::warn!(
                target: "ax_mlx::weights",
                value = other,
                "unknown mtp_norm_layout in mtplx_runtime.json; using auto-detection"
            );
            MtpNormLayout::Auto
        }
        None => MtpNormLayout::Auto,
    }
}

/// Decide whether the sidecar's norm tensors need the `+1.0` raw-HF-delta lift.
///
/// The decision is per-sidecar, not per-tensor: all norms come from one export
/// path, so either every norm is a raw delta or every norm is a shifted
/// multiplier. Raw deltas are not uniformly small — Qwen 3.6's raw
/// `q_norm`/`k_norm`/`mtp.norm` deltas have `mean_abs` between 0.21 and 1.27
/// while its raw input-layernorm sits at 0.08 — so a single sub-threshold norm
/// marks the entire sidecar raw. Deciding per tensor instead leaves the sidecar
/// in a silently mixed state that collapses draft acceptance to zero.
fn mtp_norms_need_shift(layout: MtpNormLayout, mean_abs_values: &[Option<f32>]) -> bool {
    match layout {
        MtpNormLayout::RawHfDelta => true,
        MtpNormLayout::MlxMultiplier => false,
        MtpNormLayout::Auto => mean_abs_values
            .iter()
            .flatten()
            .any(|m| *m < SANITIZED_NORM_MIN_MEAN_ABS),
    }
}

/// Apply the `+1.0` HF-delta → MLX-multiplier conversion, preserving dtype.
fn shift_mtp_norm(w: MlxArray) -> MlxArray {
    let one = MlxArray::from_f32(1.0_f32);
    let corrected = add(&astype(&w, MlxDtype::Float32, None), &one, None);
    astype(&corrected, w.dtype(), None)
}

/// Try to load MTP weights from `mtp.*` keys in `name_map`.
///
/// Returns `None` when no MTP keys are found, allowing graceful fallback to
/// n-gram speculative decoding.
fn load_mtp(
    name_map: &mut HashMap<String, MlxArray>,
    lm_head: &QuantizedWeight,
    max_depth: usize,
    draft_sampling: MlxSamplingParams,
    sidecar_bits: Option<i32>,
    draft_lm_head_spec: Option<DraftLmHeadSpec>,
    norm_layout: MtpNormLayout,
) -> Option<MtpWeights> {
    if max_depth == 0 {
        return None;
    }

    let bits = sidecar_bits; // propagated to all mtp_take_weight calls

    // Global norms and FC projection (required).
    let pre_fc_norm_embedding = mtp_take_plain(name_map, "mtp.pre_fc_norm_embedding.weight")?;
    let pre_fc_norm_hidden = mtp_take_plain(name_map, "mtp.pre_fc_norm_hidden.weight")?;
    let mtp_norm = mtp_take_plain(name_map, "mtp.norm.weight")?;
    let fc = mtp_take_weight(name_map, "mtp.fc", bits)?;

    // Layer-0 weights (the single transformer layer applied recurrently).
    let p = "mtp.layers.0";
    let attn_norm = mtp_take_plain(name_map, &format!("{p}.input_layernorm.weight"))?;
    let ffn_norm = mtp_take_plain(name_map, &format!("{p}.post_attention_layernorm.weight"))?;
    let q_norm = mtp_take_plain(name_map, &format!("{p}.self_attn.q_norm.weight"));
    let k_norm = mtp_take_plain(name_map, &format!("{p}.self_attn.k_norm.weight"));
    let q_proj = mtp_take_weight(name_map, &format!("{p}.self_attn.q_proj"), bits)?;
    let k_proj = mtp_take_weight(name_map, &format!("{p}.self_attn.k_proj"), bits)?;
    let v_proj = mtp_take_weight(name_map, &format!("{p}.self_attn.v_proj"), bits)?;
    let o_proj = mtp_take_weight(name_map, &format!("{p}.self_attn.o_proj"), bits)?;
    let router_proj = mtp_take_weight(
        name_map,
        &format!("{p}.mlp.gate"),
        mtp_router_bits_hint(bits),
    );
    let shared_expert_gate =
        mtp_take_weight(name_map, &format!("{p}.mlp.shared_expert_gate"), bits);
    let shared_gate_proj =
        mtp_take_weight(name_map, &format!("{p}.mlp.shared_expert.gate_proj"), bits);
    let shared_up_proj = mtp_take_weight(name_map, &format!("{p}.mlp.shared_expert.up_proj"), bits);
    let shared_down_proj =
        mtp_take_weight(name_map, &format!("{p}.mlp.shared_expert.down_proj"), bits);
    // Routed-expert layouts (prefer split, fall back to MLX fused packing):
    //   1) mlp.{gate,up,down}_proj stacked experts (legacy / HF-style)
    //   2) mlp.experts.gate_up_proj + mlp.experts.down_proj (Qwen3.5/3.6 MoE
    //      A3B sidecars from axquant; matches main-model FfnGateUpExpsPacked)
    let gate_exps = mtp_take_weight(name_map, &format!("{p}.mlp.gate_proj"), bits);
    let up_exps = mtp_take_weight(name_map, &format!("{p}.mlp.up_proj"), bits);
    let down_exps = mtp_take_weight(name_map, &format!("{p}.mlp.down_proj"), bits)
        .or_else(|| mtp_take_weight(name_map, &format!("{p}.mlp.experts.down_proj"), bits));
    let gate_up_exps_packed = if gate_exps.is_none() && up_exps.is_none() {
        mtp_take_weight(name_map, &format!("{p}.mlp.experts.gate_up_proj"), bits)
    } else {
        None
    };
    let has_moe_ffn = router_proj.is_some();
    let (gate_proj, up_proj, down_proj, gate_exps, up_exps, down_exps) = if has_moe_ffn {
        (None, None, None, gate_exps, up_exps, down_exps)
    } else {
        (gate_exps, up_exps, down_exps, None, None, None)
    };
    let moe_experts_complete = down_exps.is_some()
        && (gate_up_exps_packed.is_some() || (gate_exps.is_some() && up_exps.is_some()));
    if has_moe_ffn
        && (!moe_experts_complete
            || shared_gate_proj.is_none()
            || shared_up_proj.is_none()
            || shared_down_proj.is_none())
    {
        return None;
    }
    if !has_moe_ffn && (gate_proj.is_none() || up_proj.is_none() || down_proj.is_none()) {
        return None;
    }

    // Convert unshifted MTP norm weights produced by sidecars that omitted the
    // `+1.0` HF-delta → MLX-multiplier transform. Raw HF delta norms cause all
    // MTP activations to collapse to near-zero, which makes every draft token
    // the same garbage token (typically `!`). The shift decision is made once
    // for the whole sidecar — from the declared `mtp_norm_layout` when present,
    // otherwise from the norms' combined mean_abs statistics — and then applied
    // to every norm, matching what `prepare_mtp_sidecar.py` produces.
    // Must run before `ffn_layer` is built so that `attn_norm`/`ffn_norm` clones
    // inside `ffn_layer` also receive the corrected values.
    let mean_abs_values: Vec<Option<f32>> = [
        Some(&pre_fc_norm_embedding),
        Some(&pre_fc_norm_hidden),
        Some(&mtp_norm),
        Some(&attn_norm),
        Some(&ffn_norm),
        q_norm.as_ref(),
        k_norm.as_ref(),
    ]
    .into_iter()
    .flatten()
    .map(norm_mean_abs)
    .collect();
    let shift_norms = mtp_norms_need_shift(norm_layout, &mean_abs_values);
    let (pre_fc_norm_embedding, pre_fc_norm_hidden, mtp_norm, attn_norm, ffn_norm, q_norm, k_norm) =
        if shift_norms {
            if norm_layout == MtpNormLayout::Auto {
                tracing::warn!(
                    target: "ax_mlx::weights",
                    "MTP sidecar norms detected as raw HF deltas; applying the +1.0 shift to all \
                     norm tensors. Declare `mtp_norm_layout` in mtplx_runtime.json or regenerate \
                     the sidecar with scripts/prepare_mtp_sidecar.py to make this explicit."
                );
                eprintln!(
                    "[ax_mlx::weights] MTP sidecar norms auto-corrected (+1.0 shift applied to all \
                 norm tensors). Declare `mtp_norm_layout` in mtplx_runtime.json or regenerate \
                 the sidecar with scripts/prepare_mtp_sidecar.py to make this explicit."
                );
            }
            (
                shift_mtp_norm(pre_fc_norm_embedding),
                shift_mtp_norm(pre_fc_norm_hidden),
                shift_mtp_norm(mtp_norm),
                shift_mtp_norm(attn_norm),
                shift_mtp_norm(ffn_norm),
                q_norm.map(shift_mtp_norm),
                k_norm.map(shift_mtp_norm),
            )
        } else {
            (
                pre_fc_norm_embedding,
                pre_fc_norm_hidden,
                mtp_norm,
                attn_norm,
                ffn_norm,
                q_norm,
                k_norm,
            )
        };

    let ffn_layer = LayerWeights {
        attn_norm: attn_norm.clone(),
        attn_post_norm: None,
        q_norm: None,
        k_norm: None,
        q_proj: None,
        k_proj: None,
        v_proj: None,
        qkv_packed: None,
        o_proj: None,
        linear_attn: None,
        glm_mla_attn: None,
        deepseek_v4: None,
        ffn_norm: ffn_norm.clone(),
        ffn_post_norm: None,
        gate_proj,
        up_proj,
        gate_up_packed: None,
        down_proj,
        ffn_norm2: None,
        ffn_post_norm1: None,
        ffn_post_norm2: None,
        router_proj,
        router_correction_bias: None,
        router_scale: None,
        router_combined_scale: None,
        router_expert_scale: None,
        layer_scalar: None,
        per_layer_gate: None,
        per_layer_proj_w: None,
        per_layer_post_norm: None,
        shared_expert_gate,
        shared_gate_up_proj: None,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        gate_up_exps_packed,
        gate_exps,
        up_exps,
        down_exps,
        mxfp4_gate_up_exps: None,
        mxfp4_down_exps: None,
        attn_sink: None,
        rotation_smoothing_inverse: None,
        expert_stream: None,
    };

    // Infer n_heads, n_kv_heads, head_dim from projection weight shapes.
    //
    // Qwen3-next-MTP attention (Qwen3NextAttention) produces queries AND a gating
    // signal from a single q_proj: output = n_heads * head_dim * 2 (first half =
    // queries, second half = gate applied after attention with a sigmoid).
    // So q_rows = n_heads * head_dim * 2 → n_heads = q_rows / (head_dim * 2).
    //
    // head_dim is inferred from q_norm weight shape when available (most reliable),
    // otherwise probed from candidate values.
    let q_shape = q_proj.weight.shape();
    let k_shape = k_proj.weight.shape();
    let q_rows = q_shape.first().copied()? as usize;
    let k_rows = k_shape.first().copied()? as usize;
    // q_norm weight is a 1-D array of size [head_dim].
    let head_dim = q_norm
        .as_ref()
        .and_then(|qn| qn.shape().first().copied())
        .map(|d| d as usize)
        .or_else(|| {
            // Fallback: probe common head_dims assuming q_rows = n_heads * head_dim * 2.
            [256usize, 128, 64, 96]
                .iter()
                .copied()
                .find(|&d| (q_rows / 2).is_multiple_of(d))
        })?;
    // q_proj rows = n_heads * head_dim * 2 (queries + gate).
    let n_heads = q_rows / (head_dim * 2);
    let n_kv_heads = k_rows / head_dim;

    Some(MtpWeights {
        fc,
        mtp_norm,
        draft_lm_head: draft_lm_head_spec.and_then(|spec| build_draft_lm_head(lm_head, spec)),
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        attn_norm,
        ffn_norm,
        q_norm,
        k_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        ffn_layer,
        n_heads,
        n_kv_heads,
        head_dim,
        max_depth,
        draft_sampling,
    })
}

/// Detect a sibling `model.rotated.safetensors` and, when the runtime is in
/// `WeightRotationMode::Apply`, replace each layer's `gate_proj` / `up_proj`
/// with the rotated f32 version produced offline by
/// `scripts/quantize_rotated_weights.py --apply`. Fail-closed: if Apply mode
/// is selected but the rotated checkpoint is missing / incomplete / not
/// applicable, return an error rather than silently running with broken math.
fn apply_rotated_checkpoint(
    model: &mut ModelWeights,
    artifacts: &NativeModelArtifacts,
) -> Result<(), WeightLoadError> {
    use crate::weight_rotation::{WeightRotationMode, weight_rotation_mode};
    if weight_rotation_mode() != WeightRotationMode::Apply {
        return Ok(());
    }
    let rotated_path = artifacts.root_dir().join("model.rotated.safetensors");
    if !rotated_path.is_file() {
        return Err(WeightLoadError::RotatedCheckpointInvalid(format!(
            "AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION=apply requires {} (run scripts/quantize_rotated_weights.py --apply first)",
            rotated_path.display()
        )));
    }
    let rotated = load_safetensors(&rotated_path, None).map_err(|e| {
        WeightLoadError::RotatedCheckpointInvalid(format!("{}: {}", rotated_path.display(), e))
    })?;
    if rotated.is_empty() {
        return Err(WeightLoadError::RotatedCheckpointInvalid(format!(
            "{} is empty",
            rotated_path.display()
        )));
    }

    let mut replaced = 0usize;
    let mut smoothing_loaded = 0usize;
    let mut missing_layers: Vec<usize> = Vec::new();
    for (layer_idx, layer) in model.layers.iter_mut().enumerate() {
        // Per-layer AWQ-lite smoothing vector, if the offline tool produced
        // one. Stored as `ax_smoothing.layers.{idx}` of shape [hidden_size]
        // holding 1/s already (so the forward path multiplies, not divides).
        let smoothing_key = format!("ax_smoothing.layers.{}", layer_idx);
        if let Some(smoothing) = rotated.get(&smoothing_key) {
            layer.rotation_smoothing_inverse = Some(smoothing.clone());
            smoothing_loaded += 1;
        }
        for (suffix, slot) in [
            ("gate_proj", &mut layer.gate_proj),
            ("up_proj", &mut layer.up_proj),
        ] {
            let Some(target) = slot.as_mut() else {
                continue;
            };
            // Standard MLX-community Qwen naming. Other families would need
            // alternative key probes added here as P2a expands beyond Qwen.
            let key = format!(
                "language_model.model.layers.{}.mlp.{}.weight",
                layer_idx, suffix
            );
            let Some(rotated_w) = rotated.get(&key) else {
                // `target` (from `slot.as_mut()` above) is already `Some`,
                // so this layer is rotation-eligible regardless of whether
                // its gate/up got packed into `gate_up_packed` — track every
                // eligible-but-missing layer, not just the unpacked case,
                // or a partial checkpoint silently corrupts the packed
                // (i.e. the common dense-FFN) layers instead of failing.
                missing_layers.push(layer_idx);
                continue;
            };
            // Rotated tensors are EITHER stored as f32 (P2a baseline) or
            // re-quantized as u32-packed plus sibling .scales / .biases (P2b).
            // Detect by weight dtype: u32 => quantized path; anything else
            // => plain f32 path with cast to bf16.
            let stem = format!("language_model.model.layers.{}.mlp.{}", layer_idx, suffix);
            let scales_key = format!("{stem}.scales");
            let biases_key = format!("{stem}.biases");
            if rotated_w.dtype() == MlxDtype::Uint32 {
                let scales = rotated.get(&scales_key).ok_or_else(|| {
                    WeightLoadError::RotatedCheckpointInvalid(format!(
                        "missing {scales_key} for u32-packed rotated weight"
                    ))
                })?;
                let biases = rotated.get(&biases_key).ok_or_else(|| {
                    WeightLoadError::RotatedCheckpointInvalid(format!(
                        "missing {biases_key} for u32-packed rotated weight"
                    ))
                })?;
                // Infer bits from packed shape vs logical dim. For u32 packing
                // with `bits` bits per element: packed_inner = logical_inner * bits / 32.
                // logical_inner is the dim the activation rotation acts on; we
                // know the original weight's logical inner was rotation_dim,
                // which equals `original.weight` last-axis * 8 (4-bit packed)
                // or equivalent. Compute from this checkpoint's shape directly.
                let packed_shape = rotated_w.shape();
                let scales_shape = scales.shape();
                // logical_inner = group_size * groups_per_row, with groups_per_row
                // = scales last axis. Hardcoded group_size 64 (matches script).
                let groups_per_row = *scales_shape.last().unwrap_or(&0) as i64;
                let logical_inner = groups_per_row * 64;
                let packed_inner = *packed_shape.last().unwrap_or(&0) as i64;
                if logical_inner <= 0 || packed_inner <= 0 {
                    return Err(WeightLoadError::RotatedCheckpointInvalid(format!(
                        "{key}: cannot infer bits from shapes packed={packed_shape:?} scales={scales_shape:?}"
                    )));
                }
                let bits_calc = packed_inner * 32 / logical_inner;
                if !(2..=8).contains(&bits_calc) {
                    return Err(WeightLoadError::RotatedCheckpointInvalid(format!(
                        "{key}: inferred bits={bits_calc} outside 2..=8"
                    )));
                }
                target.weight = rotated_w.clone();
                target.scales = Some(scales.clone());
                target.biases = Some(biases.clone());
                target.bits = bits_calc as i32;
                target.group_size = 64;
            } else {
                // f32 path: cast to bf16, drop scales/biases, forward picks
                // the plain matmul branch.
                let cast = astype(rotated_w, MlxDtype::Bfloat16, None);
                target.weight = cast;
                target.scales = None;
                target.biases = None;
            }
            replaced += 1;
        }
        if layer.gate_up_packed.is_some()
            && let (Some(gate), Some(up)) = (&layer.gate_proj, &layer.up_proj)
        {
            layer.gate_up_packed = Some(pack_dense_ffn_gate_up_projection(gate, up)?);
        }
    }

    if replaced == 0 {
        return Err(WeightLoadError::RotatedCheckpointInvalid(format!(
            "{} contained 0 matching tensors for this model (key format mismatch?)",
            rotated_path.display()
        )));
    }
    // Apply mode rotates every eligible layer's activations unconditionally
    // (`maybe_apply_rotation_identity` in model/shared/mlp.rs has no
    // per-layer knowledge of checkpoint completeness): a layer whose weight
    // was NOT found above still gets `x @ R` applied to its input while
    // keeping its original, un-rotated `W`, producing `(x @ R) @ W^T`
    // instead of `x @ W^T` — not merely unrotated, but an actively wrong
    // orthogonal transform baked into that layer's output. A partial
    // checkpoint (e.g. `--max-tensors N` from a first-run validation pass)
    // must fail closed here rather than silently corrupting those layers.
    if !missing_layers.is_empty() {
        return Err(WeightLoadError::RotatedCheckpointInvalid(format!(
            "{} is missing rotated gate/up projections for layers {:?}; Apply mode requires \
             every eligible layer to be present, or activation-side rotation will be applied \
             against un-rotated weights for those layers",
            rotated_path.display(),
            missing_layers
        )));
    }

    tracing::info!(
        target: "ax_mlx::weight_rotation",
        replaced_tensor_count = replaced,
        smoothing_loaded = smoothing_loaded,
        rotated_checkpoint = %rotated_path.display(),
        "loaded rotated checkpoint"
    );
    eprintln!(
        "[ax_mlx::weight_rotation apply] loaded {}: {} tensors replaced, {} layers with smoothing",
        rotated_path.display(),
        replaced,
        smoothing_loaded,
    );
    Ok(())
}

/// Apply the `hf_to_mlx` weight transforms in place on a freshly loaded
/// safetensors `name_map`.
///
/// Raw HuggingFace checkpoints store ordinary RMSNorm weights as zero-centered
/// deltas (the "+1.0" is folded into the model's runtime forward pass). MLX
/// expects the weight to already be the multiplier. We add 1.0 only to the
/// RMSNorm roles that match mlx_lm's sanitize convention. Linear-attention gated
/// norms are excluded because Qwen3-Next consumes those trained scales directly.
///
/// HF also stores conv1d projection weights with axes (out, in, kernel)
/// while MLX expects (out, kernel, in). The `transpose(_, [0, 2, 1])` swap
/// brings them into MLX layout.
///
/// The companion `ensure_conv1d_mlx_layout` check downstream remains as the
/// safety net for the `WeightSanitize::None` path where a manifest mis-declares
/// raw HF conv1d layout.
fn apply_hf_sanitize_transforms(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    swap_conv1d: bool,
) {
    let one = MlxArray::from_f32_slice(&[1.0_f32]);
    for spec in specs {
        if !name_map.contains_key(&spec.name) {
            continue;
        }
        let transformed = match spec.role {
            role if is_hf_rmsnorm_lift_role(role) => {
                let tensor = name_map.get(&spec.name).expect("checked via contains_key");
                // MLX promotes (bf16/f16 + f32) to f32, which would silently
                // change the stored dtype of the norm weight. Cast back to the
                // original dtype so downstream consumers see the same shape
                // and dtype they would have for an already-sanitized
                // mlx-community weight.
                let original_dtype = tensor.dtype();
                Some(astype(&add(tensor, &one, None), original_dtype, None))
            }
            NativeTensorRole::LinearAttentionConv1d if swap_conv1d => {
                let tensor = name_map.get(&spec.name).expect("checked via contains_key");
                // `transpose` returns a stride-only view; consumers that read
                // the raw buffer (e.g. the conv1d kernel) need a contiguous
                // layout, so materialize here.
                Some(contiguous(&transpose(tensor, &[0, 2, 1], None), None))
            }
            _ => None,
        };
        if let Some(new_tensor) = transformed {
            name_map.insert(spec.name.clone(), new_tensor);
        }
    }
    // Force evaluation so subsequent inspection (e.g.
    // ensure_sanitized_linear_attention_norm) sees materialised values rather
    // than the lazy MLX op graph.
    let refs: Vec<&MlxArray> = name_map.values().collect();
    eval(&refs);
}

fn is_hf_rmsnorm_lift_role(role: NativeTensorRole) -> bool {
    matches!(
        role,
        NativeTensorRole::AttentionNorm
            | NativeTensorRole::AttentionPostNorm
            | NativeTensorRole::AttentionQNorm
            | NativeTensorRole::AttentionKNorm
            | NativeTensorRole::AttentionQaNorm
            | NativeTensorRole::AttentionKvANorm
            | NativeTensorRole::FfnNorm
            | NativeTensorRole::FfnNorm2
            | NativeTensorRole::FfnPostNorm
            | NativeTensorRole::FfnPostNorm1
            | NativeTensorRole::FfnPostNorm2
            | NativeTensorRole::PerLayerProjectionNorm
            | NativeTensorRole::PerLayerInputPostNorm
            | NativeTensorRole::DiffusionSelfConditionPreNorm
            | NativeTensorRole::FinalNorm
    )
}

/// Lower bound on `mean_abs` of a sanitized RMSNorm weight (post `+1.0` lift).
///
/// mlx_lm-sanitized norms cluster tightly around `1.0` (typically `1.0 ± 0.1`);
/// the raw HF zero-centred-delta form clusters around `0.0` (typically `< 0.05`).
/// `0.15` is comfortably between both modes — well above the deltas yet far
/// below any plausible sanitized weight, so it doubles as the auto-detect
/// trigger and the post-load fail-closed assertion.
const SANITIZED_NORM_MIN_MEAN_ABS: f32 = 0.15;

/// Compute mean_abs of a norm tensor as f32, or `None` if the sample is too
/// small to draw a conclusion. Shared by load-time auto-detection and the
/// post-sanitize verification.
fn norm_mean_abs(norm: &MlxArray) -> Option<f32> {
    let f32_norm = astype(norm, MlxDtype::Float32, None);
    eval(&[&f32_norm]);
    let data = f32_norm.data_f32();
    if data.len() < 8 {
        return None;
    }
    Some(data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32)
}

/// When the manifest sets `weight_sanitize=None`, peek at the lowest-indexed
/// block-level RMSNorm tensor (and, for hybrid models, the conv1d layout) to
/// decide whether the weights on disk are actually pre-sanitized.
///
/// Returns the sanitize mode to apply:
/// - `None`: weights look sanitized (ordinary norm baseline near 1.0, conv1d in MLX layout)
/// - `HfNormOnly`: ordinary norm needs +1.0 but conv1d is already MLX layout (mlx-community
///   quantized hybrid models: Qwen3-Coder-Next-4bit, Qwen3.5-9B-4bit, …) or the model
///   has no conv1d at all (raw HF Gemma checkpoints)
/// - `HfToMlx`: ordinary norm and conv1d are raw HF (rare for distributed mlx checkpoints)
///
/// The norm sample is restricted to block-level roles (`AttentionNorm` /
/// `FfnNorm` / `FinalNorm`): per-layer norms exist in every supported dense
/// and hybrid family, so raw zero-centered HF deltas (mean_abs near 0) are
/// detected uniformly. Family-specific adapter norms (MLA q/kv layernorms,
/// per-layer projection norms, gated linear-attention scales) are excluded —
/// several of those legitimately cluster near zero even in fully sanitized
/// checkpoints, so sampling them would false-positive.
///
/// The +1.0 norm lift only exists for two checkpoint conventions: hybrid
/// linear-attention families (mlx-community quantized hybrids ship raw norms
/// on disk) and Gemma families (HF stores zero-centered gamma deltas). For
/// every other dense family (Qwen, Llama, Mistral, …) HF and MLX store the
/// same trained RMSNorm weights, which legitimately cluster near zero — a
/// small mean_abs there carries no signal, so the probe is skipped entirely
/// rather than risking a false-positive double lift.
///
/// The conv1d layout check only applies to hybrid linear-attention models:
/// dense manifests carry no `LinearAttentionConv1d` specs, so the probe is
/// structurally a no-op for them.
fn auto_detect_weight_sanitize(
    model_family: &str,
    specs: &[NativeTensorSpec],
    name_map: &HashMap<String, MlxArray>,
) -> WeightSanitize {
    let has_linear_attention = specs
        .iter()
        .any(|s| matches!(s.role, NativeTensorRole::LinearAttentionConv1d));
    if !has_linear_attention && !model_family.contains("gemma") {
        return WeightSanitize::None;
    }
    // Sample the lowest-indexed block-level RMSNorm rather than adapter norms:
    // Qwen3-Next's linear_attn.norm is a gated scale consumed raw by mlx_lm,
    // and MLA q/kv adapter norms legitimately have small mean_abs — both can
    // sit near zero in fully sanitized checkpoints.
    let norm_spec = specs
        .iter()
        .filter(|s| {
            matches!(
                s.role,
                NativeTensorRole::AttentionNorm
                    | NativeTensorRole::FfnNorm
                    | NativeTensorRole::FinalNorm
            )
        })
        .min_by_key(|s| s.layer_index.unwrap_or(u32::MAX));
    let Some(norm_spec) = norm_spec else {
        return WeightSanitize::None;
    };
    let Some(norm) = name_map.get(&norm_spec.name) else {
        return WeightSanitize::None;
    };
    let Some(mean_abs) = norm_mean_abs(norm) else {
        return WeightSanitize::None;
    };
    let norm_needs_sanitize = mean_abs < SANITIZED_NORM_MIN_MEAN_ABS;

    let sample_layer = norm_spec.layer_index;
    let conv1d_spec = specs
        .iter()
        .find(|s| {
            matches!(s.role, NativeTensorRole::LinearAttentionConv1d)
                && s.layer_index == sample_layer
        })
        .or_else(|| {
            specs
                .iter()
                .filter(|s| matches!(s.role, NativeTensorRole::LinearAttentionConv1d))
                .min_by_key(|s| s.layer_index.unwrap_or(u32::MAX))
        });
    let conv1d_needs_swap = conv1d_spec
        .and_then(|spec| name_map.get(&spec.name))
        .map(|conv1d| {
            let shape = conv1d.shape();
            // HF layout: [conv_dim, in=1, kernel]. MLX layout: [conv_dim, kernel, in=1].
            // Detect HF by `shape[1] == 1 && shape[2] != 1` to avoid the
            // ambiguous all-ones edge case (a 1x1 conv has shape [_, 1, 1]).
            shape.len() == 3 && shape[1] == 1 && shape[2] != 1
        })
        .unwrap_or(false);

    let chosen = match (norm_needs_sanitize, conv1d_needs_swap) {
        (false, false) => WeightSanitize::None,
        (true, false) => WeightSanitize::HfNormOnly,
        (true, true) => WeightSanitize::HfToMlx,
        // Norm looks sanitized but conv1d is still HF layout. A partially
        // transformed checkpoint should not be silently patched — let the
        // downstream `ensure_conv1d_mlx_layout` check fire with its
        // diagnostic.
        (false, true) => WeightSanitize::None,
    };

    if !matches!(chosen, WeightSanitize::None) {
        tracing::warn!(
            target: "ax_mlx::weights",
            mean_abs = mean_abs,
            conv1d_needs_swap = conv1d_needs_swap,
            sanitize = ?chosen,
            "manifest weight_sanitize=None but on-disk weights look unsanitized; \
             applying {chosen:?}. Set weight_sanitize explicitly in model-manifest.json to silence \
             this warning."
        );
        eprintln!(
            "[ax_mlx::weights] auto-detected unsanitized weights \
             (norm mean_abs={mean_abs:.6}, conv1d_hf_layout={conv1d_needs_swap}); \
             applying {chosen:?} sanitize transform. Set weight_sanitize in \
             model-manifest.json to silence this warning."
        );
    }

    chosen
}

/// Resolve the sanitize transform to apply for a loaded `name_map`.
///
/// An explicit manifest `weight_sanitize` always wins; when it conflicts with
/// what the on-disk weights look like (e.g. manifest declares `HfNormOnly`
/// but the norms are still raw HF deltas *and* the conv1d is in HF layout),
/// log a warning but honor the manifest. `None` delegates to
/// `auto_detect_weight_sanitize`.
fn effective_weight_sanitize(
    model_family: &str,
    manifest_mode: WeightSanitize,
    specs: &[NativeTensorSpec],
    name_map: &HashMap<String, MlxArray>,
) -> WeightSanitize {
    match manifest_mode {
        WeightSanitize::None => auto_detect_weight_sanitize(model_family, specs, name_map),
        explicit => {
            let detected = auto_detect_weight_sanitize(model_family, specs, name_map);
            if !matches!(detected, WeightSanitize::None) && detected != explicit {
                tracing::warn!(
                    target: "ax_mlx::weights",
                    manifest = ?explicit,
                    detected = ?detected,
                    "manifest weight_sanitize conflicts with on-disk weight probe; \
                     honoring the manifest setting"
                );
            }
            explicit
        }
    }
}

/// Verify conv1d is in MLX layout `[conv_dim, kernel, 1]` (last dim = 1).
///
/// Catches manifests that mis-declare `weight_sanitize` and leave conv1d in
/// HuggingFace layout `[conv_dim, 1, kernel]`, which would produce silently
/// wrong conv outputs without any other runtime error.
fn ensure_conv1d_mlx_layout(layer_index: usize, conv1d: &MlxArray) -> Result<(), WeightLoadError> {
    let shape = conv1d.shape();
    if shape.len() != 3 || shape[2] != 1 {
        return Err(WeightLoadError::UnsanitizedWeights(format!(
            "linear attention layer {layer_index} conv1d shape {shape:?}: expected \
             [conv_dim, kernel, 1] (MLX layout). Raw HuggingFace checkpoints store conv1d as \
             [conv_dim, 1, kernel]; set weight_sanitize to \"hf_to_mlx\" or run \
             mlx_lm.convert on the unquantized source weights first \
             (re-running mlx_lm.convert on an already-quantized checkpoint corrupts the weights)."
        )));
    }
    Ok(())
}

fn gemma4_router_combined_scale(hidden_size: u32, router_scale: &MlxArray) -> MlxArray {
    let root_factor = 1.0_f32 / (hidden_size as f32).sqrt();
    let scale_arr = MlxArray::from_raw_data(
        &root_factor as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    );
    let scale_arr = astype(&scale_arr, MlxDtype::Bfloat16, None);
    multiply(router_scale, &scale_arr, None)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttentionLayout {
    Full,
    Linear,
    /// MoE-only or dense-MLP residual mixer with no attention (Nemotron-H `E`/`-`).
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FullAttentionProjectionLayout {
    /// GLM4MoELite MLA attention uses q_a/q_b + latent KV projections.
    GlmMla,
    /// KV-shared layers compute only Q and reuse a source layer's K/V.
    QOnly,
    PackedQkv,
    /// K=V full-attention layers compute Q/K and reuse K as V (Gemma4 full attention).
    SplitQkValueFromKey,
    SplitQkv,
}

fn attention_layout_for_layer(
    specs: &[NativeTensorSpec],
    layer_index: Option<u32>,
) -> Result<AttentionLayout, WeightLoadError> {
    let has_full = has_full_attention_role(specs, layer_index);
    let has_linear = has_linear_attention_role(specs, layer_index);
    let has_moe = has_role(specs, NativeTensorRole::FfnGateInp, layer_index)
        || has_role(specs, NativeTensorRole::FfnUpExps, layer_index)
        || has_role(specs, NativeTensorRole::FfnDown, layer_index);

    if has_full && has_linear {
        return Err(WeightLoadError::InvalidLayer(format!(
            "layer {layer_index:?} mixes full-attention and linear-attention tensor roles"
        )));
    }
    if has_linear {
        Ok(AttentionLayout::Linear)
    } else if has_full {
        Ok(AttentionLayout::Full)
    } else if has_moe {
        // Nemotron-H MoE / dense MLP residual mixers have no attention tensors.
        Ok(AttentionLayout::None)
    } else {
        Ok(AttentionLayout::Full)
    }
}

fn full_attention_projection_layout(
    specs: &[NativeTensorSpec],
    layer_index: Option<u32>,
    uses_shared_kv: bool,
    uses_value_from_key: bool,
) -> Result<FullAttentionProjectionLayout, WeightLoadError> {
    let has_glm_mla = has_glm_mla_attention_role(specs, layer_index);
    let has_standard_full = has_standard_full_attention_projection_role(specs, layer_index);
    if has_glm_mla {
        if uses_shared_kv || uses_value_from_key || has_standard_full {
            return Err(WeightLoadError::InvalidLayer(format!(
                "layer {layer_index:?} mixes GLM MLA attention with standard full-attention layout"
            )));
        }
        return Ok(FullAttentionProjectionLayout::GlmMla);
    }

    let has_packed = has_role(specs, NativeTensorRole::AttentionQkvPacked, layer_index);
    if uses_shared_kv {
        if has_packed {
            return Err(WeightLoadError::InvalidLayer(format!(
                "layer {layer_index:?} is KV-shared but provides packed QKV weights"
            )));
        }
        return Ok(FullAttentionProjectionLayout::QOnly);
    }
    if has_packed {
        if uses_value_from_key {
            return Err(WeightLoadError::InvalidLayer(format!(
                "layer {layer_index:?} is marked value-from-key but provides packed QKV weights"
            )));
        }
        Ok(FullAttentionProjectionLayout::PackedQkv)
    } else if uses_value_from_key {
        Ok(FullAttentionProjectionLayout::SplitQkValueFromKey)
    } else {
        Ok(FullAttentionProjectionLayout::SplitQkv)
    }
}

fn has_full_attention_role(specs: &[NativeTensorSpec], layer_index: Option<u32>) -> bool {
    [
        NativeTensorRole::AttentionO,
        NativeTensorRole::AttentionQ,
        NativeTensorRole::AttentionK,
        NativeTensorRole::AttentionV,
        NativeTensorRole::AttentionQkvPacked,
        NativeTensorRole::AttentionQa,
        NativeTensorRole::AttentionQaNorm,
        NativeTensorRole::AttentionQb,
        NativeTensorRole::AttentionKvA,
        NativeTensorRole::AttentionKvB,
        NativeTensorRole::AttentionKvANorm,
        NativeTensorRole::AttentionEmbedQ,
        NativeTensorRole::AttentionUnembedOut,
    ]
    .into_iter()
    .any(|role| has_role(specs, role, layer_index))
}

fn has_standard_full_attention_projection_role(
    specs: &[NativeTensorSpec],
    layer_index: Option<u32>,
) -> bool {
    [
        NativeTensorRole::AttentionQ,
        NativeTensorRole::AttentionK,
        NativeTensorRole::AttentionV,
        NativeTensorRole::AttentionQkvPacked,
    ]
    .into_iter()
    .any(|role| has_role(specs, role, layer_index))
}

fn has_glm_mla_attention_role(specs: &[NativeTensorSpec], layer_index: Option<u32>) -> bool {
    [
        NativeTensorRole::AttentionQa,
        NativeTensorRole::AttentionQaNorm,
        NativeTensorRole::AttentionQb,
        NativeTensorRole::AttentionKvA,
        NativeTensorRole::AttentionKvB,
        NativeTensorRole::AttentionKvANorm,
        NativeTensorRole::AttentionEmbedQ,
        NativeTensorRole::AttentionUnembedOut,
    ]
    .into_iter()
    .any(|role| has_role(specs, role, layer_index))
}

fn has_linear_attention_role(specs: &[NativeTensorSpec], layer_index: Option<u32>) -> bool {
    [
        NativeTensorRole::LinearAttentionInProjQkv,
        NativeTensorRole::LinearAttentionInProjQkvz,
        NativeTensorRole::LinearAttentionInProjZ,
        NativeTensorRole::LinearAttentionInProjA,
        NativeTensorRole::LinearAttentionInProjB,
        NativeTensorRole::LinearAttentionInProjBa,
        NativeTensorRole::LinearAttentionConv1d,
        NativeTensorRole::LinearAttentionDtBias,
        NativeTensorRole::LinearAttentionALog,
        NativeTensorRole::LinearAttentionNorm,
        NativeTensorRole::LinearAttentionOutProj,
    ]
    .into_iter()
    .any(|role| has_role(specs, role, layer_index))
}

fn load_linear_attention_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer_index: Option<u32>,
    config: &NativeLinearAttentionConfig,
) -> Result<LinearAttentionWeights, WeightLoadError> {
    let mut in_proj_qkv = try_take_weight(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionInProjQkv,
        layer_index,
        "linear_attention_in_proj_qkv",
    )?;
    let mut in_proj_z = try_take_weight(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionInProjZ,
        layer_index,
        "linear_attention_in_proj_z",
    )?;
    let mut in_proj_a = try_take_weight(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionInProjA,
        layer_index,
        "linear_attention_in_proj_a",
    )?;
    let mut in_proj_b = try_take_weight(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionInProjB,
        layer_index,
        "linear_attention_in_proj_b",
    )?;
    let mut in_proj_qkvz = try_take_weight(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionInProjQkvz,
        layer_index,
        "linear_attention_in_proj_qkvz",
    )?;
    let mut in_proj_ba = try_take_weight(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionInProjBa,
        layer_index,
        "linear_attention_in_proj_ba",
    )?;

    if linear_attention_projection_packing_enabled()
        && in_proj_qkvz.is_none()
        && in_proj_ba.is_none()
        && let (Some(qkv), Some(z), Some(a), Some(b)) =
            (&in_proj_qkv, &in_proj_z, &in_proj_a, &in_proj_b)
        && linear_attention_projection_packing_supported(qkv, z, a, b)
    {
        let (qkvz, ba) = pack_split_linear_attention_projections(config, qkv, z, a, b)?;
        in_proj_qkvz = Some(qkvz);
        in_proj_ba = Some(ba);
        in_proj_qkv = None;
        in_proj_z = None;
        in_proj_a = None;
        in_proj_b = None;
    }

    let conv1d_raw = take_weight(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionConv1d,
        layer_index,
        "linear_attention_conv1d",
    )?;
    let conv1d_dense = if let Some(scales) = &conv1d_raw.scales {
        dequantize(
            &conv1d_raw.weight,
            scales,
            conv1d_raw.biases.as_ref(),
            Some(conv1d_raw.group_size),
            Some(conv1d_raw.bits),
            None,
        )
    } else {
        conv1d_raw.weight
    };
    // Nemotron-H Mamba-2 uses conv1d bias (`use_conv_bias=true`). `take_weight`
    // already lifts `.bias` (singular) into `QuantizedWeight.linear_bias` when
    // co-located with `.weight` — do not re-fetch by name (it would be gone).
    let conv1d_bias = conv1d_raw.linear_bias.or_else(|| {
        layer_index.and_then(|li| {
            let candidates = [
                format!("backbone.layers.{li}.mixer.conv1d.bias"),
                format!("model.layers.{li}.linear_attn.conv1d.bias"),
            ];
            candidates
                .into_iter()
                .find_map(|name| name_map.remove(&name))
        })
    });
    // Mamba-2 D residual (mapped to LayerScalar at convert time).
    let d = try_take_plain(specs, name_map, NativeTensorRole::LayerScalar, layer_index)?
        .map(|arr| astype(&arr, MlxDtype::Float32, None));

    let mut linear_attn = LinearAttentionWeights {
        in_proj_qkv,
        in_proj_z,
        in_proj_a,
        in_proj_b,
        in_proj_qkvz,
        in_proj_ba,
        fused_qkvz_ba: None,
        prefill_q2_qkvz: None,
        prefill_q2_ba: None,
        conv1d_dense,
        conv1d_bias,
        // Cast at load time so the per-step linear_attention_forward does not
        // pay an astype dispatch for each layer. `gated_delta_kernel` expects
        // both as f32, matching mlx_lm's reference behaviour. For a 12-layer
        // hybrid model this removes ~24 small astype ops per decode step.
        dt_bias: astype(
            &take_weight(
                specs,
                name_map,
                NativeTensorRole::LinearAttentionDtBias,
                layer_index,
                "linear_attention_dt_bias",
            )?
            .weight,
            MlxDtype::Float32,
            None,
        ),
        a_log: astype(
            &take_weight(
                specs,
                name_map,
                NativeTensorRole::LinearAttentionALog,
                layer_index,
                "linear_attention_a_log",
            )?
            .weight,
            MlxDtype::Float32,
            None,
        ),
        d,
        norm: take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionNorm,
            layer_index,
            "linear_attention_norm",
        )?
        .weight,
        out_proj: take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionOutProj,
            layer_index,
            "linear_attention_out_proj",
        )?,
    };
    linear_attn.prepare_fused_qkvz_ba_prefill();
    linear_attn.prepare_prefill_q2_projections();
    Ok(linear_attn)
}

fn load_glm_mla_attention_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer_index: Option<u32>,
    mla_attention: &NativeMlaAttentionConfig,
    attention_head_count: u32,
) -> Result<GlmMlaAttentionWeights, WeightLoadError> {
    let q_a_proj = take_weight(
        specs,
        name_map,
        NativeTensorRole::AttentionQa,
        layer_index,
        "glm_q_a_proj",
    )?;
    let kv_a_proj = take_weight(
        specs,
        name_map,
        NativeTensorRole::AttentionKvA,
        layer_index,
        "glm_kv_a_proj",
    )?;
    let (embed_q, unembed_out) = load_mla_kv_b_weights(
        specs,
        name_map,
        layer_index,
        mla_attention,
        attention_head_count,
    )?;
    Ok(GlmMlaAttentionWeights {
        qa_kva_fused: pack_glm_mla_qa_kva_projection(&q_a_proj, &kv_a_proj)?,
        q_a_norm: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionQaNorm,
            layer_index,
            "glm_q_a_norm",
        )?
        .weight,
        q_b_proj: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionQb,
            layer_index,
            "glm_q_b_proj",
        )?,
        kv_a_norm: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionKvANorm,
            layer_index,
            "glm_kv_a_norm",
        )?
        .weight,
        embed_q,
        unembed_out,
    })
}

/// Load one DeepSeek V4 (Flash) layer's attention + hyper-connection tensors.
///
/// V4 reuses the `AttentionQa`/`AttentionQaNorm`/`AttentionQb` roles but
/// replaces the V3 MLA kv_a/kv_b pair with a fused `AttentionKv` projection
/// and a grouped `AttentionOutA`/`AttentionOutB` output LoRA, so it must NOT
/// route through `load_glm_mla_attention_weights` /
/// `split_deepseek_kv_b_projection` (which hard-require V3 MLA dims).
fn load_deepseek_v4_layer_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer_index: Option<u32>,
) -> Result<DeepseekV4LayerWeights, WeightLoadError> {
    let compressor = if has_role(specs, NativeTensorRole::CompressorKv, layer_index) {
        Some(DeepseekV4CompressorWeights {
            kv: take_weight(
                specs,
                name_map,
                NativeTensorRole::CompressorKv,
                layer_index,
                "dsv4_compressor_kv",
            )?,
            gate: take_weight(
                specs,
                name_map,
                NativeTensorRole::CompressorGate,
                layer_index,
                "dsv4_compressor_gate",
            )?,
            ape: take_weight(
                specs,
                name_map,
                NativeTensorRole::CompressorApe,
                layer_index,
                "dsv4_compressor_ape",
            )?
            .weight,
            norm: take_weight(
                specs,
                name_map,
                NativeTensorRole::CompressorNorm,
                layer_index,
                "dsv4_compressor_norm",
            )?
            .weight,
        })
    } else {
        None
    };
    let indexer = if has_role(specs, NativeTensorRole::IndexerProj, layer_index) {
        Some(DeepseekV4IndexerWeights {
            proj: take_weight(
                specs,
                name_map,
                NativeTensorRole::IndexerProj,
                layer_index,
                "dsv4_indexer_proj",
            )?,
            qb: take_weight(
                specs,
                name_map,
                NativeTensorRole::IndexerQb,
                layer_index,
                "dsv4_indexer_qb",
            )?,
            compressor_kv: take_weight(
                specs,
                name_map,
                NativeTensorRole::IndexerCompressorKv,
                layer_index,
                "dsv4_indexer_compressor_kv",
            )?,
            compressor_gate: take_weight(
                specs,
                name_map,
                NativeTensorRole::IndexerCompressorGate,
                layer_index,
                "dsv4_indexer_compressor_gate",
            )?,
            compressor_ape: take_weight(
                specs,
                name_map,
                NativeTensorRole::IndexerCompressorApe,
                layer_index,
                "dsv4_indexer_compressor_ape",
            )?
            .weight,
            compressor_norm: take_weight(
                specs,
                name_map,
                NativeTensorRole::IndexerCompressorNorm,
                layer_index,
                "dsv4_indexer_compressor_norm",
            )?
            .weight,
        })
    } else {
        None
    };
    Ok(DeepseekV4LayerWeights {
        wq_a: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionQa,
            layer_index,
            "dsv4_wq_a",
        )?,
        q_a_norm: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionQaNorm,
            layer_index,
            "dsv4_q_a_norm",
        )?
        .weight,
        wq_b: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionQb,
            layer_index,
            "dsv4_wq_b",
        )?,
        wkv: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionKv,
            layer_index,
            "dsv4_wkv",
        )?,
        kv_norm: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionKvNorm,
            layer_index,
            "dsv4_kv_norm",
        )?
        .weight,
        wo_a: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionOutA,
            layer_index,
            "dsv4_wo_a",
        )?,
        wo_b: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionOutB,
            layer_index,
            "dsv4_wo_b",
        )?,
        attn_sink: try_take_plain(specs, name_map, NativeTensorRole::AttnSink, layer_index)?,
        hc_attn_fn: take_weight(
            specs,
            name_map,
            NativeTensorRole::HcAttnFn,
            layer_index,
            "dsv4_hc_attn_fn",
        )?
        .weight,
        hc_attn_base: take_weight(
            specs,
            name_map,
            NativeTensorRole::HcAttnBase,
            layer_index,
            "dsv4_hc_attn_base",
        )?
        .weight,
        hc_attn_scale: take_weight(
            specs,
            name_map,
            NativeTensorRole::HcAttnScale,
            layer_index,
            "dsv4_hc_attn_scale",
        )?
        .weight,
        hc_ffn_fn: take_weight(
            specs,
            name_map,
            NativeTensorRole::HcFfnFn,
            layer_index,
            "dsv4_hc_ffn_fn",
        )?
        .weight,
        hc_ffn_base: take_weight(
            specs,
            name_map,
            NativeTensorRole::HcFfnBase,
            layer_index,
            "dsv4_hc_ffn_base",
        )?
        .weight,
        hc_ffn_scale: take_weight(
            specs,
            name_map,
            NativeTensorRole::HcFfnScale,
            layer_index,
            "dsv4_hc_ffn_scale",
        )?
        .weight,
        compressor,
        indexer,
        tid2eid: try_take_plain(
            specs,
            name_map,
            NativeTensorRole::FfnGateTid2Eid,
            layer_index,
        )?,
    })
}

/// Load the root-level DeepSeek V4 hyper-connection head (`hc_head_*`).
fn load_deepseek_v4_head_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
) -> Result<Option<DeepseekV4HeadWeights>, WeightLoadError> {
    if !has_role(specs, NativeTensorRole::HcHeadFn, None) {
        return Ok(None);
    }
    Ok(Some(DeepseekV4HeadWeights {
        hc_head_fn: take_weight(
            specs,
            name_map,
            NativeTensorRole::HcHeadFn,
            None,
            "dsv4_hc_head_fn",
        )?
        .weight,
        hc_head_base: take_weight(
            specs,
            name_map,
            NativeTensorRole::HcHeadBase,
            None,
            "dsv4_hc_head_base",
        )?
        .weight,
        hc_head_scale: take_weight(
            specs,
            name_map,
            NativeTensorRole::HcHeadScale,
            None,
            "dsv4_hc_head_scale",
        )?
        .weight,
    }))
}

/// Load any DeepSeek V4 MTP (nextn) predictor tensors present in the
/// manifest. All roles are optional; returns `None` when none exist.
/// `block_layer` is the nextn transformer block detached from the layer loop
/// (GGUF-layout manifests carry it at layer index `layer_count`); the block
/// must never be hash-routed (llama.cpp asserts MTP layers sit beyond the
/// hash layers), so a `tid2eid` table on it is a hard error.
fn load_deepseek_v4_nextn_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    block_layer: Option<LayerWeights>,
) -> Result<Option<DeepseekV4NextnWeights>, WeightLoadError> {
    if let Some(layer) = block_layer.as_ref()
        && let Some(v4) = layer.deepseek_v4.as_ref()
        && v4.tid2eid.is_some()
    {
        return Err(WeightLoadError::InvalidLayer(
            "DeepSeek V4 nextn (MTP) block must not carry a hash-routing tid2eid table".to_string(),
        ));
    }
    let nextn = DeepseekV4NextnWeights {
        e_proj: try_take_weight(
            specs,
            name_map,
            NativeTensorRole::NextnEproj,
            None,
            "dsv4_nextn_e_proj",
        )?,
        h_proj: try_take_weight(
            specs,
            name_map,
            NativeTensorRole::NextnHproj,
            None,
            "dsv4_nextn_h_proj",
        )?,
        eh_proj: try_take_weight(
            specs,
            name_map,
            NativeTensorRole::NextnEhProj,
            None,
            "dsv4_nextn_eh_proj",
        )?,
        enorm: try_take_plain(specs, name_map, NativeTensorRole::NextnEnorm, None)?,
        hnorm: try_take_plain(specs, name_map, NativeTensorRole::NextnHnorm, None)?,
        shared_head_norm: try_take_plain(
            specs,
            name_map,
            NativeTensorRole::NextnSharedHeadNorm,
            None,
        )?,
        embed_tokens: try_take_weight(
            specs,
            name_map,
            NativeTensorRole::NextnEmbedTokens,
            None,
            "dsv4_nextn_embed_tokens",
        )?,
        shared_head_head: try_take_weight(
            specs,
            name_map,
            NativeTensorRole::NextnSharedHeadHead,
            None,
            "dsv4_nextn_shared_head",
        )?,
        hc_head: load_deepseek_v4_nextn_hc_head(specs, name_map)?,
        layer: block_layer.map(Box::new),
    };
    Ok((!nextn.is_empty()).then_some(nextn))
}

/// Load the MTP-specific HC head from dedicated `NextnHcHead*` roles when all
/// three tensors are present. Partial sets fail closed as `None` (draft falls
/// back to the target root head with a one-shot warning at first use).
fn load_deepseek_v4_nextn_hc_head(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
) -> Result<Option<DeepseekV4HeadWeights>, WeightLoadError> {
    let has_fn = has_role(specs, NativeTensorRole::NextnHcHeadFn, None);
    let has_base = has_role(specs, NativeTensorRole::NextnHcHeadBase, None);
    let has_scale = has_role(specs, NativeTensorRole::NextnHcHeadScale, None);
    if !(has_fn || has_base || has_scale) {
        return Ok(None);
    }
    if !(has_fn && has_base && has_scale) {
        tracing::warn!(
            target: "ax_mlx::weights",
            "DeepSeek V4 MTP hc_head_* is incomplete in the manifest — draft will fall back to the target head"
        );
        return Ok(None);
    }
    Ok(Some(DeepseekV4HeadWeights {
        hc_head_fn: take_weight(
            specs,
            name_map,
            NativeTensorRole::NextnHcHeadFn,
            None,
            "dsv4_nextn_hc_head_fn",
        )?
        .weight,
        hc_head_base: take_weight(
            specs,
            name_map,
            NativeTensorRole::NextnHcHeadBase,
            None,
            "dsv4_nextn_hc_head_base",
        )?
        .weight,
        hc_head_scale: take_weight(
            specs,
            name_map,
            NativeTensorRole::NextnHcHeadScale,
            None,
            "dsv4_nextn_hc_head_scale",
        )?
        .weight,
    }))
}

/// Load the DeepSeek V4 MTP sidecar (`mtp.safetensors`) if present alongside
/// the main model. Mirrors `load_glm_mtp_sidecar`: returns `Some` only when
/// the sidecar carries a complete nextn block — input norms, an input
/// projection (fused `eh_proj` or the separate `e_proj`/`h_proj` pair), and
/// one full raw-path V4 layer (no compressor/indexer, learned-router MoE —
/// `tid2eid` is never read here). V4 manifests are gated out of the Qwen and
/// GLM sidecar loaders, so sharing the `mtp.safetensors` filename with the
/// Qwen layout is safe.
///
/// Two on-disk layouts are accepted:
/// - AXQuant (the published `AX-DeepSeek-V4-Flash-MLX-AXQ-*` artifact): all
///   block tensors under `mtp.0.*`; attention LoRA trio, `e_proj`/`h_proj`
///   and shared experts as FP8 blockwise pairs (`{base}.weight` E4M3 bytes +
///   `{base}.scale` E8M0 bytes on a 128×128 block grid), dequantized to dense
///   BF16 at load; routed experts as per-expert MXFP4
///   (`ffn.experts.{N}.w{1,2,3}.{weight,scale}`, "I8" byte payloads nibbled
///   two-per-byte, group_size 32), fused/stacked into `gate_up_exps_packed`
///   + `down_exps`.
/// - Raw-HF fallback: dense BF16 (or MLX-packed affine) tensors via
///   `mtp_take_weight`, with the stacked `ffn.experts.{gate,up,down}` triple.
///
/// `mtplx_runtime.json` is optional; AXQuant's runtime JSON carries no
/// `mtp_sidecar_bits` key, so `bits` stays `None` (shape inference in
/// `mtp_take_weight`) without warning.
fn load_deepseek_v4_mtp_sidecar(
    root: &std::path::Path,
    name_map: &mut HashMap<String, MlxArray>,
    manifest: &ax_engine_core::NativeModelManifest,
) -> Option<DeepseekV4NextnWeights> {
    if !manifest.deepseek_v4.is_enabled() {
        return None;
    }
    let sidecar = root.join("mtp.safetensors");
    if !sidecar.exists() {
        return None;
    }
    // The Rust mmap parser maps FP8 payloads (`F8_E4M3`/`F8_E8M0`) to byte
    // containers; the C `mlx_load_safetensors` rejects `F8_E8M0` scales,
    // which would skip the whole AXQuant sidecar.
    let tensors = match load_safetensors_mmap(&sidecar) {
        Ok(t) => t,
        Err(_) => return None,
    };
    if !tensors.is_empty() {
        let refs: Vec<&MlxArray> = tensors.values().collect();
        eval(&refs);
    }
    name_map.extend(tensors);

    let bits = std::fs::read(root.join("mtplx_runtime.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|v| parse_mtp_sidecar_bits_hint(&v));

    // Tensor prefixes: raw-HF extractions keep the `mtp.N.*` layout; a
    // GGUF-style extraction would keep the block at `layers.<N>.` with the
    // sidecar tensors under `layers.<N>.nextn.`. Both coincide for raw HF.
    let layer_count = manifest.layer_count;
    let bp = [
        "mtp.0".to_string(),
        "mtp.1".to_string(),
        format!("layers.{layer_count}"),
        "nextn".to_string(),
    ]
    .into_iter()
    .find(|p| name_map.contains_key(&format!("{p}.attn.wq_a.weight")));
    let np = [
        "mtp.0".to_string(),
        "mtp.1".to_string(),
        format!("layers.{layer_count}.nextn"),
        "nextn".to_string(),
    ]
    .into_iter()
    .find(|p| name_map.contains_key(&format!("{p}.enorm.weight")));
    let (Some(bp), Some(np)) = (bp, np) else {
        return None;
    };
    let bp = bp.as_str();
    let np = np.as_str();

    // Nextn input/output sidecar tensors. AXQuant sidecars store the
    // projections as FP8 blockwise (`{base}.weight` E4M3 + `{base}.scale`
    // E8M0); raw-HF fallbacks are dense BF16 handled by `mtp_take_weight`.
    let enorm = mtp_take_plain(name_map, &format!("{np}.enorm.weight"));
    let hnorm = mtp_take_plain(name_map, &format!("{np}.hnorm.weight"));
    let eh_proj = mtp_take_weight(name_map, &format!("{np}.eh_proj"), bits);
    let e_proj = mtp_take_fp8_blockscaled(name_map, &format!("{np}.e_proj"))
        .or_else(|| mtp_take_weight(name_map, &format!("{np}.e_proj"), bits));
    let h_proj = mtp_take_fp8_blockscaled(name_map, &format!("{np}.h_proj"))
        .or_else(|| mtp_take_weight(name_map, &format!("{np}.h_proj"), bits));
    let shared_head_norm = mtp_take_plain(name_map, &format!("{np}.shared_head_norm.weight"))
        .or_else(|| mtp_take_plain(name_map, &format!("{np}.norm.weight")))
        .or_else(|| mtp_take_plain(name_map, &format!("{np}.shared_head.norm.weight")));
    let embed_tokens = mtp_take_weight(name_map, &format!("{np}.embed_tokens"), bits);
    let shared_head_head = mtp_take_weight(name_map, &format!("{np}.shared_head_head"), bits)
        .or_else(|| mtp_take_weight(name_map, &format!("{np}.shared_head.head"), bits))
        .or_else(|| mtp_take_weight(name_map, &format!("{np}.head"), bits));
    // MTP HC head (vLLM DeepSeekV4MultiTokenPredictorLayer owns per-layer
    // hc_head_*). Accept both underscore and dotted AXQ naming.
    let hc_head_fn = mtp_take_plain(name_map, &format!("{np}.hc_head_fn"))
        .or_else(|| mtp_take_plain(name_map, &format!("{np}.hc_head.fn")));
    let hc_head_base = mtp_take_plain(name_map, &format!("{np}.hc_head_base"))
        .or_else(|| mtp_take_plain(name_map, &format!("{np}.hc_head.base")));
    let hc_head_scale = mtp_take_plain(name_map, &format!("{np}.hc_head_scale"))
        .or_else(|| mtp_take_plain(name_map, &format!("{np}.hc_head.scale")));
    let hc_head = match (hc_head_fn, hc_head_base, hc_head_scale) {
        (Some(hc_head_fn), Some(hc_head_base), Some(hc_head_scale)) => {
            Some(DeepseekV4HeadWeights {
                hc_head_fn,
                hc_head_base,
                hc_head_scale,
            })
        }
        (None, None, None) => None,
        _ => {
            tracing::warn!(
                target: "ax_mlx::weights",
                "DeepSeek V4 MTP sidecar hc_head_* is incomplete — draft will fall back to the target head"
            );
            None
        }
    };

    // Block: raw-path attention (q LoRA trio + fused KV + grouped output LoRA).
    let attn_norm = mtp_take_plain(name_map, &format!("{bp}.attn_norm.weight"));
    let ffn_norm = mtp_take_plain(name_map, &format!("{bp}.ffn_norm.weight"));
    let wq_a = mtp_take_fp8_blockscaled(name_map, &format!("{bp}.attn.wq_a"))
        .or_else(|| mtp_take_weight(name_map, &format!("{bp}.attn.wq_a"), bits));
    let q_a_norm = mtp_take_plain(name_map, &format!("{bp}.attn.q_norm.weight"));
    let wq_b = mtp_take_fp8_blockscaled(name_map, &format!("{bp}.attn.wq_b"))
        .or_else(|| mtp_take_weight(name_map, &format!("{bp}.attn.wq_b"), bits));
    let wkv = mtp_take_fp8_blockscaled(name_map, &format!("{bp}.attn.wkv"))
        .or_else(|| mtp_take_weight(name_map, &format!("{bp}.attn.wkv"), bits));
    let kv_norm = mtp_take_plain(name_map, &format!("{bp}.attn.kv_norm.weight"));
    // Dense FP8-dequantized `wo_a` keeps the 2-D `[o_groups*o_lora_rank, H*D/o_groups]`
    // layout; the forward's dense branch reshapes it per group itself.
    let wo_a = mtp_take_fp8_blockscaled(name_map, &format!("{bp}.attn.wo_a"))
        .or_else(|| mtp_take_weight(name_map, &format!("{bp}.attn.wo_a"), bits));
    let wo_b = mtp_take_fp8_blockscaled(name_map, &format!("{bp}.attn.wo_b"))
        .or_else(|| mtp_take_weight(name_map, &format!("{bp}.attn.wo_b"), bits));
    let attn_sink = mtp_take_plain(name_map, &format!("{bp}.attn.attn_sink"));

    // Hyper-connection branch parameters (raw `nn.Parameter`s, no `.weight`).
    let hc_attn_fn = mtp_take_plain(name_map, &format!("{bp}.hc_attn_fn"));
    let hc_attn_base = mtp_take_plain(name_map, &format!("{bp}.hc_attn_base"));
    let hc_attn_scale = mtp_take_plain(name_map, &format!("{bp}.hc_attn_scale"));
    let hc_ffn_fn = mtp_take_plain(name_map, &format!("{bp}.hc_ffn_fn"));
    let hc_ffn_base = mtp_take_plain(name_map, &format!("{bp}.hc_ffn_base"));
    let hc_ffn_scale = mtp_take_plain(name_map, &format!("{bp}.hc_ffn_scale"));

    // Learned-router MoE + shared experts (never the hash-routing table).
    let router_proj = mtp_take_weight(name_map, &format!("{bp}.ffn.gate"), bits);
    let router_correction_bias = mtp_take_plain(name_map, &format!("{bp}.ffn.gate.bias"))
        .or_else(|| mtp_take_plain(name_map, &format!("{bp}.ffn.gate.e_score_correction_bias")));
    // Routed experts: AXQuant sidecars store per-expert MXFP4 tensors
    // (`ffn.experts.{N}.w{1,2,3}`); raw-HF fallbacks store the stacked
    // `ffn.experts.{gate,up,down}` triple.
    let mxfp4_experts = manifest
        .moe
        .expert_count
        .and_then(|count| mtp_take_mxfp4_experts(name_map, bp, count));
    let (gate_up_exps_packed, gate_exps, up_exps, down_exps) =
        if let Some((gate_up, down)) = mxfp4_experts {
            (Some(gate_up), None, None, Some(down))
        } else {
            (
                None,
                mtp_take_weight(name_map, &format!("{bp}.ffn.experts.gate"), bits),
                mtp_take_weight(name_map, &format!("{bp}.ffn.experts.up"), bits),
                mtp_take_weight(name_map, &format!("{bp}.ffn.experts.down"), bits),
            )
        };
    let shared_gate_proj =
        mtp_take_fp8_blockscaled(name_map, &format!("{bp}.ffn.shared_experts.w1"))
            .or_else(|| mtp_take_weight(name_map, &format!("{bp}.ffn.shared_experts.w1"), bits));
    let shared_down_proj =
        mtp_take_fp8_blockscaled(name_map, &format!("{bp}.ffn.shared_experts.w2"))
            .or_else(|| mtp_take_weight(name_map, &format!("{bp}.ffn.shared_experts.w2"), bits));
    let shared_up_proj = mtp_take_fp8_blockscaled(name_map, &format!("{bp}.ffn.shared_experts.w3"))
        .or_else(|| mtp_take_weight(name_map, &format!("{bp}.ffn.shared_experts.w3"), bits));

    let experts_complete = down_exps.is_some()
        && (gate_up_exps_packed.is_some() || (gate_exps.is_some() && up_exps.is_some()));
    let complete = enorm.is_some()
        && hnorm.is_some()
        && (eh_proj.is_some() || (e_proj.is_some() && h_proj.is_some()))
        && experts_complete
        && [
            &attn_norm,
            &ffn_norm,
            &q_a_norm,
            &kv_norm,
            &hc_attn_fn,
            &hc_attn_base,
            &hc_attn_scale,
            &hc_ffn_fn,
            &hc_ffn_base,
            &hc_ffn_scale,
        ]
        .into_iter()
        .all(Option::is_some)
        && [
            &wq_a,
            &wq_b,
            &wkv,
            &wo_a,
            &wo_b,
            &router_proj,
            &shared_gate_proj,
            &shared_down_proj,
            &shared_up_proj,
        ]
        .into_iter()
        .all(Option::is_some);
    if !complete {
        tracing::warn!(
            target: "ax_mlx::weights",
            "DeepSeek V4 MTP sidecar is incomplete — skipping nextn block"
        );
        return None;
    }

    let layer = LayerWeights {
        attn_norm: attn_norm?,
        attn_post_norm: None,
        q_norm: None,
        k_norm: None,
        q_proj: None,
        k_proj: None,
        v_proj: None,
        qkv_packed: None,
        o_proj: None,
        linear_attn: None,
        glm_mla_attn: None,
        deepseek_v4: Some(DeepseekV4LayerWeights {
            wq_a: wq_a?,
            q_a_norm: q_a_norm?,
            wq_b: wq_b?,
            wkv: wkv?,
            kv_norm: kv_norm?,
            wo_a: wo_a?,
            wo_b: wo_b?,
            attn_sink,
            hc_attn_fn: hc_attn_fn?,
            hc_attn_base: hc_attn_base?,
            hc_attn_scale: hc_attn_scale?,
            hc_ffn_fn: hc_ffn_fn?,
            hc_ffn_base: hc_ffn_base?,
            hc_ffn_scale: hc_ffn_scale?,
            compressor: None,
            indexer: None,
            tid2eid: None,
        }),
        ffn_norm: ffn_norm?,
        ffn_post_norm: None,
        gate_proj: None,
        up_proj: None,
        gate_up_packed: None,
        down_proj: None,
        ffn_norm2: None,
        ffn_post_norm1: None,
        ffn_post_norm2: None,
        router_proj,
        router_correction_bias,
        router_scale: None,
        router_combined_scale: None,
        router_expert_scale: None,
        layer_scalar: None,
        per_layer_gate: None,
        per_layer_proj_w: None,
        per_layer_post_norm: None,
        shared_expert_gate: None,
        shared_gate_up_proj: None,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        gate_up_exps_packed,
        gate_exps,
        up_exps,
        down_exps,
        mxfp4_gate_up_exps: None,
        mxfp4_down_exps: None,
        attn_sink: None,
        rotation_smoothing_inverse: None,
        expert_stream: None,
    };

    Some(DeepseekV4NextnWeights {
        e_proj,
        h_proj,
        eh_proj,
        enorm,
        hnorm,
        shared_head_norm,
        embed_tokens,
        shared_head_head,
        hc_head,
        layer: Some(Box::new(layer)),
    })
}

fn load_mla_kv_b_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer_index: Option<u32>,
    mla_attention: &NativeMlaAttentionConfig,
    attention_head_count: u32,
) -> Result<(QuantizedWeight, QuantizedWeight), WeightLoadError> {
    if has_role(specs, NativeTensorRole::AttentionKvB, layer_index) {
        let kv_b = take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionKvB,
            layer_index,
            "deepseek_kv_b_proj",
        )?;
        return split_deepseek_kv_b_projection(kv_b, mla_attention, attention_head_count);
    }

    Ok((
        take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionEmbedQ,
            layer_index,
            "glm_embed_q",
        )?,
        take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionUnembedOut,
            layer_index,
            "glm_unembed_out",
        )?,
    ))
}

fn split_deepseek_kv_b_projection(
    kv_b: QuantizedWeight,
    mla_attention: &NativeMlaAttentionConfig,
    attention_head_count: u32,
) -> Result<(QuantizedWeight, QuantizedWeight), WeightLoadError> {
    let qk_nope_head_dim = mla_attention.qk_nope_head_dim.ok_or_else(|| {
        WeightLoadError::InvalidLayer("mla_attention.qk_nope_head_dim missing".to_string())
    })? as i32;
    let value_head_dim = mla_attention.value_head_dim.ok_or_else(|| {
        WeightLoadError::InvalidLayer("mla_attention.value_head_dim missing".to_string())
    })? as i32;
    let kv_lora_rank = mla_attention.kv_lora_rank.ok_or_else(|| {
        WeightLoadError::InvalidLayer("mla_attention.kv_lora_rank missing".to_string())
    })? as i32;
    let head_count = attention_head_count as i32;
    let head_dim = qk_nope_head_dim + value_head_dim;
    let expected_shape = vec![head_count * head_dim, kv_lora_rank];
    let was_quantized = kv_b.scales.is_some();
    let group_size = kv_b.group_size;
    let bits = kv_b.bits;

    let dense = if let Some(scales) = &kv_b.scales {
        dequantize(
            &kv_b.weight,
            scales,
            kv_b.biases.as_ref(),
            Some(group_size),
            Some(bits),
            None,
        )
    } else {
        kv_b.weight
    };
    if dense.shape() != expected_shape {
        return Err(WeightLoadError::InvalidLayer(format!(
            "deepseek_kv_b_proj must have shape {expected_shape:?}, got {:?}",
            dense.shape()
        )));
    }

    let kv_b_heads = reshape(&dense, &[head_count, head_dim, kv_lora_rank], None);
    let k_nope = slice(
        &kv_b_heads,
        &[0, 0, 0],
        &[head_count, qk_nope_head_dim, kv_lora_rank],
        &[1, 1, 1],
        None,
    );
    let embed_q = contiguous(&transpose(&k_nope, &[0, 2, 1], None), None);
    let unembed_out = contiguous(
        &slice(
            &kv_b_heads,
            &[0, qk_nope_head_dim, 0],
            &[head_count, head_dim, kv_lora_rank],
            &[1, 1, 1],
            None,
        ),
        None,
    );
    eval(&[&embed_q, &unembed_out]);

    if was_quantized {
        Ok((
            requantize_affine_weight(embed_q, group_size, bits, "deepseek_embed_q")?,
            requantize_affine_weight(unembed_out, group_size, bits, "deepseek_unembed_out")?,
        ))
    } else {
        Ok((
            QuantizedWeight {
                weight: embed_q,
                scales: None,
                biases: None,
                group_size,
                bits,

                mode: "affine".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q4_weight: None,
                decode_q4_scales: None,
                decode_q4_biases: None,
            },
            QuantizedWeight {
                weight: unembed_out,
                scales: None,
                biases: None,
                group_size,
                bits,

                mode: "affine".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q4_weight: None,
                decode_q4_scales: None,
                decode_q4_biases: None,
            },
        ))
    }
}

fn requantize_affine_weight(
    weight: MlxArray,
    group_size: i32,
    bits: i32,
    label: &str,
) -> Result<QuantizedWeight, WeightLoadError> {
    let mut parts = quantize(
        &weight,
        Some(group_size),
        Some(bits),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    if parts.len() != 3 {
        return Err(WeightLoadError::InvalidLayer(format!(
            "{label} quantization returned {} arrays, expected packed weight, scales, biases",
            parts.len()
        )));
    }
    let packed = parts.remove(0);
    let scales = parts.remove(0);
    let biases = parts.remove(0);
    Ok(QuantizedWeight {
        weight: packed,
        scales: Some(scales),
        biases: Some(biases),
        group_size,
        bits,

        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q4_weight: None,
        decode_q4_scales: None,
        decode_q4_biases: None,
    })
}

/// Concatenate two weight matrices along the output (row) dimension.
///
/// Used to fuse parallel projections that read the same input (e.g. q_a_proj
/// and kv_a_proj in GLM MLA, or Q/K/V in standard full-attention), replacing
/// multiple matmul kernel launches with one.
fn concat_quantized_weight_rows(
    a: &QuantizedWeight,
    b: &QuantizedWeight,
) -> Result<QuantizedWeight, WeightLoadError> {
    let (scales, biases) = match (&a.scales, &b.scales) {
        (Some(sa), Some(sb)) => {
            if a.group_size != b.group_size {
                return Err(WeightLoadError::InvalidLayer(format!(
                    "cannot fuse quantized projections with different group sizes: {} vs {}",
                    a.group_size, b.group_size
                )));
            }
            if a.bits != b.bits {
                return Err(WeightLoadError::InvalidLayer(format!(
                    "cannot fuse quantized projections with different bit widths: {} vs {}",
                    a.bits, b.bits
                )));
            }
            let biases = match (a.biases.as_ref(), b.biases.as_ref()) {
                (Some(ba), Some(bb)) => Some(concatenate(&[ba, bb], 0, None)),
                (None, None) => None,
                _ => {
                    return Err(WeightLoadError::InvalidLayer(
                        "cannot fuse projections where only one has quantization biases"
                            .to_string(),
                    ));
                }
            };
            (Some(concatenate(&[sa, sb], 0, None)), biases)
        }
        (None, None) => (None, None),
        _ => {
            return Err(WeightLoadError::InvalidLayer(
                "cannot fuse projections where only one has quantization scales".to_string(),
            ));
        }
    };
    let linear_bias = match (a.linear_bias.as_ref(), b.linear_bias.as_ref()) {
        (Some(ba), Some(bb)) => Some(concatenate(&[ba, bb], 0, None)),
        (None, None) => None,
        _ => {
            return Err(WeightLoadError::InvalidLayer(
                "cannot fuse projections where only one has dense linear bias".to_string(),
            ));
        }
    };
    if a.mode != b.mode {
        return Err(WeightLoadError::InvalidLayer(format!(
            "cannot fuse projections with different quant modes: {} vs {}",
            a.mode, b.mode
        )));
    }
    Ok(QuantizedWeight {
        weight: concatenate(&[&a.weight, &b.weight], 0, None),
        scales,
        biases,
        group_size: a.group_size,
        bits: a.bits,
        mode: a.mode.clone(),
        linear_bias,
        decode_weight_t: None,
        decode_q4_weight: None,
        decode_q4_scales: None,
        decode_q4_biases: None,
    })
}

fn pack_dense_ffn_gate_up_projection(
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> Result<QuantizedWeight, WeightLoadError> {
    let packed = concat_quantized_weight_rows(gate, up)?;
    eval_packed_projection(&packed);
    Ok(packed)
}

fn dense_ffn_gate_up_packing_supported(
    model_family: &str,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> bool {
    // Qwen dense FFNs keep 4/5-bit projections split: the split runtime owns
    // the decode matvec fast path. 4-bit packing regressed Qwen3.5-9B S0
    // (81 → 24 tok/s, 2026-07-24), AXQ 27B 4-bit gs32 prefill p128 (347 vs
    // 403, 2026-08-12), and seq-gated p2048 pack+compile (881 vs 889 q4,
    // 2026-08-13). Six-bit Qwen keeps both layouts so multi-token prefill
    // can use one packed gate/up while decode stays split. GLM MLA MoE
    // lite stays split.
    if model_family == "glm4_moe_lite"
        || (model_family.starts_with("qwen") && (gate.bits != 6 || up.bits != 6))
    {
        return false;
    }
    if gate.bits == 5 || up.bits == 5 || gate.mode != up.mode {
        return false;
    }
    if gate.linear_bias.is_some() != up.linear_bias.is_some() {
        return false;
    }
    match (gate.scales.as_ref(), up.scales.as_ref()) {
        (Some(_), Some(_)) => {
            gate.group_size == up.group_size
                && gate.bits == up.bits
                && gate.biases.is_some() == up.biases.is_some()
        }
        (None, None) => true,
        _ => false,
    }
}

fn pack_glm_mla_qa_kva_projection(
    q_a: &QuantizedWeight,
    kv_a: &QuantizedWeight,
) -> Result<QuantizedWeight, WeightLoadError> {
    let packed = concat_quantized_weight_rows(q_a, kv_a)?;
    eval_packed_projection(&packed);
    Ok(packed)
}

fn linear_attention_projection_packing_supported(
    qkv: &QuantizedWeight,
    z: &QuantizedWeight,
    a: &QuantizedWeight,
    b: &QuantizedWeight,
) -> bool {
    validate_linear_attention_pack_compatibility(
        "linear_attention_in_proj_qkvz",
        &[Some(qkv), Some(z)],
    )
    .is_ok()
        && validate_linear_attention_pack_compatibility(
            "linear_attention_in_proj_ba",
            &[Some(b), Some(a)],
        )
        .is_ok()
}

fn pack_split_linear_attention_projections(
    config: &NativeLinearAttentionConfig,
    qkv: &QuantizedWeight,
    z: &QuantizedWeight,
    a: &QuantizedWeight,
    b: &QuantizedWeight,
) -> Result<(QuantizedWeight, QuantizedWeight), WeightLoadError> {
    let num_key_heads = usize::try_from(config.num_key_heads.ok_or_else(|| {
        WeightLoadError::InvalidLayer(
            "linear attention projection pack requires num_key_heads".to_string(),
        )
    })?)
    .map_err(|_| {
        WeightLoadError::InvalidLayer(
            "linear attention projection pack num_key_heads does not fit usize".to_string(),
        )
    })?;
    let key_head_dim = usize::try_from(config.key_head_dim.ok_or_else(|| {
        WeightLoadError::InvalidLayer(
            "linear attention projection pack requires key_head_dim".to_string(),
        )
    })?)
    .map_err(|_| {
        WeightLoadError::InvalidLayer(
            "linear attention projection pack key_head_dim does not fit usize".to_string(),
        )
    })?;
    let num_value_heads = usize::try_from(config.num_value_heads.ok_or_else(|| {
        WeightLoadError::InvalidLayer(
            "linear attention projection pack requires num_value_heads".to_string(),
        )
    })?)
    .map_err(|_| {
        WeightLoadError::InvalidLayer(
            "linear attention projection pack num_value_heads does not fit usize".to_string(),
        )
    })?;
    let value_head_dim = usize::try_from(config.value_head_dim.ok_or_else(|| {
        WeightLoadError::InvalidLayer(
            "linear attention projection pack requires value_head_dim".to_string(),
        )
    })?)
    .map_err(|_| {
        WeightLoadError::InvalidLayer(
            "linear attention projection pack value_head_dim does not fit usize".to_string(),
        )
    })?;

    let qkvz_sources = linear_attention_qkvz_row_sources(
        num_key_heads,
        key_head_dim,
        num_value_heads,
        value_head_dim,
    )?;
    let ba_sources = linear_attention_ba_row_sources(num_key_heads, num_value_heads)?;
    let qkvz = pack_linear_attention_projection_rows(
        "linear_attention_in_proj_qkvz",
        &qkvz_sources,
        Some(qkv),
        Some(z),
        None,
        None,
    )?;
    let ba = pack_linear_attention_projection_rows(
        "linear_attention_in_proj_ba",
        &ba_sources,
        None,
        None,
        Some(b),
        Some(a),
    )?;
    eval_packed_projection(&qkvz);
    eval_packed_projection(&ba);
    Ok((qkvz, ba))
}

fn eval_packed_projection(weight: &QuantizedWeight) {
    let mut arrays = vec![&weight.weight];
    if let Some(scales) = &weight.scales {
        arrays.push(scales);
    }
    if let Some(biases) = &weight.biases {
        arrays.push(biases);
    }
    eval(&arrays);
}

fn pack_linear_attention_projection_rows(
    label: &str,
    sources: &[LinearAttentionProjectionRowSource],
    qkv: Option<&QuantizedWeight>,
    z: Option<&QuantizedWeight>,
    b: Option<&QuantizedWeight>,
    a: Option<&QuantizedWeight>,
) -> Result<QuantizedWeight, WeightLoadError> {
    let first = qkv.or(z).or(b).or(a).ok_or_else(|| {
        WeightLoadError::InvalidLayer(format!("cannot pack {label} without source projections"))
    })?;
    validate_linear_attention_pack_compatibility(label, &[qkv, z, b, a])?;

    let scales = if first.scales.is_some() {
        Some(gather_linear_attention_projection_arrays(
            label,
            sources,
            qkv.and_then(|weight| weight.scales.as_ref()),
            z.and_then(|weight| weight.scales.as_ref()),
            b.and_then(|weight| weight.scales.as_ref()),
            a.and_then(|weight| weight.scales.as_ref()),
        )?)
    } else {
        None
    };
    let biases = if first.biases.is_some() {
        Some(gather_linear_attention_projection_arrays(
            label,
            sources,
            qkv.and_then(|weight| weight.biases.as_ref()),
            z.and_then(|weight| weight.biases.as_ref()),
            b.and_then(|weight| weight.biases.as_ref()),
            a.and_then(|weight| weight.biases.as_ref()),
        )?)
    } else {
        None
    };

    Ok(QuantizedWeight {
        weight: gather_linear_attention_projection_arrays(
            label,
            sources,
            qkv.map(|weight| &weight.weight),
            z.map(|weight| &weight.weight),
            b.map(|weight| &weight.weight),
            a.map(|weight| &weight.weight),
        )?,
        scales,
        biases,
        group_size: first.group_size,
        bits: first.bits,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q4_weight: None,
        decode_q4_scales: None,
        decode_q4_biases: None,
    })
}

fn validate_linear_attention_pack_compatibility(
    label: &str,
    weights: &[Option<&QuantizedWeight>],
) -> Result<(), WeightLoadError> {
    let first = weights.iter().flatten().next().ok_or_else(|| {
        WeightLoadError::InvalidLayer(format!("cannot pack {label} without source projections"))
    })?;
    for weight in weights.iter().flatten().skip(1) {
        if weight.group_size != first.group_size {
            return Err(WeightLoadError::InvalidLayer(format!(
                "cannot pack {label} projections with different group sizes: {} vs {}",
                first.group_size, weight.group_size
            )));
        }
        if weight.bits != first.bits {
            return Err(WeightLoadError::InvalidLayer(format!(
                "cannot pack {label} projections with different bit widths: {} vs {}",
                first.bits, weight.bits
            )));
        }
        if weight.scales.is_some() != first.scales.is_some() {
            return Err(WeightLoadError::InvalidLayer(format!(
                "cannot pack {label} projections where only one has quantization scales"
            )));
        }
        if weight.biases.is_some() != first.biases.is_some() {
            return Err(WeightLoadError::InvalidLayer(format!(
                "cannot pack {label} projections where only one has quantization biases"
            )));
        }
    }
    Ok(())
}

fn gather_linear_attention_projection_arrays(
    label: &str,
    sources: &[LinearAttentionProjectionRowSource],
    qkv: Option<&MlxArray>,
    z: Option<&MlxArray>,
    b: Option<&MlxArray>,
    a: Option<&MlxArray>,
) -> Result<MlxArray, WeightLoadError> {
    if sources.is_empty() {
        return Err(WeightLoadError::InvalidLayer(format!(
            "cannot pack {label} with empty row sources"
        )));
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < sources.len() {
        let source_kind = sources[start].kind();
        let slice_start = sources[start].index();
        let mut next_index = slice_start + 1;
        let mut end = start;
        while end < sources.len()
            && sources[end].kind() == source_kind
            && sources[end].index() == next_index - 1
        {
            next_index += 1;
            end += 1;
        }
        let source_array = match source_kind {
            LinearAttentionProjectionRowKind::Qkv => qkv,
            LinearAttentionProjectionRowKind::Z => z,
            LinearAttentionProjectionRowKind::B => b,
            LinearAttentionProjectionRowKind::A => a,
        }
        .ok_or_else(|| {
            WeightLoadError::InvalidLayer(format!(
                "cannot pack {label}; missing {source_kind:?} source projection"
            ))
        })?;
        chunks.push(slice_linear_attention_projection_rows(
            label,
            source_array,
            slice_start,
            next_index - 1,
        )?);
        start = end;
    }
    let refs = chunks.iter().collect::<Vec<_>>();
    Ok(concatenate(&refs, 0, None))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LinearAttentionProjectionRowKind {
    Qkv,
    Z,
    B,
    A,
}

impl LinearAttentionProjectionRowSource {
    fn kind(self) -> LinearAttentionProjectionRowKind {
        match self {
            Self::Qkv(_) => LinearAttentionProjectionRowKind::Qkv,
            Self::Z(_) => LinearAttentionProjectionRowKind::Z,
            Self::B(_) => LinearAttentionProjectionRowKind::B,
            Self::A(_) => LinearAttentionProjectionRowKind::A,
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Qkv(index) | Self::Z(index) | Self::B(index) | Self::A(index) => index,
        }
    }
}

fn slice_linear_attention_projection_rows(
    label: &str,
    array: &MlxArray,
    start: usize,
    end: usize,
) -> Result<MlxArray, WeightLoadError> {
    let shape = array.shape();
    let Some(&row_count) = shape.first() else {
        return Err(WeightLoadError::InvalidLayer(format!(
            "cannot pack {label}; source projection has no row dimension"
        )));
    };
    let row_count = usize::try_from(row_count).map_err(|_| {
        WeightLoadError::InvalidLayer(format!(
            "cannot pack {label}; source projection row count is negative"
        ))
    })?;
    if start >= end || end > row_count {
        return Err(WeightLoadError::InvalidLayer(format!(
            "cannot pack {label}; row source exceeded input rows"
        )));
    }
    let start_i32 = i32::try_from(start).map_err(|_| {
        WeightLoadError::InvalidLayer(format!(
            "cannot pack {label}; row slice start does not fit i32"
        ))
    })?;
    let end_i32 = i32::try_from(end).map_err(|_| {
        WeightLoadError::InvalidLayer(format!(
            "cannot pack {label}; row slice end does not fit i32"
        ))
    })?;
    let mut starts = vec![0; shape.len()];
    let mut stops = shape;
    let strides = vec![1; stops.len()];
    starts[0] = start_i32;
    stops[0] = end_i32;
    Ok(slice(array, &starts, &stops, &strides, None))
}

fn linear_attention_qkvz_row_sources(
    num_key_heads: usize,
    key_head_dim: usize,
    num_value_heads: usize,
    value_head_dim: usize,
) -> Result<Vec<LinearAttentionProjectionRowSource>, WeightLoadError> {
    if num_key_heads == 0 || key_head_dim == 0 || num_value_heads == 0 || value_head_dim == 0 {
        return Err(WeightLoadError::InvalidLayer(
            "linear attention projection pack dimensions must be non-zero".to_string(),
        ));
    }
    if !num_value_heads.is_multiple_of(num_key_heads) {
        return Err(WeightLoadError::InvalidLayer(format!(
            "linear attention projection pack requires value heads divisible by key heads: {num_value_heads} vs {num_key_heads}"
        )));
    }

    let key_dim = num_key_heads * key_head_dim;
    let value_heads_per_key = num_value_heads / num_key_heads;
    let value_dim_per_key = value_heads_per_key * value_head_dim;
    let q_base = 0;
    let k_base = key_dim;
    let v_base = key_dim * 2;
    let mut rows = Vec::with_capacity(key_dim * 2 + num_value_heads * value_head_dim * 2);

    for key_head in 0..num_key_heads {
        let q_start = q_base + key_head * key_head_dim;
        let k_start = k_base + key_head * key_head_dim;
        let value_start = key_head * value_dim_per_key;
        rows.extend((q_start..q_start + key_head_dim).map(LinearAttentionProjectionRowSource::Qkv));
        rows.extend((k_start..k_start + key_head_dim).map(LinearAttentionProjectionRowSource::Qkv));
        rows.extend(
            (v_base + value_start..v_base + value_start + value_dim_per_key)
                .map(LinearAttentionProjectionRowSource::Qkv),
        );
        rows.extend(
            (value_start..value_start + value_dim_per_key)
                .map(LinearAttentionProjectionRowSource::Z),
        );
    }
    Ok(rows)
}

fn linear_attention_ba_row_sources(
    num_key_heads: usize,
    num_value_heads: usize,
) -> Result<Vec<LinearAttentionProjectionRowSource>, WeightLoadError> {
    if num_key_heads == 0 || num_value_heads == 0 {
        return Err(WeightLoadError::InvalidLayer(
            "linear attention BA pack dimensions must be non-zero".to_string(),
        ));
    }
    if !num_value_heads.is_multiple_of(num_key_heads) {
        return Err(WeightLoadError::InvalidLayer(format!(
            "linear attention BA pack requires value heads divisible by key heads: {num_value_heads} vs {num_key_heads}"
        )));
    }

    let value_heads_per_key = num_value_heads / num_key_heads;
    let mut rows = Vec::with_capacity(num_value_heads * 2);
    for key_head in 0..num_key_heads {
        let value_start = key_head * value_heads_per_key;
        rows.extend(
            (value_start..value_start + value_heads_per_key)
                .map(LinearAttentionProjectionRowSource::B),
        );
        rows.extend(
            (value_start..value_start + value_heads_per_key)
                .map(LinearAttentionProjectionRowSource::A),
        );
    }
    Ok(rows)
}

#[cfg(test)]
fn gather_linear_attention_projection_rows<T: Copy>(
    sources: &[LinearAttentionProjectionRowSource],
    qkv: &[T],
    z: &[T],
    b: &[T],
    a: &[T],
) -> Result<Vec<T>, WeightLoadError> {
    sources
        .iter()
        .map(|source| match *source {
            LinearAttentionProjectionRowSource::Qkv(index) => qkv.get(index).copied(),
            LinearAttentionProjectionRowSource::Z(index) => z.get(index).copied(),
            LinearAttentionProjectionRowSource::B(index) => b.get(index).copied(),
            LinearAttentionProjectionRowSource::A(index) => a.get(index).copied(),
        })
        .collect::<Option<Vec<T>>>()
        .ok_or_else(|| {
            WeightLoadError::InvalidLayer(
                "linear attention projection pack row source exceeded input rows".to_string(),
            )
        })
}

/// Load a weight tensor together with its `.scales` and `.biases` siblings
/// if they exist in the safetensors map (MLX affine quantization format).
fn take_weight(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    role: NativeTensorRole,
    layer_index: Option<u32>,
    label: &str,
) -> Result<QuantizedWeight, WeightLoadError> {
    let spec = specs
        .iter()
        .find(|s| s.role == role && s.layer_index == layer_index)
        .ok_or_else(|| WeightLoadError::RoleMissing(format!("{label}[{layer_index:?}]")))?;
    take_weight_spec(name_map, spec)
}

/// Load a quantized or dense linear by its exact checkpoint tensor name.
///
/// Multimodal towers are intentionally retained as `Other` manifest roles, so
/// their loaders resolve the original names while still honoring each
/// tensor's manifest quantization metadata.
pub(crate) fn take_named_weight(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    name: &str,
) -> Result<QuantizedWeight, WeightLoadError> {
    let spec = specs
        .iter()
        .find(|spec| spec.name == name)
        .ok_or_else(|| WeightLoadError::TensorMissing(name.to_string()))?;
    take_weight_spec(name_map, spec)
}

fn take_weight_spec(
    name_map: &mut HashMap<String, MlxArray>,
    spec: &NativeTensorSpec,
) -> Result<QuantizedWeight, WeightLoadError> {
    let name = spec.name.clone();
    let weight = name_map
        .remove(&name)
        .ok_or_else(|| WeightLoadError::TensorMissing(name.clone()))?;

    // Co-located sidecars:
    // - `.scales` / `.biases` (plural): MLX group-quant metadata
    // - `.bias` (singular): dense Linear bias (e.g. SwitchLinear.bias [E, out])
    //   — must NOT be treated as affine group biases (mlx-lm switch_layers.py).
    let base = name.strip_suffix(".weight").unwrap_or(&name);
    let scales = name_map.remove(&format!("{base}.scales"));
    let quant_biases = name_map.remove(&format!("{base}.biases"));
    let linear_bias = name_map.remove(&format!("{base}.bias"));
    let has_quantization_sidecars = scales.is_some() || quant_biases.is_some();

    if !spec.source_quantized && has_quantization_sidecars {
        return Err(WeightLoadError::InvalidLayer(format!(
            "tensor {name} has MLX quantization sidecar tensors but source_quantized is false"
        )));
    }

    if spec.source_quantized && scales.is_none() {
        return Err(WeightLoadError::QuantizationMissing(format!(
            "{base}.scales"
        )));
    }

    Ok(
        QuantizedWeight::with_quantization(
            weight,
            scales,
            quant_biases,
            spec.quantization.as_ref(),
        )
        .with_linear_bias(linear_bias),
    )
}

fn try_take_weight(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    role: NativeTensorRole,
    layer_index: Option<u32>,
    label: &str,
) -> Result<Option<QuantizedWeight>, WeightLoadError> {
    if has_role(specs, role, layer_index) {
        take_weight(specs, name_map, role, layer_index, label).map(Some)
    } else {
        Ok(None)
    }
}

/// Load a plain (non-quantized) weight tensor or return None if not present.
fn try_take_plain(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    role: NativeTensorRole,
    layer_index: Option<u32>,
) -> Result<Option<MlxArray>, WeightLoadError> {
    let Some(name) = specs
        .iter()
        .find(|s| s.role == role && s.layer_index == layer_index)
        .map(|s| s.name.clone())
    else {
        return Ok(None);
    };
    Ok(name_map.remove(&name))
}

/// Load openai/gpt-oss native fused MXFP4 experts and sanitize to split
/// `gate_proj` / `up_proj` / `down_proj` — matching mlx-lm `gpt_oss.Model.sanitize`.
///
/// OpenAI tensors:
///   mlp.experts.gate_up_proj_blocks/scales  (fused, even/odd rows = gate/up)
///   mlp.experts.down_proj_blocks/scales
/// After sanitize (view u32 + flatten last two dims + de-interleave), weights
/// stay packed MXFP4 (`mode=mxfp4`) for `gather_qmm` — no BF16 expand.
fn load_gpt_oss_openai_mxfp4_split_experts(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer_index: Option<u32>,
) -> Result<(QuantizedWeight, QuantizedWeight, QuantizedWeight), WeightLoadError> {
    let (gate_up_w, gate_up_s) = load_mxfp4_blocks_scales(
        specs,
        name_map,
        layer_index,
        NativeTensorRole::FfnGateUpExpsMxfp4Blocks,
        NativeTensorRole::FfnGateUpExpsMxfp4Scales,
        "gate_up_exps",
    )?;
    let (down_w, down_s) = load_mxfp4_blocks_scales(
        specs,
        name_map,
        layer_index,
        NativeTensorRole::FfnDownExpsMxfp4Blocks,
        NativeTensorRole::FfnDownExpsMxfp4Scales,
        "down_exps",
    )?;

    // De-interleave fused gate_up along the out-feature axis (dim -2):
    // even rows → gate, odd rows → up (mlx-lm gpt_oss.sanitize).
    let gate_w = contiguous(&slice_even_odd_out_rows(&gate_up_w, /*even=*/ true), None);
    let up_w = contiguous(&slice_even_odd_out_rows(&gate_up_w, /*even=*/ false), None);
    let gate_s = contiguous(&slice_even_odd_out_rows(&gate_up_s, /*even=*/ true), None);
    let up_s = contiguous(&slice_even_odd_out_rows(&gate_up_s, /*even=*/ false), None);

    // Optional dense expert biases (openai: gate_up_proj_bias / down_proj_bias).
    // mlx-lm renames and de-interleaves these to gate_proj.bias / up_proj.bias.
    let (gate_b, up_b, down_b) = take_gpt_oss_openai_expert_biases(specs, name_map, layer_index);

    eval(&[&gate_w, &up_w, &down_w, &gate_s, &up_s, &down_s]);
    if let Some(b) = &gate_b {
        eval(&[b]);
    }
    if let Some(b) = &up_b {
        eval(&[b]);
    }
    if let Some(b) = &down_b {
        eval(&[b]);
    }

    let mxfp4 = NativeTensorQuantization {
        mode: "mxfp4".to_string(),
        group_size: 32,
        bits: 4,
    };
    Ok((
        QuantizedWeight::with_quantization(gate_w, Some(gate_s), None, Some(&mxfp4))
            .with_linear_bias(gate_b),
        QuantizedWeight::with_quantization(up_w, Some(up_s), None, Some(&mxfp4))
            .with_linear_bias(up_b),
        QuantizedWeight::with_quantization(down_w, Some(down_s), None, Some(&mxfp4))
            .with_linear_bias(down_b),
    ))
}

/// Pull openai fused expert bias tensors and de-interleave gate_up bias.
fn take_gpt_oss_openai_expert_biases(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer_index: Option<u32>,
) -> (Option<MlxArray>, Option<MlxArray>, Option<MlxArray>) {
    // Discover names from the blocks tensors we already mapped (same layer).
    let gate_up_blocks = specs.iter().find(|s| {
        s.role == NativeTensorRole::FfnGateUpExpsMxfp4Blocks && s.layer_index == layer_index
    });
    let down_blocks = specs.iter().find(|s| {
        s.role == NativeTensorRole::FfnDownExpsMxfp4Blocks && s.layer_index == layer_index
    });

    let mut gate_b = None;
    let mut up_b = None;
    if let Some(spec) = gate_up_blocks {
        // model.layers.N.mlp.experts.gate_up_proj_blocks → gate_up_proj_bias
        let bias_name = spec
            .name
            .replace("gate_up_proj_blocks", "gate_up_proj_bias");
        if let Some(bias) = name_map.remove(&bias_name) {
            // even/odd on last axis for bias [E, 2*inter] → [E, inter]
            let gate = contiguous(&slice_even_odd_last_axis(&bias, true), None);
            let up = contiguous(&slice_even_odd_last_axis(&bias, false), None);
            gate_b = Some(gate);
            up_b = Some(up);
        }
    }
    let down_b = down_blocks.and_then(|spec| {
        let bias_name = spec.name.replace("down_proj_blocks", "down_proj_bias");
        name_map.remove(&bias_name)
    });
    (gate_b, up_b, down_b)
}

/// Even/odd slice on the last axis (bias de-interleave: [..., 2*I] → [..., I]).
fn slice_even_odd_last_axis(x: &MlxArray, even: bool) -> MlxArray {
    let shape = x.shape();
    let ndim = shape.len();
    assert!(ndim >= 1);
    let last = ndim - 1;
    let n = shape[last];
    let start = if even { 0 } else { 1 };
    let mut starts = vec![0i32; ndim];
    let mut stops: Vec<i32> = shape.to_vec();
    let mut strides = vec![1i32; ndim];
    starts[last] = start;
    stops[last] = n;
    strides[last] = 2;
    slice(x, &starts, &stops, &strides, None)
}

fn load_mxfp4_blocks_scales(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer_index: Option<u32>,
    blocks_role: NativeTensorRole,
    scales_role: NativeTensorRole,
    label: &str,
) -> Result<(MlxArray, MlxArray), WeightLoadError> {
    let blocks_name = specs
        .iter()
        .find(|s| s.role == blocks_role && s.layer_index == layer_index)
        .map(|s| s.name.clone())
        .ok_or_else(|| WeightLoadError::RoleMissing(format!("{label}_blocks[{layer_index:?}]")))?;
    let scales_name = specs
        .iter()
        .find(|s| s.role == scales_role && s.layer_index == layer_index)
        .map(|s| s.name.clone())
        .ok_or_else(|| WeightLoadError::RoleMissing(format!("{label}_scales[{layer_index:?}]")))?;

    let blocks = name_map
        .remove(&blocks_name)
        .ok_or(WeightLoadError::TensorMissing(blocks_name))?;
    let scales = name_map
        .remove(&scales_name)
        .ok_or(WeightLoadError::TensorMissing(scales_name))?;

    // Sanitize: u8 blocks → u32 view, flatten last two dims (mlx-lm gpt_oss.sanitize).
    let blocks_u32 = view(&blocks, MlxDtype::Uint32, None);
    let ndim = blocks_u32.ndim();
    let blocks_flat = flatten(&blocks_u32, (ndim - 2) as i32, (ndim - 1) as i32, None);
    Ok((blocks_flat, scales))
}

/// Slice even or odd rows along the expert out-feature axis (dim = ndim-2).
fn slice_even_odd_out_rows(x: &MlxArray, even: bool) -> MlxArray {
    let shape = x.shape();
    let ndim = shape.len();
    assert!(
        ndim >= 2,
        "gpt-oss expert tensor must be at least 2D, got {ndim}"
    );
    let out_axis = ndim - 2;
    let out = shape[out_axis];
    let start = if even { 0 } else { 1 };
    let mut starts = vec![0i32; ndim];
    let mut stops: Vec<i32> = shape.to_vec();
    let mut strides = vec![1i32; ndim];
    starts[out_axis] = start;
    stops[out_axis] = out;
    strides[out_axis] = 2;
    slice(x, &starts, &stops, &strides, None)
}

fn take_layer_norms(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer_index: Option<u32>,
) -> Result<(Option<MlxArray>, MlxArray), WeightLoadError> {
    // When FfnNorm (pre_feedforward_layernorm) is present, AttentionPostNorm
    // is a genuine post-attention norm applied before the residual add. When
    // FfnNorm is absent (Qwen3 and GLM4MoELite), AttentionPostNorm is the
    // pre-FFN norm applied after the attention residual.
    if has_role(specs, NativeTensorRole::FfnNorm, layer_index) {
        let attn_post_norm = try_take_plain(
            specs,
            name_map,
            NativeTensorRole::AttentionPostNorm,
            layer_index,
        )?;
        let ffn_norm = take_weight(
            specs,
            name_map,
            NativeTensorRole::FfnNorm,
            layer_index,
            "ffn_norm",
        )?
        .weight;
        Ok((attn_post_norm, ffn_norm))
    } else {
        let ffn_norm = take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionPostNorm,
            layer_index,
            "attention_post_norm",
        )?
        .weight;
        Ok((None, ffn_norm))
    }
}

fn has_role(specs: &[NativeTensorSpec], role: NativeTensorRole, layer_index: Option<u32>) -> bool {
    specs
        .iter()
        .any(|s| s.role == role && s.layer_index == layer_index)
}

#[derive(Debug, thiserror::Error)]
pub enum WeightLoadError {
    #[error("weight file not found or empty: {0}")]
    FileMissing(String),
    #[error("tensor not found: {0}")]
    TensorMissing(String),
    #[error("required tensor role missing: {0}")]
    RoleMissing(String),
    #[error("quantized tensor metadata missing: {0}")]
    QuantizationMissing(String),
    #[error("invalid layer tensor layout: {0}")]
    InvalidLayer(String),
    #[error("unsanitized weights: {0}")]
    UnsanitizedWeights(String),
    #[error("rotated checkpoint required but invalid: {0}")]
    RotatedCheckpointInvalid(String),
    #[error("pipeline weight loading does not support model family: {0}")]
    UnsupportedPipelineFamily(String),
    #[error("invalid pipeline rank assignment: {0}")]
    InvalidPipelineAssignment(String),
    #[error("invalid AXQuant vision sidecar: {0}")]
    VisionSidecarInvalid(String),
    #[error(transparent)]
    ExpertStream(#[from] crate::expert_stream::ExpertStreamError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_core::NativeTensorDataType;
    use mlx_sys::{MlxDtype, zeros};
    use std::path::Path;

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
                    conv[o * IN_DIM * KERNEL_DIM + i * KERNEL_DIM + k] =
                        (100 * o + 10 * i + k) as f32;
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

        apply_hf_sanitize_transforms(&specs, &mut name_map, true);

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

        apply_hf_sanitize_transforms(&specs, &mut name_map, false);

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
    fn fixture_dense_norms(
        norm_data: &[f32],
    ) -> (Vec<NativeTensorSpec>, HashMap<String, MlxArray>) {
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

        let chosen =
            effective_weight_sanitize("gemma4", WeightSanitize::HfToMlx, &specs, &name_map);
        assert_eq!(chosen, WeightSanitize::HfToMlx);

        let sanitized_norm_data = vec![1.0_f32; 256];
        let (specs, name_map) = fixture_dense_norms(&sanitized_norm_data);
        let chosen =
            effective_weight_sanitize("gemma4", WeightSanitize::HfNormOnly, &specs, &name_map);
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

        apply_hf_sanitize_transforms(&specs, &mut name_map, true);

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

        apply_hf_sanitize_transforms(&specs, &mut name_map, true);

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
        let weights =
            load_glm_mla_attention_weights(&specs, &mut name_map, Some(0), &mla_attention, 1)
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

        let weights =
            load_glm_mla_attention_weights(&specs, &mut name_map, Some(0), &mla_attention, 2)
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
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        }
    }

    fn invalid_layer_message(result: Result<QuantizedWeight, WeightLoadError>) -> String {
        match result {
            Err(WeightLoadError::InvalidLayer(message)) => message,
            Err(error) => panic!("expected invalid layer error, got {error}"),
            Ok(_) => panic!("expected fused GLM MLA weights to be rejected"),
        }
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
                "language_model.model.layers.0.mlp.up_proj.weight".into(),
                zeros(&[16, 2], MlxDtype::Uint32, None),
            ),
            (
                "language_model.model.layers.0.mlp.up_proj.scales".into(),
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
        let rows =
            linear_attention_qkvz_row_sources(2, 2, 4, 3).expect("valid linear attention dims");

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
        let qkvz_message =
            invalid_layer_message(linear_attention_qkvz_row_sources(3, 2, 4, 3).map(|_| {
                QuantizedWeight::new(zeros(&[1, 1], MlxDtype::Float32, None), None, None)
            }));
        let ba_message =
            invalid_layer_message(linear_attention_ba_row_sources(3, 4).map(|_| {
                QuantizedWeight::new(zeros(&[1, 1], MlxDtype::Float32, None), None, None)
            }));

        assert!(qkvz_message.contains("value heads divisible by key heads"));
        assert!(ba_message.contains("value heads divisible by key heads"));
    }

    #[test]
    fn linear_attention_qkvz_pack_oracle_gathers_rows_in_packed_order() {
        let rows =
            linear_attention_qkvz_row_sources(2, 1, 4, 2).expect("valid linear attention dims");
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

        let quantized =
            QuantizedWeight::with_quantization(weight, scales, None, Some(&quantization));

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
        assert_eq!(default_mtp_depth_without_env(3, Some(8)), 3);
        assert_eq!(default_mtp_depth_without_env(1, Some(8)), 1);
        assert_eq!(default_mtp_depth_without_env(0, Some(8)), 0);
        assert_eq!(default_mtp_depth_without_env(3, Some(4)), 3);
        assert_eq!(default_mtp_depth_without_env(3, None), 3);
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
        let manifest: ax_engine_core::NativeModelManifest =
            serde_json::from_value(serde_json::json!({
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

    fn dsv4_mtp_tensor_refs(
        tensors: &[(String, String, Vec<usize>)],
    ) -> Vec<(&str, &str, &[usize])> {
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
        let file_bytes =
            write_vision_sidecar_fixture(&dir, &[("vision_tower.weight", &[1.0], &[1])]);
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
        let result =
            load_vision_sidecar(&dir, &mut name_map).expect("absent sidecar is not an error");

        assert!(result.is_none());
        assert!(name_map.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
