use super::*;

#[derive(Clone, Copy)]
pub(crate) enum TensorMapping {
    Global(NativeTensorRole),
    PerLayer(NativeTensorRole),
}

/// Extra per-layer tensor patterns for Qwen3 MoE (mlp.gate → router; switch_mlp → experts).
pub(crate) const QWEN3_MOE_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "mlp.gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    (
        "mlp.switch_mlp.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateExps),
    ),
    (
        "mlp.switch_mlp.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "mlp.switch_mlp.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
];

/// Extra per-layer tensor patterns for GLM4MoELite.
///
/// These roles intentionally make the manifest graph-specific instead of
/// pretending GLM's MLA projections are ordinary split Q/K/V attention.
pub(crate) const GLM4_MOE_LITE_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "self_attn.q_a_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQa),
    ),
    (
        "self_attn.q_a_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQaNorm),
    ),
    (
        "self_attn.q_b_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQb),
    ),
    (
        "self_attn.kv_a_proj_with_mqa.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionKvA),
    ),
    (
        "self_attn.kv_a_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionKvANorm),
    ),
    (
        "self_attn.embed_q.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionEmbedQ),
    ),
    (
        "self_attn.unembed_out.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionUnembedOut),
    ),
    (
        "mlp.gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    (
        "mlp.gate.e_score_correction_bias",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInpCorrectionBias),
    ),
    (
        "mlp.switch_mlp.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateExps),
    ),
    (
        "mlp.switch_mlp.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "mlp.switch_mlp.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
    (
        "mlp.shared_experts.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertGate),
    ),
    (
        "mlp.shared_experts.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertUp),
    ),
    (
        "mlp.shared_experts.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertDown),
    ),
];

/// Extra per-layer tensor patterns for DeepSeek V3/V3.2.
///
/// Raw HuggingFace checkpoints store the MLA KV-B projection as
/// `kv_b_proj.weight`; the MLX runtime splits it into the same `embed_q` and
/// `unembed_out` layout used by mlx-lm at load time.
pub(crate) const DEEPSEEK_V3_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "self_attn.q_a_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQa),
    ),
    (
        "self_attn.q_a_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQaNorm),
    ),
    (
        "self_attn.q_b_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQb),
    ),
    (
        "self_attn.kv_a_proj_with_mqa.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionKvA),
    ),
    (
        "self_attn.kv_a_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionKvANorm),
    ),
    (
        "self_attn.kv_b_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionKvB),
    ),
    (
        "self_attn.embed_q.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionEmbedQ),
    ),
    (
        "self_attn.unembed_out.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionUnembedOut),
    ),
    (
        "mlp.gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    (
        "mlp.gate.e_score_correction_bias",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInpCorrectionBias),
    ),
    (
        "mlp.switch_mlp.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateExps),
    ),
    (
        "mlp.switch_mlp.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "mlp.switch_mlp.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
    (
        "mlp.shared_experts.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertGate),
    ),
    (
        "mlp.shared_experts.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertUp),
    ),
    (
        "mlp.shared_experts.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertDown),
    ),
];

/// Unlimited-OCR language MoE extras + projector / special tokens.
///
/// Language weights live under `language_model.model.layers.*` (via
/// `uses_language_model_prefix`). Vision/SAM tensors are harvested from the
/// safetensors name map at load time without explicit per-tensor roles.
pub(crate) const UNLIMITED_OCR_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "mlp.gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    (
        "mlp.switch_mlp.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateExps),
    ),
    (
        "mlp.switch_mlp.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "mlp.switch_mlp.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
    (
        "mlp.shared_experts.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertGate),
    ),
    (
        "mlp.shared_experts.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertUp),
    ),
    (
        "mlp.shared_experts.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertDown),
    ),
    // Projector + special image tokens (global).
    (
        "projector.layers.weight",
        TensorMapping::Global(NativeTensorRole::UnlimitedOcrProjector),
    ),
    (
        "image_newline",
        TensorMapping::Global(NativeTensorRole::UnlimitedOcrImageNewline),
    ),
    (
        "view_separator",
        TensorMapping::Global(NativeTensorRole::UnlimitedOcrViewSeparator),
    ),
];

/// Per-layer tensor patterns for Mixtral sparse MoE layers.
pub(crate) const MIXTRAL_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "block_sparse_moe.gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    (
        "block_sparse_moe.switch_mlp.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateExps),
    ),
    (
        "block_sparse_moe.switch_mlp.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "block_sparse_moe.switch_mlp.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
];

/// Per-layer tensor patterns for LLaMA 4 (uses feed_forward.* instead of mlp.*,
/// plus MoE experts and shared expert paths).
pub(crate) const LLAMA4_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    // Dense FFN layers (non-MoE)
    (
        "feed_forward.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGate),
    ),
    (
        "feed_forward.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUp),
    ),
    (
        "feed_forward.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDown),
    ),
    // MoE router
    (
        "feed_forward.router.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    // Packed expert weights (SwitchGLU)
    (
        "feed_forward.experts.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateExps),
    ),
    (
        "feed_forward.experts.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "feed_forward.experts.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
    // Shared expert
    (
        "feed_forward.shared_expert.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertGate),
    ),
    (
        "feed_forward.shared_expert.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertUp),
    ),
    (
        "feed_forward.shared_expert.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertDown),
    ),
];

/// Per-layer tensor patterns for GPT-OSS MoE layers.
///
/// Two published layouts are supported (no weight download required to verify):
///
/// 1. **openai/gpt-oss-*** native MXFP4: fused `gate_up_proj_blocks` /
///    `gate_up_proj_scales` + `down_proj_blocks` / `down_proj_scales`, and
///    `self_attn.sinks`.
/// 2. **mlx-community/gpt-oss-*-MXFP4-Q4**: split experts
///    `experts.{gate,up,down}_proj.weight` (+ `.scales` sidecars) and
///    `self_attn.sinks`.
pub(crate) const GPT_OSS_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    // Per-head attention sinks (both openai + mlx-community).
    (
        "self_attn.sinks",
        TensorMapping::PerLayer(NativeTensorRole::AttnSink),
    ),
    // MoE router.
    (
        "mlp.router.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    // --- openai native MXFP4 fused experts (underscore suffix) ---
    (
        "mlp.experts.gate_up_proj_blocks",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateUpExpsMxfp4Blocks),
    ),
    (
        "mlp.experts.gate_up_proj_scales",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateUpExpsMxfp4Scales),
    ),
    (
        "mlp.experts.down_proj_blocks",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExpsMxfp4Blocks),
    ),
    (
        "mlp.experts.down_proj_scales",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExpsMxfp4Scales),
    ),
    // --- mlx-community split experts (weight+scales; mode from quant config) ---
    (
        "mlp.experts.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateExps),
    ),
    (
        "mlp.experts.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "mlp.experts.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
];

/// HuggingFace tensor name patterns shared by Qwen3/Gemma4.
///
/// The HuggingFace convention is:
///   model.embed_tokens.weight
///   model.layers.{i}.self_attn.q_proj.weight
///   model.layers.{i}.self_attn.k_proj.weight
///   model.layers.{i}.self_attn.v_proj.weight
///   model.layers.{i}.self_attn.o_proj.weight
///   model.layers.{i}.self_attn.q_norm.weight        (Qwen3, Gemma4)
///   model.layers.{i}.self_attn.k_norm.weight        (Qwen3, Gemma4)
///   model.layers.{i}.input_layernorm.weight
///   model.layers.{i}.post_attention_layernorm.weight
///   model.layers.{i}.pre_feedforward_layernorm.weight
///   model.layers.{i}.post_feedforward_layernorm.weight
///   model.layers.{i}.pre_feedforward_layernorm_2.weight   (Gemma4 MoE)
///   model.layers.{i}.post_feedforward_layernorm_1.weight  (Gemma4 MoE)
///   model.layers.{i}.post_feedforward_layernorm_2.weight  (Gemma4 MoE)
///   model.layers.{i}.mlp.gate_proj.weight
///   model.layers.{i}.mlp.up_proj.weight
///   model.layers.{i}.mlp.down_proj.weight
///   model.norm.weight
///   lm_head.weight
///
/// MLX sanitises the `model.` prefix differently per family, but the
/// safetensors on disk use the HuggingFace names above.
pub(crate) const HF_STANDARD_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "model.embed_tokens.weight",
        TensorMapping::Global(NativeTensorRole::TokenEmbedding),
    ),
    (
        "model.norm.weight",
        TensorMapping::Global(NativeTensorRole::FinalNorm),
    ),
    (
        "lm_head.weight",
        TensorMapping::Global(NativeTensorRole::LmHead),
    ),
    // Per-layer input gating global weights (Gemma4 2B/4B, sanitised model. prefix)
    (
        "model.embed_tokens_per_layer.weight",
        TensorMapping::Global(NativeTensorRole::PerLayerEmbedding),
    ),
    (
        "model.per_layer_model_projection.weight",
        TensorMapping::Global(NativeTensorRole::PerLayerModelProjection),
    ),
    (
        "model.per_layer_projection_norm.weight",
        TensorMapping::Global(NativeTensorRole::PerLayerProjectionNorm),
    ),
    // per-layer attention
    (
        "self_attn.q_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQ),
    ),
    (
        "self_attn.k_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionK),
    ),
    (
        "self_attn.v_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionV),
    ),
    (
        "self_attn.o_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionO),
    ),
    (
        "self_attn.q_norm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQNorm),
    ),
    (
        "self_attn.k_norm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionKNorm),
    ),
    // per-layer attention (packed QKV variant)
    (
        "self_attn.qkv_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQkvPacked),
    ),
    // per-layer norms
    (
        "input_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionNorm),
    ),
    (
        "post_attention_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionPostNorm),
    ),
    (
        "pre_feedforward_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnNorm),
    ),
    (
        "pre_feedforward_layernorm_2.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnNorm2),
    ),
    (
        "post_feedforward_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnPostNorm),
    ),
    (
        "post_feedforward_layernorm_1.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnPostNorm1),
    ),
    (
        "post_feedforward_layernorm_2.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnPostNorm2),
    ),
    // per-layer FFN
    (
        "mlp.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGate),
    ),
    (
        "router.proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    (
        "router.scale",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInpScale),
    ),
    (
        "router.per_expert_scale",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInpExpertScale),
    ),
    (
        "layer_scalar",
        TensorMapping::PerLayer(NativeTensorRole::LayerScalar),
    ),
    // Per-layer input gating (Gemma4 2B/4B)
    (
        "per_layer_input_gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::PerLayerInputGate),
    ),
    (
        "per_layer_projection.weight",
        TensorMapping::PerLayer(NativeTensorRole::PerLayerInputProjection),
    ),
    (
        "post_per_layer_input_norm.weight",
        TensorMapping::PerLayer(NativeTensorRole::PerLayerInputPostNorm),
    ),
    (
        "mlp.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUp),
    ),
    (
        "mlp.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDown),
    ),
    (
        "mlp.shared_expert_gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertGateInp),
    ),
    (
        "mlp.shared_expert.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertGate),
    ),
    (
        "mlp.shared_expert.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertUp),
    ),
    (
        "mlp.shared_expert.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertDown),
    ),
    // packed gate+up
    (
        "mlp.gate_up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateUpPacked),
    ),
    (
        "experts.gate_up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateUpExpsPacked),
    ),
    (
        "experts.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
    (
        "experts.switch_glu.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateExps),
    ),
    (
        "experts.switch_glu.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "experts.switch_glu.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
];

/// Nemotron-H (`nemotron_h`) tensors under `backbone.layers.{N}.mixer.*`.
///
/// Each layer is a single residual mixer (Mamba-2 / attention / MoE), not an
/// attn+FFN sandwich. Global embeddings / final norm are matched separately via
/// the backbone prefix path in `match_tensor`.
pub(crate) const NEMOTRON_H_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    // Pre-mixer RMSNorm (shared role name used as the layer residual norm).
    (
        "norm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionNorm),
    ),
    // Full-attention mixer (*).
    (
        "mixer.q_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQ),
    ),
    (
        "mixer.k_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionK),
    ),
    (
        "mixer.v_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionV),
    ),
    (
        "mixer.o_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionO),
    ),
    // Mamba-2 mixer (M). Packed in_proj → LinearAttentionInProjQkvz; D reuses
    // LayerScalar so we do not grow NativeTensorRole for a one-family residual.
    (
        "mixer.in_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjQkvz),
    ),
    (
        "mixer.conv1d.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionConv1d),
    ),
    (
        "mixer.dt_bias",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionDtBias),
    ),
    (
        "mixer.A_log",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionALog),
    ),
    (
        "mixer.D",
        TensorMapping::PerLayer(NativeTensorRole::LayerScalar),
    ),
    (
        "mixer.norm.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionNorm),
    ),
    (
        "mixer.out_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionOutProj),
    ),
    // MoE mixer (E): ReLU² experts (fc1/fc2) + shared expert up/down only.
    (
        "mixer.gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    (
        "mixer.gate.e_score_correction_bias",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInpCorrectionBias),
    ),
    (
        "mixer.switch_mlp.fc1.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "mixer.switch_mlp.fc2.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
    (
        "mixer.shared_experts.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertUp),
    ),
    (
        "mixer.shared_experts.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertDown),
    ),
];

pub(crate) const QWEN35_LINEAR_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "linear_attn.in_proj_qkv.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjQkv),
    ),
    (
        "linear_attn.in_proj_qkvz.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjQkvz),
    ),
    (
        "linear_attn.in_proj_z.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjZ),
    ),
    (
        "linear_attn.in_proj_a.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjA),
    ),
    (
        "linear_attn.in_proj_b.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjB),
    ),
    (
        "linear_attn.in_proj_ba.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjBa),
    ),
    (
        "linear_attn.conv1d.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionConv1d),
    ),
    (
        "linear_attn.dt_bias",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionDtBias),
    ),
    (
        "linear_attn.A_log",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionALog),
    ),
    (
        "linear_attn.norm.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionNorm),
    ),
    (
        "linear_attn.out_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionOutProj),
    ),
];

pub(crate) const GEMMA4_ASSISTANT_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "pre_projection.weight",
        TensorMapping::Global(NativeTensorRole::AssistantPreProjection),
    ),
    (
        "post_projection.weight",
        TensorMapping::Global(NativeTensorRole::AssistantPostProjection),
    ),
];

/// EmbeddingGemma sentence-transformers Dense projection head (applied after
/// mean pooling): `dense.0` (hidden → 4*hidden) then `dense.1` (4*hidden →
/// hidden), both bias-free identity-activation linears. The backbone tensors
/// use the standard `model.layers.*` map; these two are the only extras.
pub(crate) const EMBEDDINGGEMMA_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "dense.0.weight",
        TensorMapping::Global(NativeTensorRole::EmbeddingDense0),
    ),
    (
        "dense.1.weight",
        TensorMapping::Global(NativeTensorRole::EmbeddingDense1),
    ),
];

/// Extra tensors for Qwen3-VL vision tower (WS-V2).
///
/// HF names under `visual.*` (language_model prefix is stripped by convert).
/// Layer-indexed block tensors use the synthetic layer index carried in the
/// mapped name `blocks.{i}.*` via [`match_qwen3_vl_vision_layer`].
pub(crate) const QWEN3_VL_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "visual.patch_embed.proj.weight",
        TensorMapping::Global(NativeTensorRole::Qwen3VlVisionPatchEmbed),
    ),
    (
        "visual.merger.mlp.0.weight",
        TensorMapping::Global(NativeTensorRole::Qwen3VlVisionMerger),
    ),
    (
        "visual.merger.mlp.2.weight",
        TensorMapping::Global(NativeTensorRole::Qwen3VlVisionMerger),
    ),
    (
        "visual.merger.weight",
        TensorMapping::Global(NativeTensorRole::Qwen3VlVisionMerger),
    ),
];

/// Map `visual.blocks.{i}.{suffix}` patterns for Qwen3-VL vision layers.
pub(crate) fn match_qwen3_vl_vision_layer(name: &str) -> Option<(u32, NativeTensorRole)> {
    let rest = name.strip_prefix("visual.blocks.")?;
    let (idx_s, suffix) = rest.split_once('.')?;
    let idx: u32 = idx_s.parse().ok()?;
    let role = match suffix {
        "attn.qkv.weight" => NativeTensorRole::Qwen3VlVisionLayerQkv,
        "attn.proj.weight" => NativeTensorRole::Qwen3VlVisionLayerProj,
        "norm1.weight" => NativeTensorRole::Qwen3VlVisionLayerNorm1,
        "norm2.weight" => NativeTensorRole::Qwen3VlVisionLayerNorm2,
        "mlp.fc1.weight" | "mlp.up_proj.weight" => NativeTensorRole::Qwen3VlVisionLayerFc1,
        "mlp.fc2.weight" | "mlp.down_proj.weight" => NativeTensorRole::Qwen3VlVisionLayerFc2,
        _ => return None,
    };
    Some((idx, role))
}

/// Extra global tensors for Gemma4 Unified's encoder-free multimodal path.
///
/// This mirrors vLLM's `Gemma4UnifiedVisionEmbedder` plus
/// `Gemma4MultimodalEmbedder` modules:
/// raw patches -> LayerNorm -> Dense -> LayerNorm -> factorized pos emb ->
/// LayerNorm -> multimodal projection.
pub(crate) const GEMMA4_UNIFIED_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "vision_embedder.patch_dense.weight",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPatchDense),
    ),
    (
        "vision_embedder.patch_dense.bias",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPatchDenseBias),
    ),
    (
        "vision_embedder.patch_ln1.weight",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPatchNorm1),
    ),
    (
        "vision_embedder.patch_ln1.bias",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPatchNorm1Bias),
    ),
    (
        "vision_embedder.patch_ln2.weight",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPatchNorm2),
    ),
    (
        "vision_embedder.patch_ln2.bias",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPatchNorm2Bias),
    ),
    (
        "vision_embedder.pos_embedding",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPositionEmbedding),
    ),
    (
        "vision_embedder.pos_norm.weight",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPositionNorm),
    ),
    (
        "vision_embedder.pos_norm.bias",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPositionNormBias),
    ),
    (
        "embed_vision.embedding_projection.weight",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionProjection),
    ),
    (
        "embed_audio.embedding_projection.weight",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedAudioProjection),
    ),
];

/// DiffusionGemma stores tensors under `model.decoder.*` instead of `model.*`.
/// This map handles the global tensors with that prefix.
pub(crate) const DECODER_PREFIX_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "model.decoder.embed_tokens.weight",
        TensorMapping::Global(NativeTensorRole::TokenEmbedding),
    ),
    (
        "model.decoder.norm.weight",
        TensorMapping::Global(NativeTensorRole::FinalNorm),
    ),
    (
        "model.decoder.lm_head.weight",
        TensorMapping::Global(NativeTensorRole::LmHead),
    ),
    (
        "model.decoder.self_conditioning.pre_norm.weight",
        TensorMapping::Global(NativeTensorRole::DiffusionSelfConditionPreNorm),
    ),
    (
        "model.decoder.self_conditioning.gate_proj.weight",
        TensorMapping::Global(NativeTensorRole::DiffusionSelfConditionGate),
    ),
    (
        "model.decoder.self_conditioning.up_proj.weight",
        TensorMapping::Global(NativeTensorRole::DiffusionSelfConditionUp),
    ),
    (
        "model.decoder.self_conditioning.down_proj.weight",
        TensorMapping::Global(NativeTensorRole::DiffusionSelfConditionDown),
    ),
];

/// Gemma4 and Qwen3.5+ wrap the text model under `language_model.model.`, so
/// tensor names in safetensors appear as
/// `language_model.model.layers.0.self_attn.q_proj.weight`.
/// We also accept `model.layers.…` for already-sanitised weights.
pub(crate) const LANGUAGE_MODEL_PREFIX_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "language_model.model.embed_tokens.weight",
        TensorMapping::Global(NativeTensorRole::TokenEmbedding),
    ),
    (
        "language_model.model.norm.weight",
        TensorMapping::Global(NativeTensorRole::FinalNorm),
    ),
    (
        "language_model.lm_head.weight",
        TensorMapping::Global(NativeTensorRole::LmHead),
    ),
    // Per-layer input gating global weights (Gemma4 2B/4B, language_model. prefix)
    (
        "language_model.model.embed_tokens_per_layer.weight",
        TensorMapping::Global(NativeTensorRole::PerLayerEmbedding),
    ),
    (
        "language_model.model.per_layer_model_projection.weight",
        TensorMapping::Global(NativeTensorRole::PerLayerModelProjection),
    ),
    (
        "language_model.model.per_layer_projection_norm.weight",
        TensorMapping::Global(NativeTensorRole::PerLayerProjectionNorm),
    ),
];
