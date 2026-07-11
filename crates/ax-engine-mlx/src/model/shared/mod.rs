pub(super) mod attention;
pub(super) mod linear_attention;
pub(super) mod mla;
pub(super) mod mlp;
pub(super) mod norm;
pub(super) mod rope;
pub(super) mod utils;

// Re-exports for model/mod.rs and families/ modules.
#[cfg(test)]
pub(crate) use attention::build_layer_masks;
pub(crate) use attention::{
    KVConcatBuffer, attention_mask_array, attention_with_sinks, bidirectional_attention,
    build_layer_masks_for_forward, build_layer_masks_with_media_ranges,
    direct_qk_norm_rope_route_enabled_for_family, flatten_attention_output_bhsd,
    full_precision_attention, prepare_value_bhsd, prepare_value_bhsd_from_proj,
    prepare_value_bhsd_from_proj_flat, qk_norm_bhsd_from_proj, qk_norm_rope_bhsd_from_proj,
    qk_norm_rope_bhsd_from_proj_flat, qk_norm_rope_bhsd_from_proj_with_route,
    turboquant_decode_attention_experimental,
};
pub(crate) use linear_attention::linear_attention_forward;
pub(crate) use mla::glm_mla_attention_forward;
#[cfg(test)]
pub(crate) use mlp::per_layer_input_gate;
pub(crate) use mlp::{
    attention_output_projection, attention_output_projection_batched, ffn_swiglu,
    ffn_swiglu_batched, flatten_compiled_moe_inputs, flatten_gemma4_dual_path_inputs,
    moe_experts_forward, moe_experts_forward_gemma4, moe_experts_forward_with_cloned_weights,
    moe_experts_forward_with_shared, moe_router_deepseek_v3, moe_router_gemma4, moe_router_glm,
    moe_router_gpt_oss, moe_router_qwen3, per_layer_input_gate_project, qkv_project,
    qkv_project_batched, qkv_project_embed, shared_expert_forward,
};
pub(crate) use norm::rms_norm_opt;
pub(super) use rope::build_llama3_rope_freqs;
pub(crate) use utils::scale_hidden_pub;
pub(crate) use utils::{
    ProjectionBatchPolicy, add_then_multiply_scalar, apply_final_logit_softcap, qw, qw_with_policy,
    scale_hidden, shape_element_count,
};

// Additional re-exports used by test code (via `use super::*` in #[cfg(test)] mod).
#[cfg(test)]
pub(super) use attention::{
    TurboQuantQueryReadbackArray, attention_mask_key_len, bhsd_view_from_proj, qk_norm_bshd,
    turboquant_attention_output_array_from_flat, turboquant_query_readback_array,
};
#[cfg(test)]
pub(super) use mla::{
    glm_mla_embed_q_decode, glm_mla_embed_q_prefill, glm_mla_project_and_cache_inputs,
    glm_mla_project_inputs, glm_mla_unembed_out,
};
#[cfg(test)]
pub(super) use mlp::{geglu, moe_router_glm_from_logits, swiglu, switch_gather_inputs};
#[cfg(test)]
pub(super) use utils::{QkvSlices, qkv_slices, scalar_like};
