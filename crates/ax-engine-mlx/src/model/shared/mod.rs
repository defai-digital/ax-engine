pub(super) mod attention;
pub(super) mod linear_attention;
pub(super) mod mla;
pub(super) mod mlp;
pub(super) mod norm;
pub(super) mod rope;
pub(super) mod utils;

// Re-exports for model/mod.rs and families/ modules.
pub(crate) use attention::{
    attention_mask_array, build_layer_masks, full_precision_attention, prepare_value_bhsd,
    qk_norm_bshd, turboquant_decode_attention_experimental,
};
pub(crate) use linear_attention::linear_attention_forward;
pub(crate) use mla::glm_mla_attention_forward;
pub(crate) use mlp::{
    attention_output_projection, ffn_swiglu, moe_experts_forward, moe_router_gemma4,
    moe_router_glm, moe_router_qwen3, qkv_project, shared_expert_forward,
};
pub(crate) use norm::rms_norm_opt;
pub(super) use rope::build_llama3_rope_freqs;
pub(crate) use utils::scale_hidden_pub;
pub(crate) use utils::{apply_final_logit_softcap, qw, scale_hidden, shape_element_count};

// Additional re-exports used by test code (via `use super::*` in #[cfg(test)] mod).
#[cfg(test)]
pub(super) use attention::{
    TurboQuantQueryReadbackArray, attention_mask_key_len, bhsd_view_from_proj,
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
