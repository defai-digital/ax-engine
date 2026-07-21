pub(crate) mod artifact_identity;
pub(crate) mod attention_mask;
pub mod batched_decode_certification;
pub mod batched_decode_session;
pub mod batched_kv_cache;
pub mod batched_linear_state;
pub mod batched_sampling;
pub(crate) mod diffusion;
pub mod disk_prefix_cache;
pub mod fastpath;
pub mod gemma4_assistant_mtp;
pub(crate) mod gemma4_unified;
pub mod generate;
pub mod kv_block_pool;
pub mod kv_cache;
pub mod unlimited_ocr;
// Public for the kernel-dispatch probe binaries (`src/bin/`), like the
// sibling modules; not a stable external API.
pub mod linear_attention_ops;
pub mod model;
pub mod mtp;
pub mod mtp_adaptive_gate;
pub mod ngram_accel;
pub mod per_layer_compile;
pub mod runner;
pub mod sampling;
pub mod speculation_profile;
pub mod weight_rotation;
pub mod weights;

pub mod diagnostics {
    pub use crate::model::profile::{
        EmbedProfileSnapshot, LinearAttentionProfileSnapshot, MoeProfileSnapshot,
        MoeRouterFusedSnapshot, take_embed_profile_snapshot,
        take_linear_attention_profile_snapshot, take_moe_profile_snapshot,
        take_moe_router_fused_snapshot,
    };
}

pub use runner::{EmbedCompileCacheStats, MlxPrefixCacheStore, MlxRunner, MlxSharedWeightsCell};

/// Clear process-global compiled graphs and MLX allocator caches.
pub fn clear_process_caches() {
    per_layer_compile::clear_all_layer_decode_caches();
    mlx_sys::clear_cache();
}
