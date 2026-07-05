pub(crate) mod attention_mask;
pub mod batched_decode_session;
pub mod batched_kv_cache;
pub mod batched_sampling;
pub(crate) mod diffusion;
pub mod disk_prefix_cache;
pub mod fastpath;
pub mod gemma4_assistant_mtp;
pub(crate) mod gemma4_unified;
pub mod generate;
pub mod kv_cache;
// Public for the kernel-dispatch probe binaries (`src/bin/`), like the
// sibling modules; not a stable external API.
pub mod linear_attention_ops;
pub mod model;
pub mod mtp;
pub mod ngram_accel;
pub mod per_layer_compile;
pub mod runner;
pub mod sampling;
pub mod speculation_profile;
pub mod turboquant;
pub mod turboquant_metal;
pub mod weight_rotation;
pub mod weights;

pub mod diagnostics {
    pub use crate::model::profile::{
        EmbedProfileSnapshot, LinearAttentionProfileSnapshot, MoeProfileSnapshot,
        take_embed_profile_snapshot, take_linear_attention_profile_snapshot,
        take_moe_profile_snapshot,
    };
}

pub use runner::{EmbedCompileCacheStats, MlxPrefixCacheStore, MlxRunner, MlxSharedWeightsCell};
