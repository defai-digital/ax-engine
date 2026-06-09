pub(crate) mod attention_mask;
pub mod disk_prefix_cache;
pub mod fastpath;
pub mod gemma4_assistant_mtp;
pub(crate) mod gemma4_unified;
pub mod generate;
pub mod kv_cache;
pub(crate) mod linear_attention_ops;
pub mod model;
pub mod mtp;
pub mod ngram_accel;
pub mod runner;
pub mod sampling;
pub mod speculation_profile;
pub mod turboquant;
pub mod turboquant_metal;
pub mod weight_rotation;
pub mod weights;

pub mod diagnostics {
    pub use crate::model::profile::{
        LinearAttentionProfileSnapshot, take_linear_attention_profile_snapshot,
    };
}

pub use runner::{EmbedCompileCacheStats, MlxPrefixCacheStore, MlxRunner};
