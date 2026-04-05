//! KV cache page sizing and growth policy.
//!
//! Instead of pre-allocating the full context length upfront, the KV cache
//! grows on demand in page-sized increments. The page size is tuned to the
//! model's KV stride (n_kv_heads * head_dim) to balance memory waste against
//! growth overhead.
//!
//! Target: ~256KB per page per K/V component per layer, which gives:
//!   - Tiny models  (stride ≤128):  page_size = 512 tokens
//!   - Small models (stride ~256):  page_size = 256 tokens
//!   - Medium models (stride ~1024): page_size = 64 tokens
//!   - Large models (stride ~2048+): page_size = 32 tokens

use crate::model::config::ModelConfig;

/// KV cache data type — controls memory format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvDtype {
    /// Full-precision f32 storage (4 bytes per value).
    #[default]
    F32,
    /// Quantized 8-bit storage with per-token absmax scaling (~1 byte per value).
    /// Achieves ~4× memory compression with <1% accuracy loss.
    Q8,
}

/// Target bytes per page per K/V component per layer.
const TARGET_PAGE_BYTES: usize = 256 * 1024; // 256 KB

/// Minimum page size in tokens.
const MIN_PAGE_SIZE: usize = 16;

/// Maximum page size in tokens.
const MAX_PAGE_SIZE: usize = 512;

/// KV cache configuration with model-tuned page size.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    /// Number of tokens per growth page.
    pub page_size: usize,
    /// Storage data type (F32 or Q8).
    pub dtype: KvDtype,
}

impl KvCacheConfig {
    /// Create a KvCacheConfig from model parameters with auto-tuned page size.
    ///
    /// Auto-enables Q8 quantization when the F32 KV cache would exceed 1GB at
    /// full context length. This gives ~4× memory compression with <1% accuracy
    /// loss, critical for >7B models at long context.
    pub fn from_model(config: &ModelConfig) -> Self {
        let n_kv_heads = config.n_kv_heads as usize;
        let head_dim = config.head_dim as usize;
        let n_layers = config.n_layers as usize;
        let max_seq_len = config.context_length as usize;
        let page_size = recommended_page_size(n_kv_heads, head_dim);

        // F32 KV memory: 2 (K+V) × n_layers × max_seq_len × n_kv_heads × head_dim × 4 bytes
        let f32_kv_bytes =
            2 * n_layers * max_seq_len * n_kv_heads * head_dim * std::mem::size_of::<f32>();
        let dtype = if f32_kv_bytes > 1_000_000_000 {
            tracing::info!(
                "Auto-enabling Q8 KV cache: F32 would require {:.1}GB at {max_seq_len} context",
                f32_kv_bytes as f64 / 1e9
            );
            KvDtype::Q8
        } else {
            KvDtype::F32
        };

        Self {
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            page_size,
            dtype,
        }
    }

    /// Bytes per value based on dtype.
    fn value_size(&self) -> usize {
        match self.dtype {
            // i8 data + f32 scale per token (amortized over stride)
            // Per token: stride * 1 + 4 bytes (scale)
            KvDtype::Q8 => 1,
            KvDtype::F32 => std::mem::size_of::<f32>(),
        }
    }

    /// Bytes per token per layer for one K or V component.
    pub fn bytes_per_token(&self) -> usize {
        let stride = self.n_kv_heads * self.head_dim;
        stride * self.value_size()
            + match self.dtype {
                KvDtype::Q8 => std::mem::size_of::<f32>(), // per-token scale
                KvDtype::F32 => 0,
            }
    }

    /// Bytes per page per layer for one K or V component.
    pub fn bytes_per_page(&self) -> usize {
        self.page_size * self.bytes_per_token()
    }

    /// Total initial memory allocated (K + V, all layers, 1 page each).
    pub fn initial_memory_bytes(&self) -> usize {
        2 * self.n_layers * self.bytes_per_page()
    }

    /// Total maximum memory if fully allocated (K + V, all layers, full context).
    pub fn max_memory_bytes(&self) -> usize {
        2 * self.n_layers * self.max_seq_len * self.bytes_per_token()
    }
}

/// Resolve the initial token capacity for a paged KV cache.
///
/// Current KV implementations allocate exactly one growth page up front unless
/// the model's max context is smaller than that page.
pub fn initial_token_capacity(page_size: usize, max_seq_len: usize) -> usize {
    page_size.max(1).min(max_seq_len)
}

/// Resolve the capacity required to hold `needed` tokens under a paged-growth
/// policy.
///
/// Returns `None` when `needed` exceeds `max_seq_len` or when the policy cannot
/// make further progress.
pub fn planned_capacity_for_needed(
    current_capacity: usize,
    needed: usize,
    growth_tokens: usize,
    max_seq_len: usize,
) -> Option<usize> {
    if needed > max_seq_len {
        return None;
    }

    let growth_tokens = growth_tokens.max(1);
    let mut capacity = current_capacity.max(initial_token_capacity(growth_tokens, max_seq_len));
    while capacity < needed {
        let next = (capacity + growth_tokens).min(max_seq_len);
        if next == capacity {
            return None;
        }
        capacity = next;
    }

    Some(capacity)
}

/// Recommend a page size (in tokens) based on model KV dimensions.
///
/// Targets [`TARGET_PAGE_BYTES`] per page per K/V component per layer.
/// Result is clamped to [`MIN_PAGE_SIZE`]..=[`MAX_PAGE_SIZE`] and rounded
/// to the nearest power of two for alignment.
///
/// Examples:
/// - Llama 7B  (n_kv_heads=8, head_dim=128, stride=1024): 64 tokens/page
/// - Llama 13B (n_kv_heads=8, head_dim=128, stride=1024): 64 tokens/page
/// - Llama 70B (n_kv_heads=8, head_dim=128, stride=1024): 64 tokens/page
/// - Phi-2     (n_kv_heads=32, head_dim=80, stride=2560): 32 tokens/page
/// - TinyLlama (n_kv_heads=4, head_dim=64, stride=256):  256 tokens/page
pub fn recommended_page_size(n_kv_heads: usize, head_dim: usize) -> usize {
    let stride = n_kv_heads * head_dim;
    if stride == 0 {
        return MIN_PAGE_SIZE;
    }

    let bytes_per_token = stride * std::mem::size_of::<f32>();
    let ideal = TARGET_PAGE_BYTES / bytes_per_token;

    // Round to nearest power of two, then clamp
    let rounded = ideal.next_power_of_two();
    // If rounding up doubled too much, try rounding down
    let rounded = if rounded > ideal * 3 / 2 {
        rounded / 2
    } else {
        rounded
    };

    rounded.clamp(MIN_PAGE_SIZE, MAX_PAGE_SIZE)
}

/// Tracks page allocation statistics for monitoring.
#[derive(Debug, Default, Clone)]
pub struct PageAllocator {
    /// Number of pages currently allocated.
    pub pages_allocated: usize,
    /// Number of pages at full capacity.
    pub pages_full: usize,
    /// Peak pages allocated.
    pub peak_pages: usize,
    /// Total grow operations performed.
    pub grow_count: usize,
}

impl PageAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a page growth event.
    pub fn record_grow(&mut self, new_pages: usize) {
        self.pages_allocated += new_pages;
        self.grow_count += 1;
        if self.pages_allocated > self.peak_pages {
            self.peak_pages = self.pages_allocated;
        }
    }

    /// Record that existing pages are now full.
    pub fn record_fill(&mut self, count: usize) {
        self.pages_full += count;
    }

    /// Reset stats (e.g. on cache clear).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Fraction of allocated pages that are full (0.0-1.0).
    pub fn utilization(&self) -> f64 {
        if self.pages_allocated == 0 {
            return 0.0;
        }
        (self.pages_full as f64 / self.pages_allocated as f64).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommended_page_size_small_model() {
        // TinyLlama: n_kv_heads=4, head_dim=64, stride=256
        // bytes_per_token = 256 * 4 = 1024
        // ideal = 262144 / 1024 = 256
        let ps = recommended_page_size(4, 64);
        assert_eq!(ps, 256, "TinyLlama should get 256 tokens/page");
    }

    #[test]
    fn test_recommended_page_size_medium_model() {
        // Llama 7B: n_kv_heads=8, head_dim=128, stride=1024
        // bytes_per_token = 1024 * 4 = 4096
        // ideal = 262144 / 4096 = 64
        let ps = recommended_page_size(8, 128);
        assert_eq!(ps, 64, "Llama 7B should get 64 tokens/page");
    }

    #[test]
    fn test_recommended_page_size_large_model() {
        // Phi-2 or wide model: n_kv_heads=32, head_dim=80, stride=2560
        // bytes_per_token = 2560 * 4 = 10240
        // ideal = 262144 / 10240 = 25 → round to 32
        let ps = recommended_page_size(32, 80);
        assert_eq!(ps, 32, "Wide model should get 32 tokens/page");
    }

    #[test]
    fn test_recommended_page_size_very_small() {
        // Tiny test model: n_kv_heads=1, head_dim=4, stride=4
        // bytes_per_token = 16
        // ideal = 262144 / 16 = 16384 → capped at 512
        let ps = recommended_page_size(1, 4);
        assert_eq!(ps, MAX_PAGE_SIZE, "tiny stride should cap at MAX_PAGE_SIZE");
    }

    #[test]
    fn test_recommended_page_size_very_large() {
        // Huge model: n_kv_heads=96, head_dim=256, stride=24576
        // bytes_per_token = 24576 * 4 = 98304
        // ideal = 262144 / 98304 = 2 → clamped to MIN_PAGE_SIZE
        let ps = recommended_page_size(96, 256);
        assert_eq!(
            ps, MIN_PAGE_SIZE,
            "huge stride should floor at MIN_PAGE_SIZE"
        );
    }

    #[test]
    fn test_recommended_page_size_zero() {
        assert_eq!(recommended_page_size(0, 128), MIN_PAGE_SIZE);
        assert_eq!(recommended_page_size(8, 0), MIN_PAGE_SIZE);
    }

    #[test]
    fn test_page_size_is_power_of_two() {
        for kv_heads in [1, 2, 4, 8, 16, 32, 64] {
            for head_dim in [32, 64, 80, 96, 128, 256] {
                let ps = recommended_page_size(kv_heads, head_dim);
                assert!(
                    ps.is_power_of_two(),
                    "page_size {ps} for kv_heads={kv_heads}, head_dim={head_dim} is not power of 2"
                );
            }
        }
    }

    #[test]
    fn test_kv_cache_config_from_model() {
        let model = ModelConfig {
            architecture: "llama".into(),
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            embedding_dim: 4096,
            head_dim: 128,
            intermediate_dim: 11008,
            context_length: 4096,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
            gemma4_head_dim_swa: None,
            gemma4_head_dim_global: None,
            gemma4_n_kv_heads_swa: None,
            gemma4_n_kv_heads_global: None,
            gemma4_rope_dim_swa: None,
            gemma4_rope_dim_global: None,
            final_logit_softcapping: None,
            expert_intermediate_dim: None,
        };
        let cfg = KvCacheConfig::from_model(&model);
        assert_eq!(cfg.n_layers, 32);
        assert_eq!(cfg.page_size, 64);
        assert_eq!(cfg.max_seq_len, 4096);
        // Llama 7B at 4096 ctx: F32 KV = 1.07GB > 1GB threshold → auto Q8
        assert_eq!(cfg.dtype, KvDtype::Q8);
        // Q8: stride * 1 + 4 (scale) = 1024 + 4 = 1028 bytes/token
        assert_eq!(cfg.bytes_per_token(), 1028);
    }

    #[test]
    fn test_kv_cache_config_memory_estimates() {
        let cfg = KvCacheConfig {
            n_layers: 32,
            n_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
            page_size: 64,
            dtype: KvDtype::F32,
        };
        // Per token per layer per component: 8*128*4 = 4096 bytes
        assert_eq!(cfg.bytes_per_token(), 4096);
        // Per page: 64 * 4096 = 262144 = 256KB
        assert_eq!(cfg.bytes_per_page(), 262144);
        // Initial: 2 * 32 * 262144 = 16MB
        assert_eq!(cfg.initial_memory_bytes(), 2 * 32 * 262144);
        // Max: 2 * 32 * 4096 * 4096 = 1GB
        assert_eq!(cfg.max_memory_bytes(), 2 * 32 * 4096 * 4096);
    }

    #[test]
    fn test_kv_cache_config_q8_memory() {
        let cfg = KvCacheConfig {
            n_layers: 32,
            n_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
            page_size: 64,
            dtype: KvDtype::Q8,
        };
        // Per token per layer per component: 8*128*1 + 4 (scale) = 1028 bytes
        assert_eq!(cfg.bytes_per_token(), 1028);
        // ~4× compression vs f32
        let f32_cfg = KvCacheConfig {
            dtype: KvDtype::F32,
            ..cfg.clone()
        };
        let ratio = f32_cfg.bytes_per_token() as f64 / cfg.bytes_per_token() as f64;
        assert!(
            ratio > 3.9,
            "Q8 should give ~4× compression, got {ratio:.2}×"
        );
    }

    #[test]
    fn test_initial_token_capacity_uses_one_page() {
        assert_eq!(initial_token_capacity(64, 4096), 64);
        assert_eq!(initial_token_capacity(64, 32), 32);
        assert_eq!(initial_token_capacity(0, 32), 1);
    }

    #[test]
    fn test_planned_capacity_for_needed_grows_in_page_steps() {
        assert_eq!(planned_capacity_for_needed(64, 65, 64, 4096), Some(128));
        assert_eq!(planned_capacity_for_needed(64, 191, 64, 4096), Some(192));
        assert_eq!(planned_capacity_for_needed(64, 4096, 64, 4096), Some(4096));
    }

    #[test]
    fn test_planned_capacity_for_needed_rejects_out_of_bounds_requests() {
        assert_eq!(planned_capacity_for_needed(64, 4097, 64, 4096), None);
        assert_eq!(planned_capacity_for_needed(4096, 4097, 64, 4096), None);
    }

    #[test]
    fn test_q8_auto_enable_large_model() {
        // Llama 7B: 32 layers, 8 kv_heads, 128 dim, 4096 ctx
        // F32 KV = 2 * 32 * 4096 * 8 * 128 * 4 = 1,073,741,824 = 1.07GB > 1GB → Q8
        let model = ModelConfig {
            architecture: "llama".into(),
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            embedding_dim: 4096,
            head_dim: 128,
            intermediate_dim: 11008,
            context_length: 4096,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
            gemma4_head_dim_swa: None,
            gemma4_head_dim_global: None,
            gemma4_n_kv_heads_swa: None,
            gemma4_n_kv_heads_global: None,
            gemma4_rope_dim_swa: None,
            gemma4_rope_dim_global: None,
            final_logit_softcapping: None,
            expert_intermediate_dim: None,
        };
        let cfg = KvCacheConfig::from_model(&model);
        assert_eq!(
            cfg.dtype,
            KvDtype::Q8,
            "7B at 4096 ctx should auto-enable Q8"
        );
    }

    #[test]
    fn test_q8_auto_enable_small_model_stays_f32() {
        // Small model: 4 layers, 4 kv_heads, 64 dim, 512 ctx
        // F32 KV = 2 * 4 * 512 * 4 * 64 * 4 = 4,194,304 = 4MB < 1GB → F32
        let model = ModelConfig {
            architecture: "llama".into(),
            n_layers: 4,
            n_heads: 8,
            n_kv_heads: 4,
            embedding_dim: 512,
            head_dim: 64,
            intermediate_dim: 2048,
            context_length: 512,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
            gemma4_head_dim_swa: None,
            gemma4_head_dim_global: None,
            gemma4_n_kv_heads_swa: None,
            gemma4_n_kv_heads_global: None,
            gemma4_rope_dim_swa: None,
            gemma4_rope_dim_global: None,
            final_logit_softcapping: None,
            expert_intermediate_dim: None,
        };
        let cfg = KvCacheConfig::from_model(&model);
        assert_eq!(cfg.dtype, KvDtype::F32, "small model should stay F32");
    }

    #[test]
    fn test_page_allocator_stats() {
        let mut pa = PageAllocator::new();
        assert_eq!(pa.pages_allocated, 0);
        assert_eq!(pa.utilization(), 0.0);

        pa.record_grow(1);
        assert_eq!(pa.pages_allocated, 1);
        assert_eq!(pa.grow_count, 1);
        assert_eq!(pa.peak_pages, 1);

        pa.record_grow(2);
        assert_eq!(pa.pages_allocated, 3);
        assert_eq!(pa.grow_count, 2);
        assert_eq!(pa.peak_pages, 3);

        pa.record_fill(2);
        assert!((pa.utilization() - 2.0 / 3.0).abs() < 0.01);

        pa.reset();
        assert_eq!(pa.pages_allocated, 0);
        assert_eq!(pa.grow_count, 0);
    }
}
