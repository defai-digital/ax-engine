//! Hybrid recurrent cache/state for Qwen3.5.
//!
//! Qwen3.5 mixes full-attention layers and recurrent GDN layers. Attention
//! layers still need a normal KV cache, while recurrent layers need:
//! - a short convolution history of length `conv_kernel - 1`
//! - the persistent delta-net state tensor

use super::CpuKv;
use super::page::{KvCacheConfig, KvDtype, recommended_page_size};

/// CPU-side hybrid cache for Qwen3.5.
#[derive(Debug)]
pub struct Qwen35Kv {
    attention: CpuKv,
    seq_len: usize,
    recurrent_layers: Vec<bool>,
    conv_cache_len: usize,
    conv_dim: usize,
    recurrent_state_len: usize,
    conv_states: Vec<Vec<f32>>,
    recurrent_states: Vec<Vec<f32>>,
}

impl Qwen35Kv {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        full_attention_interval: usize,
        conv_kernel: usize,
        inner_size: usize,
        state_size: usize,
        time_step_rank: usize,
        group_count: usize,
    ) -> Self {
        let page_size = recommended_page_size(n_kv_heads, head_dim);
        Self::new_with_attention_page_size(
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            page_size,
            full_attention_interval,
            conv_kernel,
            inner_size,
            state_size,
            time_step_rank,
            group_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_attention_page_size(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        attention_page_size: usize,
        full_attention_interval: usize,
        conv_kernel: usize,
        inner_size: usize,
        state_size: usize,
        time_step_rank: usize,
        group_count: usize,
    ) -> Self {
        assert!(
            full_attention_interval > 0,
            "qwen35 full_attention_interval must be > 0"
        );
        assert!(conv_kernel > 0, "qwen35 conv_kernel must be > 0");
        assert!(inner_size > 0, "qwen35 inner_size must be > 0");
        assert!(state_size > 0, "qwen35 state_size must be > 0");
        assert!(time_step_rank > 0, "qwen35 time_step_rank must be > 0");
        assert!(group_count > 0, "qwen35 group_count must be > 0");
        assert!(
            inner_size == state_size * time_step_rank,
            "qwen35 inner_size ({inner_size}) must equal state_size ({state_size}) * time_step_rank ({time_step_rank})"
        );
        assert!(
            time_step_rank.is_multiple_of(group_count),
            "qwen35 time_step_rank ({time_step_rank}) must be a multiple of group_count ({group_count})"
        );
        let recurrent_layers = (0..n_layers)
            .map(|layer| (layer + 1) % full_attention_interval != 0)
            .collect::<Vec<_>>();
        let conv_cache_len = conv_kernel.saturating_sub(1);
        let conv_dim = inner_size + 2 * group_count * state_size;
        let recurrent_state_len = time_step_rank * state_size * state_size;
        Self {
            attention: CpuKv::with_config(&KvCacheConfig {
                n_layers,
                n_kv_heads,
                head_dim,
                max_seq_len,
                page_size: attention_page_size,
                dtype: KvDtype::F32,
            }),
            seq_len: 0,
            recurrent_layers: recurrent_layers.clone(),
            conv_cache_len,
            conv_dim,
            recurrent_state_len,
            conv_states: (0..n_layers)
                .map(|layer| {
                    if recurrent_layers[layer] {
                        vec![0.0f32; conv_cache_len * conv_dim]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            recurrent_states: (0..n_layers)
                .map(|layer| {
                    if recurrent_layers[layer] {
                        vec![0.0f32; recurrent_state_len]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
        }
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn is_recurrent_layer(&self, layer: usize) -> bool {
        self.recurrent_layers[layer]
    }

    pub fn attention_append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        self.attention.append_and_advance(layer, k, v);
    }

    pub fn attention_append_batch(
        &mut self,
        layer: usize,
        k_batch: &[f32],
        v_batch: &[f32],
        n_tokens: usize,
    ) {
        self.attention
            .append_batch(layer, k_batch, v_batch, n_tokens);
    }

    pub fn attention_k_slice_including_current(&self, layer: usize, n: usize) -> &[f32] {
        self.attention.k_slice_including_current(layer, n)
    }

    pub fn attention_v_slice_including_current(&self, layer: usize, n: usize) -> &[f32] {
        self.attention.v_slice_including_current(layer, n)
    }

    pub fn conv_state_mut(&mut self, layer: usize) -> &mut [f32] {
        &mut self.conv_states[layer]
    }

    pub fn recurrent_state_mut(&mut self, layer: usize) -> &mut [f32] {
        &mut self.recurrent_states[layer]
    }

    pub fn recurrent_buffers_mut(&mut self, layer: usize) -> (&mut [f32], &mut [f32]) {
        (
            &mut self.conv_states[layer],
            &mut self.recurrent_states[layer],
        )
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
        self.attention.clear();
        for state in &mut self.conv_states {
            state.fill(0.0);
        }
        for state in &mut self.recurrent_states {
            state.fill(0.0);
        }
    }

    pub fn truncate_to(&mut self, pos: usize) {
        if pos >= self.seq_len {
            return;
        }

        if pos == 0 {
            self.clear();
            return;
        }

        // Qwen3.5 recurrent layers need historical state snapshots to support
        // arbitrary rollback. AX does not store those yet, so the only safe
        // fallback is to clear the hybrid state.
        tracing::warn!(
            pos,
            seq_len = self.seq_len,
            "Qwen35Kv truncate_to() falls back to clear() because recurrent history snapshots are not stored"
        );
        self.clear();
    }

    pub fn finalize_token(&mut self) {
        self.attention.finalize_token();
        self.seq_len += 1;
    }

    pub fn finalize_batch(&mut self, n_tokens: usize) {
        self.attention.finalize_batch(n_tokens);
        self.seq_len += n_tokens;
    }

    pub fn conv_dim(&self) -> usize {
        self.conv_dim
    }

    pub fn conv_cache_len(&self) -> usize {
        self.conv_cache_len
    }

    pub fn recurrent_state_len(&self) -> usize {
        self.recurrent_state_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen35_kv_layer_pattern() {
        let kv = Qwen35Kv::new(8, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        assert!(kv.is_recurrent_layer(0));
        assert!(kv.is_recurrent_layer(1));
        assert!(kv.is_recurrent_layer(2));
        assert!(!kv.is_recurrent_layer(3));
        assert!(kv.is_recurrent_layer(4));
        assert!(kv.is_recurrent_layer(5));
        assert!(kv.is_recurrent_layer(6));
        assert!(!kv.is_recurrent_layer(7));
    }

    #[test]
    fn test_qwen35_kv_clear_resets_state() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        kv.conv_state_mut(0).fill(1.0);
        kv.recurrent_state_mut(0).fill(2.0);
        kv.clear();
        assert_eq!(kv.seq_len(), 0);
        assert!(kv.conv_state_mut(0).iter().all(|&v| v == 0.0));
        assert!(kv.recurrent_state_mut(0).iter().all(|&v| v == 0.0));
    }

    #[test]
    #[should_panic(expected = "multiple of group_count")]
    fn test_qwen35_kv_rejects_incompatible_head_expansion() {
        let _ = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 768, 128, 6, 4);
    }
}
