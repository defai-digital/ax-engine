use mlx_sys::{MlxArray, MlxDtype, slice, slice_update, zeros};

/// Pre-allocated chunk size (tokens).  The buffer grows by this amount each time
/// the logical sequence length exceeds capacity, so the number of grow operations
/// per session is at most ceil(total_tokens / CHUNK).
const KV_CHUNK_TOKENS: usize = 256;

fn chunk_ceiling(n: usize) -> usize {
    n.div_ceil(KV_CHUNK_TOKENS) * KV_CHUNK_TOKENS
}

#[derive(Clone)]
struct LayerKV {
    /// Full backing buffer: `[1, n_kv_heads, capacity, head_dim]`.
    k: MlxArray,
    v: MlxArray,
    n_kv_heads: i32,
    head_dim: i32,
    capacity: usize,
    dtype: MlxDtype,
}

#[derive(Clone, Default)]
struct LinearLayerState {
    /// Qwen3.5 gated-delta conv tail: `[1, conv_kernel - 1, conv_dim]`.
    conv_state: Option<MlxArray>,
    /// Qwen3.5 gated-delta recurrent state: `[1, value_heads, value_dim, key_dim]`.
    recurrent_state: Option<MlxArray>,
}

/// Per-request attention cache with chunked KV pre-allocation.
///
/// Full-attention KV shape convention:
/// `[1, n_kv_heads, seq_len, head_dim]` (batch=1, SDPA-native format).
///
/// ## Growth strategy
///
/// Unlike the naive approach that calls `concatenate` on every append (O(n) data
/// movement per step), this cache pre-allocates buffers in `KV_CHUNK_TOKENS`-sized
/// blocks and uses `slice_update` to write new tokens into the pre-allocated region.
/// Buffer growth (via concatenation with zeros) happens at most every `KV_CHUNK_TOKENS`
/// steps — typically 0–1 times per request for common prompt+decode lengths.
///
/// ## Speculative rollback
///
/// `trim_to(prefix_len)` only updates `seq_len`.  The "trimmed" positions remain in
/// the backing buffer but are beyond the logical boundary, so SDPA never sees them.
/// The next `append` overwrites from `prefix_len`, restoring correctness.
pub struct MlxKVCache {
    layers: Vec<Option<LayerKV>>,
    linear_layers: Vec<LinearLayerState>,
    /// Current logical sequence length (token count cached).
    pub seq_len: usize,
}

impl Clone for MlxKVCache {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            linear_layers: self.linear_layers.clone(),
            seq_len: self.seq_len,
        }
    }
}

impl MlxKVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            linear_layers: (0..num_layers)
                .map(|_| LinearLayerState::default())
                .collect(),
            seq_len: 0,
        }
    }

    /// Append new K/V tokens for `layer` and return the full logical K/V for SDPA.
    ///
    /// `new_k` / `new_v` shape: `[1, n_kv_heads, new_tokens, head_dim]`
    ///
    /// Returns **owned** arrays sliced to `[1, n_kv_heads, seq_len + new_tokens, head_dim]`.
    pub fn append(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
    ) -> (MlxArray, MlxArray) {
        let shape = new_k.shape();
        let new_tokens = shape[2] as usize;
        let write_start = self.seq_len;
        let write_end = write_start + new_tokens;
        let n_kv_heads = shape[1];
        let head_dim = shape[3];
        let dtype = new_k.dtype();

        let entry = &mut self.layers[layer];
        match entry {
            None => {
                let capacity = chunk_ceiling(write_end);
                let buf_shape = [1i32, n_kv_heads, capacity as i32, head_dim];
                let k_buf = zeros(&buf_shape, dtype, None);
                let v_buf = zeros(&buf_shape, dtype, None);
                let start = [0i32, 0, write_start as i32, 0];
                let stop = [1i32, n_kv_heads, write_end as i32, head_dim];
                let strides = [1i32, 1, 1, 1];
                let k_out = slice_update(&k_buf, &new_k, &start, &stop, &strides, None);
                let v_out = slice_update(&v_buf, &new_v, &start, &stop, &strides, None);
                *entry = Some(LayerKV {
                    k: k_out,
                    v: v_out,
                    n_kv_heads,
                    head_dim,
                    capacity,
                    dtype,
                });
            }
            Some(lkv) => {
                if write_end > lkv.capacity {
                    // Grow: allocate a larger buffer and copy existing data.
                    let new_capacity = chunk_ceiling(write_end);
                    let buf_shape = [1i32, lkv.n_kv_heads, new_capacity as i32, lkv.head_dim];
                    let k_new = zeros(&buf_shape, lkv.dtype, None);
                    let v_new = zeros(&buf_shape, lkv.dtype, None);
                    let old_stop = [1i32, lkv.n_kv_heads, lkv.capacity as i32, lkv.head_dim];
                    let zero_start = [0i32, 0, 0, 0];
                    let ones = [1i32, 1, 1, 1];
                    lkv.k = slice_update(&k_new, &lkv.k, &zero_start, &old_stop, &ones, None);
                    lkv.v = slice_update(&v_new, &lkv.v, &zero_start, &old_stop, &ones, None);
                    lkv.capacity = new_capacity;
                }
                let start = [0i32, 0, write_start as i32, 0];
                let stop = [1i32, lkv.n_kv_heads, write_end as i32, lkv.head_dim];
                let strides = [1i32, 1, 1, 1];
                lkv.k = slice_update(&lkv.k, &new_k, &start, &stop, &strides, None);
                lkv.v = slice_update(&lkv.v, &new_v, &start, &stop, &strides, None);
            }
        }

        let lkv = self.layers[layer].as_ref().unwrap();
        let end = write_end as i32;
        let k_view = slice(
            &lkv.k,
            &[0, 0, 0, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        let v_view = slice(
            &lkv.v,
            &[0, 0, 0, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        (k_view, v_view)
    }

    /// Trim the logical boundary to `prefix_len` tokens (speculative rollback).
    ///
    /// With chunked layout this is O(1) — no array data is modified.  The backing
    /// buffer retains its pre-allocated capacity.  The next `append` writes from
    /// `prefix_len`, overwriting any rejected speculative positions.
    ///
    /// # Preconditions
    /// `prefix_len` must be ≤ `seq_len`.  Callers that pass a value beyond the
    /// current sequence length silently corrupt the cache (forward beyond trim).
    pub fn trim_to(&mut self, prefix_len: usize) {
        debug_assert!(
            prefix_len <= self.seq_len,
            "trim_to({prefix_len}) called with seq_len={} — would extend, not trim",
            self.seq_len
        );
        self.seq_len = prefix_len;
    }

    /// Collect refs to all K and V backing buffers for bulk `eval`.
    ///
    /// Pass these alongside the output token to `mlx_sys::eval()` after each
    /// decode step.  Without this, every `slice_update` append leaves the
    /// backing buffer as a lazy graph node pointing at the previous step's
    /// buffer; after N steps each buffer is a chain of N `slice_update` nodes.
    /// Evaluating here materialises the chain into a flat buffer so the next
    /// step's `slice_update` has depth-1 ancestry instead of depth-N.
    ///
    /// Mirrors mlx_lm's `mx.eval(y, cache)` pattern.
    pub fn collect_eval_refs(&self) -> Vec<&MlxArray> {
        let mut refs = Vec::with_capacity(self.layers.len() * 4);
        for lkv in self.layers.iter().flatten() {
            refs.push(&lkv.k);
            refs.push(&lkv.v);
        }
        for linear in &self.linear_layers {
            if let Some(conv_state) = &linear.conv_state {
                refs.push(conv_state);
            }
            if let Some(recurrent_state) = &linear.recurrent_state {
                refs.push(recurrent_state);
            }
        }
        refs
    }

    /// Read the cached gated-delta states for a Qwen3.5 linear-attention layer.
    pub fn linear_state(&self, layer: usize) -> (Option<&MlxArray>, Option<&MlxArray>) {
        let state = &self.linear_layers[layer];
        (state.conv_state.as_ref(), state.recurrent_state.as_ref())
    }

    /// Store the gated-delta states for a Qwen3.5 linear-attention layer.
    pub fn set_linear_state(
        &mut self,
        layer: usize,
        conv_state: MlxArray,
        recurrent_state: MlxArray,
    ) {
        let state = &mut self.linear_layers[layer];
        state.conv_state = Some(conv_state);
        state.recurrent_state = Some(recurrent_state);
    }

    /// Read K/V already written by `source_layer` during the current forward pass.
    ///
    /// Used by KV-shared layers (e.g. Gemma4 layers 24-41) that attend against
    /// a prior layer's cache instead of computing their own K/V projections.
    ///
    /// `new_tokens` must equal the number of new tokens being processed this
    /// step (matches the `write_end = seq_len + new_tokens` slice written by the
    /// source layer's `append` call earlier in the same forward pass).
    pub fn peek_source_kv(&self, source_layer: usize, new_tokens: usize) -> (MlxArray, MlxArray) {
        let lkv = self.layers[source_layer]
            .as_ref()
            .expect("KV-shared source layer has no cached KV — source layer must appear earlier");
        let end = (self.seq_len + new_tokens) as i32;
        let k_view = slice(
            &lkv.k,
            &[0, 0, 0, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        let v_view = slice(
            &lkv.v,
            &[0, 0, 0, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        (k_view, v_view)
    }

    /// Reset cache entirely (e.g., between requests).
    pub fn reset(&mut self) {
        for entry in &mut self.layers {
            *entry = None;
        }
        for state in &mut self.linear_layers {
            *state = LinearLayerState::default();
        }
        self.seq_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_state_is_eval_tracked_and_reset() {
        let mut cache = MlxKVCache::new(2);
        let conv = zeros(&[1, 3, 14], MlxDtype::Float32, None);
        let recurrent = zeros(&[1, 4, 8, 6], MlxDtype::Float32, None);

        cache.set_linear_state(1, conv, recurrent);

        let (conv_state, recurrent_state) = cache.linear_state(1);
        assert_eq!(conv_state.expect("conv state").shape(), vec![1, 3, 14]);
        assert_eq!(
            recurrent_state.expect("recurrent state").shape(),
            vec![1, 4, 8, 6]
        );
        assert_eq!(cache.collect_eval_refs().len(), 2);

        cache.reset();

        let (conv_state, recurrent_state) = cache.linear_state(1);
        assert!(conv_state.is_none());
        assert!(recurrent_state.is_none());
        assert!(cache.collect_eval_refs().is_empty());
    }

    #[test]
    fn linear_state_survives_trim_to() {
        let mut cache = MlxKVCache::new(1);
        let conv = zeros(&[1, 3, 14], MlxDtype::Float32, None);
        let recurrent = zeros(&[1, 4, 8, 6], MlxDtype::Float32, None);

        cache.seq_len = 8;
        cache.set_linear_state(0, conv, recurrent);
        cache.trim_to(4);

        assert_eq!(cache.seq_len, 4);
        let (conv_state, recurrent_state) = cache.linear_state(0);
        assert!(
            conv_state.is_some() && recurrent_state.is_some(),
            "linear recurrent state is not rolled back by seq_len trim"
        );
    }

    #[test]
    fn clone_preserves_linear_state_for_speculative_branch() {
        let mut cache = MlxKVCache::new(1);
        let conv = zeros(&[1, 3, 14], MlxDtype::Float32, None);
        let recurrent = zeros(&[1, 4, 8, 6], MlxDtype::Float32, None);

        cache.seq_len = 12;
        cache.set_linear_state(0, conv, recurrent);
        let branch = cache.clone();
        cache.reset();

        assert_eq!(branch.seq_len, 12);
        let (conv_state, recurrent_state) = branch.linear_state(0);
        assert!(conv_state.is_some());
        assert!(recurrent_state.is_some());
    }
}
