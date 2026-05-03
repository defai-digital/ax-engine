use mlx_sys::{MlxArray, concatenate, slice};

/// Per-layer KV cache stored as MLX arrays growing along the sequence axis.
///
/// Shape convention: `[1, n_kv_heads, seq_len, head_dim]` (batch=1, SDPA-native format).
pub struct MlxKVCache {
    /// One (keys, values) pair per transformer layer.
    layers: Vec<Option<(MlxArray, MlxArray)>>,
    /// Current sequence length (number of tokens cached).
    pub seq_len: usize,
}

impl MlxKVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
            seq_len: 0,
        }
    }

    /// Return keys and values for layer `i`, if any exist.
    pub fn get(&self, layer: usize) -> Option<(&MlxArray, &MlxArray)> {
        self.layers[layer].as_ref().map(|(k, v)| (k, v))
    }

    /// Append new key/value slices for layer `i` and return the full (k, v).
    ///
    /// `new_k` and `new_v` shape: `[1, n_kv_heads, new_tokens, head_dim]`
    pub fn append(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
    ) -> (&MlxArray, &MlxArray) {
        let entry = &mut self.layers[layer];
        match entry {
            None => {
                *entry = Some((new_k, new_v));
            }
            Some((cached_k, cached_v)) => {
                let k = concatenate(&[cached_k, &new_k], 2, None);
                let v = concatenate(&[cached_v, &new_v], 2, None);
                *entry = Some((k, v));
            }
        }
        let (k, v) = entry.as_ref().unwrap();
        (k, v)
    }

    /// Trim cache to `prefix_len` tokens (for prefix reuse).
    ///
    /// Slices K and V along the sequence axis (axis 2) rather than dropping the
    /// cache entirely, so the valid prefix is preserved for the next decode step.
    pub fn trim_to(&mut self, prefix_len: usize) {
        for entry in &mut self.layers {
            if let Some((k, v)) = entry {
                let shape = k.shape();
                let seq = shape.get(2).copied().unwrap_or(0) as usize;
                if prefix_len < seq {
                    // Shape: [1, n_kv_heads, seq, head_dim] — slice axis 2.
                    let n_kv_heads = shape[1];
                    let head_dim = shape[3];
                    let plen = prefix_len as i32;
                    let start = [0i32, 0, 0, 0];
                    let stop = [1i32, n_kv_heads, plen, head_dim];
                    let strides = [1i32; 4];
                    let k_trimmed = slice(k, &start, &stop, &strides, None);
                    let v_trimmed = slice(v, &start, &stop, &strides, None);
                    *entry = Some((k_trimmed, v_trimmed));
                }
            }
        }
        self.seq_len = prefix_len;
    }

    /// Reset cache entirely.
    pub fn reset(&mut self) {
        for entry in &mut self.layers {
            *entry = None;
        }
        self.seq_len = 0;
    }
}
