//! Multi-head attention with Grouped Query Attention (GQA) support.
//!
//! Standard scaled dot-product attention:
//!   Attention(Q, K, V) = softmax(Q * K^T / sqrt(head_dim)) * V
//!
//! GQA: n_heads query heads share n_kv_heads key/value heads.
//! Each KV head serves (n_heads / n_kv_heads) query heads.

use crate::compute::softmax;

/// Attention dimension parameters.
#[derive(Debug, Clone, Copy)]
pub struct AttentionParams {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionParams {
    pub fn new(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        assert!(
            n_heads >= n_kv_heads && n_heads.is_multiple_of(n_kv_heads),
            "n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})"
        );
        Self {
            n_heads,
            n_kv_heads,
            head_dim,
        }
    }

    fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    fn heads_per_kv(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    fn kv_stride(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    fn q_stride(&self) -> usize {
        self.n_heads * self.head_dim
    }
}

/// Compute dot product of two slices (NEON-vectorized on aarch64).
#[cfg(target_arch = "aarch64")]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len().min(b.len());
    let chunks = n / 4;
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    unsafe {
        for i in 0..chunks {
            let av = vld1q_f32(a.as_ptr().add(i * 4));
            let bv = vld1q_f32(b.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, av, bv);
        }
    }
    let mut sum = unsafe { vaddvq_f32(acc) };
    for (&ai, &bi) in a[chunks * 4..n].iter().zip(b[chunks * 4..n].iter()) {
        sum += ai * bi;
    }
    sum
}

#[cfg(not(target_arch = "aarch64"))]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (ai, bi) in a.iter().zip(b.iter()) {
        sum += ai * bi;
    }
    sum
}

/// Accumulate: out += weight * vec (NEON-vectorized on aarch64).
#[cfg(target_arch = "aarch64")]
fn accumulate(out: &mut [f32], weight: f32, vec: &[f32]) {
    use std::arch::aarch64::*;

    let n = out.len().min(vec.len());
    let chunks = n / 4;
    let wv = unsafe { vdupq_n_f32(weight) };
    unsafe {
        for i in 0..chunks {
            let ov = vld1q_f32(out.as_ptr().add(i * 4));
            let vv = vld1q_f32(vec.as_ptr().add(i * 4));
            vst1q_f32(out.as_mut_ptr().add(i * 4), vfmaq_f32(ov, wv, vv));
        }
    }
    for (o, &v) in out[chunks * 4..n].iter_mut().zip(vec[chunks * 4..n].iter()) {
        *o += weight * v;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn accumulate(out: &mut [f32], weight: f32, vec: &[f32]) {
    for (o, &v) in out.iter_mut().zip(vec.iter()) {
        *o += weight * v;
    }
}

/// Compute multi-head attention for a single token (decode step).
///
/// Given the current token's query vectors and the full KV cache history,
/// compute attention output for all query heads.
///
/// * `q` - Query vectors: [n_heads * head_dim] (current token only)
/// * `k_cache` - Key cache: [seq_len * n_kv_heads * head_dim]
/// * `v_cache` - Value cache: [seq_len * n_kv_heads * head_dim]
/// * `output` - Output buffer: [n_heads * head_dim]
/// * `params` - Attention dimension parameters
/// * `seq_len` - Number of tokens in KV cache (including current)
pub fn multi_head_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    output: &mut [f32],
    params: &AttentionParams,
    seq_len: usize,
) {
    let AttentionParams {
        n_heads,
        n_kv_heads,
        head_dim,
    } = *params;

    assert_eq!(q.len(), n_heads * head_dim);
    assert!(k_cache.len() >= seq_len * n_kv_heads * head_dim);
    assert!(v_cache.len() >= seq_len * n_kv_heads * head_dim);
    assert_eq!(output.len(), n_heads * head_dim);

    if seq_len == 0 {
        output.fill(0.0);
        return;
    }

    let scale = params.scale();
    let heads_per_kv = params.heads_per_kv();
    let kv_stride = params.kv_stride();

    let mut scores = vec![0.0f32; seq_len];

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];

        // Q * K^T / sqrt(d)
        for (t, score) in scores.iter_mut().enumerate() {
            let k_offset = t * kv_stride + kv_h * head_dim;
            *score = dot(q_head, &k_cache[k_offset..k_offset + head_dim]) * scale;
        }

        // Softmax
        softmax::softmax(&mut scores);

        // Weighted sum of values
        let out_head = &mut output[h * head_dim..(h + 1) * head_dim];
        out_head.fill(0.0);

        for (t, &w) in scores.iter().enumerate() {
            let v_offset = t * kv_stride + kv_h * head_dim;
            accumulate(out_head, w, &v_cache[v_offset..v_offset + head_dim]);
        }
    }
}

/// Compute multi-head attention for multiple tokens (prefill).
///
/// Processes all tokens in a sequence with causal masking
/// (each token can only attend to itself and previous tokens).
///
/// * `q` - Query vectors: [n_tokens * n_heads * head_dim]
/// * `k` - Key vectors: [n_tokens * n_kv_heads * head_dim]
/// * `v` - Value vectors: [n_tokens * n_kv_heads * head_dim]
/// * `output` - Output buffer: [n_tokens * n_heads * head_dim]
/// * `n_tokens` - Number of tokens in the sequence
/// * `params` - Attention dimension parameters
pub fn multi_head_attention_prefill(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    n_tokens: usize,
    params: &AttentionParams,
) {
    let AttentionParams {
        n_heads,
        n_kv_heads: _,
        head_dim,
    } = *params;

    let q_stride = params.q_stride();
    let kv_stride = params.kv_stride();
    let heads_per_kv = params.heads_per_kv();
    let scale = params.scale();

    assert!(q.len() >= n_tokens * q_stride);
    assert!(k.len() >= n_tokens * kv_stride);
    assert!(v.len() >= n_tokens * kv_stride);
    assert!(output.len() >= n_tokens * q_stride);

    // Pre-allocate scores buffer at max causal length to avoid per-token heap allocation
    let mut scores = vec![0.0f32; n_tokens];

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;

        for qi in 0..n_tokens {
            let q_offset = qi * q_stride + h * head_dim;
            let q_head = &q[q_offset..q_offset + head_dim];

            // Causal: token qi can attend to tokens 0..=qi
            let attend_len = qi + 1;
            let scores_slice = &mut scores[..attend_len];

            // Q * K^T / sqrt(d)
            for (t, score) in scores_slice.iter_mut().enumerate() {
                let k_offset = t * kv_stride + kv_h * head_dim;
                *score = dot(q_head, &k[k_offset..k_offset + head_dim]) * scale;
            }

            softmax::softmax(scores_slice);

            // Weighted sum of V
            let out_offset = qi * q_stride + h * head_dim;
            let out_head = &mut output[out_offset..out_offset + head_dim];
            out_head.fill(0.0);

            for (t, &w) in scores_slice.iter().enumerate() {
                let v_offset = t * kv_stride + kv_h * head_dim;
                accumulate(out_head, w, &v[v_offset..v_offset + head_dim]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> AttentionParams {
        AttentionParams::new(n_heads, n_kv_heads, head_dim)
    }

    #[test]
    fn test_attention_single_token() {
        // 1 head, head_dim=2, seq_len=1
        // softmax([s]) = [1.0] → output = V
        let q = [1.0, 0.0];
        let k = [1.0, 0.0];
        let v = [3.0, 4.0];
        let mut out = [0.0f32; 2];
        multi_head_attention(&q, &k, &v, &mut out, &params(1, 1, 2), 1);
        assert!((out[0] - 3.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 4.0).abs() < 1e-5, "out[1]={}", out[1]);
    }

    #[test]
    fn test_attention_two_tokens_uniform() {
        // Equal Q*K scores → softmax = [0.5, 0.5] → average of V rows
        let q = [1.0, 0.0];
        let k = [1.0, 0.0, 1.0, 0.0];
        let v = [2.0, 0.0, 0.0, 2.0];
        let mut out = [0.0f32; 2];
        multi_head_attention(&q, &k, &v, &mut out, &params(1, 1, 2), 2);
        assert!((out[0] - 1.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 1.0).abs() < 1e-5, "out[1]={}", out[1]);
    }

    #[test]
    fn test_attention_concentrates_on_matching_key() {
        // Large Q*K score for first key → softmax heavily favors it
        let q = [10.0, 0.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let v = [1.0, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 2];
        multi_head_attention(&q, &k, &v, &mut out, &params(1, 1, 2), 2);
        assert!(out[0] > 0.99, "expected ~1.0, got {}", out[0]);
        assert!(out[1] < 0.01, "expected ~0.0, got {}", out[1]);
    }

    #[test]
    fn test_attention_multi_head() {
        // 2 heads, 2 KV heads, head_dim=2, seq_len=1
        let q = [1.0, 0.0, 0.0, 1.0]; // head0: [1,0], head1: [0,1]
        let k = [1.0, 0.0, 0.0, 1.0]; // kv0: [1,0], kv1: [0,1]
        let v = [5.0, 6.0, 7.0, 8.0]; // kv0: [5,6], kv1: [7,8]
        let mut out = [0.0f32; 4];
        multi_head_attention(&q, &k, &v, &mut out, &params(2, 2, 2), 1);
        assert!((out[0] - 5.0).abs() < 1e-5);
        assert!((out[1] - 6.0).abs() < 1e-5);
        assert!((out[2] - 7.0).abs() < 1e-5);
        assert!((out[3] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_attention_gqa() {
        // GQA: 4 query heads, 2 KV heads, head_dim=2, seq_len=1
        // Heads 0,1 → kv_head 0; heads 2,3 → kv_head 1
        let q = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let v = [10.0, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 8];
        multi_head_attention(&q, &k, &v, &mut out, &params(4, 2, 2), 1);
        // Heads 0,1 → kv0 V = [10, 20]
        assert!((out[0] - 10.0).abs() < 1e-5);
        assert!((out[1] - 20.0).abs() < 1e-5);
        assert!((out[2] - 10.0).abs() < 1e-5);
        assert!((out[3] - 20.0).abs() < 1e-5);
        // Heads 2,3 → kv1 V = [30, 40]
        assert!((out[4] - 30.0).abs() < 1e-5);
        assert!((out[5] - 40.0).abs() < 1e-5);
        assert!((out[6] - 30.0).abs() < 1e-5);
        assert!((out[7] - 40.0).abs() < 1e-5);
    }

    #[test]
    fn test_attention_empty_sequence() {
        let q = [1.0, 0.0];
        let k: [f32; 0] = [];
        let v: [f32; 0] = [];
        let mut out = [999.0f32; 2];
        multi_head_attention(&q, &k, &v, &mut out, &params(1, 1, 2), 0);
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn test_attention_convex_combination() {
        // Output should be in convex hull of V rows
        let q = [1.0, 2.0];
        let k = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = [1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let mut out = [0.0f32; 2];
        multi_head_attention(&q, &k, &v, &mut out, &params(1, 1, 2), 3);
        assert!(out[0] >= -0.01 && out[0] <= 1.01, "out[0]={}", out[0]);
        assert!(out[1] >= -0.01 && out[1] <= 1.01, "out[1]={}", out[1]);
    }

    #[test]
    fn test_prefill_single_token() {
        let q = [1.0, 0.0];
        let k = [1.0, 0.0];
        let v = [3.0, 4.0];
        let mut out = [0.0f32; 2];
        multi_head_attention_prefill(&q, &k, &v, &mut out, 1, &params(1, 1, 2));
        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_prefill_causal_masking() {
        // Token 0 sees only itself; token 1 sees both
        let q = [1.0, 0.0, 1.0, 0.0];
        let k = [1.0, 0.0, 1.0, 0.0];
        let v = [2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 4];
        multi_head_attention_prefill(&q, &k, &v, &mut out, 2, &params(1, 1, 2));
        // Token 0: only V[0] = [2, 3]
        assert!((out[0] - 2.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-5, "out[1]={}", out[1]);
        // Token 1: uniform over V[0], V[1] → avg = [3, 4]
        assert!((out[2] - 3.0).abs() < 1e-5, "out[2]={}", out[2]);
        assert!((out[3] - 4.0).abs() < 1e-5, "out[3]={}", out[3]);
    }

    #[test]
    fn test_prefill_gqa() {
        // 2 query heads, 1 KV head, head_dim=2, 1 token
        let q = [1.0, 0.0, 0.0, 1.0];
        let k = [1.0, 0.0];
        let v = [5.0, 6.0];
        let mut out = [0.0f32; 4];
        multi_head_attention_prefill(&q, &k, &v, &mut out, 1, &params(2, 1, 2));
        assert!((out[0] - 5.0).abs() < 1e-5);
        assert!((out[1] - 6.0).abs() < 1e-5);
        assert!((out[2] - 5.0).abs() < 1e-5);
        assert!((out[3] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_prefill_three_tokens_causal() {
        // 3 tokens, 1 head, head_dim=1 (simplified)
        // Q=K=[1,1,1], V=[10,20,30]
        let p = params(1, 1, 1);
        let q = [1.0, 1.0, 1.0];
        let k = [1.0, 1.0, 1.0];
        let v = [10.0, 20.0, 30.0];
        let mut out = [0.0f32; 3];
        multi_head_attention_prefill(&q, &k, &v, &mut out, 3, &p);
        // Token 0: sees only V[0]=10
        assert!((out[0] - 10.0).abs() < 1e-4, "out[0]={}", out[0]);
        // Token 1: uniform over V[0..2] → (10+20)/2 = 15
        assert!((out[1] - 15.0).abs() < 1e-4, "out[1]={}", out[1]);
        // Token 2: uniform over V[0..3] → (10+20+30)/3 = 20
        assert!((out[2] - 20.0).abs() < 1e-4, "out[2]={}", out[2]);
    }
}
