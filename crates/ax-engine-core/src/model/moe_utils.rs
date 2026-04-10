//! Shared MoE (Mixture of Experts) utility functions.
//!
//! Used by Qwen3.5 MoE and any future MoE architecture.

use crate::gguf::tensor::GgmlType;

/// Compute byte size of `n_elements` in the given quantization type.
pub(crate) fn expert_byte_stride(dtype: GgmlType, n_elements: usize) -> usize {
    let bs = dtype.block_size();
    // Quantized GGUF tensors still allocate a full trailing block for
    // partially filled experts, so the per-expert byte stride must round up.
    n_elements.div_ceil(bs) * dtype.bytes_per_block()
}

/// Extract the raw byte slice for a single expert from a concatenated weight tensor.
pub(crate) fn expert_quant_slice<'a>(
    full: &'a [u8],
    stride: usize,
    eid: usize,
    name: &str,
) -> anyhow::Result<&'a [u8]> {
    let start = eid
        .checked_mul(stride)
        .ok_or_else(|| anyhow::anyhow!("expert slice overflow for {name}"))?;
    let end = start
        .checked_add(stride)
        .ok_or_else(|| anyhow::anyhow!("expert slice overflow for {name}"))?;
    anyhow::ensure!(
        end <= full.len(),
        "expert slice out of bounds for {name}: expert={eid}, end={end}, len={}",
        full.len()
    );
    Ok(&full[start..end])
}

/// Softmax over all experts, then select top-k.
/// Matches llama.cpp: softmax(all logits) → argsort_top_k → extract weights.
pub(crate) fn top_k_softmax(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    if logits.is_empty() || k == 0 {
        return (Vec::new(), Vec::new());
    }

    // Step 1: Softmax over ALL expert logits
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs = if !max.is_finite() {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        let mut probs: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 && sum.is_finite() {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            probs.fill(1.0 / logits.len() as f32);
        }
        probs
    };

    // Step 2: Select top-k from softmax probabilities
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    indexed.truncate(k.min(indexed.len()));

    // Step 3: Re-normalize selected weights to sum to 1
    let sel_sum: f32 = indexed.iter().map(|x| x.1).sum();
    let weights: Vec<f32> = if sel_sum > 0.0 {
        indexed.iter().map(|x| x.1 / sel_sum).collect()
    } else {
        indexed.iter().map(|_| 1.0 / indexed.len() as f32).collect()
    };

    (indexed.iter().map(|x| x.0).collect(), weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── top_k_softmax tests ──────────────────────────────────────────

    #[test]
    fn test_top_k_selects_highest_values() {
        let logits = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 0.0];
        let (ids, _weights) = top_k_softmax(&logits, 2);
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], 3); // 7.0
        assert_eq!(ids[1], 5); // 6.0
    }

    #[test]
    fn test_top_k_weights_sum_to_one() {
        let logits = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 0.0];
        let (_ids, weights) = top_k_softmax(&logits, 3);
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights sum to {sum}, expected ~1.0"
        );
    }

    #[test]
    fn test_top_k_with_k_equals_n() {
        let logits = vec![3.0, 1.0, 2.0];
        let (ids, weights) = top_k_softmax(&logits, 3);
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], 0); // 3.0 is highest
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_single_expert() {
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let (ids, weights) = top_k_softmax(&logits, 1);
        assert_eq!(ids, vec![0]);
        assert!(
            (weights[0] - 1.0).abs() < 1e-5,
            "single expert weight should be ~1.0"
        );
    }

    #[test]
    fn test_top_k_equal_logits() {
        let logits = vec![5.0, 5.0, 5.0, 5.0];
        let (ids, weights) = top_k_softmax(&logits, 2);
        assert_eq!(ids, vec![0, 1]);
        assert!((weights[0] - weights[1]).abs() < 1e-5);
        assert!((weights[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_negative_logits() {
        let logits = vec![-1.0, -5.0, -3.0, -0.5];
        let (ids, _weights) = top_k_softmax(&logits, 2);
        assert_eq!(ids[0], 3); // -0.5 is highest
        assert_eq!(ids[1], 0); // -1.0 is second
    }

    #[test]
    fn test_top_k_zero_returns_empty() {
        let logits = vec![1.0, 2.0, 3.0];
        let (ids, weights) = top_k_softmax(&logits, 0);
        assert!(ids.is_empty());
        assert!(weights.is_empty());
    }

    #[test]
    fn test_top_k_all_non_finite_logits_is_deterministic_and_normalized() {
        let logits = vec![f32::NEG_INFINITY; 4];
        let (ids, weights) = top_k_softmax(&logits, 2);
        assert_eq!(ids, vec![0, 1]);
        assert!((weights[0] - 0.5).abs() < 1e-5);
        assert!((weights[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_k_larger_than_logits_len_still_normalizes_weights() {
        let logits = vec![f32::NEG_INFINITY; 2];
        let (ids, weights) = top_k_softmax(&logits, 4);
        assert_eq!(ids, vec![0, 1]);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights sum to {sum}");
    }

    // ── expert_byte_stride tests ─────────────────────────────────────

    #[test]
    fn test_expert_byte_stride_q4k() {
        let stride = expert_byte_stride(GgmlType::Q4K, 14336 * 4096);
        assert_eq!(stride, 229376 * 144);
    }

    #[test]
    fn test_expert_byte_stride_q8_0() {
        let stride = expert_byte_stride(GgmlType::Q8_0, 4096 * 4096);
        let n_blocks = (4096 * 4096) / 32;
        assert_eq!(stride, n_blocks * 34);
    }

    #[test]
    fn test_expert_byte_stride_rounds_up_partial_quant_block() {
        let stride = expert_byte_stride(GgmlType::Q8_0, 33);
        assert_eq!(stride, 2 * GgmlType::Q8_0.bytes_per_block());
    }

    #[test]
    fn test_expert_byte_stride_f32() {
        let stride = expert_byte_stride(GgmlType::F32, 1024);
        assert_eq!(stride, 1024 * 4);
    }
}
