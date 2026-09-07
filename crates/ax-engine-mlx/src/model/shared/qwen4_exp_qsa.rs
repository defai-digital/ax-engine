//! QSA (Qwen Sparse Attention) full-attention branch for `qwen4_exp`
//! (Qwen3.8-Flash-Next layers 3, 7, …, 47).
//!
//! DSA-style selection-only sparsity: a per-layer indexer scores mean-pooled
//! block keys against per-head queries and picks the top-k blocks; attention
//! itself runs over the full-precision K/V rows at the selected positions and
//! never reads the compressed keys. The current incomplete block (≤ ratio-1
//! tokens) is always visible; there is no attention sink and no always-visible
//! local window. When the committed block count fits the budget
//! (`complete_blocks <= indexer_budget / compress_ratio`) the selection is the
//! identity and the layer falls back to plain causal SDPA.
//!
//! Authoritative spec: `.internal/planning/qwen38-flash-next-support.md`
//! sections "A. Indexer = QSA" and "E. Hybrid layer details". Cache layout:
//! `crate::kv_cache` `append_qwen4_exp_qsa` / `*_block_rows`.
//!
//! Text path only: mRoPE position ids arrive as 3 identical grids (T=H=W).
//! The interleaved mRoPE factor builder and partial-RoPE application below
//! mirror `crate::qwen3_vl::{build_interleaved_mrope, apply_interleaved_mrope}`;
//! they are intentionally duplicated (small, assert-based, qwen4_exp-specific
//! sections/θ) so the qwen3_vl path stays untouched.

use mlx_sys::{
    MlxArray, MlxDtype, ScaledDotProductAttentionMask, add, arange, astype, concatenate, divide,
    equal, eval, floor, greater_equal, less, logical_and, matmul, maximum, multiply, negative,
    power, put_along_axis, repeat_axis, reshape, scaled_dot_product_attention_with_mask, slice,
    subtract, sum_axis, transpose, where_cond, zeros,
};

use crate::kv_cache::MlxKVCache;
use crate::weights::{LayerWeights, Qwen4ExpIndexerWeights};

use super::super::config::ModelConfig;
use super::attention::flatten_attention_output_bhsd;
use super::mlp::{attention_output_projection, top_k_by_argpartition};
use super::utils::qw;

/// Geometry of one QSA layer, derived from the typed config once per forward.
/// Reference pack values in parentheses.
struct QsaGeometry {
    /// Query heads (24).
    n_heads: usize,
    /// KV heads (2) — GQA group 12.
    n_kv_heads: usize,
    /// Per-head width (256).
    head_dim: usize,
    /// Partial rotary width: first `rope_dims` of `head_dim` rotate (64).
    rope_dims: usize,
    /// Indexer query heads (4).
    idx_heads: usize,
    /// Indexer key heads (1) — raw keys are cached per token.
    idx_kv_heads: usize,
    /// Indexer head width (128).
    idx_dim: usize,
    /// Tokens per pooled block (`indexer_compress_ratio`, 4).
    ratio: usize,
    /// Blocks kept per query (`indexer_budget / ratio`, 512).
    block_topk: usize,
    eps: f32,
}

impl QsaGeometry {
    fn from_config(cfg: &ModelConfig) -> QsaGeometry {
        let Some(q4e) = cfg.qwen4_exp.as_ref() else {
            unreachable!("qwen4_exp QSA forward requires ModelConfig.qwen4_exp");
        };
        assert!(
            q4e.mrope_interleaved,
            "qwen4_exp QSA supports only interleaved mRoPE sections"
        );
        assert!(
            cfg.rope_dims > 0
                && cfg.rope_dims.is_multiple_of(2)
                && cfg.rope_dims <= cfg.head_dim
                && cfg.rope_dims <= q4e.indexer_head_dim,
            "qwen4_exp QSA partial rotary {} must be positive, even, and fit both head_dim {} and indexer dim {}",
            cfg.rope_dims,
            cfg.head_dim,
            q4e.indexer_head_dim
        );
        assert!(
            q4e.indexer_compress_ratio > 0,
            "qwen4_exp QSA indexer_compress_ratio must be positive"
        );
        let ratio = q4e.indexer_compress_ratio;
        let block_topk = q4e.indexer_budget / ratio;
        assert!(
            block_topk > 0,
            "qwen4_exp QSA indexer_budget {} must cover at least one block of {} tokens",
            q4e.indexer_budget,
            ratio
        );
        QsaGeometry {
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            rope_dims: cfg.rope_dims,
            idx_heads: q4e.indexer_n_heads,
            idx_kv_heads: q4e.indexer_kv_heads,
            idx_dim: q4e.indexer_head_dim,
            ratio,
            block_topk,
            eps: cfg.rms_norm_eps,
        }
    }
}

/// Precomputed interleaved-mRoPE factors: `[1, positions, rotary_dim]` f32
/// with `cat([freqs, freqs])` layout (matches `rotate_half` pairing).
pub(crate) struct Qwen4ExpQsaMrope {
    pub cos: MlxArray,
    pub sin: MlxArray,
}

/// Build interleaved mRoPE cos/sin factors for `positions` (T/H/W grids).
///
/// Recomposition per `sections` (`mrope_section` [11, 11, 10] in the pack):
/// the T grid drives every frequency by default, the H grid takes indices
/// 1::3, the W grid 2::3. `inv_freq[i] = theta^(-2i/rotary_dim)` and cos/sin
/// duplicate each half (`cat([f, f])`), so pairing is the NeoX-style
/// `(x1, x2) -> (x1*cos - x2*sin, x2*cos + x1*sin)` over halves of the rotary
/// width. Mirrors `qwen3_vl::build_interleaved_mrope` on CPU by design.
pub(crate) fn build_qsa_mrope(
    positions: &[[i32; 3]],
    rotary_dim: usize,
    theta: f32,
    sections: &[u32],
) -> Qwen4ExpQsaMrope {
    assert!(
        rotary_dim > 0 && rotary_dim.is_multiple_of(2),
        "qwen4_exp QSA rotary_dim {rotary_dim} must be a positive even number"
    );
    assert!(
        sections.len() == 3 && sections.iter().sum::<u32>() as usize * 2 == rotary_dim,
        "qwen4_exp QSA mrope_section {sections:?} must have 3 entries covering rotary_dim {rotary_dim}"
    );
    assert!(
        !positions.is_empty(),
        "qwen4_exp QSA mRoPE requires at least one position"
    );
    let half = rotary_dim / 2;
    let mut source_axis = vec![0usize; half];
    for (axis, section) in sections.iter().copied().enumerate().skip(1) {
        let mut index = axis;
        while index < section as usize * 3 && index < half {
            source_axis[index] = axis;
            index += 3;
        }
    }
    let inv_freq: Vec<f32> = (0..half)
        .map(|index| 1.0 / theta.powf((2 * index) as f32 / rotary_dim as f32))
        .collect();
    let mut cos_values = Vec::with_capacity(positions.len() * rotary_dim);
    let mut sin_values = Vec::with_capacity(positions.len() * rotary_dim);
    for position in positions {
        let frequencies: Vec<f32> = (0..half)
            .map(|index| position[source_axis[index]] as f32 * inv_freq[index])
            .collect();
        for _ in 0..2 {
            cos_values.extend(frequencies.iter().map(|frequency| frequency.cos()));
            sin_values.extend(frequencies.iter().map(|frequency| frequency.sin()));
        }
    }
    let shape = [1, positions.len() as i32, rotary_dim as i32];
    Qwen4ExpQsaMrope {
        cos: f32_array(&cos_values, &shape),
        sin: f32_array(&sin_values, &shape),
    }
}

/// `rotate_half` over the last dim: `cat([-x2, x1])` with `x1`/`x2` the two
/// halves. Copied from `qwen3_vl::rotate_half` (kept private there).
fn rotate_half(input: &MlxArray) -> MlxArray {
    let shape = input.shape();
    let ndim = shape.len();
    let half = shape[ndim - 1] / 2;
    let mut first_start = vec![0; ndim];
    let mut first_stop = shape.clone();
    first_stop[ndim - 1] = half;
    let first = slice(input, &first_start, &first_stop, &vec![1; ndim], None);
    first_start[ndim - 1] = half;
    first_stop[ndim - 1] = shape[ndim - 1];
    let second = slice(input, &first_start, &first_stop, &vec![1; ndim], None);
    concatenate(&[&negative(&second, None), &first], -1, None)
}

/// Apply precomputed mRoPE factors to `[B, H, S, D]` Q/K, rotating only the
/// first `rotary_dim` dims (partial rotary 0.25) and passing the rest through.
/// Mirrors `qwen3_vl::apply_interleaved_mrope`.
pub(crate) fn apply_qsa_mrope(
    tensor: &MlxArray,
    factors: &Qwen4ExpQsaMrope,
    rotary_dim: usize,
) -> MlxArray {
    let shape = tensor.shape();
    let batch = shape[0];
    let heads = shape[1];
    let seq = shape[2];
    let head_dim = shape[3];
    let rotary_dim = rotary_dim as i32;
    let rotated_input = slice(
        tensor,
        &[0, 0, 0, 0],
        &[batch, heads, seq, rotary_dim],
        &[1, 1, 1, 1],
        None,
    );
    let pass = (rotary_dim < head_dim).then(|| {
        slice(
            tensor,
            &[0, 0, 0, rotary_dim],
            &[batch, heads, seq, head_dim],
            &[1, 1, 1, 1],
            None,
        )
    });
    let cos = astype(&factors.cos, tensor.dtype(), None);
    let sin = astype(&factors.sin, tensor.dtype(), None);
    let cos = reshape(&cos, &[batch, 1, seq, rotary_dim], None);
    let sin = reshape(&sin, &[batch, 1, seq, rotary_dim], None);
    let first = multiply(&rotated_input, &cos, None);
    let second = multiply(&rotate_half(&rotated_input), &sin, None);
    let embedded = add(&first, &second, None);
    pass.map_or(embedded.clone(), |pass| {
        concatenate(&[&embedded, &pass], -1, None)
    })
}

/// Per-head RMSNorm with the `1 + γ` gain convention of qwen4_exp
/// (`Qwen4ExpTextRMSNorm`), computed in f32 over the last dim and returned in
/// f32. Callers cast back to the activation dtype where the graph requires it.
pub(crate) fn rms_norm_one_plus_gamma(x: &MlxArray, weight: &MlxArray, eps: f32) -> MlxArray {
    let shape = x.shape();
    let dim = *shape.last().unwrap_or(&0);
    assert!(dim > 0, "qwen4_exp QSA RMSNorm requires a non-scalar input");
    assert_eq!(
        weight.shape(),
        vec![dim],
        "qwen4_exp QSA RMSNorm weight must cover the normed dim"
    );
    let x32 = astype(x, MlxDtype::Float32, None);
    let sqr = multiply(&x32, &x32, None);
    let mean_sqr = divide(
        &sum_axis(&sqr, shape.len() as i32 - 1, true, None),
        &mlx_sys::ops::cached_scalar(dim as f32, MlxDtype::Float32),
        None,
    );
    let rstd = power(
        &add(
            &mean_sqr,
            &mlx_sys::ops::cached_scalar(eps, MlxDtype::Float32),
            None,
        ),
        &mlx_sys::ops::cached_scalar(-0.5, MlxDtype::Float32),
        None,
    );
    let normed = multiply(&x32, &rstd, None);
    let scale = add(
        &astype(weight, MlxDtype::Float32, None),
        &mlx_sys::ops::cached_scalar(1.0, MlxDtype::Float32),
        None,
    );
    multiply(&normed, &scale, None)
}

/// Split the packed `q_proj` output `[B, S, n_heads * 2 * head_dim]` into the
/// query `[B, n_heads, S, head_dim]` and the sigmoid gate `[B, S, n_heads *
/// head_dim]`. The pack interleaves per head: viewed `[B, S, n_heads, 2 *
/// head_dim]`, the first `head_dim` slice of each head is the query, the
/// second the gate (`utils::qkv_slices` documents the contiguous layout other
/// families use; qwen4_exp is per-head interleaved).
pub(crate) fn split_packed_q_gate(
    packed: &MlxArray,
    n_heads: usize,
    head_dim: usize,
) -> (MlxArray, MlxArray) {
    let shape = packed.shape();
    assert_eq!(
        shape.len(),
        3,
        "qwen4_exp QSA packed q_proj output must be [batch, seq, heads*2*dim]"
    );
    let (batch, seq) = (shape[0], shape[1]);
    let (heads, dim) = (n_heads as i32, head_dim as i32);
    assert_eq!(
        shape[2],
        heads * 2 * dim,
        "qwen4_exp QSA q_proj must pack query+sigmoid-gate per head"
    );
    let view = reshape(packed, &[batch, seq, heads, 2 * dim], None);
    let q = slice(
        &view,
        &[0, 0, 0, 0],
        &[batch, seq, heads, dim],
        &[1, 1, 1, 1],
        None,
    );
    let gate = slice(
        &view,
        &[0, 0, 0, dim],
        &[batch, seq, heads, 2 * dim],
        &[1, 1, 1, 1],
        None,
    );
    let q = transpose(&q, &[0, 2, 1, 3], None);
    let gate = reshape(&gate, &[batch, seq, heads * dim], None);
    (q, gate)
}

/// Indexer block scores (fp32): `Σ_h ReLU(q_h · k_block) / sqrt(idx_dim)`.
/// ReLU per indexer head, summed to ONE shared score per block for all query
/// heads. `index_q`: `[1, idx_heads, S, idx_dim]` f32 (normed + RoPE'd);
/// `block_k`: `[1, 1, n_blocks, idx_dim]` f32 (committed keys stay
/// f32-resident). Returns `[1, S, n_blocks]` f32.
pub(crate) fn qsa_indexer_scores(index_q: &MlxArray, block_k: &MlxArray) -> MlxArray {
    let q_shape = index_q.shape();
    assert_eq!(
        q_shape.len(),
        4,
        "qwen4_exp QSA indexer queries must be [1, idx_heads, seq, idx_dim]"
    );
    let idx_dim = q_shape[3];
    let k_shape = block_k.shape();
    assert_eq!(
        k_shape,
        vec![1, 1, k_shape[2], idx_dim],
        "qwen4_exp QSA block keys must be [1, 1, n_blocks, idx_dim]"
    );
    let keys = astype(block_k, MlxDtype::Float32, None);
    let keys_t = transpose(&keys, &[0, 1, 3, 2], None);
    let dots = matmul(index_q, &keys_t, None);
    let relu = maximum(
        &dots,
        &mlx_sys::ops::cached_scalar(0.0, MlxDtype::Float32),
        None,
    );
    let summed = sum_axis(&relu, 1, false, None);
    divide(
        &summed,
        &mlx_sys::ops::cached_scalar((idx_dim as f32).sqrt(), MlxDtype::Float32),
        None,
    )
}

/// Build the sparse token-visibility mask `[1, seq, total]` (Bool, true =
/// attend) from indexer block scores `[1, seq, n_blocks]` (f32).
///
/// Per query at absolute position `p` (row `i`, `p = offset + i`):
/// - Eligible blocks are complete AND causally past: `b*ratio + ratio-1 <= p`.
///   Causally-future blocks score −inf before top-k and are never selected.
/// - The top `min(block_topk, eligible)` blocks by score are selected; rows
///   with fewer eligible blocks keep exactly those (the −inf padding the
///   partition gathers is dropped by the finite check).
/// - The row's own block, when incomplete w.r.t. the row (`p % ratio !=
///   ratio-1`), is ALWAYS visible for tokens `block_start(p)..=p` — the
///   ≤ ratio-1 token tail. When the row closes its block (`p % ratio ==
///   ratio-1`) that block is eligible and arrives through selection only, so
///   the selection buffer stays within `budget + ratio - 1` tokens.
///
/// Sparse path only: callers take the dense causal fallback when
/// `n_blocks <= block_topk` (the selection is then the identity).
pub(crate) fn qsa_selection_mask(
    scores: &MlxArray,
    offset: usize,
    ratio: usize,
    block_topk: usize,
    total: usize,
) -> MlxArray {
    let shape = scores.shape();
    assert_eq!(
        shape.len(),
        3,
        "qwen4_exp QSA indexer scores must be [1, seq, n_blocks]"
    );
    let seq = shape[1] as usize;
    let n_blocks = shape[2] as usize;
    assert!(
        n_blocks > block_topk,
        "qwen4_exp QSA selection mask requires n_blocks {n_blocks} > block_topk {block_topk}; use the dense fallback"
    );
    assert_eq!(
        offset + seq,
        total,
        "qwen4_exp QSA selection mask rows must end at the total token count"
    );
    assert!(
        ratio > 0 && n_blocks * ratio <= total && total - n_blocks * ratio < ratio,
        "qwen4_exp QSA block keys must cover exactly the complete blocks of {total} tokens"
    );
    let (seq_i, nb_i, total_i) = (seq as i32, n_blocks as i32, total as i32);

    // Eligibility: block b is fully causally visible to row p when its last
    // token b*ratio + ratio-1 <= p. Future blocks score −inf.
    let q_pos = reshape(
        &arange(
            offset as f64,
            (offset + seq) as f64,
            1.0,
            MlxDtype::Int32,
            None,
        ),
        &[seq_i, 1],
        None,
    );
    let block_end = reshape(
        &arange(
            (ratio - 1) as f64,
            (n_blocks * ratio + ratio - 1) as f64,
            ratio as f64,
            MlxDtype::Int32,
            None,
        ),
        &[1, nb_i],
        None,
    );
    let eligible = reshape(
        &greater_equal(&q_pos, &block_end, None),
        &[1, seq_i, nb_i],
        None,
    );
    let neg_inf = mlx_sys::ops::cached_scalar(f32::NEG_INFINITY, MlxDtype::Float32);
    let masked = where_cond(&eligible, scores, &neg_inf, None);

    // Top-k block indices per row; selections that landed on −inf padding
    // (rows with fewer eligible blocks than block_topk) are not visible.
    let (top_idx, top_vals) = top_k_by_argpartition(&masked, n_blocks, block_topk, false);
    let valid = less(&neg_inf, &top_vals, None);
    let block_sel = put_along_axis(
        &zeros(&[1, seq_i, nb_i], MlxDtype::Bool, None),
        &top_idx,
        &valid,
        -1,
        None,
    );

    // Expand block selection to token positions and pad the incomplete tail.
    let token_sel = repeat_axis(&block_sel, ratio as i32, -1, None);
    let tail = total - n_blocks * ratio;
    let token_sel = if tail > 0 {
        concatenate(
            &[
                &token_sel,
                &zeros(&[1, seq_i, tail as i32], MlxDtype::Bool, None),
            ],
            -1,
            None,
        )
    } else {
        token_sel
    };

    // Always-visible own incomplete block: tokens [block_start(p), p] when
    // the row does not close its block. Block ids are exact in f32 at any
    // realistic context length.
    let cols = reshape(
        &arange(0.0, total as f64, 1.0, MlxDtype::Int32, None),
        &[1, total_i],
        None,
    );
    let causal = greater_equal(&q_pos, &cols, None);
    let ratio_f = mlx_sys::ops::cached_scalar(ratio as f32, MlxDtype::Float32);
    let col_block = floor(
        &divide(&astype(&cols, MlxDtype::Float32, None), &ratio_f, None),
        None,
    );
    let row_block = floor(
        &divide(&astype(&q_pos, MlxDtype::Float32, None), &ratio_f, None),
        None,
    );
    let same_block = equal(&col_block, &row_block, None);
    let row_mod = subtract(
        &astype(&q_pos, MlxDtype::Float32, None),
        &multiply(&row_block, &ratio_f, None),
        None,
    );
    let incomplete = less(
        &row_mod,
        &mlx_sys::ops::cached_scalar((ratio - 1) as f32, MlxDtype::Float32),
        None,
    );
    let own_tail = logical_and(&logical_and(&same_block, &causal, None), &incomplete, None);

    // Bool OR via maximum (mlx-sys exposes logical_and but no logical_or).
    maximum(&token_sel, &own_tail, None)
}

/// Mean-pool, normalize, RoPE, and commit the block keys that completed with
/// this forward, reusing already-committed rows across prefill chunks.
///
/// `committed`/`complete` are block counts before/after this forward's
/// append. New block `b` pools the RAW (pre-norm, pre-RoPE) index keys of
/// tokens `[b*ratio, (b+1)*ratio)`; the first new block can straddle the
/// chunk boundary (≤ ratio-1 history tokens), so its keys come from the
/// cached raw view and its block-start position row is read back from the
/// cached f32 position ids. Rows are stored in f32: pooling, k_layernorm, and
/// RoPE are already fp32, and keeping the committed keys f32-resident avoids
/// bf16 quantization noise between the commit and the fp32 scoring read-back.
#[allow(clippy::too_many_arguments)]
fn commit_completed_blocks(
    cache: &mut MlxKVCache,
    layer_idx: usize,
    new_index_k: &MlxArray,
    indexer_weights: &Qwen4ExpIndexerWeights,
    offset: usize,
    positions: &[[i32; 3]],
    committed: usize,
    complete: usize,
    geo: &QsaGeometry,
    rope_theta: f32,
    sections: &[u32],
) {
    let n_new = complete - committed;
    let (n_new_i, ratio_i, dim_i) = (n_new as i32, geo.ratio as i32, geo.idx_dim as i32);

    // Region of raw keys covering the new blocks: ≤ ratio-1 history tokens
    // from the cached view plus this forward's appended rows.
    let hist_len = offset - committed * geo.ratio;
    let region = if hist_len > 0 {
        let Some(raw_view) = cache.qwen4_exp_qsa_index_keys(layer_idx) else {
            unreachable!("qwen4_exp QSA raw index keys must exist after append");
        };
        let hist = slice(
            &raw_view,
            &[0, 0, (committed * geo.ratio) as i32, 0],
            &[1, 1, offset as i32, dim_i],
            &[1, 1, 1, 1],
            None,
        );
        concatenate(&[&hist, new_index_k], 2, None)
    } else {
        new_index_k.clone()
    };
    let region = slice(
        &region,
        &[0, 0, 0, 0],
        &[1, 1, n_new_i * ratio_i, dim_i],
        &[1, 1, 1, 1],
        None,
    );

    // fp32 mean-pool over each group of `ratio` consecutive tokens.
    let pooled = reshape(&region, &[1, 1, n_new_i, ratio_i, dim_i], None);
    let pooled = astype(&pooled, MlxDtype::Float32, None);
    let pooled = divide(
        &sum_axis(&pooled, 3, true, None),
        &mlx_sys::ops::cached_scalar(geo.ratio as f32, MlxDtype::Float32),
        None,
    );
    let pooled = reshape(&pooled, &[1, 1, n_new_i, dim_i], None);
    let normed = rms_norm_one_plus_gamma(&pooled, &indexer_weights.k_layernorm, geo.eps);

    // Block-start positions: from this forward's ids when the block starts in
    // the chunk, else read back from the cached f32 ids (only the first new
    // block can straddle the boundary).
    let mut block_positions: Vec<[i32; 3]> = Vec::with_capacity(n_new);
    let mut readback_row: Option<[i32; 3]> = None;
    if committed * geo.ratio < offset {
        let Some(pos_view) = cache.qwen4_exp_qsa_positions(layer_idx) else {
            unreachable!("qwen4_exp QSA positions must exist after append");
        };
        let row = slice(
            &pos_view,
            &[0, (committed * geo.ratio) as i32, 0],
            &[1, (committed * geo.ratio + 1) as i32, 3],
            &[1, 1, 1],
            None,
        );
        eval(&[&row]);
        let data = row.data_f32();
        assert_eq!(
            data.len(),
            3,
            "qwen4_exp QSA cached positions must carry the 3 mRoPE grids"
        );
        readback_row = Some([data[0] as i32, data[1] as i32, data[2] as i32]);
    }
    for block in committed..complete {
        let token = block * geo.ratio;
        if token >= offset {
            block_positions.push(positions[token - offset]);
        } else {
            let Some(row) = readback_row else {
                unreachable!("qwen4_exp QSA straddling block must be the first new block");
            };
            block_positions.push(row);
        }
    }
    let mrope = build_qsa_mrope(&block_positions, geo.rope_dims, rope_theta, sections);
    let roped = apply_qsa_mrope(&normed, &mrope, geo.rope_dims);

    // Committed block keys stay f32-resident (the scoring path upcasts to
    // f32 anyway; storing bf16 would quantize the keys between commits).
    let rows = astype(&roped, MlxDtype::Float32, None);
    cache.append_qwen4_exp_qsa_block_rows(layer_idx, rows);
}

/// One qwen4_exp QSA layer forward: sparse-selected full attention with the
/// sigmoid-gated output projection. Covers chunked prefill (`seq > 1`,
/// causal) and decode (`seq == 1`); both use masked SDPA — the custom sparse
/// prefill kernel is a later perf phase with identical math.
///
/// - `hidden`: `[1, seq, hidden]` attention-branch input (the hyper-connection
///   `mixed_input` output), any float dtype; the cache adopts it.
/// - `layer_weights`: generic standard fields — `q_proj` (packs
///   query+sigmoid-gate per head), `k_proj`/`v_proj`, `o_proj`, and the
///   per-head `q_norm`/`k_norm` (1+γ, applied BEFORE RoPE).
/// - `indexer_weights`: `self_attn.indexer.*` of this QSA layer.
/// - `cache`: QSA cache for `layer_idx`; K/V, raw index keys, and f32 mRoPE
///   position ids are appended here (positions stay f32 so the block-start
///   read-back needs no unsafe int accessors; grids are exact integers in
///   f32 at any realistic context length). The caller advances `seq_len`
///   after all layers, as for every other cache kind.
/// - `positions`: per-token mRoPE ids `[seq]`; the text path passes 3
///   identical grids (T=H=W=position).
///
/// Returns `[1, seq, hidden]` — `o_proj(sigmoid(gate) * attn)`.
pub(crate) fn qwen4_exp_qsa_forward(
    hidden: &MlxArray,
    layer_weights: &LayerWeights,
    indexer_weights: &Qwen4ExpIndexerWeights,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    positions: &[[i32; 3]],
    cfg: &ModelConfig,
) -> MlxArray {
    let shape = hidden.shape();
    assert_eq!(
        shape.len(),
        3,
        "qwen4_exp QSA hidden must be [batch, seq, hidden]"
    );
    let (batch, seq, width) = (shape[0], shape[1] as usize, shape[2]);
    assert_eq!(batch, 1, "qwen4_exp QSA cache supports batch=1 only");
    assert_eq!(
        width as usize, cfg.hidden_size,
        "qwen4_exp QSA hidden width must match the model hidden size"
    );
    assert_eq!(
        positions.len(),
        seq,
        "qwen4_exp QSA positions must cover every new token"
    );
    let geo = QsaGeometry::from_config(cfg);
    let Some(q4e) = cfg.qwen4_exp.as_ref() else {
        unreachable!("qwen4_exp QSA forward requires ModelConfig.qwen4_exp");
    };
    let offset = cache.seq_len();
    let total = offset + seq;
    let (kv_heads, dim) = (geo.n_kv_heads as i32, geo.head_dim as i32);
    let (idx_heads, idx_kv_heads, idx_dim) = (
        geo.idx_heads as i32,
        geo.idx_kv_heads as i32,
        geo.idx_dim as i32,
    );

    let Some(q_proj) = layer_weights.q_proj.as_ref() else {
        unreachable!("qwen4_exp QSA layer requires q_proj");
    };
    let Some(k_proj) = layer_weights.k_proj.as_ref() else {
        unreachable!("qwen4_exp QSA layer requires k_proj");
    };
    let Some(v_proj) = layer_weights.v_proj.as_ref() else {
        unreachable!("qwen4_exp QSA layer requires v_proj");
    };
    let Some(o_proj) = layer_weights.o_proj.as_ref() else {
        unreachable!("qwen4_exp QSA layer requires o_proj");
    };
    let Some(q_norm) = layer_weights.q_norm.as_ref() else {
        unreachable!("qwen4_exp QSA layer requires q_norm");
    };
    let Some(k_norm) = layer_weights.k_norm.as_ref() else {
        unreachable!("qwen4_exp QSA layer requires k_norm");
    };

    let mrope = build_qsa_mrope(positions, geo.rope_dims, cfg.rope_theta, &q4e.mrope_section);

    // Attention projections; q/k norms (1+γ) BEFORE RoPE; partial RoPE.
    let packed = qw(hidden, q_proj);
    let (q, gate) = split_packed_q_gate(&packed, geo.n_heads, geo.head_dim);
    let k = qw(hidden, k_proj);
    let v = qw(hidden, v_proj);
    let k = reshape(&k, &[1, seq as i32, kv_heads, dim], None);
    let v = reshape(&v, &[1, seq as i32, kv_heads, dim], None);
    let k = transpose(&k, &[0, 2, 1, 3], None);
    let v = transpose(&v, &[0, 2, 1, 3], None);
    let q = astype(
        &rms_norm_one_plus_gamma(&q, q_norm, geo.eps),
        hidden.dtype(),
        None,
    );
    let k = astype(
        &rms_norm_one_plus_gamma(&k, k_norm, geo.eps),
        hidden.dtype(),
        None,
    );
    let q = apply_qsa_mrope(&q, &mrope, geo.rope_dims);
    let k = apply_qsa_mrope(&k, &mrope, geo.rope_dims);

    // Indexer: fused projection → per-head queries (normed + RoPE'd, f32) and
    // one RAW key head (pre-norm, pre-RoPE) cached per token.
    let idx = qw(hidden, &indexer_weights.index_qk_proj);
    let idx_shape = idx.shape();
    assert_eq!(
        idx_shape[2],
        (idx_heads + idx_kv_heads) * idx_dim,
        "qwen4_exp QSA index_qk_proj must produce (idx_heads + idx_kv_heads) * idx_dim"
    );
    let idx_view = reshape(
        &idx,
        &[1, seq as i32, idx_heads + idx_kv_heads, idx_dim],
        None,
    );
    let index_q = slice(
        &idx_view,
        &[0, 0, 0, 0],
        &[1, seq as i32, idx_heads, idx_dim],
        &[1, 1, 1, 1],
        None,
    );
    let index_q = transpose(&index_q, &[0, 2, 1, 3], None);
    let index_k = slice(
        &idx_view,
        &[0, 0, idx_heads, 0],
        &[1, seq as i32, idx_heads + idx_kv_heads, idx_dim],
        &[1, 1, 1, 1],
        None,
    );
    let index_k = transpose(&index_k, &[0, 2, 1, 3], None);
    let index_q = rms_norm_one_plus_gamma(&index_q, &indexer_weights.q_layernorm, geo.eps);
    let index_q = apply_qsa_mrope(&index_q, &mrope, geo.rope_dims);

    // Append K/V + raw index keys + f32 mRoPE ids; logical K/V span `total`.
    let pos_flat: Vec<f32> = positions
        .iter()
        .flat_map(|position| position.map(|grid| grid as f32))
        .collect();
    let pos_array = f32_array(&pos_flat, &[1, seq as i32, 3]);
    let (full_k, full_v) =
        cache.append_qwen4_exp_qsa(layer_idx, k, v, index_k.clone(), pos_array, geo.ratio);

    // Commit blocks that completed with this append; reuse cached rows.
    let committed = cache.qwen4_exp_qsa_committed_blocks(layer_idx);
    let complete = total / geo.ratio;
    if complete > committed {
        commit_completed_blocks(
            cache,
            layer_idx,
            &index_k,
            indexer_weights,
            offset,
            positions,
            committed,
            complete,
            &geo,
            cfg.rope_theta,
            &q4e.mrope_section,
        );
    }

    // Dense fallback: the selection is the identity while every row's
    // eligible blocks fit the budget (complete blocks <= budget / ratio).
    let attn = if complete <= geo.block_topk {
        let mask = if seq > 1 {
            ScaledDotProductAttentionMask::Causal
        } else {
            ScaledDotProductAttentionMask::None
        };
        scaled_dot_product_attention_with_mask(&q, &full_k, &full_v, cfg.query_scale, mask, None)
    } else {
        let Some(block_k) = cache.qwen4_exp_qsa_block_keys(layer_idx) else {
            unreachable!("qwen4_exp QSA sparse path requires committed block keys");
        };
        let scores = qsa_indexer_scores(&index_q, &block_k);
        let mask = qsa_selection_mask(&scores, offset, geo.ratio, geo.block_topk, total);
        scaled_dot_product_attention_with_mask(
            &q,
            &full_k,
            &full_v,
            cfg.query_scale,
            ScaledDotProductAttentionMask::Array(&mask),
            None,
        )
    };

    let flat = flatten_attention_output_bhsd(&attn, seq, geo.n_heads, geo.head_dim);
    attention_output_projection(&flat, Some(&gate), o_proj)
}

fn f32_array(values: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        shape,
        MlxDtype::Float32,
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::model::config::ModelConfig;
    use crate::weights::{LayerWeights, QuantizedWeight};
    use mlx_sys::eval;

    // Toy geometry: GQA group 2, partial rotary 0.5, indexer 2+1 heads,
    // ratio 2, budget 4 → block_topk 2 (sparse once 3+ blocks complete).
    const HID: usize = 8;
    const NH: usize = 2;
    const KV: usize = 1;
    const D: usize = 8;
    const ROT: usize = 4;
    const THETA: f32 = 10_000.0;
    const SECTIONS: [u32; 3] = [1, 1, 0];
    const IH: usize = 2;
    const IKV: usize = 1;
    const ID: usize = 8;
    const RATIO: usize = 2;
    const TOPK: usize = 2;
    const EPS: f32 = 1e-6;

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    fn eval_f32(array: &MlxArray) -> Vec<f32> {
        eval(&[array]);
        array.data_f32().to_vec()
    }

    /// Deterministic pseudo-random fill (no external deps).
    fn fill(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5)
            .collect()
    }

    fn scaled_fill(len: usize, seed: f32, scale: f32) -> Vec<f32> {
        fill(len, seed).iter().map(|v| v * scale).collect()
    }

    fn sigmoid_f32(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what} length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "{what} mismatch at {i}: {a} vs {e} (tol {tol})"
            );
        }
    }

    fn text_positions(offset: usize, seq: usize) -> Vec<[i32; 3]> {
        (offset..offset + seq).map(|p| [p as i32; 3]).collect()
    }

    fn test_config(indexer_budget: usize) -> ModelConfig {
        let value = serde_json::json!({
            "schema_version": ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
            "model_family": "qwen4_exp",
            "tensor_format": "safetensors",
            "layer_count": 1,
            "hidden_size": HID,
            "attention_head_count": NH,
            "attention_head_dim": D,
            "kv_head_count": KV,
            "vocab_size": 32,
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
            "rms_norm_eps": EPS as f64,
            "qwen4_exp": {
                "indexer_budget": indexer_budget,
                "indexer_compress_ratio": RATIO,
                "indexer_head_dim": ID,
                "indexer_kv_heads": IKV,
                "indexer_n_heads": IH,
                "mrope_section": SECTIONS,
                "mrope_interleaved": true,
                "partial_rotary_factor": 0.5
            },
            "tensors": []
        });
        let manifest: ax_engine_core::NativeModelManifest =
            serde_json::from_value(value).expect("qwen4_exp test manifest");
        ModelConfig::from_manifest(&manifest)
    }

    struct TestWeights {
        layer: LayerWeights,
        indexer: Qwen4ExpIndexerWeights,
        q_proj: Vec<f32>,
        k_proj: Vec<f32>,
        v_proj: Vec<f32>,
        o_proj: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        idx_qk: Vec<f32>,
        idx_q_ln: Vec<f32>,
        idx_k_ln: Vec<f32>,
    }

    fn dense_weight(data: &[f32], out_dim: usize, in_dim: usize) -> QuantizedWeight {
        QuantizedWeight::new(
            array_f32(data, &[out_dim as i32, in_dim as i32]),
            None,
            None,
        )
    }

    // idx seed 0.37 keeps every sparse test row free of exact top-k score
    // ties (ReLU-clamped zeros), whose argpartition order is unspecified.
    fn test_weights() -> TestWeights {
        test_weights_with_idx_seed(0.37)
    }

    fn test_weights_with_idx_seed(idx_seed: f32) -> TestWeights {
        let q_proj = scaled_fill(NH * 2 * D * HID, 0.11, 0.4);
        let k_proj = scaled_fill(KV * D * HID, 0.13, 0.4);
        let v_proj = scaled_fill(KV * D * HID, 0.17, 0.4);
        let o_proj = scaled_fill(HID * NH * D, 0.19, 0.3);
        let q_norm = scaled_fill(D, 0.23, 0.2);
        let k_norm = scaled_fill(D, 0.29, 0.2);
        let idx_qk = scaled_fill((IH + IKV) * ID * HID, idx_seed, 0.4);
        let idx_q_ln = scaled_fill(ID, 0.37, 0.2);
        let idx_k_ln = scaled_fill(ID, 0.41, 0.2);
        let dummy = array_f32(&[0.0], &[1]);
        let layer = LayerWeights {
            attn_norm: dummy.clone(),
            attn_post_norm: None,
            q_norm: Some(array_f32(&q_norm, &[D as i32])),
            k_norm: Some(array_f32(&k_norm, &[D as i32])),
            q_proj: Some(dense_weight(&q_proj, NH * 2 * D, HID)),
            k_proj: Some(dense_weight(&k_proj, KV * D, HID)),
            v_proj: Some(dense_weight(&v_proj, KV * D, HID)),
            qkv_packed: None,
            attn_out_gate: None,
            o_proj: Some(dense_weight(&o_proj, HID, NH * D)),
            linear_attn: None,
            glm_mla_attn: None,
            deepseek_v4: None,
            qwen4_exp: None,
            ffn_norm: dummy,
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: None,
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
            expert_stream: None,
        };
        let indexer = Qwen4ExpIndexerWeights {
            index_qk_proj: dense_weight(&idx_qk, (IH + IKV) * ID, HID),
            q_layernorm: array_f32(&idx_q_ln, &[ID as i32]),
            k_layernorm: array_f32(&idx_k_ln, &[ID as i32]),
        };
        TestWeights {
            layer,
            indexer,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            idx_qk,
            idx_q_ln,
            idx_k_ln,
        }
    }

    // ── CPU reference (independent f32 implementation of planning §A/§E) ──

    /// Row-major `x [rows, in_dim] @ w [out_dim, in_dim]^T`.
    fn ref_matmul(x: &[f32], rows: usize, in_dim: usize, w: &[f32], out_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * out_dim];
        for r in 0..rows {
            for o in 0..out_dim {
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    acc += x[r * in_dim + i] * w[o * in_dim + i];
                }
                out[r * out_dim + o] = acc;
            }
        }
        out
    }

    /// Row-major `[rows, dim]` RMSNorm with the 1+γ gain, f32.
    fn ref_rmsnorm(x: &[f32], dim: usize, w: &[f32]) -> Vec<f32> {
        let rows = x.len() / dim;
        let mut out = vec![0.0; x.len()];
        for r in 0..rows {
            let row = &x[r * dim..(r + 1) * dim];
            let mean_sqr: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let rstd = (mean_sqr + EPS).powf(-0.5);
            for (e, v) in row.iter().enumerate() {
                out[r * dim + e] = v * rstd * (1.0 + w[e]);
            }
        }
        out
    }

    /// Interleaved mRoPE cos/sin factors `[seq, rot]` (cat([f, f]) layout).
    fn ref_mrope(positions: &[[i32; 3]]) -> (Vec<f32>, Vec<f32>) {
        let half = ROT / 2;
        let mut source_axis = vec![0usize; half];
        for (axis, section) in SECTIONS.iter().copied().enumerate().skip(1) {
            let mut index = axis;
            while index < section as usize * 3 && index < half {
                source_axis[index] = axis;
                index += 3;
            }
        }
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / THETA.powf((2 * i) as f32 / ROT as f32))
            .collect();
        let mut cos = vec![0.0; positions.len() * ROT];
        let mut sin = vec![0.0; positions.len() * ROT];
        for (s, position) in positions.iter().enumerate() {
            for i in 0..half {
                let f = position[source_axis[i]] as f32 * inv_freq[i];
                cos[s * ROT + i] = f.cos();
                cos[s * ROT + half + i] = f.cos();
                sin[s * ROT + i] = f.sin();
                sin[s * ROT + half + i] = f.sin();
            }
        }
        (cos, sin)
    }

    /// Apply partial RoPE in place: `x` is `[heads][seq][dim]`, factors
    /// `[seq][rot]`; only the first `rot` dims rotate (NeoX pairing).
    fn ref_rope(x: &mut [f32], heads: usize, seq: usize, dim: usize, cos: &[f32], sin: &[f32]) {
        let half = ROT / 2;
        for h in 0..heads {
            for s in 0..seq {
                let base = (h * seq + s) * dim;
                for i in 0..half {
                    let x1 = x[base + i];
                    let x2 = x[base + half + i];
                    x[base + i] = x1 * cos[s * ROT + i] - x2 * sin[s * ROT + i];
                    x[base + half + i] =
                        x2 * cos[s * ROT + half + i] + x1 * sin[s * ROT + half + i];
                }
            }
        }
    }

    #[derive(Default)]
    struct RefCache {
        /// Post-norm/rope K rows `[total][KV*D]`.
        k: Vec<f32>,
        /// V rows `[total][KV*D]`.
        v: Vec<f32>,
        /// RAW (pre-norm, pre-RoPE) index keys `[total][ID]`.
        idx_raw: Vec<f32>,
        /// Per-token mRoPE ids.
        pos: Vec<[i32; 3]>,
        /// Committed pooled block keys `[n_blocks][ID]`.
        block_k: Vec<f32>,
    }

    fn ref_topk_selection(scores_row: &[f32], p: usize, nb: usize, topk: usize) -> Vec<bool> {
        let mut eligible: Vec<usize> = (0..nb).filter(|b| b * RATIO + RATIO - 1 <= p).collect();
        eligible.sort_by(|a, b| {
            scores_row[*b]
                .partial_cmp(&scores_row[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut selected = vec![false; nb];
        for b in eligible.into_iter().take(topk) {
            selected[b] = true;
        }
        selected
    }

    /// Full QSA layer forward on CPU: returns `[seq][HID]`.
    fn ref_qsa_forward(
        w: &TestWeights,
        cache: &mut RefCache,
        hidden: &[f32],
        seq: usize,
        positions: &[[i32; 3]],
        topk: usize,
    ) -> Vec<f32> {
        let offset = cache.pos.len();
        let total = offset + seq;
        let query_scale = 1.0 / (D as f32).sqrt();

        let packed = ref_matmul(hidden, seq, HID, &w.q_proj, NH * 2 * D);
        let mut q = vec![0.0; NH * seq * D];
        let mut gate = vec![0.0; NH * seq * D];
        for s in 0..seq {
            for h in 0..NH {
                for e in 0..D {
                    q[(h * seq + s) * D + e] = packed[s * NH * 2 * D + h * 2 * D + e];
                    gate[(h * seq + s) * D + e] = packed[s * NH * 2 * D + h * 2 * D + D + e];
                }
            }
        }
        let k_new = ref_matmul(hidden, seq, HID, &w.k_proj, KV * D);
        let v_new = ref_matmul(hidden, seq, HID, &w.v_proj, KV * D);
        let mut kh = vec![0.0; KV * seq * D];
        for s in 0..seq {
            for h in 0..KV {
                for e in 0..D {
                    kh[(h * seq + s) * D + e] = k_new[s * KV * D + h * D + e];
                }
            }
        }
        // q/k norms (1+γ) BEFORE RoPE.
        let mut q = ref_rmsnorm(&q, D, &w.q_norm);
        let mut kh = ref_rmsnorm(&kh, D, &w.k_norm);
        let (cos, sin) = ref_mrope(positions);
        ref_rope(&mut q, NH, seq, D, &cos, &sin);
        ref_rope(&mut kh, KV, seq, D, &cos, &sin);
        // Cache K/V rows back in token-major layout.
        for s in 0..seq {
            for h in 0..KV {
                cache
                    .k
                    .extend_from_slice(&kh[(h * seq + s) * D..(h * seq + s) * D + D]);
                cache
                    .v
                    .extend_from_slice(&v_new[s * KV * D + h * D..s * KV * D + h * D + D]);
            }
        }

        let idx = ref_matmul(hidden, seq, HID, &w.idx_qk, (IH + IKV) * ID);
        let mut iq = vec![0.0; IH * seq * ID];
        for s in 0..seq {
            for h in 0..IH {
                for e in 0..ID {
                    iq[(h * seq + s) * ID + e] = idx[s * (IH + IKV) * ID + h * ID + e];
                }
            }
            cache
                .idx_raw
                .extend_from_slice(&idx[s * (IH + IKV) * ID + IH * ID..(s + 1) * (IH + IKV) * ID]);
        }
        cache.pos.extend_from_slice(positions);
        let mut iq = ref_rmsnorm(&iq, ID, &w.idx_q_ln);
        ref_rope(&mut iq, IH, seq, ID, &cos, &sin);

        // Commit newly completed blocks from the RAW cached keys.
        let committed = cache.block_k.len() / ID;
        let complete = total / RATIO;
        for b in committed..complete {
            let mut pooled = vec![0.0f32; ID];
            for t in b * RATIO..(b + 1) * RATIO {
                for (e, v) in pooled.iter_mut().enumerate() {
                    *v += cache.idx_raw[t * ID + e] / RATIO as f32;
                }
            }
            let mut row = ref_rmsnorm(&pooled, ID, &w.idx_k_ln);
            let (bc, bs) = ref_mrope(&[cache.pos[b * RATIO]]);
            ref_rope(&mut row, 1, 1, ID, &bc, &bs);
            cache.block_k.extend_from_slice(&row);
        }
        let nb = complete;

        // Indexer scores for this chunk's rows.
        let mut scores = vec![0.0f32; seq * nb];
        for i in 0..seq {
            for b in 0..nb {
                let mut acc = 0.0f32;
                for h in 0..IH {
                    let mut dot = 0.0f32;
                    for e in 0..ID {
                        dot += iq[(h * seq + i) * ID + e] * cache.block_k[b * ID + e];
                    }
                    acc += dot.max(0.0);
                }
                scores[i * nb + b] = acc / (ID as f32).sqrt();
            }
        }

        let kv_group = NH / KV;
        let mut attn = vec![0.0f32; seq * NH * D];
        for i in 0..seq {
            let p = offset + i;
            let selected = if nb > topk {
                Some(ref_topk_selection(
                    &scores[i * nb..(i + 1) * nb],
                    p,
                    nb,
                    topk,
                ))
            } else {
                None
            };
            let own_start = p - (p % RATIO);
            let allowed: Vec<bool> = (0..total)
                .map(|j| match selected.as_ref() {
                    None => j <= p,
                    Some(sel) => {
                        let in_selected_block = j < nb * RATIO && sel[j / RATIO];
                        let in_own_tail = p % RATIO != RATIO - 1 && j >= own_start && j <= p;
                        in_selected_block || in_own_tail
                    }
                })
                .collect();
            for h in 0..NH {
                let kvh = h / kv_group;
                let mut logits = vec![f32::NEG_INFINITY; total];
                for (j, logit) in logits.iter_mut().enumerate() {
                    if allowed[j] {
                        let mut dot = 0.0f32;
                        for e in 0..D {
                            dot += q[(h * seq + i) * D + e] * cache.k[j * KV * D + kvh * D + e];
                        }
                        *logit = dot * query_scale;
                    }
                }
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0f32;
                for logit in logits.iter_mut() {
                    *logit = (*logit - max).exp();
                    denom += *logit;
                }
                for e in 0..D {
                    let mut acc = 0.0f32;
                    for (j, weight) in logits.iter().enumerate() {
                        acc += weight / denom * cache.v[j * KV * D + kvh * D + e];
                    }
                    attn[i * NH * D + h * D + e] = acc;
                }
            }
        }

        let gated: Vec<f32> = (0..seq * NH * D)
            .map(|idx| {
                let s = idx / (NH * D);
                let h = (idx / D) % NH;
                let e = idx % D;
                sigmoid_f32(gate[(h * seq + s) * D + e]) * attn[idx]
            })
            .collect();
        ref_matmul(&gated, seq, NH * D, &w.o_proj, HID)
    }

    fn hidden_input(seq: usize, seed: f32) -> MlxArray {
        array_f32(
            &scaled_fill(seq * HID, seed, 1.0),
            &[1, seq as i32, HID as i32],
        )
    }

    fn hidden_data(seq: usize, seed: f32) -> Vec<f32> {
        scaled_fill(seq * HID, seed, 1.0)
    }

    fn mlx_forward(
        w: &TestWeights,
        cache: &mut MlxKVCache,
        cfg: &ModelConfig,
        hidden: &MlxArray,
        positions: &[[i32; 3]],
    ) -> Vec<f32> {
        let out = qwen4_exp_qsa_forward(hidden, &w.layer, &w.indexer, cache, 0, positions, cfg);
        let data = eval_f32(&out);
        cache.advance(positions.len());
        data
    }

    // ── Unit tests ──

    #[test]
    fn packed_q_splits_query_and_gate_per_head() {
        let seq = 2;
        let data: Vec<f32> = (0..seq * NH * 2 * D).map(|i| i as f32 * 0.25).collect();
        let packed = array_f32(&data, &[1, seq as i32, (NH * 2 * D) as i32]);
        let (q, gate) = split_packed_q_gate(&packed, NH, D);
        assert_eq!(q.shape(), vec![1, NH as i32, seq as i32, D as i32]);
        assert_eq!(gate.shape(), vec![1, seq as i32, (NH * D) as i32]);
        let q_data = eval_f32(&mlx_sys::contiguous(&q, None));
        let gate_data = eval_f32(&mlx_sys::contiguous(&gate, None));
        for s in 0..seq {
            for h in 0..NH {
                for e in 0..D {
                    let packed_base = s * NH * 2 * D + h * 2 * D;
                    assert_eq!(q_data[(h * seq + s) * D + e], data[packed_base + e]);
                    assert_eq!(gate_data[s * NH * D + h * D + e], data[packed_base + D + e]);
                }
            }
        }
    }

    #[test]
    fn rms_norm_one_plus_gamma_matches_manual() {
        let rows = 3;
        let x = array_f32(&scaled_fill(rows * D, 0.43, 1.0), &[rows as i32, D as i32]);
        let w = array_f32(&scaled_fill(D, 0.47, 0.5), &[D as i32]);
        let out = rms_norm_one_plus_gamma(&x, &w, EPS);
        assert_eq!(out.dtype(), MlxDtype::Float32);
        let expected = ref_rmsnorm(
            &scaled_fill(rows * D, 0.43, 1.0),
            D,
            &scaled_fill(D, 0.47, 0.5),
        );
        assert_close(&eval_f32(&out), &expected, 1e-5, "rms_norm 1+gamma");
    }

    #[test]
    fn qk_norm_applies_before_rope() {
        let seq = 3;
        let x_data = scaled_fill(NH * seq * D, 0.53, 1.0);
        let w_data = scaled_fill(D, 0.59, 0.4);
        let x = array_f32(&x_data, &[1, NH as i32, seq as i32, D as i32]);
        let w = array_f32(&w_data, &[D as i32]);
        let positions = text_positions(0, seq);
        let mrope = build_qsa_mrope(&positions, ROT, THETA, &SECTIONS);

        let normed_then_roped = apply_qsa_mrope(&rms_norm_one_plus_gamma(&x, &w, EPS), &mrope, ROT);
        let roped_unnormed = apply_qsa_mrope(&x, &mrope, ROT);
        let got = eval_f32(&normed_then_roped);
        let other = eval_f32(&roped_unnormed);
        assert!(
            got.iter()
                .zip(other.iter())
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "norm-before-rope must differ from rope of the unnormed input"
        );

        let mut expected = ref_rmsnorm(&x_data, D, &w_data);
        let (cos, sin) = ref_mrope(&positions);
        ref_rope(&mut expected, NH, seq, D, &cos, &sin);
        assert_close(&got, &expected, 1e-5, "norm-then-rope ordering");
    }

    #[test]
    fn interleaved_mrope_matches_cpu_reference_text_positions() {
        let positions = text_positions(0, 5);
        let factors = build_qsa_mrope(&positions, ROT, THETA, &SECTIONS);
        let (cos, sin) = ref_mrope(&positions);
        assert_close(&eval_f32(&factors.cos), &cos, 1e-6, "mrope cos (T=H=W)");
        assert_close(&eval_f32(&factors.sin), &sin, 1e-6, "mrope sin (T=H=W)");
    }

    #[test]
    fn interleaved_mrope_uses_section_axes_for_distinct_grids() {
        // Distinct T/H/W grids pin the section recomposition: with
        // SECTIONS [1,1,0] over half=2, frequency 0 reads the T grid and
        // frequency 1 the H grid; W is unused.
        let positions: Vec<[i32; 3]> = vec![[3, 40, 500], [7, 80, 900]];
        let factors = build_qsa_mrope(&positions, ROT, THETA, &SECTIONS);
        let cos = eval_f32(&factors.cos);
        let half = ROT / 2;
        for (s, position) in positions.iter().enumerate() {
            for i in 0..half {
                let axis = if i == 0 { 0 } else { 1 };
                let f = position[axis] as f32 * (1.0 / THETA.powf((2 * i) as f32 / ROT as f32));
                assert!((cos[s * ROT + i] - f.cos()).abs() < 1e-6);
                assert!((cos[s * ROT + half + i] - f.cos()).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn indexer_scores_match_cpu_reference() {
        let (seq, nb) = (3, 4);
        let q_data = scaled_fill(IH * seq * ID, 0.61, 1.0);
        let k_data = scaled_fill(nb * ID, 0.67, 1.0);
        let index_q = array_f32(&q_data, &[1, IH as i32, seq as i32, ID as i32]);
        let block_k = array_f32(&k_data, &[1, 1, nb as i32, ID as i32]);
        let scores = qsa_indexer_scores(&index_q, &block_k);
        assert_eq!(scores.shape(), vec![1, seq as i32, nb as i32]);

        let mut expected = vec![0.0f32; seq * nb];
        for i in 0..seq {
            for b in 0..nb {
                let mut acc = 0.0f32;
                for h in 0..IH {
                    let mut dot = 0.0f32;
                    for e in 0..ID {
                        dot += q_data[(h * seq + i) * ID + e] * k_data[b * ID + e];
                    }
                    acc += dot.max(0.0);
                }
                expected[i * nb + b] = acc / (ID as f32).sqrt();
            }
        }
        assert_close(&eval_f32(&scores), &expected, 1e-5, "indexer scores");
    }

    #[test]
    fn selection_mask_applies_causal_topk_and_visible_tail() {
        // ratio 2, topk 2, nb 3 complete blocks over tokens [0,6) plus a
        // 1-token global tail (token 6); rows p = 4,5,6 (offset 4), total 7.
        let (seq, nb, offset, total) = (3usize, 3usize, 4usize, 7usize);
        let score_data: Vec<f32> = vec![
            0.9, 0.1, 0.7, // p=4: eligible {0,1} → both kept + own tail {4}
            0.2, 0.8, 0.5, // p=5: eligible {0,1,2} → top-2 {1,2}
            0.3, 0.9, 0.4, // p=6: eligible {0,1,2} → top-2 {1,2} + tail {6}
        ];
        let scores = array_f32(&score_data, &[1, seq as i32, nb as i32]);
        let mask = qsa_selection_mask(&scores, offset, RATIO, TOPK, total);
        assert_eq!(mask.shape(), vec![1, seq as i32, total as i32]);
        assert_eq!(mask.dtype(), MlxDtype::Bool);
        let got = eval_f32(&astype(&mask, MlxDtype::Float32, None));

        let mut expected = vec![0.0f32; seq * total];
        for i in 0..seq {
            let p = offset + i;
            let selected = ref_topk_selection(&score_data[i * nb..(i + 1) * nb], p, nb, TOPK);
            for j in 0..total {
                let in_selected_block = j < nb * RATIO && selected[j / RATIO];
                let in_own_tail = p % RATIO != RATIO - 1 && j >= p - (p % RATIO) && j <= p;
                expected[i * total + j] = (in_selected_block || in_own_tail) as u8 as f32;
            }
        }
        assert_eq!(got, expected, "selection mask positions");

        // Row p=4 (p % ratio == 0): own block 2 is causally incomplete, so
        // token 4 is visible even though block 2 was not selected, while the
        // causally-future tokens 5 (same block) and 6 (global tail) stay
        // masked.
        assert_eq!(got[4], 1.0, "own incomplete block must be visible");
        assert_eq!(got[5], 0.0, "future token of own block stays masked");
        assert_eq!(got[6], 0.0, "future global-tail token stays masked");
        // Row p=5 closes block 2: no tail; tokens 4..=5 visible only because
        // block 2 scored into the top-2, and unselected block 0 stays masked.
        assert_eq!(got[total + 4], 1.0, "selected own block visible");
        assert_eq!(got[total], 0.0, "unselected block 0 masked for p=5");
        // Row p=6 sits in the global tail block: always visible to itself.
        assert_eq!(
            got[2 * total + 6],
            1.0,
            "global tail is the own incomplete block"
        );
    }

    // ── End-to-end tests vs the CPU reference ──

    #[test]
    fn dense_fallback_matches_cpu_reference() {
        let cfg = test_config(1024); // block_topk 512 → always dense here
        let w = test_weights();
        let mut cache = MlxKVCache::new(1);
        let mut ref_cache = RefCache::default();

        let seq = 5;
        let positions = text_positions(0, seq);
        let got = mlx_forward(&w, &mut cache, &cfg, &hidden_input(seq, 0.71), &positions);
        let expected = ref_qsa_forward(
            &w,
            &mut ref_cache,
            &hidden_data(seq, 0.71),
            seq,
            &positions,
            512,
        );
        assert_close(&got, &expected, 2e-4, "dense prefill");

        // Committed blocks and pooled keys track the reference even on the
        // dense path (the indexer state is maintained regardless).
        assert_eq!(cache.qwen4_exp_qsa_committed_blocks(0), seq / RATIO);
        let block_k = cache
            .qwen4_exp_qsa_block_keys(0)
            .expect("committed block keys");
        assert_close(
            &eval_f32(&block_k),
            &ref_cache.block_k,
            1e-4,
            "dense committed block keys",
        );

        // Dense decode: one token, no mask, matches the reference continuation.
        let decode_hidden = hidden_input(1, 0.73);
        let decode_data = hidden_data(1, 0.73);
        let positions = text_positions(seq, 1);
        let got = mlx_forward(&w, &mut cache, &cfg, &decode_hidden, &positions);
        let expected = ref_qsa_forward(&w, &mut ref_cache, &decode_data, 1, &positions, 512);
        assert_close(&got, &expected, 2e-4, "dense decode");
    }

    #[test]
    fn sparse_selection_matches_cpu_reference_across_chunks() {
        let cfg = test_config(RATIO * TOPK); // block_topk = TOPK
        let w = test_weights();
        let mut cache = MlxKVCache::new(1);
        let mut ref_cache = RefCache::default();

        // Chunk 1: 3 tokens → 1 complete block + a straddling token.
        let positions1 = text_positions(0, 3);
        let got1 = mlx_forward(&w, &mut cache, &cfg, &hidden_input(3, 0.83), &positions1);
        let expected1 = ref_qsa_forward(
            &w,
            &mut ref_cache,
            &hidden_data(3, 0.83),
            3,
            &positions1,
            TOPK,
        );
        assert_close(&got1, &expected1, 2e-4, "sparse chunk 1");
        assert_eq!(cache.qwen4_exp_qsa_committed_blocks(0), 1);

        // Chunk 2: 4 tokens at offset 3 → 7 total, 3 blocks; block 1 pools
        // one history token (straddle) and block selection activates
        // (3 blocks > topk 2). Block-start positions of the straddling block
        // come from the cached f32 ids.
        let positions2 = text_positions(3, 4);
        let got2 = mlx_forward(&w, &mut cache, &cfg, &hidden_input(4, 0.89), &positions2);
        let expected2 = ref_qsa_forward(
            &w,
            &mut ref_cache,
            &hidden_data(4, 0.89),
            4,
            &positions2,
            TOPK,
        );
        assert_close(&got2, &expected2, 2e-4, "sparse chunk 2");
        assert_eq!(cache.qwen4_exp_qsa_committed_blocks(0), 3);

        // Committed block keys were reused, not recomputed: rows match the
        // single-pass reference pooling exactly.
        let block_k = cache
            .qwen4_exp_qsa_block_keys(0)
            .expect("committed block keys");
        assert_close(
            &eval_f32(&block_k),
            &ref_cache.block_k,
            1e-4,
            "reused block keys across chunks",
        );

        // Cached f32 mRoPE ids cover all 7 tokens.
        let pos = cache.qwen4_exp_qsa_positions(0).expect("cached positions");
        let pos_data = eval_f32(&pos);
        assert_eq!(pos_data.len(), 7 * 3);
        for (t, chunk) in pos_data.chunks(3).enumerate() {
            assert_eq!(chunk, &[t as f32, t as f32, t as f32]);
        }
    }

    #[test]
    fn decode_matches_prefill_in_sparse_regime() {
        let cfg = test_config(RATIO * TOPK);
        let w = test_weights();

        // Ground truth: one-shot 8-token prefill on CPU.
        let full_positions = text_positions(0, 8);
        let mut full_hidden = hidden_data(3, 0.97);
        full_hidden.extend(hidden_data(4, 1.01));
        full_hidden.extend(hidden_data(1, 1.03));
        let mut ref_cache = RefCache::default();
        let expected = ref_qsa_forward(&w, &mut ref_cache, &full_hidden, 8, &full_positions, TOPK);

        // Path A: chunked prefill 3+4 then decode 1 (T=8, 4 blocks > topk 2).
        let mut cache_a = MlxKVCache::new(1);
        mlx_forward(
            &w,
            &mut cache_a,
            &cfg,
            &hidden_input(3, 0.97),
            &text_positions(0, 3),
        );
        mlx_forward(
            &w,
            &mut cache_a,
            &cfg,
            &hidden_input(4, 1.01),
            &text_positions(3, 4),
        );
        let decode_a = mlx_forward(
            &w,
            &mut cache_a,
            &cfg,
            &hidden_input(1, 1.03),
            &text_positions(7, 1),
        );

        // Path B: one-shot 8-token prefill; the last row must agree.
        let mut cache_b = MlxKVCache::new(1);
        let full = array_f32(&full_hidden, &[1, 8, HID as i32]);
        let prefill_b = mlx_forward(&w, &mut cache_b, &cfg, &full, &full_positions);

        let expected_row = &expected[7 * HID..8 * HID];
        assert_close(&decode_a, expected_row, 2e-4, "decode vs CPU reference");
        assert_close(
            &prefill_b[7 * HID..8 * HID],
            expected_row,
            2e-4,
            "one-shot prefill last row vs CPU reference",
        );
        assert_close(
            &decode_a,
            &prefill_b[7 * HID..8 * HID],
            2e-4,
            "decode vs prefill row",
        );

        // Both paths committed the same 4 blocks.
        assert_eq!(cache_a.qwen4_exp_qsa_committed_blocks(0), 4);
        assert_eq!(cache_b.qwen4_exp_qsa_committed_blocks(0), 4);
    }

    #[test]
    fn committed_block_keys_stay_f32_with_bf16_activations() {
        let cfg = test_config(RATIO * TOPK);
        let mut w = test_weights();
        // Production dtypes: bf16 activations and projections, so the raw
        // index-key cache adopts bf16. The committed pooled block keys must
        // still stay f32-resident (the scoring path reads them in fp32).
        let cast = |qw: &QuantizedWeight| {
            QuantizedWeight::new(astype(&qw.weight, MlxDtype::Bfloat16, None), None, None)
        };
        w.layer.q_proj = w.layer.q_proj.as_ref().map(cast);
        w.layer.k_proj = w.layer.k_proj.as_ref().map(cast);
        w.layer.v_proj = w.layer.v_proj.as_ref().map(cast);
        w.layer.o_proj = w.layer.o_proj.as_ref().map(cast);
        w.indexer.index_qk_proj = cast(&w.indexer.index_qk_proj);

        // Two complete blocks (block-start RoPE is non-trivial on the second).
        let seq = 2 * RATIO;
        let hidden = astype(&hidden_input(seq, 0.91), MlxDtype::Bfloat16, None);
        let positions = text_positions(0, seq);
        let mut cache = MlxKVCache::new(1);
        let _ = qwen4_exp_qsa_forward(
            &hidden, &w.layer, &w.indexer, &mut cache, 0, &positions, &cfg,
        );
        cache.advance(seq);

        let block_k = cache
            .qwen4_exp_qsa_block_keys(0)
            .expect("committed block keys");
        assert_eq!(block_k.shape(), vec![1, 1, 2, ID as i32]);
        assert_eq!(
            block_k.dtype(),
            MlxDtype::Float32,
            "committed block keys must stay f32-resident"
        );
        let raw = cache.qwen4_exp_qsa_index_keys(0).expect("raw index keys");
        assert_eq!(
            raw.dtype(),
            MlxDtype::Bfloat16,
            "raw per-token index keys keep the activation dtype"
        );

        // CPU f32 reference from the actual cached RAW (bf16) keys: mean-pool
        // each block, k_layernorm (1+γ), RoPE at the block-start position.
        // f32 storage keeps the commit path exact; bf16 storage would inject
        // ~1e-3 quantization noise here.
        let raw = eval_f32(&astype(&raw, MlxDtype::Float32, None));
        assert_eq!(raw.len(), seq * ID);
        let mut expected = Vec::new();
        for b in 0..2 {
            let mut pooled = vec![0.0f32; ID];
            for t in b * RATIO..(b + 1) * RATIO {
                for (e, value) in pooled.iter_mut().enumerate() {
                    *value += raw[t * ID + e] / RATIO as f32;
                }
            }
            let mut row = ref_rmsnorm(&pooled, ID, &w.idx_k_ln);
            let (cos, sin) = ref_mrope(&[positions[b * RATIO]]);
            ref_rope(&mut row, 1, 1, ID, &cos, &sin);
            expected.extend_from_slice(&row);
        }
        assert_close(
            &eval_f32(&block_k),
            &expected,
            1e-6,
            "f32-resident block keys",
        );
    }
}
