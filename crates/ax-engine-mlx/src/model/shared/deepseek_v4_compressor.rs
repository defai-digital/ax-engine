//! DeepSeek V4 (Flash) sliding-window compressor and LID sparse indexer
//! (Phase 3).
//!
//! Compress layers (ratio 4 = CSA / 128 = HCA) summarize every completed
//! block of `ratio` tokens into ONE compressed-K row that attention
//! concatenates after the raw sliding-window latent K (K doubles as V):
//!
//! - Per token at position `p`: `kv_state = W_kv(x)` and
//!   `score_state = W_gate(x) + ape[p % ratio]` (width `coff*dim`,
//!   `coff = 1 + (ratio == 4)`), buffered in F32 between forwards.
//! - When block `b` completes, its row is
//!   `RMSNorm(Σ_j softmax(scores_j) ⊙ kv_j)` over the block window — for
//!   CSA the window is the overlapping pair of blocks `b-1` and `b` (8
//!   rows; each token's state row carries a "previous-half" and a
//!   "current-half" channel set), for HCA just block `b`. The row's pe
//!   slice is RoPE'd at the block-start position `b*ratio` with the
//!   compress-layer rotary configuration (`compress_rope_theta`, YaRN-scaled
//!   when the manifest carries rope_scaling) and appended to the
//!   compressed-K cache.
//! - CSA layers additionally run a dedicated indexer compressor (state
//!   width `2*index_head_dim`) into a separate indexer-K cache, and pick
//!   the top-`index_topk` blocks per query by the lightning-indexer score
//!   `Σ_heads relu(K_idx·q_h) * w_h` (llama.cpp `build_lid_top_k`).
//!
//! Authoritative references: llama.cpp `src/models/deepseek4.cpp`
//! (`build_hca_compressed_kv_from_state`, `build_overlap_compressed_kv_from_state`,
//! `build_lid_top_k`, `build_top_k_mask`) and `src/llama-kv-cache-dsv4.cpp`
//! (`dsv4_build_comp_plan`); vLLM `vllm/models/deepseek_v4/compressor.py` /
//! `attention.py` (`DeepseekV4Indexer`).
//!
//! Unlike llama.cpp's host-side per-ubatch plan, block scheduling here is
//! pure arithmetic on `token_offset`/`seq`: prefill compresses every block
//! completed by the chunk in one vectorized pass, decode compresses exactly
//! when `(pos + 1) % ratio == 0`. llama.cpp's CSA "dummy block" padding is
//! a fixed-shape graph artifact and is not reproduced (MLX shapes are
//! dynamic). The optional hadamard `k_rot` is skipped for the same reason
//! as the raw path: it is an orthogonal involution applied to both sides of
//! every dot product (and undone on the output), so it cancels exactly.

use mlx_sys::{
    MlxArray, MlxDtype, add, arange, astype, concatenate, cos, eval, greater_equal, less,
    logical_and, matmul, maximum, multiply, outer, power, reshape, rms_norm, rope, sin, slice,
    slice_last_dim, softmax_precise, stack, subtract, sum_axis, take, topk_axis, transpose,
    where_cond, zeros,
};

use super::super::config::{DeepseekV4Config, ModelConfig};
use super::deepseek_v4_attention::compress_rope_setup;
use super::utils::{qw, scalar_like};
use crate::kv_cache::MlxKVCache;
use crate::weights::{DeepseekV4IndexerWeights, QuantizedWeight};

/// Borrowed view of one compressor pipeline's weights (the main compressor
/// or the indexer compressor — same quartet, different widths).
struct CompressorWeightsView<'a> {
    kv: &'a QuantizedWeight,
    gate: &'a QuantizedWeight,
    ape: &'a MlxArray,
    norm: &'a MlxArray,
    /// Compressed-row width (`head_dim` main / `index_head_dim` indexer).
    row_dim: usize,
}

/// Committed compressed rows after one forward's compressor update.
#[derive(Default)]
pub(crate) struct DeepseekV4CompFrame {
    /// Committed compressed-K rows `[1, 1, n_rows, head_dim]` (raw-K dtype),
    /// `None` when no block has completed yet.
    pub k_rows: Option<MlxArray>,
    /// `k_rows` row count (== completed blocks).
    pub n_rows: usize,
    /// Committed indexer-K rows `[1, 1, n_rows, index_head_dim]` (CSA only).
    pub indexer_rows: Option<MlxArray>,
}

/// Phase-3 compressor update for one compress layer: compute per-token
/// states for this chunk, compress every block the chunk completes
/// (vectorized over blocks), commit the rows, and retain the partial-window
/// states the next forward still needs. Runs the indexer compressor in
/// lockstep on CSA layers. Returns the committed rows for the attention
/// concat; `n_rows == 0` until the first block completes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deepseek_v4_compressor_update(
    cfg: &ModelConfig,
    v4_cfg: &DeepseekV4Config,
    v4_w: &crate::weights::DeepseekV4LayerWeights,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    ratio: usize,
    out_dtype: MlxDtype,
) -> DeepseekV4CompFrame {
    let seq = x.shape()[1] as usize;
    let overlap = ratio == 4;
    let comp_w = v4_w
        .compressor
        .as_ref()
        .expect("DeepSeek V4 compress layer must carry compressor weights");
    cache.deepseek_v4_comp_ensure(layer_idx, false, ratio, overlap);
    compress_pipeline(
        cfg.rms_norm_eps,
        &CompressorWeightsView {
            kv: &comp_w.kv,
            gate: &comp_w.gate,
            ape: &comp_w.ape,
            norm: &comp_w.norm,
            row_dim: v4_cfg.head_dim,
        },
        v4_cfg.qk_rope_head_dim,
        v4_cfg,
        x,
        cache,
        layer_idx,
        false,
        token_offset,
        seq,
        ratio,
        overlap,
        out_dtype,
    );

    if overlap {
        let idx_w = v4_w
            .indexer
            .as_ref()
            .expect("DeepSeek V4 CSA layer must carry indexer weights");
        cache.deepseek_v4_comp_ensure(layer_idx, true, ratio, true);
        compress_pipeline(
            cfg.rms_norm_eps,
            &CompressorWeightsView {
                kv: &idx_w.compressor_kv,
                gate: &idx_w.compressor_gate,
                ape: &idx_w.compressor_ape,
                norm: &idx_w.compressor_norm,
                row_dim: v4_cfg.index_head_dim,
            },
            v4_cfg.qk_rope_head_dim,
            v4_cfg,
            x,
            cache,
            layer_idx,
            true,
            token_offset,
            seq,
            ratio,
            true,
            out_dtype,
        );
    }

    let n_rows = cache.deepseek_v4_comp_committed(layer_idx, false);
    let k_rows = cache.deepseek_v4_comp_k(layer_idx, false);
    let indexer_rows = if overlap {
        cache.deepseek_v4_comp_k(layer_idx, true)
    } else {
        None
    };
    DeepseekV4CompFrame {
        k_rows,
        n_rows,
        indexer_rows,
    }
}

/// One compressor pipeline's per-forward update: per-token states → block
/// compression → commit + retain.
#[allow(clippy::too_many_arguments)]
fn compress_pipeline(
    rms_eps: f32,
    w: &CompressorWeightsView<'_>,
    rot: usize,
    v4_cfg: &DeepseekV4Config,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    indexer: bool,
    token_offset: usize,
    seq: usize,
    ratio: usize,
    overlap: bool,
    out_dtype: MlxDtype,
) {
    let width = w.row_dim * if overlap { 2 } else { 1 };
    let seq_i = seq as i32;
    let width_i = width as i32;

    // Per-token states (F32, matching llama.cpp's ring precision):
    // kv_state = W_kv(x); score_state = W_gate(x) + ape[p % ratio].
    let kv_state = astype(&qw(x, w.kv), MlxDtype::Float32, None);
    let kv_state = reshape(&kv_state, &[seq_i, width_i], None);
    let score_state = astype(&qw(x, w.gate), MlxDtype::Float32, None);
    let score_state = reshape(&score_state, &[seq_i, width_i], None);
    let score_state = add(
        &score_state,
        &ape_rows(w.ape, token_offset, seq, ratio),
        None,
    );

    // Prepend the buffered partial-window states. Invariant: the buffer
    // covers [base, token_offset), so the concat spans [base, end).
    let end = token_offset + seq;
    let (base, kv_all, score_all) = match cache.deepseek_v4_comp_states(layer_idx, indexer) {
        Some((base, buf_kv, buf_score)) => {
            debug_assert_eq!(
                base + buf_kv.shape()[0] as usize,
                token_offset,
                "DeepSeek V4 compressor buffer must abut the incoming chunk"
            );
            (
                base,
                concatenate(&[&buf_kv, &kv_state], 0, None),
                concatenate(&[&buf_score, &score_state], 0, None),
            )
        }
        None => (token_offset, kv_state, score_state),
    };

    // Compress every block this chunk completes (prefill: possibly many at
    // once; decode: at most one, exactly when crossing a boundary).
    let committed = cache.deepseek_v4_comp_committed(layer_idx, indexer);
    let target = end / ratio;
    if target > committed {
        let rows = compress_blocks(
            &kv_all, &score_all, base, committed, target, ratio, overlap, w.row_dim, rot, v4_cfg,
            w.norm, rms_eps,
        );
        let n_new = (target - committed) as i32;
        let rows = astype(&rows, out_dtype, None);
        let rows = reshape(&rows, &[1, 1, n_new, w.row_dim as i32], None);
        cache.append_deepseek_v4_comp_rows(layer_idx, indexer, rows);
    }
    let committed = committed.max(target);

    // Retain the states the next forward still reads: the overlap
    // compressor keeps the just-completed block as the next block's
    // previous half; both keep the partial current block. `keep_from` is
    // monotone in the normal flow; the `max(base)` clamp only engages after
    // a draft rollback deeper than the retained window (documented on
    // `rewind_deepseek_v4_comps`), where the straddling rows are lost.
    let keep_from = if overlap && committed > 0 {
        (committed - 1) * ratio
    } else {
        committed * ratio
    }
    .max(base);
    if keep_from == end {
        // Exact block boundary with nothing partial to retain: store an
        // empty buffer (avoids a zero-row slice) — the next forward's
        // `None` branch restarts the buffer at `token_offset == keep_from`.
        cache.deepseek_v4_comp_replace_states(layer_idx, indexer, keep_from, None, None);
        return;
    }
    let start = (keep_from - base) as i32;
    let total = kv_all.shape()[0];
    let new_kv = slice(&kv_all, &[start, 0], &[total, width_i], &[1, 1], None);
    let new_score = slice(&score_all, &[start, 0], &[total, width_i], &[1, 1], None);
    cache.deepseek_v4_comp_replace_states(
        layer_idx,
        indexer,
        keep_from,
        Some(new_kv),
        Some(new_score),
    );
}

/// Absolute-positional-embedding rows for this chunk: `ape[(p % ratio)]`
/// per token, `[seq, width]` F32. The HF/vLLM checkpoint layout is
/// `[ratio, coff*dim]` (gather rows); ggml's `{coff*dim, ratio}` layout is
/// accepted too (gather columns) since the converter maps the HF tensor
/// verbatim and the weights doc follows ggml.
fn ape_rows(ape: &MlxArray, token_offset: usize, seq: usize, ratio: usize) -> MlxArray {
    let idx_data: Vec<i32> = (0..seq)
        .map(|i| ((token_offset + i) % ratio) as i32)
        .collect();
    let idx = MlxArray::from_raw_data(
        idx_data.as_ptr().cast(),
        std::mem::size_of_val(idx_data.as_slice()),
        &[seq as i32],
        MlxDtype::Int32,
    );
    // Materialize while `idx_data` is alive so the array owns its bytes.
    eval(&[&idx]);
    let shape = ape.shape();
    let rows = if shape[0] as usize == ratio {
        take(ape, &idx, 0, None)
    } else {
        assert_eq!(
            shape[1] as usize, ratio,
            "DeepSeek V4 compressor ape must be [ratio, coff*dim] or [coff*dim, ratio]"
        );
        transpose(&take(ape, &idx, 1, None), &[1, 0], None)
    };
    astype(&rows, MlxDtype::Float32, None)
}

/// Compress blocks `[committed, target)` from the buffered states.
/// `kv_all`/`score_all` are the F32 `[rows, coff*dim]` states spanning
/// positions `[base, ..)`. Returns the new rows `[n_new, dim]` F32 —
/// RMSNormed, pe slice RoPE'd at the block-start positions.
#[allow(clippy::too_many_arguments)]
fn compress_blocks(
    kv_all: &MlxArray,
    score_all: &MlxArray,
    base: usize,
    committed: usize,
    target: usize,
    ratio: usize,
    overlap: bool,
    row_dim: usize,
    rot: usize,
    v4_cfg: &DeepseekV4Config,
    norm: &MlxArray,
    rms_eps: f32,
) -> MlxArray {
    let n_new = (target - committed) as i32;
    let r = ratio as i32;
    let d = row_dim as i32;
    let width = kv_all.shape()[1];

    // Current-block rows: positions [committed*r, target*r). The overlap
    // compressor reads the SECOND half of each state row here (the
    // "current-window member" channels).
    let (cur_lo, cur_hi) = if overlap { (d, width) } else { (0, d) };
    let cur_kv = gather_positions(
        kv_all,
        base,
        (committed * ratio) as i64,
        target * ratio,
        cur_lo,
        cur_hi,
        false,
    );
    let cur_score = gather_positions(
        score_all,
        base,
        (committed * ratio) as i64,
        target * ratio,
        cur_lo,
        cur_hi,
        true,
    );
    let cur_kv = reshape(&cur_kv, &[n_new, r, d], None);
    let cur_score = reshape(&cur_score, &[n_new, r, d], None);

    let (values, scores) = if overlap {
        // Previous-block rows: positions [(committed-1)*r, (target-1)*r),
        // FIRST half channels. Block 0 has no previous block — llama.cpp
        // appends a synthetic zero/-inf row (`dsv4_append_zero_row`) whose
        // softmax weight is exactly 0; `gather_positions` produces the same
        // effect for every position below `base`.
        let prev_lo = 0;
        let prev_kv = gather_positions(
            kv_all,
            base,
            (committed * ratio) as i64 - ratio as i64,
            target * ratio - ratio,
            prev_lo,
            d,
            false,
        );
        let prev_score = gather_positions(
            score_all,
            base,
            (committed * ratio) as i64 - ratio as i64,
            target * ratio - ratio,
            prev_lo,
            d,
            true,
        );
        let prev_kv = reshape(&prev_kv, &[n_new, r, d], None);
        let prev_score = reshape(&prev_score, &[n_new, r, d], None);
        (
            concatenate(&[&prev_kv, &cur_kv], 1, None),
            concatenate(&[&prev_score, &cur_score], 1, None),
        )
    } else {
        (cur_kv, cur_score)
    };

    // softmax over the window per channel, weighted sum → [n_new, dim]
    // (llama.cpp ggml_soft_max + ggml_sum_rows on the permuted layout).
    let weights = softmax_precise(&scores, 1, None);
    let comp = sum_axis(&multiply(&values, &weights, None), 1, false, None);
    let comp = rms_norm(&comp, Some(norm), rms_eps, None);

    // RoPE the pe slice at the block-start positions b*ratio
    // (`state_write_pos` in llama.cpp's plan), then reassemble the row.
    let pe = block_start_rope(&comp, committed, ratio, rot, v4_cfg, row_dim);
    let nope = slice_last_dim(&comp, 0, d - rot as i32, None);
    concatenate(&[&nope, &pe], -1, None)
}

/// Gather `pos_end - pos_start` rows of positions `[pos_start, pos_end)`
/// from `all` (whose row 0 is position `base`), channels `[lo, hi)`.
/// Positions below `base` (the first block's missing previous half, or rows
/// lost to a deep draft rollback) are filled with zeros for kv states and
/// `-inf` for score states — the exact effect of llama.cpp's
/// `dsv4_append_zero_row`: zero softmax weight times a zero value.
fn gather_positions(
    all: &MlxArray,
    base: usize,
    pos_start: i64,
    pos_end: usize,
    lo: i32,
    hi: i32,
    fill_neg_inf: bool,
) -> MlxArray {
    let needed = (pos_end as i64 - pos_start) as usize;
    let avail_from = (base as i64).max(pos_start).max(0);
    // Clamp both sides so the output is exactly `needed` rows even when the
    // requested window sits entirely below `base` (deep-rollback fallout).
    let n_missing = ((avail_from - pos_start) as usize).min(needed);
    let n_avail = needed - n_missing;
    let width = hi - lo;
    let mut parts: Vec<MlxArray> = Vec::with_capacity(2);
    if n_missing > 0 {
        let fill = zeros(&[n_missing as i32, width], MlxDtype::Float32, None);
        if fill_neg_inf {
            parts.push(add(
                &fill,
                &scalar_like(f32::NEG_INFINITY, MlxDtype::Float32),
                None,
            ));
        } else {
            parts.push(fill);
        }
    }
    if n_avail > 0 {
        let start = (avail_from - base as i64) as i32;
        let stop = start + n_avail as i32;
        parts.push(slice(all, &[start, lo], &[stop, hi], &[1, 1], None));
    }
    let refs: Vec<&MlxArray> = parts.iter().collect();
    concatenate(&refs, 0, None)
}

/// RoPE the pe slice of compressed rows at the block-start positions.
/// `comp` is `[n_new, row_dim]`; row `i` belongs to block `committed + i`,
/// rotated at position `(committed + i) * ratio`. MLX `rope` takes a scalar
/// offset, so the strided positions are computed directly: GPT-J
/// interleaved pairs `(2j, 2j+1)`. The frequency set comes from
/// [`compress_rope_setup`]: plain `compress_rope_theta` inv_freq
/// (`theta^(-2j/rot)`) without rope_scaling, otherwise the reciprocated
/// YaRN divisors (`build_yarn_rope_freqs` returns divisors because MLX
/// `rope` reciprocates internally; the manual angle computation here needs
/// the inverse frequencies themselves) with the YaRN attn factor applied to
/// the pe slice before rotation — the same rotary configuration the
/// attention path uses for compress layers (llama.cpp shares one rotary
/// cache per layer).
fn block_start_rope(
    comp: &MlxArray,
    committed: usize,
    ratio: usize,
    rot: usize,
    v4_cfg: &DeepseekV4Config,
    row_dim: usize,
) -> MlxArray {
    let n = comp.shape()[0];
    let half = (rot / 2) as i32;
    let (rope_base, rope_freqs, pe_scale) = compress_rope_setup(v4_cfg);
    let pe = slice_last_dim(comp, (row_dim - rot) as i32, row_dim as i32, None);
    let pe = if (pe_scale - 1.0).abs() > 1e-6 {
        multiply(&pe, &scalar_like(pe_scale, MlxDtype::Float32), None)
    } else {
        pe
    };

    let positions = arange(
        committed as f64,
        (committed + n as usize) as f64,
        1.0,
        MlxDtype::Float32,
        None,
    );
    let positions = multiply(
        &positions,
        &scalar_like(ratio as f32, MlxDtype::Float32),
        None,
    );
    let exponents = arange(0.0, half as f64, 1.0, MlxDtype::Float32, None);
    let exponents = multiply(
        &exponents,
        &scalar_like(-2.0 / rot as f32, MlxDtype::Float32),
        None,
    );
    let inv_freq = if let Some(theta) = rope_base {
        power(&scalar_like(theta, MlxDtype::Float32), &exponents, None)
    } else if let Some(freqs) = &rope_freqs {
        power(freqs, &scalar_like(-1.0, MlxDtype::Float32), None)
    } else {
        // compress_rope_setup always yields one of the two arms above.
        power(
            &scalar_like(v4_cfg.compress_rope_theta, MlxDtype::Float32),
            &exponents,
            None,
        )
    };
    let angles = outer(&positions, &inv_freq, None); // [n, rot/2]
    let cosines = cos(&angles, None);
    let sines = sin(&angles, None);

    let pairs = reshape(&pe, &[n, half, 2], None);
    let x0 = reshape(&slice_last_dim(&pairs, 0, 1, None), &[n, half], None);
    let x1 = reshape(&slice_last_dim(&pairs, 1, 2, None), &[n, half], None);
    // [n, 1, rot/2] broadcast over the pair axis.
    let cosines = reshape(&cosines, &[n, half], None);
    let sines = reshape(&sines, &[n, half], None);
    let out0 = subtract(
        &multiply(&x0, &cosines, None),
        &multiply(&x1, &sines, None),
        None,
    );
    let out1 = add(
        &multiply(&x0, &sines, None),
        &multiply(&x1, &cosines, None),
        None,
    );
    let out = stack(&[&out0, &out1], -1, None);
    reshape(&out, &[n, (half * 2)], None)
}

/// Boolean compressed-row visibility mask `[seq, n_rows]`: row `j` is
/// visible to the query at absolute position `p` iff block `j` had
/// completed by then, `(j + 1) * ratio <= p + 1`
/// (`plan.n_visible = (pos + 1) / ratio` in llama.cpp). This is the full
/// HCA compressed mask and the base mask the CSA top-k selection is ANDed
/// with (llama.cpp adds both -inf masks; AND of booleans is equivalent).
pub(crate) fn deepseek_v4_visibility_mask(
    seq: usize,
    token_offset: usize,
    ratio: usize,
    n_rows: usize,
) -> MlxArray {
    let visible_data: Vec<i32> = (0..seq)
        .map(|i| ((token_offset + i + 1) / ratio).min(n_rows) as i32)
        .collect();
    let visible = MlxArray::from_raw_data(
        visible_data.as_ptr().cast(),
        std::mem::size_of_val(visible_data.as_slice()),
        &[seq as i32, 1],
        MlxDtype::Int32,
    );
    // Materialize while `visible_data` is alive so the array owns its bytes.
    eval(&[&visible]);
    let rows = arange(0.0, n_rows as f64, 1.0, MlxDtype::Int32, None);
    let rows = reshape(&rows, &[1, n_rows as i32], None);
    less(&rows, &visible, None)
}

/// CSA compressed-row mask `[seq, n_idx]`: visibility AND top-`index_topk`
/// lightning-indexer selection (llama.cpp `build_lid_top_k` +
/// `build_top_k_mask`).
///
/// Score for query `i`, block row `j`:
/// `S[i, j] = Σ_h relu(K_idx[j] · q_rope[i, h]) * w[i, h]` with
/// `q_rope = wq_b_indexer(qr)` (nope/pe split, pe RoPE'd at the query
/// positions with the compress-layer rotary configuration — YaRN-scaled
/// `compress_rope_theta` when rope_scaling is present) and per-token head
/// weights
/// `w = indexer_proj(x) / sqrt(I * Hi)`. Invisible rows score `-inf`
/// before selection, and the final mask is ANDed with visibility, so
/// invisible rows stay masked even when fewer than `topk` rows are visible.
/// When all committed rows fit (`n_idx <= topk`) the selection is a no-op —
/// the short-context path. Ties at the k-th score select every tied row
/// (ggml's `top_k` is exact); the difference is additive-mask-only and
/// visibility-bounded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deepseek_v4_lid_top_k_mask(
    v4_cfg: &DeepseekV4Config,
    idx_w: &DeepseekV4IndexerWeights,
    qr: &MlxArray,
    x: &MlxArray,
    indexer_rows: &MlxArray,
    token_offset: usize,
    seq: usize,
    ratio: usize,
) -> MlxArray {
    let n_idx = indexer_rows.shape()[2] as usize;
    let vis = deepseek_v4_visibility_mask(seq, token_offset, ratio, n_idx);
    let topk = v4_cfg.index_topk;
    if n_idx <= topk {
        return vis;
    }

    let i_dim = v4_cfg.index_head_dim as i32;
    let hi = v4_cfg.index_n_heads as i32;
    let rot = v4_cfg.qk_rope_head_dim;
    let seq_i = seq as i32;

    // indexer_q = wq_b(qr) → [seq, Hi, I]; rope the pe slice at the query
    // positions (consecutive, so MLX rope's scalar offset applies) with the
    // compress-layer YaRN configuration — llama.cpp rotates indexer q/k with
    // the same rotary cache as the layer's compressed attention.
    let (rope_base, rope_freqs, pe_scale) = compress_rope_setup(v4_cfg);
    let q = qw(qr, &idx_w.qb);
    let q = reshape(&q, &[seq_i, hi, i_dim], None);
    let q_nope = slice_last_dim(&q, 0, i_dim - rot as i32, None);
    let q_pe = slice_last_dim(&q, i_dim - rot as i32, i_dim, None);
    let q_pe = if (pe_scale - 1.0).abs() > 1e-6 {
        multiply(&q_pe, &scalar_like(pe_scale, q_pe.dtype()), None)
    } else {
        q_pe
    };
    let q_pe = rope(
        &q_pe,
        rot as i32,
        true,
        rope_base,
        1.0,
        token_offset as i32,
        rope_freqs.as_ref(),
        None,
    );
    let q = astype(
        &concatenate(&[&q_nope, &q_pe], -1, None),
        MlxDtype::Float32,
        None,
    );

    // Per-token head weights w_i = indexer_proj(x) / sqrt(I * Hi).
    let head_scale = 1.0 / ((v4_cfg.index_head_dim * v4_cfg.index_n_heads) as f32).sqrt();
    let w = qw(x, &idx_w.proj);
    let w = multiply(&w, &scalar_like(head_scale, MlxDtype::Float32), None);
    let w = astype(&w, MlxDtype::Float32, None);
    let w = reshape(&w, &[seq_i, hi, 1], None);

    // scores[i, j] = Σ_h relu(K[j]·q[i, h]) * w[i, h]; invisible → -inf.
    let k = reshape(indexer_rows, &[n_idx as i32, i_dim], None);
    let k = astype(&k, MlxDtype::Float32, None);
    let k_t = transpose(&k, &[1, 0], None);
    let dots = matmul(&q, &k_t, None); // [seq, Hi, n_idx]
    let act = maximum(&dots, &scalar_like(0.0, MlxDtype::Float32), None);
    let scores = sum_axis(&multiply(&act, &w, None), 1, false, None); // [seq, n_idx]
    let neg_inf = scalar_like(f32::NEG_INFINITY, MlxDtype::Float32);
    let scores = where_cond(&vis, &scores, &neg_inf, None);

    // Select the top-min(topk, n_idx) rows per query by thresholding at the
    // k-th largest score. MLX `topk` returns the k largest values in
    // ASCENDING order, so the k-th largest is element 0, not element k-1.
    let k = (topk.min(n_idx)) as i32;
    let top_vals = topk_axis(&scores, k, -1, None);
    let threshold = slice_last_dim(&top_vals, 0, 1, None);
    let selected = greater_equal(&scores, &threshold, None);
    logical_and(&vis, &selected, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{
        DeepseekV4CompressorWeights, DeepseekV4IndexerWeights, DeepseekV4LayerWeights,
    };
    use mlx_sys::eval;

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    /// Deterministic pseudo-random fill (no external deps).
    fn fill(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5)
            .collect()
    }

    fn dense(rows: usize, cols: usize, seed: f32) -> QuantizedWeight {
        QuantizedWeight::new(
            array_f32(&fill(rows * cols, seed), &[rows as i32, cols as i32]),
            None,
            None,
        )
    }

    fn f32_data(a: &MlxArray) -> Vec<f32> {
        eval(&[a]);
        a.data_f32().to_vec()
    }

    /// Boolean mask contents as 0.0/1.0 floats (avoids raw-pointer reads).
    fn bool_data(mask: &MlxArray) -> Vec<f32> {
        f32_data(&astype(mask, MlxDtype::Float32, None))
    }

    /// Host-side softmax-weighted sum + RMSNorm reference for one block.
    /// `window_kv`/`window_scores`: `[rows, dim]` flattened row-major.
    fn host_compress_block(
        window_kv: &[f64],
        window_scores: &[f64],
        rows: usize,
        dim: usize,
        norm: &[f64],
        eps: f64,
    ) -> Vec<f64> {
        let mut out = vec![0.0f64; dim];
        for d in 0..dim {
            let mut max = f64::NEG_INFINITY;
            for r in 0..rows {
                max = max.max(window_scores[r * dim + d]);
            }
            let mut denom = 0.0f64;
            let mut acc = 0.0f64;
            for r in 0..rows {
                let w = (window_scores[r * dim + d] - max).exp();
                denom += w;
                acc += w * window_kv[r * dim + d];
            }
            out[d] = acc / denom;
        }
        let rms = (out.iter().map(|v| v * v).sum::<f64>() / dim as f64 + eps).sqrt();
        for d in 0..dim {
            out[d] = out[d] / rms * norm[d];
        }
        out
    }

    /// Minimal config for the rope paths: plain `compress_rope_theta`, no
    /// YaRN scaling (the pre-YaRN-fix behavior these tests pin).
    fn rope_test_cfg(theta: f32) -> DeepseekV4Config {
        DeepseekV4Config {
            head_dim: 16,
            qk_rope_head_dim: TROT,
            q_lora_rank: TRQ,
            o_lora_rank: 4,
            o_groups: 2,
            index_topk: 2,
            index_n_heads: THI,
            index_head_dim: TI,
            compress_rope_theta: theta,
            compress_rope_scaling: None,
            has_attn_sinks: true,
            compress_ratios: vec![4],
            hc_mult: 4,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-5,
            num_hash_layers: 0,
            num_nextn_predict_layers: 0,
            scoring_func: None,
            swiglu_limit: 7.0,
        }
    }

    /// RoPE one compressed row's pe slice via the MLX op at the block-start
    /// position — the reference `block_start_rope` is checked against.
    fn mlx_rope_row(row: &[f32], row_dim: usize, rot: usize, theta: f32, pos: usize) -> Vec<f32> {
        // MLX fast rope requires at least 3 dims — [1, 1, row_dim], not 2-D.
        let arr = array_f32(row, &[1, 1, row_dim as i32]);
        let pe = slice_last_dim(&arr, (row_dim - rot) as i32, row_dim as i32, None);
        let roped = rope(
            &pe,
            rot as i32,
            true,
            Some(theta),
            1.0,
            pos as i32,
            None,
            None,
        );
        f32_data(&roped)
    }

    #[test]
    fn block_start_rope_matches_per_row_mlx_rope() {
        let (n, row_dim, rot, ratio, committed) = (3usize, 16usize, 4usize, 4usize, 2usize);
        let theta = 50000.0f32;
        let comp = array_f32(&fill(n * row_dim, 0.37), &[n as i32, row_dim as i32]);
        let roped = block_start_rope(&comp, committed, ratio, rot, &rope_test_cfg(theta), row_dim);
        let roped = f32_data(&roped);
        let comp_data = f32_data(&comp);
        for i in 0..n {
            let row = &comp_data[i * row_dim..(i + 1) * row_dim];
            let expect = mlx_rope_row(row, row_dim, rot, theta, (committed + i) * ratio);
            for j in 0..rot {
                let actual = roped[i * rot + j];
                assert!(
                    (actual - expect[j]).abs() < 1e-5,
                    "row {i} pe[{j}]: {actual} vs {}",
                    expect[j]
                );
            }
        }
    }

    #[test]
    fn block_start_rope_yarn_applies_scaled_freqs_and_pe_factor() {
        use super::super::rope::build_yarn_rope_freqs;
        use crate::model::config::DeepseekV4CompressRopeScaling;

        let (n, row_dim, rot, ratio, committed) = (2usize, 16usize, 4usize, 4usize, 1usize);
        let theta = 50000.0f32;
        let factor = 16.0f32;
        let mut cfg = rope_test_cfg(theta);
        cfg.compress_rope_scaling = Some(DeepseekV4CompressRopeScaling {
            factor,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_context_len: 65536,
        });
        let comp = array_f32(&fill(n * row_dim, 0.37), &[n as i32, row_dim as i32]);
        let roped = f32_data(&block_start_rope(
            &comp, committed, ratio, rot, &cfg, row_dim,
        ));
        let plain = f32_data(&block_start_rope(
            &comp,
            committed,
            ratio,
            rot,
            &rope_test_cfg(theta),
            row_dim,
        ));
        assert_ne!(roped, plain, "YaRN path must differ from plain theta");

        // Host reference: pe * (1/(1+0.1*ln(factor))), rotated with
        // inv_freq = 1/yarn_divisor at each block-start position.
        let (divisors, mscale) =
            build_yarn_rope_freqs(rot, theta, factor, 65536, 32.0, 1.0, 1.0, 0.0);
        let divisors = f32_data(&divisors);
        let pe_scale = 1.0 / mscale;
        let half = rot / 2;
        let comp_data = f32_data(&comp);
        for i in 0..n {
            let pos = ((committed + i) * ratio) as f32;
            for j in 0..half {
                let angle = pos / divisors[j];
                let (x0, x1) = (
                    comp_data[i * row_dim + (row_dim - rot) + 2 * j] * pe_scale,
                    comp_data[i * row_dim + (row_dim - rot) + 2 * j + 1] * pe_scale,
                );
                let e0 = x0 * angle.cos() - x1 * angle.sin();
                let e1 = x0 * angle.sin() + x1 * angle.cos();
                let a0 = roped[i * rot + 2 * j];
                let a1 = roped[i * rot + 2 * j + 1];
                assert!((a0 - e0).abs() < 1e-4, "row {i} pair {j} re: {a0} vs {e0}");
                assert!((a1 - e1).abs() < 1e-4, "row {i} pair {j} im: {a1} vs {e1}");
            }
        }
    }

    #[test]
    fn compress_blocks_overlap_matches_hand_computed() {
        // CSA overlap: r=4, D=8, rot=2, two blocks in one shot (committed 0 → 2).
        let (r, dim, rot, width) = (4usize, 8usize, 2usize, 16usize);
        let eps = 1e-6f32;
        let theta = 50000.0f32;
        let kv_data = fill(2 * r * width, 0.21);
        let score_data = fill(2 * r * width, 0.53);
        let norm_data = fill(dim, 0.71);
        let kv_all = array_f32(&kv_data, &[(2 * r) as i32, width as i32]);
        let score_all = array_f32(&score_data, &[(2 * r) as i32, width as i32]);
        let norm = array_f32(&norm_data, &[dim as i32]);

        let out = compress_blocks(
            &kv_all,
            &score_all,
            0,
            0,
            2,
            r,
            true,
            dim,
            rot,
            &rope_test_cfg(theta),
            &norm,
            eps,
        );
        let out = f32_data(&out);

        // Host reference, one block at a time. Block 0's previous half is the
        // synthetic zero/-inf row (zero softmax weight), so it compresses
        // exactly its own 4 rows — the same as HCA on that block.
        let norm64: Vec<f64> = norm_data.iter().map(|v| *v as f64).collect();
        for b in 0..2usize {
            let mut window_kv = Vec::new();
            let mut window_scores = Vec::new();
            if b > 0 {
                // previous block rows, FIRST half channels.
                for p in 0..r {
                    for d in 0..dim {
                        window_kv.push(kv_data[p * width + d] as f64);
                        window_scores.push(score_data[p * width + d] as f64);
                    }
                }
            }
            // current block rows, SECOND half channels.
            for p in (b * r)..((b + 1) * r) {
                for d in 0..dim {
                    window_kv.push(kv_data[p * width + dim + d] as f64);
                    window_scores.push(score_data[p * width + dim + d] as f64);
                }
            }
            let rows = window_kv.len() / dim;
            let expect =
                host_compress_block(&window_kv, &window_scores, rows, dim, &norm64, eps as f64);
            // RoPE the pe slice at block start b*r via the MLX op.
            let expect_f32: Vec<f32> = expect.iter().map(|v| *v as f32).collect();
            let expect_pe = mlx_rope_row(&expect_f32, dim, rot, theta, b * r);
            for d in 0..(dim - rot) {
                let actual = out[b * dim + d];
                assert!(
                    (actual - expect_f32[d]).abs() < 1e-4,
                    "block {b} nope[{d}]: {actual} vs {}",
                    expect_f32[d]
                );
            }
            for d in 0..rot {
                let actual = out[b * dim + (dim - rot) + d];
                assert!(
                    (actual - expect_pe[d]).abs() < 1e-4,
                    "block {b} pe[{d}]: {actual} vs {}",
                    expect_pe[d]
                );
            }
        }
    }

    #[test]
    fn compress_blocks_hca_matches_hand_computed() {
        // HCA (no overlap): r=3, D=8, rot=2, coff=1.
        let (r, dim, rot) = (3usize, 8usize, 2usize);
        let eps = 1e-6f32;
        let theta = 10000.0f32;
        let kv_data = fill(2 * r * dim, 0.31);
        let score_data = fill(2 * r * dim, 0.43);
        let norm_data = fill(dim, 0.61);
        let kv_all = array_f32(&kv_data, &[(2 * r) as i32, dim as i32]);
        let score_all = array_f32(&score_data, &[(2 * r) as i32, dim as i32]);
        let norm = array_f32(&norm_data, &[dim as i32]);

        let out = compress_blocks(
            &kv_all,
            &score_all,
            0,
            0,
            2,
            r,
            false,
            dim,
            rot,
            &rope_test_cfg(theta),
            &norm,
            eps,
        );
        let out = f32_data(&out);

        let norm64: Vec<f64> = norm_data.iter().map(|v| *v as f64).collect();
        for b in 0..2usize {
            let mut window_kv = Vec::new();
            let mut window_scores = Vec::new();
            for p in (b * r)..((b + 1) * r) {
                for d in 0..dim {
                    window_kv.push(kv_data[p * dim + d] as f64);
                    window_scores.push(score_data[p * dim + d] as f64);
                }
            }
            let expect =
                host_compress_block(&window_kv, &window_scores, r, dim, &norm64, eps as f64);
            let expect_f32: Vec<f32> = expect.iter().map(|v| *v as f32).collect();
            let expect_pe = mlx_rope_row(&expect_f32, dim, rot, theta, b * r);
            for d in 0..(dim - rot) {
                let actual = out[b * dim + d];
                assert!(
                    (actual - expect_f32[d]).abs() < 1e-4,
                    "block {b} nope[{d}]: {actual} vs {}",
                    expect_f32[d]
                );
            }
            for d in 0..rot {
                let actual = out[b * dim + (dim - rot) + d];
                assert!(
                    (actual - expect_pe[d]).abs() < 1e-4,
                    "block {b} pe[{d}]: {actual} vs {}",
                    expect_pe[d]
                );
            }
        }
    }

    #[test]
    fn ape_rows_gather_hf_and_ggml_layouts() {
        let (ratio, width, seq, offset) = (4usize, 6usize, 5usize, 3usize);
        // HF/vLLM layout [ratio, width].
        let hf = array_f32(&fill(ratio * width, 0.9), &[ratio as i32, width as i32]);
        let rows = ape_rows(&hf, offset, seq, ratio);
        let rows_data = f32_data(&rows);
        let hf_data = fill(ratio * width, 0.9);
        for i in 0..seq {
            let slot = (offset + i) % ratio;
            for d in 0..width {
                assert_eq!(rows_data[i * width + d], hf_data[slot * width + d]);
            }
        }
        // ggml layout [width, ratio] — same table transposed.
        let mut ggml_data = vec![0.0f32; ratio * width];
        for s in 0..ratio {
            for d in 0..width {
                ggml_data[d * ratio + s] = hf_data[s * width + d];
            }
        }
        let ggml = array_f32(&ggml_data, &[width as i32, ratio as i32]);
        let rows2 = ape_rows(&ggml, offset, seq, ratio);
        assert_eq!(f32_data(&rows2), rows_data);
    }

    #[test]
    fn visibility_mask_marks_only_completed_blocks() {
        // r=4, n_rows=3: query at p sees floor((p+1)/4) rows.
        let mask = deepseek_v4_visibility_mask(6, 0, 4, 3);
        let bits = bool_data(&mask);
        let expect: [f32; 18] = [
            0., 0., 0., // p=0 → 0
            0., 0., 0., // p=1 → 0
            0., 0., 0., // p=2 → 0
            1., 0., 0., // p=3 → 1
            1., 0., 0., // p=4 → 1
            1., 0., 0., // p=5 → 1
        ];
        assert_eq!(bits, &expect);

        // Decode-style single query at p=11 (block 2 just completed).
        let mask = deepseek_v4_visibility_mask(1, 11, 4, 3);
        assert_eq!(bool_data(&mask), &[1.0, 1.0, 1.0]);
    }

    /// Tiny indexer config: I=8, Hi=2, rot=4, R_q=16 (=Hi*I so qb can be an
    /// identity-ish map from qr to indexer_q), E=8, topk=2.
    const TI: usize = 8;
    const THI: usize = 2;
    const TROT: usize = 4;
    const TRQ: usize = 16;
    const TE: usize = 8;

    fn lid_test_cfg(topk: usize) -> DeepseekV4Config {
        DeepseekV4Config {
            head_dim: 16,
            qk_rope_head_dim: TROT,
            q_lora_rank: TRQ,
            o_lora_rank: 4,
            o_groups: 2,
            index_topk: topk,
            index_n_heads: THI,
            index_head_dim: TI,
            compress_rope_theta: 50000.0,
            compress_rope_scaling: None,
            has_attn_sinks: true,
            compress_ratios: vec![4],
            hc_mult: 4,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-5,
            num_hash_layers: 0,
            num_nextn_predict_layers: 0,
            scoring_func: None,
            swiglu_limit: 7.0,
        }
    }

    fn lid_test_weights() -> DeepseekV4IndexerWeights {
        DeepseekV4IndexerWeights {
            proj: dense(THI, TE, 0.29),
            qb: dense(THI * TI, TRQ, 0.31),
            compressor_kv: dense(2 * TI, TE, 0.37),
            compressor_gate: dense(2 * TI, TE, 0.41),
            compressor_ape: array_f32(&fill(4 * 2 * TI, 0.43), &[4, (2 * TI) as i32]),
            compressor_norm: array_f32(&fill(TI, 0.47), &[TI as i32]),
        }
    }

    #[test]
    fn lid_top_k_mask_matches_host_computed_scores() {
        // offset=12: every query sees 3 of the 4 rows → topk=2 forces a
        // real selection among visible rows.
        let (seq, n_idx, topk, ratio, offset) = (3usize, 4usize, 2usize, 4usize, 12usize);
        let v4_cfg = lid_test_cfg(topk);
        let idx_w = lid_test_weights();
        let qr = array_f32(&fill(seq * TRQ, 0.51), &[1, seq as i32, TRQ as i32]);
        let x = array_f32(&fill(seq * TE, 0.57), &[1, seq as i32, TE as i32]);
        let k_rows = array_f32(&fill(n_idx * TI, 0.61), &[1, 1, n_idx as i32, TI as i32]);

        let mask =
            deepseek_v4_lid_top_k_mask(&v4_cfg, &idx_w, &qr, &x, &k_rows, offset, seq, ratio);
        let mask_bits = bool_data(&mask);

        // Host oracle: recompute indexer_q (rope included) with the same ops,
        // then score on the host.
        let q = qw(&qr, &idx_w.qb);
        let q = reshape(&q, &[seq as i32, THI as i32, TI as i32], None);
        let q_nope = slice_last_dim(&q, 0, (TI - TROT) as i32, None);
        let q_pe = slice_last_dim(&q, (TI - TROT) as i32, TI as i32, None);
        let q_pe = rope(
            &q_pe,
            TROT as i32,
            true,
            Some(v4_cfg.compress_rope_theta),
            1.0,
            offset as i32,
            None,
            None,
        );
        let q = concatenate(&[&q_nope, &q_pe], -1, None);
        let q_data = f32_data(&q);
        let k_data = f32_data(&k_rows);
        let head_scale = 1.0 / ((TI * THI) as f64).sqrt();
        let w_proj = fill(THI * TE, 0.29);
        let x_data = fill(seq * TE, 0.57);

        for i in 0..seq {
            let n_visible = ((offset + i + 1) / ratio).min(n_idx);
            let mut scores = vec![0.0f64; n_idx];
            for j in 0..n_idx {
                for h in 0..THI {
                    let mut dot = 0.0f64;
                    for d in 0..TI {
                        dot += q_data[(i * THI + h) * TI + d] as f64 * k_data[j * TI + d] as f64;
                    }
                    let mut w = 0.0f64;
                    for e in 0..TE {
                        w += w_proj[h * TE + e] as f64 * x_data[i * TE + e] as f64;
                    }
                    scores[j] += dot.max(0.0) * w * head_scale;
                }
            }
            // top-min(topk, n_idx) by score among visible rows.
            let mut order: Vec<usize> = (0..n_visible).collect();
            order.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap());
            let mut expect = vec![0.0f32; n_idx];
            for &j in order.iter().take(topk.min(n_idx)) {
                expect[j] = 1.0;
            }
            assert_eq!(
                &mask_bits[i * n_idx..(i + 1) * n_idx],
                expect.as_slice(),
                "query {i}: scores {scores:?}"
            );
        }
    }

    #[test]
    fn lid_short_context_selects_all_visible() {
        // n_idx <= topk → selection is a no-op, mask == visibility.
        let (seq, n_idx, ratio, offset) = (2usize, 2usize, 4usize, 4usize);
        let v4_cfg = lid_test_cfg(8);
        let idx_w = lid_test_weights();
        let qr = array_f32(&fill(seq * TRQ, 0.51), &[1, seq as i32, TRQ as i32]);
        let x = array_f32(&fill(seq * TE, 0.57), &[1, seq as i32, TE as i32]);
        let k_rows = array_f32(&fill(n_idx * TI, 0.61), &[1, 1, n_idx as i32, TI as i32]);
        let mask =
            deepseek_v4_lid_top_k_mask(&v4_cfg, &idx_w, &qr, &x, &k_rows, offset, seq, ratio);
        // p=4 → 1 visible, p=5 → 1 visible.
        assert_eq!(bool_data(&mask), &[1.0, 0.0, 1.0, 0.0]);
    }

    /// Full-layer weights with compressor (and indexer on CSA) for the
    /// pipeline schedule tests. Mirrors the attention test builder.
    fn pipeline_test_weights(ratio: usize, indexer: bool) -> crate::weights::LayerWeights {
        const E: usize = 64;
        const D: usize = 16;
        const H: usize = 2;
        const G: usize = 2;
        const R_O: usize = 4;
        const R_Q: usize = 8;
        let overlap = ratio == 4;
        let coff = if overlap { 2 } else { 1 };
        let compressor = Some(DeepseekV4CompressorWeights {
            kv: dense(coff * D, E, 0.11),
            gate: dense(coff * D, E, 0.13),
            ape: array_f32(
                &fill(ratio * coff * D, 0.17),
                &[ratio as i32, (coff * D) as i32],
            ),
            norm: array_f32(&fill(D, 0.19), &[D as i32]),
        });
        let indexer = indexer.then(|| DeepseekV4IndexerWeights {
            proj: dense(THI, E, 0.23),
            qb: dense(THI * TI, R_Q, 0.27),
            compressor_kv: dense(2 * TI, E, 0.29),
            compressor_gate: dense(2 * TI, E, 0.31),
            compressor_ape: array_f32(
                &fill(ratio * 2 * TI, 0.37),
                &[ratio as i32, (2 * TI) as i32],
            ),
            compressor_norm: array_f32(&fill(TI, 0.41), &[TI as i32]),
        });
        crate::weights::LayerWeights {
            attn_norm: array_f32(&fill(E, 0.9), &[E as i32]),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            attn_out_gate: None,
            o_proj: None,
            linear_attn: None,
            glm_mla_attn: None,
            deepseek_v4: Some(DeepseekV4LayerWeights {
                wq_a: dense(R_Q, E, 0.11),
                q_a_norm: array_f32(&fill(R_Q, 0.8), &[R_Q as i32]),
                wq_b: dense(H * D, R_Q, 0.13),
                wkv: dense(D, E, 0.17),
                kv_norm: array_f32(&fill(D, 0.8), &[D as i32]),
                wo_a: dense(G * R_O, H * D / G, 0.19),
                wo_b: dense(E, G * R_O, 0.23),
                attn_sink: Some(array_f32(&[-1.0, -2.0], &[H as i32])),
                hc_attn_fn: array_f32(&[1.0], &[1]),
                hc_attn_base: array_f32(&[1.0], &[1]),
                hc_attn_scale: array_f32(&[1.0], &[1]),
                hc_ffn_fn: array_f32(&[1.0], &[1]),
                hc_ffn_base: array_f32(&[1.0], &[1]),
                hc_ffn_scale: array_f32(&[1.0], &[1]),
                compressor,
                indexer,
                tid2eid: None,
            }),
            ffn_norm: array_f32(&fill(E, 0.9), &[E as i32]),
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
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
            expert_stream: None,
        }
    }

    fn pipeline_test_cfg(ratio: u32) -> ModelConfig {
        const E: usize = 64;
        const D: usize = 16;
        const H: usize = 2;
        const ROT: usize = 4;
        ModelConfig {
            compile_cache_identity: 1,
            model_family: "deepseek_v4".to_string(),
            layer_count: 1,
            hidden_size: E,
            intermediate_size: 8,
            n_heads: H,
            n_kv_heads: 1,
            head_dim: D,
            vocab_size: 16,
            rope_theta: 10000.0,
            rope_dims: ROT,
            attn_output_gate: false,
            query_scale: 1.0 / (D as f32).sqrt(),
            final_logit_softcapping: None,
            final_logits_scale: None,
            post_norm_eps: 1e-6,
            embed_norm_no_weight: false,
            moe_expert_count: 0,
            moe_experts_per_token: 0,
            moe_expert_intermediate_size: 0,
            layer_configs: Vec::new(),
            global_sliding_window: None,
            protected_prefix_sliding_window: None,
            gemma4_moe_router: false,
            uses_geglu: false,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            linear_attention: None,
            mla_attention: None,
            glm_router: None,
            deepseek_v4: Some(DeepseekV4Config {
                head_dim: D,
                qk_rope_head_dim: ROT,
                q_lora_rank: 8,
                o_lora_rank: 4,
                o_groups: 2,
                index_topk: 8,
                index_n_heads: THI,
                index_head_dim: TI,
                compress_rope_theta: 50000.0,
                compress_rope_scaling: None,
                has_attn_sinks: true,
                compress_ratios: vec![ratio],
                hc_mult: 4,
                hc_sinkhorn_iters: 3,
                hc_eps: 1e-5,
                num_hash_layers: 0,
                num_nextn_predict_layers: 0,
                scoring_func: None,
                swiglu_limit: 7.0,
            }),
            rms_norm_eps: 1e-6,
            rope_freqs: None,
            rope_mscale: 1.0,
            no_rope_layer_interval: 0,
            attn_temperature_floor: 8192.0,
            attn_temperature_scale: 0.1,
            intermediate_size_mlp: 0,
            moe_layer_freq: 1,
            moe_first_dense_layers: 0,
            moe_shared_expert_count: 0,
            moe_sigmoid_routing: false,
            moe_routed_scaling_factor: 1.0,
            moe_n_group: 1,
            moe_topk_group: 1,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: None,
            gpt_oss_uses_mxfp4_experts: false,
            generation_kind: ax_engine_core::GenerationKind::Autoregressive,
            kv_cache_quant: vec![None; 1],
        }
    }

    /// Drive one forward's worth of compressor scheduling: raw append (creates
    /// the cache entry), compressor update at `token_offset`, then advance.
    fn pipeline_step(
        cfg: &ModelConfig,
        w: &crate::weights::LayerWeights,
        cache: &mut MlxKVCache,
        x: &MlxArray,
        token_offset: usize,
        ratio: usize,
    ) -> DeepseekV4CompFrame {
        let seq = x.shape()[1] as usize;
        let kv = zeros(&[1, 1, seq as i32, 16], MlxDtype::Float32, None);
        let _ = cache.append_deepseek_v4(0, kv);
        let v4_cfg = cfg.deepseek_v4.as_ref().expect("v4 cfg");
        let v4_w = w.deepseek_v4.as_ref().expect("v4 weights");
        let frame = deepseek_v4_compressor_update(
            cfg,
            v4_cfg,
            v4_w,
            x,
            cache,
            0,
            token_offset,
            ratio,
            MlxDtype::Float32,
        );
        cache.advance(seq);
        frame
    }

    #[test]
    fn csa_pipeline_commits_rows_at_block_boundaries() {
        const E: usize = 64;
        let ratio = 4usize;
        let cfg = pipeline_test_cfg(4);
        let w = pipeline_test_weights(ratio, true);
        let mut cache = MlxKVCache::new(1);

        // Prefill 6 tokens: block 0 completes (positions 0..=3), 4..=5 partial.
        let x = array_f32(&fill(6 * E, 0.63), &[1, 6, E as i32]);
        let frame = pipeline_step(&cfg, &w, &mut cache, &x, 0, ratio);
        assert_eq!(frame.n_rows, 1);
        assert_eq!(cache.deepseek_v4_comp_committed(0, false), 1);
        assert_eq!(cache.deepseek_v4_comp_committed(0, true), 1);
        // CSA retains the just-completed block as the next block's previous
        // half: buffer spans [0, 6).
        let (base, kv_buf, score_buf) = cache.deepseek_v4_comp_states(0, false).expect("states");
        assert_eq!(base, 0);
        assert_eq!(kv_buf.shape(), vec![6, 32]);
        assert_eq!(score_buf.shape(), vec![6, 32]);

        // Decode pos 6: no boundary crossed.
        let x = array_f32(&fill(E, 0.67), &[1, 1, E as i32]);
        let frame = pipeline_step(&cfg, &w, &mut cache, &x, 6, ratio);
        assert_eq!(frame.n_rows, 1);
        let (_, kv_buf, _) = cache.deepseek_v4_comp_states(0, false).expect("states");
        assert_eq!(kv_buf.shape()[0], 7);

        // Decode pos 7: block 1 completes → second row; buffer rewinds to
        // block 1's rows ([4, 8)).
        let x = array_f32(&fill(E, 0.71), &[1, 1, E as i32]);
        let frame = pipeline_step(&cfg, &w, &mut cache, &x, 7, ratio);
        assert_eq!(frame.n_rows, 2);
        assert_eq!(cache.deepseek_v4_comp_committed(0, true), 2);
        let (base, kv_buf, _) = cache.deepseek_v4_comp_states(0, false).expect("states");
        assert_eq!(base, 4);
        assert_eq!(kv_buf.shape()[0], 4);
        let rows = frame.k_rows.expect("committed rows");
        assert_eq!(rows.shape(), vec![1, 1, 2, 16]);
        assert!(f32_data(&rows).iter().all(|v| v.is_finite()));
        let idx_rows = frame.indexer_rows.expect("indexer rows");
        assert_eq!(idx_rows.shape(), vec![1, 1, 2, TI as i32]);
        assert!(f32_data(&idx_rows).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn hca_pipeline_commits_rows_at_block_boundaries() {
        const E: usize = 64;
        // r=8 stands in for the real HCA ratio 128 (config ratios are a plain
        // Vec<u32>; the pipeline is parameterized by ratio).
        let ratio = 8usize;
        let cfg = pipeline_test_cfg(8);
        let w = pipeline_test_weights(ratio, false);
        let mut cache = MlxKVCache::new(1);

        // Prefill 10 tokens: block 0 (0..=7) completes, 8..=9 partial.
        let x = array_f32(&fill(10 * E, 0.63), &[1, 10, E as i32]);
        let frame = pipeline_step(&cfg, &w, &mut cache, &x, 0, ratio);
        assert_eq!(frame.n_rows, 1);
        assert!(frame.indexer_rows.is_none());
        // HCA retains only the partial current block: [8, 10).
        let (base, kv_buf, _) = cache.deepseek_v4_comp_states(0, false).expect("states");
        assert_eq!(base, 8);
        assert_eq!(kv_buf.shape(), vec![2, 16]);

        // Decode to the end of block 1 (pos 15): second row, buffer clears
        // (exact boundary — nothing partial to retain).
        for pos in 10..16 {
            let x = array_f32(&fill(E, 0.01 * pos as f32), &[1, 1, E as i32]);
            let frame = pipeline_step(&cfg, &w, &mut cache, &x, pos, ratio);
            if pos < 15 {
                assert_eq!(frame.n_rows, 1, "pos {pos} must not commit yet");
            } else {
                assert_eq!(frame.n_rows, 2, "pos 15 completes block 1");
            }
        }
        assert!(cache.deepseek_v4_comp_states(0, false).is_none());
    }

    #[test]
    fn pipeline_committed_row_matches_hand_computed_block() {
        // CSA r=4: prefill exactly one block; row 0 must equal the
        // hand-computed compression of positions 0..=3 (first block → only
        // its own current-half channels participate).
        const E: usize = 64;
        const D: usize = 16;
        const ROT: usize = 4;
        let ratio = 4usize;
        let cfg = pipeline_test_cfg(4);
        let w = pipeline_test_weights(ratio, true);
        let mut cache = MlxKVCache::new(1);

        let x_data = fill(4 * E, 0.73);
        let x = array_f32(&x_data, &[1, 4, E as i32]);
        let frame = pipeline_step(&cfg, &w, &mut cache, &x, 0, ratio);
        let rows = frame.k_rows.expect("one committed row");
        let row = f32_data(&rows);

        // Host reference: kv/score states = x @ W^T (+ ape slot p%4), second
        // half channels; softmax-weighted sum; RMSNorm; rope at position 0.
        let v4_w = w.deepseek_v4.as_ref().expect("v4 weights");
        let comp_w = v4_w.compressor.as_ref().expect("compressor");
        let wkv = fill(2 * D * E, 0.11);
        let wgate = fill(2 * D * E, 0.13);
        let ape = fill(ratio * 2 * D, 0.17);
        let norm = fill(D, 0.19);
        let mut window_kv = Vec::new();
        let mut window_scores = Vec::new();
        for p in 0..ratio {
            for d in 0..D {
                let mut kv = 0.0f64;
                let mut sc = 0.0f64;
                for e in 0..E {
                    kv += wkv[(D + d) * E + e] as f64 * x_data[p * E + e] as f64;
                    sc += wgate[(D + d) * E + e] as f64 * x_data[p * E + e] as f64;
                }
                sc += ape[p * 2 * D + (D + d)] as f64;
                window_kv.push(kv);
                window_scores.push(sc);
            }
        }
        let norm64: Vec<f64> = norm.iter().map(|v| *v as f64).collect();
        let expect = host_compress_block(
            &window_kv,
            &window_scores,
            ratio,
            D,
            &norm64,
            cfg.rms_norm_eps as f64,
        );
        let expect_f32: Vec<f32> = expect.iter().map(|v| *v as f32).collect();
        let v4_cfg = cfg.deepseek_v4.as_ref().expect("v4 cfg");
        let expect_pe = mlx_rope_row(&expect_f32, D, ROT, v4_cfg.compress_rope_theta, 0);
        for d in 0..(D - ROT) {
            assert!(
                (row[d] - expect_f32[d]).abs() < 1e-4,
                "nope[{d}]: {} vs {}",
                row[d],
                expect_f32[d]
            );
        }
        for d in 0..ROT {
            assert!(
                (row[D - ROT + d] - expect_pe[d]).abs() < 1e-4,
                "pe[{d}]: {} vs {}",
                row[D - ROT + d],
                expect_pe[d]
            );
        }
        // Silence unused warning for the weight handles used only for seeds.
        let _ = (comp_w, &v4_w.indexer);
    }
}
