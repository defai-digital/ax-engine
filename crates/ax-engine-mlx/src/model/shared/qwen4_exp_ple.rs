//! PLE (per-layer n-gram embedding) floor for Qwen4-Exp (`qwen4_exp`).
//!
//! One PLE layer per model (0-indexed decoder layer 1). Per forward it:
//! 1. hashes the raw token ids (plus the cached last-`context_len` tokens)
//!    into `[L, ngram_heads]` row ids of ONE logical n-gram table — bigram
//!    and trigram mixes, EOS-segment aware;
//! 2. gathers those rows from the lazily-loaded sharded table (8-bit g32
//!    affine; only the touched shards are mmap-loaded, the 320M-row join is
//!    never materialized);
//! 3. applies the gated wrap: a per-stream sigmoid gate over the grouped
//!    key/query norms, the value projection, and a dilated (d=3) causal
//!    depthwise conv with a cached tail.
//!
//! The trunk adds the returned `[1, L, hc*hidden]` value to all `hc_count`
//! residual streams at the top of decoder layer 1 and persists the state
//! rings via `MlxKVCache::set_qwen4_exp_ple_state`.
//!
//! Authoritative spec: `.internal/planning/qwen38-flash-next-support.md`
//! section "B. PLE floor" (HF `Qwen4ExpPle`). All wrap math runs in f32; the
//! output is cast back to the residual stream dtype.
//!
//! The n-gram hash runs on the host in i64: the reference relies on every
//! `token * multiplier` product staying below 2^63 (verified for the shipped
//! constants: max id 248,044 x max mult 23,703,573,157,769 ~= 5.88e18 <
//! 9.22e18), and mlx-sys exposes no int64 xor/remainder ops. `wrapping_mul`
//! keeps a pathological pack defined-behaved; with positive multipliers and
//! non-negative token ids the XOR-accumulated mixes stay non-negative, so `%`
//! is a plain remainder.

use mlx_sys::{
    MlxArray, MlxDtype, add, astype, concatenate, conv1d, dequantize_with_mode, divide, equal,
    eval, greater_equal, maximum, multiply, negative, power, reshape, sigmoid, slice, sum_axis,
    take, view, where_cond, zeros,
};

use crate::weights::{QuantizedWeight, Qwen4ExpPleTableShard, Qwen4ExpPleWeights};

use super::{grouped_rms_norm, qw};

/// Depthwise-conv dilation of the PLE wrap (planning doc section B.6). Not
/// stored in the pack; fixed by the reference architecture. With kernel 4 the
/// cached conv tail is `(kernel - 1) * dilation` = 9 rows.
const PLE_CONV_DILATION: i32 = 3;

/// Floor inside the signed-sqrt gate: `sign(g) * sqrt(max(|g|, eps))`.
const PLE_SIGNED_SQRT_EPS: f32 = 1e-6;

/// Read an I64 MLX buffer to the host. mlx-sys has no `data_i64`, so the
/// eval'd array is reinterpreted as little-endian u32 word pairs and each
/// i64 is rebuilt from its two words.
fn read_i64_buffer(array: &MlxArray) -> Result<Vec<i64>, String> {
    if array.dtype() != MlxDtype::Int64 {
        return Err("qwen4_exp PLE hash buffers must be I64".to_string());
    }
    eval(&[array]);
    let words = view(array, MlxDtype::Uint32, None);
    eval(&[&words]);
    let data = words.data_u32();
    Ok(data
        .chunks_exact(2)
        .map(|pair| (u64::from(pair[0]) | (u64::from(pair[1]) << 32)) as i64)
        .collect())
}

/// EOS-segment-aware n-gram hash context loaded from the pack's I64 buffers
/// (`ple.ple_embedding.{layer_multipliers,ngram_heads_vocab_sizes,ngram_heads_offsets}`).
/// The pack ships the exact splitmix64 multipliers; they are read, never
/// recomputed.
pub(crate) struct Qwen4ExpPleHash {
    /// Odd splitmix64 multipliers per shift (`layer_multipliers`), `[ngram_size]`.
    mults: Vec<i64>,
    /// Per-head prime vocab sizes, `[ngram_heads]` (bigram heads first).
    vocab_sizes: Vec<i64>,
    /// Per-head cumulative row offsets into the logical table, `[ngram_heads]`.
    offsets: Vec<i64>,
    /// Heads per hash order (`heads_per_ngram` config).
    heads_per_ngram: usize,
}

impl Qwen4ExpPleHash {
    /// Build the hash context from the raw pack buffers, asserting the
    /// buffer counts against the configured geometry (`ngram_size` orders,
    /// `(ngram_size - 1) * heads_per_ngram` heads).
    fn from_buffers(
        layer_multipliers: &MlxArray,
        ngram_heads_vocab_sizes: &MlxArray,
        ngram_heads_offsets: &MlxArray,
        ngram_size: usize,
        heads_per_ngram: usize,
    ) -> Result<Self, String> {
        let mults = read_i64_buffer(layer_multipliers)?;
        let vocab_sizes = read_i64_buffer(ngram_heads_vocab_sizes)?;
        let offsets = read_i64_buffer(ngram_heads_offsets)?;
        if ngram_size < 2 || heads_per_ngram == 0 {
            return Err(format!(
                "qwen4_exp PLE geometry must have ngram_size >= 2 and heads_per_ngram >= 1, \
                 got ngram_size {ngram_size}, heads_per_ngram {heads_per_ngram}"
            ));
        }
        if mults.len() != ngram_size {
            return Err(format!(
                "qwen4_exp PLE layer_multipliers has {} entries, config ngram_size is {ngram_size}",
                mults.len()
            ));
        }
        let heads = (ngram_size - 1) * heads_per_ngram;
        if vocab_sizes.len() != heads || offsets.len() != heads {
            return Err(format!(
                "qwen4_exp PLE ngram head buffers must hold {heads} entries \
                 ((ngram_size - 1) * heads_per_ngram), got {} vocab sizes and {} offsets",
                vocab_sizes.len(),
                offsets.len()
            ));
        }
        if mults.iter().any(|mult| *mult <= 0) {
            return Err(
                "qwen4_exp PLE layer_multipliers must be positive odd i64 values".to_string(),
            );
        }
        if vocab_sizes.iter().any(|size| *size <= 0) {
            return Err("qwen4_exp PLE ngram_heads_vocab_sizes must be positive".to_string());
        }
        if offsets.iter().any(|offset| *offset < 0) {
            return Err("qwen4_exp PLE ngram_heads_offsets must be non-negative".to_string());
        }
        Ok(Self {
            mults,
            vocab_sizes,
            offsets,
            heads_per_ngram,
        })
    }

    /// Tokens of hash context carried across calls (`ngram_size - 1`).
    fn context_len(&self) -> usize {
        self.mults.len() - 1
    }

    /// Total n-gram heads (`(ngram_size - 1) * heads_per_ngram`).
    fn head_count(&self) -> usize {
        self.vocab_sizes.len()
    }

    /// Global table row ids for `tokens`, laid out row-major `[L, ngram_heads]`
    /// (head index is fastest). `history` holds the raw last `context_len`
    /// tokens of the previous call (`eos_token_id`-filled at sequence start).
    fn hash_ids(&self, tokens: &[i64], history: &[i64], eos_token_id: i64) -> Vec<i64> {
        let context = self.context_len();
        assert_eq!(
            history.len(),
            context,
            "qwen4_exp PLE hash history must hold context_len tokens"
        );
        let ngram_size = self.mults.len();
        let mut full = Vec::with_capacity(context + tokens.len());
        full.extend_from_slice(history);
        full.extend_from_slice(tokens);
        // shifted[s][i] = full[i - s], except positions at/before the previous
        // EOS (which would cross a segment boundary): those stay eos_token_id.
        // Rows are EOS-filled up front and only overwritten when the source
        // position is strictly after the most recent EOS before i. Only the
        // current-token positions (i >= context) are consumed below.
        let mut shifted: Vec<Vec<i64>> = vec![vec![eos_token_id; full.len()]; ngram_size];
        let mut last_eos: i64 = -1;
        for (i, &tok) in full.iter().enumerate() {
            for (s, row) in shifted.iter_mut().enumerate() {
                if i >= s && (i - s) as i64 > last_eos {
                    row[i] = full[i - s];
                }
            }
            if tok == eos_token_id {
                last_eos = i as i64;
            }
        }
        let heads = self.head_count();
        let mut ids = vec![0i64; tokens.len() * heads];
        for (pos, out) in ids.chunks_exact_mut(heads).enumerate() {
            let i = context + pos;
            for order in 2..=ngram_size {
                let mut mixed: i64 = 0;
                for (s, row) in shifted.iter().take(order).enumerate() {
                    mixed ^= row[i].wrapping_mul(self.mults[s]);
                }
                let base = (order - 2) * self.heads_per_ngram;
                for h in 0..self.heads_per_ngram {
                    let head = base + h;
                    out[head] = mixed % self.vocab_sizes[head] + self.offsets[head];
                }
            }
        }
        ids
    }
}

/// Replace masked (padding) positions with EOS, as the reference does when a
/// conv mask exists. Returns the token ids the hash and history see.
fn effective_tokens(
    input_ids: &[u32],
    conv_mask: Option<&[bool]>,
    eos_token_id: i64,
) -> Result<Vec<i64>, String> {
    if let Some(mask) = conv_mask {
        if mask.len() != input_ids.len() {
            return Err(format!(
                "qwen4_exp PLE conv mask has {} entries for {} input ids",
                mask.len(),
                input_ids.len()
            ));
        }
        Ok(input_ids
            .iter()
            .zip(mask.iter())
            .map(|(&id, &real)| if real { i64::from(id) } else { eos_token_id })
            .collect())
    } else {
        Ok(input_ids.iter().map(|&id| i64::from(id)).collect())
    }
}

/// One shard of the logical n-gram table as the gather sees it. Production
/// sources are `Qwen4ExpPleTableShard`s (lazy mmap per shard); tests
/// substitute in-memory quantized weights and track loads.
trait NgramShardSource {
    /// Load-state probe; only the laziness tests consume it.
    #[cfg_attr(not(test), allow(dead_code))]
    fn shard_is_loaded(&self) -> bool;
    fn shard_quantized(&self) -> Result<&QuantizedWeight, String>;
}

impl NgramShardSource for &Qwen4ExpPleTableShard {
    fn shard_is_loaded(&self) -> bool {
        Qwen4ExpPleTableShard::is_loaded(self)
    }

    fn shard_quantized(&self) -> Result<&QuantizedWeight, String> {
        self.load().map_err(|error| error.to_string())
    }
}

/// One shard's worth of pending row gathers.
struct ShardGatherGroup {
    shard: usize,
    /// Local row ids inside the shard (same order as `positions`).
    locals: Vec<u32>,
    /// Flat `[L * heads]` positions these rows answer.
    positions: Vec<u32>,
}

/// Bucket global table ids by shard (`id / rows_per_shard`), preserving
/// per-shard position order so the inverse permutation in
/// [`gather_ngram_rows`] is exact. Shards not touched by `ids` never appear
/// in the output, so they are never loaded.
fn group_ids_by_shard(
    ids: &[i64],
    rows_per_shard: i64,
    shard_count: usize,
) -> Result<Vec<ShardGatherGroup>, String> {
    if rows_per_shard <= 0 || shard_count == 0 {
        return Err(format!(
            "qwen4_exp PLE n-gram table geometry is invalid: rows_per_shard {rows_per_shard}, \
             shard_count {shard_count}"
        ));
    }
    let total_rows = rows_per_shard * shard_count as i64;
    let mut groups: Vec<ShardGatherGroup> = Vec::new();
    let mut group_of_shard = vec![usize::MAX; shard_count];
    for (position, &id) in ids.iter().enumerate() {
        if id < 0 || id >= total_rows {
            return Err(format!(
                "qwen4_exp PLE n-gram id {id} is outside the {shard_count}-shard table \
                 ({total_rows} rows)"
            ));
        }
        let shard = (id / rows_per_shard) as usize;
        let local = u32::try_from(id % rows_per_shard)
            .map_err(|_| "qwen4_exp PLE n-gram local row id exceeds u32".to_string())?;
        let position = u32::try_from(position)
            .map_err(|_| "qwen4_exp PLE gather position count exceeds u32".to_string())?;
        if group_of_shard[shard] == usize::MAX {
            group_of_shard[shard] = groups.len();
            groups.push(ShardGatherGroup {
                shard,
                locals: Vec::new(),
                positions: Vec::new(),
            });
        }
        let group = &mut groups[group_of_shard[shard]];
        group.locals.push(local);
        group.positions.push(position);
    }
    Ok(groups)
}

/// Gather `local_ids` rows out of an already-loaded quantized shard — the
/// same per-tensor `take` + `dequantize` path `embed_tokens_arr` uses for
/// the 8-bit g32 token embedding. Returns `[local_ids.len(), dim]` f32.
fn gather_quantized_rows(qw: &QuantizedWeight, local_ids: &[u32]) -> Result<MlxArray, String> {
    let Some(scales) = &qw.scales else {
        return Err("qwen4_exp PLE n-gram shard must be quantized (no scales)".to_string());
    };
    let indices = MlxArray::from_raw_data(
        local_ids.as_ptr() as *const u8,
        std::mem::size_of_val(local_ids),
        &[local_ids.len() as i32],
        MlxDtype::Uint32,
    );
    let row_w = take(&qw.weight, &indices, 0, None);
    let row_s = take(scales, &indices, 0, None);
    let row_b = qw.biases.as_ref().map(|b| take(b, &indices, 0, None));
    Ok(dequantize_with_mode(
        &row_w,
        &row_s,
        row_b.as_ref(),
        Some(qw.group_size),
        Some(qw.bits),
        qw.mlx_quantization_mode(),
        None,
        None,
        None,
    ))
}

/// Gather global table rows for `ids` (flat `[L * heads]`, row-major) into
/// `[ids.len(), head_dim]` f32. Only the shards the ids touch are loaded;
/// each loaded shard contributes one row gather, and a single inverse-
/// permutation `take` restores id order.
///
/// The result is materialized here and owns its data on return:
/// `from_raw_data` borrows the host index vectors, so the graph consuming
/// them must be evaluated before they drop (same contract as
/// `batched_decode_validity_mask`).
fn gather_ngram_rows<S: NgramShardSource>(
    shards: &[S],
    ids: &[i64],
    rows_per_shard: i64,
) -> Result<MlxArray, String> {
    if ids.is_empty() {
        return Err("qwen4_exp PLE gather requires at least one id".to_string());
    }
    let groups = group_ids_by_shard(ids, rows_per_shard, shards.len())?;
    let mut parts: Vec<MlxArray> = Vec::with_capacity(groups.len());
    let mut order: Vec<u32> = Vec::with_capacity(ids.len());
    for group in &groups {
        let shard = shards
            .get(group.shard)
            .ok_or_else(|| format!("qwen4_exp PLE shard index {} out of range", group.shard))?;
        parts.push(gather_quantized_rows(
            shard.shard_quantized()?,
            &group.locals,
        )?);
        order.extend_from_slice(&group.positions);
    }
    let part_refs: Vec<&MlxArray> = parts.iter().collect();
    let gathered = concatenate(&part_refs, 0, None);
    // `gathered` row j answers flat position order[j]; invert the permutation
    // so row p of the result answers ids[p].
    let mut inverse = vec![0u32; ids.len()];
    for (j, &position) in order.iter().enumerate() {
        inverse[position as usize] = u32::try_from(j)
            .map_err(|_| "qwen4_exp PLE gather row count exceeds u32".to_string())?;
    }
    let perm = MlxArray::from_raw_data(
        inverse.as_ptr() as *const u8,
        std::mem::size_of_val(inverse.as_slice()),
        &[inverse.len() as i32],
        MlxDtype::Uint32,
    );
    let rows = take(&gathered, &perm, 0, None);
    eval(&[&rows]);
    Ok(rows)
}

/// Dilated causal depthwise conv over `[cached_tail, x]` — mirrors
/// `linear_attention_conv1d` (gated-delta layers) but with dilation 3, so
/// the cached tail is `(kernel - 1) * dilation` rows (9 at k=4/d=3) instead
/// of `kernel - 1`. Returns the raw conv output (the wrap applies SiLU) and
/// the new tail.
fn ple_conv1d(
    x: &MlxArray,
    conv_weight: &MlxArray,
    cached_state: Option<&MlxArray>,
) -> (MlxArray, MlxArray) {
    let shape = x.shape();
    let (batch, channels) = (shape[0], shape[2]);
    let weight_shape = conv_weight.shape();
    let kernel = weight_shape[1];
    let tail = (kernel - 1) * PLE_CONV_DILATION;
    assert_eq!(
        weight_shape,
        vec![channels, kernel, 1],
        "qwen4_exp PLE conv weight must be [channels, kernel, 1]"
    );
    let conv_state = cached_state
        .cloned()
        .unwrap_or_else(|| zeros(&[batch, tail, channels], x.dtype(), None));
    assert_eq!(
        conv_state.shape(),
        vec![batch, tail, channels],
        "qwen4_exp PLE conv state must be [batch, (kernel - 1) * dilation, channels]"
    );
    let conv_input = concatenate(&[&conv_state, x], 1, None);
    let total = conv_input.shape()[1];
    let new_state = slice(
        &conv_input,
        &[0, total - tail, 0],
        &[batch, total, channels],
        &[1, 1, 1],
        None,
    );
    let conv_out = conv1d(
        &conv_input,
        conv_weight,
        1,
        0,
        PLE_CONV_DILATION,
        channels,
        None,
    );
    (conv_out, new_state)
}

/// `sign(g) * sqrt(max(|g|, 1e-6))` elementwise in f32. `sign(0) = 0` matches
/// the reference: an exactly-zero gate stays zero.
fn signed_sqrt(g: &MlxArray) -> MlxArray {
    let zero = mlx_sys::ops::cached_scalar(0.0, MlxDtype::Float32);
    let abs = maximum(g, &negative(g, None), None);
    let floored = maximum(
        &abs,
        &mlx_sys::ops::cached_scalar(PLE_SIGNED_SQRT_EPS, MlxDtype::Float32),
        None,
    );
    let root = power(
        &floored,
        &mlx_sys::ops::cached_scalar(0.5, MlxDtype::Float32),
        None,
    );
    let nonneg = greater_equal(g, &zero, None);
    let signed = where_cond(&nonneg, &root, &negative(&root, None), None);
    where_cond(&equal(g, &zero, None), &zero, &signed, None)
}

/// Borrowed PLE wrap tensors (split out of `Qwen4ExpPleWeights` so tests can
/// drive [`ple_wrap`] with synthetic tensors).
pub(crate) struct Qwen4ExpPleWrapWeights<'a> {
    pub key_proj: &'a QuantizedWeight,
    pub value_proj: &'a QuantizedWeight,
    pub norm_key: &'a MlxArray,
    pub norm_query: &'a MlxArray,
    pub norm_conv: &'a MlxArray,
    pub conv1d: &'a MlxArray,
}

impl<'a> From<&'a Qwen4ExpPleWeights> for Qwen4ExpPleWrapWeights<'a> {
    fn from(weights: &'a Qwen4ExpPleWeights) -> Self {
        Self {
            key_proj: &weights.key_proj,
            value_proj: &weights.value_proj,
            norm_key: &weights.norm_key,
            norm_query: &weights.norm_query,
            norm_conv: &weights.norm_conv,
            conv1d: &weights.conv1d,
        }
    }
}

/// The PLE gated wrap over a gathered n-gram embedding: per-stream sigmoid
/// gate (`signed_sqrt(sum(key*query) / sqrt(hidden))` of the grouped key and
/// query norms) times the value projection, plus `silu` of the dilated
/// depthwise conv over the re-normed gated stream. `emb` is
/// `[1, L, hidden]` f32 (the flattened `[L, heads, head_dim]` gather);
/// `hidden` is the packed residual stream `[1, L, hc*hidden]`. Returns the
/// PLE output in the residual dtype and the new conv-input ring (f32): the
/// last `(kernel - 1) * dilation + QWEN4_EXP_PLE_REWIND_WINDOW` re-normed
/// gated-stream rows, whose last `(kernel - 1) * dilation` rows are the conv
/// tail the next forward consumes.
fn ple_wrap(
    hidden: &MlxArray,
    emb: &MlxArray,
    weights: &Qwen4ExpPleWrapWeights,
    cached_conv_ring: Option<&MlxArray>,
    hc_count: usize,
    hidden_size: usize,
    eps: f32,
) -> (MlxArray, MlxArray) {
    let shape = hidden.shape();
    let (batch, seq) = (shape[0], shape[1]);
    let (hc, hidden_i) = (hc_count as i32, hidden_size as i32);
    let key = grouped_rms_norm(
        &qw(emb, weights.key_proj),
        weights.norm_key,
        hc_count,
        hidden_size,
        eps,
    );
    let query = grouped_rms_norm(hidden, weights.norm_query, hc_count, hidden_size, eps);
    let value = astype(&qw(emb, weights.value_proj), MlxDtype::Float32, None);
    // Per-stream scalar gate g = sum(key * query) / sqrt(hidden).
    let key_streams = reshape(&key, &[batch, seq, hc, hidden_i], None);
    let query_streams = reshape(&query, &[batch, seq, hc, hidden_i], None);
    let dot = sum_axis(
        &multiply(&key_streams, &query_streams, None),
        3,
        false,
        None,
    );
    let g = divide(
        &dot,
        &mlx_sys::ops::cached_scalar((hidden_size as f32).sqrt(), MlxDtype::Float32),
        None,
    );
    let gate = sigmoid(&signed_sqrt(&g), None);
    let gated = reshape(
        &multiply(
            &reshape(&gate, &[batch, seq, hc, 1], None),
            &reshape(&value, &[batch, seq, 1, hidden_i], None),
            None,
        ),
        &[batch, seq, hc * hidden_i],
        None,
    );
    // Dilated depthwise conv branch over the re-normed gated stream.
    let normed = grouped_rms_norm(&gated, weights.norm_conv, hc_count, hidden_size, eps);
    let conv_weight = astype(weights.conv1d, MlxDtype::Float32, None);
    let tail = (conv_weight.shape()[1] - 1) * PLE_CONV_DILATION;
    let channels = normed.shape()[2];
    // The conv kernel consumes the last `tail` rows of the cached ring (zeros
    // at sequence start). The new ring keeps an extra rewind window of input
    // rows beyond the tail so `trim_to` can rebuild the exact tail at the
    // rewind point without a full re-forward.
    let ring_base = cached_conv_ring
        .cloned()
        .unwrap_or_else(|| zeros(&[batch, tail, channels], normed.dtype(), None));
    let ring_rows = ring_base.shape()[1];
    assert!(
        ring_rows >= tail && ring_base.shape()[2] == channels,
        "qwen4_exp PLE conv ring must hold at least the (kernel - 1) * dilation tail rows"
    );
    let conv_state = slice(
        &ring_base,
        &[0, ring_rows - tail, 0],
        &[1, ring_rows, channels],
        &[1, 1, 1],
        None,
    );
    let (conv_out, _) = ple_conv1d(&normed, &conv_weight, Some(&conv_state));
    let ring_full = concatenate(&[&ring_base, &normed], 1, None);
    let total = ring_full.shape()[1];
    let keep = (tail + crate::kv_cache::QWEN4_EXP_PLE_REWIND_WINDOW as i32).min(total);
    let new_ring = slice(
        &ring_full,
        &[0, total - keep, 0],
        &[1, total, channels],
        &[1, 1, 1],
        None,
    );
    let out = add(&gated, &mlx_sys::ops::silu(&conv_out, None), None);
    (astype(&out, hidden.dtype(), None), new_ring)
}

/// Output of one PLE forward — the value the trunk adds to all `hc_count`
/// residual streams, plus the state rings to persist on the KV cache.
pub(crate) struct Qwen4ExpPleForwardOutput {
    /// `[1, L, hc*hidden]` in the residual stream dtype.
    pub output: MlxArray,
    /// Conv-input ring `[1, min((kernel - 1) * dilation + rewind window,
    /// (kernel - 1) * dilation + covered), hc*hidden]` f32; the forward
    /// consumes its last `(kernel - 1) * dilation` rows as the conv tail and
    /// `trim_to` rewinds it row-wise after a draft rollback.
    pub conv_ring: MlxArray,
    /// Effective token-id ring: the last `context_len + rewind window` ids
    /// (post pad→EOS substitution, EOS-filled at sequence start). The hash
    /// consumes the last `context_len` entries as its cross-call history.
    pub token_ring: Vec<i64>,
}

/// Run the qwen4_exp PLE floor for one forward — a prefill chunk or a single
/// decode token through the same cached-state path.
///
/// * `hidden`: packed residual stream `[1, L, hc_count * hidden]` (batch is
///   fixed to 1 — the KV cache holds a single PLE state slot).
/// * `input_ids`: raw token ids `[L]` on the host (the hash is pure i64).
/// * `conv_mask`: optional `[L]` real-token flags; padded positions hash as
///   `eos_token_id` (reference behavior when a conv mask exists).
/// * `ple_state`: `(conv_ring, token_ring)` from
///   `MlxKVCache::qwen4_exp_ple_state`; both `None` on the first forward
///   (zero conv tail, all-EOS hash context).
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen4_exp_ple_forward(
    hidden: &MlxArray,
    input_ids: &[u32],
    conv_mask: Option<&[bool]>,
    eos_token_id: u32,
    ngram_size: usize,
    heads_per_ngram: usize,
    hc_count: usize,
    eps: f32,
    weights: &Qwen4ExpPleWeights,
    ple_state: (Option<&MlxArray>, Option<&[i64]>),
) -> Result<Qwen4ExpPleForwardOutput, String> {
    let shape = hidden.shape();
    if shape.len() != 3 || shape[0] != 1 {
        return Err(format!(
            "qwen4_exp PLE hidden must be [1, seq, hc*hidden], got {shape:?}"
        ));
    }
    let seq = shape[1] as usize;
    if seq == 0 || input_ids.len() != seq {
        return Err(format!(
            "qwen4_exp PLE got {} input ids for seq {seq}",
            input_ids.len()
        ));
    }
    let width = shape[2] as usize;
    if hc_count == 0 || !width.is_multiple_of(hc_count) {
        return Err(format!(
            "qwen4_exp PLE packed width {width} is not divisible by hc_count {hc_count}"
        ));
    }
    let hidden_size = width / hc_count;

    let hash = Qwen4ExpPleHash::from_buffers(
        &weights.layer_multipliers,
        &weights.ngram_heads_vocab_sizes,
        &weights.ngram_heads_offsets,
        ngram_size,
        heads_per_ngram,
    )?;
    let context = hash.context_len();
    let table = &weights.table;
    let shard_count = table.shard_count();
    let first_shard = table
        .shard(0)
        .ok_or_else(|| "qwen4_exp PLE n-gram table has no shards".to_string())?;
    let head_dim = first_shard.dim();
    if head_dim <= 0 || hash.head_count() as i64 * head_dim != hidden_size as i64 {
        return Err(format!(
            "qwen4_exp PLE {} heads x dim {head_dim} does not match hidden size {hidden_size}",
            hash.head_count()
        ));
    }
    // The shard-index-by-division scheme needs uniform shard row counts.
    let rows_per_shard = first_shard.rows();
    let shards: Vec<&Qwen4ExpPleTableShard> = (0..shard_count)
        .map(|index| table.shard(index))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "qwen4_exp PLE n-gram table shard index out of range".to_string())?;
    for shard in &shards {
        if shard.rows() != rows_per_shard || shard.dim() != head_dim {
            return Err(
                "qwen4_exp PLE n-gram shards must share uniform row counts and dims".to_string(),
            );
        }
    }

    let eos = i64::from(eos_token_id);
    let tokens = effective_tokens(input_ids, conv_mask, eos)?;
    let mut token_ring: Vec<i64> = match ple_state.1 {
        Some(ring) => {
            if ring.len() < context {
                return Err(format!(
                    "qwen4_exp PLE token ring holds {} ids, context_len is {context}",
                    ring.len()
                ));
            }
            ring.to_vec()
        }
        None => vec![eos; context],
    };
    // The hash context is the ring's tail BEFORE this forward's ids.
    let history = token_ring[token_ring.len() - context..].to_vec();
    let ids = hash.hash_ids(&tokens, &history, eos);

    let gathered = gather_ngram_rows(&shards, &ids, rows_per_shard)?;
    let emb = reshape(&gathered, &[1, seq as i32, hidden_size as i32], None);
    let wrap_weights = Qwen4ExpPleWrapWeights::from(weights);
    let (output, conv_ring) = ple_wrap(
        hidden,
        &emb,
        &wrap_weights,
        ple_state.0,
        hc_count,
        hidden_size,
        eps,
    );

    // Persist the state rings (eval'd so the cache does not pin this graph).
    token_ring.extend_from_slice(&tokens);
    let keep = context + crate::kv_cache::QWEN4_EXP_PLE_REWIND_WINDOW;
    if token_ring.len() > keep {
        token_ring.drain(..token_ring.len() - keep);
    }
    eval(&[&conv_ring]);
    Ok(Qwen4ExpPleForwardOutput {
        output,
        conv_ring,
        token_ring,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use ax_engine_core::NativeTensorQuantization;
    use mlx_sys::{MlxQuantizationMode, quantize};
    use std::cell::Cell;

    /// Planning-doc section B.4 verified hash multipliers (splitmix64, seed
    /// 1234 + 10007*0 for PLE layer index 0).
    const MULTS: [i64; 3] = [23703573157769, 20109073645365, 8052911324071];
    /// Planning-doc section B.2: the first 16 primes after 19,999,999.
    const PRIMES: [i64; 16] = [
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
        20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
    ];
    const EOS: i64 = 248044;

    /// Tiny-table geometry: 4 shards x 64 rows x dim 32, 8-bit g32 affine
    /// (MLX quantize supports group sizes 32/64/128 only).
    const TEST_SHARDS: usize = 4;
    const TEST_ROWS: i64 = 64;
    const TEST_DIM: usize = 32;
    /// Wrap geometry: 4 heads (2 per order) x dim 32 -> hidden 128, 2 streams.
    const TEST_HEADS: usize = 4;
    const TEST_HEADS_PER_NGRAM: usize = 2;
    const TEST_HIDDEN: usize = TEST_HEADS * TEST_DIM;
    const TEST_HC: usize = 2;
    const TEST_WIDTH: usize = TEST_HC * TEST_HIDDEN;
    const TEST_SEQ: usize = 5;
    const EPS: f32 = 1e-6;

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    fn i64_buffer(values: &[i64]) -> MlxArray {
        MlxArray::from_raw_data(
            values.as_ptr() as *const u8,
            std::mem::size_of_val(values),
            &[values.len() as i32],
            MlxDtype::Int64,
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

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what} length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "{what} mismatch at {i}: {a} vs {e} (tol {tol})"
            );
        }
    }

    fn cumulative_offsets(sizes: &[i64]) -> Vec<i64> {
        let mut offsets = Vec::with_capacity(sizes.len());
        let mut acc = 0i64;
        for size in sizes {
            offsets.push(acc);
            acc += size;
        }
        offsets
    }

    /// Reference hash context with the planning-doc constants (8 bigram +
    /// 8 trigram heads over the 16 verified primes).
    fn test_hash() -> Qwen4ExpPleHash {
        Qwen4ExpPleHash {
            mults: MULTS.to_vec(),
            vocab_sizes: PRIMES.to_vec(),
            offsets: cumulative_offsets(&PRIMES),
            heads_per_ngram: 8,
        }
    }

    /// Tiny-table hash context: 2 heads per order, small primes, offsets
    /// placing each order's heads in different shards of the 4x64 table.
    fn tiny_hash() -> Qwen4ExpPleHash {
        Qwen4ExpPleHash {
            mults: MULTS.to_vec(),
            vocab_sizes: vec![13, 17, 19, 23],
            offsets: vec![0, 64, 128, 192],
            heads_per_ngram: TEST_HEADS_PER_NGRAM,
        }
    }

    struct FakeShard {
        qw: QuantizedWeight,
        loads: Cell<usize>,
    }

    impl FakeShard {
        fn from_dense(dense: &MlxArray, group_size: i32) -> Self {
            let parts = quantize(
                dense,
                Some(group_size),
                Some(8),
                MlxQuantizationMode::Affine,
                None,
                None,
            );
            assert!(
                parts.len() >= 3,
                "quantize must return weight/scales/biases"
            );
            eval(&[&parts[0], &parts[1], &parts[2]]);
            let quantization = NativeTensorQuantization {
                mode: "affine".to_string(),
                group_size: group_size as u32,
                bits: 8,
            };
            Self {
                qw: QuantizedWeight::with_quantization(
                    parts[0].clone(),
                    Some(parts[1].clone()),
                    Some(parts[2].clone()),
                    Some(&quantization),
                ),
                loads: Cell::new(0),
            }
        }
    }

    impl NgramShardSource for FakeShard {
        fn shard_is_loaded(&self) -> bool {
            self.loads.get() > 0
        }

        fn shard_quantized(&self) -> Result<&QuantizedWeight, String> {
            self.loads.set(self.loads.get() + 1);
            Ok(&self.qw)
        }
    }

    /// Dense table contents plus the 4 fake shards over them.
    fn build_fake_table(seed: f32, scale: f32) -> (Vec<f32>, Vec<FakeShard>) {
        let rows = TEST_ROWS as usize;
        let dense = scaled_fill(TEST_SHARDS * rows * TEST_DIM, seed, scale);
        let shards = (0..TEST_SHARDS)
            .map(|s| {
                let chunk = &dense[s * rows * TEST_DIM..(s + 1) * rows * TEST_DIM];
                FakeShard::from_dense(
                    &array_f32(chunk, &[TEST_ROWS as i32, TEST_DIM as i32]),
                    TEST_DIM as i32,
                )
            })
            .collect();
        (dense, shards)
    }

    /// Row-major `x [rows, in_dim] @ w [out_dim, in_dim]^T`.
    fn manual_matmul(x: &[f32], rows: usize, w: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
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

    /// CPU reference for `grouped_rms_norm` on row-major `[rows, hc*hidden]`.
    fn manual_grouped_norm(x: &[f32], w: &[f32], hc: usize, hidden: usize, eps: f32) -> Vec<f32> {
        let width = hc * hidden;
        let rows = x.len() / width;
        let mut out = vec![0.0; x.len()];
        for r in 0..rows {
            for g in 0..hc {
                let base = r * width + g * hidden;
                let group = &x[base..base + hidden];
                let mean_sqr: f32 = group.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
                let rstd = (mean_sqr + eps).powf(-0.5);
                for (e, v) in group.iter().enumerate() {
                    out[base + e] = v * rstd * (1.0 + w[g * hidden + e]);
                }
            }
        }
        out
    }

    fn sigmoid_f32(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    /// CPU reference for the dilated causal depthwise conv over
    /// `[state(tail), x]`: `out[c, t] = sum_k w[c, k] * input[c, t + k*d]`.
    fn manual_dilated_conv(
        x: &[f32],
        state: &[f32],
        w: &[f32],
        channels: usize,
        kernel: usize,
        dilation: usize,
        seq: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let tail = (kernel - 1) * dilation;
        assert_eq!(state.len(), tail * channels);
        let mut input = state.to_vec();
        input.extend_from_slice(x);
        let mut out = vec![0.0; seq * channels];
        for t in 0..seq {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for k in 0..kernel {
                    acc += w[c * kernel + k] * input[(t + k * dilation) * channels + c];
                }
                out[t * channels + c] = acc;
            }
        }
        let total = tail + seq;
        let new_state = input[(total - tail) * channels..].to_vec();
        (out, new_state)
    }

    /// CPU reference for the whole PLE wrap on row-major `[seq, *]` data.
    #[allow(clippy::too_many_arguments)]
    fn reference_wrap(
        emb: &[f32],
        hidden: &[f32],
        seq: usize,
        hc: usize,
        hidden_size: usize,
        key_w: &[f32],
        value_w: &[f32],
        norm_k: &[f32],
        norm_q: &[f32],
        norm_c: &[f32],
        conv_w: &[f32],
        conv_state: &[f32],
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let width = hc * hidden_size;
        let key = manual_grouped_norm(
            &manual_matmul(emb, seq, key_w, width, hidden_size),
            norm_k,
            hc,
            hidden_size,
            eps,
        );
        let query = manual_grouped_norm(hidden, norm_q, hc, hidden_size, eps);
        let value = manual_matmul(emb, seq, value_w, hidden_size, hidden_size);
        let mut gated = vec![0.0; seq * width];
        for r in 0..seq {
            for g in 0..hc {
                let base = r * width + g * hidden_size;
                let dot: f32 = (0..hidden_size)
                    .map(|e| key[base + e] * query[base + e])
                    .sum();
                let raw = dot / (hidden_size as f32).sqrt();
                // Rust f32::signum(0.0) is 1.0; the reference's sign(0) is 0.
                let gate = if raw == 0.0 {
                    0.0
                } else {
                    raw.signum() * raw.abs().max(PLE_SIGNED_SQRT_EPS).sqrt()
                };
                let sig = sigmoid_f32(gate);
                for e in 0..hidden_size {
                    gated[base + e] = sig * value[r * hidden_size + e];
                }
            }
        }
        let normed = manual_grouped_norm(&gated, norm_c, hc, hidden_size, eps);
        let kernel = conv_w.len() / width;
        let (conv_out, new_state) = manual_dilated_conv(
            &normed,
            conv_state,
            conv_w,
            width,
            kernel,
            PLE_CONV_DILATION as usize,
            seq,
        );
        let mut out = vec![0.0; seq * width];
        for (i, o) in out.iter_mut().enumerate() {
            *o = gated[i] + conv_out[i] * sigmoid_f32(conv_out[i]);
        }
        (out, new_state)
    }

    /// Synthetic wrap tensors plus their raw f32 data for the CPU reference.
    struct WrapRig {
        weights: Qwen4ExpPleWrapWeights<'static>,
        // Raw data (row-major) behind the boxed arrays.
        key_w: Vec<f32>,
        value_w: Vec<f32>,
        norm_k: Vec<f32>,
        norm_q: Vec<f32>,
        norm_c: Vec<f32>,
        conv_w: Vec<f32>,
    }

    fn build_wrap_rig() -> WrapRig {
        let key_w = scaled_fill(TEST_WIDTH * TEST_HIDDEN, 0.13, 0.15);
        let value_w = scaled_fill(TEST_HIDDEN * TEST_HIDDEN, 0.17, 0.15);
        let norm_k = scaled_fill(TEST_WIDTH, 0.19, 0.4);
        let norm_q = scaled_fill(TEST_WIDTH, 0.23, 0.4);
        let norm_c = scaled_fill(TEST_WIDTH, 0.29, 0.4);
        let conv_w = scaled_fill(TEST_WIDTH * 4, 0.31, 0.3);
        // Leak the backing arrays: the rig is built once per test and the
        // 'static borrow keeps the wrap-weights struct simple.
        let key_proj = Box::leak(Box::new(QuantizedWeight::new(
            array_f32(&key_w, &[TEST_WIDTH as i32, TEST_HIDDEN as i32]),
            None,
            None,
        )));
        let value_proj = Box::leak(Box::new(QuantizedWeight::new(
            array_f32(&value_w, &[TEST_HIDDEN as i32, TEST_HIDDEN as i32]),
            None,
            None,
        )));
        let norm_k_arr = Box::leak(Box::new(array_f32(&norm_k, &[TEST_WIDTH as i32])));
        let norm_q_arr = Box::leak(Box::new(array_f32(&norm_q, &[TEST_WIDTH as i32])));
        let norm_c_arr = Box::leak(Box::new(array_f32(&norm_c, &[TEST_WIDTH as i32])));
        let conv_arr = Box::leak(Box::new(array_f32(&conv_w, &[TEST_WIDTH as i32, 4, 1])));
        WrapRig {
            weights: Qwen4ExpPleWrapWeights {
                key_proj,
                value_proj,
                norm_key: norm_k_arr,
                norm_query: norm_q_arr,
                norm_conv: norm_c_arr,
                conv1d: conv_arr,
            },
            key_w,
            value_w,
            norm_k,
            norm_q,
            norm_c,
            conv_w,
        }
    }

    /// Gathered embedding ids for the tiny table over `TEST_SEQ` tokens.
    fn tiny_ids() -> (Qwen4ExpPleHash, Vec<i64>, Vec<i64>) {
        let hash = tiny_hash();
        let tokens = vec![7, 42, 9, 13, 77];
        let ids = hash.hash_ids(&tokens, &[EOS, EOS], EOS);
        (hash, tokens, ids)
    }

    #[test]
    fn i64_buffers_round_trip_through_u32_view() {
        let mut values: Vec<i64> = MULTS.to_vec();
        values.extend_from_slice(&PRIMES);
        values.extend_from_slice(&[0, 1, -1, i64::MAX, i64::MIN]);
        let back = read_i64_buffer(&i64_buffer(&values)).unwrap();
        assert_eq!(back, values);
    }

    #[test]
    fn hash_from_buffers_validates_counts_and_signs() {
        let offsets = cumulative_offsets(&PRIMES);
        let ok = Qwen4ExpPleHash::from_buffers(
            &i64_buffer(&MULTS),
            &i64_buffer(&PRIMES),
            &i64_buffer(&offsets),
            3,
            8,
        )
        .unwrap();
        assert_eq!(ok.mults, MULTS);
        assert_eq!(ok.vocab_sizes, PRIMES);
        assert_eq!(ok.offsets, offsets);
        assert_eq!(ok.context_len(), 2);
        assert_eq!(ok.head_count(), 16);

        let short_mults = Qwen4ExpPleHash::from_buffers(
            &i64_buffer(&MULTS[..2]),
            &i64_buffer(&PRIMES),
            &i64_buffer(&offsets),
            3,
            8,
        );
        assert!(
            short_mults.is_err(),
            "multiplier count must match ngram_size"
        );

        let short_heads = Qwen4ExpPleHash::from_buffers(
            &i64_buffer(&MULTS),
            &i64_buffer(&PRIMES[..15]),
            &i64_buffer(&offsets),
            3,
            8,
        );
        assert!(
            short_heads.is_err(),
            "head buffers must match (ngram_size-1)*heads_per_ngram"
        );

        let bad_sign = Qwen4ExpPleHash::from_buffers(
            &i64_buffer(&[-23703573157769, 20109073645365, 8052911324071]),
            &i64_buffer(&PRIMES),
            &i64_buffer(&offsets),
            3,
            8,
        );
        assert!(bad_sign.is_err(), "multipliers must be positive");

        let bad_geometry = Qwen4ExpPleHash::from_buffers(
            &i64_buffer(&MULTS),
            &i64_buffer(&PRIMES),
            &i64_buffer(&offsets),
            1,
            8,
        );
        assert!(bad_geometry.is_err(), "ngram_size must be at least 2");
    }

    #[test]
    fn hash_ids_match_planning_doc_constants() {
        // Hand-computed with the doc literals (not the test constants) so a
        // transcription error in MULTS/PRIMES/EOS fails here.
        let (m0, m1, m2) = (23703573157769i64, 20109073645365i64, 8052911324071i64);
        let e = 248044i64;
        // full = [e, e, 5, 9]: position of 5 sees EOS-only context; position
        // of 9 sees (5, EOS) — the trigram context stops at the history EOS.
        let big0 = 5i64.wrapping_mul(m0) ^ e.wrapping_mul(m1);
        let tri0 = big0 ^ e.wrapping_mul(m2);
        let big1 = 9i64.wrapping_mul(m0) ^ 5i64.wrapping_mul(m1);
        let tri1 = big1 ^ e.wrapping_mul(m2);

        let hash = test_hash();
        let ids = hash.hash_ids(&[5, 9], &[EOS, EOS], EOS);
        let again = hash.hash_ids(&[5, 9], &[EOS, EOS], EOS);
        assert_eq!(ids, again, "hash must be deterministic");
        assert_eq!(ids.len(), 2 * 16);
        let offsets = cumulative_offsets(&PRIMES);
        for h in 0..8 {
            assert_eq!(
                ids[h],
                big0 % PRIMES[h] + offsets[h],
                "bigram head {h} pos 0"
            );
            assert_eq!(
                ids[8 + h],
                tri0 % PRIMES[8 + h] + offsets[8 + h],
                "trigram head {h} pos 0"
            );
            assert_eq!(
                ids[16 + h],
                big1 % PRIMES[h] + offsets[h],
                "bigram head {h} pos 1"
            );
            assert_eq!(
                ids[24 + h],
                tri1 % PRIMES[8 + h] + offsets[8 + h],
                "trigram head {h} pos 1"
            );
        }
    }

    #[test]
    fn hash_ids_reset_trigram_at_eos_boundary() {
        let (m0, m1, m2) = (23703573157769i64, 20109073645365i64, 8052911324071i64);
        let e = 248044i64;
        // full = [e, e, 5, EOS, 9].
        // Position of EOS: bigram (5, EOS) survives; trigram is (EOS, EOS)
        // because the source position 1 sits at the previous EOS.
        let big_at_eos = e.wrapping_mul(m0) ^ 5i64.wrapping_mul(m1);
        let tri_at_eos = big_at_eos ^ e.wrapping_mul(m2);
        // Position after EOS: full reset — the trigram source 5 is at/before
        // the EOS and must become EOS, not 5.
        let big9 = 9i64.wrapping_mul(m0) ^ e.wrapping_mul(m1);
        let tri9 = big9 ^ e.wrapping_mul(m2);

        let hash = test_hash();
        let ids = hash.hash_ids(&[5, EOS, 9], &[EOS, EOS], EOS);
        let offsets = cumulative_offsets(&PRIMES);
        for h in 0..8 {
            assert_eq!(
                ids[16 + h],
                big_at_eos % PRIMES[h] + offsets[h],
                "bigram head {h} at EOS"
            );
            assert_eq!(
                ids[24 + h],
                tri_at_eos % PRIMES[8 + h] + offsets[8 + h],
                "trigram head {h} at EOS"
            );
            assert_eq!(
                ids[32 + h],
                big9 % PRIMES[h] + offsets[h],
                "bigram head {h} after EOS"
            );
            assert_eq!(
                ids[40 + h],
                tri9 % PRIMES[8 + h] + offsets[8 + h],
                "trigram head {h} after EOS"
            );
        }
        // The reset is the point: a trigram leaking the pre-EOS token (5*m2
        // instead of e*m2) cannot reproduce every head's id at once.
        let leaked: Vec<i64> = (0..8)
            .map(|h| {
                (9i64.wrapping_mul(m0) ^ e.wrapping_mul(m1) ^ 5i64.wrapping_mul(m2)) % PRIMES[8 + h]
                    + offsets[8 + h]
            })
            .collect();
        assert_ne!(
            ids[40..48],
            leaked[..],
            "trigram after EOS must not leak the pre-EOS token"
        );
    }

    #[test]
    fn hash_ids_carry_across_chunk_boundary() {
        let tokens = [11i64, 22, 33, 44, 55];
        let hash = test_hash();
        let one_shot = hash.hash_ids(&tokens, &[EOS, EOS], EOS);
        // Chunked prefill: A = [11, 22], then B = [33, 44, 55] carrying the
        // last two raw tokens of A as history.
        let chunk_a = hash.hash_ids(&tokens[..2], &[EOS, EOS], EOS);
        let chunk_b = hash.hash_ids(&tokens[2..], &[11, 22], EOS);
        assert_eq!(one_shot[..chunk_a.len()], chunk_a[..]);
        assert_eq!(one_shot[chunk_a.len()..], chunk_b[..]);
        // Decode-style carry: history [44, 55] then one token 66 matches the
        // one-shot hash of the extended sequence at its last position.
        let extended = [tokens.as_slice(), &[66]].concat();
        let one_shot_ext = hash.hash_ids(&extended, &[EOS, EOS], EOS);
        let decode = hash.hash_ids(&[66], &[44, 55], EOS);
        assert_eq!(one_shot_ext[one_shot_ext.len() - 16..], decode[..]);
    }

    #[test]
    fn effective_tokens_replace_padding_with_eos() {
        let ids = [10u32, 20, 30];
        let masked = effective_tokens(&ids, Some(&[true, false, true]), EOS).unwrap();
        assert_eq!(masked, vec![10, EOS, 30]);
        let raw = effective_tokens(&ids, None, EOS).unwrap();
        assert_eq!(raw, vec![10, 20, 30]);
        let bad = effective_tokens(&ids, Some(&[true, false]), EOS);
        assert!(bad.is_err(), "mask length must match input ids");
    }

    #[test]
    fn group_ids_by_shard_buckets_and_rejects_out_of_range() {
        let groups = group_ids_by_shard(&[0, 63, 64, 127, 130], TEST_ROWS, TEST_SHARDS).unwrap();
        assert_eq!(groups.len(), 3, "shard 3 is untouched and must not appear");
        assert_eq!(groups[0].shard, 0);
        assert_eq!(groups[0].locals, vec![0, 63]);
        assert_eq!(groups[0].positions, vec![0, 1]);
        assert_eq!(groups[1].shard, 1);
        assert_eq!(groups[1].locals, vec![0, 63]);
        assert_eq!(groups[1].positions, vec![2, 3]);
        assert_eq!(groups[2].shard, 2);
        assert_eq!(groups[2].locals, vec![2]);
        assert_eq!(groups[2].positions, vec![4]);

        let total = TEST_ROWS * TEST_SHARDS as i64;
        assert!(group_ids_by_shard(&[total], TEST_ROWS, TEST_SHARDS).is_err());
        assert!(group_ids_by_shard(&[-1], TEST_ROWS, TEST_SHARDS).is_err());
        assert!(group_ids_by_shard(&[0], 0, TEST_SHARDS).is_err());
        assert!(group_ids_by_shard(&[0], TEST_ROWS, 0).is_err());
    }

    #[test]
    fn gather_ngram_rows_loads_only_touched_shards() {
        let (dense, shards) = build_fake_table(0.31, 0.4);
        // Ids in shards 0 and 3 only.
        let ids = [5i64, 60, 195, 250];
        let gathered = gather_ngram_rows(&shards, &ids, TEST_ROWS).unwrap();
        assert_eq!(gathered.shape(), vec![ids.len() as i32, TEST_DIM as i32]);
        assert!(shards[0].shard_is_loaded(), "shard 0 must be loaded");
        assert!(!shards[1].shard_is_loaded(), "shard 1 must stay unloaded");
        assert!(!shards[2].shard_is_loaded(), "shard 2 must stay unloaded");
        assert!(shards[3].shard_is_loaded(), "shard 3 must be loaded");
        assert_eq!(shards[0].loads.get(), 1, "each touched shard loads once");

        // 8-bit affine over [-0.4, 0.4]: half-step error ~ 0.8/510.
        let rows = eval_f32(&gathered);
        let mut expected = Vec::new();
        for id in ids {
            expected
                .extend_from_slice(&dense[id as usize * TEST_DIM..(id as usize + 1) * TEST_DIM]);
        }
        assert_close(&rows, &expected, 4e-3, "gathered rows vs dense");
    }

    #[test]
    fn ple_conv1d_matches_cpu_dilated_conv_and_carries_state() {
        let channels = 8usize;
        let kernel = 4usize;
        let w_data = scaled_fill(channels * kernel, 0.17, 0.5);
        let x_data = fill(6 * channels, 0.23);
        let w = array_f32(&w_data, &[channels as i32, kernel as i32, 1]);
        let x = array_f32(&x_data, &[1, 6, channels as i32]);
        let zero_state = vec![0.0; 9 * channels];

        let (out, state) = ple_conv1d(&x, &w, None);
        assert_eq!(out.shape(), vec![1, 6, channels as i32]);
        assert_eq!(state.shape(), vec![1, 9, channels as i32]);
        let (ref_out, ref_state) = manual_dilated_conv(
            &x_data,
            &zero_state,
            &w_data,
            channels,
            kernel,
            PLE_CONV_DILATION as usize,
            6,
        );
        assert_close(&eval_f32(&out), &ref_out, 1e-6, "dilated conv one-shot");
        assert_close(&eval_f32(&state), &ref_state, 1e-6, "conv tail one-shot");

        // Chunked: [0..4) then [4..6) with the carried tail must equal one-shot.
        let x_a = slice(&x, &[0, 0, 0], &[1, 4, channels as i32], &[1, 1, 1], None);
        let x_b = slice(&x, &[0, 4, 0], &[1, 6, channels as i32], &[1, 1, 1], None);
        let (out_a, state_a) = ple_conv1d(&x_a, &w, None);
        let (out_b, state_b) = ple_conv1d(&x_b, &w, Some(&state_a));
        assert_close(
            &eval_f32(&out_a),
            &ref_out[..4 * channels],
            1e-6,
            "conv chunk A",
        );
        assert_close(
            &eval_f32(&out_b),
            &ref_out[4 * channels..],
            1e-6,
            "conv chunk B",
        );
        assert_close(
            &eval_f32(&state_b),
            &ref_state,
            1e-6,
            "conv tail after chunk B",
        );
    }

    #[test]
    fn signed_sqrt_matches_reference_formula() {
        let values = [0.0f32, 1e-8, -1e-8, 0.25, -4.0, 1e-6, -1e-6];
        let g = array_f32(&values, &[values.len() as i32]);
        let out = eval_f32(&signed_sqrt(&g));
        for (v, o) in values.iter().zip(out.iter()) {
            // Rust f32::signum(0.0) is 1.0; the reference's sign(0) is 0.
            let expected = if *v == 0.0 {
                0.0
            } else {
                v.signum() * v.abs().max(PLE_SIGNED_SQRT_EPS).sqrt()
            };
            assert!(
                (o - expected).abs() < 1e-6,
                "signed_sqrt({v}): {o} vs {expected}"
            );
        }
    }

    #[test]
    fn ple_wrap_matches_cpu_reference_end_to_end() {
        let (dense, shards) = build_fake_table(0.41, 0.05);
        let (_, _, ids) = tiny_ids();
        let gathered = gather_ngram_rows(&shards, &ids, TEST_ROWS).unwrap();
        for shard in &shards {
            assert!(
                shard.shard_is_loaded(),
                "tiny-table offsets touch every shard"
            );
        }
        let emb = reshape(&gathered, &[1, TEST_SEQ as i32, TEST_HIDDEN as i32], None);
        let rig = build_wrap_rig();
        let hidden_data = fill(TEST_SEQ * TEST_WIDTH, 0.37);
        let hidden = array_f32(&hidden_data, &[1, TEST_SEQ as i32, TEST_WIDTH as i32]);

        let (out, ring) = ple_wrap(&hidden, &emb, &rig.weights, None, TEST_HC, TEST_HIDDEN, EPS);
        assert_eq!(out.shape(), vec![1, TEST_SEQ as i32, TEST_WIDTH as i32]);
        assert_eq!(out.dtype(), MlxDtype::Float32);
        // The ring holds the 9-row conv tail plus this chunk's input rows
        // (all inside the rewind window).
        assert_eq!(
            ring.shape(),
            vec![1, 9 + TEST_SEQ as i32, TEST_WIDTH as i32]
        );

        // CPU reference over the dense (pre-quantization) table rows.
        let mut ref_emb = Vec::with_capacity(TEST_SEQ * TEST_HIDDEN);
        for id in &ids {
            ref_emb
                .extend_from_slice(&dense[*id as usize * TEST_DIM..(*id as usize + 1) * TEST_DIM]);
        }
        let zero_state = vec![0.0; 9 * TEST_WIDTH];
        let (ref_out, ref_state) = reference_wrap(
            &ref_emb,
            &hidden_data,
            TEST_SEQ,
            TEST_HC,
            TEST_HIDDEN,
            &rig.key_w,
            &rig.value_w,
            &rig.norm_k,
            &rig.norm_q,
            &rig.norm_c,
            &rig.conv_w,
            &zero_state,
            EPS,
        );
        // Tolerance is driven by the 8-bit table quantization error.
        assert_close(
            &eval_f32(&out),
            &ref_out,
            2e-2,
            "wrap output vs CPU reference",
        );
        // The ring's last 9 rows are the conv tail the next forward consumes.
        let ring_host = eval_f32(&ring);
        assert_close(
            &ring_host[ring_host.len() - 9 * TEST_WIDTH..],
            &ref_state,
            2e-2,
            "wrap conv tail vs CPU reference",
        );

        // bf16 residual streams round-trip through the wrap dtype boundary.
        let hidden_bf16 = astype(&hidden, MlxDtype::Bfloat16, None);
        let (out_bf16, _) = ple_wrap(
            &hidden_bf16,
            &emb,
            &rig.weights,
            None,
            TEST_HC,
            TEST_HIDDEN,
            EPS,
        );
        assert_eq!(out_bf16.dtype(), MlxDtype::Bfloat16);
        assert_eq!(
            out_bf16.shape(),
            vec![1, TEST_SEQ as i32, TEST_WIDTH as i32]
        );
    }

    #[test]
    fn ple_wrap_chunked_state_matches_one_shot() {
        let (dense, shards) = build_fake_table(0.41, 0.05);
        let (_, _, ids) = tiny_ids();
        let gathered = gather_ngram_rows(&shards, &ids, TEST_ROWS).unwrap();
        let emb = reshape(&gathered, &[1, TEST_SEQ as i32, TEST_HIDDEN as i32], None);
        let rig = build_wrap_rig();
        let hidden_data = fill(TEST_SEQ * TEST_WIDTH, 0.37);
        let hidden = array_f32(&hidden_data, &[1, TEST_SEQ as i32, TEST_WIDTH as i32]);

        let (out_full, state_full) =
            ple_wrap(&hidden, &emb, &rig.weights, None, TEST_HC, TEST_HIDDEN, EPS);
        let split = 3i32;
        let hidden_a = slice(
            &hidden,
            &[0, 0, 0],
            &[1, split, TEST_WIDTH as i32],
            &[1, 1, 1],
            None,
        );
        let hidden_b = slice(
            &hidden,
            &[0, split, 0],
            &[1, TEST_SEQ as i32, TEST_WIDTH as i32],
            &[1, 1, 1],
            None,
        );
        let emb_a = slice(
            &emb,
            &[0, 0, 0],
            &[1, split, TEST_HIDDEN as i32],
            &[1, 1, 1],
            None,
        );
        let emb_b = slice(
            &emb,
            &[0, split, 0],
            &[1, TEST_SEQ as i32, TEST_HIDDEN as i32],
            &[1, 1, 1],
            None,
        );
        let (out_a, state_a) = ple_wrap(
            &hidden_a,
            &emb_a,
            &rig.weights,
            None,
            TEST_HC,
            TEST_HIDDEN,
            EPS,
        );
        let (out_b, state_b) = ple_wrap(
            &hidden_b,
            &emb_b,
            &rig.weights,
            Some(&state_a),
            TEST_HC,
            TEST_HIDDEN,
            EPS,
        );

        // Chunked == one-shot (production-vs-production, tight tolerance).
        let full = eval_f32(&out_full);
        assert_close(
            &eval_f32(&out_a),
            &full[..split as usize * TEST_WIDTH],
            1e-5,
            "chunk A vs one-shot",
        );
        assert_close(
            &eval_f32(&out_b),
            &full[split as usize * TEST_WIDTH..],
            1e-5,
            "chunk B vs one-shot",
        );
        assert_close(
            &eval_f32(&state_b),
            &eval_f32(&state_full),
            1e-5,
            "final tails chunked vs one-shot",
        );

        // And the chunk-B output still matches the CPU reference with the
        // carried state (covers the decode-style continuation path).
        let mut ref_emb = Vec::with_capacity(TEST_SEQ * TEST_HIDDEN);
        for id in &ids {
            ref_emb
                .extend_from_slice(&dense[*id as usize * TEST_DIM..(*id as usize + 1) * TEST_DIM]);
        }
        let zero_state = vec![0.0; 9 * TEST_WIDTH];
        let (_, ref_state_a) = reference_wrap(
            &ref_emb[..split as usize * TEST_HIDDEN],
            &hidden_data[..split as usize * TEST_WIDTH],
            split as usize,
            TEST_HC,
            TEST_HIDDEN,
            &rig.key_w,
            &rig.value_w,
            &rig.norm_k,
            &rig.norm_q,
            &rig.norm_c,
            &rig.conv_w,
            &zero_state,
            EPS,
        );
        let (ref_out_b, _) = reference_wrap(
            &ref_emb[split as usize * TEST_HIDDEN..],
            &hidden_data[split as usize * TEST_WIDTH..],
            TEST_SEQ - split as usize,
            TEST_HC,
            TEST_HIDDEN,
            &rig.key_w,
            &rig.value_w,
            &rig.norm_k,
            &rig.norm_q,
            &rig.norm_c,
            &rig.conv_w,
            &ref_state_a,
            EPS,
        );
        assert_close(
            &eval_f32(&out_b),
            &ref_out_b,
            2e-2,
            "chunk B vs CPU reference",
        );
    }
}
