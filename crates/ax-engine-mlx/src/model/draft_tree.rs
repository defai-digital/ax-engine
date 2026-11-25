// SPDX-License-Identifier: Apache-2.0

//! One-forward verify of a DFlash2 draft tree on the dense Qwen gated-delta
//! stack, and the commit of its accepted path.
//!
//! The rows of the forward are the nodes of a [`TreeLayout`]: a row sees the
//! cached prefix, itself and its ancestors, so every root path is verified as
//! the chain it spells. See `docs/DFLASH2-TREE-VERIFY.md`.

use mlx_sys::{MlxArray, MlxDtype, add, astype, concatenate, reshape, rms_norm, slice, take};

use super::shared::replay_linear_attention_mtp_prefix;
use super::{
    ModelConfig, embed_tokens_arr, lm_head_verify_window_projection, qwen_visual_rope_offset,
};
use crate::dflash2_tree::TreeLayout;
use crate::kv_cache::MlxKVCache;
use crate::weights::ModelWeights;

/// One target forward over a DFlash2 draft tree.
pub(crate) struct TreeVerifyForward {
    /// `[rows, vocab]`, in the head's own dtype.
    pub logits: MlxArray,
    /// `[1, rows, hidden]` after the final norm.
    pub post_norm: MlxArray,
    /// Per full-attention layer: the rows' new `(layer, k, v)`,
    /// `[1, heads, rows, head_dim]`, as appended to the cache in row order.
    pub attention_rows: Vec<(usize, MlxArray, MlxArray)>,
    /// Per gated-delta layer: the rows' `(layer, qkv, a, b)` projections.
    pub linear_rows: Vec<(usize, MlxArray, MlxArray, MlxArray)>,
}

/// True when [`forward_tree_verify`] can run this model: a dense Qwen
/// gated-delta stack. A routed-expert layer keeps the chain verify, because
/// the tree's layer tail is the dense FFN.
pub(crate) fn model_takes_tree(cfg: &ModelConfig, weights: &ModelWeights) -> bool {
    crate::fastpath::qwen_linear_throughput_family(&cfg.model_family)
        && cfg.linear_attention.is_some()
        && cfg.deepseek_v4.is_none()
        && cfg.hidden_size_per_layer_input == 0
        && cfg.hidden_states_scale.is_none()
        && !weights
            .layers
            .iter()
            .any(|layer| layer.router_proj.is_some() || layer.expert_stream.is_some())
}

/// Verifies every row of a draft tree in one forward.
///
/// Row `i` sees the cached prefix, itself and its ancestors, at position
/// `token_offset + depth[i]`. Full-attention layers append all rows' K/V to
/// `cache` in row order (the caller advances it and later keeps the accepted
/// path); gated-delta layers read their state and leave it for the caller's
/// replay. `None` when the model is not a dense Qwen gated-delta stack or a
/// layer declines; the cache may then hold appended rows the caller must not
/// advance over.
pub(crate) fn forward_tree_verify(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    tree: &TreeLayout,
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> Option<TreeVerifyForward> {
    if !model_takes_tree(cfg, weights)
        || qwen_visual_rope_offset(weights, cache, token_offset) != token_offset
    {
        return None;
    }
    let rows = tree.rows;
    let prefix = i32::try_from(token_offset).ok()?;
    let mut hidden = embed_tokens_arr(&tree.ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(
        &hidden,
        crate::fastpath::activation_dtype_for_family(&cfg.model_family),
        None,
    );
    hidden = super::shared::utils::maybe_weightless_embed_norm(cfg, hidden);

    // Each chunk's mask: the whole prefix, then the tree's own visibility.
    let chunk_rows = super::families::standard::TREE_ATTENTION_CHUNK_ROWS;
    let mut masks = Vec::new();
    let mut start = 0;
    while start < rows {
        let end = (start + chunk_rows).min(rows);
        let visible = slice(&tree.visible, &[start, 0], &[end, rows], &[1, 1], None);
        masks.push(if prefix > 0 {
            let open = vec![1u8; ((end - start) * prefix) as usize];
            let open = MlxArray::from_raw_data(
                open.as_ptr(),
                open.len(),
                &[end - start, prefix],
                MlxDtype::Bool,
            );
            concatenate(&[&open, &visible], 1, None)
        } else {
            visible
        });
        start = end;
    }
    let offset = MlxArray::from_raw_data(
        (&prefix as *const i32).cast::<u8>(),
        std::mem::size_of::<i32>(),
        &[1],
        MlxDtype::Int32,
    );
    let positions = add(&tree.depth, &offset, None);

    let layer_count = weights.layers.len();
    let submit_interval = crate::fastpath::verify_submit_interval_for_build(
        rows as usize,
        layer_count,
        crate::fastpath::mtp_verify_submit_layer_interval(),
        &cfg.model_family,
    );
    let mut attention_rows = Vec::new();
    let mut linear_rows = Vec::new();
    for (li, layer_w) in weights.layers.iter().enumerate() {
        if cfg.is_linear_attention_layer(li) {
            let (output, qkv, a, b) = super::families::qwen3_linear::layer_forward_tree_verify(
                cfg,
                layer_w,
                &hidden,
                cache,
                li,
                &tree.depth,
                &tree.tail_src,
            )?;
            linear_rows.push((li, qkv, a, b));
            hidden = output;
        } else {
            let (output, k, v) = super::families::standard::layer_forward_tree_verify(
                cfg, layer_w, &hidden, cache, li, &positions, &masks,
            )?;
            attention_rows.push((li, k, v));
            hidden = output;
        }
        super::layer_capture::observe(li, &hidden);
        let submitted =
            submit_interval > 0 && li + 1 < layer_count && (li + 1) % submit_interval == 0;
        if (submitted || crate::fastpath::pipeline_hint_should_fire(li, layer_count))
            && mlx_sys::try_async_eval(&[&hidden]).is_err()
        {
            // The caller drops the cache clone these rows were appended to.
            return None;
        }
    }

    let post_norm = rms_norm(&hidden, Some(weights.final_norm()), cfg.rms_norm_eps, None);
    let logits = lm_head_verify_window_projection(
        &post_norm,
        &weights.lm_head,
        &cfg.model_family,
        rows,
        cfg.hidden_size as i32,
    );
    let logits = reshape(&logits, &[rows, cfg.vocab_size as i32], None);
    Some(TreeVerifyForward {
        logits,
        post_norm,
        attention_rows,
        linear_rows,
    })
}

/// Keeps the accepted path of a tree verify in `cache`.
///
/// `cache` is the verify clone, advanced over all tree rows; `source` is the
/// cache it was cloned from, whose gated-delta state is the path's start.
/// `path` lists the kept rows from the anchor down (`path[0] == 0`). The
/// leading rows that sit where the path wants them stay; the K/V of the rest
/// is rewritten from the forward's captured rows. Each gated-delta layer
/// replays the path's projections from `source`'s state. `false` leaves
/// `cache` unusable: the caller drops it.
pub(crate) fn commit_tree_path(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    source: &MlxKVCache,
    cache: &mut MlxKVCache,
    forward: &TreeVerifyForward,
    token_offset: usize,
    path: &[usize],
) -> bool {
    if path.first() != Some(&0) {
        return false;
    }
    let in_place = path
        .iter()
        .enumerate()
        .take_while(|(slot, row)| slot == *row)
        .count();
    let moved = path.len() - in_place;
    let row_ids = |rows: &[usize]| {
        let ids: Vec<u32> = rows.iter().map(|row| *row as u32).collect();
        MlxArray::from_raw_data(
            ids.as_ptr().cast::<u8>(),
            std::mem::size_of_val(ids.as_slice()),
            &[ids.len() as i32],
            MlxDtype::Uint32,
        )
    };
    if !cache.trim_to(token_offset + in_place) {
        return false;
    }
    if moved > 0 {
        let tail = row_ids(&path[in_place..]);
        for (layer, k, v) in &forward.attention_rows {
            let _ = cache.append(*layer, take(k, &tail, 2, None), take(v, &tail, 2, None));
        }
        cache.advance(moved);
        if cache.hard_cap_exhausted() {
            return false;
        }
    }
    let all = (moved > 0).then(|| row_ids(path));
    for (layer, qkv, a, b) in &forward.linear_rows {
        let gather = |rows: &MlxArray| match all.as_ref() {
            Some(ids) => take(rows, ids, 1, None),
            None => rows.clone(),
        };
        cache.set_linear_mtp_projection_stash(*layer, gather(qkv), gather(a), gather(b));
        let Some(layer_weights) = weights.layers.get(*layer) else {
            return false;
        };
        if !replay_linear_attention_mtp_prefix(
            cfg,
            layer_weights,
            source,
            cache,
            *layer,
            path.len(),
        ) {
            return false;
        }
    }
    true
}
