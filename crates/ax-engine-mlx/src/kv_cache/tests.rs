use super::*;
use mlx_sys::astype;

fn contiguous_layer(cache: &MlxKVCache, layer: usize) -> &LayerKV {
    cache.layers[layer]
        .as_ref()
        .and_then(FaLayerStorage::as_contiguous)
        .expect("contiguous FA layer")
}

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
    assert!(cache.trim_to(4));

    assert_eq!(cache.seq_len, 4);
    let (conv_state, recurrent_state) = cache.linear_state(0);
    assert!(
        conv_state.is_some() && recurrent_state.is_some(),
        "linear recurrent state is not rolled back by seq_len trim"
    );
}

#[test]
fn trim_to_does_not_extend_logical_sequence() {
    let mut cache = MlxKVCache::new(1);
    cache.seq_len = 8;

    assert!(!cache.trim_to(12));

    assert_eq!(
        cache.seq_len, 8,
        "invalid rollback points must not expose unwritten KV slots"
    );
}

#[test]
fn clone_preserves_linear_state_for_draft_branch() {
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

#[test]
fn linear_prefix_checkpoint_restores_complete_transient_state() {
    let shaped = |value: f32, shape: &[i32]| {
        let count = shape.iter().map(|dim| *dim as usize).product();
        mlx_sys::reshape(&MlxArray::from_f32_slice(&vec![value; count]), shape, None)
    };
    let mut cache = MlxKVCache::new(1);
    cache.set_linear_state(0, shaped(1.0, &[1, 2, 3]), shaped(2.0, &[1, 2, 2, 2]));
    cache.begin_linear_prefix_capture(1);
    cache.set_linear_prefix_checkpoint(0, shaped(3.0, &[1, 2, 3]), shaped(4.0, &[1, 2, 2, 2]));
    cache.set_linear_state(0, shaped(5.0, &[1, 2, 3]), shaped(6.0, &[1, 2, 2, 2]));

    assert!(cache.linear_prefix_refs(0).is_some());
    assert!(cache.linear_prefix_refs(1).is_none());
    assert!(cache.restore_linear_prefix_checkpoint());
    assert_eq!(cache.linear_prefix_capture_after(), None);
    assert!(
        cache.linear_prefix_refs(0).is_none(),
        "restore consumes the transient prefix refs"
    );
    let (conv, recurrent) = cache.linear_state(0);
    let conv = conv.expect("restored conv state");
    let recurrent = recurrent.expect("restored recurrent state");
    eval(&[conv, recurrent]);
    assert!(conv.data_f32().iter().all(|value| *value == 3.0));
    assert!(recurrent.data_f32().iter().all(|value| *value == 4.0));
}

#[test]
fn clone_drops_transient_linear_prefix_checkpoint() {
    let mut cache = MlxKVCache::new(1);
    cache.set_linear_state(
        0,
        zeros(&[1, 2, 3], MlxDtype::Float32, None),
        zeros(&[1, 2, 2, 2], MlxDtype::Float32, None),
    );
    cache.begin_linear_prefix_capture(1);
    cache.set_linear_prefix_checkpoint(
        0,
        zeros(&[1, 2, 3], MlxDtype::Float32, None),
        zeros(&[1, 2, 2, 2], MlxDtype::Float32, None),
    );

    let mut branch = cache.clone();
    assert_eq!(branch.linear_prefix_capture_after(), None);
    assert!(!branch.restore_linear_prefix_checkpoint());
    let (conv, recurrent) = branch.linear_state(0);
    assert!(conv.is_some() && recurrent.is_some());
}

#[test]
fn linear_mtp_projection_stash_is_transient() {
    let mut cache = MlxKVCache::new(1);
    cache.set_linear_state(
        0,
        zeros(&[1, 2, 3], MlxDtype::Float32, None),
        zeros(&[1, 2, 2, 2], MlxDtype::Float32, None),
    );
    cache.begin_linear_prefix_capture(1);
    cache.set_linear_mtp_projection_stash(
        0,
        zeros(&[1, 3, 8], MlxDtype::Bfloat16, None),
        zeros(&[1, 3, 2], MlxDtype::Bfloat16, None),
        zeros(&[1, 3, 2], MlxDtype::Bfloat16, None),
    );
    cache.set_linear_mtp_processed_stash(
        0,
        zeros(&[1, 3, 1, 4], MlxDtype::Bfloat16, None),
        zeros(&[1, 3, 1, 4], MlxDtype::Bfloat16, None),
        zeros(&[1, 3, 2, 4], MlxDtype::Bfloat16, None),
    );
    assert!(cache.linear_mtp_projection_stash(0).is_some());
    assert!(cache.linear_mtp_processed_stash(0).is_some());

    let branch = cache.clone();
    assert!(branch.linear_mtp_projection_stash(0).is_none());
    assert!(branch.linear_mtp_processed_stash(0).is_none());

    cache.clear_linear_prefix_checkpoint();
    assert!(cache.linear_mtp_projection_stash(0).is_none());
    assert!(cache.linear_mtp_processed_stash(0).is_none());
}

#[test]
fn linear_mtp_tape_stash_is_transient() {
    let mut cache = MlxKVCache::new(1);
    cache.begin_linear_prefix_capture(1);
    cache.set_linear_mtp_tape_stash(
        0,
        zeros(&[1, 3, 8], MlxDtype::Bfloat16, None),
        zeros(&[1, 3, 2], MlxDtype::Bfloat16, None),
        zeros(&[1, 3, 2, 4], MlxDtype::Float32, None),
    );
    assert!(cache.linear_mtp_tape_stash(0).is_some());
    assert!(cache.linear_mtp_projection_stash(0).is_none());

    let branch = cache.clone();
    assert!(branch.linear_mtp_tape_stash(0).is_none());

    cache.clear_linear_prefix_checkpoint();
    assert!(cache.linear_mtp_tape_stash(0).is_none());
}

#[test]
fn whole_verify_backing_reserves_capacity_without_advancing_length() {
    let mut cache = MlxKVCache::new_contiguous(1);
    let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
    let _ = cache.append(0, k, v);
    cache.advance(3);

    let (backing_k, backing_v) = cache
        .prepare_whole_verify_layer_kv(0, 300)
        .expect("dense backing");
    assert_eq!(backing_k.shape(), vec![1, 2, 512, 4]);
    assert_eq!(backing_v.shape(), vec![1, 2, 512, 4]);
    assert_eq!(cache.seq_len(), 3);
    assert_eq!(contiguous_layer(&cache, 0).capacity, 512);
}

#[test]
fn glm_mla_cache_appends_latent_and_rope_key_history() {
    let mut cache = MlxKVCache::new(1);
    let kv_latent = zeros(&[1, 1, 2, 512], MlxDtype::Bfloat16, None);
    let k_pe = zeros(&[1, 1, 2, 64], MlxDtype::Bfloat16, None);

    let (latent_history, rope_history) = cache.append_glm_mla(0, kv_latent, k_pe);

    assert_eq!(latent_history.shape(), vec![1, 1, 2, 512]);
    assert_eq!(rope_history.shape(), vec![1, 1, 2, 64]);
    assert_eq!(cache.collect_eval_refs().len(), 2);
    cache.seq_len = 2;

    let kv_latent = zeros(&[1, 1, 1, 512], MlxDtype::Bfloat16, None);
    let k_pe = zeros(&[1, 1, 1, 64], MlxDtype::Bfloat16, None);
    let (latent_history, rope_history) = cache.append_glm_mla(0, kv_latent, k_pe);

    assert_eq!(latent_history.shape(), vec![1, 1, 3, 512]);
    assert_eq!(rope_history.shape(), vec![1, 1, 3, 64]);
    assert_eq!(cache.collect_eval_refs().len(), 2);
}

#[test]
fn usage_snapshot_tracks_glm_mla_compressed_cache_bytes() {
    let mut cache = MlxKVCache::new(1);
    let kv_latent = zeros(&[1, 1, 2, 512], MlxDtype::Bfloat16, None);
    let k_pe = zeros(&[1, 1, 2, 64], MlxDtype::Bfloat16, None);

    cache.append_glm_mla(0, kv_latent, k_pe);
    cache.seq_len = 2;

    let usage = cache.usage_snapshot();
    assert_eq!(usage.logical_tokens, 2);
    assert_eq!(usage.capacity_tokens, 2);
    assert_eq!(usage.full_attention_layers, 1);
    assert_eq!(usage.logical_bytes, 2304);
    assert_eq!(usage.capacity_bytes, 2304);
    assert_eq!(usage.growth_count, 1);
}

#[test]
fn reset_clears_glm_mla_cache_eval_refs() {
    let mut cache = MlxKVCache::new(1);
    let kv_latent = zeros(&[1, 1, 1, 512], MlxDtype::Bfloat16, None);
    let k_pe = zeros(&[1, 1, 1, 64], MlxDtype::Bfloat16, None);

    cache.append_glm_mla(0, kv_latent, k_pe);
    cache.seq_len = 1;
    assert_eq!(cache.collect_eval_refs().len(), 2);

    cache.reset();

    assert_eq!(cache.seq_len, 0);
    assert!(cache.collect_eval_refs().is_empty());
    assert_eq!(cache.usage_snapshot(), MlxKVCacheUsage::default());
}

#[test]
fn usage_snapshot_tracks_full_attention_capacity_and_growth() {
    let mut cache = MlxKVCache::new(1);
    let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);

    cache.append(0, k, v);
    cache.seq_len = 3;

    let usage = cache.usage_snapshot();
    assert_eq!(usage.logical_tokens, 3);
    assert_eq!(usage.capacity_tokens, 3);
    assert_eq!(usage.full_attention_layers, 1);
    assert_eq!(usage.logical_bytes, 96);
    assert_eq!(usage.capacity_bytes, 96);
    assert_eq!(usage.growth_count, 1);
}

#[test]
fn aligned_prefill_grow_concatenates_instead_of_zeros() {
    let mut cache = MlxKVCache::new(1);
    let k0 = zeros(&[1, 2, 1024, 4], MlxDtype::Bfloat16, None);
    let v0 = zeros(&[1, 2, 1024, 4], MlxDtype::Bfloat16, None);
    let _ = cache.append(0, k0, v0);
    cache.advance(1024);
    assert_eq!(contiguous_layer(&cache, 0).capacity, 1024);

    let k1 = zeros(&[1, 2, 1024, 4], MlxDtype::Bfloat16, None);
    let v1 = zeros(&[1, 2, 1024, 4], MlxDtype::Bfloat16, None);
    let (k_view, v_view) = cache.append(0, k1, v1);
    cache.advance(1024);

    assert_eq!(k_view.shape(), vec![1, 2, 2048, 4]);
    assert_eq!(v_view.shape(), vec![1, 2, 2048, 4]);
    assert_eq!(contiguous_layer(&cache, 0).capacity, 2048);
    assert_eq!(cache.usage_snapshot().capacity_tokens, 2048);
    assert_eq!(contiguous_layer(&cache, 0).k.shape(), vec![1, 2, 2048, 4]);
    let layer = contiguous_layer(&cache, 0);
    assert_eq!(
        layer.last_k_view.as_ref().map(MlxArray::shape),
        Some(layer.k.shape()),
        "full-buffer view must skip the identity slice"
    );
}

#[test]
fn peek_layer_full_kv_ignores_retained_last_view() {
    let mut cache = MlxKVCache::new(1);
    let k = zeros(&[1, 2, 5, 4], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 2, 5, 4], MlxDtype::Bfloat16, None);

    cache.append_with_retained_window(0, k, v, Some(3));
    cache.seq_len = 5;

    let (retained_k, _) = cache.peek_layer_kv(0).expect("retained view");
    let (full_k, _) = cache.peek_layer_full_kv(0).expect("full view");
    assert_eq!(retained_k.shape(), vec![1, 2, 3, 4]);
    assert_eq!(full_k.shape(), vec![1, 2, 5, 4]);
}

#[test]
fn multi_token_append_retains_window_plus_seq_view() {
    // Prefill 8 tokens (full view), then append a 4-token chunk with the
    // multi-token retained bound window + seq - 1 = 3 + 4 - 1 = 6: the
    // returned view must be the last 6 tokens of the 12-token history,
    // contents intact, while full storage stays available for rollback
    // and prefix-cache snapshots.
    let head_dim = 2usize;
    let fill = |start: usize, tokens: usize| -> MlxArray {
        let data: Vec<f32> = (start..start + tokens * head_dim)
            .map(|i| i as f32)
            .collect();
        let flat = MlxArray::from_f32_slice(&data);
        mlx_sys::reshape(&flat, &[1, 1, tokens as i32, head_dim as i32], None)
    };
    let read_f32 = |arr: &MlxArray| -> Vec<f32> {
        let arr = astype(arr, MlxDtype::Float32, None);
        eval(&[&arr]);
        let len = arr.nbytes() / std::mem::size_of::<f32>();
        let ptr = arr.data_raw() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    };

    let mut cache = MlxKVCache::new(1);
    cache.append(0, fill(0, 8), fill(100, 8));
    cache.seq_len = 8;

    let (k_view, v_view) = cache.append_with_retained_window(0, fill(16, 4), fill(116, 4), Some(6));
    cache.seq_len = 12;

    assert_eq!(k_view.shape(), vec![1, 1, 6, head_dim as i32]);
    assert_eq!(v_view.shape(), vec![1, 1, 6, head_dim as i32]);
    // Last 6 tokens = prompt tokens 6..8 (values 12..16) + the 4 new
    // tokens (values 16..24).
    let expected_k: Vec<f32> = (12..24).map(|i| i as f32).collect();
    let expected_v: Vec<f32> = (112..124).map(|i| i as f32).collect();
    assert_eq!(read_f32(&k_view), expected_k);
    assert_eq!(read_f32(&v_view), expected_v);

    let (full_k, full_v) = cache.peek_layer_full_kv(0).expect("full view");
    assert_eq!(full_k.shape(), vec![1, 1, 12, head_dim as i32]);
    let full_k_data = read_f32(&full_k);
    assert_eq!(
        &full_k_data[..16],
        (0..16).map(|i| i as f32).collect::<Vec<_>>()
    );
    assert_eq!(
        &full_k_data[16..],
        (16..24).map(|i| i as f32).collect::<Vec<_>>()
    );
    assert_eq!(read_f32(&full_v).len(), 24);
}

#[test]
fn protected_prefix_ring_retains_prefill_and_rotates_only_decode_tokens() {
    let token =
        |value: f32| mlx_sys::reshape(&MlxArray::from_f32_slice(&[value]), &[1, 1, 1, 1], None);
    let read = |arr: &MlxArray| {
        let arr = astype(arr, MlxDtype::Float32, None);
        eval(&[&arr]);
        let len = arr.nbytes() / std::mem::size_of::<f32>();
        let ptr = arr.data_raw() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    };

    let mut cache = MlxKVCache::new_contiguous(1);
    let prefill = mlx_sys::reshape(
        &MlxArray::from_f32_slice(&[0.0, 1.0, 2.0]),
        &[1, 1, 3, 1],
        None,
    );
    cache.append(0, prefill.clone(), prefill);
    cache.advance(3);

    let mut last_k = None;
    for value in [3.0, 4.0, 5.0, 6.0] {
        let kv = cache.append_with_protected_prefix_window_for_attention(
            0,
            token(value),
            token(value + 100.0),
            2,
        );
        let (k, _) = kv.into_dense();
        last_k = Some(k);
        cache.advance(1);
    }

    let last_k = last_k.expect("decode produced a KV view");
    assert_eq!(last_k.shape(), vec![1, 1, 5, 1]);
    assert_eq!(read(&last_k), vec![0.0, 1.0, 2.0, 5.0, 6.0]);
    let ring = contiguous_layer(&cache, 0)
        .protected_prefix_ring
        .as_ref()
        .expect("protected-prefix ring initialized");
    assert_eq!((ring.prefix_len, ring.window), (3, 2));
    assert!(!cache.trim_to(6), "ring decode must decline rollback");
    assert_eq!(cache.seq_len(), 7);
    assert!(cache.has_rotated_sliding_layers());
}

#[test]
fn usage_snapshot_tracks_sliding_window_trim_opportunity() {
    let mut cache = MlxKVCache::new(1);
    let k = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);

    cache.append(0, k, v);
    cache.seq_len = 300;

    let usage = cache.usage_snapshot_with_layer_windows(&[Some(128)]);
    assert_eq!(usage.capacity_tokens, 300);
    assert_eq!(usage.sliding_window_layers, 1);
    assert_eq!(usage.sliding_window_retained_tokens, 128);
    assert_eq!(usage.sliding_window_reclaimable_capacity_tokens, 44);
    assert_eq!(usage.sliding_window_reclaimable_capacity_bytes, 1408);
}

#[test]
fn usage_snapshot_ignores_unwritten_sliding_window_layers() {
    let mut cache = MlxKVCache::new(2);
    let k = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);

    cache.append(0, k, v);
    cache.seq_len = 300;

    let usage = cache.usage_snapshot_with_layer_windows(&[Some(128), Some(128)]);
    assert_eq!(usage.full_attention_layers, 1);
    assert_eq!(usage.sliding_window_layers, 1);
    assert_eq!(usage.sliding_window_reclaimable_capacity_tokens, 44);
}

#[test]
fn usage_snapshot_does_not_report_reclaimable_capacity_inside_window() {
    let mut cache = MlxKVCache::new(1);
    let k = zeros(&[1, 2, 120, 4], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 2, 120, 4], MlxDtype::Bfloat16, None);

    cache.append(0, k, v);
    cache.seq_len = 120;

    let usage = cache.usage_snapshot_with_layer_windows(&[Some(512)]);
    assert_eq!(usage.capacity_tokens, 120);
    assert_eq!(usage.sliding_window_layers, 1);
    assert_eq!(usage.sliding_window_retained_tokens, 120);
    assert_eq!(usage.sliding_window_reclaimable_capacity_tokens, 0);
    assert_eq!(usage.sliding_window_reclaimable_capacity_bytes, 0);
}

#[test]
#[should_panic(expected = "matching K/V shapes")]
fn append_rejects_mismatched_kv_shapes() {
    let mut cache = MlxKVCache::new(1);
    let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 2, 4, 4], MlxDtype::Bfloat16, None);

    let _ = cache.append(0, k, v);
}

#[test]
#[should_panic(expected = "cannot change head_dim")]
fn append_rejects_existing_layer_shape_drift() {
    let mut cache = MlxKVCache::new(1);
    let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
    let _ = cache.append(0, k, v);
    cache.seq_len = 3;

    let k = zeros(&[1, 2, 1, 5], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 2, 1, 5], MlxDtype::Bfloat16, None);
    let _ = cache.append(0, k, v);
}

#[test]
#[should_panic(expected = "requires matching K/V dtypes")]
fn append_rejects_mismatched_kv_dtypes() {
    let mut cache = MlxKVCache::new(1);
    let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 2, 3, 4], MlxDtype::Float32, None);

    let _ = cache.append(0, k, v);
}

#[test]
fn usage_snapshot_tracks_linear_state_bytes() {
    let mut cache = MlxKVCache::new(1);
    let conv = zeros(&[1, 3, 14], MlxDtype::Float32, None);
    let recurrent = zeros(&[1, 4, 8, 6], MlxDtype::Float32, None);

    cache.set_linear_state(0, conv, recurrent);

    let usage = cache.usage_snapshot();
    assert_eq!(usage.linear_state_layers, 1);
    assert_eq!(usage.linear_state_bytes, 936);
}

#[test]
fn peek_source_kv_reuses_cached_views_from_append() {
    use mlx_sys::eval;
    // Two-layer cache: layer 0 is the source, layer 1 is a KV-shared consumer.
    let mut cache = MlxKVCache::new(2);

    let k = zeros(&[1, 1, 4, 8], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 1, 4, 8], MlxDtype::Bfloat16, None);
    let (k_from_append, _) = cache.append(0, k, v);
    cache.seq_len = 4;

    let (k_from_peek, _) = cache.peek_source_kv(0, 0);

    // Materialise both arrays. If peek returned the same lazy node as append,
    // the results must be numerically identical (same shape and dtype).
    eval(&[&k_from_append, &k_from_peek]);
    assert_eq!(k_from_append.shape(), k_from_peek.shape());
    assert_eq!(k_from_append.dtype(), k_from_peek.dtype());

    // After a buffer grow, last_k_view is cleared and peek falls back to a
    // fresh slice — verify the fallback also produces the correct shape.
    let k2 = zeros(&[1, 1, 300, 8], MlxDtype::Bfloat16, None);
    let v2 = zeros(&[1, 1, 300, 8], MlxDtype::Bfloat16, None);
    cache.append(0, k2, v2);
    cache.seq_len = 304;

    let (k_grow, _) = cache.peek_source_kv(0, 0);
    eval(&[&k_grow]);
    assert_eq!(k_grow.shape(), vec![1, 1, 304, 8]);
}

#[test]
fn append_with_retained_window_returns_windowed_cached_views() {
    use mlx_sys::eval;

    let mut cache = MlxKVCache::new(2);
    let k = zeros(&[1, 1, 6, 8], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 1, 6, 8], MlxDtype::Bfloat16, None);

    let (k_from_append, v_from_append) = cache.append_with_retained_window(0, k, v, Some(4));
    cache.seq_len = 6;
    let (k_from_peek, v_from_peek) = cache.peek_source_kv(0, 0);

    eval(&[&k_from_append, &v_from_append, &k_from_peek, &v_from_peek]);
    assert_eq!(k_from_append.shape(), vec![1, 1, 4, 8]);
    assert_eq!(v_from_append.shape(), vec![1, 1, 4, 8]);
    assert_eq!(k_from_peek.shape(), vec![1, 1, 4, 8]);
    assert_eq!(v_from_peek.shape(), vec![1, 1, 4, 8]);
}

#[test]
fn rotating_sliding_decode_uses_bounded_backing_store() {
    let mut cache = MlxKVCache::new(1);
    cache.set_rotating_sliding_decode(true);

    let k = zeros(&[1, 1, 6, 8], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 1, 6, 8], MlxDtype::Bfloat16, None);
    let (prefill_k, _) = cache.append(0, k, v);
    cache.seq_len = 6;
    assert_eq!(prefill_k.shape(), vec![1, 1, 6, 8]);

    let next_k = zeros(&[1, 1, 1, 8], MlxDtype::Bfloat16, None);
    let next_v = zeros(&[1, 1, 1, 8], MlxDtype::Bfloat16, None);
    let (decode_k, decode_v) = cache.append_with_retained_window(0, next_k, next_v, Some(4));

    let lkv = contiguous_layer(&cache, 0);
    assert_eq!(lkv.capacity, 4);
    assert_eq!(lkv.rotating_window, Some(4));
    assert_eq!(decode_k.shape(), vec![1, 1, 4, 8]);
    assert_eq!(decode_v.shape(), vec![1, 1, 4, 8]);
}

#[test]
fn trim_to_rejects_rollback_after_rotating_sliding_decode() {
    let mut cache = MlxKVCache::new(1);
    cache.set_rotating_sliding_decode(true);
    let k = zeros(&[1, 1, 4, 8], MlxDtype::Bfloat16, None);
    let v = zeros(&[1, 1, 4, 8], MlxDtype::Bfloat16, None);
    cache.append(0, k, v);
    cache.seq_len = 4;

    let next_k = zeros(&[1, 1, 1, 8], MlxDtype::Bfloat16, None);
    let next_v = zeros(&[1, 1, 1, 8], MlxDtype::Bfloat16, None);
    cache.append_with_retained_window(0, next_k, next_v, Some(4));
    cache.seq_len = 5;

    assert!(!cache.trim_to(4));
    assert_eq!(cache.seq_len, 5);
}

// ── Bounded-rollback rotating ring tests ──

/// `[1, 1, len, head_dim]` f32 array where token row `i` is filled with
/// `values[i]`, so slot contents are identifiable after ring writes.
fn tokens_f32(values: &[f32], head_dim: usize) -> MlxArray {
    let data: Vec<f32> = values
        .iter()
        .flat_map(|&value| std::iter::repeat_n(value, head_dim))
        .collect();
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data.as_slice()),
        &[1, 1, values.len() as i32, head_dim as i32],
        MlxDtype::Float32,
    )
}

/// First element of each token row of a `[1, 1, len, head_dim]` array.
fn token_row_values(arr: &MlxArray, head_dim: usize) -> Vec<f32> {
    eval(&[arr]);
    arr.data_f32().chunks(head_dim).map(|row| row[0]).collect()
}

#[test]
fn sliding_ring_layout_gates_by_mode_seq_and_crossing() {
    let mut cache = MlxKVCache::new(1);
    cache.seq_len = 10;
    // Rotation disabled: never a ring.
    assert_eq!(cache.sliding_ring_layout(Some(4), 1), None);
    cache.set_rotating_sliding_decode(true);
    // Pure mode: single-token only.
    let pure = cache.sliding_ring_layout(Some(4), 1).expect("pure ring");
    assert_eq!((pure.window, pure.capacity, pure.write_start), (4, 4, 10));
    assert!(!pure.needs_mask(1));
    assert_eq!(cache.sliding_ring_layout(Some(4), 2), None);
    // Bounded mode: multi-token up to the slack, always masked.
    cache.set_rotating_sliding_slack(3);
    let ring = cache.sliding_ring_layout(Some(4), 3).expect("bounded ring");
    assert_eq!((ring.window, ring.capacity, ring.write_start), (4, 7, 10));
    assert!(ring.needs_mask(1));
    assert_eq!(cache.sliding_ring_layout(Some(4), 4), None);
    // Not yet crossing the window, and no window at all: ordered path.
    cache.seq_len = 2;
    assert_eq!(cache.sliding_ring_layout(Some(4), 2), None);
    assert_eq!(cache.sliding_ring_layout(Some(4), 3), None);
    cache.seq_len = 4;
    assert!(cache.sliding_ring_layout(Some(4), 3).is_some());
    cache.seq_len = 10;
    assert_eq!(cache.sliding_ring_layout(None, 1), None);
}

#[test]
fn cold_ring_multi_token_append_returns_capacity_shaped_kv() {
    // First append past the window on an empty layer must cold-init a
    // capacity ring (not ordered windowed view) so SDPA masks sized to
    // capacity broadcast against the returned K/V.
    const HD: usize = 4;
    let mut cache = MlxKVCache::new(1);
    cache.set_rotating_sliding_decode(true);
    // Multi-token ring eligibility requires seq <= slack.
    cache.set_rotating_sliding_slack(7); // window 4 → capacity 11
    cache.seq_len = 0;
    let values: Vec<f32> = (1..=7).map(|v| v as f32).collect();
    let k = tokens_f32(&values, HD);
    let v = tokens_f32(&values, HD);
    let (ck, cv) = cache.append_with_retained_window(0, k, v, Some(4));
    cache.seq_len = 7;
    assert_eq!(ck.shape(), vec![1, 1, 11, HD as i32]);
    assert_eq!(cv.shape(), vec![1, 1, 11, HD as i32]);
    let lkv = contiguous_layer(&cache, 0);
    assert_eq!(lkv.rotating_window, Some(4));
    assert_eq!(lkv.capacity, 11);
    assert_eq!(
        token_row_values(&lkv.k, HD),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn bounded_ring_multi_token_append_places_tokens_by_slot_and_wraps() {
    const HD: usize = 4;
    let mut cache = MlxKVCache::new(1);
    cache.set_rotating_sliding_decode(true);
    cache.set_rotating_sliding_slack(3); // window 4 → capacity 7
    // Prefill 4 tokens (values 1..=4 for tokens 0..=3).
    let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
    let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
    cache.append(0, k, v);
    cache.seq_len = 4;

    // 3-token verify-style append crosses the window: tokens 4, 5, 6.
    let k = tokens_f32(&[5.0, 6.0, 7.0], HD);
    let v = tokens_f32(&[5.0, 6.0, 7.0], HD);
    let (ck, _) = cache.append_with_retained_window(0, k, v, Some(4));
    cache.seq_len = 7;
    assert!(cache.has_rotated_sliding_layers());

    let lkv = contiguous_layer(&cache, 0);
    assert_eq!(lkv.rotating_window, Some(4));
    assert_eq!(lkv.capacity, 7);
    assert_eq!(ck.shape(), vec![1, 1, 7, HD as i32]);
    // Conversion copies tokens 1..=3 (window - 1 back from write_start 4)
    // to slots 1..=3; new tokens 4..=6 land at slots 4..=6; slot 0 (token
    // 0's slot) was outside the copy range and stays zero.
    assert_eq!(
        token_row_values(&lkv.k, HD),
        vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    );

    // Two single-token appends wrap: token 7 → slot 0, token 8 → slot 1.
    for (t, value) in [(7usize, 8.0f32), (8, 9.0)] {
        let k = tokens_f32(&[value], HD);
        let v = tokens_f32(&[value], HD);
        cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = t + 1;
    }
    let lkv = contiguous_layer(&cache, 0);
    assert_eq!(
        token_row_values(&lkv.k, HD),
        vec![8.0, 9.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    );
}

#[test]
fn ordered_cache_reports_no_rotated_sliding_layers() {
    let mut cache = MlxKVCache::new(1);
    let k = tokens_f32(&[1.0, 2.0], 4);
    let v = tokens_f32(&[3.0, 4.0], 4);
    cache.append(0, k, v);
    cache.advance(2);
    assert!(!cache.has_rotated_sliding_layers());
}

#[test]
fn bounded_ring_trim_within_slack_rewrites_same_slots() {
    const HD: usize = 4;
    let mut cache = MlxKVCache::new(1);
    cache.set_rotating_sliding_decode(true);
    cache.set_rotating_sliding_slack(3);
    let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
    let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
    cache.append(0, k, v);
    cache.seq_len = 4;
    // Verify forward: draft tokens 4, 5, 6 (values 5, 6, 7).
    let k = tokens_f32(&[5.0, 6.0, 7.0], HD);
    let v = tokens_f32(&[5.0, 6.0, 7.0], HD);
    cache.append_with_retained_window(0, k, v, Some(4));
    cache.seq_len = 7;

    // Reject the last two draft tokens (rollback depth 2 <= slack 3).
    assert!(cache.trim_to(5));
    assert_eq!(cache.seq_len, 5);

    // The corrected continuation rewrites tokens 5 and 6 with new values;
    // they land in the same slots the rejected tokens occupied.
    let k = tokens_f32(&[60.0, 70.0], HD);
    let v = tokens_f32(&[60.0, 70.0], HD);
    cache.append_with_retained_window(0, k, v, Some(4));
    cache.seq_len = 7;
    let lkv = contiguous_layer(&cache, 0);
    assert_eq!(
        token_row_values(&lkv.k, HD),
        vec![0.0, 2.0, 3.0, 4.0, 5.0, 60.0, 70.0]
    );

    // Rollback deeper than the slack is refused (fail-closed).
    assert!(!cache.trim_to(3));
    assert_eq!(cache.seq_len, 7);
    // Pure rings (slack 0 on another cache) still refuse any real trim —
    // covered by trim_to_rejects_rollback_after_rotating_sliding_decode.
}

/// The Gemma4 assistant drafter reads target KV between appends via
/// `peek_layer_kv` — including right after a verify rollback, when the
/// cached views are cleared and the ordered `[0, seq_len)` fallback
/// would slice past a ring's capacity. Rotated layers must return the
/// full ring plus geometry for the slot-validity mask.
#[test]
fn peek_layer_kv_returns_full_ring_after_rollback() {
    const HD: usize = 4;
    let mut cache = MlxKVCache::new(1);
    cache.set_rotating_sliding_decode(true);
    cache.set_rotating_sliding_slack(3); // window 4 → capacity 7
    let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
    let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
    cache.append(0, k, v);
    cache.seq_len = 4;
    let k = tokens_f32(&[5.0, 6.0, 7.0], HD);
    let v = tokens_f32(&[5.0, 6.0, 7.0], HD);
    cache.append_with_retained_window(0, k, v, Some(4));
    cache.seq_len = 7;
    // Partial reject: views are cleared, seq_len (5) < ring capacity (7).
    assert!(cache.trim_to(5));

    let (k, _v) = cache.peek_layer_kv(0).expect("layer peek");
    assert_eq!(k.shape(), vec![1, 1, 7, HD as i32]);
    let ring = cache.layer_sliding_ring(0).expect("ring geometry");
    assert_eq!((ring.window, ring.capacity, ring.write_start), (4, 7, 5));

    // The end-anchored drafter mask keeps exactly the last `window` live
    // tokens (1..=4) and excludes the rolled-back slots (tokens 5, 6 at
    // slots 5, 6) and token 0's never-copied slot 0.
    let mask = crate::attention_mask::create_ring_sliding_mask(
        1,
        ring.window,
        ring.capacity,
        ring.write_start - 1,
    );
    eval(&[&mask]);
    let bits: Vec<u8> =
        unsafe { std::slice::from_raw_parts(mask.data_raw(), mask.nbytes()).to_vec() };
    assert_eq!(bits, vec![0, 1, 1, 1, 1, 0, 0]);

    // Un-rotated layers keep the ordered peek contract.
    let plain = MlxKVCache::new(1);
    assert!(plain.layer_sliding_ring(0).is_none());
}

/// End-to-end oracle: masked SDPA over the ring (unordered slots + slot-
/// validity mask) must equal unmasked SDPA over the ordered sliding
/// window from a plain cache, through a conversion → wrap → rollback →
/// rewrite trajectory. This is the property that makes bounded rings a
/// drop-in for ordered window views.
#[test]
fn bounded_ring_sdpa_matches_ordered_window_reference() {
    use crate::attention_mask::create_ring_sliding_mask;
    use mlx_sys::{ScaledDotProductAttentionMask, scaled_dot_product_attention_with_mask};
    const HD: usize = 4;
    const WINDOW: usize = 4;
    const SLACK: usize = 3;
    let scale = 1.0 / (HD as f32).sqrt();

    let mut ring_cache = MlxKVCache::new(1);
    ring_cache.set_rotating_sliding_decode(true);
    ring_cache.set_rotating_sliding_slack(SLACK);
    let mut plain_cache = MlxKVCache::new(1);

    // Distinct K/V rows per token so misplaced slots change the output.
    let tok = |t: usize| ((t + 1) as f32) * 0.25;
    let prefill: Vec<f32> = (0..WINDOW).map(tok).collect();
    for cache in [&mut ring_cache, &mut plain_cache] {
        let k = tokens_f32(&prefill, HD);
        let v = tokens_f32(&prefill, HD);
        cache.append(0, k, v);
        cache.seq_len = WINDOW;
    }

    // Trajectory: 3-token verify (tokens 4-6), reject 2 (trim to 5),
    // 2-token re-verify (tokens 5-6 with new values), then a single-token
    // step (token 7) that wraps the ring.
    struct Step {
        values: Vec<f32>,
        trim_to: Option<usize>,
    }
    let steps = [
        Step {
            values: vec![tok(4), tok(5), tok(6)],
            trim_to: Some(5),
        },
        Step {
            values: vec![9.5, 10.5],
            trim_to: None,
        },
        Step {
            values: vec![11.5],
            trim_to: None,
        },
    ];

    for step in steps {
        let seq = step.values.len();
        let write_start = ring_cache.seq_len;
        assert_eq!(write_start, plain_cache.seq_len, "caches stay in sync");
        let q = tokens_f32(&step.values, HD); // queries; values arbitrary

        // Ring side: append with the raw window, mask over the ring.
        let ring = ring_cache
            .sliding_ring_layout(Some(WINDOW), seq)
            .expect("trajectory stays on the ring past the window");
        let k = tokens_f32(&step.values, HD);
        let v = tokens_f32(&step.values, HD);
        let (ring_k, ring_v) = ring_cache.append_with_retained_window(0, k, v, Some(WINDOW));
        let ring_mask = create_ring_sliding_mask(seq, ring.window, ring.capacity, ring.write_start);
        let ring_out = scaled_dot_product_attention_with_mask(
            &q,
            &ring_k,
            &ring_v,
            scale,
            ScaledDotProductAttentionMask::Array(&ring_mask),
            None,
        );

        // Plain side: ordered append; reference is the ordered full view
        // with the standard causal sliding-window mask, so both sides run
        // the same masked-SDPA kernel and the comparison isolates ring
        // slot/mask placement (kernel-level masked-vs-unmasked numeric
        // drift is ~1e-3 in f32 and not what this test is about).
        let k = tokens_f32(&step.values, HD);
        let v = tokens_f32(&step.values, HD);
        plain_cache.append_with_retained_window(0, k, v, None);
        let plain = contiguous_layer(&plain_cache, 0);
        let write_end = (write_start + seq) as i32;
        let ones = [1i32, 1, 1, 1];
        let ordered_k = slice(
            &plain.k,
            &[0, 0, 0, 0],
            &[1, 1, write_end, HD as i32],
            &ones,
            None,
        );
        let ordered_v = slice(
            &plain.v,
            &[0, 0, 0, 0],
            &[1, 1, write_end, HD as i32],
            &ones,
            None,
        );
        let ordered_mask =
            crate::attention_mask::create_causal_mask(seq, write_start, Some(WINDOW));
        let want = scaled_dot_product_attention_with_mask(
            &q,
            &ordered_k,
            &ordered_v,
            scale,
            ScaledDotProductAttentionMask::Array(&ordered_mask),
            None,
        );
        eval(&[&ring_out, &want]);
        let got = ring_out.data_f32().to_vec();
        let want = want.data_f32().to_vec();
        for i in 0..seq {
            for d in 0..HD {
                let g = got[i * HD + d];
                let w = want[i * HD + d];
                assert!(
                    (g - w).abs() < 1e-5,
                    "step query {i} dim {d}: ring {g} vs ordered {w}"
                );
            }
        }

        let new_len = write_start + seq;
        ring_cache.seq_len = new_len;
        plain_cache.seq_len = new_len;
        if let Some(target) = step.trim_to {
            assert!(ring_cache.trim_to(target), "trim within slack");
            assert!(plain_cache.trim_to(target));
        }
    }
}

// ── F3 M1 serialize / deserialize round-trip tests ──

fn build_fa_array_f32(seq_len: usize, n_kv_heads: i32, head_dim: i32) -> MlxArray {
    let total = (n_kv_heads as usize) * seq_len * (head_dim as usize);
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data.as_slice()),
        &[1, n_kv_heads, seq_len as i32, head_dim],
        MlxDtype::Float32,
    )
}

fn build_mla_latent_f32(seq_len: usize, inner: i32) -> MlxArray {
    let total = seq_len * (inner as usize);
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.0007).collect();
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data.as_slice()),
        &[1, 1, seq_len as i32, inner],
        MlxDtype::Float32,
    )
}

fn host_f32(arr: &MlxArray) -> Vec<f32> {
    // Materialize a tight C-contiguous buffer first: `data_f32` on a
    // lazy slice of an over-allocated KV buffer can otherwise surface
    // capacity padding and break token-exact comparisons.
    let tight = contiguous(arr, None);
    eval(&[&tight]);
    tight.data_f32().to_vec()
}

/// Build a contiguous f32 tensor with an explicit shape, mirroring
/// `build_fa_array_f32` but for the arbitrary ranks the draft-cursor
/// codec round-trips (QSA K/V are rank 4, the index cache rank 3, the
/// `stream_hidden` row rank 3).
fn build_f32(shape: &[i32], scale: f32) -> MlxArray {
    let total: usize = shape.iter().map(|&d| d.max(0) as usize).product();
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * scale).collect();
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data.as_slice()),
        shape,
        MlxDtype::Float32,
    )
}

/// One QSA layer with `tokens` tokens: keys/values are
/// `[1, tokens, kv_heads, head_dim]` and the index cache is
/// `[1, tokens, index_head_dim]`. `position` must equal `tokens` for the
/// state to pass `from_serialized_layers`' self-consistency check.
fn draft_cursor_fixture(tokens: usize) -> (crate::model::qwen4_exp::Qwen4ExpState, MlxArray) {
    let keys = build_f32(&[1, tokens as i32, 2, 4], 0.001);
    let values = build_f32(&[1, tokens as i32, 2, 4], 0.002);
    let index_keys = build_f32(&[1, tokens as i32, 4], 0.003);
    let state = crate::model::qwen4_exp::Qwen4ExpState::synthetic_qsa_state(
        1701, tokens, keys, values, index_keys,
    )
    .expect("synthetic QSA state");
    let stream_hidden = build_f32(&[1, 1, 8], 0.004);
    eval(&[&stream_hidden]);
    (state, stream_hidden)
}

#[test]
fn serialize_qwen4_exp_draft_cursor_roundtrips_values() {
    let (state, stream_hidden) = draft_cursor_fixture(3);

    let bytes = MlxKVCache::serialize_qwen4_exp_draft_cursor(&state, &stream_hidden);
    assert_eq!(&bytes[..4], b"AXMC", "cursor magic must be AXMC");
    assert_ne!(&bytes[..4], b"AXKB", "cursor magic must differ from trunk");

    let (restored, restored_hidden) =
        MlxKVCache::try_deserialize_qwen4_exp_draft_cursor(&bytes).expect("round-trip");

    // The whole payload must re-serialize byte-for-byte identically: this
    // catches any shape/dtype/order drift the value checks might miss.
    let reserialized = MlxKVCache::serialize_qwen4_exp_draft_cursor(&restored, &restored_hidden);
    assert_eq!(
        bytes, reserialized,
        "cursor payload must round-trip exactly"
    );

    // Field-level fidelity: tensors must survive byte-for-byte, not
    // approximately. `arrays()` returns QSA keys, values, then index.
    let original = state.arrays();
    let recovered = restored.arrays();
    assert_eq!(original.len(), 3);
    assert_eq!(recovered.len(), original.len());
    for (before, after) in original.iter().zip(recovered.iter()) {
        assert_eq!(before.shape(), after.shape());
        assert_eq!(before.dtype(), after.dtype());
        assert_eq!(host_f32(before), host_f32(after));
    }
    assert_eq!(host_f32(&stream_hidden), host_f32(&restored_hidden));

    // A cursor payload is not a trunk blob and vice versa: the distinct
    // magics must make each reader reject the other's payload.
    assert!(matches!(
        MlxKVCache::try_deserialize_from_bytes(&bytes),
        Err(MlxKVCacheSerializeError::BadMagic)
    ));
}

#[test]
fn draft_cursor_codec_rejects_foreign_and_truncated_payloads() {
    // A trunk payload (AXKB) must fail at the header check immediately
    // rather than being misparsed as a cursor.
    let trunk = MlxKVCache::new(2).serialize_to_bytes();
    assert!(matches!(
        MlxKVCache::try_deserialize_qwen4_exp_draft_cursor(&trunk),
        Err(MlxKVCacheSerializeError::BadMagic)
    ));

    // Any non-AXMC magic is rejected before any structural parse.
    let mut foreign = Vec::new();
    foreign.extend_from_slice(b"ZZZZ");
    foreign.extend_from_slice(&1u32.to_le_bytes());
    foreign.extend_from_slice(&1u64.to_le_bytes());
    foreign.extend_from_slice(&1u32.to_le_bytes());
    assert!(matches!(
        MlxKVCache::try_deserialize_qwen4_exp_draft_cursor(&foreign),
        Err(MlxKVCacheSerializeError::BadMagic)
    ));

    // Truncated header and truncated body both fail closed.
    assert!(matches!(
        MlxKVCache::try_deserialize_qwen4_exp_draft_cursor(b"AXM"),
        Err(MlxKVCacheSerializeError::UnexpectedEof)
    ));
    let mut truncated = Vec::new();
    truncated.extend_from_slice(b"AXMC");
    truncated.extend_from_slice(&1u32.to_le_bytes());
    assert!(matches!(
        MlxKVCache::try_deserialize_qwen4_exp_draft_cursor(&truncated),
        Err(MlxKVCacheSerializeError::UnexpectedEof)
    ));
}

#[test]
fn serialize_empty_cache_roundtrips() {
    let cache = MlxKVCache::new(3);
    let bytes = cache.serialize_to_bytes();
    let restored = MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip");
    assert_eq!(restored.seq_len, 0);
    assert_eq!(restored.layers.len(), 3);
    assert!(restored.layers.iter().all(Option::is_none));
    assert!(restored.glm_mla_layers.iter().all(Option::is_none));
    assert!(
        restored
            .linear_layers
            .iter()
            .all(|l| l.conv_state.is_none() && l.recurrent_state.is_none())
    );
}

#[test]
fn serialize_fa_cache_roundtrips_values() {
    // Two FA layers, varying head counts to catch shape mistakes.
    let mut cache = MlxKVCache::new(2);
    let seq_len = 4;
    let k0 = build_fa_array_f32(seq_len, 2, 8);
    let v0 = build_fa_array_f32(seq_len, 2, 8);
    let k1 = build_fa_array_f32(seq_len, 4, 16);
    let v1 = build_fa_array_f32(seq_len, 4, 16);
    cache.layers[0] = Some(FaLayerStorage::Contiguous(LayerKV {
        last_k_view: None,
        last_v_view: None,
        n_kv_heads: 2,
        head_dim: 8,
        capacity: seq_len,
        rotating_window: None,
        protected_prefix_ring: None,
        dtype: MlxDtype::Float32,
        k: k0,
        v: v0,
    }));
    cache.layers[1] = Some(FaLayerStorage::Contiguous(LayerKV {
        last_k_view: None,
        last_v_view: None,
        n_kv_heads: 4,
        head_dim: 16,
        capacity: seq_len,
        rotating_window: None,
        protected_prefix_ring: None,
        dtype: MlxDtype::Float32,
        k: k1,
        v: v1,
    }));
    cache.seq_len = seq_len;
    cache.growth_count = 7;

    let bytes = cache.serialize_to_bytes();
    let restored = MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip");

    assert_eq!(restored.seq_len, seq_len);
    assert_eq!(restored.growth_count, 7);
    for layer in 0..2 {
        let orig = contiguous_layer(&cache, layer);
        let back = contiguous_layer(&restored, layer);
        assert_eq!(back.n_kv_heads, orig.n_kv_heads);
        assert_eq!(back.head_dim, orig.head_dim);
        assert_eq!(back.capacity, orig.capacity);
        assert_eq!(back.dtype, orig.dtype);
        assert_eq!(host_f32(&back.k), host_f32(&orig.k));
        assert_eq!(host_f32(&back.v), host_f32(&orig.v));
    }
}

#[test]
fn serialize_mla_cache_roundtrips_values() {
    let mut cache = MlxKVCache::new(1);
    let seq_len = 6;
    let latent_dim = 4;
    let rope_dim = 2;
    let kv_latent = build_mla_latent_f32(seq_len, latent_dim);
    let k_pe = build_mla_latent_f32(seq_len, rope_dim);
    cache.glm_mla_layers[0] = Some(GlmMlaLayerCache {
        latent_dim,
        rope_dim,
        capacity: seq_len,
        dtype: MlxDtype::Float32,
        kv_latent,
        k_pe,
    });
    cache.seq_len = seq_len;

    let bytes = cache.serialize_to_bytes();
    let restored = MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip");

    let orig = cache.glm_mla_layers[0].as_ref().unwrap();
    let back = restored.glm_mla_layers[0]
        .as_ref()
        .expect("mla layer present");
    assert_eq!(back.latent_dim, orig.latent_dim);
    assert_eq!(back.rope_dim, orig.rope_dim);
    assert_eq!(back.capacity, orig.capacity);
    assert_eq!(back.dtype, orig.dtype);
    assert_eq!(host_f32(&back.kv_latent), host_f32(&orig.kv_latent));
    assert_eq!(host_f32(&back.k_pe), host_f32(&orig.k_pe));
}

#[test]
fn deserialize_rejects_bad_magic() {
    let bytes = b"NOPE\x00\x00\x00\x00".to_vec();
    let result = MlxKVCache::try_deserialize_from_bytes(&bytes);
    assert!(matches!(
        result,
        Err(MlxKVCacheSerializeError::BadMagic) | Err(MlxKVCacheSerializeError::UnexpectedEof)
    ));
}

#[test]
fn deserialize_rejects_unsupported_version() {
    let mut payload = MlxKVCache::SERIALIZE_MAGIC.to_vec();
    payload.extend_from_slice(&99u32.to_le_bytes()); // wrong version
    payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
    payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
    payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
    payload.extend_from_slice(&0u32.to_le_bytes()); // layer_count
    payload.extend_from_slice(&0u32.to_le_bytes()); // reserved
    let result = MlxKVCache::try_deserialize_from_bytes(&payload);
    assert!(matches!(
        result,
        Err(MlxKVCacheSerializeError::UnsupportedVersion(99))
    ));
}

#[test]
fn deserialize_rejects_truncated_payload() {
    let cache = MlxKVCache::new(1);
    let bytes = cache.serialize_to_bytes();
    // Cut off the last byte to simulate a torn write.
    let truncated = &bytes[..bytes.len() - 1];
    let result = MlxKVCache::try_deserialize_from_bytes(truncated);
    assert!(matches!(
        result,
        Err(MlxKVCacheSerializeError::UnexpectedEof)
    ));
}

#[test]
fn verify_restored_snapshot_rejects_all_empty_layers() {
    let mut cache = MlxKVCache::new_contiguous(2);
    cache.set_seq_len(8);
    // Round-trip to prove the wire format itself accepts this payload;
    // the structural check is the only thing standing between it and
    // adoption as a valid 8-token prefix.
    let restored = MlxKVCache::try_deserialize_from_bytes(&cache.serialize_to_bytes()).unwrap();
    assert!(matches!(
        restored.verify_restored_snapshot(2, 8, None),
        Err(MlxKVCacheSerializeError::EmptySnapshot)
    ));
}

#[test]
fn verify_restored_snapshot_rejects_layer_count_mismatch() {
    let mut cache = MlxKVCache::new_contiguous(2);
    cache.set_seq_len(8);
    let restored = MlxKVCache::try_deserialize_from_bytes(&cache.serialize_to_bytes()).unwrap();
    assert!(matches!(
        restored.verify_restored_snapshot(4, 8, None),
        Err(MlxKVCacheSerializeError::LayerCountMismatch {
            expected: 4,
            actual: 2
        })
    ));
}

#[test]
fn verify_restored_snapshot_rejects_token_count_mismatch() {
    let mut cache = MlxKVCache::new_contiguous(2);
    cache.set_seq_len(8);
    assert!(matches!(
        cache.verify_restored_snapshot(2, 4, None),
        Err(MlxKVCacheSerializeError::TokenCountMismatch {
            expected: 4,
            actual: 8
        })
    ));
}

#[test]
fn verify_restored_snapshot_rejects_one_sided_linear_state() {
    let mut cache = MlxKVCache::new_contiguous(2);
    cache.set_seq_len(8);
    // Forge a payload whose linear layer carries conv state but no
    // recurrent state — the serializer writes it, the reader accepts
    // it, and only the completeness check rejects it.
    cache.linear_layers[0].conv_state = Some(MlxArray::from_f32_slice(&[0.0]));
    let restored = MlxKVCache::try_deserialize_from_bytes(&cache.serialize_to_bytes()).unwrap();
    assert!(matches!(
        restored.verify_restored_snapshot(2, 8, Some(&[true, false])),
        Err(MlxKVCacheSerializeError::IncompleteLinearLayer(0))
    ));
}

#[test]
fn verify_restored_snapshot_accepts_complete_linear_snapshot() {
    let mut cache = MlxKVCache::new_contiguous(2);
    cache.set_seq_len(8);
    cache.set_linear_state(
        0,
        MlxArray::from_f32_slice(&[0.0]),
        MlxArray::from_f32_slice(&[0.0]),
    );
    let restored = MlxKVCache::try_deserialize_from_bytes(&cache.serialize_to_bytes()).unwrap();
    assert!(
        restored
            .verify_restored_snapshot(2, 8, Some(&[true, false]))
            .is_ok()
    );
}

#[test]
fn deserialize_rejects_an_absurd_layer_count_before_allocating() {
    let mut payload = MlxKVCache::new(2).serialize_to_bytes();
    let layer_count_at = MlxKVCache::SERIALIZE_MAGIC.len() + 4 + 8 + 8 + 8;
    payload[layer_count_at..layer_count_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let err = MlxKVCache::try_deserialize_from_bytes(&payload)
        .err()
        .expect("a corrupted layer count must be rejected");
    assert!(
        matches!(err, MlxKVCacheSerializeError::BadShape(count) if count == u32::MAX as usize),
        "{err:?}"
    );
}

#[test]
fn deserialize_rejects_undersized_byte_count() {
    // Hand-craft a payload whose tensor header declares a shape
    // requiring more bytes than `byte_count` advertises. Without
    // the pre-validation guard, `MlxArray::from_managed_data` would
    // panic; with it, we surface a structured `BadShape` error.
    let mut payload = Vec::new();
    payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
    payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
    payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
    payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
    payload.extend_from_slice(&1u32.to_le_bytes()); // layer_count
    payload.extend_from_slice(&0u32.to_le_bytes()); // reserved

    // Single FA layer
    payload.push(MlxKVCache::LAYER_KIND_FA);
    payload.extend_from_slice(&[0u8; 7]);
    payload.extend_from_slice(&0u64.to_le_bytes()); // rotating_window: none
    // K tensor header: f32, 4-dim shape [1, 2, 4, 8] = 64 elements × 4 bytes
    payload.push(MlxKVCache::dtype_to_tag(MlxDtype::Float32));
    payload.push(4);
    payload.extend_from_slice(&[0u8; 6]);
    payload.extend_from_slice(&1i32.to_le_bytes());
    payload.extend_from_slice(&2i32.to_le_bytes());
    payload.extend_from_slice(&4i32.to_le_bytes());
    payload.extend_from_slice(&8i32.to_le_bytes());
    // Declared byte_count = 1 (too small for the declared shape)
    payload.extend_from_slice(&1u64.to_le_bytes());
    payload.push(0u8);

    let err = MlxKVCache::try_deserialize_from_bytes(&payload)
        .err()
        .expect("undersized byte_count must be rejected");
    assert!(
        matches!(err, MlxKVCacheSerializeError::BadShape(4)),
        "expected BadShape(4), got {err:?}"
    );
}

#[test]
fn deserialize_rejects_oversized_byte_count() {
    // A huge declared byte_count must fail closed before allocation.
    let mut payload = Vec::new();
    payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
    payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
    payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
    payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
    payload.extend_from_slice(&1u32.to_le_bytes()); // layer_count
    payload.extend_from_slice(&0u32.to_le_bytes()); // reserved

    payload.push(MlxKVCache::LAYER_KIND_FA);
    payload.extend_from_slice(&[0u8; 7]);
    payload.extend_from_slice(&0u64.to_le_bytes()); // rotating_window: none
    // K tensor: f32, shape [1,1,1,1] = 4 bytes required
    payload.push(MlxKVCache::dtype_to_tag(MlxDtype::Float32));
    payload.push(4);
    payload.extend_from_slice(&[0u8; 6]);
    payload.extend_from_slice(&1i32.to_le_bytes());
    payload.extend_from_slice(&1i32.to_le_bytes());
    payload.extend_from_slice(&1i32.to_le_bytes());
    payload.extend_from_slice(&1i32.to_le_bytes());
    // Declared byte_count far larger than required (would OOM/abort without the check)
    payload.extend_from_slice(&(1u64 << 40).to_le_bytes());

    let err = MlxKVCache::try_deserialize_from_bytes(&payload)
        .err()
        .expect("oversized byte_count must be rejected before allocation");
    assert!(
        matches!(err, MlxKVCacheSerializeError::BadShape(4)),
        "expected BadShape(4), got {err:?}"
    );
}

#[test]
fn deserialize_rejects_unknown_dtype_tag() {
    // Hand-craft a payload whose tensor header carries a dtype tag that
    // is not in `dtype_from_tag`'s match table. The deserializer must
    // fail-close (PRD §7.1 "per-layer cache type, tensor shape, dtype")
    // rather than silently accept an unknown dtype and trip MLX later.
    let mut payload = Vec::new();
    payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
    payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
    payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
    payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
    payload.extend_from_slice(&1u32.to_le_bytes()); // layer_count
    payload.extend_from_slice(&0u32.to_le_bytes()); // reserved

    // Single FA layer with one tensor carrying an invalid dtype.
    payload.push(MlxKVCache::LAYER_KIND_FA);
    payload.extend_from_slice(&[0u8; 7]);
    payload.extend_from_slice(&0u64.to_le_bytes()); // rotating_window: none
    // 0xEE is not a valid dtype tag in dtype_from_tag's table.
    payload.push(0xEE);
    payload.push(4); // ndim
    payload.extend_from_slice(&[0u8; 6]); // reserved
    payload.extend_from_slice(&1i32.to_le_bytes());
    payload.extend_from_slice(&1i32.to_le_bytes());
    payload.extend_from_slice(&1i32.to_le_bytes());
    payload.extend_from_slice(&1i32.to_le_bytes());
    payload.extend_from_slice(&4u64.to_le_bytes()); // byte_count

    let err = MlxKVCache::try_deserialize_from_bytes(&payload)
        .err()
        .expect("unknown dtype tag must be rejected");
    assert!(
        matches!(err, MlxKVCacheSerializeError::UnknownDtype(0xEE)),
        "expected UnknownDtype(0xEE), got {err:?}"
    );
}

#[test]
fn deserialize_rejects_unknown_layer_kind() {
    // PRD §7.1 requires per-layer cache type validation. An unknown
    // discriminator byte at the layer header position must fail-close.
    let mut payload = Vec::new();
    payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
    payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
    payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
    payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
    payload.extend_from_slice(&1u32.to_le_bytes()); // layer_count
    payload.extend_from_slice(&0u32.to_le_bytes()); // reserved

    // 0x7F is intentionally outside the four documented layer kinds
    // (EMPTY/FA/MLA/LINEAR) but inside `u8`'s range.
    payload.push(0x7F);
    payload.extend_from_slice(&[0u8; 7]);

    let err = MlxKVCache::try_deserialize_from_bytes(&payload)
        .err()
        .expect("unknown layer kind must be rejected");
    assert!(
        matches!(err, MlxKVCacheSerializeError::UnknownLayerKind(0x7F)),
        "expected UnknownLayerKind(0x7F), got {err:?}"
    );
}

#[test]
fn deserialize_rejects_zero_rank_tensor() {
    // Tensor rank 0 is never valid for an FA layer; the deserializer's
    // shape guard (`ndim == 0 || ndim > 4`) must reject before any
    // MlxArray construction is attempted. This is the dtype-aware
    // complement to `deserialize_rejects_undersized_byte_count`.
    let mut payload = Vec::new();
    payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
    payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());

    payload.push(MlxKVCache::LAYER_KIND_FA);
    payload.extend_from_slice(&[0u8; 7]);
    payload.extend_from_slice(&0u64.to_le_bytes()); // rotating_window: none
    payload.push(MlxKVCache::dtype_to_tag(MlxDtype::Float32));
    payload.push(0); // ndim = 0 (invalid)
    payload.extend_from_slice(&[0u8; 6]);
    payload.extend_from_slice(&[0u8; 16]); // 4 i32 shape entries
    payload.extend_from_slice(&0u64.to_le_bytes());

    let err = MlxKVCache::try_deserialize_from_bytes(&payload)
        .err()
        .expect("rank 0 tensor must be rejected");
    assert!(
        matches!(err, MlxKVCacheSerializeError::BadShape(0)),
        "expected BadShape(0), got {err:?}"
    );
}

#[test]
fn serialize_trims_fa_capacity_to_logical_seq_len() {
    // The backing buffer holds `capacity` tokens but only `seq_len`
    // are logical; the payload must carry the logical prefix only.
    let capacity = 8usize;
    let seq_len = 3usize;
    let head_dim = 4;
    let n_kv_heads = 2;
    let mut cache = MlxKVCache::new(1);
    let k = build_fa_array_f32(capacity, n_kv_heads, head_dim);
    let v = build_fa_array_f32(capacity, n_kv_heads, head_dim);
    // The slice is a strided view; materialize it before reading host
    // bytes (data_f32 on a strided view walks the backing buffer
    // linearly — the same hazard serialize_tensor_logical guards).
    let expected_k = host_f32(&contiguous(
        &slice(
            &k,
            &[0, 0, 0, 0],
            &[1, n_kv_heads, seq_len as i32, head_dim],
            &[1, 1, 1, 1],
            None,
        ),
        None,
    ));
    cache.layers[0] = Some(FaLayerStorage::Contiguous(LayerKV {
        last_k_view: None,
        last_v_view: None,
        n_kv_heads,
        head_dim,
        capacity,
        rotating_window: None,
        protected_prefix_ring: None,
        dtype: MlxDtype::Float32,
        k,
        v,
    }));
    cache.seq_len = seq_len;

    let trimmed_bytes = cache.serialize_to_bytes();
    let restored = MlxKVCache::try_deserialize_from_bytes(&trimmed_bytes).expect("round-trip");
    assert_eq!(restored.seq_len, seq_len);
    let restored_layer = contiguous_layer(&restored, 0);
    assert_eq!(
        restored_layer.capacity, seq_len,
        "restored capacity must equal the logical length, not the source capacity"
    );
    assert_eq!(
        host_f32(&restored_layer.k),
        expected_k,
        "restored K must match the logical prefix of the source buffer"
    );

    // Payload must scale with seq_len, not capacity: the same cache
    // reporting full capacity as logical length serializes strictly more.
    cache.seq_len = capacity;
    let full_bytes = cache.serialize_to_bytes();
    assert!(
        trimmed_bytes.len() < full_bytes.len(),
        "trimmed payload ({}) must be smaller than full-capacity payload ({})",
        trimmed_bytes.len(),
        full_bytes.len()
    );
}

#[test]
fn serialize_materializes_exact_length_strided_fa_views() {
    let backing_tokens = 8usize;
    let seq_len = 3usize;
    let head_dim = 4;
    let n_kv_heads = 2;
    let base_k = build_fa_array_f32(backing_tokens, n_kv_heads, head_dim);
    let base_v = build_fa_array_f32(backing_tokens, n_kv_heads, head_dim);
    let stop = [1, n_kv_heads, seq_len as i32, head_dim];
    let k = slice(&base_k, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None);
    let v = slice(&base_v, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None);
    let expected_k = host_f32(&contiguous(&k, None));

    let mut cache = MlxKVCache::new_contiguous(1);
    cache.layers[0] = Some(FaLayerStorage::Contiguous(LayerKV {
        last_k_view: None,
        last_v_view: None,
        n_kv_heads,
        head_dim,
        capacity: seq_len,
        rotating_window: None,
        protected_prefix_ring: None,
        dtype: MlxDtype::Float32,
        k,
        v,
    }));
    cache.seq_len = seq_len;

    let restored = MlxKVCache::try_deserialize_from_bytes(&cache.serialize_to_bytes())
        .expect("strided exact-length round-trip");
    assert_eq!(host_f32(&contiguous_layer(&restored, 0).k), expected_k,);
}

#[test]
fn serialize_roundtrips_rope_offset() {
    let mut cache = MlxKVCache::new(1);
    cache.seq_len = 4;
    cache.rope_offset = 9;

    let restored =
        MlxKVCache::try_deserialize_from_bytes(&cache.serialize_to_bytes()).expect("round-trip");

    assert_eq!(restored.seq_len, 4);
    assert_eq!(
        restored.rope_offset, 9,
        "rope_offset is positional state and must survive the round trip"
    );
}

#[test]
fn serialize_roundtrips_rotated_ring_geometry_and_appends_continue() {
    const HD: usize = 4;
    // Build a bounded ring exactly like the rotating-decode tests:
    // window 4, slack 3 → capacity 7, tokens 0..=8 with 7 and 8 wrapped.
    let mut cache = MlxKVCache::new(1);
    cache.set_rotating_sliding_decode(true);
    cache.set_rotating_sliding_slack(3);
    let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
    let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
    cache.append(0, k, v);
    cache.seq_len = 4;
    for (t, value) in [(4usize, 5.0f32), (5, 6.0), (6, 7.0), (7, 8.0), (8, 9.0)] {
        let k = tokens_f32(&[value], HD);
        let v = tokens_f32(&[value], HD);
        cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = t + 1;
    }
    let expected_slots = token_row_values(&contiguous_layer(&cache, 0).k, HD);
    assert_eq!(expected_slots, vec![8.0, 9.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    let restored =
        MlxKVCache::try_deserialize_from_bytes(&cache.serialize_to_bytes()).expect("round-trip");
    let lkv = contiguous_layer(&restored, 0);
    assert_eq!(lkv.rotating_window, Some(4), "ring window must survive");
    assert_eq!(lkv.capacity, 7, "ring capacity must survive");
    assert_eq!(restored.rotating_sliding_slack(), 3, "slack re-latched");
    assert_eq!(
        token_row_values(&lkv.k, HD),
        expected_slots,
        "slot-ordered ring contents must survive byte-identical"
    );

    // Post-restore decode must keep rotating: token 9 lands at slot
    // 9 % 7 = 2, not at logical position 9 of an ordered buffer.
    let mut restored = restored;
    let k = tokens_f32(&[10.0], HD);
    let v = tokens_f32(&[10.0], HD);
    restored.append_with_retained_window(0, k, v, Some(4));
    restored.seq_len = 10;
    let lkv = contiguous_layer(&restored, 0);
    assert_eq!(lkv.capacity, 7, "restored ring must not regrow");
    assert_eq!(
        token_row_values(&lkv.k, HD),
        vec![8.0, 9.0, 10.0, 4.0, 5.0, 6.0, 7.0]
    );
}

#[test]
#[should_panic(expected = "ordered KV append on rotated ring layer")]
fn ordered_append_on_rotated_ring_fails_closed() {
    const HD: usize = 4;
    let mut cache = MlxKVCache::new(1);
    cache.set_rotating_sliding_decode(true);
    let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
    let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
    cache.append(0, k, v);
    cache.seq_len = 4;
    // Convert to a pure ring (window 4, slack 0).
    let k = tokens_f32(&[5.0], HD);
    let v = tokens_f32(&[5.0], HD);
    cache.append_with_retained_window(0, k, v, Some(4));
    cache.seq_len = 5;

    // A 2-token forward is not ring-eligible in pure mode; before the
    // fail-closed assert this fell through to the ordered path and
    // silently grew the ring, copying slots as a token-ordered prefix.
    let k = tokens_f32(&[6.0, 7.0], HD);
    let v = tokens_f32(&[6.0, 7.0], HD);
    let _ = cache.append_with_retained_window(0, k, v, Some(4));
}

#[test]
fn deserialized_cache_outlives_input_buffer() {
    // Regression test for the lifetime bug fixed alongside this
    // commit: `from_raw_data` borrows its data pointer (per the
    // mlx-sys array.rs:80 doc, "MLX does **not** copy"), so handing
    // it a slice of the caller's input buffer would leave the
    // deserialised array dangling once that buffer is freed.
    // `try_deserialize_from_bytes` must construct arrays that own
    // their data via `from_managed_data` + a heap-Box deleter.
    //
    // This test arranges the scenario explicitly: build an input
    // buffer, deserialise from it, drop the buffer, then read the
    // cache's tensors. With the fix, every read returns the
    // original byte pattern; without the fix this exhibits
    // undefined behaviour (typically reads bogus values or
    // SIGBUS / SIGSEGV under MLX's evaluator).
    let seq_len = 4;
    let head_dim = 8;
    let n_kv_heads = 2;
    let original = {
        let mut cache = MlxKVCache::new(1);
        let k = build_fa_array_f32(seq_len, n_kv_heads, head_dim);
        let v = build_fa_array_f32(seq_len, n_kv_heads, head_dim);
        cache.layers[0] = Some(FaLayerStorage::Contiguous(LayerKV {
            last_k_view: None,
            last_v_view: None,
            n_kv_heads,
            head_dim,
            capacity: seq_len,
            rotating_window: None,
            protected_prefix_ring: None,
            dtype: MlxDtype::Float32,
            k,
            v,
        }));
        cache.seq_len = seq_len;
        cache
    };
    let expected_k = host_f32(&contiguous_layer(&original, 0).k);
    let expected_v = host_f32(&contiguous_layer(&original, 0).v);

    let restored = {
        let bytes = original.serialize_to_bytes();
        MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip")
        // `bytes` drops here. The restored cache must remain valid.
    };

    // Read the restored tensors AFTER the input buffer has been
    // dropped. If `read_tensor` had borrowed the slice, this would
    // be UB; the managed-data + heap-owned pattern keeps it sound.
    let restored_k = host_f32(&contiguous_layer(&restored, 0).k);
    let restored_v = host_f32(&contiguous_layer(&restored, 0).v);
    assert_eq!(restored_k, expected_k);
    assert_eq!(restored_v, expected_v);
}

// ── PR4 FA block-pool path: token-exact oracle vs contiguous ──

fn fa_token_values(seq_len: usize, n_kv_heads: i32, head_dim: i32, base: f32) -> MlxArray {
    let total = (n_kv_heads as usize) * seq_len * (head_dim as usize);
    let data: Vec<f32> = (0..total).map(|i| base + (i as f32) * 0.01).collect();
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data.as_slice()),
        &[1, n_kv_heads, seq_len as i32, head_dim],
        MlxDtype::Float32,
    )
}

#[test]
fn fa_paged_append_trim_oracle_matches_contiguous() {
    // block_size=4 so multi-block growth + partial last block exercise
    // materialize and free_blocks_beyond.
    let config = FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 32,
        hard_cap: false,
    };
    let mut paged = MlxKVCache::new_with_fa_block_pool(1, config);
    let mut contig = MlxKVCache::new_contiguous(1);
    assert!(paged.fa_block_pool_enabled());
    assert!(!contig.fa_block_pool_enabled());

    let steps: &[(usize, f32)] = &[(3, 1.0), (5, 2.0), (1, 3.0), (6, 4.0)];
    let n_kv_heads = 2i32;
    let head_dim = 4i32;
    let mut seq = 0usize;
    for &(n, base) in steps {
        let k = fa_token_values(n, n_kv_heads, head_dim, base);
        let v = fa_token_values(n, n_kv_heads, head_dim, base + 0.5);
        let (pk, pv) = paged.append(0, k.clone(), v.clone());
        let (ck, cv) = contig.append(0, k, v);
        seq += n;
        paged.advance(n);
        contig.advance(n);
        eval(&[&pk, &pv, &ck, &cv]);
        assert_eq!(
            host_f32(&pk),
            host_f32(&ck),
            "K mismatch after append to {seq}"
        );
        assert_eq!(
            host_f32(&pv),
            host_f32(&cv),
            "V mismatch after append to {seq}"
        );
        assert_eq!(pk.shape(), ck.shape());
    }

    // Trim into the middle of a block; trailing full blocks free.
    assert!(paged.trim_to(7));
    assert!(contig.trim_to(7));
    let (pk, pv) = paged.logical_layer_kv(0).expect("paged layer");
    let (ck, cv) = contig.logical_layer_kv(0).expect("contig layer");
    eval(&[&pk, &pv, &ck, &cv]);
    assert_eq!(host_f32(&pk), host_f32(&ck), "K mismatch after trim_to(7)");
    assert_eq!(host_f32(&pv), host_f32(&cv), "V mismatch after trim_to(7)");

    // Re-append after trim overwrites the trimmed region.
    let k = fa_token_values(3, n_kv_heads, head_dim, 9.0);
    let v = fa_token_values(3, n_kv_heads, head_dim, 9.5);
    let (pk, pv) = paged.append(0, k.clone(), v.clone());
    let (ck, cv) = contig.append(0, k, v);
    paged.advance(3);
    contig.advance(3);
    eval(&[&pk, &pv, &ck, &cv]);
    assert_eq!(host_f32(&pk), host_f32(&ck), "K mismatch after re-append");
    assert_eq!(host_f32(&pv), host_f32(&cv), "V mismatch after re-append");

    // Serialize materializes dense; round-trip matches contiguous values.
    let paged_bytes = paged.serialize_to_bytes();
    let contig_bytes = contig.serialize_to_bytes();
    // Growth counts may differ (block vs chunk grow), so compare tensors
    // rather than full wire equality.
    let p_restored = MlxKVCache::try_deserialize_from_bytes(&paged_bytes).expect("paged serialize");
    let c_restored =
        MlxKVCache::try_deserialize_from_bytes(&contig_bytes).expect("contig serialize");
    assert_eq!(p_restored.seq_len(), c_restored.seq_len());
    assert_eq!(
        host_f32(&contiguous_layer(&p_restored, 0).k),
        host_f32(&contiguous_layer(&c_restored, 0).k)
    );
    assert_eq!(
        host_f32(&contiguous_layer(&p_restored, 0).v),
        host_f32(&contiguous_layer(&c_restored, 0).v)
    );
    assert!(!p_restored.fa_block_pool_enabled());
}

#[test]
fn fa_paged_pool_exhaustion_demotes_to_contiguous() {
    let config = FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 2, // only 8 tokens of private capacity
        hard_cap: false,
    };
    let mut cache = MlxKVCache::new_with_fa_block_pool(1, config);
    let k = fa_token_values(8, 1, 4, 1.0);
    let v = fa_token_values(8, 1, 4, 2.0);
    let _ = cache.append(0, k, v);
    cache.advance(8);
    assert_eq!(cache.fa_block_pool_available(), Some(0));

    let k2 = fa_token_values(1, 1, 4, 3.0);
    let v2 = fa_token_values(1, 1, 4, 4.0);
    let (out_k, _) = cache.append(0, k2, v2);
    cache.advance(1);
    eval(&[&out_k]);
    assert_eq!(out_k.shape()[2], 9);
    let usage = cache.usage_snapshot();
    assert_eq!(
        usage.paged_pool_exhaustion_fallbacks, 1,
        "pool exhaustion must demote rather than panic"
    );
    // Layer is now contiguous; further appends stay contiguous.
    assert!(matches!(
        cache.layers[0],
        Some(FaLayerStorage::Contiguous(_))
    ));
}

/// Regression for the shared-pool sizing hazard: `FaBlockPool` capacity is
/// one budget shared by every pure-FA layer in the cache (PR4 scope), not
/// a per-layer slab. A layer that exhausts the shared pool must demote to
/// contiguous storage on its own without disturbing sibling layers that
/// already hold private blocks from the same pool.
#[test]
fn fa_paged_pool_shared_across_layers_demotes_independently() {
    let config = FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 3, // 12 tokens shared across both layers
        hard_cap: false,
    };
    let mut cache = MlxKVCache::new_with_fa_block_pool(2, config);

    // Layer 0 claims 2 of the 3 blocks (8 tokens), leaving 1 block free.
    let k0 = fa_token_values(8, 1, 4, 1.0);
    let v0 = fa_token_values(8, 1, 4, 2.0);
    let _ = cache.append(0, k0, v0);

    // Layer 1 needs 2 blocks (8 tokens) but only 1 remains in the shared
    // pool: this must demote layer 1, not corrupt or evict layer 0.
    let k1 = fa_token_values(8, 1, 4, 10.0);
    let v1 = fa_token_values(8, 1, 4, 20.0);
    let (out_k1, out_v1) = cache.append(1, k1, v1);
    cache.advance(8);
    eval(&[&out_k1, &out_v1]);

    let usage = cache.usage_snapshot();
    assert_eq!(
        usage.paged_pool_exhaustion_fallbacks, 1,
        "layer 1 must demote exactly once when the shared pool runs out"
    );
    assert!(
        matches!(cache.layers[0], Some(FaLayerStorage::Paged(_))),
        "layer 0 must remain paged; it did not exhaust the pool"
    );
    assert!(
        matches!(cache.layers[1], Some(FaLayerStorage::Contiguous(_))),
        "layer 1 must demote to contiguous once its own append exhausts the shared pool"
    );

    // Layer 0's data must be intact after layer 1's demotion freed blocks
    // from the same shared pool.
    let (layer0_k, layer0_v) = cache.peek_layer_full_kv(0).expect("layer 0 still resident");
    eval(&[&layer0_k, &layer0_v]);
    let expected_k = fa_token_values(8, 1, 4, 1.0);
    let expected_v = fa_token_values(8, 1, 4, 2.0);
    eval(&[&expected_k, &expected_v]);
    assert_eq!(layer0_k.data_f32(), expected_k.data_f32());
    assert_eq!(layer0_v.data_f32(), expected_v.data_f32());
}

/// Token-exact oracle for the exhaustion→demotion path itself, not just
/// shape/counters (`fa_paged_pool_exhaustion_demotes_to_contiguous`) or a
/// sibling layer's untouched values
/// (`fa_paged_pool_shared_across_layers_demotes_independently`).
/// `fa_paged_append_trim_oracle_matches_contiguous`'s pool is oversized
/// and never demotes, so it never exercises this path either. This test
/// forces demotion mid-sequence and then keeps comparing the demoted
/// layer's own output — across further appends and a trim after
/// demotion — against a contiguous-only cache fed the identical logical
/// token sequence.
#[test]
fn fa_paged_pool_exhaustion_demotion_matches_contiguous_oracle() {
    let config = FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 2, // 8 tokens of private capacity; the 2nd append exhausts it
        hard_cap: false,
    };
    let mut paged = MlxKVCache::new_with_fa_block_pool(1, config);
    let mut contig = MlxKVCache::new_contiguous(1);
    let n_kv_heads = 1i32;
    let head_dim = 4i32;

    // Fills the private pool exactly; no demotion yet.
    let steps_before_demotion: &[(usize, f32)] = &[(8, 1.0)];
    // write_start=8 needs a 3rd block the pool doesn't have: demotes.
    let demoting_step: (usize, f32) = (1, 3.0);
    // Continues on the now-contiguous layer after demotion.
    let steps_after_demotion: &[(usize, f32)] = &[(5, 5.0), (2, 7.0)];

    let mut seq = 0usize;
    for &(n, base) in steps_before_demotion {
        let k = fa_token_values(n, n_kv_heads, head_dim, base);
        let v = fa_token_values(n, n_kv_heads, head_dim, base + 0.5);
        let _ = paged.append(0, k.clone(), v.clone());
        let _ = contig.append(0, k, v);
        paged.advance(n);
        contig.advance(n);
        seq += n;
    }
    assert_eq!(paged.fa_block_pool_available(), Some(0));

    let (n, base) = demoting_step;
    let k = fa_token_values(n, n_kv_heads, head_dim, base);
    let v = fa_token_values(n, n_kv_heads, head_dim, base + 0.5);
    let _ = paged.append(0, k.clone(), v.clone());
    let _ = contig.append(0, k, v);
    paged.advance(n);
    contig.advance(n);
    seq += n;
    let usage = paged.usage_snapshot();
    assert_eq!(
        usage.paged_pool_exhaustion_fallbacks, 1,
        "this step must actually trigger demotion, or the test proves nothing"
    );
    assert!(
        matches!(paged.layers[0], Some(FaLayerStorage::Contiguous(_))),
        "layer must be demoted to contiguous storage after exhaustion"
    );

    // Value-check right at the demotion boundary, before further appends
    // can paper over a corrupted materialize.
    {
        let (pk, pv) = paged.logical_layer_kv(0).expect("demoted layer");
        let (ck, cv) = contig.logical_layer_kv(0).expect("contig layer");
        eval(&[&pk, &pv, &ck, &cv]);
        assert_eq!(
            host_f32(&pk),
            host_f32(&ck),
            "K mismatch right after demotion"
        );
        assert_eq!(
            host_f32(&pv),
            host_f32(&cv),
            "V mismatch right after demotion"
        );
        assert_eq!(pk.shape(), ck.shape());
    }

    for &(n, base) in steps_after_demotion {
        let k = fa_token_values(n, n_kv_heads, head_dim, base);
        let v = fa_token_values(n, n_kv_heads, head_dim, base + 0.5);
        let (pk, pv) = paged.append(0, k.clone(), v.clone());
        let (ck, cv) = contig.append(0, k, v);
        paged.advance(n);
        contig.advance(n);
        seq += n;
        eval(&[&pk, &pv, &ck, &cv]);
        assert_eq!(
            host_f32(&pk),
            host_f32(&ck),
            "K mismatch after append to {seq}"
        );
        assert_eq!(
            host_f32(&pv),
            host_f32(&cv),
            "V mismatch after append to {seq}"
        );
    }

    // Trim into the post-demotion buffer; re-append across the trim
    // boundary, mirroring the non-demoted oracle test.
    let trim_len = seq - 4;
    assert!(paged.trim_to(trim_len));
    assert!(contig.trim_to(trim_len));
    let (pk, pv) = paged.logical_layer_kv(0).expect("demoted layer");
    let (ck, cv) = contig.logical_layer_kv(0).expect("contig layer");
    eval(&[&pk, &pv, &ck, &cv]);
    assert_eq!(
        host_f32(&pk),
        host_f32(&ck),
        "K mismatch after trim_to({trim_len})"
    );
    assert_eq!(
        host_f32(&pv),
        host_f32(&cv),
        "V mismatch after trim_to({trim_len})"
    );

    let k = fa_token_values(3, n_kv_heads, head_dim, 42.0);
    let v = fa_token_values(3, n_kv_heads, head_dim, 42.5);
    let (pk, pv) = paged.append(0, k.clone(), v.clone());
    let (ck, cv) = contig.append(0, k, v);
    paged.advance(3);
    contig.advance(3);
    eval(&[&pk, &pv, &ck, &cv]);
    assert_eq!(
        host_f32(&pk),
        host_f32(&ck),
        "K mismatch after re-append past trim"
    );
    assert_eq!(
        host_f32(&pv),
        host_f32(&cv),
        "V mismatch after re-append past trim"
    );
    assert_eq!(pk.shape(), ck.shape());
}

#[test]
fn fa_paged_pool_hard_cap_exhaustion_sticks_flag() {
    // Operator-set AX_MLX_FA_KV_BLOCK_POOL_MAX_BLOCKS must fail closed:
    // exhaustion under `hard_cap: true` sticks hard_cap_exhausted so the
    // runner can fail the request, instead of silently succeeding like
    // the default fail-soft path.
    let config = FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 2, // 8 tokens of private capacity
        hard_cap: true,
    };
    let mut cache = MlxKVCache::new_with_fa_block_pool(1, config);
    assert!(!cache.hard_cap_exhausted());

    let k = fa_token_values(8, 1, 4, 1.0);
    let v = fa_token_values(8, 1, 4, 2.0);
    let _ = cache.append(0, k, v);
    cache.advance(8);
    assert!(
        !cache.hard_cap_exhausted(),
        "exactly filling capacity is not exhaustion"
    );

    let k2 = fa_token_values(1, 1, 4, 3.0);
    let v2 = fa_token_values(1, 1, 4, 4.0);
    let (out_k, _) = cache.append(0, k2, v2);
    cache.advance(1);
    eval(&[&out_k]);
    assert_eq!(
        out_k.shape()[2],
        9,
        "hard-cap demotion still produces a correct (if soon-to-be-\
             discarded) forward, matching the fail-soft materialize path"
    );
    assert!(
        cache.hard_cap_exhausted(),
        "exhaustion under an explicit hard cap must stick the flag"
    );
    let usage = cache.usage_snapshot();
    assert_eq!(usage.paged_pool_exhaustion_fallbacks, 1);
}

#[test]
fn fa_paged_pool_exhaustion_without_hard_cap_does_not_stick_flag() {
    // Regression: the default (no explicit operator override) scaffold
    // behavior must remain fail-soft and never set hard_cap_exhausted.
    let config = FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 2,
        hard_cap: false,
    };
    let mut cache = MlxKVCache::new_with_fa_block_pool(1, config);
    let k = fa_token_values(8, 1, 4, 1.0);
    let v = fa_token_values(8, 1, 4, 2.0);
    let _ = cache.append(0, k, v);
    cache.advance(8);
    let k2 = fa_token_values(1, 1, 4, 3.0);
    let v2 = fa_token_values(1, 1, 4, 4.0);
    let _ = cache.append(0, k2, v2);
    cache.advance(1);
    assert!(!cache.hard_cap_exhausted());
    assert_eq!(cache.usage_snapshot().paged_pool_exhaustion_fallbacks, 1);
}

#[test]
fn fa_paged_clone_diverges_without_double_free() {
    let config = FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 16,
        hard_cap: false,
    };
    let mut a = MlxKVCache::new_with_fa_block_pool(1, config);
    let k = fa_token_values(5, 1, 4, 1.0);
    let v = fa_token_values(5, 1, 4, 2.0);
    let _ = a.append(0, k, v);
    a.advance(5);
    let pool = a.fa_pool.as_ref().expect("paged pool").clone();
    assert_eq!(pool.snapshot().allocated_blocks, 2);
    let b = a.clone();
    assert!(a.shares_fa_block_pool_with(&b));
    assert!(a.is_native_fa_shareable());
    assert!(b.is_native_fa_shareable());
    assert_eq!(pool.snapshot().allocated_blocks, 2);
    assert_eq!(pool.snapshot().shared_blocks, 2);
    assert_eq!(a.additional_fa_blocks_for_append(1), Some(1));
    assert!(a.trim_to(2));
    // Clone still owns both blocks after the source releases its tail.
    assert_eq!(pool.snapshot().allocated_blocks, 2);
    assert_eq!(pool.snapshot().shared_blocks, 1);
    assert_eq!(a.additional_fa_blocks_for_append(1), Some(1));
    let (bk, _) = b.logical_layer_kv(0).expect("clone layer");
    eval(&[&bk]);
    let b_before = host_f32(&bk);
    assert_eq!(bk.shape()[2], 5);
    let k2 = fa_token_values(1, 1, 4, 9.0);
    let v2 = fa_token_values(1, 1, 4, 9.5);
    let (ak, _) = a.append(0, k2, v2);
    a.advance(1);
    eval(&[&ak]);
    assert_eq!(ak.shape()[2], 3);
    assert_eq!(pool.snapshot().allocated_blocks, 3);
    assert_eq!(pool.snapshot().shared_blocks, 0);
    assert_eq!(a.usage_snapshot().paged_cow_copies, 1);
    let (bk_after, _) = b.logical_layer_kv(0).expect("clone layer after divergence");
    eval(&[&bk_after]);
    assert_eq!(host_f32(&bk_after), b_before);

    // A dense compiled/speculative replacement releases only this view's
    // paged IDs and becomes ineligible for native physical adoption.
    let dense_k = fa_token_values(3, 1, 4, 4.0);
    let dense_v = fa_token_values(3, 1, 4, 5.0);
    a.set_layer_kv_logical(0, dense_k, dense_v, 3);
    assert!(!a.is_native_fa_shareable());
    assert_eq!(pool.snapshot().allocated_blocks, 2);

    // Drop both; every reference returns to the one shared pool exactly once.
    drop(a);
    assert_eq!(pool.snapshot().allocated_blocks, 2);
    drop(b);
    assert_eq!(pool.snapshot().allocated_blocks, 0);
}

#[test]
fn dense_standard_fa_snapshot_repages_transactionally() {
    let mut dense = MlxKVCache::new_contiguous(2);
    for layer in 0..2 {
        let k = fa_token_values(5, 1, 4, 1.0 + layer as f32);
        let v = fa_token_values(5, 1, 4, 3.0 + layer as f32);
        let _ = dense.append(layer, k, v);
    }
    dense.advance(5);
    let restored =
        MlxKVCache::try_deserialize_from_bytes(&dense.serialize_to_bytes()).expect("dense restore");

    let too_small = SharedFaBlockPool::new(FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 3,
        hard_cap: true,
    })
    .expect("small pool");
    assert_eq!(restored.fa_blocks_required_for_repage(&too_small), Ok(4));
    assert!(matches!(
        restored.clone_repage_into_shared_fa_pool(too_small.clone()),
        Err(FaBlockPoolError::Exhausted {
            requested: 4,
            available: 3
        })
    ));
    assert_eq!(too_small.snapshot().allocated_blocks, 0);

    let pool = SharedFaBlockPool::new(FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 4,
        hard_cap: true,
    })
    .expect("exact pool");
    let mut restored = restored;
    restored.mrope_position_delta = -3;
    let repaged = restored
        .clone_repage_into_shared_fa_pool(pool.clone())
        .expect("repage");
    assert!(repaged.is_native_fa_shareable());
    assert_eq!(pool.snapshot().allocated_blocks, 4);
    assert_eq!(
        repaged.mrope_position_delta, -3,
        "multimodal RoPE state must survive the repage"
    );
    for layer in 0..2 {
        let (dense_k, dense_v) = restored.logical_layer_kv(layer).expect("dense layer");
        let (paged_k, paged_v) = repaged.logical_layer_kv(layer).expect("paged layer");
        eval(&[&dense_k, &dense_v, &paged_k, &paged_v]);
        assert_eq!(host_f32(&paged_k), host_f32(&dense_k));
        assert_eq!(host_f32(&paged_v), host_f32(&dense_v));
    }
    drop(repaged);
    assert_eq!(pool.snapshot().allocated_blocks, 0);
}

#[test]
fn fixed_slab_clone_cow_preserves_source_and_releases_every_block() {
    let pool = SharedFaBlockPool::new_with_native_slab_storage(FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 16,
        hard_cap: true,
    })
    .expect("central pool");
    let mut branch = MlxKVCache::new_with_shared_fa_block_pool(1, pool.clone());
    let initial_k = fa_token_values(5, 1, 4, 1.0);
    let initial_v = fa_token_values(5, 1, 4, 2.0);
    let _ = branch.append(0, initial_k, initial_v);
    branch.advance(5);
    assert!(branch.is_native_fa_shareable());
    assert_eq!(pool.snapshot().allocated_blocks, 2);

    let source = branch.clone();
    assert!(source.is_native_fa_shareable());
    assert_eq!(pool.snapshot().shared_blocks, 2);
    let (source_k_before, source_v_before) =
        source.logical_layer_kv(0).expect("source central layer");
    eval(&[&source_k_before, &source_v_before]);
    let source_k_before = host_f32(&source_k_before);
    let source_v_before = host_f32(&source_v_before);

    let next_k = fa_token_values(1, 1, 4, 9.0);
    let next_v = fa_token_values(1, 1, 4, 9.5);
    let attention = branch.append_with_retained_window_for_attention(0, next_k, next_v, None);
    branch.advance(1);
    let MlxAttentionKv::Paged(view) = attention else {
        panic!("eligible single-token central append must return a block-table view");
    };
    assert_eq!(view.key_len, 6);
    assert_eq!(view.block_ids.len(), 2);
    let (branch_k, branch_v) = view.materialize();
    eval(&[&branch_k, &branch_v]);
    assert_eq!(branch_k.shape(), vec![1, 1, 6, 4]);
    assert_eq!(
        &host_f32(&branch_k)[20..],
        fa_token_values(1, 1, 4, 9.0).data_f32()
    );
    assert_eq!(
        &host_f32(&branch_v)[20..],
        fa_token_values(1, 1, 4, 9.5).data_f32()
    );
    assert_eq!(branch.usage_snapshot().paged_cow_copies, 1);
    assert_eq!(pool.snapshot().allocated_blocks, 3);
    assert_eq!(pool.snapshot().shared_blocks, 1);

    let (source_k_after, source_v_after) =
        source.logical_layer_kv(0).expect("source after branch COW");
    eval(&[&source_k_after, &source_v_after]);
    assert_eq!(host_f32(&source_k_after), source_k_before);
    assert_eq!(host_f32(&source_v_after), source_v_before);

    drop(branch);
    assert_eq!(pool.snapshot().allocated_blocks, 2);
    drop(source);
    assert_eq!(pool.snapshot().allocated_blocks, 0);
}

#[test]
fn dense_snapshot_repages_into_fixed_slabs_without_dense_block_handles() {
    let mut dense = MlxKVCache::new_contiguous(1);
    let expected_k = fa_token_values(5, 2, 8, 1.0);
    let expected_v = fa_token_values(5, 2, 8, 2.0);
    let _ = dense.append(0, expected_k.clone(), expected_v.clone());
    dense.advance(5);

    let pool = SharedFaBlockPool::new_with_native_slab_storage(FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 2,
        hard_cap: true,
    })
    .expect("central pool");
    let repaged = dense
        .clone_repage_into_shared_fa_pool(pool.clone())
        .expect("central repage");
    assert!(repaged.is_native_fa_shareable());
    let Some(FaLayerStorage::Paged(layer)) = repaged.layers[0].as_ref() else {
        panic!("repage must produce paged storage");
    };
    assert!(layer.slab_storage);
    assert!(layer.k_blocks.is_empty());
    assert!(layer.v_blocks.is_empty());
    let (actual_k, actual_v) = repaged.logical_layer_kv(0).expect("central layer");
    eval(&[&actual_k, &actual_v]);
    eval(&[&expected_k, &expected_v]);
    assert_eq!(host_f32(&actual_k), host_f32(&expected_k));
    assert_eq!(host_f32(&actual_v), host_f32(&expected_v));
    drop(repaged);
    assert_eq!(pool.snapshot().allocated_blocks, 0);
}

// ── Phase 3b: per-layer KV-cache quantization ──

/// Guards `AX_KV_QUANT` across concurrently-running tests.
///
/// `set_kv_quant_table` reads the env var live, so the kill-switch test
/// holds the write lock while `AX_KV_QUANT=0` is set, and every other
/// KV-quant test holds a read lock for its whole scope. Read-locking only
/// the mutating test (the pre-fix state) left the readers racing the
/// kill-switch scope: a table set while `AX_KV_QUANT=0` was visible is
/// silently dropped and `layer_is_quantized` asserts fail intermittently.
static KV_QUANT_ENV_LOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());

struct KvQuantEnvGuard {
    previous: Option<String>,
}

impl KvQuantEnvGuard {
    fn set(value: &str) -> Self {
        let previous = std::env::var(AX_KV_QUANT_ENV).ok();
        // SAFETY: the KV_QUANT_ENV_LOCK write lock is held for the whole
        // test scope, so no reader observes a mid-mutation environment.
        unsafe { std::env::set_var(AX_KV_QUANT_ENV, value) };
        Self { previous }
    }
}

impl Drop for KvQuantEnvGuard {
    fn drop(&mut self) {
        // SAFETY: the KV_QUANT_ENV_LOCK write lock is held for the whole
        // test scope, so no reader observes a mid-mutation environment.
        unsafe {
            match &self.previous {
                Some(value) => std::env::set_var(AX_KV_QUANT_ENV, value),
                None => std::env::remove_var(AX_KV_QUANT_ENV),
            }
        }
    }
}

/// Deterministic pseudo-random `[1, heads, tokens, dim]` bf16 tensor with
/// values in [-2, 2) — bounded so per-bits quant error bounds are
/// predictable. Measured MLX affine worst case against these groups:
/// 4-bit ≤ 0.25 (one quantization step at group extremes), 8-bit ≤ 0.008.
fn kv_tokens_bf16(tokens: usize, heads: usize, dim: usize, seed: u32) -> MlxArray {
    let total = heads * tokens * dim;
    let mut x = seed.max(1);
    let data: Vec<f32> = (0..total)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            ((x >> 8) % 1000) as f32 / 250.0 - 2.0
        })
        .collect();
    let dense = MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data.as_slice()),
        &[1, heads as i32, tokens as i32, dim as i32],
        MlxDtype::Float32,
    );
    let bf16 = astype(&dense, MlxDtype::Bfloat16, None);
    // Materialize before `data` drops: `from_raw_data` may borrow the
    // caller's buffer, and eval reads it while it is still alive.
    eval(&[&bf16]);
    bf16
}

/// f32 host copy of a (possibly bf16, possibly strided) tensor.
fn host_values(arr: &MlxArray) -> Vec<f32> {
    let tight = contiguous(&astype(arr, MlxDtype::Float32, None), None);
    eval(&[&tight]);
    tight.data_f32().to_vec()
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "error comparison needs equal lengths");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn spec(bits: u32, group_size: u32) -> Option<KvQuantSpec> {
    Some(KvQuantSpec { bits, group_size })
}

#[test]
fn kv_quant_mixed_table_matches_dense_within_bits_bounds() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    const H: usize = 2;
    const D: usize = 128;
    let table = vec![spec(8, 64), None, spec(4, 32)];
    let mut quant = MlxKVCache::new_contiguous(3);
    quant.set_kv_quant_table(table);
    let mut dense = MlxKVCache::new_contiguous(3);

    // Steps cross the 256-token chunk twice (300 → 400 → 556), exercising
    // fresh-layer creation and quantized buffer growth on both sides.
    let steps = [(300usize, 7u32), (100, 11), (156, 13)];
    for (n, seed) in steps {
        for layer in 0..3 {
            let k = kv_tokens_bf16(n, H, D, seed + layer as u32);
            let v = kv_tokens_bf16(n, H, D, seed + 100 + layer as u32);
            let (qk, qv) = quant.append(layer, k.clone(), v.clone());
            let (dk, dv) = dense.append(layer, k, v);
            assert_eq!(qk.shape(), dk.shape());
            assert_eq!(qv.shape(), dv.shape());
            assert_eq!(qk.dtype(), MlxDtype::Bfloat16);
            assert_eq!(qv.dtype(), MlxDtype::Bfloat16);
        }
        quant.advance(n);
        dense.advance(n);
    }
    assert_eq!(quant.seq_len(), 556);

    assert!(quant.layer_is_quantized(0));
    assert!(!quant.layer_is_quantized(1));
    assert!(quant.layer_is_quantized(2));
    assert!(quant.has_quantized_layers());

    let mut err8 = 0.0f32;
    let mut err4 = 0.0f32;
    for layer in 0..3 {
        let (qk, qv) = quant.peek_layer_full_kv(layer).expect("quant view");
        let (dk, dv) = dense.peek_layer_full_kv(layer).expect("dense view");
        assert_eq!(qk.shape(), vec![1, H as i32, 556, D as i32]);
        assert_eq!(qk.dtype(), MlxDtype::Bfloat16);
        let k_err = max_abs_err(&host_values(&qk), &host_values(&dk));
        let v_err = max_abs_err(&host_values(&qv), &host_values(&dv));
        let err = k_err.max(v_err);
        match layer {
            // Full-precision layer: identical storage, exact match.
            1 => assert_eq!(err, 0.0, "full-precision layer must match exactly"),
            0 => err8 = err,
            2 => err4 = err,
            _ => unreachable!(),
        }
    }
    assert!(err4 > 0.0, "4-bit quantization must measurably differ");
    assert!(err8 > 0.0, "8-bit quantization must measurably differ");
    assert!(
        err8 < err4,
        "8-bit error ({err8}) must be tighter than 4-bit ({err4})"
    );
    assert!(err4 <= 0.28, "4-bit error {err4} exceeds group bound 0.28");
    assert!(err8 <= 0.03, "8-bit error {err8} exceeds group bound 0.03");
}

#[test]
fn kv_quant_decode_steps_after_prefill_stay_consistent() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    const H: usize = 2;
    const D: usize = 128;
    let table = vec![spec(8, 64), spec(4, 32)];
    let mut quant = MlxKVCache::new_contiguous(2);
    quant.set_kv_quant_table(table);
    let mut dense = MlxKVCache::new_contiguous(2);

    for layer in 0..2 {
        let k = kv_tokens_bf16(128, H, D, 5 + layer as u32);
        let v = kv_tokens_bf16(128, H, D, 105 + layer as u32);
        let _ = quant.append(layer, k.clone(), v.clone());
        let _ = dense.append(layer, k, v);
    }
    quant.advance(128);
    dense.advance(128);

    // Decode-step pattern: many single-token appends after prefill, over
    // both the fresh-layer and steady-state quantized paths.
    for step in 0..16u32 {
        for layer in 0..2 {
            let k = kv_tokens_bf16(1, H, D, 1000 + step * 10 + layer as u32);
            let v = kv_tokens_bf16(1, H, D, 2000 + step * 10 + layer as u32);
            let (qk, _) = quant.append(layer, k.clone(), v.clone());
            let (dk, _) = dense.append(layer, k, v);
            assert_eq!(qk.shape(), dk.shape());
            assert_eq!(qk.shape()[2], 128 + step as i32 + 1);
        }
        quant.advance(1);
        dense.advance(1);
    }

    for layer in 0..2 {
        let (qk, _) = quant.peek_layer_full_kv(layer).expect("quant view");
        let (dk, _) = dense.peek_layer_full_kv(layer).expect("dense view");
        let err = max_abs_err(&host_values(&qk), &host_values(&dk));
        let bound = if layer == 0 { 0.03 } else { 0.28 };
        assert!(
            err <= bound,
            "layer {layer} decode drift {err} exceeds bound {bound}"
        );
    }
}

#[test]
fn kv_quant_trim_to_then_reappend_stays_consistent() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    const H: usize = 1;
    const D: usize = 128;
    let mut quant = MlxKVCache::new_contiguous(1);
    quant.set_kv_quant_table(vec![spec(4, 32)]);
    let mut dense = MlxKVCache::new_contiguous(1);

    let append_both = |quant: &mut MlxKVCache, dense: &mut MlxKVCache, n: usize, seed: u32| {
        let k = kv_tokens_bf16(n, H, D, seed);
        let v = kv_tokens_bf16(n, H, D, seed + 50);
        let _ = quant.append(0, k.clone(), v.clone());
        let _ = dense.append(0, k, v);
        quant.advance(n);
        dense.advance(n);
    };
    append_both(&mut quant, &mut dense, 100, 3);
    append_both(&mut quant, &mut dense, 20, 17);

    assert!(quant.trim_to(110));
    assert!(dense.trim_to(110));
    assert!(quant.layer_is_quantized(0), "trim keeps quantized storage");

    // Re-append over the trimmed region with corrected tokens.
    append_both(&mut quant, &mut dense, 15, 29);
    assert_eq!(quant.seq_len(), 125);
    assert!(quant.layer_is_quantized(0));

    let (qk, qv) = quant.peek_layer_full_kv(0).expect("quant view");
    let (dk, dv) = dense.peek_layer_full_kv(0).expect("dense view");
    assert_eq!(qk.shape(), vec![1, H as i32, 125, D as i32]);
    let err = max_abs_err(&host_values(&qk), &host_values(&dk))
        .max(max_abs_err(&host_values(&qv), &host_values(&dv)));
    assert!(err <= 0.28, "post-trim re-append drift {err} exceeds bound");
    // The corrected tokens must actually hold the corrected values:
    // token 110..125 derive from seed 29, not the rejected seed 17 draft.
    let expected = host_values(&kv_tokens_bf16(15, H, D, 29));
    let got = host_values(&qk);
    let tail = &got[(110 * D)..];
    assert!(
        max_abs_err(tail, &expected[..15 * D]) <= 0.28,
        "re-appended region must hold corrected tokens"
    );
}

#[test]
fn kv_quant_serialize_roundtrip_and_requantize_on_first_append() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    const H: usize = 2;
    const D: usize = 128;
    let table = vec![spec(8, 64), spec(4, 32)];
    let mut quant = MlxKVCache::new_contiguous(2);
    quant.set_kv_quant_table(table.clone());
    let mut dense = MlxKVCache::new_contiguous(2);
    for layer in 0..2 {
        let k = kv_tokens_bf16(96, H, D, 31 + layer as u32);
        let v = kv_tokens_bf16(96, H, D, 131 + layer as u32);
        let _ = quant.append(layer, k.clone(), v.clone());
        let _ = dense.append(layer, k, v);
    }
    quant.advance(96);
    dense.advance(96);

    let bytes = quant.serialize_to_bytes();
    let mut restored = MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip");
    // Wire format is dense-only: restored layers are contiguous dense.
    assert!(!restored.has_quantized_layers());
    for layer in 0..2 {
        let (rk, rv) = restored.peek_layer_full_kv(layer).expect("restored view");
        let (dk, dv) = dense.peek_layer_full_kv(layer).expect("dense view");
        let bound = if layer == 0 { 0.03 } else { 0.28 };
        let err = max_abs_err(&host_values(&rk), &host_values(&dk))
            .max(max_abs_err(&host_values(&rv), &host_values(&dv)));
        assert!(
            err <= bound,
            "layer {layer} restore drift {err} exceeds bound {bound}"
        );
    }

    // Re-quantize on first append after restore: the dense prefix is
    // quantized wholesale and the layer returns to quantized storage.
    restored.set_kv_quant_table(table);
    for layer in 0..2 {
        let k = kv_tokens_bf16(5, H, D, 211 + layer as u32);
        let v = kv_tokens_bf16(5, H, D, 311 + layer as u32);
        let (rk, _) = restored.append(layer, k.clone(), v.clone());
        let (dk, _) = dense.append(layer, k, v);
        assert_eq!(rk.shape(), dk.shape());
        assert!(restored.layer_is_quantized(layer));
    }
    restored.advance(5);
    dense.advance(5);
    for layer in 0..2 {
        let (rk, _) = restored.peek_layer_full_kv(layer).expect("restored view");
        let (dk, _) = dense.peek_layer_full_kv(layer).expect("dense view");
        // Post-restore prefixes were quantized twice (append, then
        // re-quantize on first post-restore append), so the compounding
        // doubles the single-pass bound.
        let bound = if layer == 0 { 0.06 } else { 0.4 };
        let err = max_abs_err(&host_values(&rk), &host_values(&dk));
        assert!(err <= bound, "layer {layer} post-restore drift {err}");
    }
}

#[test]
fn kv_quant_usage_snapshot_reflects_packed_sizes() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    const H: usize = 2;
    const D: usize = 128;
    let mut quant = MlxKVCache::new_contiguous(1);
    quant.set_kv_quant_table(vec![spec(4, 32)]);
    let mut dense = MlxKVCache::new_contiguous(1);
    let k = kv_tokens_bf16(200, H, D, 41);
    let v = kv_tokens_bf16(200, H, D, 141);
    let _ = quant.append(0, k.clone(), v.clone());
    let _ = dense.append(0, k, v);
    quant.advance(200);
    dense.advance(200);

    // Per token (K+V, H=2): packed 128*4/32 u32 = 64 B, scales+biases
    // 2 × 128/32 bf16 = 16 B → 80 B/head/tensor → 320 B/token packed vs
    // 2 × 128 × 2 × 2 = 1024 B/token dense.
    let usage = quant.usage_snapshot();
    assert_eq!(usage.logical_tokens, 200);
    assert_eq!(usage.capacity_tokens, 200);
    assert_eq!(usage.logical_bytes, 320 * 200);
    assert_eq!(usage.capacity_bytes, 320 * 200);
    assert_eq!(usage.quantized_layers, 1);
    let dense_usage = dense.usage_snapshot();
    assert_eq!(dense_usage.logical_bytes, 1024 * 200);
    assert_eq!(dense_usage.quantized_layers, 0);
    assert!(
        usage.logical_bytes * 3 < dense_usage.logical_bytes,
        "4-bit packed storage ({} B) must be well under dense ({} B)",
        usage.logical_bytes,
        dense_usage.logical_bytes
    );
}

#[test]
fn kv_quant_env_zero_disables_table() {
    let _lock = KV_QUANT_ENV_LOCK.write().expect("env lock");
    let _guard = KvQuantEnvGuard::set("0");
    let mut cache = MlxKVCache::new_contiguous(1);
    cache.set_kv_quant_table(vec![spec(4, 32)]);
    let k = kv_tokens_bf16(16, 1, 128, 7);
    let v = kv_tokens_bf16(16, 1, 128, 77);
    let (ck, _) = cache.append(0, k.clone(), v.clone());
    cache.advance(16);

    assert!(!cache.layer_is_quantized(0));
    assert!(!cache.has_quantized_layers());
    assert_eq!(
        host_values(&ck),
        host_values(&k),
        "AX_KV_QUANT=0 must behave as if every spec were None"
    );
}

#[test]
fn kv_quant_table_length_mismatch_is_ignored() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    let mut cache = MlxKVCache::new_contiguous(3);
    cache.set_kv_quant_table(vec![spec(4, 32), None]);
    let k = kv_tokens_bf16(8, 1, 128, 9);
    let v = kv_tokens_bf16(8, 1, 128, 99);
    let _ = cache.append(0, k.clone(), v.clone());
    cache.advance(8);
    assert!(!cache.has_quantized_layers());
}

#[test]
fn kv_quant_invalid_spec_is_rejected_per_layer() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    let mut cache = MlxKVCache::new_contiguous(2);
    cache.set_kv_quant_table(vec![
        Some(KvQuantSpec {
            bits: 3,
            group_size: 32,
        }),
        spec(8, 64),
    ]);
    for layer in 0..2 {
        let k = kv_tokens_bf16(8, 1, 128, 13 + layer as u32);
        let v = kv_tokens_bf16(8, 1, 128, 113 + layer as u32);
        let _ = cache.append(layer, k, v);
    }
    cache.advance(8);
    assert!(!cache.layer_is_quantized(0), "bits=3 spec must be rejected");
    assert!(cache.layer_is_quantized(1));
}

#[test]
fn kv_quant_ring_engagement_demotes_layer_to_dense() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    const D: usize = 128;
    let mut cache = MlxKVCache::new_contiguous(1);
    cache.set_kv_quant_table(vec![spec(8, 64)]);
    cache.set_rotating_sliding_decode(true);
    let k = kv_tokens_bf16(6, 1, D, 19);
    let v = kv_tokens_bf16(6, 1, D, 119);
    let _ = cache.append(0, k, v);
    cache.advance(6);
    assert!(cache.layer_is_quantized(0));

    // Ring-eligible single-token append: quantized rings are out of
    // scope, so the layer demotes to dense and converts to a ring.
    let k = kv_tokens_bf16(1, 1, D, 23);
    let v = kv_tokens_bf16(1, 1, D, 123);
    let (rk, _) = cache.append_with_retained_window(0, k, v, Some(4));
    cache.advance(1);
    assert!(!cache.layer_is_quantized(0));
    assert!(cache.has_rotated_sliding_layers());
    assert_eq!(rk.shape(), vec![1, 1, 4, D as i32]);
    let ring = cache.layer_sliding_ring(0).expect("ring geometry");
    assert_eq!((ring.window, ring.capacity), (4, 4));
}

#[test]
fn kv_quant_protected_prefix_ring_demotes_layer_to_dense() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    const D: usize = 128;
    let mut cache = MlxKVCache::new_contiguous(1);
    cache.set_kv_quant_table(vec![spec(8, 64)]);
    let k = kv_tokens_bf16(3, 1, D, 29);
    let v = kv_tokens_bf16(3, 1, D, 129);
    let _ = cache.append(0, k, v);
    cache.advance(3);
    assert!(cache.layer_is_quantized(0));

    let k = kv_tokens_bf16(1, 1, D, 31);
    let v = kv_tokens_bf16(1, 1, D, 131);
    let kv = cache.append_with_protected_prefix_window_for_attention(0, k, v, 2);
    cache.advance(1);
    let MlxAttentionKv::Dense { k, .. } = kv else {
        panic!("protected-prefix decode always returns dense views");
    };
    assert_eq!(k.shape(), vec![1, 1, 4, D as i32]);
    assert!(!cache.layer_is_quantized(0));
    let lkv = contiguous_layer(&cache, 0);
    assert!(lkv.protected_prefix_ring.is_some());
}

#[test]
fn kv_quant_layers_never_take_paged_route() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    let config = FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 32,
        hard_cap: false,
    };
    let pool_layer_cache = MlxKVCache::new_with_fa_block_pool(2, config);
    let pool = pool_layer_cache.fa_pool.as_ref().expect("pool").clone();
    let mut cache = pool_layer_cache;
    cache.set_kv_quant_table(vec![spec(4, 32), None]);

    for layer in 0..2 {
        let k = kv_tokens_bf16(8, 1, 128, 37 + layer as u32);
        let v = kv_tokens_bf16(8, 1, 128, 137 + layer as u32);
        let _ = cache.append(layer, k, v);
    }
    cache.advance(8);
    // Spec'd layer quantized contiguous; spec-less layer paged.
    assert!(cache.layer_is_quantized(0));
    assert!(matches!(cache.layers[1], Some(FaLayerStorage::Paged(_))));
    assert!(!cache.is_native_fa_shareable());
    assert_eq!(pool.snapshot().allocated_blocks, 2);

    // Single-token decode on the quantized layer must return the dense
    // route, never a paged attention view.
    let k = kv_tokens_bf16(1, 1, 128, 41);
    let v = kv_tokens_bf16(1, 1, 128, 141);
    let attention = cache.append_with_retained_window_for_attention(0, k, v, None);
    cache.advance(1);
    assert!(
        matches!(attention, MlxAttentionKv::Dense { .. }),
        "quantized layers must always take the Dense attention route"
    );
    assert_eq!(pool.snapshot().allocated_blocks, 2);
}

#[test]
fn kv_quant_clone_deep_copies_buffers_and_spec() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    const D: usize = 128;
    let mut cache = MlxKVCache::new_contiguous(1);
    cache.set_kv_quant_table(vec![spec(8, 64)]);
    let k = kv_tokens_bf16(32, 1, D, 43);
    let v = kv_tokens_bf16(32, 1, D, 143);
    let _ = cache.append(0, k, v);
    cache.advance(32);

    let mut branch = cache.clone();
    assert!(branch.layer_is_quantized(0));
    let (bk, _) = branch.peek_layer_full_kv(0).expect("branch view");
    let (ck, _) = cache.peek_layer_full_kv(0).expect("source view");
    assert_eq!(host_values(&bk), host_values(&ck));

    // Diverging appends must not disturb each other's contents.
    let k = kv_tokens_bf16(1, 1, D, 47);
    let v = kv_tokens_bf16(1, 1, D, 147);
    let _ = branch.append(0, k, v);
    branch.advance(1);
    let k = kv_tokens_bf16(1, 1, D, 53);
    let v = kv_tokens_bf16(1, 1, D, 153);
    let _ = cache.append(0, k, v);
    cache.advance(1);

    let (bk, _) = branch.peek_layer_full_kv(0).expect("branch view");
    let (ck, _) = cache.peek_layer_full_kv(0).expect("source view");
    let branch_tail = host_values(&bk)[(32 * D)..].to_vec();
    let cache_tail = host_values(&ck)[(32 * D)..].to_vec();
    let expected_branch = host_values(&kv_tokens_bf16(1, 1, D, 47));
    let expected_cache = host_values(&kv_tokens_bf16(1, 1, D, 53));
    assert!(max_abs_err(&branch_tail, &expected_branch) <= 0.03);
    assert!(max_abs_err(&cache_tail, &expected_cache) <= 0.03);
}

#[test]
fn kv_quant_repage_skips_quantized_layers() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    const D: usize = 128;
    let mut dense_source = MlxKVCache::new_contiguous(2);
    dense_source.set_kv_quant_table(vec![spec(4, 32), None]);
    let mut control = MlxKVCache::new_contiguous(2);
    for layer in 0..2 {
        let k = kv_tokens_bf16(5, 1, D, 59 + layer as u32);
        let v = kv_tokens_bf16(5, 1, D, 159 + layer as u32);
        let _ = dense_source.append(layer, k.clone(), v.clone());
        let _ = control.append(layer, k, v);
    }
    dense_source.advance(5);
    control.advance(5);

    // Only the dense layer claims blocks: 5 tokens → 2 blocks of 4.
    let pool = SharedFaBlockPool::new(FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 2,
        hard_cap: true,
    })
    .expect("pool");
    assert_eq!(dense_source.fa_blocks_required_for_repage(&pool), Ok(2));
    let repaged = dense_source
        .clone_repage_into_shared_fa_pool(pool.clone())
        .expect("repage");
    assert_eq!(pool.snapshot().allocated_blocks, 2);
    assert!(
        matches!(repaged.layers[0], Some(FaLayerStorage::Contiguous(_))),
        "quantized layer must stay contiguous in the repaged clone"
    );
    assert!(
        matches!(repaged.layers[1], Some(FaLayerStorage::Paged(_))),
        "dense layer must repage into the pool"
    );
    let (rk, _) = repaged.logical_layer_kv(0).expect("repaged quant layer");
    let (ck, _) = control.logical_layer_kv(0).expect("control layer");
    assert!(max_abs_err(&host_values(&rk), &host_values(&ck)) <= 0.28);
    let (rk, _) = repaged.logical_layer_kv(1).expect("repaged dense layer");
    let (ck, _) = control.logical_layer_kv(1).expect("control layer");
    assert_eq!(host_values(&rk), host_values(&ck));

    // The copied table re-quantizes the skipped layer on its next append.
    let mut repaged = repaged;
    let k = kv_tokens_bf16(1, 1, D, 61);
    let v = kv_tokens_bf16(1, 1, D, 161);
    let _ = repaged.append(0, k, v);
    repaged.advance(1);
    assert!(repaged.layer_is_quantized(0));
    drop(repaged);
    assert_eq!(pool.snapshot().allocated_blocks, 0);
}

#[test]
fn kv_quant_table_injection_demotes_already_paged_layer() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    // Prefix-restore adoption can inject the table onto a cache whose
    // layers already hold paged storage; the spec'd layer must demote to
    // contiguous at injection and quantize on its next append.
    let config = FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 16,
        hard_cap: false,
    };
    let mut cache = MlxKVCache::new_with_fa_block_pool(1, config);
    let pool = cache.fa_pool.as_ref().expect("pool").clone();
    let k = kv_tokens_bf16(8, 1, 128, 71);
    let v = kv_tokens_bf16(8, 1, 128, 171);
    let _ = cache.append(0, k, v);
    cache.advance(8);
    assert!(matches!(cache.layers[0], Some(FaLayerStorage::Paged(_))));
    assert_eq!(pool.snapshot().allocated_blocks, 2);

    cache.set_kv_quant_table(vec![spec(8, 64)]);
    assert!(
        matches!(cache.layers[0], Some(FaLayerStorage::Contiguous(_))),
        "injection must demote the paged layer to contiguous"
    );
    assert_eq!(pool.snapshot().allocated_blocks, 0);

    let k = kv_tokens_bf16(1, 1, 128, 73);
    let v = kv_tokens_bf16(1, 1, 128, 173);
    let (ck, _) = cache.append(0, k, v);
    cache.advance(1);
    assert!(cache.layer_is_quantized(0));
    assert_eq!(ck.shape(), vec![1, 1, 9, 128]);
}

/// The pipeline.rs cache-creation path: a `ModelConfig` carrying a
/// `kv_cache_quant` table must reach the cache it creates.
#[test]
fn kv_quant_config_table_reaches_cache_via_pipeline_creation() {
    let _env = KV_QUANT_ENV_LOCK.read().expect("env lock");
    let config = crate::model::ModelConfig {
        compile_cache_identity: 1,
        model_family: "qwen3".to_string(),
        layer_count: 2,
        hidden_size: 16,
        intermediate_size: 32,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 128,
        vocab_size: 32,
        rope_theta: 10000.0,
        rope_dims: 128,
        attn_output_gate: false,
        query_scale: 1.0,
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
        deepseek_v4: None,
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
        generation_kind: ax_engine_core::GenerationKind::Autoregressive,
        kv_cache_quant: vec![spec(8, 64), None],
    };

    // Mirrors pipeline.rs `execute`'s or_insert_with closure.
    let mut cache = MlxKVCache::new_contiguous(config.layer_count);
    cache.set_kv_quant_table(config.kv_cache_quant.clone());

    for layer in 0..2 {
        let k = kv_tokens_bf16(8, 1, 128, 67 + layer as u32);
        let v = kv_tokens_bf16(8, 1, 128, 167 + layer as u32);
        let _ = cache.append(layer, k, v);
    }
    cache.advance(8);
    assert!(cache.layer_is_quantized(0));
    assert!(!cache.layer_is_quantized(1));
}

#[test]
fn deepseek_v4_append_prefill_then_decode_reads_back_full_window() {
    let mut cache = MlxKVCache::new_contiguous(2);
    let head_dim = 4;
    // Prefill batch of 3 tokens, then two single-token decode appends.
    let prefill = build_mla_latent_f32(3, head_dim);
    let full = cache.append_deepseek_v4(0, prefill.clone());
    assert_eq!(full.shape(), vec![1, 1, 3, head_dim]);
    cache.advance(3);

    let step4 = build_mla_latent_f32(1, head_dim);
    let full = cache.append_deepseek_v4(0, step4);
    assert_eq!(full.shape(), vec![1, 1, 4, head_dim]);
    cache.advance(1);
    let step5 = build_mla_latent_f32(1, head_dim);
    let full = cache.append_deepseek_v4(0, step5);
    assert_eq!(full.shape(), vec![1, 1, 5, head_dim]);
    cache.advance(1);

    // The last-3 window holds prefill token 2 plus the two decode steps.
    let view = cache.deepseek_v4_k_window(0, 3).expect("window read-back");
    assert_eq!(view.shape(), vec![1, 1, 3, head_dim]);
    let mut expected: Vec<f32> = (8..12).map(|i| (i as f32) * 0.0007).collect();
    expected.extend((0..4).map(|i| (i as f32) * 0.0007));
    expected.extend((0..4).map(|i| (i as f32) * 0.0007));
    assert_eq!(host_f32(&view), expected);

    // Full window (0 = no sliding window) covers every appended token.
    let full = cache.deepseek_v4_k_window(0, 0).expect("full window");
    assert_eq!(full.shape(), vec![1, 1, 5, head_dim]);

    // A second layer stays independent.
    assert!(cache.deepseek_v4_k_window(1, 0).is_none());
    let state = cache.deepseek_v4_layer_state(0).expect("state view");
    assert_eq!(state.head_dim, head_dim);
}

#[test]
fn deepseek_v4_trim_and_reset_semantics_match_siblings() {
    let mut cache = MlxKVCache::new_contiguous(1);
    let head_dim = 4;
    let prefill = build_mla_latent_f32(4, head_dim);
    let _ = cache.append_deepseek_v4(0, prefill.clone());
    cache.advance(4);

    // Draft rollback: trim drops the logical boundary; the next append
    // overwrites from the trim point.
    assert!(cache.trim_to(2));
    let replacement = build_mla_latent_f32(1, head_dim);
    let full = cache.append_deepseek_v4(0, replacement.clone());
    assert_eq!(full.shape(), vec![1, 1, 3, head_dim]);
    cache.advance(1);
    let window = cache.deepseek_v4_k_window(0, 1).expect("last token");
    assert_eq!(host_f32(&window), host_f32(&replacement));

    // Reset clears V4 state alongside the sibling caches.
    cache.reset();
    assert!(cache.deepseek_v4_k_window(0, 0).is_none());
    assert!(cache.deepseek_v4_layer_state(0).is_none());
}

#[test]
fn deepseek_v4_comp_rows_chunked_append_and_readback() {
    let mut cache = MlxKVCache::new_contiguous(1);
    let head_dim = 4;
    let _ = cache.append_deepseek_v4(0, build_mla_latent_f32(8, head_dim));
    cache.advance(8);

    cache.deepseek_v4_comp_ensure(0, false, 4, true);
    assert_eq!(cache.deepseek_v4_comp_committed(0, false), 0);
    assert!(cache.deepseek_v4_comp_k(0, false).is_none());

    // First append: two rows (block rows are [1, 1, n, dim] like the raw K).
    let rows_a = build_mla_latent_f32(2, head_dim);
    let view = cache.append_deepseek_v4_comp_rows(0, false, rows_a.clone());
    assert_eq!(view.shape(), vec![1, 1, 2, head_dim]);
    assert_eq!(host_f32(&view), host_f32(&rows_a));

    // Second append: one row; the committed view spans all three rows.
    let rows_b = build_mla_latent_f32(1, head_dim);
    let view = cache.append_deepseek_v4_comp_rows(0, false, rows_b.clone());
    assert_eq!(view.shape(), vec![1, 1, 3, head_dim]);
    assert_eq!(cache.deepseek_v4_comp_committed(0, false), 3);
    let mut expected = host_f32(&rows_a);
    expected.extend(host_f32(&rows_b));
    assert_eq!(host_f32(&view), expected);
    let read = cache.deepseek_v4_comp_k(0, false).expect("committed rows");
    assert_eq!(host_f32(&read), expected);

    // The indexer pipeline is independent.
    assert_eq!(cache.deepseek_v4_comp_committed(0, true), 0);
    cache.deepseek_v4_comp_ensure(0, true, 4, true);
    let idx_rows = build_mla_latent_f32(1, head_dim);
    let idx_view = cache.append_deepseek_v4_comp_rows(0, true, idx_rows.clone());
    assert_eq!(idx_view.shape(), vec![1, 1, 1, head_dim]);
    assert_eq!(host_f32(&idx_view), host_f32(&idx_rows));
    assert_eq!(cache.deepseek_v4_comp_committed(0, false), 3);
}

#[test]
fn deepseek_v4_comp_states_and_trim_rewind() {
    let mut cache = MlxKVCache::new_contiguous(1);
    let head_dim = 4;
    let _ = cache.append_deepseek_v4(0, build_mla_latent_f32(10, head_dim));
    cache.advance(10);

    // Overlap pipeline (r=4): committed 2 blocks, states buffered over
    // [4, 10) — the previous block plus the partial current one.
    cache.deepseek_v4_comp_ensure(0, false, 4, true);
    let rows = build_mla_latent_f32(2, head_dim);
    let _ = cache.append_deepseek_v4_comp_rows(0, false, rows);
    // Compressor states are 2D `[rows, state_width]` (F32).
    let kv_states = mlx_sys::reshape(&build_mla_latent_f32(6, head_dim), &[6, head_dim], None);
    let score_states = mlx_sys::reshape(&build_mla_latent_f32(6, head_dim), &[6, head_dim], None);
    cache.deepseek_v4_comp_replace_states(0, false, 4, Some(kv_states.clone()), Some(score_states));
    let (base, kv, _score) = cache.deepseek_v4_comp_states(0, false).expect("states");
    assert_eq!(base, 4);
    assert_eq!(host_f32(&kv), host_f32(&kv_states));

    // Covered trim to 9: committed stays 2 (9 / 4), the buffer re-slices
    // to [4, 9) and the overlap keep-from stays at block 1's start.
    assert!(cache.trim_to(9));
    assert_eq!(cache.deepseek_v4_comp_committed(0, false), 2);
    let (base, kv, _score) = cache.deepseek_v4_comp_states(0, false).expect("states");
    assert_eq!(base, 4);
    assert_eq!(kv.shape(), vec![5, head_dim]);

    // Trim back to 3 drops the second committed row; the retained buffer
    // no longer covers the rewind range, so it clears (documented
    // deep-rollback limitation) and restarts empty at 3.
    assert!(cache.trim_to(3));
    assert_eq!(cache.deepseek_v4_comp_committed(0, false), 0);
    assert!(cache.deepseek_v4_comp_states(0, false).is_none());
}
