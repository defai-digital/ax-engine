use super::*;

use crate::backend::Backend;
use crate::backend::metal::MetalBackend;

#[test]
fn test_qwen35_kv_layer_pattern() {
    let kv = Qwen3_5Kv::new(8, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
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
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    kv.conv_state_mut(0).fill(1.0);
    kv.recurrent_state_mut(0).fill(2.0);
    kv.finalize_token();
    kv.clear();
    assert_eq!(kv.seq_len(), 0);
    assert!(kv.conv_state_mut(0).iter().all(|&v| v == 0.0));
    assert!(kv.recurrent_state_mut(0).iter().all(|&v| v == 0.0));
    assert_eq!(kv.recurrent_seqlen_offset(kv.active_slot()), 0);
}

#[test]
fn test_qwen35_kv_finalize_batch_for_slots_keeps_shared_timeline_aligned() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();

    kv.finalize_batch_for_slots(&[0, slot1], 2);

    assert_eq!(kv.seq_len(), 2);
    assert_eq!(kv.recurrent_seqlen_offset(0), 2);
    assert_eq!(kv.recurrent_seqlen_offset(slot1), 2);
}

#[test]
fn test_qwen35_kv_finalize_batch_uses_explicit_batch_slot_indices() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();

    kv.set_batch_slot_indices(&[0, slot1]);
    kv.finalize_batch(2);

    assert_eq!(kv.seq_len(), 2);
    assert_eq!(kv.recurrent_seqlen_offset(0), 2);
    assert_eq!(kv.recurrent_seqlen_offset(slot1), 2);
    assert_eq!(kv.recurrent_batch_slot_indices().as_ref(), &[0]);
}

#[test]
fn test_qwen35_kv_recurrent_batch_slot_indices_fallback_to_active_slot() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();

    assert_eq!(kv.recurrent_batch_slot_indices().as_ref(), &[0]);
    kv.set_active_slot(slot1);
    assert_eq!(kv.recurrent_batch_slot_indices().as_ref(), &[slot1]);

    kv.set_batch_slot_indices(&[slot1]);
    assert_eq!(kv.recurrent_batch_slot_indices().as_ref(), &[slot1]);

    kv.clear_batch_slot_indices();
    assert_eq!(kv.recurrent_batch_slot_indices().as_ref(), &[slot1]);
}

#[test]
#[should_panic(expected = "qwen35 recurrent slot batch requires the active slot to be included")]
fn test_qwen35_kv_rejects_setting_batch_slot_indices_without_active_slot() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();

    kv.set_batch_slot_indices(&[slot1]);
}

#[test]
#[should_panic(
    expected = "qwen35 shared-attention finalize requires the active recurrent slot to be included"
)]
fn test_qwen35_kv_rejects_finalizing_shared_timeline_without_active_slot() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();
    kv.finalize_batch_for_slots(&[slot1], 1);
}

#[test]
#[should_panic(
    expected = "qwen35 recurrent slot batch slot 1 has seqlen_offset 1 != shared attention seq_len 0"
)]
fn test_qwen35_kv_rejects_finalizing_misaligned_multi_slot_shared_timeline() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();
    kv.set_recurrent_seqlen_offset(slot1, 1);
    kv.finalize_batch_for_slots(&[0, slot1], 1);
}

#[test]
#[should_panic(expected = "multiple of group_count")]
fn test_qwen35_kv_rejects_incompatible_head_expansion() {
    let _ = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 768, 128, 6, 4);
}

#[test]
fn test_qwen35_kv_snapshot_restore_round_trips_slot_state() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();
    kv.conv_state_for_slot_mut(slot1, 0).fill(1.5);
    kv.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
    kv.set_recurrent_seqlen_offset(slot1, 11);

    let snapshot = kv.recurrent_slot_snapshot(slot1);

    kv.conv_state_for_slot_mut(slot1, 0).fill(0.0);
    kv.recurrent_state_for_slot_mut(slot1, 0).fill(0.0);
    kv.set_recurrent_seqlen_offset(slot1, 0);
    kv.restore_recurrent_slot(slot1, &snapshot);

    assert!(kv.conv_state_for_slot(slot1, 0).iter().all(|&v| v == 1.5));
    assert!(
        kv.recurrent_state_for_slot(slot1, 0)
            .iter()
            .all(|&v| v == 2.5)
    );
    assert_eq!(kv.recurrent_seqlen_offset(slot1), 11);
}

#[test]
fn test_qwen35_kv_clone_recurrent_slot_copies_state_and_offset() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();
    let slot2 = kv.allocate_recurrent_slot();

    kv.conv_state_for_slot_mut(slot1, 0).fill(1.25);
    kv.recurrent_state_for_slot_mut(slot1, 0).fill(2.75);
    kv.set_recurrent_seqlen_offset(slot1, 9);

    kv.clone_recurrent_slot(slot1, slot2);

    assert!(kv.conv_state_for_slot(slot2, 0).iter().all(|&v| v == 1.25));
    assert!(
        kv.recurrent_state_for_slot(slot2, 0)
            .iter()
            .all(|&v| v == 2.75)
    );
    assert_eq!(kv.recurrent_seqlen_offset(slot2), 9);
}

#[test]
fn test_qwen35_kv_restore_recurrent_slot_allocates_missing_slot() {
    let mut source = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = source.allocate_recurrent_slot();
    source.conv_state_for_slot_mut(slot1, 0).fill(1.25);
    source.recurrent_state_for_slot_mut(slot1, 0).fill(2.75);
    source.set_recurrent_seqlen_offset(slot1, 13);
    let snapshot = source.recurrent_slot_snapshot(slot1);

    let mut restored = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    restored.restore_recurrent_slot(slot1, &snapshot);

    assert!(
        restored
            .conv_state_for_slot(slot1, 0)
            .iter()
            .all(|&v| v == 1.25)
    );
    assert!(
        restored
            .recurrent_state_for_slot(slot1, 0)
            .iter()
            .all(|&v| v == 2.75)
    );
    assert_eq!(restored.recurrent_seqlen_offset(slot1), 13);
}

#[test]
fn test_qwen35_kv_allocates_and_reuses_recurrent_slots() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot = kv.allocate_recurrent_slot();
    assert_eq!(slot, 1);

    let mut snapshot = kv.recurrent_slot_snapshot(slot);
    snapshot.conv_states[0].fill(3.0);
    snapshot.conv_states[1].fill(3.0);
    snapshot.conv_states[2].fill(3.0);
    snapshot.recurrent_states[0].fill(4.0);
    snapshot.recurrent_states[1].fill(4.0);
    snapshot.recurrent_states[2].fill(4.0);
    snapshot.seqlen_offset = 9;
    kv.restore_recurrent_slot(slot, &snapshot);

    kv.free_recurrent_slot(slot);

    let reused = kv.allocate_recurrent_slot();
    assert_eq!(reused, slot);
    let snapshot = kv.recurrent_slot_snapshot(reused);
    assert!(
        snapshot
            .conv_states
            .iter()
            .all(|state| state.iter().all(|&v| v == 0.0))
    );
    assert!(
        snapshot
            .recurrent_states
            .iter()
            .all(|state| state.iter().all(|&v| v == 0.0))
    );
    assert_eq!(snapshot.seqlen_offset, 0);
}

#[test]
#[should_panic(expected = "cannot activate free qwen35 recurrent slot")]
fn test_qwen35_kv_rejects_activating_freed_slot() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot = kv.allocate_recurrent_slot();
    kv.free_recurrent_slot(slot);
    kv.set_active_slot(slot);
}

#[test]
#[should_panic(expected = "cannot free inactive qwen35 recurrent slot")]
fn test_qwen35_kv_rejects_freeing_same_slot_twice() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot = kv.allocate_recurrent_slot();
    kv.free_recurrent_slot(slot);
    kv.free_recurrent_slot(slot);
}

#[test]
#[should_panic(expected = "cannot free reserved qwen35 recurrent slot 0")]
fn test_qwen35_kv_rejects_freeing_reserved_slot_zero() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();
    kv.set_active_slot(slot1);
    kv.free_recurrent_slot(0);
}

#[test]
#[should_panic(expected = "qwen35 recurrent slot is not allocated")]
fn test_qwen35_kv_rejects_reading_freed_slot_state() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot = kv.allocate_recurrent_slot();
    kv.free_recurrent_slot(slot);
    let _ = kv.conv_state_for_slot(slot, 0);
}

#[test]
#[should_panic(expected = "qwen35 recurrent slot is not allocated")]
fn test_qwen35_kv_rejects_snapshotting_freed_slot() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot = kv.allocate_recurrent_slot();
    kv.free_recurrent_slot(slot);
    let _ = kv.recurrent_slot_snapshot(slot);
}

#[test]
fn test_qwen35_kv_slot_specific_access_stays_isolated() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();

    kv.conv_state_for_slot_mut(0, 0).fill(1.0);
    kv.recurrent_state_for_slot_mut(0, 0).fill(2.0);
    kv.conv_state_for_slot_mut(slot1, 0).fill(3.0);
    kv.recurrent_state_for_slot_mut(slot1, 0).fill(4.0);

    kv.set_active_slot(slot1);
    assert!(kv.conv_state_mut(0).iter().all(|&v| v == 3.0));
    assert!(kv.recurrent_state_mut(0).iter().all(|&v| v == 4.0));
    assert_eq!(kv.recurrent_seqlen_offset(kv.active_slot()), 0);

    kv.set_active_slot(0);
    assert!(kv.conv_state_mut(0).iter().all(|&v| v == 1.0));
    assert!(kv.recurrent_state_mut(0).iter().all(|&v| v == 2.0));
    assert_eq!(kv.recurrent_seqlen_offset(kv.active_slot()), 0);
}

#[test]
#[should_panic(
    expected = "cannot activate qwen35 recurrent slot with seqlen_offset 1 != shared attention seq_len 0"
)]
fn test_qwen35_kv_rejects_activating_slot_with_mismatched_shared_seq_len() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();
    kv.set_recurrent_seqlen_offset(slot1, 1);
    kv.set_active_slot(slot1);
}

#[test]
fn test_qwen35_kv_gather_and_scatter_recurrent_state_batch() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();
    let slot_indices = [slot1, 0];

    kv.conv_state_for_slot_mut(0, 0).fill(1.0);
    kv.recurrent_state_for_slot_mut(0, 0).fill(2.0);
    kv.conv_state_for_slot_mut(slot1, 0).fill(3.0);
    kv.recurrent_state_for_slot_mut(slot1, 0).fill(4.0);

    let conv_state_len = kv.conv_state_for_slot(0, 0).len();
    let recurrent_state_len = kv.recurrent_state_for_slot(0, 0).len();
    let mut conv_batch = vec![0.0; slot_indices.len() * conv_state_len];
    let mut recurrent_batch = vec![0.0; slot_indices.len() * recurrent_state_len];
    kv.gather_recurrent_state_batch(&slot_indices, 0, &mut conv_batch, &mut recurrent_batch);

    assert!(conv_batch[..conv_state_len].iter().all(|&v| v == 3.0));
    assert!(
        recurrent_batch[..recurrent_state_len]
            .iter()
            .all(|&v| v == 4.0)
    );
    assert!(conv_batch[conv_state_len..].iter().all(|&v| v == 1.0));
    assert!(
        recurrent_batch[recurrent_state_len..]
            .iter()
            .all(|&v| v == 2.0)
    );

    conv_batch[..conv_state_len].fill(5.0);
    recurrent_batch[..recurrent_state_len].fill(6.0);
    conv_batch[conv_state_len..].fill(7.0);
    recurrent_batch[recurrent_state_len..].fill(8.0);
    kv.scatter_recurrent_state_batch(&slot_indices, 0, &conv_batch, &recurrent_batch);

    assert!(kv.conv_state_for_slot(0, 0).iter().all(|&v| v == 7.0));
    assert!(kv.recurrent_state_for_slot(0, 0).iter().all(|&v| v == 8.0));
    assert!(kv.conv_state_for_slot(slot1, 0).iter().all(|&v| v == 5.0));
    assert!(
        kv.recurrent_state_for_slot(slot1, 0)
            .iter()
            .all(|&v| v == 6.0)
    );
}

#[test]
fn test_qwen35_kv_prepare_recurrent_state_batch_single_slot_writes_through() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot_indices = [0usize];

    {
        let mut prepared = kv.prepare_recurrent_state_batch(&slot_indices, 0);
        {
            let mut state_batch = prepared.state_batch();
            state_batch.conv_state_for_slot_mut(0).fill(1.25);
            state_batch.recurrent_state_for_slot_mut(0).fill(2.5);
        }
        prepared.finish();
    }

    assert!(kv.conv_state_for_slot(0, 0).iter().all(|&v| v == 1.25));
    assert!(kv.recurrent_state_for_slot(0, 0).iter().all(|&v| v == 2.5));
}

#[test]
fn test_qwen35_kv_prepare_recurrent_state_batch_multi_slot_scatter_on_finish() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();
    let slot_indices = [0usize, slot1];

    {
        let mut prepared = kv.prepare_recurrent_state_batch(&slot_indices, 0);
        {
            let mut state_batch = prepared.state_batch();
            state_batch.conv_state_for_slot_mut(0).fill(3.0);
            state_batch.recurrent_state_for_slot_mut(0).fill(4.0);
            state_batch.conv_state_for_slot_mut(1).fill(5.0);
            state_batch.recurrent_state_for_slot_mut(1).fill(6.0);
        }
        prepared.finish();
    }

    assert!(kv.conv_state_for_slot(0, 0).iter().all(|&v| v == 3.0));
    assert!(kv.recurrent_state_for_slot(0, 0).iter().all(|&v| v == 4.0));
    assert!(kv.conv_state_for_slot(slot1, 0).iter().all(|&v| v == 5.0));
    assert!(
        kv.recurrent_state_for_slot(slot1, 0)
            .iter()
            .all(|&v| v == 6.0)
    );
}

#[test]
fn test_qwen35_kv_prepare_recurrent_state_batch_single_slot_materializes_backend_owned_gpu_state() {
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let layer = 0usize;
    kv.enable_gpu_recurrent_state(&metal_ops.device).unwrap();

    let _ = kv.note_backend_conv_state_update(0, layer);
    let _ = kv.note_backend_recurrent_state_update(0, layer);
    let (conv_gpu, recurrent_gpu) = kv.gpu_recurrent_buffers_mut(0, layer).unwrap();
    unsafe {
        conv_gpu.as_mut_slice::<f32>().fill(1.25);
        recurrent_gpu.as_mut_slice::<f32>().fill(2.5);
    }

    let slot_indices = [0usize];
    let mut prepared = kv.prepare_recurrent_state_batch(&slot_indices, layer);
    assert_eq!(
        prepared.kind(),
        Qwen3_5PreparedRecurrentStateBatchKind::CpuDirectMaterializedFromBackend
    );
    {
        let mut state_batch = prepared.state_batch();
        assert_eq!(state_batch.slot_count(), 1);
        assert!(
            state_batch
                .conv_state_for_slot_mut(0)
                .iter()
                .all(|&v| v == 1.25)
        );
        assert!(
            state_batch
                .recurrent_state_for_slot_mut(0)
                .iter()
                .all(|&v| v == 2.5)
        );
    }
    prepared.finish();

    assert!(!kv.conv_state_cpu_stale(0, layer));
    assert!(!kv.recurrent_state_cpu_stale(0, layer));
}

#[test]
fn test_qwen35_kv_prepare_recurrent_state_batch_multi_slot_materializes_backend_owned_gpu_state() {
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let layer = 0usize;
    let slot1 = kv.allocate_recurrent_slot();
    kv.enable_gpu_recurrent_state(&metal_ops.device).unwrap();

    let _ = kv.note_backend_conv_state_update(0, layer);
    let _ = kv.note_backend_recurrent_state_update(0, layer);
    let _ = kv.note_backend_conv_state_update(slot1, layer);
    let _ = kv.note_backend_recurrent_state_update(slot1, layer);
    let (conv0, rec0) = kv.gpu_recurrent_buffers_mut(0, layer).unwrap();
    unsafe {
        conv0.as_mut_slice::<f32>().fill(3.0);
        rec0.as_mut_slice::<f32>().fill(4.0);
    }
    let (conv1, rec1) = kv.gpu_recurrent_buffers_mut(slot1, layer).unwrap();
    unsafe {
        conv1.as_mut_slice::<f32>().fill(5.0);
        rec1.as_mut_slice::<f32>().fill(6.0);
    }

    let slot_indices = [0usize, slot1];
    let mut prepared = kv.prepare_recurrent_state_batch(&slot_indices, layer);
    assert_eq!(
        prepared.kind(),
        Qwen3_5PreparedRecurrentStateBatchKind::CpuGatheredMaterializedFromBackend
    );
    {
        let mut state_batch = prepared.state_batch();
        assert!(
            state_batch
                .conv_state_for_slot_mut(0)
                .iter()
                .all(|&v| v == 3.0)
        );
        assert!(
            state_batch
                .recurrent_state_for_slot_mut(0)
                .iter()
                .all(|&v| v == 4.0)
        );
        assert!(
            state_batch
                .conv_state_for_slot_mut(1)
                .iter()
                .all(|&v| v == 5.0)
        );
        assert!(
            state_batch
                .recurrent_state_for_slot_mut(1)
                .iter()
                .all(|&v| v == 6.0)
        );
    }
    prepared.finish();

    assert!(!kv.conv_state_cpu_stale(0, layer));
    assert!(!kv.recurrent_state_cpu_stale(0, layer));
    assert!(!kv.conv_state_cpu_stale(slot1, layer));
    assert!(!kv.recurrent_state_cpu_stale(slot1, layer));
}

#[test]
fn test_qwen35_kv_state_generation_changes_on_cpu_mutation_and_reset() {
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let layer = 0usize;
    let base_generation = kv.recurrent_state_generation(0, layer);
    let base_conv_generation = kv.conv_state_generation(0, layer);

    kv.conv_state_for_slot_mut(0, layer)[0] = 1.0;
    let after_conv_mutation = kv.recurrent_state_generation(0, layer);
    let after_conv_cpu_generation = kv.conv_state_generation(0, layer);
    assert_eq!(after_conv_mutation, base_generation);
    assert!(after_conv_cpu_generation > base_conv_generation);

    kv.recurrent_state_for_slot_mut(0, layer)[0] = 2.0;
    let after_recurrent_mutation = kv.recurrent_state_generation(0, layer);
    assert!(after_recurrent_mutation > after_conv_mutation);

    let snapshot = kv.recurrent_slot_snapshot(0);
    kv.clear();
    let after_clear = kv.recurrent_state_generation(0, layer);
    assert!(after_clear > after_recurrent_mutation);

    kv.restore_recurrent_slot(0, &snapshot);
    let after_restore = kv.recurrent_state_generation(0, layer);
    assert!(after_restore > after_clear);
}

#[test]
fn test_qwen35_kv_conv_only_cpu_mutation_keeps_backend_recurrent_state_stale() {
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let layer = 0usize;

    assert!(!kv.recurrent_state_cpu_stale(0, layer));
    let backend_generation = kv.note_backend_recurrent_state_update(0, layer);
    assert!(kv.recurrent_state_cpu_stale(0, layer));

    kv.conv_state_for_slot_mut(0, layer)[0] = 1.0;
    assert!(
        kv.recurrent_state_cpu_stale(0, layer),
        "conv-only CPU mutation must not mark backend-owned recurrent state as materialized"
    );
    assert!(
        kv.recurrent_state_generation(0, layer) == backend_generation,
        "conv-only CPU mutation must not advance recurrent generation"
    );
    assert!(
        kv.conv_state_generation(0, layer) > 1,
        "conv-only CPU mutation should still advance conv-state generation"
    );

    kv.recurrent_state_for_slot_mut(0, layer)[0] = 2.0;
    assert!(
        !kv.recurrent_state_cpu_stale(0, layer),
        "CPU recurrent mutation should make recurrent state materialized again"
    );
}

#[test]
fn test_qwen35_kv_layer_state_owner_tracks_cpu_backend_and_split() {
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let layer = 0usize;

    assert_eq!(
        kv.layer_state_owner(0, layer),
        Qwen3_5LayerStateOwner::CpuMaterialized
    );
    assert_eq!(
        kv.conv_state_owner(0, layer),
        Qwen3_5StateOwner::CpuMaterialized
    );
    assert_eq!(
        kv.recurrent_state_owner(0, layer),
        Qwen3_5StateOwner::CpuMaterialized
    );

    kv.note_backend_recurrent_state_update(0, layer);
    assert_eq!(
        kv.layer_state_owner(0, layer),
        Qwen3_5LayerStateOwner::Split
    );
    assert_eq!(
        kv.conv_state_owner(0, layer),
        Qwen3_5StateOwner::CpuMaterialized
    );
    assert_eq!(
        kv.recurrent_state_owner(0, layer),
        Qwen3_5StateOwner::BackendOwned
    );

    kv.note_backend_conv_state_update(0, layer);
    assert_eq!(
        kv.layer_state_owner(0, layer),
        Qwen3_5LayerStateOwner::BackendOwned
    );

    kv.conv_state_for_slot_mut(0, layer)[0] = 1.0;
    assert_eq!(
        kv.layer_state_owner(0, layer),
        Qwen3_5LayerStateOwner::Split
    );

    kv.recurrent_buffers_for_slot_mut(0, layer).1[0] = 2.0;
    assert_eq!(
        kv.layer_state_owner(0, layer),
        Qwen3_5LayerStateOwner::CpuMaterialized
    );
}

#[test]
fn test_qwen35_kv_pristine_zero_flags_track_clear_cpu_write_and_backend_write() {
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let layer = 0usize;

    assert!(kv.conv_state_pristine_zero(0, layer));
    assert!(kv.recurrent_state_pristine_zero(0, layer));

    kv.conv_state_for_slot_mut(0, layer)[0] = 1.0;
    assert!(!kv.conv_state_pristine_zero(0, layer));
    assert!(kv.recurrent_state_pristine_zero(0, layer));

    kv.clear();
    assert!(kv.conv_state_pristine_zero(0, layer));
    assert!(kv.recurrent_state_pristine_zero(0, layer));

    kv.note_backend_conv_state_update(0, layer);
    assert!(!kv.conv_state_pristine_zero(0, layer));
    assert!(kv.recurrent_state_pristine_zero(0, layer));

    kv.clear();
    kv.note_backend_recurrent_state_update(0, layer);
    assert!(kv.conv_state_pristine_zero(0, layer));
    assert!(!kv.recurrent_state_pristine_zero(0, layer));
}

#[test]
fn test_qwen35_kv_mark_layer_state_backend_owned_makes_cpu_copy_stale_without_bump() {
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let layer = 0usize;
    let conv_generation = kv.conv_state_generation(0, layer);
    let recurrent_generation = kv.recurrent_state_generation(0, layer);

    kv.mark_layer_state_backend_owned(0, layer);

    assert!(kv.conv_state_cpu_stale(0, layer));
    assert!(kv.recurrent_state_cpu_stale(0, layer));
    assert_eq!(kv.conv_state_generation(0, layer), conv_generation);
    assert_eq!(
        kv.recurrent_state_generation(0, layer),
        recurrent_generation
    );
    assert_eq!(
        kv.layer_state_owner(0, layer),
        Qwen3_5LayerStateOwner::BackendOwned
    );
}

#[test]
#[should_panic(
    expected = "cannot snapshot qwen35 recurrent slot while backend-owned recurrent state is not materialized on CPU"
)]
fn test_qwen35_kv_rejects_snapshotting_slot_with_backend_owned_recurrent_state() {
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    kv.note_backend_recurrent_state_update(0, 0);
    let _ = kv.recurrent_slot_snapshot(0);
}

#[test]
#[should_panic(
    expected = "cannot snapshot qwen35 recurrent slot while backend-owned recurrent state is not materialized on CPU"
)]
fn test_qwen35_kv_rejects_full_snapshot_with_backend_owned_recurrent_state() {
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    kv.note_backend_recurrent_state_update(0, 0);
    let _ = kv.snapshot_active_slot();
}

#[test]
fn test_qwen35_kv_full_snapshot_restores_attention_and_recurrent_state() {
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let slot1 = kv.allocate_recurrent_slot();
    kv.set_active_slot(slot1);

    kv.conv_state_for_slot_mut(slot1, 0).fill(1.5);
    kv.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
    kv.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
    kv.finalize_token();

    let snapshot = kv.snapshot_active_slot();

    kv.clear();
    kv.restore_snapshot(&snapshot);

    assert_eq!(kv.active_slot(), slot1);
    assert_eq!(kv.seq_len(), 1);
    assert_eq!(kv.attention_k_slice_including_current(3, 1), &[10.0, 11.0]);
    assert_eq!(kv.attention_v_slice_including_current(3, 1), &[12.0, 13.0]);
    assert!(kv.conv_state_for_slot(slot1, 0).iter().all(|&v| v == 1.5));
    assert!(
        kv.recurrent_state_for_slot(slot1, 0)
            .iter()
            .all(|&v| v == 2.5)
    );
    assert_eq!(kv.recurrent_seqlen_offset(slot1), 1);
}

#[test]
fn test_qwen35_kv_attention_snapshot_restores_shared_timeline_only() {
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let slot1 = kv.allocate_recurrent_slot();
    kv.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
    kv.finalize_token();

    let snapshot = kv.attention_snapshot();

    kv.attention_append(3, &[20.0, 21.0], &[22.0, 23.0]);
    kv.finalize_token();
    kv.set_recurrent_seqlen_offset(slot1, 1);
    kv.conv_state_for_slot_mut(slot1, 0).fill(7.0);
    kv.recurrent_state_for_slot_mut(slot1, 0).fill(8.0);

    kv.restore_attention_snapshot(&snapshot);

    assert_eq!(kv.seq_len(), 1);
    assert_eq!(kv.attention_k_slice_including_current(3, 1), &[10.0, 11.0]);
    assert_eq!(kv.attention_v_slice_including_current(3, 1), &[12.0, 13.0]);
    assert_eq!(kv.recurrent_seqlen_offset(slot1), 1);
    assert!(kv.conv_state_for_slot(slot1, 0).iter().all(|&v| v == 7.0));
    assert!(
        kv.recurrent_state_for_slot(slot1, 0)
            .iter()
            .all(|&v| v == 8.0)
    );
}

#[test]
fn test_qwen35_kv_sync_attention_cpu_from_gpu_if_needed_restores_cpu_mirror() {
    let device = ax_engine_metal::MetalDevice::new().expect("metal device");
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    kv.enable_gpu_attention(&device, GpuKvDtype::F32)
        .expect("enable gpu attention");

    let k = [10.0f32, 11.0];
    let v = [12.0f32, 13.0];
    let gpu_attention = kv
        .attention_gpu
        .as_mut()
        .expect("gpu attention should exist");
    gpu_attention.append_layer(3, &k, &v);
    gpu_attention.finalize_token();

    kv.mark_attention_cpu_dirty();
    kv.finalize_token();

    assert_eq!(kv.attention_k_slice_including_current(3, 1), &[0.0, 0.0]);
    assert_eq!(kv.attention_v_slice_including_current(3, 1), &[0.0, 0.0]);

    kv.sync_attention_cpu_from_gpu_if_needed();

    assert_eq!(kv.attention_k_slice_including_current(3, 1), &k);
    assert_eq!(kv.attention_v_slice_including_current(3, 1), &v);
}

#[test]
fn test_qwen35_kv_incremental_attention_cpu_sync_preserves_valid_prefix() {
    let device = ax_engine_metal::MetalDevice::new().expect("metal device");
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    kv.enable_gpu_attention(&device, GpuKvDtype::F32)
        .expect("enable gpu attention");

    kv.attention_append(3, &[1.0, 2.0], &[3.0, 4.0]);
    kv.finalize_token();

    let gpu_attention = kv
        .attention_gpu
        .as_mut()
        .expect("gpu attention should exist");
    gpu_attention.append_layer(3, &[5.0, 6.0], &[7.0, 8.0]);
    gpu_attention.finalize_token();

    kv.mark_attention_cpu_dirty();
    kv.finalize_token();
    kv.sync_attention_cpu_from_gpu_if_needed();

    assert_eq!(
        kv.attention_k_slice_including_current(3, 2),
        &[1.0, 2.0, 5.0, 6.0]
    );
    assert_eq!(
        kv.attention_v_slice_including_current(3, 2),
        &[3.0, 4.0, 7.0, 8.0]
    );
}

#[test]
fn test_qwen35_kv_truncate_attention_to_preserves_valid_cpu_prefix_when_gpu_dirty() {
    let device = ax_engine_metal::MetalDevice::new().expect("metal device");
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    kv.enable_gpu_attention(&device, GpuKvDtype::F32)
        .expect("enable gpu attention");

    kv.attention_append(3, &[1.0, 2.0], &[3.0, 4.0]);
    kv.finalize_token();

    {
        let gpu_attention = kv.gpu_attention_mut().expect("gpu attention should exist");
        gpu_attention.append_layer(3, &[5.0, 6.0], &[7.0, 8.0]);
        gpu_attention.finalize_token();
    }
    kv.mark_attention_cpu_dirty();
    kv.finalize_token();

    kv.truncate_attention_to(1);

    assert_eq!(kv.seq_len(), 1);
    assert_eq!(kv.attention_k_slice_including_current(3, 1), &[1.0, 2.0]);
    assert_eq!(kv.attention_v_slice_including_current(3, 1), &[3.0, 4.0]);
    kv.sync_attention_cpu_from_gpu_if_needed();
    assert_eq!(kv.attention_k_slice_including_current(3, 1), &[1.0, 2.0]);
    assert_eq!(kv.attention_v_slice_including_current(3, 1), &[3.0, 4.0]);
}

#[test]
#[should_panic(
    expected = "cannot snapshot inactive qwen35 recurrent slot while attention KV is shared"
)]
fn test_qwen35_kv_rejects_full_snapshot_for_inactive_slot() {
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let slot1 = kv.allocate_recurrent_slot();
    let _ = kv.snapshot_slot(slot1);
}

#[test]
#[should_panic(
    expected = "cannot set active qwen35 recurrent slot seqlen_offset 1 != shared attention seq_len 0"
)]
fn test_qwen35_kv_rejects_setting_active_slot_seq_len_offset_to_mismatched_shared_seq_len() {
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    kv.set_recurrent_seqlen_offset(0, 1);
}

#[test]
#[should_panic(
    expected = "cannot restore active qwen35 recurrent slot with seqlen_offset 1 != shared attention seq_len 0"
)]
fn test_qwen35_kv_rejects_restoring_active_slot_with_mismatched_shared_seq_len() {
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let mut snapshot = kv.recurrent_slot_snapshot(0);
    snapshot.seqlen_offset = 1;
    kv.restore_recurrent_slot(0, &snapshot);
}

#[test]
fn test_qwen35_kv_full_snapshot_restores_into_fresh_kv() {
    let mut source = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let slot1 = source.allocate_recurrent_slot();
    source.set_active_slot(slot1);

    source.conv_state_for_slot_mut(slot1, 0).fill(1.5);
    source.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
    source.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
    source.finalize_token();

    let snapshot = source.snapshot_active_slot();

    let mut restored = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    restored.restore_snapshot(&snapshot);

    assert_eq!(restored.active_slot(), slot1);
    assert_eq!(restored.seq_len(), 1);
    assert_eq!(
        restored.attention_k_slice_including_current(3, 1),
        &[10.0, 11.0]
    );
    assert_eq!(
        restored.attention_v_slice_including_current(3, 1),
        &[12.0, 13.0]
    );
    assert!(
        restored
            .conv_state_for_slot(slot1, 0)
            .iter()
            .all(|&v| v == 1.5)
    );
    assert!(
        restored
            .recurrent_state_for_slot(slot1, 0)
            .iter()
            .all(|&v| v == 2.5)
    );
    assert_eq!(restored.recurrent_seqlen_offset(slot1), 1);
}

#[test]
fn test_qwen35_kv_restore_snapshot_syncs_gpu_recurrent_state() {
    let device = ax_engine_metal::MetalDevice::new().expect("metal device");

    let mut source = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let slot1 = source.allocate_recurrent_slot();
    source.set_active_slot(slot1);
    source.conv_state_for_slot_mut(slot1, 0).fill(1.5);
    source.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
    source.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
    source.finalize_token();
    let snapshot = source.snapshot_active_slot();

    let mut restored = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    restored
        .enable_gpu_recurrent_state(&device)
        .expect("enable gpu recurrent state");
    restored.restore_snapshot(&snapshot);

    let (conv_buf, rec_buf) = restored
        .gpu_recurrent_buffers(slot1, 0)
        .expect("gpu recurrent buffers should exist after restore");
    let conv_len = restored.conv_state_for_slot(slot1, 0).len();
    let recurrent_len = restored.recurrent_state_for_slot(slot1, 0).len();
    unsafe {
        assert!(
            conv_buf.as_slice::<f32>()[..conv_len]
                .iter()
                .all(|&value| value == 1.5)
        );
        assert!(
            rec_buf.as_slice::<f32>()[..recurrent_len]
                .iter()
                .all(|&value| value == 2.5)
        );
    }
}

#[test]
fn test_qwen35_kv_snapshot_active_slot_syncs_gpu_attention_before_capture() {
    let device = ax_engine_metal::MetalDevice::new().expect("metal device");
    let mut kv = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let slot1 = kv.allocate_recurrent_slot();
    kv.set_active_slot(slot1);
    kv.enable_gpu_attention(&device, GpuKvDtype::F32)
        .expect("enable gpu attention");

    let k = [10.0f32, 11.0];
    let v = [12.0f32, 13.0];
    let gpu_attention = kv
        .attention_gpu
        .as_mut()
        .expect("gpu attention should exist");
    gpu_attention.append_layer(3, &k, &v);
    gpu_attention.finalize_token();

    kv.mark_attention_cpu_dirty();
    kv.finalize_token();

    let snapshot = kv.snapshot_active_slot();

    let mut restored = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    restored.restore_snapshot(&snapshot);

    assert_eq!(restored.seq_len(), 1);
    assert_eq!(restored.attention_k_slice_including_current(3, 1), &k);
    assert_eq!(restored.attention_v_slice_including_current(3, 1), &v);
}

#[test]
fn test_qwen35_kv_restore_snapshot_clears_unrelated_allocated_slots() {
    let mut source = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let slot1 = source.allocate_recurrent_slot();
    source.set_active_slot(slot1);
    source.conv_state_for_slot_mut(slot1, 0).fill(1.5);
    source.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
    source.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
    source.finalize_token();
    let snapshot = source.snapshot_active_slot();

    let mut restored = Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let slot1_restored = restored.allocate_recurrent_slot();
    let slot2 = restored.allocate_recurrent_slot();
    assert_eq!(slot1_restored, slot1);
    restored.conv_state_for_slot_mut(slot2, 0).fill(7.0);
    restored.recurrent_state_for_slot_mut(slot2, 0).fill(8.0);

    restored.restore_snapshot(&snapshot);

    assert_eq!(restored.active_slot(), slot1);
    assert!(!restored.has_recurrent_slot(slot2));
    let reused = restored.allocate_recurrent_slot();
    assert_eq!(reused, slot2);
    assert!(
        restored
            .conv_state_for_slot(reused, 0)
            .iter()
            .all(|&v| v == 0.0)
    );
    assert!(
        restored
            .recurrent_state_for_slot(reused, 0)
            .iter()
            .all(|&v| v == 0.0)
    );
}

#[test]
fn test_qwen35_kv_27b_shaped_snapshot_restore_clears_unrelated_slots() {
    let mut source = Qwen3_5Kv::new(
        4,   // keep layer count small while exercising 27B-shaped recurrent strides
        4,   // Qwen3.5-27B KV heads
        256, // Qwen3.5-27B head dim
        16, 4, 4, 6144, // 48 value heads * 128 value head dim
        128,  // linear key head dim
        48,   // linear value heads
        16,   // linear key heads / group count
    );
    let slot1 = source.allocate_recurrent_slot();
    source.set_active_slot(slot1);
    source.conv_state_for_slot_mut(slot1, 0).fill(0.25);
    source.recurrent_state_for_slot_mut(slot1, 0).fill(0.5);
    source.attention_append(3, &vec![1.0; 4 * 256], &vec![2.0; 4 * 256]);
    source.finalize_token();
    let snapshot = source.snapshot_active_slot();

    let mut restored = Qwen3_5Kv::new(4, 4, 256, 16, 4, 4, 6144, 128, 48, 16);
    let slot1_restored = restored.allocate_recurrent_slot();
    let slot2 = restored.allocate_recurrent_slot();
    assert_eq!(slot1_restored, slot1);
    restored.conv_state_for_slot_mut(slot2, 0).fill(7.0);
    restored.recurrent_state_for_slot_mut(slot2, 0).fill(8.0);

    restored.restore_snapshot(&snapshot);

    assert_eq!(restored.active_slot(), slot1);
    assert!(!restored.has_recurrent_slot(slot2));
    assert_eq!(
        restored.attention_k_slice_including_current(3, 1)[..8],
        [1.0; 8]
    );
    assert_eq!(
        restored.attention_v_slice_including_current(3, 1)[..8],
        [2.0; 8]
    );
}

#[test]
#[should_panic(expected = "qwen35 recurrent slot batch must not contain duplicate slots")]
fn test_qwen35_kv_rejects_gathering_duplicate_slot_batch() {
    let kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let conv_state_len = kv.conv_state_for_slot(0, 0).len();
    let recurrent_state_len = kv.recurrent_state_for_slot(0, 0).len();
    let mut conv_batch = vec![0.0; 2 * conv_state_len];
    let mut recurrent_batch = vec![0.0; 2 * recurrent_state_len];
    kv.gather_recurrent_state_batch(&[0, 0], 0, &mut conv_batch, &mut recurrent_batch);
}

#[test]
#[should_panic(
    expected = "qwen35 recurrent slot batch slot 1 has seqlen_offset 1 != shared attention seq_len 0"
)]
fn test_qwen35_kv_rejects_gathering_slot_batch_with_misaligned_shared_seq_len() {
    let mut kv = Qwen3_5Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
    let slot1 = kv.allocate_recurrent_slot();
    kv.set_recurrent_seqlen_offset(slot1, 1);
    let conv_state_len = kv.conv_state_for_slot(0, 0).len();
    let recurrent_state_len = kv.recurrent_state_for_slot(0, 0).len();
    let mut conv_batch = vec![0.0; 2 * conv_state_len];
    let mut recurrent_batch = vec![0.0; 2 * recurrent_state_len];
    kv.gather_recurrent_state_batch(&[0, slot1], 0, &mut conv_batch, &mut recurrent_batch);
}
