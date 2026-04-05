use super::*;

#[test]
fn test_model_kv_cpu_snapshot_round_trip() {
    let mut kv = ModelKv::Cpu(CpuKv::new(1, 1, 2, 16));
    if let ModelKv::Cpu(cpu_kv) = &mut kv {
        cpu_kv.append_and_advance(0, &[1.0, 2.0], &[3.0, 4.0]);
        cpu_kv.finalize_token();
    }

    let snapshot = kv.snapshot().expect("cpu kv should support snapshots");
    kv.clear();
    kv.restore_snapshot(&snapshot)
        .expect("cpu restore should succeed");

    let restored = kv.as_cpu_mut().expect("expected cpu kv");
    assert_eq!(restored.seq_len(), 1);
    assert_eq!(restored.k_slice(0, 1), &[1.0, 2.0]);
    assert_eq!(restored.v_slice(0, 1), &[3.0, 4.0]);
}

#[test]
fn test_model_kv_qwen35_snapshot_round_trip() {
    let mut kv = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));
    if let ModelKv::Qwen35(qwen_kv) = &mut kv {
        let slot1 = qwen_kv.allocate_recurrent_slot();
        qwen_kv.set_active_slot(slot1);
        qwen_kv.conv_state_for_slot_mut(slot1, 0).fill(1.5);
        qwen_kv.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
        qwen_kv.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
        qwen_kv.finalize_token();
    }

    let snapshot = kv.snapshot().expect("qwen35 kv should support snapshots");
    kv.clear();
    kv.restore_snapshot(&snapshot)
        .expect("qwen35 restore should succeed");

    let restored = kv.as_qwen35_mut().expect("expected qwen35 kv");
    assert_eq!(restored.active_slot(), 1);
    assert_eq!(restored.seq_len(), 1);
    assert_eq!(
        restored.attention_k_slice_including_current(3, 1),
        &[10.0, 11.0]
    );
    assert_eq!(
        restored.attention_v_slice_including_current(3, 1),
        &[12.0, 13.0]
    );
    assert!(restored.conv_state_for_slot(1, 0).iter().all(|&v| v == 1.5));
    assert!(
        restored
            .recurrent_state_for_slot(1, 0)
            .iter()
            .all(|&v| v == 2.5)
    );
}

#[test]
fn test_model_kv_qwen3_5_alias_accessors_match_qwen35_variant() {
    let mut kv = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));

    assert!(kv.as_qwen3_5().is_some());
    assert!(kv.as_qwen3_5_mut().is_some());
    assert_eq!(kv.as_qwen3_5().expect("expected qwen3_5 kv").seq_len(), 0);
}

#[test]
fn test_model_kv_qwen35_snapshot_restores_into_fresh_kv() {
    let mut source = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));
    if let ModelKv::Qwen35(qwen_kv) = &mut source {
        let slot1 = qwen_kv.allocate_recurrent_slot();
        qwen_kv.set_active_slot(slot1);
        qwen_kv.conv_state_for_slot_mut(slot1, 0).fill(1.5);
        qwen_kv.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
        qwen_kv.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
        qwen_kv.finalize_token();
    }

    let snapshot = source
        .snapshot()
        .expect("qwen35 kv should support snapshots");

    let mut restored = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));
    restored
        .restore_snapshot(&snapshot)
        .expect("qwen35 restore into fresh kv should succeed");

    let restored_qwen = restored.as_qwen35_mut().expect("expected qwen35 kv");
    assert_eq!(restored_qwen.active_slot(), 1);
    assert_eq!(restored_qwen.seq_len(), 1);
    assert_eq!(
        restored_qwen.attention_k_slice_including_current(3, 1),
        &[10.0, 11.0]
    );
    assert_eq!(
        restored_qwen.attention_v_slice_including_current(3, 1),
        &[12.0, 13.0]
    );
    assert!(
        restored_qwen
            .conv_state_for_slot(1, 0)
            .iter()
            .all(|&v| v == 1.5)
    );
    assert!(
        restored_qwen
            .recurrent_state_for_slot(1, 0)
            .iter()
            .all(|&v| v == 2.5)
    );
}

#[test]
fn test_model_kv_gpu_snapshot_is_unsupported() {
    assert!(matches!(
        ModelKv::Cpu(CpuKv::new(1, 1, 2, 16)).snapshot(),
        Some(ModelKvSnapshot::Cpu(_))
    ));
}

#[test]
fn test_model_kv_restore_rejects_mismatched_snapshot_variant() {
    let mut kv = ModelKv::Cpu(CpuKv::new(1, 1, 2, 16));
    let qwen_snapshot = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)))
        .snapshot()
        .expect("qwen35 snapshot should exist");
    let err = kv
        .restore_snapshot(&qwen_snapshot)
        .expect_err("mismatched restore should fail");
    assert!(
        err.to_string()
            .contains("cannot restore qwen35 snapshot into cpu kv")
    );
}

#[test]
fn test_model_kv_with_qwen35_batch_slot_indices_clears_after_success() {
    let mut kv = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));
    let slot1 = kv
        .as_qwen35_mut()
        .expect("expected qwen35 kv")
        .allocate_recurrent_slot();

    kv.with_qwen35_batch_slot_indices(&[0, slot1], |kv| {
        let qwen_kv = kv.as_qwen35().expect("expected qwen35 kv");
        assert_eq!(qwen_kv.recurrent_batch_slot_indices().as_ref(), &[0, slot1]);
        Ok(())
    })
    .expect("scoped qwen35 batch slots should succeed");

    let qwen_kv = kv.as_qwen35().expect("expected qwen35 kv");
    assert_eq!(qwen_kv.recurrent_batch_slot_indices().as_ref(), &[0]);
}

#[test]
fn test_model_kv_with_qwen35_batch_slot_indices_clears_after_error() {
    let mut kv = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));
    let slot1 = kv
        .as_qwen35_mut()
        .expect("expected qwen35 kv")
        .allocate_recurrent_slot();

    let err = kv
        .with_qwen35_batch_slot_indices(&[0, slot1], |_kv| -> anyhow::Result<()> {
            anyhow::bail!("boom")
        })
        .expect_err("closure error should propagate");
    assert!(err.to_string().contains("boom"));

    let qwen_kv = kv.as_qwen35().expect("expected qwen35 kv");
    assert_eq!(qwen_kv.recurrent_batch_slot_indices().as_ref(), &[0]);
}

#[test]
fn test_model_kv_with_qwen35_batch_slot_indices_rejects_non_qwen35_kv() {
    let mut kv = ModelKv::Cpu(CpuKv::new(1, 1, 2, 16));
    let err = kv
        .with_qwen35_batch_slot_indices(&[0], |_kv| Ok(()))
        .expect_err("non-qwen35 kv should be rejected");
    assert!(
        err.to_string()
            .contains("qwen35 batch slots require ModelKv::Qwen35")
    );
}

#[test]
fn test_model_kv_with_qwen35_shared_timeline_branches_restores_active_and_frees_slots() {
    let mut kv = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));

    kv.with_qwen35_shared_timeline_branches(3, |kv, slot_indices| {
        assert_eq!(slot_indices, &[0, 1, 2]);
        kv.set_qwen35_active_slot(slot_indices[1])?;
        Ok(())
    })
    .expect("scoped qwen35 branches should succeed");

    assert_eq!(
        kv.qwen35_active_slot()
            .expect("qwen35 active slot query should succeed"),
        0
    );
    let reused = kv
        .allocate_qwen35_recurrent_slot()
        .expect("freed qwen35 branch slot should be reusable");
    assert_eq!(reused, 1);
}

#[test]
fn test_model_kv_with_qwen35_shared_timeline_branches_cleans_up_after_error() {
    let mut kv = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));

    let err = kv
        .with_qwen35_shared_timeline_branches(2, |kv, slot_indices| -> anyhow::Result<()> {
            kv.set_qwen35_active_slot(slot_indices[1])?;
            anyhow::bail!("boom");
        })
        .expect_err("closure error should propagate");
    assert!(err.to_string().contains("boom"));

    assert_eq!(
        kv.qwen35_active_slot()
            .expect("qwen35 active slot query should succeed"),
        0
    );
    let reused = kv
        .allocate_qwen35_recurrent_slot()
        .expect("freed qwen35 branch slot should be reusable");
    assert_eq!(reused, 1);
}

#[test]
fn test_model_kv_with_qwen35_shared_timeline_branches_from_slot_restores_original_active() {
    let mut kv = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));
    let source_slot = kv
        .allocate_qwen35_recurrent_slot()
        .expect("qwen35 slot allocation should succeed");

    {
        let qwen_kv = kv.as_qwen35_mut().expect("expected qwen35 kv");
        qwen_kv.set_recurrent_seqlen_offset(source_slot, 0);
    }

    kv.with_qwen35_shared_timeline_branches_from_slot(source_slot, 2, |kv, slot_indices| {
        assert_eq!(slot_indices, &[source_slot, 2]);
        assert_eq!(
            kv.qwen35_active_slot()
                .expect("qwen35 active slot query should succeed"),
            source_slot
        );
        Ok(())
    })
    .expect("scoped qwen35 branches from slot should succeed");

    assert_eq!(
        kv.qwen35_active_slot()
            .expect("qwen35 active slot query should succeed"),
        0
    );
    let reused = kv
        .allocate_qwen35_recurrent_slot()
        .expect("freed qwen35 branch slot should be reusable");
    assert_eq!(reused, 2);
}

#[test]
fn test_model_kv_with_qwen35_shared_timeline_branches_from_slot_commits_back_to_original_active() {
    let mut kv = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));
    let source_slot = kv
        .allocate_qwen35_recurrent_slot()
        .expect("qwen35 slot allocation should succeed");

    kv.with_qwen35_shared_timeline_branches_from_slot(source_slot, 2, |kv, slot_indices| {
        let qwen_kv = kv.as_qwen35_mut().expect("expected qwen35 kv");
        qwen_kv.finalize_batch_for_slots(slot_indices, 1);
        Ok(())
    })
    .expect("scoped qwen35 branches from slot should succeed");

    let qwen_kv = kv.as_qwen35().expect("expected qwen35 kv");
    assert_eq!(qwen_kv.active_slot(), 0);
    assert_eq!(qwen_kv.seq_len(), 1);
    assert_eq!(qwen_kv.recurrent_seqlen_offset(0), 1);
}

#[test]
fn test_model_kv_with_qwen35_shared_timeline_branches_rejects_non_qwen35_kv() {
    let mut kv = ModelKv::Cpu(CpuKv::new(1, 1, 2, 16));
    let err = kv
        .with_qwen35_shared_timeline_branches(2, |_kv, _slots| Ok(()))
        .expect_err("non-qwen35 kv should be rejected");
    assert!(
        err.to_string()
            .contains("qwen35 active slot requires ModelKv::Qwen35")
    );
}

#[test]
fn test_model_kv_qwen35_slot_lifecycle_wrappers() {
    let mut kv = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));
    let slot1 = kv
        .allocate_qwen35_recurrent_slot()
        .expect("qwen35 slot allocation should succeed");
    assert_eq!(slot1, 1);

    kv.set_qwen35_active_slot(slot1)
        .expect("qwen35 active slot switch should succeed");
    assert_eq!(
        kv.qwen35_active_slot()
            .expect("qwen35 active slot query should succeed"),
        slot1
    );

    kv.set_qwen35_active_slot(0)
        .expect("returning to slot 0 should succeed");
    kv.free_qwen35_recurrent_slot(slot1)
        .expect("qwen35 slot free should succeed");
}

#[test]
fn test_model_kv_qwen35_recurrent_slot_snapshot_wrappers_round_trip() {
    let mut kv = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));
    let slot1 = kv
        .allocate_qwen35_recurrent_slot()
        .expect("qwen35 slot allocation should succeed");

    {
        let qwen_kv = kv.as_qwen35_mut().expect("expected qwen35 kv");
        qwen_kv.conv_state_for_slot_mut(slot1, 0).fill(1.25);
        qwen_kv.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
        qwen_kv.set_recurrent_seqlen_offset(slot1, 7);
    }

    let snapshot = kv
        .snapshot_qwen35_recurrent_slot(slot1)
        .expect("qwen35 recurrent slot snapshot should succeed");

    {
        let qwen_kv = kv.as_qwen35_mut().expect("expected qwen35 kv");
        qwen_kv.conv_state_for_slot_mut(slot1, 0).fill(0.0);
        qwen_kv.recurrent_state_for_slot_mut(slot1, 0).fill(0.0);
        qwen_kv.set_recurrent_seqlen_offset(slot1, 0);
    }

    kv.restore_qwen35_recurrent_slot(slot1, &snapshot)
        .expect("qwen35 recurrent slot restore should succeed");

    let qwen_kv = kv.as_qwen35().expect("expected qwen35 kv");
    assert!(
        qwen_kv
            .conv_state_for_slot(slot1, 0)
            .iter()
            .all(|&v| v == 1.25)
    );
    assert!(
        qwen_kv
            .recurrent_state_for_slot(slot1, 0)
            .iter()
            .all(|&v| v == 2.5)
    );
    assert_eq!(qwen_kv.recurrent_seqlen_offset(slot1), 7);
}

#[test]
fn test_model_kv_qwen35_attention_snapshot_wrapper_round_trip() {
    let mut kv = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));
    let slot1 = kv
        .allocate_qwen35_recurrent_slot()
        .expect("qwen35 slot allocation should succeed");

    {
        let qwen_kv = kv.as_qwen35_mut().expect("expected qwen35 kv");
        qwen_kv.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
        qwen_kv.finalize_token();
    }

    let snapshot = kv
        .snapshot_qwen35_attention_timeline()
        .expect("qwen35 attention snapshot should succeed");

    {
        let qwen_kv = kv.as_qwen35_mut().expect("expected qwen35 kv");
        qwen_kv.attention_append(3, &[20.0, 21.0], &[22.0, 23.0]);
        qwen_kv.finalize_token();
        qwen_kv.set_recurrent_seqlen_offset(slot1, 1);
    }

    kv.restore_qwen35_attention_timeline(&snapshot)
        .expect("qwen35 attention restore should succeed");

    let qwen_kv = kv.as_qwen35().expect("expected qwen35 kv");
    assert_eq!(qwen_kv.seq_len(), 1);
    assert_eq!(
        qwen_kv.attention_k_slice_including_current(3, 1),
        &[10.0, 11.0]
    );
    assert_eq!(
        qwen_kv.attention_v_slice_including_current(3, 1),
        &[12.0, 13.0]
    );
    assert_eq!(qwen_kv.recurrent_seqlen_offset(slot1), 1);
}

#[test]
fn test_model_kv_qwen35_clone_recurrent_slot_wrapper() {
    let mut kv = ModelKv::Qwen35(Box::new(Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2)));
    let slot1 = kv
        .allocate_qwen35_recurrent_slot()
        .expect("qwen35 slot allocation should succeed");
    let slot2 = kv
        .allocate_qwen35_recurrent_slot()
        .expect("qwen35 slot allocation should succeed");

    {
        let qwen_kv = kv.as_qwen35_mut().expect("expected qwen35 kv");
        qwen_kv.conv_state_for_slot_mut(slot1, 0).fill(1.25);
        qwen_kv.recurrent_state_for_slot_mut(slot1, 0).fill(2.75);
        qwen_kv.set_recurrent_seqlen_offset(slot1, 5);
    }

    kv.clone_qwen35_recurrent_slot(slot1, slot2)
        .expect("qwen35 slot clone should succeed");

    let qwen_kv = kv.as_qwen35().expect("expected qwen35 kv");
    assert!(
        qwen_kv
            .conv_state_for_slot(slot2, 0)
            .iter()
            .all(|&v| v == 1.25)
    );
    assert!(
        qwen_kv
            .recurrent_state_for_slot(slot2, 0)
            .iter()
            .all(|&v| v == 2.75)
    );
    assert_eq!(qwen_kv.recurrent_seqlen_offset(slot2), 5);
}

#[test]
fn test_model_kv_qwen35_slot_wrappers_reject_non_qwen35_kv() {
    let mut kv = ModelKv::Cpu(CpuKv::new(1, 1, 2, 16));
    let err = kv
        .allocate_qwen35_recurrent_slot()
        .expect_err("non-qwen35 kv should reject slot allocation");
    assert!(
        err.to_string()
            .contains("qwen35 recurrent slots require ModelKv::Qwen35")
    );

    let err = kv
        .qwen35_active_slot()
        .expect_err("non-qwen35 kv should reject active slot query");
    assert!(
        err.to_string()
            .contains("qwen35 active slot requires ModelKv::Qwen35")
    );
}
