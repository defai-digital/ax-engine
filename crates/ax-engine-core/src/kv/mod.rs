//! KV cache — single-owner design.
//!
//! # v2 Architecture Decision
//!
//! v1 had a split-brain design: `KvCache` (CPU) + `GpuKvCache` (GPU) with an
//! `advance_by()` call to keep seq_len in sync when the GPU path ran prefill
//! but the CPU path ran decode. This caused a P1 panic:
//!
//!   GPU batch prefill → `kv_cache.advance_by(209)` increments seq_len to 209
//!   CPU decode → `kv_cache.append()` tries to write at offset=209*stride
//!   KvCache buffer capacity was still 128 → out-of-range slice panic
//!
//! v2 solution: `ModelKv` is a single enum. The forward pass holds one `ModelKv`.
//! The backend creates the right variant at model initialization.
//!
//! - `MetalBackend` / `HybridBackend` → `ModelKv::Gpu`
//! - `CpuBackend` / `HybridCpuDecode` → `ModelKv::Cpu`
//!
//! `forward_batch_gpu_unified` takes `&mut GpuKv` directly — it is
//! architecturally impossible to have GPU batch prefill + CPU decode in the
//! same pass. If the caller holds `ModelKv::Cpu`, `forward_batch` falls
//! through to serial prefill automatically.
//!
//! There is no `advance_by()`.

use anyhow::bail;

pub mod cpu_kv;
pub mod gpu_kv;
pub mod page;
pub mod qwen35_kv;

pub use cpu_kv::{CpuKv, CpuKvSnapshot};
pub use gpu_kv::{GpuKv, GpuKvDtype};
pub(crate) use qwen35_kv::Qwen35LayerStateOwner;
pub use qwen35_kv::{
    Qwen35AttentionSnapshot, Qwen35Kv, Qwen35KvSnapshot, Qwen35RecurrentSlotSnapshot,
};

#[derive(Debug, Clone)]
pub enum ModelKvSnapshot {
    Cpu(CpuKvSnapshot),
    Qwen35(Qwen35KvSnapshot),
}

/// Single-owner KV cache. Eliminates the v1 CPU+GPU split-brain.
pub enum ModelKv {
    /// CPU RAM storage (f32). Used with CpuBackend or HybridCpuDecode.
    Cpu(CpuKv),
    /// GPU Metal buffer storage (f32 or f16). Used with MetalBackend or HybridBackend.
    Gpu(GpuKv),
    /// Hybrid recurrent state + attention KV for Qwen3.5.
    Qwen35(Qwen35Kv),
}

impl ModelKv {
    /// Current sequence length (number of tokens with KV stored).
    pub fn seq_len(&self) -> usize {
        match self {
            Self::Cpu(c) => c.seq_len(),
            Self::Gpu(g) => g.seq_len(),
            Self::Qwen35(q) => q.seq_len(),
        }
    }

    /// Attempt to get a mutable reference to the GPU KV cache.
    ///
    /// Returns `None` if this is a CPU KV cache. Forward passes use this to
    /// decide whether to take the GPU batch path — if None, they fall through
    /// to serial CPU prefill.
    pub fn as_gpu_mut(&mut self) -> Option<&mut GpuKv> {
        match self {
            Self::Gpu(g) => Some(g),
            Self::Cpu(_) => None,
            Self::Qwen35(_) => None,
        }
    }

    /// Attempt to get an immutable reference to the GPU KV cache.
    pub fn as_gpu(&self) -> Option<&GpuKv> {
        match self {
            Self::Gpu(g) => Some(g),
            Self::Cpu(_) => None,
            Self::Qwen35(_) => None,
        }
    }

    /// Get a reference to the CPU KV cache.
    ///
    /// Returns `None` if this is a GPU KV cache.
    pub fn as_cpu_mut(&mut self) -> Option<&mut CpuKv> {
        match self {
            Self::Cpu(c) => Some(c),
            Self::Gpu(_) => None,
            Self::Qwen35(_) => None,
        }
    }

    /// Get a reference to the Qwen3.5 hybrid cache/state.
    pub fn as_qwen35(&self) -> Option<&Qwen35Kv> {
        match self {
            Self::Qwen35(q) => Some(q),
            Self::Cpu(_) | Self::Gpu(_) => None,
        }
    }

    /// Get a mutable reference to the Qwen3.5 hybrid cache/state.
    pub fn as_qwen35_mut(&mut self) -> Option<&mut Qwen35Kv> {
        match self {
            Self::Qwen35(q) => Some(q),
            Self::Cpu(_) | Self::Gpu(_) => None,
        }
    }

    /// True if KV data is GPU-resident.
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu(_))
    }

    /// Snapshot the current KV state when the backing implementation supports it.
    ///
    /// GPU KV snapshotting is not supported yet because AX does not maintain a
    /// CPU mirror of Metal buffers in v2.
    pub fn snapshot(&self) -> Option<ModelKvSnapshot> {
        match self {
            Self::Cpu(c) => Some(ModelKvSnapshot::Cpu(c.snapshot())),
            Self::Gpu(_) => None,
            Self::Qwen35(q) => Some(ModelKvSnapshot::Qwen35(q.snapshot_active_slot())),
        }
    }

    /// Restore a previously captured KV snapshot.
    pub fn restore_snapshot(&mut self, snapshot: &ModelKvSnapshot) -> anyhow::Result<()> {
        match (self, snapshot) {
            (Self::Cpu(kv), ModelKvSnapshot::Cpu(snapshot)) => {
                kv.restore(snapshot);
                Ok(())
            }
            (Self::Qwen35(kv), ModelKvSnapshot::Qwen35(snapshot)) => {
                kv.restore_snapshot(snapshot);
                Ok(())
            }
            (Self::Gpu(_), _) => bail!("GPU KV snapshot restore is not supported"),
            (Self::Cpu(_), ModelKvSnapshot::Qwen35(_)) => {
                bail!("cannot restore qwen35 snapshot into cpu kv")
            }
            (Self::Qwen35(_), ModelKvSnapshot::Cpu(_)) => {
                bail!("cannot restore cpu snapshot into qwen35 kv")
            }
        }
    }

    /// Reset to zero tokens (keeps allocated memory/buffers).
    pub fn clear(&mut self) {
        match self {
            Self::Cpu(c) => c.clear(),
            Self::Gpu(g) => g.clear(),
            Self::Qwen35(q) => q.clear(),
        }
    }

    /// Rewind seq_len to `pos` without freeing memory or Metal buffers.
    ///
    /// Used by speculative decoding to roll back the KV state after a rejected
    /// draft token. Data beyond `pos` is overwritten on the next append.
    pub fn truncate_to(&mut self, pos: usize) {
        match self {
            Self::Cpu(c) => c.truncate_to(pos),
            Self::Gpu(g) => g.truncate_to(pos),
            Self::Qwen35(q) => q.truncate_to(pos),
        }
    }

    /// Temporarily apply a Qwen3.5 shared-timeline recurrent slot batch for one operation.
    ///
    /// This is intended for fan-out style prefill where the same token stream
    /// should advance multiple recurrent slots while attention KV remains shared.
    /// The configured slot batch is cleared before returning, even if `f`
    /// returns an error.
    pub fn with_qwen35_batch_slot_indices<R>(
        &mut self,
        slot_indices: &[usize],
        f: impl FnOnce(&mut Self) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        {
            let qwen_kv = self
                .as_qwen35_mut()
                .ok_or_else(|| anyhow::anyhow!("qwen35 batch slots require ModelKv::Qwen35"))?;
            qwen_kv.set_batch_slot_indices(slot_indices);
        }

        let result = f(self);

        if let Some(qwen_kv) = self.as_qwen35_mut() {
            qwen_kv.clear_batch_slot_indices();
        }

        result
    }

    /// Temporarily fork shared-timeline Qwen3.5 recurrent branches from the active slot.
    ///
    /// The closure receives a slot list containing the source active slot
    /// followed by any freshly allocated branch slots. All newly allocated
    /// branch slots are freed before returning, even if `f` returns an error.
    pub fn with_qwen35_shared_timeline_branches<R>(
        &mut self,
        slot_count: usize,
        f: impl FnOnce(&mut Self, &[usize]) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        let source_slot = self.qwen35_active_slot()?;
        self.with_qwen35_shared_timeline_branches_from_slot(source_slot, slot_count, f)
    }

    /// Temporarily fork shared-timeline Qwen3.5 recurrent branches from a specific source slot.
    ///
    /// The helper temporarily switches the active slot to `source_slot` for the
    /// duration of the closure, allocates and clones any additional branch
    /// slots, then restores the original active slot and frees the temporary
    /// branch slots before returning.
    pub fn with_qwen35_shared_timeline_branches_from_slot<R>(
        &mut self,
        source_slot: usize,
        slot_count: usize,
        f: impl FnOnce(&mut Self, &[usize]) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        anyhow::ensure!(
            slot_count > 0,
            "qwen35 shared timeline slot count must be >= 1"
        );

        let original_active_slot = self.qwen35_active_slot()?;
        let mut branch_slots = Vec::with_capacity(slot_count.saturating_sub(1));
        let mut slot_indices = Vec::with_capacity(slot_count);
        slot_indices.push(source_slot);

        {
            let qwen_kv = self.as_qwen35_mut().ok_or_else(|| {
                anyhow::anyhow!("qwen35 shared timeline requires ModelKv::Qwen35")
            })?;
            qwen_kv.set_active_slot(source_slot);
            for _ in 1..slot_count {
                let slot_idx = qwen_kv.allocate_recurrent_slot();
                qwen_kv.clone_recurrent_slot(source_slot, slot_idx);
                branch_slots.push(slot_idx);
                slot_indices.push(slot_idx);
            }
        }

        let result = f(self, &slot_indices);

        if let Some(qwen_kv) = self.as_qwen35_mut() {
            qwen_kv.clear_batch_slot_indices();
            if original_active_slot != source_slot
                && qwen_kv.has_recurrent_slot(source_slot)
                && qwen_kv.has_recurrent_slot(original_active_slot)
            {
                qwen_kv.clone_recurrent_slot(source_slot, original_active_slot);
            }
            if qwen_kv.has_recurrent_slot(original_active_slot)
                && qwen_kv.active_slot() != original_active_slot
            {
                qwen_kv.set_active_slot(original_active_slot);
            }
            for slot_idx in branch_slots.into_iter().rev() {
                if qwen_kv.has_recurrent_slot(slot_idx) {
                    qwen_kv.free_recurrent_slot(slot_idx);
                }
            }
        }

        result
    }

    /// Allocate a new recurrent slot for a Qwen3.5 hybrid KV.
    pub fn allocate_qwen35_recurrent_slot(&mut self) -> anyhow::Result<usize> {
        let qwen_kv = self
            .as_qwen35_mut()
            .ok_or_else(|| anyhow::anyhow!("qwen35 recurrent slots require ModelKv::Qwen35"))?;
        Ok(qwen_kv.allocate_recurrent_slot())
    }

    /// Free a recurrent slot in a Qwen3.5 hybrid KV.
    pub fn free_qwen35_recurrent_slot(&mut self, slot_idx: usize) -> anyhow::Result<()> {
        let qwen_kv = self
            .as_qwen35_mut()
            .ok_or_else(|| anyhow::anyhow!("qwen35 recurrent slots require ModelKv::Qwen35"))?;
        qwen_kv.free_recurrent_slot(slot_idx);
        Ok(())
    }

    /// Return the currently active recurrent slot for a Qwen3.5 hybrid KV.
    pub fn qwen35_active_slot(&self) -> anyhow::Result<usize> {
        let qwen_kv = self
            .as_qwen35()
            .ok_or_else(|| anyhow::anyhow!("qwen35 active slot requires ModelKv::Qwen35"))?;
        Ok(qwen_kv.active_slot())
    }

    /// Set the active recurrent slot for a Qwen3.5 hybrid KV.
    pub fn set_qwen35_active_slot(&mut self, slot_idx: usize) -> anyhow::Result<()> {
        let qwen_kv = self
            .as_qwen35_mut()
            .ok_or_else(|| anyhow::anyhow!("qwen35 active slot requires ModelKv::Qwen35"))?;
        qwen_kv.set_active_slot(slot_idx);
        Ok(())
    }

    /// Snapshot the recurrent state for a specific Qwen3.5 slot.
    pub fn snapshot_qwen35_recurrent_slot(
        &self,
        slot_idx: usize,
    ) -> anyhow::Result<Qwen35RecurrentSlotSnapshot> {
        let qwen_kv = self.as_qwen35().ok_or_else(|| {
            anyhow::anyhow!("qwen35 recurrent slot snapshots require ModelKv::Qwen35")
        })?;
        Ok(qwen_kv.recurrent_slot_snapshot(slot_idx))
    }

    /// Restore the recurrent state for a specific Qwen3.5 slot.
    pub fn restore_qwen35_recurrent_slot(
        &mut self,
        slot_idx: usize,
        snapshot: &Qwen35RecurrentSlotSnapshot,
    ) -> anyhow::Result<()> {
        let qwen_kv = self.as_qwen35_mut().ok_or_else(|| {
            anyhow::anyhow!("qwen35 recurrent slot snapshots require ModelKv::Qwen35")
        })?;
        qwen_kv.restore_recurrent_slot(slot_idx, snapshot);
        Ok(())
    }

    /// Snapshot the shared attention timeline for a Qwen3.5 hybrid KV.
    pub fn snapshot_qwen35_attention_timeline(&self) -> anyhow::Result<Qwen35AttentionSnapshot> {
        let qwen_kv = self
            .as_qwen35()
            .ok_or_else(|| anyhow::anyhow!("qwen35 attention snapshots require ModelKv::Qwen35"))?;
        Ok(qwen_kv.attention_snapshot())
    }

    /// Restore the shared attention timeline for a Qwen3.5 hybrid KV.
    pub fn restore_qwen35_attention_timeline(
        &mut self,
        snapshot: &Qwen35AttentionSnapshot,
    ) -> anyhow::Result<()> {
        let qwen_kv = self
            .as_qwen35_mut()
            .ok_or_else(|| anyhow::anyhow!("qwen35 attention snapshots require ModelKv::Qwen35"))?;
        qwen_kv.restore_attention_snapshot(snapshot);
        Ok(())
    }

    /// Truncate the shared Qwen3.5 attention timeline to a finalized prefix.
    pub fn truncate_qwen35_attention_timeline(&mut self, pos: usize) -> anyhow::Result<()> {
        let qwen_kv = self
            .as_qwen35_mut()
            .ok_or_else(|| anyhow::anyhow!("qwen35 attention truncate requires ModelKv::Qwen35"))?;
        qwen_kv.truncate_attention_to(pos);
        Ok(())
    }

    /// Materialize the Qwen3.5 shared attention timeline from GPU if needed.
    pub fn sync_qwen35_attention_timeline_if_needed(&mut self) -> anyhow::Result<()> {
        let qwen_kv = self
            .as_qwen35_mut()
            .ok_or_else(|| anyhow::anyhow!("qwen35 attention sync requires ModelKv::Qwen35"))?;
        qwen_kv.sync_attention_cpu_from_gpu_if_needed();
        Ok(())
    }

    /// Clone the recurrent state from one Qwen3.5 slot into another slot.
    pub fn clone_qwen35_recurrent_slot(
        &mut self,
        src_slot_idx: usize,
        dst_slot_idx: usize,
    ) -> anyhow::Result<()> {
        let qwen_kv = self.as_qwen35_mut().ok_or_else(|| {
            anyhow::anyhow!("qwen35 recurrent slot snapshots require ModelKv::Qwen35")
        })?;
        qwen_kv.clone_recurrent_slot(src_slot_idx, dst_slot_idx);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
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
        let mut kv = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
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
    fn test_model_kv_qwen35_snapshot_restores_into_fresh_kv() {
        let mut source = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
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

        let mut restored = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
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
        let qwen_snapshot = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2))
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
        let mut kv = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
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
        let mut kv = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
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
        let mut kv = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));

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
        let mut kv = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));

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
        let mut kv = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
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
    fn test_model_kv_with_qwen35_shared_timeline_branches_from_slot_commits_back_to_original_active()
     {
        let mut kv = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
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
        let mut kv = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
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
        let mut kv = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
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
        let mut kv = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
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
        let mut kv = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
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
}
