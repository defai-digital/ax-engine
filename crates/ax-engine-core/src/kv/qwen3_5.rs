//! Qwen3.5-specific `ModelKv` helpers that operate on hybrid recurrent state.

use super::{ModelKv, Qwen3_5AttentionSnapshot, Qwen3_5RecurrentSlotSnapshot};

impl ModelKv {
    /// Get a reference to the Qwen3.5 hybrid cache/state.
    pub fn as_qwen35(&self) -> Option<&super::Qwen3_5Kv> {
        match self {
            Self::Qwen35(q) => Some(q),
            Self::Cpu(_) | Self::Gpu(_) => None,
        }
    }

    /// Get a reference to the Qwen3.5 hybrid cache/state via the normalized name.
    pub fn as_qwen3_5(&self) -> Option<&super::Qwen3_5Kv> {
        self.as_qwen35()
    }

    /// Get a mutable reference to the Qwen3.5 hybrid cache/state.
    pub fn as_qwen35_mut(&mut self) -> Option<&mut super::Qwen3_5Kv> {
        match self {
            Self::Qwen35(q) => Some(q),
            Self::Cpu(_) | Self::Gpu(_) => None,
        }
    }

    /// Get a mutable reference to the Qwen3.5 hybrid cache/state via the normalized name.
    pub fn as_qwen3_5_mut(&mut self) -> Option<&mut super::Qwen3_5Kv> {
        self.as_qwen35_mut()
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
        &mut self,
        slot_idx: usize,
    ) -> anyhow::Result<Qwen3_5RecurrentSlotSnapshot> {
        let qwen_kv = self.as_qwen35_mut().ok_or_else(|| {
            anyhow::anyhow!("qwen35 recurrent slot snapshots require ModelKv::Qwen35")
        })?;
        Ok(qwen_kv.recurrent_slot_snapshot(slot_idx))
    }

    /// Restore the recurrent state for a specific Qwen3.5 slot.
    pub fn restore_qwen35_recurrent_slot(
        &mut self,
        slot_idx: usize,
        snapshot: &Qwen3_5RecurrentSlotSnapshot,
    ) -> anyhow::Result<()> {
        let qwen_kv = self.as_qwen35_mut().ok_or_else(|| {
            anyhow::anyhow!("qwen35 recurrent slot snapshots require ModelKv::Qwen35")
        })?;
        qwen_kv.restore_recurrent_slot(slot_idx, snapshot);
        Ok(())
    }

    /// Snapshot the shared attention timeline for a Qwen3.5 hybrid KV.
    pub fn snapshot_qwen35_attention_timeline(&self) -> anyhow::Result<Qwen3_5AttentionSnapshot> {
        let qwen_kv = self
            .as_qwen35()
            .ok_or_else(|| anyhow::anyhow!("qwen35 attention snapshots require ModelKv::Qwen35"))?;
        Ok(qwen_kv.attention_snapshot())
    }

    /// Restore the shared attention timeline for a Qwen3.5 hybrid KV.
    pub fn restore_qwen35_attention_timeline(
        &mut self,
        snapshot: &Qwen3_5AttentionSnapshot,
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
